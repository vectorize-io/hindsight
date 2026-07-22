"""Tests for the extension bank-scoped-table registry seam.

Extensions declare bank-scoped tables via ``TenantExtension.extra_bank_tables``
so those tables participate in core's per-tenant data lifecycle:

* admin backup/restore includes them (``_effective_backup_tables``);
* ``MemoryEngine.delete_bank`` sweeps them so no orphaned rows survive.

The descriptor validation and the backup-list computation are pure-Python; the
delete_bank sweep runs against a real Postgres via the ``memory`` fixture.
"""

import uuid

import pytest

from hindsight_api import RequestContext
from hindsight_api.admin import cli as admin_cli
from hindsight_api.admin.cli import BACKUP_TABLES, _effective_backup_tables
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.extensions.bank_tables import BankScopedTable
from hindsight_api.extensions.tenant import Tenant, TenantContext, TenantExtension


class _StubTenant(TenantExtension):
    """Minimal TenantExtension that only declares extra bank tables."""

    def __init__(self, specs: list[BankScopedTable]):
        super().__init__(config={})
        self._specs = specs

    async def authenticate(self, context: RequestContext) -> TenantContext:  # pragma: no cover - unused
        raise NotImplementedError

    async def list_tenants(self) -> list[Tenant]:  # pragma: no cover - unused
        return []

    def extra_bank_tables(self) -> list[BankScopedTable]:
        return self._specs


# ---------------------------------------------------------------------------
# Descriptor validation
# ---------------------------------------------------------------------------


def test_defaults_and_valid_identifier():
    spec = BankScopedTable(name="privacy_events")
    assert spec.bank_id_column == "bank_id"
    assert spec.include_in_backup is True
    assert spec.delete_with_bank is True


@pytest.mark.parametrize("bad", ["has-dash", "with space", "semi;colon", "", "1leading", "a.b"])
def test_invalid_table_name_rejected(bad):
    with pytest.raises(ValueError, match="not a valid SQL identifier"):
        BankScopedTable(name=bad)


def test_invalid_bank_id_column_rejected():
    with pytest.raises(ValueError, match="not a valid SQL identifier"):
        BankScopedTable(name="ok_table", bank_id_column="bank id")


# ---------------------------------------------------------------------------
# Effective backup-table computation
# ---------------------------------------------------------------------------


def test_effective_backup_tables_no_extension(monkeypatch):
    monkeypatch.setattr(admin_cli, "load_extension", lambda *a, **k: None)
    assert _effective_backup_tables() == BACKUP_TABLES


def test_effective_backup_tables_appends_after_core(monkeypatch):
    ext = _StubTenant(
        [
            BankScopedTable(name="privacy_events"),
            BankScopedTable(name="privacy_exports", include_in_backup=False),  # excluded
            BankScopedTable(name="banks"),  # dup of a core table — deduped
        ]
    )
    monkeypatch.setattr(admin_cli, "load_extension", lambda *a, **k: ext)

    result = _effective_backup_tables()

    # Core list preserved in order and comes first.
    assert result[: len(BACKUP_TABLES)] == BACKUP_TABLES
    # Only backup-participating, non-duplicate extension tables appended.
    assert result[len(BACKUP_TABLES) :] == ["privacy_events"]
    assert "privacy_exports" not in result
    assert result.count("banks") == 1


# ---------------------------------------------------------------------------
# delete_bank sweep (real DB)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_bank_sweeps_extension_tables(memory: MemoryEngine, request_context: RequestContext, monkeypatch):
    swept = "ext_receipts_swept"
    kept = "ext_receipts_kept"
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        for tbl in (swept, kept):
            await conn.execute(f"CREATE TABLE IF NOT EXISTS {tbl} (id uuid PRIMARY KEY, bank_id text NOT NULL)")
            await conn.execute(f"TRUNCATE {tbl}")

    bank_a = f"test-ext-a-{uuid.uuid4().hex[:8]}"
    bank_b = f"test-ext-b-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_a, request_context=request_context)
    await memory.get_bank_profile(bank_id=bank_b, request_context=request_context)

    async with pool.acquire() as conn:
        for tbl in (swept, kept):
            for bank in (bank_a, bank_b):
                await conn.execute(f"INSERT INTO {tbl} (id, bank_id) VALUES ($1, $2)", uuid.uuid4(), bank)

    # Declare the tables on the live tenant extension; `kept` opts out of
    # teardown. Patch only extra_bank_tables so authentication still works.
    specs = [
        BankScopedTable(name=swept),
        BankScopedTable(name=kept, delete_with_bank=False),
        BankScopedTable(name="does_not_exist_table"),  # unprovisioned → skipped, must not error
    ]
    monkeypatch.setattr(memory._tenant_extension, "extra_bank_tables", lambda: specs)

    await memory.delete_bank(bank_a, request_context=request_context)

    async with pool.acquire() as conn:
        # `swept` lost bank_a's rows, kept bank_b's.
        assert await conn.fetchval(f"SELECT count(*) FROM {swept} WHERE bank_id = $1", bank_a) == 0
        assert await conn.fetchval(f"SELECT count(*) FROM {swept} WHERE bank_id = $1", bank_b) == 1
        # `kept` opted out — both banks' rows survive.
        assert await conn.fetchval(f"SELECT count(*) FROM {kept} WHERE bank_id = $1", bank_a) == 1

        await conn.execute(f"DROP TABLE IF EXISTS {swept}")
        await conn.execute(f"DROP TABLE IF EXISTS {kept}")

    await memory.delete_bank(bank_b, request_context=request_context)
