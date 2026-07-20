"""Tests for the server-side maintenance discovery routines.

``public.banks_needing_consolidation()`` and
``public.schemas_with_expired_rows(table, ts_col, days)`` are installed by the
maintenance-routines migration and loop over every schema holding the relevant
table in a single round-trip. These tests drive them directly against pg0.
"""

import importlib.util
import uuid
from types import SimpleNamespace
from pathlib import Path

import pytest

from hindsight_api.engine.memory_engine import MemoryEngine


def _load_repair_migration():
    """Import the repair migration by path (filename starts with a digit, so it
    is not importable as a normal module name)."""
    path = (
        Path(__file__).resolve().parent.parent
        / "hindsight_api/alembic/versions/b2d4f6a8c1e3_repair_maintenance_routines_public.py"
    )
    spec = importlib.util.spec_from_file_location("_repair_maintenance_routines", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("target_schema", "expected"),
    [
        (None, True),  # base-schema run (no target_schema)
        ("", True),  # falsy schema behaves like the base run
        ("public", True),  # the case #2056 regressed: explicit public must install
        ("tenant_xyz", False),  # per-tenant run skips to avoid concurrent CREATE
    ],
)
def test_repair_gate_installs_on_public_and_base_runs(target_schema, expected):
    """Regression for #2056: the maintenance routines live in ``public`` and must
    be (re)created on both the base run and the explicit ``target_schema=public``
    run — the runtime always migrates an explicit ``public`` schema, so gating on
    ``not target_schema`` alone silently skipped function creation."""
    migration = _load_repair_migration()
    assert migration._should_install_public_routines(target_schema) is expected


async def _make_bank(memory: MemoryEngine, request_context, suffix: str) -> str:
    bank_id = f"maint-{suffix}-{uuid.uuid4().hex[:8]}"
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    return bank_id


async def _insert_fact(
    conn, bank_id: str, *, fact_type: str = "experience", consolidated: bool = False, failed: bool = False
) -> None:
    await conn.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, created_at, consolidated_at, consolidation_failed_at)
        VALUES ($1, $2, 'a fact', $3, now(),
                CASE WHEN $4 THEN now() ELSE NULL END,
                CASE WHEN $5 THEN now() ELSE NULL END)
        """,
        uuid.uuid4(),
        bank_id,
        fact_type,
        consolidated,
        failed,
    )


@pytest.mark.asyncio
async def test_banks_needing_consolidation_filters(memory: MemoryEngine, request_context):
    """Returns only banks with eligible-but-unscheduled facts, auto-consolidation
    not bank-disabled, and no in-flight consolidation op."""
    eligible = await _make_bank(memory, request_context, "eligible")
    eligible_world = await _make_bank(memory, request_context, "world")
    all_consolidated = await _make_bank(memory, request_context, "done")
    all_failed = await _make_bank(memory, request_context, "failed")
    in_flight = await _make_bank(memory, request_context, "inflight")
    bank_disabled = await _make_bank(memory, request_context, "disabled")

    async with memory._pool.acquire() as conn:
        await _insert_fact(conn, eligible)
        await _insert_fact(conn, eligible_world, fact_type="world")
        await _insert_fact(conn, all_consolidated, consolidated=True)
        await _insert_fact(conn, all_failed, failed=True)

        await _insert_fact(conn, in_flight)
        await conn.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
            VALUES ($1, $2, 'consolidation', 'pending', '{}'::jsonb)
            """,
            uuid.uuid4(),
            in_flight,
        )

        await _insert_fact(conn, bank_disabled)
        await conn.execute(
            "UPDATE banks SET config = '{\"enable_auto_consolidation\": false}'::jsonb WHERE bank_id = $1",
            bank_disabled,
        )

        rows = await conn.fetch("SELECT schema_name, bank_id FROM public.banks_needing_consolidation()")

    returned = {r["bank_id"] for r in rows}
    assert eligible in returned
    assert eligible_world in returned
    assert all_consolidated not in returned
    assert all_failed not in returned
    assert in_flight not in returned
    assert bank_disabled not in returned


@pytest.mark.asyncio
async def test_banks_needing_consolidation_includes_in_flight_after_completion(memory: MemoryEngine, request_context):
    """A bank whose only consolidation op is already completed is still eligible
    (only pending/processing ops suppress re-scheduling)."""
    bank = await _make_bank(memory, request_context, "completed-op")
    async with memory._pool.acquire() as conn:
        await _insert_fact(conn, bank)
        await conn.execute(
            """
            INSERT INTO async_operations (operation_id, bank_id, operation_type, status, task_payload)
            VALUES ($1, $2, 'consolidation', 'completed', '{}'::jsonb)
            """,
            uuid.uuid4(),
            bank,
        )
        rows = await conn.fetch("SELECT bank_id FROM public.banks_needing_consolidation()")
    assert bank in {r["bank_id"] for r in rows}


@pytest.mark.asyncio
async def test_banks_needing_consolidation_skips_schema_with_vanished_table(memory: MemoryEngine):
    """A schema discovered via its ``memory_units`` table but missing the
    ``banks`` table the routine joins must be skipped, not abort the scan.

    This reproduces the time-of-check/time-of-use race deterministically: the
    routine snapshots schemas owning ``memory_units`` from ``pg_class`` and then
    joins each schema's ``banks`` table. A tenant being dropped or migrated (and,
    in the test suite, the concurrent multi-tenant maintenance test) can leave a
    schema whose ``banks`` table is gone. Before the fix the dynamic query raised
    ``undefined_table`` and aborted the whole routine (migration c7e9f1a3b5d2)."""
    schema = f"mtvanish{uuid.uuid4().hex[:8]}"
    try:
        async with memory._pool.acquire() as conn:
            await conn.execute(f'CREATE SCHEMA "{schema}"')
            # Discovered by the FOR loop (has memory_units) but the JOIN target
            # `banks` is absent — exactly a half-built / vanishing schema.
            await conn.execute(f'CREATE TABLE "{schema}".memory_units (LIKE public.memory_units INCLUDING DEFAULTS)')

            # Must not raise; the bad schema is simply skipped.
            rows = await conn.fetch("SELECT schema_name, bank_id FROM public.banks_needing_consolidation()")
            assert schema not in {r["schema_name"] for r in rows}
    finally:
        async with memory._pool.acquire() as conn:
            await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


@pytest.mark.asyncio
async def test_schemas_with_expired_rows(memory: MemoryEngine):
    """Returns schemas holding a row older than p_days; respects the p_days<=0 guard."""
    async with memory._pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO audit_log (action, transport, started_at) VALUES ('t', 'system', now() - INTERVAL '10 days')"
        )

        # 7-day cutoff: the 10-day-old row makes 'public' expired.
        expired_7 = await conn.fetch("SELECT * FROM public.schemas_with_expired_rows('audit_log', 'started_at', 7)")
        assert "public" in {r[0] for r in expired_7}

        # 100-year cutoff: nothing is that old.
        expired_century = await conn.fetch(
            "SELECT * FROM public.schemas_with_expired_rows('audit_log', 'started_at', 36500)"
        )
        assert "public" not in {r[0] for r in expired_century}

        # Disabled retention (days <= 0): always empty.
        disabled = await conn.fetch("SELECT * FROM public.schemas_with_expired_rows('audit_log', 'started_at', 0)")
        assert len(disabled) == 0


def _load_schema_local_migration():
    """Import the #2638 schema-local install migration by path."""
    path = (
        Path(__file__).resolve().parent.parent
        / "hindsight_api/alembic/versions/b6d2f8a4c1e7_maintenance_routines_schema_local.py"
    )
    spec = importlib.util.spec_from_file_location("_maint_routines_schema_local", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeAlembicContext:
    """Stands in for ``alembic.context``, whose ``config`` only exists inside a
    live migration run."""

    def __init__(self, target_schema: str | None) -> None:
        self.config = SimpleNamespace(get_main_option=lambda name: target_schema if name == "target_schema" else None)


def _capture_upgrade(migration, monkeypatch, target_schema: str | None) -> list[str]:
    """Run the migration's ``_pg_upgrade`` for ``target_schema``, capturing SQL."""
    monkeypatch.setattr(migration, "context", _FakeAlembicContext(target_schema))
    executed: list[str] = []
    monkeypatch.setattr(migration.op, "execute", lambda sql: executed.append(str(sql)))
    migration._pg_upgrade()
    return executed


@pytest.mark.parametrize(
    ("target_schema", "expected_prefix"),
    [
        (None, ""),  # base run: search_path resolves to public
        ("public", '"public".'),
        ("tenant_xyz", '"tenant_xyz".'),  # the #2638 case: must NOT be skipped
    ],
)
def test_schema_local_install_runs_for_every_target_schema(monkeypatch, target_schema, expected_prefix):
    """Regression for #2638: a deploy migrated into a non-``public`` schema must
    still get the routines. They now go into the run's *own* schema on every run,
    so no gate can skip them — and because each process writes only its own
    ``pg_proc`` row, no cross-process lock is needed to make that safe.
    """
    migration = _load_schema_local_migration()
    joined = "\n".join(_capture_upgrade(migration, monkeypatch, target_schema))

    for routine in ("banks_needing_consolidation", "schemas_with_expired_rows", "mental_models_with_cron"):
        assert f"CREATE OR REPLACE FUNCTION {expected_prefix}{routine}" in joined
    # The public-only gate that caused #2638 must not come back...
    assert not hasattr(migration, "_should_install_public_routines")
    # ...and neither must an advisory lock (unusable behind poolers; see #2690).
    assert "advisory" not in joined.lower()


@pytest.mark.asyncio
async def test_routines_callable_from_non_public_schema(memory: MemoryEngine, request_context, monkeypatch):
    """End-to-end #2638: with the deployment schema set to a non-``public`` schema,
    the migration installs the routines there and ``_routine()`` resolves to that
    copy, which returns the same cross-tenant results as the ``public`` one.
    """
    from hindsight_api.engine import maintenance

    migration = _load_schema_local_migration()
    schema = f"tenant_{uuid.uuid4().hex[:8]}"

    bank_id = await _make_bank(memory, request_context, "nonpublic")
    async with memory._pool.acquire() as conn:
        await _insert_fact(conn, bank_id)
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        try:
            # Drive the migration exactly as a per-schema run would.
            for stmt in _capture_upgrade(migration, monkeypatch, schema):
                await conn.execute(stmt)

            # The loop's qualifier now points at the non-public copy...
            # Config is a read-only proxy, so swap the accessor rather than the field.
            monkeypatch.setattr(maintenance, "get_config", lambda: SimpleNamespace(database_schema=schema))
            assert maintenance._routine("banks_needing_consolidation") == f'"{schema}".banks_needing_consolidation'

            # ...and that copy works: it enumerates pg_class database-wide, so it
            # still finds the bank living in the public schema.
            rows = await conn.fetch(f'SELECT schema_name, bank_id FROM "{schema}".banks_needing_consolidation()')
            assert bank_id in {r["bank_id"] for r in rows}
        finally:
            await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


@pytest.mark.parametrize("target_schema", [None, "public"])
def test_schema_local_downgrade_leaves_public_copies_alone(monkeypatch, target_schema):
    """Downgrade must only drop the copies this migration uniquely owns.

    The ``public`` copies belong to e5f6a7b8c9d0 / f4d1c2b3a5e6, which are still
    applied when this one is downgraded — dropping them here would strand those
    migrations without the functions they claim to have installed.
    """
    migration = _load_schema_local_migration()
    monkeypatch.setattr(migration, "context", _FakeAlembicContext(target_schema))
    executed: list[str] = []
    monkeypatch.setattr(migration.op, "execute", lambda sql: executed.append(str(sql)))

    migration._pg_downgrade()

    assert executed == []


def test_schema_local_downgrade_drops_non_public_copies(monkeypatch):
    """A non-``public`` run's copies exist only because of this migration, so its
    downgrade must remove them."""
    migration = _load_schema_local_migration()
    monkeypatch.setattr(migration, "context", _FakeAlembicContext("tenant_xyz"))
    executed: list[str] = []
    monkeypatch.setattr(migration.op, "execute", lambda sql: executed.append(str(sql)))

    migration._pg_downgrade()

    joined = "\n".join(executed)
    for routine in ("banks_needing_consolidation", "schemas_with_expired_rows", "mental_models_with_cron"):
        assert f'DROP FUNCTION IF EXISTS "tenant_xyz".{routine}' in joined
