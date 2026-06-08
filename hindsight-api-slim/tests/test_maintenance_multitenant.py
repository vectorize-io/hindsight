"""Multi-tenant maintenance-loop test.

Provisions 100 tenant schemas and verifies that each of the loop's three jobs —
audit-log retention, llm-request retention, and consolidation reconcile —
affects only the tenants that should be affected, leaving the rest untouched.

Schemas are provisioned cheaply by cloning just the five tables the loop touches
(`CREATE TABLE ... LIKE public.<t> INCLUDING DEFAULTS`); the server-side routines
discover them by table presence, exactly as they would real tenant schemas.
"""

import uuid

import pytest
import pytest_asyncio

from hindsight_api.engine.maintenance import MaintenanceLoop
from hindsight_api.engine.memory_engine import MemoryEngine, _current_schema
from hindsight_api.extensions.builtin.tenant import DefaultTenantExtension
from hindsight_api.extensions.tenant import Tenant

N_TENANTS = 100
_CLONED_TABLES = ("banks", "memory_units", "async_operations", "audit_log", "llm_requests")


class _StaticTenantExtension(DefaultTenantExtension):
    """Lists a fixed set of tenants (each with a tenant_id) for the reconcile sweep."""

    def __init__(self, tenants: list[Tenant]) -> None:
        super().__init__(config={})
        self._tenants = list(tenants)

    async def list_tenants(self) -> list[Tenant]:
        return list(self._tenants)


@pytest_asyncio.fixture
async def hundred_tenant_schemas(memory: MemoryEngine):
    """Create N_TENANTS isolated schemas cloning the loop's tables; drop them after."""
    prefix = f"mt{uuid.uuid4().hex[:8]}"
    schemas = [f"{prefix}_{i:03d}" for i in range(N_TENANTS)]
    async with memory._pool.acquire() as conn:
        for s in schemas:
            await conn.execute(f'CREATE SCHEMA "{s}"')
            for table in _CLONED_TABLES:
                await conn.execute(f'CREATE TABLE "{s}".{table} (LIKE public.{table} INCLUDING DEFAULTS)')
    try:
        yield prefix, schemas
    finally:
        async with memory._pool.acquire() as conn:
            for s in schemas:
                await conn.execute(f'DROP SCHEMA IF EXISTS "{s}" CASCADE')


async def _expired_schemas(memory: MemoryEngine, table: str, ts_col: str, days: int) -> set[str]:
    async with memory._pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM public.schemas_with_expired_rows($1, $2, $3)", table, ts_col, days)
    return {r[0] for r in rows}


async def _banks_needing(memory: MemoryEngine) -> set[tuple[str, str]]:
    async with memory._pool.acquire() as conn:
        rows = await conn.fetch("SELECT schema_name, bank_id FROM public.banks_needing_consolidation()")
    return {(r["schema_name"], r["bank_id"]) for r in rows}


@pytest.mark.asyncio
async def test_maintenance_loop_targets_only_affected_tenants(
    memory: MemoryEngine, hundred_tenant_schemas, monkeypatch
):
    prefix, schemas = hundred_tenant_schemas
    schema_set = set(schemas)
    loop = MaintenanceLoop(memory)

    # Per-tenant categories (deterministic by index):
    #   consolidation (i % 4): 0=eligible, 1=auto-consolidation disabled, 2=in-flight op, 3=already consolidated
    #   audit_log retention:   i % 2 == 0 has a 10-day-old row (the rest only recent)
    #   llm_requests retention: i % 3 == 0 has a 3-day-old row (the rest only recent)
    eligible: set[tuple[str, str]] = set()
    not_eligible: set[tuple[str, str]] = set()
    audit_with_old: set[str] = set()
    llm_with_old: set[str] = set()

    async with memory._pool.acquire() as conn:
        for i, s in enumerate(schemas):
            bank = f"{prefix}-bank-{i}"
            cat = i % 4
            if cat == 1:
                cfg = '{"enable_auto_consolidation": false}'
            else:
                cfg = '{"enable_observations": true, "enable_auto_consolidation": true}'
            await conn.execute(f'INSERT INTO "{s}".banks (bank_id, config) VALUES ($1, $2::jsonb)', bank, cfg)
            await conn.execute(
                f'INSERT INTO "{s}".memory_units (id, bank_id, text, fact_type, created_at, consolidated_at) '
                f"VALUES ($1, $2, 'f', 'experience', now(), CASE WHEN $3 THEN now() ELSE NULL END)",
                uuid.uuid4(),
                bank,
                cat == 3,  # already consolidated
            )
            if cat == 2:  # in-flight consolidation op
                await conn.execute(
                    f'INSERT INTO "{s}".async_operations (operation_id, bank_id, operation_type, status, task_payload) '
                    f"VALUES ($1, $2, 'consolidation', 'pending', '{{}}'::jsonb)",
                    uuid.uuid4(),
                    bank,
                )
            (eligible if cat == 0 else not_eligible).add((s, bank))

            await conn.execute(
                f"INSERT INTO \"{s}\".audit_log (action, transport, started_at) VALUES ('NEW', 'system', now())"
            )
            if i % 2 == 0:
                await conn.execute(
                    f"INSERT INTO \"{s}\".audit_log (action, transport, started_at) "
                    f"VALUES ('OLD', 'system', now() - INTERVAL '10 days')"
                )
                audit_with_old.add(s)

            await conn.execute(f"INSERT INTO \"{s}\".llm_requests (status, started_at) VALUES ('success', now())")
            if i % 3 == 0:
                await conn.execute(
                    f"INSERT INTO \"{s}\".llm_requests (status, started_at) "
                    f"VALUES ('success', now() - INTERVAL '3 days')"
                )
                llm_with_old.add(s)

    # ── 1. audit_log retention ────────────────────────────────────────────────
    # Discovery targets exactly the tenants holding an expired row.
    assert (await _expired_schemas(memory, "audit_log", "started_at", 7)) & schema_set == audit_with_old
    await loop._purge_expired("audit_log", "started_at", 7)
    async with memory._pool.acquire() as conn:
        for s in schemas:
            old = await conn.fetchval(f"SELECT count(*) FROM \"{s}\".audit_log WHERE action = 'OLD'")
            new = await conn.fetchval(f"SELECT count(*) FROM \"{s}\".audit_log WHERE action = 'NEW'")
            assert old == 0, f"{s}: expired audit row not purged"
            assert new == 1, f"{s}: recent audit row wrongly deleted"

    # ── 2. llm_requests retention ─────────────────────────────────────────────
    assert (await _expired_schemas(memory, "llm_requests", "started_at", 1)) & schema_set == llm_with_old
    await loop._purge_expired("llm_requests", "started_at", 1)
    async with memory._pool.acquire() as conn:
        for s in schemas:
            total = await conn.fetchval(f'SELECT count(*) FROM "{s}".llm_requests')
            recent = await conn.fetchval(
                f"SELECT count(*) FROM \"{s}\".llm_requests WHERE started_at > now() - INTERVAL '1 day'"
            )
            assert total == 1, f"{s}: expected only the recent llm_requests row to remain"
            assert recent == 1, f"{s}: recent llm_requests row wrongly deleted"

    # ── 3. consolidation reconcile ────────────────────────────────────────────
    # Discovery returns exactly the eligible banks among ours (not disabled/in-flight/consolidated).
    discovered_banks = await _banks_needing(memory)
    assert {(s, b) for (s, b) in discovered_banks if s in schema_set} == eligible
    assert discovered_banks.isdisjoint(not_eligible)

    monkeypatch.setattr(
        memory,
        "_tenant_extension",
        _StaticTenantExtension([Tenant(schema=s, tenant_id=f"tid-{i}") for i, s in enumerate(schemas)]),
    )
    submitted: list[tuple[str | None, str]] = []

    async def _record(*, bank_id, request_context, observation_scopes=None):
        # Capture the schema the op is being enqueued into (set on the contextvar by the loop).
        submitted.append((_current_schema.get(), bank_id))
        return {"operation_id": str(uuid.uuid4())}

    monkeypatch.setattr(memory, "submit_async_consolidation", _record)

    await loop._run_reconcile()

    ours_submitted = {(s, b) for (s, b) in submitted if s in schema_set}
    # Exactly the eligible tenants were reconciled — into their own schema — and nobody else.
    assert ours_submitted == eligible
