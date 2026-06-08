"""Tests for the server-side maintenance discovery routines.

``public.banks_needing_consolidation()`` and
``public.schemas_with_expired_rows(table, ts_col, days)`` are installed by the
maintenance-routines migration and loop over every schema holding the relevant
table in a single round-trip. These tests drive them directly against pg0.
"""

import importlib.util
import uuid
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


async def _insert_fact(conn, bank_id: str, *, fact_type: str = "experience", consolidated: bool = False, failed: bool = False) -> None:
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
async def test_schemas_with_expired_rows(memory: MemoryEngine):
    """Returns schemas holding a row older than p_days; respects the p_days<=0 guard."""
    async with memory._pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO audit_log (action, transport, started_at) VALUES ('t', 'system', now() - INTERVAL '10 days')"
        )

        # 7-day cutoff: the 10-day-old row makes 'public' expired.
        expired_7 = await conn.fetch(
            "SELECT * FROM public.schemas_with_expired_rows('audit_log', 'started_at', 7)"
        )
        assert "public" in {r[0] for r in expired_7}

        # 100-year cutoff: nothing is that old.
        expired_century = await conn.fetch(
            "SELECT * FROM public.schemas_with_expired_rows('audit_log', 'started_at', 36500)"
        )
        assert "public" not in {r[0] for r in expired_century}

        # Disabled retention (days <= 0): always empty.
        disabled = await conn.fetch("SELECT * FROM public.schemas_with_expired_rows('audit_log', 'started_at', 0)")
        assert len(disabled) == 0
