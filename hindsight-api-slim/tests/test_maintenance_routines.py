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


def _capture_upgrade(migration, monkeypatch, target_schema: str | None, configured_schema: str = "public") -> list[str]:
    """Run the migration's ``_pg_upgrade`` for ``target_schema``, capturing SQL.

    ``configured_schema`` is the deployment's ``HINDSIGHT_API_DATABASE_SCHEMA``;
    the migration installs only when the run targets it.
    """
    monkeypatch.setattr(migration, "context", _FakeAlembicContext(target_schema))
    if hasattr(migration, "get_config"):
        monkeypatch.setattr(migration, "get_config", lambda: SimpleNamespace(database_schema=configured_schema))
    executed: list[str] = []
    monkeypatch.setattr(migration.op, "execute", lambda sql: executed.append(str(sql)))
    migration._pg_upgrade()
    return executed


@pytest.mark.parametrize(
    ("target_schema", "configured", "expected_prefix"),
    [
        (None, "public", ""),  # base run: search_path resolves to the configured schema
        ("public", "public", '"public".'),  # default deployment
        ("hs_tenant", "hs_tenant", '"hs_tenant".'),  # the #2638 case: single-tenant, non-public
    ],
)
def test_routines_install_in_the_configured_schema(monkeypatch, target_schema, configured, expected_prefix):
    """Regression for #2638: the install must follow the *configured* schema.

    The old gate compared ``target_schema`` against the literal ``"public"``, so a
    deployment living in a dedicated non-``public`` schema never installed the
    routines at all. Comparing against ``get_config().database_schema`` installs
    exactly one copy, in the schema ``fq_routine`` actually calls.
    """
    migration = _load_schema_local_migration()
    joined = "\n".join(_capture_upgrade(migration, monkeypatch, target_schema, configured))

    for routine in ("banks_needing_consolidation", "schemas_with_expired_rows", "mental_models_with_cron"):
        assert f"CREATE OR REPLACE FUNCTION {expected_prefix}{routine}" in joined
    # The public-only gate that caused #2638 must not come back...
    assert not hasattr(migration, "_should_install_public_routines")
    # ...and neither must an advisory lock (unusable behind poolers; see #2817).
    assert "advisory" not in joined.lower()


def test_tenant_runs_install_nothing_and_clean_up_strays(monkeypatch):
    """A tenant schema must not carry its own copy.

    These routines are database-global — each enumerates ``pg_class`` across every
    schema — so only the copy in the configured schema is ever called. A per-tenant
    copy is dead weight, and the first cut of this migration created one per
    tenant, so tenant runs drop rather than merely skip.
    """
    migration = _load_schema_local_migration()
    joined = "\n".join(_capture_upgrade(migration, monkeypatch, "tenant_xyz", configured_schema="public"))

    assert "CREATE OR REPLACE FUNCTION" not in joined
    for routine in ("banks_needing_consolidation", "schemas_with_expired_rows", "mental_models_with_cron"):
        assert f'DROP FUNCTION IF EXISTS "tenant_xyz".{routine}' in joined


@pytest.mark.asyncio
async def test_routines_callable_from_non_public_schema(memory: MemoryEngine, request_context, monkeypatch):
    """End-to-end #2638: with the deployment schema set to a non-``public`` schema,
    the migration installs the routines there and ``fq_routine()`` resolves to that
    copy, which returns the same cross-tenant results as the ``public`` one.
    """
    from hindsight_api.engine import schema as schema_mod

    migration = _load_schema_local_migration()
    schema = f"tenant_{uuid.uuid4().hex[:8]}"

    bank_id = await _make_bank(memory, request_context, "nonpublic")
    async with memory._pool.acquire() as conn:
        await _insert_fact(conn, bank_id)
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        try:
            # Drive the migration exactly as a per-schema run would.
            # Configured schema == this schema, i.e. a single-tenant deployment
            # living in a dedicated non-public schema (the #2638 shape).
            for stmt in _capture_upgrade(migration, monkeypatch, schema, configured_schema=schema):
                await conn.execute(stmt)

            # The loop's qualifier now points at the non-public copy...
            # Config is a read-only proxy, so swap the accessor rather than the field.
            monkeypatch.setattr(schema_mod, "get_config", lambda: SimpleNamespace(database_schema=schema))
            assert schema_mod.fq_routine("banks_needing_consolidation") == f'"{schema}".banks_needing_consolidation'

            # ...and that copy works: it enumerates pg_class database-wide, so it
            # still finds the bank living in the public schema.
            rows = await conn.fetch(f'SELECT schema_name, bank_id FROM "{schema}".banks_needing_consolidation()')
            assert bank_id in {r["bank_id"] for r in rows}
        finally:
            await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


def _capture_downgrade(migration, monkeypatch, target_schema, configured_schema) -> list[str]:
    monkeypatch.setattr(migration, "context", _FakeAlembicContext(target_schema))
    monkeypatch.setattr(migration, "get_config", lambda: SimpleNamespace(database_schema=configured_schema))
    executed: list[str] = []
    monkeypatch.setattr(migration.op, "execute", lambda sql: executed.append(str(sql)))
    migration._pg_downgrade()
    return executed


@pytest.mark.parametrize("target_schema", [None, "public"])
def test_downgrade_leaves_public_copies_alone(monkeypatch, target_schema):
    """When the configured schema is ``public`` the copies there belong to
    e5f6a7b8c9d0 / f4d1c2b3a5e6, which are still applied when this migration is
    downgraded — dropping them here would strand those migrations without the
    functions they claim to have installed."""
    migration = _load_schema_local_migration()
    assert _capture_downgrade(migration, monkeypatch, target_schema, "public") == []


def test_downgrade_drops_the_copy_it_owns(monkeypatch):
    """On a non-``public`` deployment the copy exists only because of this
    migration, so its downgrade must remove it."""
    migration = _load_schema_local_migration()
    joined = "\n".join(_capture_downgrade(migration, monkeypatch, "hs_tenant", "hs_tenant"))

    for routine in ("banks_needing_consolidation", "schemas_with_expired_rows", "mental_models_with_cron"):
        assert f'DROP FUNCTION IF EXISTS "hs_tenant".{routine}' in joined


def test_downgrade_ignores_tenant_runs(monkeypatch):
    """A tenant run never owned a copy under the current install policy, so its
    downgrade has nothing to undo."""
    migration = _load_schema_local_migration()
    assert _capture_downgrade(migration, monkeypatch, "tenant_xyz", "hs_tenant") == []


def _load_expired_operations_migration():
    """Import the #2708-followup operation-discovery routine migration by path."""
    path = (
        Path(__file__).resolve().parent.parent
        / "hindsight_api/alembic/versions/d7b2f8a1c934_add_schemas_with_expired_operations_routine.py"
    )
    spec = importlib.util.spec_from_file_location("_schemas_with_expired_operations", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("target_schema", "configured", "expected_prefix"),
    [
        (None, "public", ""),  # base run: search_path resolves to the configured schema
        ("public", "public", '"public".'),  # default deployment
        ("hs_tenant", "hs_tenant", '"hs_tenant".'),  # the #2638 case: single-tenant, non-public
    ],
)
def test_expired_operations_routine_installs_in_the_configured_schema(
    monkeypatch, target_schema, configured, expected_prefix
):
    """The operation-discovery routine must follow ``b6d2f8a4c1e7`` (#2824).

    One copy, in the schema the deployment is *configured* to use — which is the
    one ``fq_routine`` calls. Gating on the literal ``"public"`` is what left
    non-``public`` deployments without the routine (#2638), and installing into
    every schema would leave a dead duplicate per tenant.
    """
    migration = _load_expired_operations_migration()
    joined = "\n".join(_capture_upgrade(migration, monkeypatch, target_schema, configured))

    assert f"CREATE OR REPLACE FUNCTION {expected_prefix}schemas_with_expired_operations" in joined
    assert not hasattr(migration, "_should_install_public_routines")
    # Advisory locks are unusable behind poolers / managed PG (see #2817) — one
    # install run is what makes concurrent runs safe without one.
    assert "advisory" not in joined.lower()


def test_expired_operations_tenant_runs_install_nothing(monkeypatch):
    """Tenant schemas must not carry their own copy; the run drops instead."""
    migration = _load_expired_operations_migration()
    joined = "\n".join(_capture_upgrade(migration, monkeypatch, "tenant_xyz", configured_schema="public"))

    assert "CREATE OR REPLACE FUNCTION" not in joined
    assert 'DROP FUNCTION IF EXISTS "tenant_xyz".schemas_with_expired_operations' in joined


@pytest.mark.asyncio
async def test_schemas_with_expired_operations(memory: MemoryEngine):
    """Returns schemas holding an expired *terminal* operation only.

    This is the discovery half of the worker's cleanup sweep: pending and
    processing rows are never prunable, so an old pending row must not make a
    schema look like it has work. Mirrors ``schemas_with_expired_rows`` for the
    ``p_days <= 0`` (retention disabled) guard.

    Runs against a throwaway schema rather than ``public``: the suite shares one
    database, so terminal operations left behind by other tests (or a previous
    run) would make ``public`` eligible no matter what this test inserts.
    """
    schema = f"mtops{uuid.uuid4().hex[:8]}"

    async def _insert_op(conn, status: str, age_days: int) -> None:
        await conn.execute(
            f'''
            INSERT INTO "{schema}".async_operations
                (operation_id, bank_id, operation_type, status, task_payload, updated_at)
            VALUES ($1, 'b', 'retain', $2, '{{}}'::jsonb, now() - make_interval(days => $3))
            ''',
            uuid.uuid4(),
            status,
            age_days,
        )

    async def _expired(conn, days: int) -> set[str]:
        rows = await conn.fetch("SELECT * FROM public.schemas_with_expired_operations($1)", days)
        return {r[0] for r in rows}

    try:
        async with memory._pool.acquire() as conn:
            await conn.execute(f'CREATE SCHEMA "{schema}"')
            await conn.execute(
                f'CREATE TABLE "{schema}".async_operations (LIKE public.async_operations INCLUDING DEFAULTS)'
            )

            # Old, but not terminal: must not make the schema eligible on its own.
            await _insert_op(conn, "pending", 10)
            await _insert_op(conn, "processing", 10)
            assert schema not in await _expired(conn, 7)

            await _insert_op(conn, "completed", 10)
            assert schema in await _expired(conn, 7)

            # Cutoff older than any row: nothing to prune.
            assert schema not in await _expired(conn, 36500)

            # Disabled retention (days <= 0): always empty, so the worker skips the sweep.
            assert await _expired(conn, 0) == set()
    finally:
        async with memory._pool.acquire() as conn:
            await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
