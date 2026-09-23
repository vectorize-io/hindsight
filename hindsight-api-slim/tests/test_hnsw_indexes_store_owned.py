"""Per-bank vector indexes must not be created for a bank whose memories live outside SQL.

A store-owned bank writes no ``memory_units`` rows, so its three partial indexes can
only ever be empty. Empty is not free: Postgres plans against every index on a relation,
so they are charged to every OTHER statement that names the shared table. A tenant with
27,315 store-owned banks carried 82,795 such indexes and paid ~975 ms of planning for a
query returning zero rows (#4615).

Two halves, and only one of them is loud:

* creation must skip them — the fix;
* an SQL-owned bank must still get all three — the half that fails SILENTLY if the
  condition is inverted, because recall still answers, just without its ANN index.

Creation only. Nothing here drops an index a bank already carries, and that is the
point: shedding the 82,795 that already exist is an operator's decision, not something
a deploy or a write does behind their back. This suite therefore also pins that the
indexes an existing store-owned bank has are left exactly where they are.

Runs via: uv run pytest tests/test_hnsw_indexes_store_owned.py -v
"""

import uuid

import pytest

import hindsight_api.engine.memories as memories_mod
from hindsight_api.engine import memory_engine as memory_engine_module
from hindsight_api.engine import vector_index_health
from hindsight_api.engine.retain import bank_utils
from hindsight_api.engine.retain.bank_utils import _vector_index_clause
from hindsight_api.engine.vector_index_health import reconcile_bank_vector_indexes
from hindsight_api.models import RequestContext
from tests.test_memories_extension import InMemoryMemories


@pytest.fixture
def request_ctx():
    return RequestContext(api_key=None, api_key_id=None, tenant_id=None, internal=False)


@pytest.fixture(autouse=True)
def eager(monkeypatch):
    """Pin eager mode on for every test here, rather than inheriting the default.

    These tests are about WHICH banks get indexes, not about the size threshold, and
    the two policies are alternatives: with ``HINDSIGHT_API_VECTOR_INDEX_MIN_ROWS``
    set, creation builds nothing for *any* bank — which would fail the SQL-owned
    assertions and, worse, make the store-owned ones pass vacuously. Patched on every
    module that imported the helper by name; a missed one silently restores the
    ambient default (the same trap ``test_hnsw_indexes.py::threshold_set`` documents).
    """
    for module in (vector_index_health, bank_utils, memory_engine_module):
        monkeypatch.setattr(module, "per_bank_indexes_are_eager", lambda: True)


@pytest.fixture
def store_owned(monkeypatch):
    """Route ``get_memories()`` to a store that owns its memory rows.

    Patched on the ``memories`` module rather than on each importer: every call site
    under test reaches the accessor through it, and patching one importer by name would
    leave the others on the real store and quietly pass.
    """
    store = InMemoryMemories()
    monkeypatch.setattr(memories_mod, "get_memories", lambda: store)
    return store


async def _bank_indexes(pool, bank_id: str) -> list[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE tablename = 'memory_units'
              AND indexname LIKE 'idx_mu_emb_%'
              AND indexdef LIKE $1
            ORDER BY indexname
            """,
            f"%bank_id = '{bank_id}'%",
        )
    return [row["indexname"] for row in rows]


@pytest.mark.asyncio
async def test_store_owned_bank_gets_no_vector_indexes(memory, request_ctx, store_owned):
    """The fix. Three indexes on a table this bank will never write a row to."""
    bank_id = f"test_so_none_{uuid.uuid4().hex[:8]}"
    try:
        await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_ctx)

        indexes = await _bank_indexes(memory._pool, bank_id)
        assert indexes == [], f"a store-owned bank must get no per-bank vector indexes, got: {indexes}"
    finally:
        await memory.delete_bank(bank_id, request_context=request_ctx)


@pytest.mark.asyncio
async def test_a_sql_owned_bank_still_gets_all_three(memory, request_ctx):
    """The silent half: inverting the condition strips ANN from every ordinary bank.

    Nothing fails when that happens — recall falls back to the exact ``(bank_id,
    fact_type)`` scan and returns the same rows, more slowly — so only an explicit
    assertion catches it.
    """
    if _vector_index_clause() is None:
        pytest.skip("configured vector backend does not use per-bank indexes")

    bank_id = f"test_so_sql_{uuid.uuid4().hex[:8]}"
    try:
        await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_ctx)

        assert len(await _bank_indexes(memory._pool, bank_id)) == 3
    finally:
        await memory.delete_bank(bank_id, request_context=request_ctx)


@pytest.mark.asyncio
async def test_indexes_an_existing_store_owned_bank_already_has_are_left_alone(memory, request_ctx, monkeypatch):
    """Adopting a memories store must not silently drop what is already built.

    The bank is created SQL-owned so it really gets its three indexes, then the store
    takes it over — which is what a deployment that adopts a memories store after the
    fact looks like, and what flipping the extension on does to every existing bank at
    once. Dropping tens of thousands of indexes is an operator's decision with its own
    timing: this change only stops NEW ones being created, so the reconcile must leave
    these exactly where they are.

    Asserted through ``reconcile_bank_vector_indexes`` because that is the one path
    that could take them away — the maintenance operation and ``repair-bank`` both end
    up here, so a future entitlement rule that forgets store-owned banks fails here.
    """
    index_clause = _vector_index_clause()
    if index_clause is None:
        pytest.skip("configured vector backend does not use per-bank indexes")

    bank_id = f"test_so_keep_{uuid.uuid4().hex[:8]}"
    try:
        await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_ctx)
        before = await _bank_indexes(memory._pool, bank_id)
        assert len(before) == 3, "setup: the bank needs indexes that could be dropped"

        monkeypatch.setattr(memories_mod, "get_memories", lambda: InMemoryMemories())

        async with memory._pool.acquire() as conn:
            result = await reconcile_bank_vector_indexes(conn, "public", bank_id, index_clause)

        assert result.dropped == 0, f"a reconcile must not shed a store-owned bank's existing indexes, got {result}"
        assert await _bank_indexes(memory._pool, bank_id) == before
    finally:
        await memory.delete_bank(bank_id, request_context=request_ctx)


@pytest.mark.asyncio
async def test_restoring_a_bank_into_a_store_owned_deployment_creates_no_indexes(memory, request_ctx, store_owned):
    """The import seam, which does NOT go through the fresh-INSERT gate.

    A restored ``banks`` row already exists by the time the bank is set up, so import
    calls :func:`create_bank_vector_indexes` directly to give the bank the coverage the
    gate would otherwise have skipped (#2645). That call reproduces here exactly as the
    importer makes it — with the bank row already in place — because it is the one path
    that would still have built three empty indexes per restored bank after the
    creation-time guard was added.
    """
    bank_id = f"test_so_import_{uuid.uuid4().hex[:8]}"
    try:
        await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_ctx)

        backend = await memory._get_backend()
        async with memory._pool.acquire() as conn:
            internal_id = await conn.fetchval("SELECT internal_id FROM banks WHERE bank_id = $1", bank_id)
            await bank_utils.create_bank_vector_indexes(conn, bank_id, str(internal_id), ops=backend.ops)

        assert await _bank_indexes(memory._pool, bank_id) == []
    finally:
        await memory.delete_bank(bank_id, request_context=request_ctx)
