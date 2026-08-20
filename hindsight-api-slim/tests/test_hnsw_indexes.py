"""
Tests for per-bank vector index lifecycle and UNION ALL retrieval.

Covers:
- _bank_index_name deterministic naming
- At the shipped default (threshold off) a bank gets its three
  per-(bank, fact_type) indexes when it is created, and a retain finds them
  already there — the pre-#3485 outcome, reached the pre-#3485 way
- With a threshold set, bank creation issues no index DDL: an unearned index
  covers no rows while still being planned against by every other bank
- Per-bank vector indexes dropped on bank deletion, which is where the
  internal_id they are named after is still known
- retrieve_semantic_bm25_combined_sql groups results correctly by fact_type and source
"""

import uuid
from datetime import datetime, timezone

import pytest

from hindsight_api.engine import memory_engine as memory_engine_module
from hindsight_api.engine import vector_index_health
from hindsight_api.engine.db_utils import retry_with_backoff
from hindsight_api.engine.retain import bank_utils
from hindsight_api.engine.retain.bank_utils import (
    _BANK_INDEX_FACT_TYPES,
    _bank_index_name,
    _vector_index_clause,
)
from hindsight_api.engine.vector_index_health import CoverageTrigger


@pytest.fixture
def threshold_set(monkeypatch):
    """Turn the size threshold on for this test.

    The suite runs at the shipped default, where the threshold is off and every
    bank is given its indexes at creation. A test about what happens *with* a
    threshold has to say so — and has to patch every module that imported the
    helper by name, since a missed one silently leaves the default in place and
    turns the test into a vacuous pass.
    """
    for module in (vector_index_health, bank_utils, memory_engine_module):
        monkeypatch.setattr(module, "per_bank_indexes_are_eager", lambda: False)


# ---------------------------------------------------------------------------
# Unit tests — no DB required
# ---------------------------------------------------------------------------


class TestBankIndexName:
    def test_deterministic(self):
        uid = "550e8400-e29b-41d4-a716-446655440000"
        assert _bank_index_name("world", uid) == _bank_index_name("world", uid)

    def test_strips_dashes(self):
        uid = "550e8400-e29b-41d4-a716-446655440000"
        name = _bank_index_name("world", uid)
        # uid16 should be hex chars only
        assert "-" not in name

    def test_uses_first_16_hex_chars(self):
        uid = "550e8400-e29b-41d4-a716-446655440000"
        uid16 = uid.replace("-", "")[:16]  # "550e8400e29b41d4"
        assert name_ends_with(name=_bank_index_name("world", uid), suffix=uid16)

    def test_suffix_per_fact_type(self):
        uid = "550e8400-e29b-41d4-a716-446655440000"
        names = {ft: _bank_index_name(ft, uid) for ft in _BANK_INDEX_FACT_TYPES}
        # All three names must be distinct
        assert len(set(names.values())) == 3

    def test_all_fact_types_covered(self):
        assert set(_BANK_INDEX_FACT_TYPES) == {"world", "experience", "observation"}

    def test_fits_pg_identifier_limit(self):
        # PostgreSQL max identifier length is 63 chars
        uid = "f" * 32  # simulated UUID without dashes
        for ft in _BANK_INDEX_FACT_TYPES:
            assert len(_bank_index_name(ft, uid)) <= 63


def name_ends_with(name: str, suffix: str) -> bool:
    return name.endswith(suffix)


# ---------------------------------------------------------------------------
# Integration tests — require DB (memory fixture)
# ---------------------------------------------------------------------------


async def _get_bank_vector_indexes(pool, bank_id: str) -> list[str]:
    """Return index names for memory_units that match the per-bank pattern."""
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


async def _build_bank_vector_indexes(pool, bank_id: str) -> list[str]:
    """Give a bank its three partial indexes, as the maintenance sweep would.

    Used by the delete-path test: retain no longer creates them, so a bank has
    to be given them before deletion can be asked to take them away.
    """
    index_clause = _vector_index_clause()
    assert index_clause is not None
    async with pool.acquire() as conn:
        internal_id = str(await conn.fetchval("SELECT internal_id FROM banks WHERE bank_id = $1", bank_id))
        literal = await conn.fetchval("SELECT quote_literal($1::text)", bank_id)
        names = []
        for ft in _BANK_INDEX_FACT_TYPES:
            name = _bank_index_name(ft, internal_id)
            # CONCURRENTLY, and retried: a plain CREATE INDEX takes ShareLock on
            # the shared memory_units table, which forms a deadlock cycle with
            # another xdist worker's DROP INDEX CONCURRENTLY (ShareUpdateExclusive)
            # — observed as a three-way cycle in CI.
            await retry_with_backoff(
                lambda name=name, ft=ft: conn.execute(
                    f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} ON memory_units {index_clause} "
                    f"WHERE fact_type = '{ft}' AND bank_id = {literal}"
                )
            )
            names.append(name)
    return names


@pytest.mark.asyncio
async def test_retain_still_ends_up_with_per_bank_indexes(memory, request_context):
    """At the shipped default a retained bank has the coverage it always had.

    The threshold defaults to off, so the bank was given its three indexes when
    it was created and retain adds rows to indexes that already exist. No
    vector_index_maintenance operation is involved at all — which is the point:
    the default deployment behaves exactly as it did before the threshold.
    """
    bank_id = f"test_hnsw_create_{uuid.uuid4().hex[:8]}"
    try:
        await memory.retain_async(
            bank_id=bank_id,
            content="Alice is a software engineer.",
            request_context=request_context,
        )
        indexes = await _get_bank_vector_indexes(memory._pool, bank_id)
        assert indexes, "a retained bank should end up with per-bank vector indexes at the default threshold"
        for name in indexes:
            assert "_" in name
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_bank_creation_alone_creates_no_vector_indexes(memory, request_context, threshold_set):
    """With a threshold set, creating a bank must issue no index DDL.

    A fresh bank has earned nothing, so its three indexes would cover no rows
    while still being locked and planned against by every other bank's queries.
    That is the difference that makes bank count stop being a ceiling (#3485),
    and it is also what takes CREATE INDEX's ShareLock on the shared
    memory_units table off the request path.
    """
    bank_id = f"test_hnsw_empty_{uuid.uuid4().hex[:8]}"
    try:
        await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)

        indexes = await _get_bank_vector_indexes(memory._pool, bank_id)
        assert indexes == [], f"bank creation must not create vector indexes, got: {indexes}"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_delete_bank_drops_vector_indexes(memory, request_context):
    """delete_bank must drop the per-bank vector indexes a large bank had.

    Still the one request path that issues vector-index DDL: an index outliving
    its bank would be charged to every surviving bank's query planning forever,
    and nothing else knows the internal_id it is named after.
    """
    bank_id = f"test_hnsw_drop_{uuid.uuid4().hex[:8]}"

    await memory.retain_async(
        bank_id=bank_id,
        content="Bob is a data scientist.",
        request_context=request_context,
    )
    await _build_bank_vector_indexes(memory._pool, bank_id)
    indexes_before = await _get_bank_vector_indexes(memory._pool, bank_id)
    assert len(indexes_before) == 3, "setup: the bank should have indexes to drop"

    await memory.delete_bank(bank_id, request_context=request_context)

    indexes_after = await _get_bank_vector_indexes(memory._pool, bank_id)
    assert indexes_after == [], f"Indexes should be dropped after bank deletion, got: {indexes_after}"


@pytest.mark.asyncio
async def test_delete_bank_drops_indexes_before_data(memory, request_context, monkeypatch):
    """delete_bank must drop the per-bank vector indexes BEFORE the delete transaction.

    Regression test for #3485: when the cluster-wide index count approaches the
    Postgres lock-table budget, planning ANY statement against memory_units
    fails — including the delete DML. DROP INDEX (a utility statement) only
    locks its own index and the table, so it keeps working; dropping first is
    what keeps bank deletion usable at high bank counts.
    """
    bank_id = f"test_hnsw_order_{uuid.uuid4().hex[:8]}"
    await memory.retain_async(
        bank_id=bank_id,
        content="Dana is a research scientist.",
        request_context=request_context,
    )

    backend = await memory._get_backend()
    real = backend.ops.drop_bank_vector_indexes
    drop_rows_present: bool | None = None
    drop_profile_present: bool | None = None

    async def recording_drop(conn, schema, internal_id, fact_types):
        # At drop time the bank's memory rows and profile must still exist —
        # the drop runs before the delete transaction, not after it.
        nonlocal drop_rows_present, drop_profile_present
        drop_rows_present = (await conn.fetchval("SELECT COUNT(*) FROM memory_units WHERE bank_id = $1", bank_id)) > 0
        drop_profile_present = (
            await conn.fetchval("SELECT internal_id FROM banks WHERE bank_id = $1", bank_id)
        ) is not None
        return await real(conn, schema, internal_id, fact_types)

    monkeypatch.setattr(backend.ops, "drop_bank_vector_indexes", recording_drop)

    await memory.delete_bank(bank_id, request_context=request_context)

    assert drop_rows_present is True, "indexes must be dropped while the bank's rows still exist"
    assert drop_profile_present is True, "indexes must be dropped before the bank profile is deleted"
    assert await _get_bank_vector_indexes(memory._pool, bank_id) == []


@pytest.mark.asyncio
async def test_delete_bank_fact_type_scoped_keeps_indexes(memory, request_context):
    """A fact_type-scoped delete must not drop the bank's vector indexes.

    Only the full-delete path drops indexes: the bank survives a
    fact_type-scoped delete, so its remaining partitions still earn them, and
    nothing else knows the internal_id they are named after.
    """
    bank_id = f"test_hnsw_facttype_{uuid.uuid4().hex[:8]}"
    try:
        await memory.retain_async(
            bank_id=bank_id,
            content="Eve is a data analyst.",
            request_context=request_context,
        )
        await _build_bank_vector_indexes(memory._pool, bank_id)
        indexes_before = await _get_bank_vector_indexes(memory._pool, bank_id)
        assert len(indexes_before) == 3, "setup: the bank should have indexes to keep"

        await memory.delete_bank(bank_id, fact_type="world", request_context=request_context)

        indexes_after = await _get_bank_vector_indexes(memory._pool, bank_id)
        assert indexes_after == indexes_before, (
            f"fact_type-scoped delete must keep the bank's indexes, got: {indexes_after}"
        )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_retain_idempotent_bank_creation(memory, request_context):
    """Retaining twice must not error, and must not duplicate or rebuild indexes.

    At the shipped default the second retain must not reach the coverage
    machinery at all: no operation, and no query behind the submit either.
    """
    bank_id = f"test_hnsw_idem_{uuid.uuid4().hex[:8]}"
    try:
        await memory.retain_async(
            bank_id=bank_id,
            content="Carol is a product manager.",
            request_context=request_context,
        )
        after_first = await _get_bank_vector_indexes(memory._pool, bank_id)

        await memory.retain_async(
            bank_id=bank_id,
            content="Carol joined the company in 2022.",
            request_context=request_context,
        )

        assert await _get_bank_vector_indexes(memory._pool, bank_id) == after_first
        submitted = await memory.submit_async_vector_index_maintenance(
            bank_id=bank_id, request_context=request_context, trigger=CoverageTrigger.GREW
        )
        assert submitted["no_work"] is True, "a settled bank must stop queueing maintenance"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_retrieve_semantic_bm25_grouped_by_fact_type(memory, request_context):
    """Combined retrieval groups typed semantic and BM25 candidates by fact type."""
    from hindsight_api.engine.search.retrieval import retrieve_semantic_bm25_combined_sql

    bank_id = f"test_retrieval_{uuid.uuid4().hex[:8]}"
    try:
        await memory.retain_async(
            bank_id=bank_id,
            content=("Alice is a software engineer at TechCorp. She visited Paris in 2023 for a conference."),
            context="background",
            event_date=datetime(2023, 6, 1, tzinfo=timezone.utc),
            request_context=request_context,
        )

        query_emb = memory.embeddings.encode(["software engineer Alice"])
        query_emb_str = str(query_emb[0])

        fact_types = ["world", "experience"]
        async with memory._pool.acquire() as conn:
            results = await retrieve_semantic_bm25_combined_sql(
                conn=conn,
                query_emb_str=query_emb_str,
                query_text="software engineer Alice",
                bank_id=bank_id,
                fact_types=fact_types,
                limit=5,
            )

        # Must return an entry for every requested fact_type
        assert set(results.keys()) == set(fact_types)

        for ft, result in results.items():
            # Semantic and BM25 lists must be lists
            assert isinstance(result.semantic, list)
            assert isinstance(result.bm25, list)
            # All semantic results must declare the correct fact_type
            for r in result.semantic:
                assert r.fact_type == ft, f"Semantic result has wrong fact_type: {r.fact_type}"
            # All BM25 results must declare the correct fact_type
            for r in result.bm25:
                assert r.fact_type == ft, f"BM25 result has wrong fact_type: {r.fact_type}"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_fetch_unit_dates_ignores_noncanonical_uuid_inputs(memory, request_context):
    """The indexed UUID lookup preserves the old text-comparison input behavior."""
    from hindsight_api.engine.db.ops_postgresql import PostgreSQLOps

    bank_id = f"test_unit_dates_{uuid.uuid4().hex[:8]}"
    try:
        await memory.retain_async(
            bank_id=bank_id,
            content="Alice joined TechCorp in 2023.",
            request_context=request_context,
        )

        async with memory._pool.acquire() as conn:
            unit_id = await conn.fetchval(
                "SELECT id::text FROM memory_units WHERE bank_id = $1 ORDER BY created_at LIMIT 1",
                bank_id,
            )
            rows = await PostgreSQLOps().fetch_unit_dates(
                conn,
                "memory_units",
                [unit_id, "not-a-uuid", unit_id.upper(), unit_id.replace("-", "")],
            )

        assert [str(row["id"]) for row in rows] == [unit_id]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_recall_reuses_semantic_pool_for_graph_seeds(memory, request_context, monkeypatch):
    """Default recall must not issue a second ANN query for graph entry points."""
    from hindsight_api.engine.search import link_expansion_retrieval

    async def fail_find_semantic_seeds(*args, **kwargs):
        raise AssertionError("default recall should reuse the combined semantic candidate pool")

    bank_id = f"test_graph_seed_reuse_{uuid.uuid4().hex[:8]}"
    try:
        await memory.retain_async(
            bank_id=bank_id,
            content="Alice is a software engineer at TechCorp.",
            request_context=request_context,
        )
        monkeypatch.setattr(link_expansion_retrieval, "_find_semantic_seeds", fail_find_semantic_seeds)

        result = await memory.recall_async(
            bank_id=bank_id,
            query="Where does Alice work?",
            fact_type=["world"],
            request_context=request_context,
        )

        assert result.results
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
# Asserts *how* the graph arm seeds — that recall calls link_expansion_retrieval's
# _find_semantic_seeds — rather than what it returns. A store with its own graph
# retrieval never goes through that function, so the assertion is specific to the
# SQL retrieval path.
@pytest.mark.memory_backend_incompatible
async def test_recall_keeps_graph_seed_query_for_stricter_semantic_floor(memory, request_context, monkeypatch):
    """A semantic floor above the graph floor must retain the dedicated seed query."""
    from hindsight_api.engine.response_models import MinScores
    from hindsight_api.engine.search import link_expansion_retrieval

    original_find_semantic_seeds = link_expansion_retrieval._find_semantic_seeds
    graph_seed_fact_types: list[str] = []

    async def record_find_semantic_seeds(*args, **kwargs):
        graph_seed_fact_types.append(args[3])
        return await original_find_semantic_seeds(*args, **kwargs)

    bank_id = f"test_graph_seed_fallback_{uuid.uuid4().hex[:8]}"
    try:
        await memory.retain_async(
            bank_id=bank_id,
            content="Alice is a software engineer at TechCorp.",
            request_context=request_context,
        )
        monkeypatch.setattr(link_expansion_retrieval, "_find_semantic_seeds", record_find_semantic_seeds)

        await memory.recall_async(
            bank_id=bank_id,
            query="Where does Alice work?",
            fact_type=["world"],
            min_scores=MinScores(semantic=0.9),
            request_context=request_context,
        )

        assert graph_seed_fact_types == ["world"]
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
