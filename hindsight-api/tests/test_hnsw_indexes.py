"""
Tests for per-bank HNSW index lifecycle and UNION ALL retrieval.

Covers:
- _hnsw_index_name deterministic naming
- Per-bank HNSW indexes created on bank creation (retain_async / ensure_bank_exists)
- Per-bank HNSW indexes dropped on bank deletion
- retrieve_semantic_bm25_combined groups results correctly by fact_type and source
"""
import uuid
from datetime import datetime, timezone

import pytest

from hindsight_api.engine.retain.bank_utils import _HNSW_FACT_TYPES, _hnsw_index_name


# ---------------------------------------------------------------------------
# Unit tests — no DB required
# ---------------------------------------------------------------------------


class TestHnswIndexName:
    def test_deterministic(self):
        uid = "550e8400-e29b-41d4-a716-446655440000"
        assert _hnsw_index_name("world", uid) == _hnsw_index_name("world", uid)

    def test_strips_dashes(self):
        uid = "550e8400-e29b-41d4-a716-446655440000"
        name = _hnsw_index_name("world", uid)
        # uid16 should be hex chars only
        assert "-" not in name

    def test_uses_first_16_hex_chars(self):
        uid = "550e8400-e29b-41d4-a716-446655440000"
        uid16 = uid.replace("-", "")[:16]  # "550e8400e29b41d4"
        assert name_ends_with(name=_hnsw_index_name("world", uid), suffix=uid16)

    def test_suffix_per_fact_type(self):
        uid = "550e8400-e29b-41d4-a716-446655440000"
        names = {ft: _hnsw_index_name(ft, uid) for ft in _HNSW_FACT_TYPES}
        # All three names must be distinct
        assert len(set(names.values())) == 3

    def test_all_fact_types_covered(self):
        assert set(_HNSW_FACT_TYPES) == {"world", "experience", "observation"}

    def test_fits_pg_identifier_limit(self):
        # PostgreSQL max identifier length is 63 chars
        uid = "f" * 32  # simulated UUID without dashes
        for ft in _HNSW_FACT_TYPES:
            assert len(_hnsw_index_name(ft, uid)) <= 63


def name_ends_with(name: str, suffix: str) -> bool:
    return name.endswith(suffix)


# ---------------------------------------------------------------------------
# Integration tests — require DB (memory fixture)
# ---------------------------------------------------------------------------


async def _get_bank_hnsw_indexes(pool, bank_id: str) -> list[str]:
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


@pytest.mark.asyncio
async def test_retain_creates_per_bank_hnsw_indexes(memory, request_context):
    """retain_async on a new bank must create 3 per-(bank, fact_type) HNSW indexes."""
    bank_id = f"test_hnsw_create_{uuid.uuid4().hex[:8]}"
    try:
        await memory.retain_async(
            bank_id=bank_id,
            content="Alice is a software engineer.",
            request_context=request_context,
        )
        indexes = await _get_bank_hnsw_indexes(memory._pool, bank_id)
        assert len(indexes) == 3, f"Expected 3 per-bank HNSW indexes, got: {indexes}"
        for ft_short in _HNSW_FACT_TYPES.values():
            assert any(ft_short in idx for idx in indexes), (
                f"Missing index for fact_type short '{ft_short}' in {indexes}"
            )
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_delete_bank_drops_hnsw_indexes(memory, request_context):
    """delete_bank must drop all per-bank HNSW indexes."""
    bank_id = f"test_hnsw_drop_{uuid.uuid4().hex[:8]}"

    await memory.retain_async(
        bank_id=bank_id,
        content="Bob is a data scientist.",
        request_context=request_context,
    )
    # Verify indexes exist before deletion
    indexes_before = await _get_bank_hnsw_indexes(memory._pool, bank_id)
    assert len(indexes_before) == 3

    await memory.delete_bank(bank_id, request_context=request_context)

    indexes_after = await _get_bank_hnsw_indexes(memory._pool, bank_id)
    assert indexes_after == [], f"Indexes should be dropped after bank deletion, got: {indexes_after}"


@pytest.mark.asyncio
async def test_retain_idempotent_bank_creation(memory, request_context):
    """Retaining into the same bank twice must not error and still have exactly 3 indexes."""
    bank_id = f"test_hnsw_idem_{uuid.uuid4().hex[:8]}"
    try:
        await memory.retain_async(
            bank_id=bank_id,
            content="Carol is a product manager.",
            request_context=request_context,
        )
        await memory.retain_async(
            bank_id=bank_id,
            content="Carol joined the company in 2022.",
            request_context=request_context,
        )
        indexes = await _get_bank_hnsw_indexes(memory._pool, bank_id)
        assert len(indexes) == 3
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
async def test_retrieve_semantic_bm25_grouped_by_fact_type(memory, request_context):
    """
    retrieve_semantic_bm25_combined must return a dict keyed by fact_type with
    (semantic_list, bm25_list) tuples.  All returned facts must belong to their
    declared fact_type.
    """
    from hindsight_api.engine.search.retrieval import retrieve_semantic_bm25_combined

    bank_id = f"test_retrieval_{uuid.uuid4().hex[:8]}"
    try:
        await memory.retain_async(
            bank_id=bank_id,
            content=(
                "Alice is a software engineer at TechCorp. "
                "She visited Paris in 2023 for a conference."
            ),
            context="background",
            event_date=datetime(2023, 6, 1, tzinfo=timezone.utc),
            request_context=request_context,
        )

        query_emb = memory.embeddings.encode(["software engineer Alice"])
        query_emb_str = str(query_emb[0])

        fact_types = ["world", "experience"]
        async with memory._pool.acquire() as conn:
            results = await retrieve_semantic_bm25_combined(
                conn=conn,
                query_emb_str=query_emb_str,
                query_text="software engineer Alice",
                bank_id=bank_id,
                fact_types=fact_types,
                limit=5,
            )

        # Must return an entry for every requested fact_type
        assert set(results.keys()) == set(fact_types)

        for ft, (sem, bm25) in results.items():
            # Semantic and BM25 lists must be lists
            assert isinstance(sem, list)
            assert isinstance(bm25, list)
            # All semantic results must declare the correct fact_type
            for r in sem:
                assert r.fact_type == ft, f"Semantic result has wrong fact_type: {r.fact_type}"
            # All BM25 results must declare the correct fact_type
            for r in bm25:
                assert r.fact_type == ft, f"BM25 result has wrong fact_type: {r.fact_type}"

    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
