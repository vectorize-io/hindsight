"""Tests for created_after / created_before time-range filtering in recall.

Inserts memory_units with known timestamps directly via SQL, then verifies
that recall_async respects the time bounds — never returning memories
outside the requested range.

No LLM required — uses mock provider.
"""

import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.engine.retain import embedding_utils

# Tests in this file insert memory_units with shared hardcoded UUIDs and
# memory_units.id is a global PK, so parallel xdist workers running these
# tests simultaneously hit pk_memory_units conflicts. Share an xdist group
# so the eight tests serialize on the same worker.
pytestmark = pytest.mark.xdist_group("recall_time_range")

# Three points in time, each 1 hour apart
T1 = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
T3 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# Stable UUIDs for the three facts (deterministic for assertion readability)
ID_OLD = "00000000-0000-0000-0000-000000000001"
ID_MID = "00000000-0000-0000-0000-000000000002"
ID_NEW = "00000000-0000-0000-0000-000000000003"

RC = RequestContext(tenant_id="default")


async def _insert_fact(
    conn,
    *,
    fact_id: str,
    text: str,
    bank_id: str,
    embedding_str: str,
    created_at: datetime,
    updated_at: datetime | None = None,
    fact_type: str = "world",
) -> None:
    """Insert a memory_unit with a specific created_at/updated_at timestamp."""
    updated = updated_at or created_at
    await conn.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5::vector, $6, $7)
        """,
        fact_id,
        bank_id,
        text,
        fact_type,
        embedding_str,
        created_at,
        updated,
    )


@pytest_asyncio.fixture
async def seeded_memory(memory_no_llm_verify: MemoryEngine):
    """Insert three facts at T1, T2, T3 and return the engine."""
    engine = memory_no_llm_verify
    bank_id = f"test-time-range-{uuid.uuid4().hex[:8]}"

    await engine.get_bank_profile(bank_id, request_context=RC)

    # Generate real embeddings so semantic retrieval works
    embeddings = await embedding_utils.generate_embeddings_batch(
        engine.embeddings,
        ["the cat sat on the mat", "dogs are loyal animals", "birds can fly in the sky"],
    )

    def _to_str(emb: list[float]) -> str:
        return "[" + ",".join(str(v) for v in emb) + "]"

    pool = await engine._get_pool()
    async with pool.acquire() as conn:
        # Defensive cleanup: clear any rows left behind by an interrupted
        # previous run of this fixture (test process killed before teardown).
        # Without this, pk_memory_units rejects the next insert with the same
        # hardcoded IDs.
        await conn.execute(
            "DELETE FROM memory_units WHERE id IN ($1, $2, $3)",
            ID_OLD,
            ID_MID,
            ID_NEW,
        )
        await _insert_fact(
            conn,
            fact_id=ID_OLD,
            text="the cat sat on the mat",
            bank_id=bank_id,
            embedding_str=_to_str(embeddings[0]),
            created_at=T1,
            updated_at=T1,
        )
        await _insert_fact(
            conn,
            fact_id=ID_MID,
            text="dogs are loyal animals",
            bank_id=bank_id,
            embedding_str=_to_str(embeddings[1]),
            created_at=T2,
            updated_at=T2,
        )
        await _insert_fact(
            conn,
            fact_id=ID_NEW,
            text="birds can fly in the sky",
            bank_id=bank_id,
            embedding_str=_to_str(embeddings[2]),
            created_at=T3,
            updated_at=T3,
        )

    yield engine, bank_id

    await engine.delete_bank(bank_id, request_context=RC)


def _result_ids(result) -> set[str]:
    return {str(r.id) for r in result.results}


class TestRecallTimeRange:
    """Verify created_after / created_before filtering at the recall level."""

    @pytest.mark.memory_backend_incompatible
    async def test_no_filter_returns_all(self, seeded_memory):
        engine, bank_id = seeded_memory
        result = await engine.recall_async(
            bank_id=bank_id,
            query="animals and nature",
            request_context=RC,
            max_tokens=10000,
        )
        ids = _result_ids(result)
        assert ID_OLD in ids
        assert ID_MID in ids
        assert ID_NEW in ids

    @pytest.mark.memory_backend_incompatible
    async def test_created_after_excludes_old(self, seeded_memory):
        """created_after=T1 excludes fact-old (updated_at == T1, not > T1)."""
        engine, bank_id = seeded_memory
        result = await engine.recall_async(
            bank_id=bank_id,
            query="animals and nature",
            request_context=RC,
            max_tokens=10000,
            created_after=T1,
        )
        ids = _result_ids(result)
        assert ID_OLD not in ids, "fact-old (updated_at=T1) must be excluded by created_after=T1"
        assert ID_MID in ids
        assert ID_NEW in ids

    @pytest.mark.memory_backend_incompatible
    async def test_created_after_excludes_old_and_mid(self, seeded_memory):
        """created_after=T2 returns only fact-new."""
        engine, bank_id = seeded_memory
        result = await engine.recall_async(
            bank_id=bank_id,
            query="animals and nature",
            request_context=RC,
            max_tokens=10000,
            created_after=T2,
        )
        ids = _result_ids(result)
        assert ID_OLD not in ids
        assert ID_MID not in ids, "fact-mid (updated_at=T2) must be excluded by created_after=T2"
        assert ID_NEW in ids

    @pytest.mark.memory_backend_incompatible
    async def test_created_before_excludes_new(self, seeded_memory):
        """created_before=T3 excludes fact-new (updated_at == T3, not < T3)."""
        engine, bank_id = seeded_memory
        result = await engine.recall_async(
            bank_id=bank_id,
            query="animals and nature",
            request_context=RC,
            max_tokens=10000,
            created_before=T3,
        )
        ids = _result_ids(result)
        assert ID_OLD in ids
        assert ID_MID in ids
        assert ID_NEW not in ids, "fact-new (updated_at=T3) must be excluded by created_before=T3"

    @pytest.mark.memory_backend_incompatible
    async def test_created_before_excludes_mid_and_new(self, seeded_memory):
        """created_before=T2 returns only fact-old."""
        engine, bank_id = seeded_memory
        result = await engine.recall_async(
            bank_id=bank_id,
            query="animals and nature",
            request_context=RC,
            max_tokens=10000,
            created_before=T2,
        )
        ids = _result_ids(result)
        assert ID_OLD in ids
        assert ID_MID not in ids
        assert ID_NEW not in ids

    @pytest.mark.memory_backend_incompatible
    async def test_range_both_bounds(self, seeded_memory):
        """created_after=T1, created_before=T3 returns only fact-mid."""
        engine, bank_id = seeded_memory
        result = await engine.recall_async(
            bank_id=bank_id,
            query="animals and nature",
            request_context=RC,
            max_tokens=10000,
            created_after=T1,
            created_before=T3,
        )
        ids = _result_ids(result)
        assert ID_OLD not in ids
        assert ID_MID in ids, "fact-mid (T2) must be in range (T1, T3)"
        assert ID_NEW not in ids

    async def test_empty_range_returns_nothing(self, seeded_memory):
        """A range after all facts returns empty results."""
        engine, bank_id = seeded_memory
        result = await engine.recall_async(
            bank_id=bank_id,
            query="animals and nature",
            request_context=RC,
            max_tokens=10000,
            created_after=T3,
        )
        assert len(result.results) == 0, f"Expected no results after T3, got: {_result_ids(result)}"

    @pytest.mark.memory_backend_incompatible
    async def test_updated_at_catches_consolidation_updates(self, seeded_memory):
        """A fact created at T1 but updated at T3 appears with created_after=T2."""
        engine, bank_id = seeded_memory

        # Simulate consolidation updating fact-old
        pool = await engine._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE memory_units SET updated_at = $1 WHERE id = $2",
                T3,
                ID_OLD,
            )

        result = await engine.recall_async(
            bank_id=bank_id,
            query="animals and nature",
            request_context=RC,
            max_tokens=10000,
            created_after=T2,
        )
        ids = _result_ids(result)
        assert ID_OLD in ids, "fact-old created at T1, updated at T3 — created_after=T2 must find it via updated_at"
        assert ID_NEW in ids
