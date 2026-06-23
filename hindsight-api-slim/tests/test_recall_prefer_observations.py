"""Tests for the recall `prefer_observations` deduplication flag.

When the caller recalls raw facts ('world'/'experience') together with
'observation' and sets prefer_observations=True, any raw fact that a returned
observation was consolidated from (tracked via memory_units.source_memory_ids)
is dropped so the observation supersedes it — no duplicate content.

Dedup is provenance-based, not semantic: a raw fact that is semantically
similar to an observation but NOT listed in its source_memory_ids must survive.

No LLM required — inserts memory_units directly via SQL with real embeddings.
"""

import uuid

import pytest
import pytest_asyncio

from hindsight_api import MemoryEngine, RequestContext
from hindsight_api.engine.retain import embedding_utils

RC = RequestContext(tenant_id="default")

QUERY = "Alice mountain hiking"

# Two raw facts the observation is consolidated from (must be dropped when the
# flag is on), one raw fact that is semantically similar but NOT a source (must
# survive), and the observation itself.
SRC1_TEXT = "Alice loves hiking in the mountains"
SRC2_TEXT = "Alice hikes the Alps every summer"
NON_SRC_TEXT = "Alice enjoys exploring mountain hiking trails"
OBS_TEXT = "Alice is an avid mountain hiker"


async def _insert_unit(
    conn,
    *,
    unit_id: str,
    text: str,
    bank_id: str,
    embedding_str: str,
    fact_type: str = "world",
    source_memory_ids: list[uuid.UUID] | None = None,
) -> None:
    await conn.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, embedding, source_memory_ids)
        VALUES ($1, $2, $3, $4, $5::vector, $6::uuid[])
        """,
        unit_id,
        bank_id,
        text,
        fact_type,
        embedding_str,
        source_memory_ids,
    )


def _to_str(emb: list[float]) -> str:
    return "[" + ",".join(str(v) for v in emb) + "]"


def _result_ids(result) -> set[str]:
    return {str(r.id) for r in result.results}


@pytest_asyncio.fixture
async def seeded_obs_memory(memory_no_llm_verify: MemoryEngine):
    """Seed two source facts, one non-source fact, and an observation over the two sources."""
    engine = memory_no_llm_verify
    bank_id = f"test-prefer-obs-{uuid.uuid4().hex[:8]}"
    await engine.get_bank_profile(bank_id, request_context=RC)

    src1_id = str(uuid.uuid4())
    src2_id = str(uuid.uuid4())
    non_src_id = str(uuid.uuid4())
    obs_id = str(uuid.uuid4())

    embeddings = await embedding_utils.generate_embeddings_batch(
        engine.embeddings,
        [SRC1_TEXT, SRC2_TEXT, NON_SRC_TEXT, OBS_TEXT],
    )

    pool = await engine._get_pool()
    async with pool.acquire() as conn:
        await _insert_unit(conn, unit_id=src1_id, text=SRC1_TEXT, bank_id=bank_id, embedding_str=_to_str(embeddings[0]))
        await _insert_unit(conn, unit_id=src2_id, text=SRC2_TEXT, bank_id=bank_id, embedding_str=_to_str(embeddings[1]))
        await _insert_unit(
            conn, unit_id=non_src_id, text=NON_SRC_TEXT, bank_id=bank_id, embedding_str=_to_str(embeddings[2])
        )
        await _insert_unit(
            conn,
            unit_id=obs_id,
            text=OBS_TEXT,
            bank_id=bank_id,
            embedding_str=_to_str(embeddings[3]),
            fact_type="observation",
            source_memory_ids=[uuid.UUID(src1_id), uuid.UUID(src2_id)],
        )

    ids = {"src1": src1_id, "src2": src2_id, "non_src": non_src_id, "obs": obs_id}
    yield engine, bank_id, ids

    await engine.delete_bank(bank_id, request_context=RC)


class TestPreferObservations:
    async def test_disabled_returns_sources_and_observation(self, seeded_obs_memory):
        """Without the flag, the source facts AND the observation are all returned."""
        engine, bank_id, ids = seeded_obs_memory
        result = await engine.recall_async(
            bank_id=bank_id,
            query=QUERY,
            request_context=RC,
            fact_type=["world", "experience", "observation"],
            prefer_observations=False,
            max_tokens=10000,
        )
        found = _result_ids(result)
        assert ids["src1"] in found
        assert ids["src2"] in found
        assert ids["obs"] in found

    async def test_enabled_drops_source_facts_keeps_observation(self, seeded_obs_memory):
        """With the flag, the observation supersedes the facts it was consolidated from."""
        engine, bank_id, ids = seeded_obs_memory
        result = await engine.recall_async(
            bank_id=bank_id,
            query=QUERY,
            request_context=RC,
            fact_type=["world", "experience", "observation"],
            prefer_observations=True,
            max_tokens=10000,
        )
        found = _result_ids(result)
        assert ids["obs"] in found, "the observation must remain"
        assert ids["src1"] not in found, "source fact 1 is superseded by the observation"
        assert ids["src2"] not in found, "source fact 2 is superseded by the observation"

    async def test_enabled_keeps_non_source_fact(self, seeded_obs_memory):
        """Dedup is provenance-based: a similar fact NOT in source_memory_ids survives."""
        engine, bank_id, ids = seeded_obs_memory
        result = await engine.recall_async(
            bank_id=bank_id,
            query=QUERY,
            request_context=RC,
            fact_type=["world", "experience", "observation"],
            prefer_observations=True,
            max_tokens=10000,
        )
        found = _result_ids(result)
        assert ids["non_src"] in found, "a non-source fact must not be dropped, even if semantically similar"

    async def test_noop_without_observation_type(self, seeded_obs_memory):
        """The flag is a no-op when 'observation' is not among the requested types."""
        engine, bank_id, ids = seeded_obs_memory
        result = await engine.recall_async(
            bank_id=bank_id,
            query=QUERY,
            request_context=RC,
            fact_type=["world", "experience"],
            prefer_observations=True,
            max_tokens=10000,
        )
        found = _result_ids(result)
        assert ids["src1"] in found
        assert ids["src2"] in found


def test_flag_is_opt_in_by_default():
    """prefer_observations is opt-in: off at the API surface and the engine method.

    The engine default in particular must stay False so internal callers — notably
    consolidation, which needs the raw facts it folds into observations — are never
    silently deduped.
    """
    import inspect

    from hindsight_api.api.http import RecallRequest
    from hindsight_api.engine.memory_engine import MemoryEngine

    assert RecallRequest(query="anything").prefer_observations is False
    engine_default = inspect.signature(MemoryEngine.recall_async).parameters["prefer_observations"].default
    assert engine_default is False
