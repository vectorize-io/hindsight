"""Tests for consolidation_llm_parallelism (intra-op tag-group parallelism).

The parallel dispatcher uses the *tag group* as the unit of concurrency: batches
within a group run serially (they share an observation scope), while distinct
groups run concurrently. These tests prove that running with parallelism > 1
produces the same result as the sequential default — i.e. the deferred per-batch
stat merge has no lost-update / double-count race and drops no memories.
"""

import uuid
from unittest.mock import patch

import pytest

from hindsight_api.config import HindsightConfig, _get_raw_config
from hindsight_api.engine.consolidation.consolidator import run_consolidation_job
from hindsight_api.engine.memory_engine import MemoryEngine

# Distinct tags per item → one tag group per memory. This (a) exercises the
# parallel dispatch path (parallel unit = tag group, needs > 1 group) and
# (b) keeps observation scopes disjoint, so create/update decisions don't depend
# on processing order — making the sequential/parallel comparison deterministic.
_CORPUS = [
    ("Alice prefers tea over coffee.", ["alice"]),
    ("Bob bikes to work every day.", ["bob"]),
    ("Carol studies astrophysics at night.", ["carol"]),
    ("Dave collects vinyl records from the 70s.", ["dave"]),
    ("Erin runs marathons on the weekend.", ["erin"]),
    ("Frank plays the cello in an orchestra.", ["frank"]),
]

_STAT_KEYS = [
    "memories_processed",
    "observations_created",
    "observations_updated",
    "observations_merged",
    "observations_deleted",
    "actions_executed",
    "skipped",
    "memories_failed",
]


@pytest.fixture(autouse=True)
def enable_observations():
    config = _get_raw_config()
    original = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original


def _make_config(**overrides):
    raw = _get_raw_config()
    return type(raw)(
        **{
            **{f: getattr(raw, f) for f in raw.__dataclass_fields__},
            **overrides,
        }
    )


async def _seed_corpus(memory: MemoryEngine, bank_id: str, request_context) -> None:
    """Retain the corpus with consolidation disabled so a backlog builds up."""
    await memory.get_bank_profile(bank_id=bank_id, request_context=request_context)
    no_obs = _make_config(enable_observations=False)
    with patch.object(memory._config_resolver, "resolve_full_config", return_value=no_obs):
        for content, tags in _CORPUS:
            await memory.retain_batch_async(
                bank_id=bank_id,
                contents=[{"content": content}],
                document_tags=tags,
                request_context=request_context,
            )


async def _consolidate(memory: MemoryEngine, bank_id: str, parallelism: int, request_context) -> dict:
    # llm_batch_size=1 forces one batch per memory so the dispatcher sees the
    # full set of tag groups (one per memory here).
    cfg = _make_config(consolidation_llm_parallelism=parallelism, consolidation_llm_batch_size=1)
    with (
        patch.object(memory._config_resolver, "resolve_full_config", return_value=cfg),
        patch.object(memory, "submit_async_consolidation"),
    ):
        return await run_consolidation_job(
            memory_engine=memory,
            bank_id=bank_id,
            request_context=request_context,
        )


async def _count_unconsolidated(memory: MemoryEngine, bank_id: str) -> int:
    async with memory._pool.acquire() as conn:
        return await conn.fetchval(
            """
            SELECT COUNT(*) FROM memory_units
            WHERE bank_id = $1 AND consolidated_at IS NULL
              AND consolidation_failed_at IS NULL AND fact_type IN ('experience', 'world')
            """,
            bank_id,
        )


@pytest.mark.asyncio
async def test_parallel_matches_sequential(memory: MemoryEngine, request_context):
    """Consolidating the same corpus with parallelism=4 yields identical stats to
    parallelism=1, and leaves no memory unconsolidated."""
    seq_bank = f"test-consol-seq-{uuid.uuid4().hex[:8]}"
    par_bank = f"test-consol-par-{uuid.uuid4().hex[:8]}"
    await _seed_corpus(memory, seq_bank, request_context)
    await _seed_corpus(memory, par_bank, request_context)

    # Sanity: the corpus formed more than one tag group, so the parallel branch
    # (parallel unit = tag group) actually engages.
    async with memory._pool.acquire() as conn:
        distinct_groups = await conn.fetchval(
            """
            SELECT COUNT(DISTINCT tags::text) FROM memory_units
            WHERE bank_id = $1 AND consolidated_at IS NULL AND fact_type IN ('experience', 'world')
            """,
            par_bank,
        )
    assert distinct_groups > 1, f"expected multiple tag groups to exercise parallel dispatch, got {distinct_groups}"

    seq_result = await _consolidate(memory, seq_bank, 1, request_context)
    par_result = await _consolidate(memory, par_bank, 4, request_context)

    assert seq_result["status"] == "completed"
    assert par_result["status"] == "completed"

    seq_stats = {k: seq_result.get(k) for k in _STAT_KEYS}
    par_stats = {k: par_result.get(k) for k in _STAT_KEYS}
    assert par_stats == seq_stats, f"parallel stats {par_stats} != sequential {seq_stats}"
    assert par_result["memories_processed"] > 0

    # Neither run dropped a memory.
    assert await _count_unconsolidated(memory, seq_bank) == 0
    assert await _count_unconsolidated(memory, par_bank) == 0

    await memory.delete_bank(seq_bank, request_context=request_context)
    await memory.delete_bank(par_bank, request_context=request_context)


def test_parallelism_registered_as_configurable():
    """The flag is per-bank overridable (hierarchical), like consolidation_llm_batch_size."""
    assert "consolidation_llm_parallelism" in HindsightConfig._CONFIGURABLE_FIELDS
    # Defaults to sequential.
    assert _get_raw_config().consolidation_llm_parallelism >= 1
