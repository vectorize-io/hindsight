"""The consolidation round is picked fairly across scope groups (#4823).

A strictly oldest-first round holds only the group that owns the oldest facts, and the
dispatcher parallelises across *groups* — so one group in the round means one LLM call at a
time whatever ``consolidation_llm_parallelism`` says, and the other groups wait for that
group's whole backlog. Two levels:

* ``_fair_group_slice`` in isolation — deterministic, no LLM, no database.
* the real job, asserting the composition of the rounds it actually fetched.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import pytest

from hindsight_api.engine.consolidation import consolidator as C
from hindsight_api.engine.memories import StoredMemory
from hindsight_api.engine.memory_engine import MemoryEngine
from tests.test_consolidation_scope_parallelism import (  # noqa: F401  (fixture import)
    _mock_llm_one_obs_per_fact,
    _override_config,
    enable_observations,
)


def _mem(tags: list[str], observation_scopes: Any = None) -> StoredMemory:
    """A candidate fact carrying only what the grouping key reads: tags and scope spec."""
    return StoredMemory(
        unit_id=str(uuid.uuid4()),
        text="a fact",
        fact_type="experience",
        tags=tags,
        observation_scopes=observation_scopes,
    )


def _keys(memories: list[StoredMemory]) -> list[tuple[str, ...]]:
    return [C._consolidation_batch_key({"tags": m.tags, "observation_scopes": m.observation_scopes}) for m in memories]


class TestFairGroupSlice:
    def test_one_group_holding_the_oldest_facts_does_not_take_the_whole_round(self):
        # 20 untagged facts first (the draining backlog), then one fact in each of 3 groups.
        oldest_first = [_mem([]) for _ in range(20)] + [_mem([f"user:{n}"]) for n in ("alice", "bob", "carol")]

        taken = C._fair_group_slice(oldest_first, limit=8, quota=2)

        assert len(set(_keys(taken))) == 4, "every waiting group should get a slot in the round"
        assert sum(1 for m in taken if not m.tags) == 2, "the big group is capped at its quota"

    def test_fills_the_round_from_the_groups_that_have_facts(self):
        # Only two groups exist, so the quota cannot fill an 8-fact round: take what there is.
        oldest_first = [_mem([]) for _ in range(20)] + [_mem(["user:alice"]) for _ in range(20)]

        taken = C._fair_group_slice(oldest_first, limit=8, quota=2)

        assert len(taken) == 4
        assert len(set(_keys(taken))) == 2

    def test_stops_at_the_limit(self):
        taken = C._fair_group_slice([_mem([f"user:{i}"]) for i in range(50)], limit=8, quota=2)
        assert len(taken) == 8

    def test_keys_on_the_resolved_scope_not_raw_tags(self):
        # Two memories with different tags both target the untagged ``shared`` scope, so they
        # are ONE group and share the quota — the grouping key the dispatcher uses, not tags.
        oldest_first = [_mem(["user:alice"], "shared"), _mem(["user:bob"], "shared")]

        taken = C._fair_group_slice(oldest_first, limit=8, quota=1)

        assert len(taken) == 1


async def _insert(conn, bank_id: str, text: str, tags: list[str], created_at: datetime) -> None:
    """Insert one experience memory at an exact ``created_at``.

    Direct SQL because the point of the test is the *arrival order* of facts across groups,
    and no public write path lets a caller backdate a fact — retain stamps ``now()``.
    """
    await conn.execute(
        """
        INSERT INTO memory_units (id, bank_id, text, fact_type, tags, created_at)
        VALUES ($1, $2, $3, 'experience', $4, $5)
        """,
        uuid.uuid4(),
        bank_id,
        text,
        tags,
        created_at,
    )


@pytest.mark.asyncio
@pytest.mark.memory_backend_incompatible
async def test_round_holds_every_waiting_group_not_just_the_oldest(memory: MemoryEngine, request_context):
    """The first round must contain all four groups, so four LLM calls can run at once.

    Before the fair fetch, rounds 1-2 held only the untagged group — one LLM call at a time
    with three parallel slots idle, while alice/bob/carol waited for the backlog to drain.
    """
    bank_id = f"test-fair-fetch-{uuid.uuid4().hex[:8]}"
    await memory.ensure_bank_profile(bank_id=bank_id, request_context=request_context)
    try:
        base = datetime.now(timezone.utc) - timedelta(hours=2)
        async with memory._pool.acquire() as conn:
            for i in range(8):  # the big group owns every one of the oldest facts
                await _insert(conn, bank_id, f"untagged fact {i}", [], base + timedelta(seconds=i))
            for i, name in enumerate(("alice", "bob", "carol")):
                await _insert(
                    conn, bank_id, f"{name} likes tea", [f"user:{name}"], base + timedelta(minutes=1, seconds=i)
                )

        rounds: list[set[tuple[str, ...]]] = []
        original_fetch = C._fetch_unconsolidated_rows

        async def spy(*args, **kwargs):
            rows = await original_fetch(*args, **kwargs)
            rounds.append({C._consolidation_batch_key(r) for r in rows})
            return rows

        wrapper, _ = _mock_llm_one_obs_per_fact()
        original_llm = memory._consolidation_llm_config
        memory._consolidation_llm_config = wrapper
        try:
            with (
                _override_config(
                    memory,
                    consolidation_batch_size=4,
                    consolidation_llm_batch_size=1,
                    consolidation_llm_parallelism=4,
                ),
                patch.object(memory, "submit_async_consolidation"),
                patch.object(C, "_fetch_unconsolidated_rows", spy),
            ):
                result = await C.run_consolidation_job(
                    memory_engine=memory, bank_id=bank_id, request_context=request_context
                )
        finally:
            memory._consolidation_llm_config = original_llm

        assert result["status"] == "completed"
        assert len(rounds[0]) == 4, f"first round should hold all four groups, held {sorted(rounds[0])}"
        # Fairness must not lose or skip facts: all 11 still consolidate.
        assert result["memories_processed"] == 11
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
