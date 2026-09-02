"""Deterministic tests for the dedicated observation_history table.

The history of an observation lives in its own table (one row per change),
written by consolidator._append_observation_history and read back by
MemoryEngine.get_observation_history. These tests exercise that path directly
(no LLM) so the array/timestamptz binding and the per-observation cap are
covered without depending on consolidation deciding to issue an UPDATE.
"""

import uuid
from typing import Any

import pytest

from hindsight_api.config import clear_config_cache
from hindsight_api.engine.consolidation import consolidator as consolidator_mod
from hindsight_api.engine.consolidation.consolidator import _ObservationHistorySnapshot
from hindsight_api.engine.memory_engine import MemoryEngine


def _entry(i: int) -> _ObservationHistorySnapshot:
    return _ObservationHistorySnapshot(
        previous_text=f"v{i}",
        previous_tags=[f"tag{i}"],
        previous_occurred_start=None,
        previous_occurred_end=None,
        previous_mentioned_at="2025-01-01T00:00:00Z",
        new_source_memory_ids=[],
    )


@pytest.mark.asyncio
class TestObservationHistory:
    # Seeds the observation with a raw INSERT INTO memory_units, which a store-owned bank leaves
    # empty — the store has no such observation, so the engine correctly returns None.
    @pytest.mark.memory_backend_incompatible
    async def test_append_read_and_cap(
        self, memory: MemoryEngine, request_context: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Snapshots are appended as rows, returned oldest-first, and trimmed to
        the configured cap (the oldest over-cap rows are deleted on each write)."""
        monkeypatch.setenv("HINDSIGHT_API_OBSERVATION_HISTORY_MAX_ENTRIES", "3")
        clear_config_cache()

        bank_id = f"test-obs-hist-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        obs_id = uuid.uuid4()
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memory_units (id, bank_id, text, event_date, fact_type)
                VALUES ($1, $2, 'current observation', now(), 'observation')
                """,
                obs_id,
                bank_id,
            )
            # 5 updates → 5 snapshots → trimmed to the most recent 3.
            for i in range(1, 6):
                await consolidator_mod._append_observation_history(conn, bank_id, str(obs_id), _entry(i), 3)

        history = await memory.get_observation_history(bank_id, str(obs_id), request_context=request_context)

        # Capped to the most-recent 3 (v3, v4, v5), returned oldest-first.
        assert [h["previous_text"] for h in history] == ["v3", "v4", "v5"]
        # Array column round-trips (the binding the mental-model path doesn't exercise).
        assert history[0]["previous_tags"] == ["tag3"]

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_history_is_reclaimed_when_observations_are_cleared(
        self, memory: MemoryEngine, request_context: Any
    ) -> None:
        """Regression: the observation_history→memory_units FK was dropped (so history can be
        recorded for observations kept outside SQL), which also removed the cascade that used to
        reclaim history when an observation went away. clear_observations must now delete the
        history explicitly — otherwise rows accumulate on the default deployment forever."""
        bank_id = f"test-obs-hist-clear-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        obs_id = uuid.uuid4()
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memory_units (id, bank_id, text, event_date, fact_type)
                VALUES ($1, $2, 'current observation', now(), 'observation')
                """,
                obs_id,
                bank_id,
            )
            await consolidator_mod._append_observation_history(conn, bank_id, str(obs_id), _entry(1), 5)
            before = await conn.fetchval("SELECT COUNT(*) FROM observation_history WHERE bank_id = $1", bank_id)
        assert before == 1

        await memory.clear_observations(bank_id, request_context=request_context)

        async with pool.acquire() as conn:
            after = await conn.fetchval("SELECT COUNT(*) FROM observation_history WHERE bank_id = $1", bank_id)
        assert after == 0, "observation_history rows leaked after clear_observations"

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_history_is_reclaimed_when_observations_deleted_by_type(
        self, memory: MemoryEngine, request_context: Any
    ) -> None:
        """The typed delete (``DELETE /banks/{id}/memories?type=observation``) removes the
        observation rows directly, bypassing the stale-observation sweep — so it must clean the
        history itself, or the snapshots outlive the observations they belonged to."""
        bank_id = f"test-obs-hist-type-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)

        obs_id = uuid.uuid4()
        pool = await memory._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memory_units (id, bank_id, text, event_date, fact_type)
                VALUES ($1, $2, 'current observation', now(), 'observation')
                """,
                obs_id,
                bank_id,
            )
            await consolidator_mod._append_observation_history(conn, bank_id, str(obs_id), _entry(1), 5)

        await memory.delete_bank(bank_id, fact_type="observation", request_context=request_context)

        async with pool.acquire() as conn:
            after = await conn.fetchval("SELECT COUNT(*) FROM observation_history WHERE bank_id = $1", bank_id)
        assert after == 0, "observation_history rows leaked after typed observation delete"

        await memory.delete_bank(bank_id, request_context=request_context)

    async def test_returns_none_for_missing_observation(self, memory: MemoryEngine, request_context: Any) -> None:
        bank_id = f"test-obs-hist-{uuid.uuid4().hex[:8]}"
        await memory.get_bank_profile(bank_id, request_context=request_context)
        result = await memory.get_observation_history(bank_id, str(uuid.uuid4()), request_context=request_context)
        assert result is None
        await memory.delete_bank(bank_id, request_context=request_context)
