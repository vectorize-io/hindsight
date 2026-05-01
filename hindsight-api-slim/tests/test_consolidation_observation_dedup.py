"""Unit test for observation deduplication during consolidation (#1284 fix).

This test validates the new _find_semantic_duplicate_observation and
_merge_into_observation helpers without needing a real PostgreSQL instance
or sentence-transformers — it mocks the connection and the embedding call.
"""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from hindsight_api.engine.consolidation.consolidator import (
    _find_semantic_duplicate_observation,
    _merge_into_observation,
    _create_observation_directly,
)
from hindsight_api.engine.memory_engine import fq_table


@pytest.fixture
def mock_conn():
    conn = AsyncMock()
    conn.backend_type = "postgresql"
    return conn


@pytest.fixture
def mock_memory_engine():
    mem = MagicMock()
    mem._backend.ops.uses_observation_sources_table = False
    mem.embeddings = MagicMock()
    return mem


class TestFindSemanticDuplicateObservation:
    """Tests for _find_semantic_duplicate_observation."""

    async def test_no_match_below_threshold(self, mock_conn):
        mock_conn.fetchrow = AsyncMock(return_value=None)
        result = await _find_semantic_duplicate_observation(
            mock_conn, "bank-1", "[0.1, 0.2]", threshold=0.95
        )
        assert result is None
        # Verify query was executed with correct bank_id
        args = mock_conn.fetchrow.call_args[0]
        assert "bank_id = $2" in mock_conn.fetchrow.call_args[0][0]
        assert args[1] == "[0.1, 0.2]"
        assert args[2] == "bank-1"

    async def test_match_above_threshold(self, mock_conn):
        mock_conn.fetchrow = AsyncMock(
            return_value={"id": UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890"), "similarity": 0.95}
        )
        result = await _find_semantic_duplicate_observation(
            mock_conn, "bank-1", "[0.1, 0.2]", threshold=0.90
        )
        assert result is not None
        assert result[0] == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert result[1] == 0.95

    async def test_match_with_tags_filter(self, mock_conn):
        mock_conn.fetchrow = AsyncMock(
            return_value={"id": UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890"), "similarity": 0.93}
        )
        result = await _find_semantic_duplicate_observation(
            mock_conn, "bank-1", "[0.1, 0.2]", tags=["user:42"], threshold=0.90
        )
        assert result is not None
        # Verify tags filter was injected into the query
        sql = mock_conn.fetchrow.call_args[0][0]
        assert "tags" in sql.lower()
        # Verify params contain raw list (not json.dumps)
        params = mock_conn.fetchrow.call_args[0][1:]
        assert ["user:42"] in params

    async def test_threshold_zero_disabled(self, mock_conn):
        """When threshold is 0, the helper should still execute but accept any match."""
        mock_conn.fetchrow = AsyncMock(
            return_value={"id": UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890"), "similarity": 0.5}
        )
        result = await _find_semantic_duplicate_observation(
            mock_conn, "bank-1", "[0.1, 0.2]", threshold=0.0
        )
        assert result is not None
        assert result[1] == 0.5


class TestMergeIntoObservation:
    """Tests for _merge_into_observation."""

    async def test_merge_updates_fields(self, mock_conn, mock_memory_engine):
        obs_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        existing_text = "Original observation text"
        existing_tags = ["user:42"]
        existing_sources = [UUID("11111111-1111-1111-1111-111111111111")]

        mock_conn.fetchrow = AsyncMock(
            return_value={
                "text": existing_text,
                "source_memory_ids": existing_sources,
                "tags": existing_tags,
                "occurred_start": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "occurred_end": datetime(2024, 1, 2, tzinfo=timezone.utc),
                "mentioned_at": datetime(2024, 1, 3, tzinfo=timezone.utc),
            }
        )
        mock_conn.execute = AsyncMock()

        # Patch the embedding call to avoid sentence-transformers dependency
        with patch(
            "hindsight_api.engine.consolidation.consolidator.embedding_utils.generate_embeddings_batch",
            new=AsyncMock(return_value=[[0.1, 0.2, 0.3]]),
        ):
            await _merge_into_observation(
                conn=mock_conn,
                memory_engine=mock_memory_engine,
                bank_id="bank-1",
                observation_id=obs_id,
                new_text="Updated observation text",
                source_memory_ids=[UUID("22222222-2222-2222-2222-222222222222")],
                source_fact_tags=["session:99"],
                source_occurred_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                source_occurred_end=datetime(2024, 1, 5, tzinfo=timezone.utc),
                source_mentioned_at=datetime(2024, 1, 4, tzinfo=timezone.utc),
            )

        # Verify UPDATE was issued
        assert mock_conn.execute.call_count == 1
        sql = mock_conn.execute.call_args[0][0]
        assert "UPDATE" in sql.upper()

        # Verify the new text and merged tags
        args = mock_conn.execute.call_args[0]
        assert args[1] == "Updated observation text"
        # tags should be merged
        merged_tags = args[10]
        assert set(merged_tags) == {"user:42", "session:99"}

    async def test_merge_skips_deleted_sources(self, mock_conn, mock_memory_engine):
        """If all new source memories are dead, the merge should be skipped."""
        mock_conn.fetchrow = AsyncMock(return_value=None)
        # Override _filter_live_source_memories to return empty
        with patch(
            "hindsight_api.engine.consolidation.consolidator._filter_live_source_memories",
            new=AsyncMock(return_value=[]),
        ):
            await _merge_into_observation(
                conn=mock_conn,
                memory_engine=mock_memory_engine,
                bank_id="bank-1",
                observation_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                new_text="Updated text",
                source_memory_ids=[UUID("22222222-2222-2222-2222-222222222222")],
            )
        # No UPDATE should be issued
        mock_conn.execute.assert_not_called()


class TestCreateObservationDirectlyDedup:
    """Tests that _create_observation_directly respects the dedup_threshold."""

    async def test_dedup_path(self, mock_conn, mock_memory_engine):
        """When a duplicate exists, _create_observation_directly returns 'updated'."""
        obs_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

        # First, _find_semantic_duplicate_observation will be called
        # Second, _merge_into_observation will be called
        with patch(
            "hindsight_api.engine.consolidation.consolidator._find_semantic_duplicate_observation",
            new=AsyncMock(return_value=(obs_id, 0.95)),
        ):
            with patch(
                "hindsight_api.engine.consolidation.consolidator._merge_into_observation",
                new=AsyncMock(),
            ):
                result = await _create_observation_directly(
                    conn=mock_conn,
                    memory_engine=mock_memory_engine,
                    bank_id="bank-1",
                    source_memory_ids=[UUID("22222222-2222-2222-2222-222222222222")],
                    observation_text="Duplicate text",
                    dedup_threshold=0.90,
                )
        assert result["action"] == "updated"
        assert result["observation_id"] == obs_id

    async def test_no_dedup_when_threshold_zero(self, mock_conn, mock_memory_engine):
        """When threshold is 0, the normal INSERT path runs."""
        mock_conn.fetchrow = AsyncMock(return_value={"id": UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")})

        # Patch the embedding call
        with patch(
            "hindsight_api.engine.consolidation.consolidator.embedding_utils.generate_embeddings_batch",
            new=AsyncMock(return_value=[[0.1, 0.2, 0.3]]),
        ):
            result = await _create_observation_directly(
                conn=mock_conn,
                memory_engine=mock_memory_engine,
                bank_id="bank-1",
                source_memory_ids=[UUID("22222222-2222-2222-2222-222222222222")],
                observation_text="New text",
                dedup_threshold=0.0,
            )
        assert result["action"] == "created"
        # An INSERT should have been issued
        sql = mock_conn.fetchrow.call_args[0][0]
        assert "INSERT" in sql.upper()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
