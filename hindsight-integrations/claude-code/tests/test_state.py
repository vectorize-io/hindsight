"""Unit tests for lib/state.py — retention tracking and compaction detection."""

import json

import pytest
from lib.state import commit_retention, plan_retention, read_state, track_retention, write_state


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path):
    """Point all state operations at a temp directory."""
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path))


# ---------------------------------------------------------------------------
# track_retention — core compaction detection
# ---------------------------------------------------------------------------


class TestTrackRetention:
    def test_first_call_returns_chunk_zero(self):
        progress = track_retention("sess-1", 10)
        assert progress.chunk_index == 0
        assert progress.compacted is False
        assert progress.start_index == 0

    def test_growing_transcript_advances_chunk_and_starts_at_last_count(self):
        track_retention("sess-1", 4)
        progress = track_retention("sess-1", 8)
        assert progress.chunk_index == 1
        assert progress.compacted is False
        assert progress.start_index == 4

    def test_equal_count_keeps_same_chunk_and_skips_all_messages(self):
        track_retention("sess-1", 5)
        progress = track_retention("sess-1", 5)
        assert progress.chunk_index == 0
        assert progress.compacted is False
        assert progress.start_index == 5

    def test_shrinking_transcript_triggers_compaction(self):
        track_retention("sess-1", 10)
        progress = track_retention("sess-1", 3)
        assert progress.chunk_index == 1
        assert progress.compacted is True
        assert progress.start_index == 0

    def test_multiple_compactions_increment_chunk(self):
        track_retention("sess-1", 10)

        progress = track_retention("sess-1", 3)
        assert progress.chunk_index == 1
        assert progress.compacted is True
        assert progress.start_index == 0

        # Grow again after compaction
        track_retention("sess-1", 8)

        # Second compaction
        progress = track_retention("sess-1", 2)
        assert progress.chunk_index == 3
        assert progress.compacted is True
        assert progress.start_index == 0

    def test_growth_after_compaction_advances_to_next_delta_chunk(self):
        track_retention("sess-1", 10)
        track_retention("sess-1", 3)  # compaction -> chunk 1

        progress = track_retention("sess-1", 6)
        assert progress.chunk_index == 2
        assert progress.compacted is False
        assert progress.start_index == 3

    def test_sessions_are_independent(self):
        track_retention("sess-a", 10)
        track_retention("sess-b", 20)

        # Compaction on sess-a only
        progress_a = track_retention("sess-a", 3)
        progress_b = track_retention("sess-b", 25)

        assert progress_a.chunk_index == 1
        assert progress_a.compacted is True
        assert progress_a.start_index == 0
        assert progress_b.chunk_index == 1
        assert progress_b.compacted is False
        assert progress_b.start_index == 20

    def test_persists_across_calls(self, tmp_path):
        """State file is written to disk and survives between calls."""
        track_retention("sess-1", 10)

        # Verify the state file exists
        state_file = tmp_path / "state" / "retention_tracking.json"
        assert state_file.exists()

        data = json.loads(state_file.read_text())
        assert data["sess-1"]["message_count"] == 10
        assert data["sess-1"]["chunk"] == 0

    def test_compaction_from_one_message(self):
        """Edge case: transcript shrinks to a single message."""
        track_retention("sess-1", 50)
        progress = track_retention("sess-1", 1)
        assert progress.chunk_index == 1
        assert progress.compacted is True
        assert progress.start_index == 0

    def test_shrink_by_one_triggers_compaction(self):
        """Even shrinking by a single message counts as compaction."""
        track_retention("sess-1", 10)
        progress = track_retention("sess-1", 9)
        assert progress.chunk_index == 1
        assert progress.compacted is True
        assert progress.start_index == 0


class TestPlanCommitRetention:
    def test_plan_does_not_persist_checkpoint(self):
        progress = plan_retention("sess-1", 4)
        assert progress.chunk_index == 0
        assert progress.compacted is False
        assert progress.start_index == 0
        assert read_state("retention_tracking.json", {}) == {}

    def test_commit_advances_next_plan(self):
        commit_retention("sess-1", 4, 0)
        progress = plan_retention("sess-1", 7)
        assert progress.chunk_index == 1
        assert progress.compacted is False
        assert progress.start_index == 4


# ---------------------------------------------------------------------------
# read_state / write_state basics
# ---------------------------------------------------------------------------


class TestReadWriteState:
    def test_read_nonexistent_returns_default(self):
        assert read_state("does_not_exist.json") is None
        assert read_state("does_not_exist.json", {"key": "val"}) == {"key": "val"}

    def test_write_then_read_roundtrips(self):
        write_state("test_roundtrip.json", {"foo": 42})
        assert read_state("test_roundtrip.json") == {"foo": 42}

    def test_write_overwrites_previous(self):
        write_state("test_overwrite.json", {"v": 1})
        write_state("test_overwrite.json", {"v": 2})
        assert read_state("test_overwrite.json") == {"v": 2}
