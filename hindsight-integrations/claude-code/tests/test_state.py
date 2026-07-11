"""Unit tests for lib/state.py — retention tracking and compaction detection."""

import json

import pytest

from lib.state import commit_delta_retention, plan_delta_retention, read_state, track_retention, write_state


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path):
    """Point all state operations at a temp directory."""
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path))


# ---------------------------------------------------------------------------
# track_retention — core compaction detection
# ---------------------------------------------------------------------------


class TestTrackRetention:
    def test_first_call_returns_chunk_zero(self):
        chunk, compacted = track_retention("sess-1", 10)
        assert chunk == 0
        assert compacted is False

    def test_growing_transcript_keeps_same_chunk(self):
        track_retention("sess-1", 4)
        chunk, compacted = track_retention("sess-1", 8)
        assert chunk == 0
        assert compacted is False

    def test_equal_count_keeps_same_chunk(self):
        track_retention("sess-1", 5)
        chunk, compacted = track_retention("sess-1", 5)
        assert chunk == 0
        assert compacted is False

    def test_shrinking_transcript_triggers_compaction(self):
        track_retention("sess-1", 10)
        chunk, compacted = track_retention("sess-1", 3)
        assert chunk == 1
        assert compacted is True

    def test_multiple_compactions_increment_chunk(self):
        track_retention("sess-1", 10)

        chunk, compacted = track_retention("sess-1", 3)
        assert chunk == 1
        assert compacted is True

        # Grow again after compaction
        track_retention("sess-1", 8)

        # Second compaction
        chunk, compacted = track_retention("sess-1", 2)
        assert chunk == 2
        assert compacted is True

    def test_growth_after_compaction_stays_on_same_chunk(self):
        track_retention("sess-1", 10)
        track_retention("sess-1", 3)  # compaction → chunk 1

        chunk, compacted = track_retention("sess-1", 6)
        assert chunk == 1
        assert compacted is False

    def test_sessions_are_independent(self):
        track_retention("sess-a", 10)
        track_retention("sess-b", 20)

        # Compaction on sess-a only
        chunk_a, compacted_a = track_retention("sess-a", 3)
        chunk_b, compacted_b = track_retention("sess-b", 25)

        assert chunk_a == 1
        assert compacted_a is True
        assert chunk_b == 0
        assert compacted_b is False

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
        chunk, compacted = track_retention("sess-1", 1)
        assert chunk == 1
        assert compacted is True

    def test_shrink_by_one_triggers_compaction(self):
        """Even shrinking by a single message counts as compaction."""
        track_retention("sess-1", 10)
        chunk, compacted = track_retention("sess-1", 9)
        assert chunk == 1
        assert compacted is True


class TestDeltaRetention:
    def test_first_plan_starts_at_zero_with_plain_document(self):
        start, document_index, compacted = plan_delta_retention("sess-1", 4)
        assert start == 0
        assert document_index == 0
        assert compacted is False

    def test_successful_commit_advances_start_and_document_index(self):
        commit_delta_retention("sess-1", 4, 0)
        start, document_index, compacted = plan_delta_retention("sess-1", 7)
        assert start == 4
        assert document_index == 1
        assert compacted is False

    def test_plan_does_not_commit_on_its_own(self):
        plan_delta_retention("sess-1", 4)
        start, document_index, compacted = plan_delta_retention("sess-1", 4)
        assert start == 0
        assert document_index == 0
        assert compacted is False

    def test_compaction_resets_start_but_keeps_next_document(self):
        commit_delta_retention("sess-1", 10, 2)
        start, document_index, compacted = plan_delta_retention("sess-1", 3)
        assert start == 0
        assert document_index == 3
        assert compacted is True

    def test_old_compaction_state_shape_does_not_reuse_plain_document(self):
        write_state("retention_tracking.json", {"sess-1": {"message_count": 5, "chunk": 0}})
        start, document_index, compacted = plan_delta_retention("sess-1", 8)
        assert start == 5
        assert document_index == 1
        assert compacted is False


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
