"""Unit tests for lib/state.py — retention tracking and compact segmentation."""

import json

import pytest

from lib.state import mark_precompact, read_state, track_retention, write_state


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path):
    """Point all state operations at a temp directory."""
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path))


# ---------------------------------------------------------------------------
# track_retention / mark_precompact — retention document segmentation
# ---------------------------------------------------------------------------


class TestTrackRetention:
    def test_first_call_returns_chunk_zero(self):
        chunk, start_index = track_retention("sess-1", 10)
        assert chunk == 0
        assert start_index == 0

    def test_growing_transcript_keeps_same_chunk(self):
        track_retention("sess-1", 4)
        chunk, start_index = track_retention("sess-1", 8)
        assert chunk == 0
        assert start_index == 0

    def test_equal_count_keeps_same_chunk(self):
        track_retention("sess-1", 5)
        chunk, start_index = track_retention("sess-1", 5)
        assert chunk == 0
        assert start_index == 0

    def test_sessions_are_independent(self):
        track_retention("sess-a", 10)
        track_retention("sess-b", 20)

        mark_precompact("sess-a", 10)
        chunk_a, start_a = track_retention("sess-a", 12)
        chunk_b, start_b = track_retention("sess-b", 25)

        assert chunk_a == 1
        assert start_a == 10
        assert chunk_b == 0
        assert start_b == 0

    def test_persists_across_calls(self, tmp_path):
        """State file is written to disk and survives between calls."""
        track_retention("sess-1", 10)

        # Verify the state file exists
        state_file = tmp_path / "state" / "retention_tracking.json"
        assert state_file.exists()

        data = json.loads(state_file.read_text())
        assert data["sess-1"]["message_count"] == 10
        assert data["sess-1"]["chunk"] == 0

    def test_precompact_marks_next_segment_start(self):
        chunk, start_index = mark_precompact("sess-1", 10)
        assert chunk == 1
        assert start_index == 10

        chunk, start_index = track_retention("sess-1", 14)
        assert chunk == 1
        assert start_index == 10

    def test_multiple_precompacts_increment_chunks(self):
        mark_precompact("sess-1", 10)
        track_retention("sess-1", 14)

        chunk, start_index = mark_precompact("sess-1", 14)
        assert chunk == 2
        assert start_index == 14

        chunk, start_index = track_retention("sess-1", 17)
        assert chunk == 2
        assert start_index == 14


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
