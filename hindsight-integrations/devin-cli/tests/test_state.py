"""Unit tests for lib/state.py — retention tracking and compaction detection."""

import json
from unittest.mock import patch

import pytest

from lib import state as state_module
from lib.state import (
    commit_retention,
    increment_turn_count,
    plan_retention,
    read_state,
    track_retention,
    write_state,
)


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path):
    """Point all state operations at a temp directory."""
    monkeypatch.setenv("HINDSIGHT_DEVIN_CLI_DATA_DIR", str(tmp_path))


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
        assert commit_retention("sess-1", 4, plan_retention("sess-1", 4)) is True
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


# ---------------------------------------------------------------------------
# Concurrency and durability invariants
# ---------------------------------------------------------------------------


class TestConcurrentRetainHooks:
    """Retain hooks overlap: an async Stop can still be running at the next one.

    They plan from independent snapshots, so commits can arrive out of order.
    """

    def test_a_stale_commit_cannot_roll_the_checkpoint_back(self):
        slow = plan_retention("sess-1", 20)
        fast = plan_retention("sess-1", 30)

        assert commit_retention("sess-1", 30, fast) is True
        assert commit_retention("sess-1", 20, slow) is False

        entry = read_state("retention_tracking.json", {})["sess-1"]
        assert entry["message_count"] == 30, (
            "the slower hook's stale checkpoint overwrote the newer one; "
            "messages 21-30 would be retained a second time on the next run"
        )

    def test_compaction_still_lowers_the_checkpoint(self):
        commit_retention("sess-1", 30, plan_retention("sess-1", 30))

        # A compacted session legitimately reports fewer messages than the
        # checkpoint. That must not be mistaken for a stale concurrent commit.
        compacted = plan_retention("sess-1", 5)
        assert compacted.compacted is True
        assert commit_retention("sess-1", 5, compacted) is True
        assert read_state("retention_tracking.json", {})["sess-1"]["message_count"] == 5

    def test_a_compaction_back_to_an_observed_count_still_blocks_a_stale_commit(self):
        """The count alone is not a version: compaction can hand it back.

        A checkpoint of 20 is not the same checkpoint just because it says 20
        again — the chunk moved on in between, and the messages the stale hook
        is about to mark retained belong to a transcript that no longer exists.
        """
        commit_retention("sess-1", 20, plan_retention("sess-1", 20))

        stale = plan_retention("sess-1", 25)  # observes count=20
        commit_retention("sess-1", 30, plan_retention("sess-1", 30))
        compaction = plan_retention("sess-1", 20)  # 20 < 30, so a new chunk
        assert compaction.compacted is True
        assert commit_retention("sess-1", 20, compaction) is True

        # The stored count matches what `stale` observed, but nothing else does.
        assert commit_retention("sess-1", 25, stale) is False

        entry = read_state("retention_tracking.json", {})["sess-1"]
        assert entry["message_count"] == 20, "a stale hook advanced the checkpoint past a compaction"
        assert entry["chunk"] == compaction.chunk_index, "a stale hook rolled the chunk backwards"


class TestWriteStateSurfacesFailures:
    def test_write_failure_raises_instead_of_reporting_success(self, monkeypatch, tmp_path):
        """A silently dropped checkpoint is worse than a failed hook.

        The caller would believe the turn counter or retention checkpoint was
        persisted and re-send (or skip) messages on the next run.

        Driven through a genuinely unwritable state directory rather than a
        patched `open`, so it keeps testing the contract and not whichever
        primitive write_state happens to stage the file with.
        """
        readonly = tmp_path / "readonly"
        (readonly / "state").mkdir(parents=True)
        (readonly / "state").chmod(0o500)
        monkeypatch.setenv("HINDSIGHT_DEVIN_CLI_DATA_DIR", str(readonly))

        try:
            with pytest.raises(OSError):
                write_state("test_failure.json", {"v": 1})
        finally:
            (readonly / "state").chmod(0o700)

    def test_concurrent_writers_do_not_share_a_staging_file(self):
        """A fixed `<path>.tmp` lets one writer os.replace() another's partial file.

        Every writer must stage somewhere unique, so no leftover staging file
        can be adopted by a different process mid-write.
        """
        seen = []
        real_mkstemp = state_module.tempfile.mkstemp

        def _record(*args, **kwargs):
            fd, path = real_mkstemp(*args, **kwargs)
            seen.append(path)
            return fd, path

        with patch.object(state_module.tempfile, "mkstemp", _record):
            write_state("test_concurrent.json", {"v": 1})
            write_state("test_concurrent.json", {"v": 2})

        assert len(set(seen)) == 2, f"staging paths collided: {seen}"
        assert read_state("test_concurrent.json") == {"v": 2}


class TestMalformedStateFilesDegradeInsteadOfCrashing:
    """State files are on disk between runs and nothing here repairs them.

    read_state already degraded to the default for *invalid* JSON, so a file
    holding valid-but-wrong JSON crashing the hook was a distinction no caller
    could act on — and the crash would repeat on every hook run forever.
    """

    @pytest.mark.parametrize("bad_value", [None, [], "a string", 42])
    def test_a_non_object_state_file_reads_as_the_default(self, bad_value):
        write_state("retention_tracking.json", bad_value)

        assert read_state("retention_tracking.json", {}) == {}

    @pytest.mark.parametrize("bad_value", [None, [], "a string"])
    def test_plan_retention_survives_a_non_object_state_file(self, bad_value):
        write_state("retention_tracking.json", bad_value)

        progress = plan_retention("sess-1", 4)

        assert progress.start_index == 0
        assert progress.chunk_index == 0

    @pytest.mark.parametrize(
        "bad_entry",
        [{}, {"message_count": None}, {"message_count": "3"}, {"message_count": [3]}, [], "x", 7],
    )
    def test_an_unusable_session_entry_reads_as_no_checkpoint(self, bad_entry):
        write_state("retention_tracking.json", {"sess-1": bad_entry})

        progress = plan_retention("sess-1", 4)

        assert progress.start_index == 0
        assert progress.observed_message_count == 0

    def test_a_partial_entry_keeps_the_field_it_does_have(self):
        """Defaulting is per-field: a readable message_count is still a checkpoint."""
        write_state("retention_tracking.json", {"sess-1": {"message_count": 3}})

        progress = plan_retention("sess-1", 4)

        assert progress.observed_message_count == 3
        assert progress.start_index == 3
        assert progress.chunk_index == 1

    def test_a_malformed_entry_is_repaired_by_the_next_successful_retain(self):
        write_state("retention_tracking.json", {"sess-1": {"chunk": 2}})

        progress = plan_retention("sess-1", 4)
        assert commit_retention("sess-1", 4, progress) is True

        assert read_state("retention_tracking.json", {})["sess-1"] == {"message_count": 4, "chunk": 2}

    def test_read_state_leaves_non_dict_defaults_alone(self):
        """The coercion is keyed on the caller asking for a dict, not on the file."""
        write_state("test_list_state.json", [1, 2, 3])

        assert read_state("test_list_state.json") == [1, 2, 3]
        assert read_state("test_list_state.json", []) == [1, 2, 3]

    def test_invalid_utf8_reads_as_the_default(self):
        """A torn write raises UnicodeDecodeError, not JSONDecodeError or OSError."""
        path = state_module._state_file("test_binary.json")
        with open(path, "wb") as fh:
            fh.write(b'{"a": "\xff\xfe not utf-8"}')

        assert read_state("test_binary.json", {}) == {}


class TestSessionCapNeverEvictsTheSessionBeingWritten:
    """The cap runs inside the same locked update that just wrote the entry.

    Evicting it there raised KeyError out of increment_turn_count, so the turn
    counter never returned and the state write never completed.
    """

    def _fill(self, count, prefix="old"):
        return {f"{prefix}-{i:06d}": i for i in range(count)}

    def test_a_new_session_survives_an_over_cap_file(self, monkeypatch):
        monkeypatch.setattr(state_module, "_MAX_TRACKED_SESSIONS", 10)
        # "aaa" sorts before every "old-*" key, so under the previous
        # lexicographic eviction this session was the first one dropped.
        write_state("turns.json", self._fill(20))

        assert increment_turn_count("aaa-brand-new") == 1
        assert read_state("turns.json", {})["aaa-brand-new"] == 1

    def test_the_busiest_session_is_not_evicted_before_idle_ones(self, monkeypatch):
        monkeypatch.setattr(state_module, "_MAX_TRACKED_SESSIONS", 10)
        write_state("turns.json", {})

        # A session that keeps taking turns while 20 one-shot sessions come and
        # go. Its id sorts before every "idle-*" key, so the previous
        # lexicographic eviction dropped it repeatedly and its count kept
        # restarting at 1 even though it was the most active session on the box.
        for i in range(20):
            increment_turn_count(f"idle-{i:03d}")
            increment_turn_count("aaa-active")

        turns = read_state("turns.json", {})
        assert turns.get("aaa-active") == 20, "the most recently used session was evicted"

    def test_the_file_is_still_capped(self, monkeypatch):
        monkeypatch.setattr(state_module, "_MAX_TRACKED_SESSIONS", 10)
        write_state("turns.json", self._fill(20))

        increment_turn_count("new-session")

        assert len(read_state("turns.json", {})) <= 20


class TestMalformedCheckpointsAreRepairable:
    """plan_retention and commit_retention must read a stored value identically.

    plan_retention coerces a malformed message_count to 0; commit_retention
    used to compare the raw stored value, so the compare-and-swap between them
    could never match. The entry was never overwritten and the session
    re-retained from offset 0 on every single run.
    """

    @pytest.mark.parametrize("bad_count", ["4", None, 4.5, [4]])
    def test_a_malformed_count_is_overwritten_rather_than_deadlocked(self, bad_count):
        write_state("retention_tracking.json", {"sess-1": {"message_count": bad_count, "chunk": 0}})

        progress = plan_retention("sess-1", 6)
        committed = commit_retention("sess-1", 6, progress)

        assert committed is True, "CAS could not match, so the bad entry would persist forever"
        assert read_state("retention_tracking.json", {})["sess-1"]["message_count"] == 6

    @pytest.mark.parametrize("bad_count", [-1, -99, True, False])
    def test_a_negative_or_boolean_count_reads_as_no_checkpoint(self, bad_count):
        """A stored -1 would slice the retain from index -1.

        Only the final message would be sent, and the real total then committed
        as the checkpoint — permanently skipping everything before it. Reading
        it as 0 re-retains the session instead, which is recoverable. bool is
        excluded explicitly because it is a subclass of int, so True would
        otherwise pass as the checkpoint 1.
        """
        write_state("retention_tracking.json", {"sess-1": {"message_count": bad_count, "chunk": 0}})

        progress = plan_retention("sess-1", 5)

        assert progress.start_index == 0
        assert progress.observed_message_count == 0

    def test_a_negative_chunk_index_reads_as_zero(self):
        write_state("retention_tracking.json", {"sess-1": {"message_count": 0, "chunk": -3}})

        assert plan_retention("sess-1", 5).chunk_index == 0

    def test_a_genuinely_stale_commit_is_still_rejected(self):
        """The coercion must not weaken the concurrency guard it sits inside."""
        write_state("retention_tracking.json", {})
        stale = plan_retention("sess-1", 10)
        assert commit_retention("sess-1", 30, plan_retention("sess-1", 30)) is True

        assert commit_retention("sess-1", 10, stale) is False
        assert read_state("retention_tracking.json", {})["sess-1"]["message_count"] == 30


class TestLockFailureIsNotSilentlyIgnored:
    """Yielding unlocked is worse than failing.

    `_file_lock` used to swallow an acquisition error and run the body anyway,
    so `locked_read_modify_write` did its read-modify-write with no
    interprocess synchronisation while looking exactly like it held the lock —
    two hooks read the same counter and one overwrote the other's increment,
    which is the corruption the lock exists to prevent.

    "The state directory is unwritable anyway, so the write would fail too" is
    not a defence: a stale `turns.lock` with the wrong owner or mode blocks the
    lock while `turns.json` stays perfectly writable.
    """

    def _unwritable_dir(self, tmp_path):
        d = tmp_path / "readonly"
        d.mkdir()
        d.chmod(0o500)
        return d

    def test_the_body_does_not_run_when_the_lock_cannot_be_taken(self, tmp_path):
        d = self._unwritable_dir(tmp_path)
        entered = False

        with pytest.raises(OSError):
            with state_module._file_lock(str(d / "x.lock")):
                entered = True

        assert not entered, "the guarded body ran without the lock"

    def test_a_read_modify_write_surfaces_the_failure(self, tmp_path, monkeypatch):
        """The lock is unavailable while the state file is perfectly writable.

        That split matters: with both unwritable, the write would raise anyway
        and the test would pass without the lock ever being checked. Only a
        writable state file proves the failure came from the lock.
        """
        d = self._unwritable_dir(tmp_path)
        monkeypatch.setattr(
            state_module,
            "_state_file",
            lambda name: str((d if name.endswith(".lock") else tmp_path) / name),
        )

        with pytest.raises(OSError):
            state_module.locked_read_modify_write("turns.json", "turns.lock", lambda data: (data, None))

    def test_a_lock_that_can_be_taken_still_yields(self, tmp_path):
        """Control: the raise must be about acquisition, not about locking at all."""
        entered = False

        with state_module._file_lock(str(tmp_path / "ok.lock")):
            entered = True

        assert entered
