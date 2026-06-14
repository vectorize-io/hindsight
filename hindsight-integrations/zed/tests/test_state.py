"""Tests for daemon state persistence."""

from hindsight_zed.state import DaemonState


def test_needs_retain_new_thread(tmp_path):
    st = DaemonState(path=tmp_path / "s.json")
    assert st.needs_retain("t1", "2026-06-10T00:00:00Z") is True


def test_mark_and_skip_unchanged(tmp_path):
    st = DaemonState(path=tmp_path / "s.json")
    st.mark_retained("t1", "2026-06-10T00:00:00Z")
    assert st.needs_retain("t1", "2026-06-10T00:00:00Z") is False
    # advanced timestamp → retain again
    assert st.needs_retain("t1", "2026-06-10T01:00:00Z") is True


def test_persistence_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    st = DaemonState(path=p)
    st.mark_retained("t1", "2026-06-10T00:00:00Z")
    st.save()
    reloaded = DaemonState.load(p)
    assert reloaded.needs_retain("t1", "2026-06-10T00:00:00Z") is False


def test_load_missing_file(tmp_path):
    st = DaemonState.load(tmp_path / "nope.json")
    assert st.retained == {}
