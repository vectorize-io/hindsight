"""Daemon flow tests.

Exercise the recall→inject and capture→retain pipeline against a synthetic
threads.db and a fake Hindsight client (no network, no real Zed).
"""

import json
import sqlite3

import zstandard

from hindsight_zed.config import ZedConfig
from hindsight_zed.daemon import poll_once, process_thread
from hindsight_zed.rules_file import BEGIN_MARKER
from hindsight_zed.state import DaemonState
from hindsight_zed.threads_db import ThreadMessage, ZedThread


class FakeClient:
    """Records recall/retain calls; returns canned recall results."""

    def __init__(self, recall_results=None):
        self.recall_results = recall_results or []
        self.recall_calls = []
        self.retain_calls = []

    def recall(self, bank_id, query, **kw):
        self.recall_calls.append((bank_id, query))
        return {"results": self.recall_results}

    def retain(self, bank_id, content, **kw):
        self.retain_calls.append((bank_id, content, kw))
        return {"ok": True}


def _thread(project, *, id="t1", updated="2026-06-10T10:00:00Z"):
    return ZedThread(
        id=id,
        title="Session",
        updated_at=updated,
        messages=[ThreadMessage("user", "How do I fix the parser?"), ThreadMessage("assistant", "Guard for empty.")],
        folder_paths=[str(project)],
    )


def test_process_thread_recalls_and_writes_block(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    client = FakeClient(recall_results=[{"text": "The user prefers pytest", "type": "world"}])
    cfg = ZedConfig(bank_prefix="zed")
    state = DaemonState(path=tmp_path / "s.json")

    process_thread(_thread(project), client, cfg, state)

    # Recalled against the latest user message, scoped to the project's bank.
    assert client.recall_calls
    bank, query = client.recall_calls[0]
    assert bank == "zed-proj"
    assert "parser" in query
    # Memory block written into the project's instruction file (.rules here).
    rules = (project / ".rules").read_text()
    assert BEGIN_MARKER in rules
    assert "The user prefers pytest" in rules
    # Retained the transcript.
    assert client.retain_calls
    assert client.retain_calls[0][0] == "zed-proj"
    assert "[user]" in client.retain_calls[0][1]


def test_process_thread_retain_dedup(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    client = FakeClient()
    cfg = ZedConfig()
    state = DaemonState(path=tmp_path / "s.json")

    process_thread(_thread(project), client, cfg, state)
    process_thread(_thread(project), client, cfg, state)  # same updated_at → skip retain
    assert len(client.retain_calls) == 1
    # But an advanced thread retains again.
    process_thread(_thread(project, updated="2026-06-10T11:00:00Z"), client, cfg, state)
    assert len(client.retain_calls) == 2


def test_process_thread_skips_when_no_project(tmp_path):
    client = FakeClient()
    cfg = ZedConfig()
    state = DaemonState(path=tmp_path / "s.json")
    thread = ZedThread(
        id="t", title="x", updated_at="2026-06-10T10:00:00Z", messages=[ThreadMessage("user", "hi")], folder_paths=[]
    )
    process_thread(thread, client, cfg, state)
    assert not client.recall_calls and not client.retain_calls


def test_auto_recall_off(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    client = FakeClient()
    cfg = ZedConfig(auto_recall=False)
    process_thread(_thread(project), client, cfg, DaemonState(path=tmp_path / "s.json"))
    assert not client.recall_calls
    assert not (project / ".rules").exists()
    assert client.retain_calls  # retain still happens


def test_recall_failure_does_not_block_retain(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()

    class Boom(FakeClient):
        def recall(self, *a, **k):
            raise RuntimeError("server down")

    client = Boom()
    process_thread(_thread(project), client, ZedConfig(), DaemonState(path=tmp_path / "s.json"))
    assert client.retain_calls  # retain still ran despite recall error


# ── poll_once against a real synthetic threads.db ─────────────────────────────


def _make_threads_db(tmp_path, project):
    db = tmp_path / "threads.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE threads (id TEXT PRIMARY KEY, summary TEXT, updated_at TEXT, "
        "data_type TEXT, data BLOB, folder_paths TEXT)"
    )
    doc = {
        "version": "0.3.0",
        "title": "Session",
        "messages": [
            {"User": {"id": "u", "content": [{"Text": "fix the parser"}]}},
            {"Agent": {"content": [{"Text": "guard empties"}], "tool_results": {}}},
        ],
    }
    obj = zstandard.ZstdCompressor(level=3).compressobj()
    data = obj.compress(json.dumps(doc).encode()) + obj.flush()
    conn.execute(
        "INSERT INTO threads VALUES (?,?,?,?,?,?)",
        ("t1", "Session", "2026-06-10T10:00:00Z", "zstd", data, json.dumps([str(project)])),
    )
    conn.commit()
    conn.close()
    return db


def test_poll_once_end_to_end(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    db = _make_threads_db(tmp_path, project)
    client = FakeClient(recall_results=[{"text": "prefers pytest", "type": "world"}])
    cfg = ZedConfig(bank_prefix="zed")
    state = DaemonState(path=tmp_path / "s.json")

    high = poll_once(db, client, cfg, state, since=None)

    assert high == "2026-06-10T10:00:00Z"
    assert client.recall_calls and client.retain_calls
    assert "prefers pytest" in (project / ".rules").read_text()
    # Polling again with the high-water mark yields no reprocessing.
    client.recall_calls.clear()
    poll_once(db, client, cfg, state, since=high)
    assert not client.recall_calls


# ── Auth/connection error escalation ──────────────────────────────────────────

import logging  # noqa: E402

from hindsight_zed.client import HindsightHTTPError  # noqa: E402


class AuthFailClient(FakeClient):
    """A bad token fails *every* call (401/403), like the real server."""

    def __init__(self, status=401):
        super().__init__()
        self.status = status

    def recall(self, *a, **k):
        raise HindsightHTTPError(self.status, "http://x/recall", "denied")

    def retain(self, *a, **k):
        raise HindsightHTTPError(self.status, "http://x/retain", "denied")


def test_auth_error_escalated_to_warning(tmp_path, caplog):
    project = tmp_path / "proj"
    project.mkdir()
    state = DaemonState(path=tmp_path / "s.json")
    with caplog.at_level(logging.WARNING, logger="hindsight_zed.daemon"):
        process_thread(_thread(project), AuthFailClient(401), ZedConfig(), state)
    assert any("HINDSIGHT_API_TOKEN" in r.getMessage() for r in caplog.records)
    assert ("auth", "zed-proj") in state.warned


def test_auth_warning_deduped_across_polls(tmp_path, caplog):
    project = tmp_path / "proj"
    project.mkdir()
    client = AuthFailClient(403)
    state = DaemonState(path=tmp_path / "s.json")
    with caplog.at_level(logging.WARNING, logger="hindsight_zed.daemon"):
        process_thread(_thread(project), client, ZedConfig(), state)
        process_thread(_thread(project, id="t2"), client, ZedConfig(), state)
    # One WARNING total despite recall+retain failing on two polls (4 failures).
    warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warns) == 1


def test_connection_error_escalated(tmp_path, caplog):
    import urllib.error

    project = tmp_path / "proj"
    project.mkdir()

    class Down(FakeClient):
        def recall(self, *a, **k):
            raise urllib.error.URLError("connection refused")

        def retain(self, *a, **k):
            raise urllib.error.URLError("connection refused")

    state = DaemonState(path=tmp_path / "s.json")
    with caplog.at_level(logging.WARNING, logger="hindsight_zed.daemon"):
        process_thread(_thread(project), Down(), ZedConfig(), state)
    assert any("could not reach" in r.getMessage() for r in caplog.records)
    assert ("conn", "zed-proj") in state.warned


def test_transient_error_stays_debug(tmp_path, caplog):
    project = tmp_path / "proj"
    project.mkdir()

    class Boom(FakeClient):
        def recall(self, *a, **k):
            raise RuntimeError("weird transient")

        def retain(self, *a, **k):
            raise RuntimeError("weird transient")

    state = DaemonState(path=tmp_path / "s.json")
    with caplog.at_level(logging.WARNING, logger="hindsight_zed.daemon"):
        process_thread(_thread(project), Boom(), ZedConfig(), state)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
    assert not state.warned


def test_warning_cleared_on_recovery(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    state = DaemonState(path=tmp_path / "s.json")

    class Flappy(FakeClient):
        def __init__(self):
            super().__init__(recall_results=[{"text": "ok", "type": "world"}])
            self.fail = True

        def recall(self, *a, **k):
            if self.fail:
                raise HindsightHTTPError(401, "http://x", "no")
            return super().recall(*a, **k)

        def retain(self, *a, **k):
            if self.fail:
                raise HindsightHTTPError(401, "http://x", "no")
            return super().retain(*a, **k)

    client = Flappy()
    process_thread(_thread(project), client, ZedConfig(), state)
    assert ("auth", "zed-proj") in state.warned
    # User fixes the token → both calls succeed → warning state cleared.
    client.fail = False
    process_thread(_thread(project, id="t2"), client, ZedConfig(), state)
    assert ("auth", "zed-proj") not in state.warned
