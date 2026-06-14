"""Tests for the Zed threads.db reader.

Zed isn't installed in CI, so these synthesize real ``threads.db`` fixtures —
a SQLite database with zstd-compressed JSON blobs in every on-disk format the
reader must handle — and assert the extracted transcript.
"""

import json
import sqlite3
from pathlib import Path

import pytest
import zstandard

from hindsight_zed.threads_db import (
    ZedThread,
    default_threads_db_path,
    parse_thread_json,
    read_threads,
)


# ── Fixture helpers ───────────────────────────────────────────────────────────


def _make_db(tmp_path: Path) -> Path:
    db = tmp_path / "threads.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE threads (
            id TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            data_type TEXT NOT NULL,
            data BLOB NOT NULL,
            parent_id TEXT,
            folder_paths TEXT,
            folder_paths_order TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()
    return db


def _insert(db: Path, *, id, summary, updated_at, doc, compress=True, folder_paths=None):
    raw = json.dumps(doc).encode("utf-8")
    if compress:
        # Mimic Zed exactly: it writes streaming zstd frames (zstd::encode_all)
        # whose header does NOT declare the decompressed content size. Use the
        # streaming compressobj rather than .compress() (which would write the
        # size and mask the real-world frame the reader must handle).
        obj = zstandard.ZstdCompressor(level=3).compressobj()
        data = obj.compress(raw) + obj.flush()
        data_type = "zstd"
    else:
        data, data_type = raw, "json"
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO threads (id, summary, updated_at, data_type, data, folder_paths) VALUES (?, ?, ?, ?, ?, ?)",
        (id, summary, updated_at, data_type, data, json.dumps(folder_paths) if folder_paths else None),
    )
    conn.commit()
    conn.close()


# ── Sample documents per format ───────────────────────────────────────────────

CURRENT_DOC = {
    "version": "0.3.0",
    "title": "Fix the parser bug",
    "updated_at": "2026-06-10T14:22:05Z",
    "messages": [
        {"User": {"id": "u1", "content": [{"Text": "Why does the reader crash on empty input?"}]}},
        {
            "Agent": {
                "content": [
                    {"Thinking": {"text": "internal", "signature": None}},
                    {"Text": "It crashes on a zero-length blob. Guard for empty data."},
                    {"ToolUse": {"id": "t1", "name": "read_file", "input": {}}},
                ],
                "tool_results": {},
            }
        },
        "Resume",  # unit variant → bare string, must not break parsing
        {"User": {"id": "u2", "content": [{"Mention": {"uri": "x", "content": "see @reader.rs"}}]}},
    ],
}

LEGACY_020_DOC = {
    "version": "0.2.0",
    "summary": "Old thread",
    "updated_at": "2026-01-02T00:00:00Z",
    "messages": [
        {"id": 0, "role": "user", "segments": [{"type": "text", "text": "hello from legacy"}]},
        {"id": 1, "role": "system", "segments": [{"type": "text", "text": "system prompt — dropped"}]},
        {"id": 2, "role": "assistant", "segments": [{"type": "text", "text": "legacy reply"}]},
    ],
}

OLDEST_DOC = {
    # no "version" key — messages carry a flat "text" field
    "summary": "Oldest thread",
    "updated_at": "2025-12-01T00:00:00Z",
    "messages": [
        {"id": 0, "role": "user", "text": "oldest user line"},
        {"id": 1, "role": "assistant", "text": "oldest assistant line"},
    ],
}


# ── Per-format parsing ────────────────────────────────────────────────────────


def test_current_format_roles_and_text():
    msgs = parse_thread_json(json.dumps(CURRENT_DOC))
    assert [m.role for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0].text == "Why does the reader crash on empty input?"
    # Thinking/ToolUse excluded; only the plain Text block kept.
    assert msgs[1].text == "It crashes on a zero-length blob. Guard for empty data."
    # Mention contributes its resolved content text.
    assert msgs[2].text == "see @reader.rs"


def test_resume_bare_string_does_not_break():
    # The "Resume" string element must be skipped, not crash.
    msgs = parse_thread_json(json.dumps(CURRENT_DOC))
    assert all(isinstance(m.text, str) for m in msgs)


def test_legacy_020_segments_and_system_dropped():
    msgs = parse_thread_json(json.dumps(LEGACY_020_DOC))
    assert [m.role for m in msgs] == ["user", "assistant"]  # system dropped
    assert msgs[0].text == "hello from legacy"
    assert msgs[1].text == "legacy reply"


def test_oldest_flat_text_format():
    msgs = parse_thread_json(json.dumps(OLDEST_DOC))
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[0].text == "oldest user line"


def test_unknown_version_skipped():
    assert parse_thread_json(json.dumps({"version": "9.9.9", "messages": []})) is None


def test_malformed_json_returns_none():
    assert parse_thread_json("{not json") is None


# ── End-to-end DB reading ─────────────────────────────────────────────────────


def test_read_threads_mixed_versions(tmp_path):
    db = _make_db(tmp_path)
    _insert(
        db, id="a", summary="cur", updated_at="2026-06-10T10:00:00Z", doc=CURRENT_DOC, folder_paths=["/Users/me/proj-a"]
    )
    _insert(db, id="b", summary="leg", updated_at="2026-06-10T09:00:00Z", doc=LEGACY_020_DOC)
    _insert(db, id="c", summary="old", updated_at="2026-06-10T08:00:00Z", doc=OLDEST_DOC)

    threads = read_threads(db)
    by_id = {t.id: t for t in threads}
    assert set(by_id) == {"a", "b", "c"}
    assert by_id["a"].title == "Fix the parser bug"
    assert by_id["a"].folder_paths == ["/Users/me/proj-a"]
    assert len(by_id["a"].messages) == 3
    assert len(by_id["b"].messages) == 2


def test_read_threads_uncompressed_json_row(tmp_path):
    db = _make_db(tmp_path)
    _insert(db, id="j", summary="json", updated_at="2026-06-10T10:00:00Z", doc=CURRENT_DOC, compress=False)
    threads = read_threads(db)
    assert len(threads) == 1
    assert threads[0].messages[0].text.startswith("Why does the reader crash")


def test_read_threads_since_filter(tmp_path):
    db = _make_db(tmp_path)
    _insert(db, id="old", summary="o", updated_at="2026-06-10T08:00:00Z", doc=CURRENT_DOC)
    _insert(db, id="new", summary="n", updated_at="2026-06-10T12:00:00Z", doc=CURRENT_DOC)
    recent = read_threads(db, since="2026-06-10T10:00:00Z")
    assert [t.id for t in recent] == ["new"]


def test_read_threads_missing_db_returns_empty(tmp_path):
    assert read_threads(tmp_path / "nope.db") == []


def test_read_threads_unknown_version_skipped_at_db_level(tmp_path):
    db = _make_db(tmp_path)
    _insert(db, id="future", summary="f", updated_at="2026-06-10T10:00:00Z", doc={"version": "9.9.9", "messages": []})
    _insert(db, id="ok", summary="g", updated_at="2026-06-10T11:00:00Z", doc=CURRENT_DOC)
    threads = read_threads(db)
    assert [t.id for t in threads] == ["ok"]


def test_opens_readonly_does_not_lock(tmp_path):
    # Reading must not block a concurrent writer — open read-only and verify we
    # can still write to the db afterward.
    db = _make_db(tmp_path)
    _insert(db, id="a", summary="x", updated_at="2026-06-10T10:00:00Z", doc=CURRENT_DOC)
    read_threads(db)
    # Writing still works (no lingering lock from the reader).
    _insert(db, id="b", summary="y", updated_at="2026-06-10T11:00:00Z", doc=CURRENT_DOC)
    assert len(read_threads(db)) == 2


# ── Path resolution ───────────────────────────────────────────────────────────


def test_default_path_env_override(monkeypatch):
    monkeypatch.setenv("ZED_THREADS_DB", "/custom/threads.db")
    assert default_threads_db_path() == Path("/custom/threads.db")


def test_default_path_macos(monkeypatch):
    monkeypatch.delenv("ZED_THREADS_DB", raising=False)
    monkeypatch.setattr("sys.platform", "darwin")
    p = default_threads_db_path()
    assert p.parts[-3:] == ("Zed", "threads", "threads.db")


def test_default_path_linux(monkeypatch):
    monkeypatch.delenv("ZED_THREADS_DB", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    p = default_threads_db_path()
    assert p.parts[-3:] == ("zed", "threads", "threads.db")
