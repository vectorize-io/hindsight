"""Reader for Zed's agent thread database (``threads.db``).

Zed stores every AI-assistant conversation in a SQLite database. This module
opens that database read-only, decompresses each thread, and extracts a plain
``[role] text`` transcript plus the project paths the thread belongs to — so a
background process can retain finished conversations into Hindsight.

The on-disk format has gone through several revisions and Zed only rewrites a
row to the current version when that thread is next saved, so a live database
can hold a mix of all of them at once. We parse, in order of the top-level
``version`` field:

  - ``"0.3.0"`` (current): messages are externally-tagged ``{"User": {...}}`` /
    ``{"Agent": {...}}`` objects; the role is implied by the variant key, and
    text lives in a ``content`` array of ``{"Text": "..."}`` blocks. A unit
    ``Resume`` message serializes as the bare string ``"Resume"`` rather than an
    object, so message elements may be either strings or dicts.
  - ``"0.2.0"`` / ``"0.1.0"`` (legacy): messages have an explicit lowercase
    ``role`` and a ``segments`` array of ``{"type": "text", "text": "..."}``.
  - no ``version`` field (oldest legacy): messages have an explicit ``role`` and
    a flat ``text`` string.

Source of truth: zed-industries/zed ``crates/agent/src/db.rs``,
``legacy_thread.rs``, and ``thread.rs`` (verified against ``main``).
"""

import json
import os
import sqlite3
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import zstandard  # type: ignore
except ImportError:  # pragma: no cover - resolved at import time, exercised in tests via fallback
    zstandard = None


# ── Thread / message models ──────────────────────────────────────────────────


@dataclass
class ThreadMessage:
    """A single turn extracted from a Zed thread."""

    role: str  # "user" | "assistant" | "system"
    text: str


@dataclass
class ZedThread:
    """A parsed Zed conversation thread."""

    id: str
    title: str
    updated_at: str
    messages: list[ThreadMessage] = field(default_factory=list)
    # Absolute project paths this thread was opened against (from the
    # ``folder_paths`` column). Used to map a thread to its per-project bank.
    folder_paths: list[str] = field(default_factory=list)


# ── Database location ─────────────────────────────────────────────────────────


def default_threads_db_path() -> Path:
    """Return the platform-default path to Zed's ``threads.db``.

    Mirrors Zed's ``paths::data_dir().join("threads").join("threads.db")``.
    """
    override = os.environ.get("ZED_THREADS_DB")
    if override:
        return Path(override)

    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Zed" / "threads" / "threads.db"
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else home / "AppData" / "Local"
        return root / "Zed" / "threads" / "threads.db"
    # Linux / *nix — XDG_DATA_HOME or ~/.local/share, lowercase "zed".
    xdg = os.environ.get("XDG_DATA_HOME")
    root = Path(xdg) if xdg else home / ".local" / "share"
    return root / "zed" / "threads" / "threads.db"


# ── Blob decoding ─────────────────────────────────────────────────────────────


def _decompress(data: bytes, data_type: str) -> str:
    """Decode a thread ``data`` blob into its inner JSON string.

    ``data_type`` is ``"zstd"`` (current writes) or ``"json"`` (uncompressed).
    """
    if data_type == "json":
        return data.decode("utf-8")
    if data_type == "zstd":
        if zstandard is not None:
            # Stream-decode rather than ZstdDecompressor.decompress(): Zed writes
            # frames via zstd::encode_all, whose header does NOT declare the
            # decompressed content size, and the one-shot decompress() refuses
            # such frames ("could not determine content size in frame header").
            # The streaming reader has no such requirement.
            import io

            with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)) as reader:
                return reader.read().decode("utf-8")
        # Fallback so the package has no hard runtime dep: a zstd frame can be
        # decoded by zlib only if it is not actually zstd. We never expect this
        # path in practice, but raise a clear error rather than silently fail.
        raise RuntimeError("thread is zstd-compressed but the 'zstandard' package is not installed")
    raise ValueError(f"unknown thread data_type: {data_type!r}")


# ── Message extraction (per version) ──────────────────────────────────────────


def _text_from_user_content(content: Any) -> str:
    """Join the text of a current-format ``User`` message ``content`` array."""
    parts: list[str] = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        if "Text" in block and isinstance(block["Text"], str):
            parts.append(block["Text"])
        elif "Mention" in block and isinstance(block["Mention"], dict):
            mention_text = block["Mention"].get("content")
            if isinstance(mention_text, str):
                parts.append(mention_text)
        # "Image" blocks carry no text — skipped.
    return "\n".join(p for p in parts if p)


def _text_from_agent_content(content: Any) -> str:
    """Join the text of a current-format ``Agent`` message ``content`` array.

    Only plain ``Text`` blocks are kept — ``Thinking``, ``RedactedThinking`` and
    ``ToolUse`` are model-internal and excluded from the retained transcript.
    """
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, dict) and "Text" in block and isinstance(block["Text"], str):
            parts.append(block["Text"])
    return "\n".join(p for p in parts if p)


def _messages_from_current(messages: Any) -> list[ThreadMessage]:
    """Parse the current (``0.3.0``) externally-tagged message array."""
    out: list[ThreadMessage] = []
    for msg in messages or []:
        # Unit variants (e.g. ``Resume``) serialize as a bare string.
        if isinstance(msg, str):
            continue
        if not isinstance(msg, dict):
            continue
        if "User" in msg and isinstance(msg["User"], dict):
            text = _text_from_user_content(msg["User"].get("content"))
            if text.strip():
                out.append(ThreadMessage(role="user", text=text))
        elif "Agent" in msg and isinstance(msg["Agent"], dict):
            text = _text_from_agent_content(msg["Agent"].get("content"))
            if text.strip():
                out.append(ThreadMessage(role="assistant", text=text))
        # "Compaction" / "Resume" carry no user-facing transcript text.
    return out


def _text_from_segments(segments: Any) -> str:
    """Join the text of legacy ``segments`` (``0.1.0`` / ``0.2.0``)."""
    parts: list[str] = []
    for seg in segments or []:
        if isinstance(seg, dict) and seg.get("type") == "text":
            text = seg.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(p for p in parts if p)


def _messages_from_legacy(messages: Any) -> list[ThreadMessage]:
    """Parse legacy messages that carry an explicit ``role``.

    Handles both the ``segments`` array (``0.1.0`` / ``0.2.0``) and the oldest
    flat ``text`` field (no ``version``). ``system`` messages are dropped to
    match Zed's own upgrade behaviour.
    """
    out: list[ThreadMessage] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue  # drop "system" and anything unexpected
        if "segments" in msg:
            text = _text_from_segments(msg.get("segments"))
        else:
            raw = msg.get("text")
            text = raw if isinstance(raw, str) else ""
        if text.strip():
            out.append(ThreadMessage(role=role, text=text))
    return out


def parse_thread_json(raw: str) -> Optional[list[ThreadMessage]]:
    """Parse a decompressed thread JSON string into transcript messages.

    Returns ``None`` if the version is unrecognized (so the caller can skip it).
    """
    try:
        doc = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None

    version = doc.get("version")
    messages = doc.get("messages")
    if version == "0.3.0":
        return _messages_from_current(messages)
    if version in ("0.2.0", "0.1.0") or version is None:
        return _messages_from_legacy(messages)
    # Unknown future version — skip rather than guess.
    return None


def _thread_title(doc: dict, column_summary: str) -> str:
    """Resolve a thread title, preferring the JSON over the column."""
    for key in ("title", "summary"):
        val = doc.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return column_summary or "Untitled"


def _folder_paths(raw: Optional[str]) -> list[str]:
    """Parse the ``folder_paths`` column (a JSON-serialized list of paths)."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []
    if isinstance(parsed, list):
        return [p for p in parsed if isinstance(p, str)]
    return []


# ── Public API ────────────────────────────────────────────────────────────────


def read_threads(db_path: Path, since: Optional[str] = None) -> list[ZedThread]:
    """Read and parse all threads from ``db_path``.

    Opens the database read-only so it is safe to run alongside a live Zed.
    When ``since`` (an RFC3339 string) is given, only threads with a strictly
    greater ``updated_at`` are returned — the cheap way to poll for new activity.
    """
    if not db_path.exists():
        return []

    uri = f"file:{db_path}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(threads)")}
        has_folders = "folder_paths" in cols
        select = (
            "SELECT id, summary, updated_at, data_type, data"
            + (", folder_paths" if has_folders else "")
            + " FROM threads"
        )
        params: tuple = ()
        if since is not None:
            select += " WHERE updated_at > ?"
            params = (since,)
        rows = conn.execute(select, params).fetchall()
    finally:
        conn.close()

    threads: list[ZedThread] = []
    for row in rows:
        thread_id, summary, updated_at, data_type, data = row[0], row[1], row[2], row[3], row[4]
        folder_raw = row[5] if has_folders and len(row) > 5 else None
        try:
            raw = _decompress(data, data_type)
            doc = json.loads(raw)
        except (RuntimeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        messages = parse_thread_json(raw)
        if messages is None:
            continue
        threads.append(
            ZedThread(
                id=str(thread_id),
                title=_thread_title(doc, summary),
                updated_at=str(updated_at),
                messages=messages,
                folder_paths=_folder_paths(folder_raw),
            )
        )
    return threads
