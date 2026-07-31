"""Reads Devin CLI's local session database to reconstruct a conversation transcript.

Devin CLI hooks are ephemeral processes, like Claude Code's, but their stdin is
much sparser: no `transcript_path`, no `cwd`. What Devin CLI hooks *do* get
(as of CLI 3000.3.22) is a stable `session_id` on every event.

Devin CLI persists every session's conversation to a local SQLite database at
`~/.local/share/devin/cli/sessions.db` (schema: `sessions`, `message_nodes`).
This is undocumented — there is no public API or stability guarantee for it —
but it is written live (WAL mode) as the conversation progresses, and the
`session_id` in hook stdin matches `sessions.id` / `message_nodes.session_id`
exactly. This module reads it defensively: any schema mismatch or missing file
degrades to an empty transcript rather than raising, so a future CLI release
that changes the storage layer disables retain/multi-turn-recall instead of
crashing the hook.

`message_nodes` stores one row per message *as sent to the model in a given API
call*, so the same system/rules messages (and even the same user/assistant
turns) get re-inserted verbatim on every subsequent turn as the context window
is replayed. Rows carry a stable `message_id` in their JSON payload, so we
de-duplicate by first occurrence (in `node_id` order) to reconstruct the same
linear, one-entry-per-turn transcript Claude Code's JSONL transcript file
represents.
"""

import json
import os
import sqlite3
import urllib.parse

# Matches the CLI's own data directory resolution: $XDG_DATA_HOME (or the
# platform default) / devin / cli / sessions.db. Only macOS/Linux paths are
# handled here — Windows uses %APPDATA%\devin\cli\sessions.db, out of scope
# until this integration ships Windows support.
_DEFAULT_DB_RELATIVE = os.path.join(".local", "share", "devin", "cli", "sessions.db")


def sessions_db_path() -> str:
    """Resolve the path to Devin CLI's session database."""
    override = os.environ.get("HINDSIGHT_DEVIN_SESSIONS_DB")
    if override:
        return override
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return os.path.join(xdg_data_home, "devin", "cli", "sessions.db")
    return os.path.join(os.path.expanduser("~"), _DEFAULT_DB_RELATIVE)


def read_session_messages(session_id: str, db_path: str = None) -> list:
    """Return a session's messages as a de-duplicated, chronological list.

    Each entry is a flat dict: {"role": ..., "content": ...} and, when present,
    "tool_calls" (the raw OpenAI-style tool call list Devin CLI stored for that
    turn). This is the same flat shape `lib/content.py`'s transcript formatters
    already accept (their "testing / future compatibility" branch), so recall.py
    and retain.py can feed this straight into the shared Claude Code formatting
    logic without modification.

    Returns an empty list if the database is missing, the session has no rows,
    or the schema doesn't match what this module expects — never raises.
    """
    if not session_id:
        return []

    path = db_path or sessions_db_path()
    if not os.path.isfile(path):
        return []

    messages = []
    seen_message_ids = set()
    try:
        # Read-only URI connection: safe to open alongside the CLI's own
        # writer, including while it holds the database in WAL mode.
        #
        # The path is percent-encoded because SQLite parses a URI filename: a
        # bare `?` in it starts the parameter list and a bare `%` begins an
        # escape, so a HINDSIGHT_DEVIN_SESSIONS_DB containing either would open
        # some other filename — or none — and degrade to an empty transcript
        # while the real database sat there readable. `/` is kept literal so
        # the path still parses as a path.
        conn = sqlite3.connect(f"file:{urllib.parse.quote(path)}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return []

    try:
        cur = conn.execute(
            "SELECT chat_message FROM message_nodes WHERE session_id = ? ORDER BY node_id ASC",
            (session_id,),
        )
        for (raw,) in cur:
            try:
                msg = json.loads(raw)
            # json.loads() accepts bytes and decodes them as UTF-8, so a BLOB
            # column holding invalid bytes raises UnicodeDecodeError — which
            # would abort the whole read rather than skipping the one row.
            except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                continue
            if not isinstance(msg, dict):
                continue

            # Dedup only on hashable scalar ids. A JSON object or array here would
            # raise TypeError from the set operations below, and the only handler
            # in scope catches sqlite3.Error — so it would escape the "never
            # raises" contract and crash the hook.
            message_id = msg.get("message_id")
            # bool excluded explicitly: it subclasses int, so `"message_id":
            # true` would otherwise become a dedup key shared by every such row.
            if isinstance(message_id, bool) or not isinstance(message_id, (str, int)):
                message_id = None
            # `is not None`, not truthiness: 0 and "" are valid ids, and treating
            # them as absent means every replayed row carrying one is re-emitted
            # instead of de-duplicated.
            if message_id is not None and message_id in seen_message_ids:
                continue

            # Type-checked, not just truth-checked. Every consumer treats `role`
            # as a string — comparing it to "user"/"assistant", and formatting it
            # into the transcript — so a row carrying an object or a number here
            # would leave the reader's "never raises" contract intact only to
            # break the formatter it hands the transcript to.
            role = msg.get("role")
            if not isinstance(role, str) or not role:
                continue

            # Marked seen only once the row is accepted: recording it before the
            # role check would let a malformed row suppress a later valid row
            # carrying the same id, silently dropping that turn.
            if message_id is not None:
                seen_message_ids.add(message_id)

            entry = {"role": role, "content": msg.get("content") or ""}
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    return messages


def fold_tool_calls_into_content(entry: dict) -> object:
    """Convert a {"role", "content", "tool_calls"} entry into content.py's block shape.

    lib/content.py expects `content` to be either a plain string or a list of
    content blocks ({"type": "text", ...} / {"type": "tool_use", ...}). Devin CLI
    stores tool calls in a separate OpenAI-style `tool_calls` field instead of
    inline blocks, so this only needs to run when `retainToolCalls` is enabled.
    """
    tool_calls = entry.get("tool_calls")
    # Type-checked, not just truth-checked: the loop below iterates it, and
    # read_session_messages copies this field through on nothing more than
    # truthiness, so a row storing a scalar (`"tool_calls": 1`) would reach
    # `for call in ...` and raise TypeError inside the retain hook.
    if not isinstance(tool_calls, list) or not tool_calls:
        return entry.get("content", "")

    blocks = []
    text = entry.get("content") or ""
    if text:
        blocks.append({"type": "text", "text": text})
    for call in tool_calls:
        # `.get("function", {})` would still yield None when the key is present
        # with a null value, which is exactly what a truncated tool call looks like.
        fn = call.get("function") if isinstance(call, dict) else None
        if not isinstance(fn, dict):
            fn = {}
        name = fn.get("name", "unknown")
        raw_args = fn.get("arguments")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
        except (json.JSONDecodeError, TypeError):
            args = {}
        block = {"type": "tool_use", "name": name, "input": args}
        # Carry the call id through. content.py suppresses an operational
        # Hindsight tool's *result* by matching `tool_use_id` against the id of
        # the tool_use it dropped — that is the half of the anti-feedback-loop
        # guard that stops recalled memories being retained again. Synthesising
        # a block with no id silently disables it, and the id is right here in
        # the OpenAI-style call we are converting.
        call_id = call.get("id") if isinstance(call, dict) else None
        if isinstance(call_id, str) and call_id:
            block["id"] = call_id
        blocks.append(block)
    return blocks
