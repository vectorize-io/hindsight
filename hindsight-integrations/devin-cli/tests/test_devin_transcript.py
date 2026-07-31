"""Tests for lib/devin_transcript.py — reading Devin CLI's session database.

This module reads an *undocumented* CLI internal: there is no public API or
stability guarantee for `sessions.db`. The strategy that makes that acceptable
is that every failure path degrades to an empty transcript, so a future CLI
release that changes the storage layer turns retain into a no-op instead of
raising out of a hook (which Devin CLI would read as "block this turn").

`TestSchemaAssumptions` states exactly what the module depends on, and
`TestAgainstRealDevinCliDatabase` checks those assumptions against the actual
CLI database when one is present — a local signal for anyone running the suite
on a machine with Devin CLI installed. It skips in CI, so it can never be the
thing that keeps the schema honest on its own; the degradation tests are.
"""

import json
import os
import sqlite3

import pytest
from conftest import insert_message

from lib.devin_transcript import (
    fold_tool_calls_into_content,
    read_session_messages,
    sessions_db_path,
)

# The columns of `message_nodes` that lib/devin_transcript.py actually reads or
# orders by. Keep in sync with the module's SELECT — this list is what
# TestAgainstRealDevinCliDatabase validates against a live CLI install.
REQUIRED_COLUMNS = {"session_id", "node_id", "chat_message"}

# Keys read out of the JSON payload in `chat_message`.
REQUIRED_PAYLOAD_KEYS = {"message_id", "role", "content"}


class TestReadSessionMessages:
    def test_missing_db_returns_empty(self, tmp_path):
        assert read_session_messages("sess-1", db_path=str(tmp_path / "missing.db")) == []

    def test_empty_session_id_returns_empty(self, sessions_db):
        assert read_session_messages("", db_path=sessions_db) == []

    def test_unknown_session_returns_empty(self, sessions_db):
        insert_message(sessions_db, "sess-1", 0, "m1", "user", "hi")
        assert read_session_messages("sess-does-not-exist", db_path=sessions_db) == []

    def test_reads_messages_in_order(self, sessions_db):
        insert_message(sessions_db, "sess-1", 0, "m1", "system", "system prompt")
        insert_message(sessions_db, "sess-1", 1, "m2", "user", "hello")
        insert_message(sessions_db, "sess-1", 2, "m3", "assistant", "hi there")

        messages = read_session_messages("sess-1", db_path=sessions_db)

        assert [m["role"] for m in messages] == ["system", "user", "assistant"]
        assert messages[1]["content"] == "hello"
        assert messages[2]["content"] == "hi there"

    def test_deduplicates_by_message_id_keeping_first_occurrence(self, sessions_db):
        # Devin CLI re-inserts the same system/rules messages on every turn —
        # only the first occurrence (by node_id) should survive.
        insert_message(sessions_db, "sess-1", 0, "sys", "system", "rules v1")
        insert_message(sessions_db, "sess-1", 1, "u1", "user", "first question")
        insert_message(sessions_db, "sess-1", 2, "sys", "system", "rules v1")  # re-sent, same message_id
        insert_message(sessions_db, "sess-1", 3, "a1", "assistant", "first answer")
        insert_message(sessions_db, "sess-1", 4, "sys", "system", "rules v1")
        insert_message(sessions_db, "sess-1", 5, "u1", "user", "first question")
        insert_message(sessions_db, "sess-1", 6, "u2", "user", "second question")

        messages = read_session_messages("sess-1", db_path=sessions_db)

        assert [(m["role"], m["content"]) for m in messages] == [
            ("system", "rules v1"),
            ("user", "first question"),
            ("assistant", "first answer"),
            ("user", "second question"),
        ]

    def test_skips_rows_without_message_id_role_pairing_gracefully(self, sessions_db):
        insert_message(sessions_db, "sess-1", 0, "m1", "user", "ok")
        # Row with no role at all (malformed) — should just be skipped, not raise.
        conn = sqlite3.connect(sessions_db)
        conn.execute(
            "INSERT INTO message_nodes (session_id, node_id, parent_node_id, chat_message, created_at) "
            "VALUES (?, ?, ?, ?, 0)",
            ("sess-1", 1, 0, json.dumps({"message_id": "bad"})),
        )
        conn.commit()
        conn.close()

        messages = read_session_messages("sess-1", db_path=sessions_db)
        assert len(messages) == 1
        assert messages[0]["content"] == "ok"

    @pytest.mark.parametrize("bad_role", [{"name": "user"}, ["user"], 7, True])
    def test_skips_rows_whose_role_is_not_a_string(self, sessions_db, bad_role):
        """A truthy non-string role passes an emptiness check and breaks the formatter.

        This reader promises never to raise, but it hands its output straight to
        code that compares `role` to "user" and formats it into the transcript.
        Letting a dict through moves the crash rather than preventing it.
        """
        insert_message(sessions_db, "sess-1", 0, "m1", "user", "ok")
        conn = sqlite3.connect(sessions_db)
        conn.execute(
            "INSERT INTO message_nodes (session_id, node_id, parent_node_id, chat_message, created_at) "
            "VALUES (?, ?, ?, ?, 0)",
            ("sess-1", 1, 0, json.dumps({"message_id": "m2", "role": bad_role, "content": "x"})),
        )
        conn.commit()
        conn.close()

        messages = read_session_messages("sess-1", db_path=sessions_db)
        assert [m["role"] for m in messages] == ["user"]

    @pytest.mark.parametrize("falsy_id", [0, ""])
    def test_a_falsy_but_valid_message_id_is_still_deduplicated(self, sessions_db, falsy_id):
        """0 and "" are ids, not absences.

        Devin CLI replays the whole context window on every turn, so a row that
        is not deduplicated appears once per subsequent turn — the duplication
        this reader exists to undo.
        """
        for node_id in range(3):
            conn = sqlite3.connect(sessions_db)
            conn.execute(
                "INSERT INTO message_nodes (session_id, node_id, parent_node_id, chat_message, created_at) "
                "VALUES (?, ?, ?, ?, 0)",
                ("sess-1", node_id, None, json.dumps({"message_id": falsy_id, "role": "system", "content": "rules"})),
            )
            conn.commit()
            conn.close()

        messages = read_session_messages("sess-1", db_path=sessions_db)

        assert len(messages) == 1, f"message_id {falsy_id!r} was treated as absent"

    def test_a_malformed_row_does_not_suppress_a_later_valid_row(self, sessions_db):
        """The dedup set must only record ids of rows we actually kept.

        A roleless row sharing an id with a real turn would otherwise mask it,
        dropping that turn from every recall and retain for the session.
        """
        conn = sqlite3.connect(sessions_db)
        conn.execute(
            "INSERT INTO message_nodes (session_id, node_id, parent_node_id, chat_message, created_at) "
            "VALUES (?, ?, ?, ?, 0)",
            ("sess-1", 0, None, json.dumps({"message_id": "m1"})),  # no role
        )
        conn.commit()
        conn.close()
        insert_message(sessions_db, "sess-1", 1, "m1", "user", "the real turn")

        messages = read_session_messages("sess-1", db_path=sessions_db)

        assert [m["content"] for m in messages] == ["the real turn"]

    def test_an_unhashable_message_id_degrades_instead_of_raising(self, sessions_db):
        """`message_id` comes from an undocumented CLI payload, so its type is not ours.

        A JSON object or array there raises TypeError from the dedup set, which the
        sqlite3-only handler would not catch — breaking the "never raises" contract
        and taking the hook down with it.
        """
        conn = sqlite3.connect(sessions_db)
        conn.execute(
            "INSERT INTO message_nodes (session_id, node_id, parent_node_id, chat_message, created_at) "
            "VALUES (?, ?, ?, ?, 0)",
            ("sess-1", 0, None, json.dumps({"message_id": {"nested": "id"}, "role": "user", "content": "hi"})),
        )
        conn.commit()
        conn.close()

        messages = read_session_messages("sess-1", db_path=sessions_db)

        assert [m["content"] for m in messages] == ["hi"]

    def test_carries_tool_calls_when_present(self, sessions_db):
        tool_calls = [
            {"id": "tc1", "type": "function", "function": {"name": "read", "arguments": '{"file_path": "a.py"}'}}
        ]
        insert_message(sessions_db, "sess-1", 0, "a1", "assistant", "", tool_calls=tool_calls)

        messages = read_session_messages("sess-1", db_path=sessions_db)

        assert messages[0]["tool_calls"] == tool_calls


class TestFoldToolCallsIntoContent:
    def test_plain_content_passes_through_unchanged(self):
        entry = {"role": "user", "content": "hello"}
        assert fold_tool_calls_into_content(entry) == "hello"

    def test_folds_tool_calls_into_content_blocks(self):
        entry = {
            "role": "assistant",
            "content": "Let me check that file.",
            "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "read", "arguments": '{"file_path": "a.py"}'}}
            ],
        }

        blocks = fold_tool_calls_into_content(entry)

        assert blocks == [
            {"type": "text", "text": "Let me check that file."},
            {"type": "tool_use", "name": "read", "input": {"file_path": "a.py"}, "id": "tc1"},
        ]

    def test_handles_malformed_arguments_gracefully(self):
        entry = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "read", "arguments": "not json"}}],
        }

        blocks = fold_tool_calls_into_content(entry)

        assert blocks == [{"type": "tool_use", "name": "read", "input": {}, "id": "tc1"}]

    def test_handles_a_null_function_field(self):
        """A truncated tool call carries `function: null`, not a missing key.

        `dict.get(key, default)` returns the default only when the key is absent,
        so the null value would reach `.get("name")` and raise AttributeError.
        """
        entry = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "tc1", "type": "function", "function": None}],
        }

        blocks = fold_tool_calls_into_content(entry)

        assert blocks == [{"type": "tool_use", "name": "unknown", "input": {}, "id": "tc1"}]

    @pytest.mark.parametrize("tool_calls", [1, "read", {"id": "tc1"}, True])
    def test_a_non_list_tool_calls_degrades_to_plain_content(self, tool_calls):
        """`read_session_messages` copies this field through on truthiness alone.

        A scalar therefore reaches `for call in tool_calls` and raises
        TypeError inside the retain hook.
        """
        entry = {"role": "assistant", "content": "just text", "tool_calls": tool_calls}

        assert fold_tool_calls_into_content(entry) == "just text"

    def test_the_call_id_is_carried_onto_the_tool_use_block(self):
        """content.py suppresses an operational tool's *result* by matching this id.

        Synthesising a block without it silently disables half the
        anti-feedback-loop guard, so recalled memories are retained again.
        """
        entry = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "mcp__hindsight__recall", "arguments": "{}"},
                }
            ],
        }

        blocks = fold_tool_calls_into_content(entry)

        assert blocks[0]["id"] == "call_abc"

    @pytest.mark.parametrize("bad_id", [None, 7, {"a": 1}, ""])
    def test_a_missing_or_unusable_call_id_is_simply_absent(self, bad_id):
        entry = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": bad_id, "type": "function", "function": {"name": "read", "arguments": "{}"}}],
        }

        blocks = fold_tool_calls_into_content(entry)

        assert "id" not in blocks[0]


class TestSchemaAssumptions:
    """Pin what the module needs from `sessions.db`, and prove it degrades.

    These are the tests that make reading an undocumented internal defensible.
    Each one removes something the module assumes and asserts an empty list
    rather than an exception, because an exception here becomes a nonzero hook
    exit and Devin CLI treats exit code 2 as "block the turn".
    """

    def test_fixture_matches_the_documented_column_set(self, sessions_db):
        """Guard the fixture itself, so it can't silently drift from reality."""
        conn = sqlite3.connect(sessions_db)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(message_nodes)")}
        conn.close()
        assert REQUIRED_COLUMNS <= columns

    def test_missing_table_degrades_to_empty(self, tmp_path):
        db = tmp_path / "no-table.db"
        sqlite3.connect(str(db)).close()

        assert read_session_messages("sess-1", db_path=str(db)) == []

    def test_renamed_column_degrades_to_empty(self, tmp_path):
        """A CLI release renaming `chat_message` disables retain, not the hook."""
        db = tmp_path / "renamed.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE message_nodes (  row_id INTEGER PRIMARY KEY, session_id TEXT, node_id INTEGER, payload TEXT)"
        )
        conn.execute(
            "INSERT INTO message_nodes (session_id, node_id, payload) VALUES (?, ?, ?)",
            ("sess-1", 0, json.dumps({"message_id": "m1", "role": "user", "content": "hi"})),
        )
        conn.commit()
        conn.close()

        assert read_session_messages("sess-1", db_path=str(db)) == []

    def test_non_json_payload_degrades_to_empty(self, sessions_db):
        conn = sqlite3.connect(sessions_db)
        conn.execute(
            "INSERT INTO message_nodes (session_id, node_id, parent_node_id, chat_message, created_at) "
            "VALUES (?, ?, ?, ?, 0)",
            ("sess-1", 0, None, "<not json at all>"),
        )
        conn.commit()
        conn.close()

        assert read_session_messages("sess-1", db_path=sessions_db) == []

    def test_json_payload_that_is_not_an_object_is_skipped(self, sessions_db):
        conn = sqlite3.connect(sessions_db)
        conn.execute(
            "INSERT INTO message_nodes (session_id, node_id, parent_node_id, chat_message, created_at) "
            "VALUES (?, ?, ?, ?, 0)",
            ("sess-1", 0, None, json.dumps(["a", "list"])),
        )
        conn.commit()
        conn.close()

        assert read_session_messages("sess-1", db_path=sessions_db) == []

    def test_a_corrupt_database_file_degrades_to_empty(self, tmp_path):
        db = tmp_path / "corrupt.db"
        db.write_bytes(b"this is not an sqlite file" * 100)

        assert read_session_messages("sess-1", db_path=str(db)) == []

    def test_read_only_connection_never_creates_a_database(self, tmp_path):
        """The plugin must not fabricate a sessions.db the CLI would then see."""
        db = tmp_path / "absent.db"

        assert read_session_messages("sess-1", db_path=str(db)) == []
        assert not db.exists()

    def test_reads_while_a_writer_holds_an_open_transaction(self, sessions_db):
        """WAL mode: the CLI writes live during a session; reads must not block.

        Without WAL + a read-only connection this deadlocks until the 5s
        timeout, which would blow the Stop hook's budget on every turn.
        """
        conn = sqlite3.connect(sessions_db)
        conn.execute("PRAGMA journal_mode=WAL")
        insert_message(sessions_db, "sess-1", 0, "m1", "user", "committed")
        conn.execute("BEGIN")
        conn.execute(
            "INSERT INTO message_nodes (session_id, node_id, parent_node_id, chat_message, created_at) "
            "VALUES (?, ?, ?, ?, 0)",
            ("sess-1", 1, 0, json.dumps({"message_id": "m2", "role": "user", "content": "pending"})),
        )
        try:
            messages = read_session_messages("sess-1", db_path=sessions_db)
        finally:
            conn.rollback()
            conn.close()

        assert [m["content"] for m in messages] == ["committed"]


REAL_DB = sessions_db_path()


@pytest.mark.skipif(
    not os.path.isfile(REAL_DB),
    reason="Devin CLI is not installed on this machine (no sessions.db)",
)
class TestAgainstRealDevinCliDatabase:
    """Validate the schema assumptions against a live Devin CLI install.

    Skipped in CI by design — CI has no Devin CLI. When it does run (a
    developer's machine, or a user debugging why retain went quiet), a failure
    here is the earliest possible warning that a CLI upgrade moved the schema
    out from under this integration. Nothing reads session *content*; only
    `PRAGMA table_info`.
    """

    def test_message_nodes_still_has_the_columns_we_query(self):
        conn = sqlite3.connect(f"file:{REAL_DB}?mode=ro", uri=True, timeout=5)
        try:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(message_nodes)")}
        finally:
            conn.close()

        missing = REQUIRED_COLUMNS - columns
        assert not missing, (
            f"Devin CLI's message_nodes no longer has {sorted(missing)}. "
            f"lib/devin_transcript.py degrades to an empty transcript, so "
            f"auto-retain is now silently disabled — update the read path."
        )

    def test_the_modules_query_executes_against_the_real_schema(self):
        """Run the exact SELECT with a session_id that matches nothing."""
        conn = sqlite3.connect(f"file:{REAL_DB}?mode=ro", uri=True, timeout=5)
        try:
            rows = conn.execute(
                "SELECT chat_message FROM message_nodes WHERE session_id = ? ORDER BY node_id ASC",
                ("__hindsight_test_no_such_session__",),
            ).fetchall()
        finally:
            conn.close()

        assert rows == []


class TestDatabasePathsNeedingUriEscaping:
    """SQLite parses a `file:` URI, so `?` and `%` in the path are syntax.

    A HINDSIGHT_DEVIN_SESSIONS_DB containing either used to open some other
    filename — or none — and degrade to an empty transcript while the real
    database sat right there, readable.
    """

    @pytest.mark.parametrize("dirname", ["with?question", "with%percent", "with #hash"])
    def test_a_path_needing_escaping_is_still_read(self, tmp_path, dirname):
        db_dir = tmp_path / dirname
        db_dir.mkdir()
        db_path = db_dir / "sessions.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE message_nodes (node_id INTEGER, session_id TEXT, chat_message TEXT)")
        conn.execute(
            "INSERT INTO message_nodes VALUES (1, 'sess-1', ?)",
            (json.dumps({"message_id": "m1", "role": "user", "content": "hello"}),),
        )
        conn.commit()
        conn.close()

        messages = read_session_messages("sess-1", str(db_path))

        assert [m["content"] for m in messages] == ["hello"], f"path {str(db_path)!r} was not opened"

    def test_the_connection_is_still_read_only(self, tmp_path):
        """Escaping must not drop the ?mode=ro parameter."""
        db_dir = tmp_path / "with?question"
        db_dir.mkdir()
        db_path = db_dir / "sessions.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE message_nodes (node_id INTEGER, session_id TEXT, chat_message TEXT)")
        conn.commit()
        conn.close()

        import urllib.parse

        escaped = urllib.parse.quote(str(db_path))
        ro = sqlite3.connect(f"file:{escaped}?mode=ro", uri=True)
        try:
            with pytest.raises(sqlite3.OperationalError):
                ro.execute("INSERT INTO message_nodes VALUES (9, 'x', '{}')")
        finally:
            ro.close()
