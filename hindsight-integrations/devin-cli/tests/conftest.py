"""Shared fixtures for the Hindsight Devin CLI plugin tests."""

import json
import os
import sqlite3
import sys

import pytest

# Make scripts/ importable as the root — the hook scripts do:
#   sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# so lib.* imports resolve relative to scripts/
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))


@pytest.fixture()
def state_dir(tmp_path, monkeypatch):
    """Isolated state directory — prevents tests from touching real state files."""
    d = tmp_path / "state"
    monkeypatch.setenv("HINDSIGHT_DEVIN_CLI_DATA_DIR", str(tmp_path))
    return d


@pytest.fixture()
def default_config(tmp_path, monkeypatch):
    """Load config with no overrides, isolated from real settings.json."""
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({}))
    monkeypatch.setattr("lib.config.plugin_root", lambda: str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))  # keep ~/.hindsight/devin-cli.json out of the way
    for key in list(os.environ):
        if key.startswith("HINDSIGHT_"):
            monkeypatch.delenv(key, raising=False)
    from lib.config import load_config

    return load_config()


def make_hook_input(prompt="What is the capital of France?", session_id="sess-abc123"):
    return {"prompt": prompt, "session_id": session_id}


def make_recall_response(memories):
    """Build a fake /recall API response."""
    return {"results": memories}


def make_memory(text, mem_type="experience", mentioned_at="2024-01-15"):
    return {"text": text, "type": mem_type, "mentioned_at": mentioned_at}


class FakeHTTPResponse:
    """Minimal urllib response mock."""

    def __init__(self, data: dict, status: int = 200):
        self.status = status
        self._data = json.dumps(data).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


@pytest.fixture()
def sessions_db(tmp_path):
    """Build a throwaway sessions.db with the CLI's real message_nodes schema."""
    db_path = tmp_path / "sessions.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE sessions (
          id TEXT PRIMARY KEY,
          working_directory TEXT NOT NULL,
          backend_type TEXT NOT NULL,
          model TEXT NOT NULL,
          agent_mode TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          last_activity_at INTEGER NOT NULL, title TEXT, main_chain_id INTEGER,
          shell_last_seen_index INTEGER DEFAULT 0, cogs_json TEXT, workspace_dirs TEXT,
          hidden INTEGER NOT NULL DEFAULT 0, metadata TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE message_nodes (
          row_id INTEGER PRIMARY KEY AUTOINCREMENT,
          session_id TEXT NOT NULL,
          node_id INTEGER NOT NULL,
          parent_node_id INTEGER,
          chat_message TEXT NOT NULL,
          created_at INTEGER NOT NULL, metadata TEXT,
          UNIQUE(session_id, node_id)
        )
        """
    )
    conn.commit()
    conn.close()
    return str(db_path)


def insert_message(db_path, session_id, node_id, message_id, role, content, tool_calls=None):
    payload = {"message_id": message_id, "role": role, "content": content}
    if tool_calls:
        payload["tool_calls"] = tool_calls
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO message_nodes (session_id, node_id, parent_node_id, chat_message, created_at) "
        "VALUES (?, ?, ?, ?, 0)",
        (session_id, node_id, node_id - 1 if node_id else None, json.dumps(payload)),
    )
    conn.commit()
    conn.close()
