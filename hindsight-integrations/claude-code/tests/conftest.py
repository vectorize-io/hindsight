"""Shared fixtures for Hindsight Claude Code plugin tests."""

import io
import json
import os
import sys
import tempfile

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
    d.mkdir()
    monkeypatch.setenv("CLAUDE_PLUGIN_DATA", str(tmp_path))
    return d


@pytest.fixture()
def plugin_root(tmp_path):
    """Temp plugin root with a minimal settings.json."""
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({}))
    return tmp_path


@pytest.fixture()
def default_config(plugin_root, monkeypatch):
    """Load config with no overrides, isolated from real settings.json."""
    monkeypatch.setenv("CLAUDE_PLUGIN_ROOT", str(plugin_root))
    # Strip any real HINDSIGHT_* env vars that might bleed in
    for key in list(os.environ):
        if key.startswith("HINDSIGHT_"):
            monkeypatch.delenv(key, raising=False)
    from lib.config import load_config

    return load_config()


def make_hook_input(
    prompt="What is the capital of France?",
    session_id="sess-abc123",
    cwd="/home/user/myproject",
    transcript_path="",
):
    return {
        "prompt": prompt,
        "session_id": session_id,
        "cwd": cwd,
        "transcript_path": transcript_path,
    }


def make_transcript_file(tmp_path, messages):
    """Write messages as a JSONL transcript file (flat test format)."""
    f = tmp_path / "transcript.jsonl"
    lines = [json.dumps(m) for m in messages]
    f.write_text("\n".join(lines))
    return str(f)


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
