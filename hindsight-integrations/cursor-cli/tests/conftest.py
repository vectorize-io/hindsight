"""Shared fixtures for the Hindsight Cursor CLI integration tests."""

import io
import json
import os
import sys

import pytest

# Make the packaged scripts/ importable as the root — the hook scripts do:
#   sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# so lib.* imports resolve relative to scripts/
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "hindsight_cursor_cli", "hooks", "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))


def make_hook_input(
    prompt="What is the capital of France?",
    conversation_id="sess-abc123",
    cwd="/home/user/myproject",
    transcript_path="",
    workspace_roots=None,
    **extras,
):
    """Build a Cursor-style hook input dict.

    `beforeSubmitPrompt` only carries `prompt` + `attachments`, but
    common fields like `conversation_id`, `transcript_path`, and
    `workspace_roots` are also injected by the runtime, so we include
    them in the fixture for parity with how a real hook sees the world.
    """
    payload = {
        "prompt": prompt,
        "conversation_id": conversation_id,
        "session_id": conversation_id,  # legacy alias — bank.py accepts both
        "cwd": cwd,
        "transcript_path": transcript_path,
        "workspace_roots": workspace_roots or ["/home/user/myproject"],
        "hook_event_name": "beforeSubmitPrompt",
        "model": "composer-2",
        "cursor_version": "0.45.0",
    }
    payload.update(extras)
    return payload


def make_transcript_file(tmp_path, messages, cursor_format=False):
    """Write messages as a JSONL transcript file.

    By default writes the flat format {role, content} which
    read_transcript() accepts. Set cursor_format=True to write the
    real Cursor SDK envelope {type, message: {role, content: [TextBlock]}}.
    """
    f = tmp_path / "transcript-test.jsonl"
    lines = []
    for msg in messages:
        if cursor_format:
            role = msg["role"]
            text = msg["content"]
            envelope = {
                "type": role,  # Cursor uses "user" and "assistant" as the event type
                "message": {
                    "role": role,
                    "content": [{"type": "text", "text": text}],
                },
            }
            lines.append(json.dumps(envelope))
        else:
            lines.append(json.dumps(msg))
    f.write_text("\n".join(lines))
    return str(f)


def make_memory(text, mem_type="experience", mentioned_at="2024-01-15"):
    return {"text": text, "type": mem_type, "mentioned_at": mentioned_at}


def make_user_config(tmp_path, overrides=None):
    """Write a ~/.hindsight/cursor-cli.json in tmp_path with test defaults."""
    hindsight_dir = tmp_path / ".hindsight"
    hindsight_dir.mkdir(exist_ok=True)
    config = {"retainEveryNTurns": 1}
    if overrides:
        config.update(overrides)
    (hindsight_dir / "cursor-cli.json").write_text(json.dumps(config))


class FakeHTTPResponse:
    """Minimal urllib response mock."""

    def __init__(self, data, status=200):
        self.status = status
        self._data = json.dumps(data).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass
