"""Shared fixtures for Hindsight Codex plugin tests."""

import io
import json
import os
import sys

import pytest

# Make scripts/ importable as the root — the hook scripts do:
#   sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# so lib.* imports resolve relative to scripts/
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))


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


def make_transcript_file(tmp_path, messages, codex_format=False):
    """Write messages as a JSONL transcript file.

    By default writes flat format {role, content} which read_transcript() accepts.
    Set codex_format=True to write actual Codex response_item format.
    """
    f = tmp_path / "rollout-test.jsonl"
    lines = []
    for msg in messages:
        if codex_format:
            role = msg["role"]
            text = msg["content"]
            content_type = "input_text" if role == "user" else "output_text"
            entry = {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": role,
                    "content": [{"type": content_type, "text": text}],
                },
            }
            if role == "assistant":
                entry["payload"]["phase"] = "final_answer"
            lines.append(json.dumps(entry))
        else:
            lines.append(json.dumps(msg))
    f.write_text("\n".join(lines))
    return str(f)


def make_memory(text, mem_type="experience", mentioned_at="2024-01-15"):
    return {"text": text, "type": mem_type, "mentioned_at": mentioned_at}


def make_user_config(tmp_path, overrides=None):
    """Write a ~/.hindsight/codex.json in tmp_path with test defaults."""
    hindsight_dir = tmp_path / ".hindsight"
    hindsight_dir.mkdir(exist_ok=True)
    config = {"retainEveryNTurns": 1}
    if overrides:
        config.update(overrides)
    (hindsight_dir / "codex.json").write_text(json.dumps(config))


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
