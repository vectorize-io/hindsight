"""Shared fixtures for the Hindsight ZCode integration tests."""

import json
import os
import sys

# Make the packaged scripts/ importable as the root — the hook scripts do:
#   sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# so lib.* imports resolve relative to scripts/
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "hindsight_zcode", "hooks", "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))


def make_hook_input(
    prompt="What is the capital of France?",
    session_id="sess-abc123",
    cwd="/home/user/myproject",
    transcript_path="",
    workspace_roots=None,
    **extras,
):
    """Build a ZCode (Claude Code protocol) hook input dict.

    ZCode embeds the Claude Code agent runtime, so hook payloads carry
    `prompt`, `session_id`, `transcript_path`, and `cwd`. We also include
    `workspace_roots` for parity with how bank derivation sees the world.
    """
    payload = {
        "prompt": prompt,
        "session_id": session_id,
        "cwd": cwd,
        "transcript_path": transcript_path,
        "workspace_roots": workspace_roots or ["/home/user/myproject"],
        "hook_event_name": "UserPromptSubmit",
    }
    payload.update(extras)
    return payload


def make_transcript_file(tmp_path, messages, envelope_format=False, zcode_format=False):
    """Write messages as a JSONL transcript file.

    By default writes the flat format {role, content} which
    read_transcript() accepts. Set envelope_format=True to write the typed
    transcript envelope {type, message: {role, content: [TextBlock]}}.
    Set zcode_format=True to write the ephemeral Stop-hook shape
    {message: {role, content: [TextBlock]}} with NO top-level type.
    """
    f = tmp_path / "transcript-test.jsonl"
    lines = []
    for msg in messages:
        role = msg["role"]
        text = msg["content"]
        if zcode_format:
            envelope = {"message": {"role": role, "content": [{"type": "text", "text": text}]}}
            lines.append(json.dumps(envelope))
        elif envelope_format:
            envelope = {
                "type": role,  # "user" / "assistant" as the event type
                "message": {"role": role, "content": [{"type": "text", "text": text}]},
            }
            lines.append(json.dumps(envelope))
        else:
            lines.append(json.dumps(msg))
    f.write_text("\n".join(lines))
    return str(f)


def make_stop_input(response_text="world", session_id="sess-abc123", cwd="/home/user/myproject", **extras):
    """Build a ZCode `Stop`-hook input dict.

    ZCode's Stop payload carries the assistant reply in `responseText` (and a
    truncated `responsePreview`) plus both `session_id` and `sessionId`.
    """
    payload = {
        "responseText": response_text,
        "responsePreview": response_text[:100],
        "session_id": session_id,
        "sessionId": session_id,
        "cwd": cwd,
        "workspace_roots": ["/home/user/myproject"],
        "toolCallCount": 0,
        "hookEventName": "Stop",
    }
    payload.update(extras)
    return payload


def stash_prompt(tmp_path, session_id, prompt):
    """Seed the state file the recall hook would write, so retain can pair it."""
    state_dir = tmp_path / ".hindsight" / "zcode" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / f"last_prompt_{session_id}.json").write_text(
        json.dumps({"prompt": prompt, "ts": "2024-01-15T00:00:00Z"})
    )


def make_memory(text, mem_type="experience", mentioned_at="2024-01-15"):
    return {"text": text, "type": mem_type, "mentioned_at": mentioned_at}


def make_user_config(tmp_path, overrides=None):
    """Write a ~/.hindsight/zcode.json in tmp_path with test defaults."""
    hindsight_dir = tmp_path / ".hindsight"
    hindsight_dir.mkdir(exist_ok=True)
    config = {"retainEveryNTurns": 1}
    if overrides:
        config.update(overrides)
    (hindsight_dir / "zcode.json").write_text(json.dumps(config))


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
