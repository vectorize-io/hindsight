"""Shared fixtures for the Hindsight GitHub Copilot CLI integration tests."""

import json
import os
import sys

# Make the packaged scripts/ importable as the root — the hook scripts do:
#   sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# so lib.* imports resolve relative to scripts/
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "hindsight_copilot_cli", "hooks", "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))


def make_hook_input(
    session_id="sess-abc123",
    cwd="/home/user/myproject",
    transcript_path="",
    initial_prompt=None,
    source="new",
    reason=None,
    stop_reason=None,
    **extras,
):
    """Build a Copilot-CLI-style (camelCase) hook input dict.

    Covers the union of fields across `sessionStart`/`agentStop`/`sessionEnd`
    payloads; callers only need to pass what's relevant for the hook under
    test — unused fields are simply extra keys the real CLI wouldn't send,
    which every hook script ignores gracefully.
    """
    payload = {
        "sessionId": session_id,
        "timestamp": 1704614400000,
        "cwd": cwd,
    }
    if transcript_path:
        payload["transcriptPath"] = transcript_path
    if initial_prompt is not None:
        payload["initialPrompt"] = initial_prompt
    if source is not None:
        payload["source"] = source
    if reason is not None:
        payload["reason"] = reason
    if stop_reason is not None:
        payload["stopReason"] = stop_reason
    payload.update(extras)
    return payload


def make_subagent_hook_input(
    session_id="sess-abc123",
    cwd="/home/user/myproject",
    transcript_path="/tmp/sub-transcript.jsonl",
    agent_name="explore",
    agent_display_name=None,
    agent_description=None,
    **extras,
):
    """Build a Copilot-CLI-style `subagentStart` hook input dict."""
    payload = {
        "sessionId": session_id,
        "timestamp": 1704614400000,
        "cwd": cwd,
        "transcriptPath": transcript_path,
        "agentName": agent_name,
    }
    if agent_display_name is not None:
        payload["agentDisplayName"] = agent_display_name
    if agent_description is not None:
        payload["agentDescription"] = agent_description
    payload.update(extras)
    return payload


def make_transcript_file(tmp_path, messages, sdk_format=False):
    """Write messages as a JSONL transcript file.

    By default writes the flat format {role, content} which
    read_transcript() accepts. Set sdk_format=True to write the
    type-nested SDK envelope {type, message: {role, content: [TextBlock]}}
    that Copilot CLI's docs describe.
    """
    f = tmp_path / "transcript-test.jsonl"
    lines = []
    for msg in messages:
        if sdk_format:
            role = msg["role"]
            text = msg["content"]
            envelope = {
                "type": role,
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
    """Write a ~/.hindsight/copilot-cli.json in tmp_path with test defaults."""
    hindsight_dir = tmp_path / ".hindsight"
    hindsight_dir.mkdir(exist_ok=True)
    config = {"retainEveryNTurns": 1}
    if overrides:
        config.update(overrides)
    (hindsight_dir / "copilot-cli.json").write_text(json.dumps(config))


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
