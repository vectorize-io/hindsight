"""Drive one real Claude Code session through a scripted list of turns.

**One process for the whole session, fed over stream-json.** The obvious
alternative — `claude -p` once per turn with `--resume` — fires SessionStart on
every turn, so the plugin re-injects its tool guide before each prompt and the
measured search rate is an artifact of the harness. Verified on Claude Code
2.1.274: stream-json gives 1 SessionStart, N UserPromptSubmit and N Stop for N
prompts, which is the shape a human session has.

The next prompt is sent only after the previous turn's `result` event, so the
turns stay ordered and each one's hooks have run before the next begins.
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hindsight_system_evals.coding_agents.sandbox import Sandbox

#: A turn that has not produced its result by now is treated as hung. Generous:
#: a turn may run tools, and the plugin's own reflect injection can add ~20s.
TURN_TIMEOUT_SECONDS = 600.0


#: How Claude Code names a tool of the plugin's MCP server; stripped so a call reads as the plugin names it.
_MCP_PREFIX = "mcp__hindsight__"


@dataclass
class ToolCall:
    name: str
    #: The arguments as the agent sent them: whatever JSON that tool's schema takes.
    input: dict[str, Any]


@dataclass
class Turn:
    prompt: str
    reply: str
    calls: list[ToolCall] = field(default_factory=list)

    @property
    def tools(self) -> list[str]:
        return [call.name for call in self.calls]


@dataclass
class SessionResult:
    session_id: str
    prompts: list[str]
    turns: list[Turn] = field(default_factory=list)
    #: Every tool the session was offered, from its init event, MCP prefix stripped. Lets a test
    #: that asserts a tool was NOT called tell "chose not to" from "never had it".
    offered: list[str] = field(default_factory=list)
    transcript: Path | None = None
    error: str | None = None

    @property
    def replies(self) -> list[str]:
        return [t.reply for t in self.turns]

    @property
    def completed_turns(self) -> int:
        return len(self.turns)


def run_session(
    sandbox: Sandbox,
    prompts: list[str],
    *,
    model: str = "opus",
    log_path: Path | None = None,
    debug_path: Path | None = None,
) -> SessionResult:
    session_id = str(uuid.uuid4())
    result = SessionResult(session_id=session_id, prompts=list(prompts))
    log = log_path.open("w", encoding="utf-8") if log_path else None

    process = subprocess.Popen(
        [
            "claude",
            "--print",
            "--model",
            model,
            "--session-id",
            session_id,
            "--permission-mode",
            "bypassPermissions",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            # Hindsight's tools and nothing else — see Sandbox.mcp_config.
            "--mcp-config",
            str(sandbox.mcp_config),
            "--strict-mcp-config",
            # The API debug log is the only place the FULL prompt is visible:
            # what a hook injected, where it landed, and what the model was
            # actually holding when it chose not to search. Without it a low
            # rate can only be stared at.
            *(["--debug-file", str(debug_path), "--debug", "api,hooks"] if debug_path else []),
        ],
        cwd=sandbox.workdir,
        env=sandbox.env(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def send(text: str) -> None:
        assert process.stdin is not None
        process.stdin.write(
            json.dumps({"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": text}]}})
            + "\n"
        )
        process.stdin.flush()

    calls: list[ToolCall] = []
    try:
        send(prompts[0])
        deadline = time.monotonic() + TURN_TIMEOUT_SECONDS
        assert process.stdout is not None
        for line in process.stdout:
            if log:
                log.write(line)
            if time.monotonic() > deadline:
                result.error = f"turn {len(result.replies) + 1} exceeded {TURN_TIMEOUT_SECONDS}s"
                break
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "system" and event.get("subtype") == "init":
                result.offered = _offered(event)
                continue
            if event.get("type") == "assistant":
                calls += _tool_calls(event)
                continue
            if event.get("type") != "result":
                continue
            done = len(result.turns)
            result.turns.append(Turn(prompt=prompts[done], reply=str(event.get("result", "")), calls=calls))
            calls = []
            if event.get("is_error"):
                result.error = f"turn {len(result.turns)} returned an error: {event.get('result')}"
                break
            if len(result.turns) >= len(prompts):
                break
            send(prompts[len(result.turns)])
            deadline = time.monotonic() + TURN_TIMEOUT_SECONDS
    finally:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
        if log:
            log.close()

    if result.error is None and result.completed_turns < len(prompts):
        stderr = (process.stderr.read() if process.stderr else "") or ""
        result.error = f"session ended after {result.completed_turns}/{len(prompts)} turns: {stderr[-500:]}"

    transcript = _find_transcript(sandbox, session_id)
    result.transcript = transcript
    return result


def _tool_calls(event: dict[str, Any]) -> list[ToolCall]:
    """The tool calls in one assistant message of the stream-json output."""
    content = (event.get("message") or {}).get("content") or []
    return [
        ToolCall(name=str(block.get("name", "")).removeprefix(_MCP_PREFIX), input=block.get("input") or {})
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]


def _offered(event: dict[str, Any]) -> list[str]:
    """The tools a session's init event offers, named as the plugin names them."""
    return [str(t).removeprefix(_MCP_PREFIX) for t in event.get("tools") or []]


def _find_transcript(sandbox: Sandbox, session_id: str) -> Path | None:
    """Claude names the transcript after the session id, under a slug of the cwd."""
    projects = sandbox.home / ".claude" / "projects"
    matches = list(projects.glob(f"*/{session_id}.jsonl")) if projects.exists() else []
    return matches[0] if matches else None
