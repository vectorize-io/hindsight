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

from hindsight_system_evals.coding_agents.sandbox import Sandbox

#: A turn that has not produced its result by now is treated as hung. Generous:
#: a turn may run tools, and the plugin's own reflect injection can add ~20s.
TURN_TIMEOUT_SECONDS = 600.0


@dataclass
class SessionResult:
    session_id: str
    prompts: list[str]
    replies: list[str] = field(default_factory=list)
    transcript: Path | None = None
    error: str | None = None

    @property
    def completed_turns(self) -> int:
        return len(self.replies)


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
            if event.get("type") != "result":
                continue
            result.replies.append(str(event.get("result", "")))
            if event.get("is_error"):
                result.error = f"turn {len(result.replies)} returned an error: {event.get('result')}"
                break
            if len(result.replies) >= len(prompts):
                break
            send(prompts[len(result.replies)])
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


def _find_transcript(sandbox: Sandbox, session_id: str) -> Path | None:
    """Claude names the transcript after the session id, under a slug of the cwd."""
    projects = sandbox.home / ".claude" / "projects"
    matches = list(projects.glob(f"*/{session_id}.jsonl")) if projects.exists() else []
    return matches[0] if matches else None
