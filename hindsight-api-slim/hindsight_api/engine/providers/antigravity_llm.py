"""Google Antigravity LLM provider driving the official ``agy`` CLI headlessly.

Lets a Google AI subscription that includes Antigravity back Hindsight's internal
LLM work, the same way ``cursor``, ``claude-code`` and ``openai-codex`` already do
for theirs (issue #5766). Hindsight only launches Google's own binary; it never
reads or stores the OAuth credentials, which ``agy`` keeps in its own login.

Everything but the subprocess is ``CursorLLM``'s: prompt flattening, retries,
JSON parsing, metrics and spans. What differs is how the turn is run:

- The prompt goes in on stdin as one ``stream-json`` message, never in argv —
  Linux caps a single argument at 128KB, and a reflect or consolidation prompt
  can be larger (verified: a 190KB prompt round-trips on agy 1.5.15).
- ``agy`` enforces a JSON schema natively (``--json-schema``), so structured
  output is model-enforced, not just requested in the prompt.

Verified against agy 1.5.15.
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from hindsight_api.engine.llm_interface import ProviderAuthError
from hindsight_api.engine.providers.cursor_llm import CursorLLM, _CliTurn

logger = logging.getLogger(__name__)

#: Every permission namespace ``agy`` has, denied. The CLI's tools (file reads,
#: shell, web, MCP, subagents) all route through these, and Hindsight's prompts
#: carry arbitrary retained text, so nothing in them may reach a file or a shell.
#: Passed per run via ``--settings``, which is never saved, so the operator's own
#: ``settings.json`` is untouched. Verified: with these rules a prompt asking to
#: read, ``cat`` or ``ls`` a canary file is refused with "Matches user-configured
#: deny rule". Namespaces from https://antigravity.google/docs/permissions.md.
_SESSION_SETTINGS = json.dumps(
    {
        "toolPermission": "strict",
        "permissions": {
            "deny": [
                f"{ns}(*)"
                for ns in (
                    "read_file",
                    "write_file",
                    "read_url",
                    "execute_url",
                    "download_url",
                    "command",
                    "unsandboxed",
                    "mcp",
                )
            ]
        },
    }
)

#: The agent's cwd: an empty directory, so the server's own files stay out of reach
#: even for a tool a later CLI adds outside the namespaces above.
_workspace_lock = threading.Lock()
_workspace: str | None = None


def _get_workspace() -> str:
    global _workspace
    with _workspace_lock:
        if _workspace is None:
            _workspace = tempfile.mkdtemp(prefix="hindsight-antigravity-ws-")
        return _workspace


def _resolve_binary() -> str:
    path = shutil.which("agy")
    if path:
        return path
    raise RuntimeError(
        "agy CLI not found on PATH.\n\n"
        "To set up the Antigravity provider:\n"
        "1. Install the Antigravity CLI (https://antigravity.google)\n"
        "2. Log in once by running: agy\n"
        "3. Verify: agy models"
    )


class _AgyUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class _AgyResult(BaseModel):
    status: str = ""
    response: str = ""
    #: The schema-validated answer, present when ``--json-schema`` was passed. Free-form
    #: JSON with no fixed shape: it is whatever schema the caller asked for.
    structured_output: Any = None
    error: str = ""
    usage: _AgyUsage = Field(default_factory=_AgyUsage)


class _AgyEvent(BaseModel):
    """The ``{"event": "result", "result": {...}}`` line a stream-json run ends with."""

    event: str
    result: _AgyResult | None = None


class AntigravityLLM(CursorLLM):
    """LLM provider backed by a Google Antigravity subscription via the ``agy`` CLI."""

    def _resolve_binary(self) -> str:
        return _resolve_binary()

    async def _run(self, prompt: str, scope: str, schema: dict[str, Any] | None) -> _CliTurn:
        args = [
            self._binary,
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--disable-slash-commands",
            "--settings",
            _SESSION_SETTINGS,
        ]
        if self.model:
            args += ["--model", self.model]
        if schema is not None:
            args += ["--json-schema", json.dumps(schema)]

        # agy runs the operator's hooks.json even in headless mode (verified on 1.5.15).
        # With the hindsight coding-agents integration installed, those hooks recall and
        # retain into Hindsight itself — the recursive loop of #1751 — and the recall hook
        # alone added ~30s per call. That integration skips its hooks under this variable.
        env = {**os.environ, "NO_COLOR": "1", "HINDSIGHT_DISABLE_HOOKS": "1"}
        payload = json.dumps({"event": "user", "message": {"role": "user", "content": prompt}}) + "\n"

        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=_get_workspace(),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(payload.encode()), timeout=self.timeout)
        except TimeoutError as e:
            proc.kill()
            await proc.wait()
            raise TimeoutError(
                f"agy did not answer within {self.timeout}s. Raise "
                "HINDSIGHT_API_LLM_TIMEOUT / HINDSIGHT_API_REFLECT_LLM_TIMEOUT: an agent CLI turn is "
                "slower than a chat-completions request."
            ) from e
        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()
            raise

        result: _AgyResult | None = None
        for line in stdout.decode(errors="replace").splitlines():
            if not line.startswith("{"):
                continue
            try:
                event = _AgyEvent.model_validate_json(line)
            except ValidationError:
                continue
            if event.event == "result" and event.result is not None:
                result = event.result

        if result is None:
            detail = (stderr.decode(errors="replace").strip() or stdout.decode(errors="replace").strip())[-2000:]
            raise RuntimeError(f"agy exited with {proc.returncode} and no result: {detail}")
        if result.status != "SUCCESS":
            message = f"agy reported an error: {result.error or result.status}"
            # Signed out, agy opens a browser login on the server and waits 60s for it
            # ("authentication failed or timed out"). Every retry would repeat that wait,
            # so a missing login is permanent until someone runs `agy` and signs in.
            if "authentication" in result.error.lower():
                raise ProviderAuthError(f"{message}. Run `agy` once as the user running Hindsight and sign in.")
            raise RuntimeError(message)

        # With a schema, read the validated object, not the reply text: the text can carry
        # the JSON twice (the model's own answer, then the final structured one), which
        # json.loads rejects as "Extra data" — seen live on reflect's tool-selection calls.
        text = result.response
        if schema is not None and result.structured_output is not None:
            text = json.dumps(result.structured_output)
        logger.debug(f"Antigravity: scope={scope} returned {len(text)} chars")
        return _CliTurn(
            text=text,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
        )
