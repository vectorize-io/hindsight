"""Cursor LLM provider driving the ``cursor-agent`` CLI in headless mode.

Lets a Cursor subscription back Hindsight's internal LLM work, the same way
``claude-code``, ``github-copilot`` and ``openai-codex`` already do for theirs
(issue #4501).

Unlike those, Cursor's CLI exposes no caller-facing tool or schema surface:
``cursor-agent --help`` has ``--output-format text|json|stream-json`` and
nothing for response formats, JSON schemas, temperature or tool definitions.
So both structured output and tool calling are prompt-level emulations here —
we ask for JSON and parse it back, retrying on a parse failure, which is the
same weaker technique ``claude_code_llm.py`` uses for structured output.
"""

import asyncio
import json
import logging
import os
import pathlib
import re
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hindsight_api.engine.llm_interface import (
    LLM_TOOL_CHOICE_AUTO,
    LLMInterface,
    LLMToolChoice,
    LLMToolChoiceMode,
    ProviderContentPolicyError,
)
from hindsight_api.engine.llm_trace import LLMResponseUsage, stash_response_usage
from hindsight_api.engine.response_models import (
    LLMCallResult,
    LLMToolCall,
    LLMToolCallResult,
    TokenUsage,
)
from hindsight_api.engine.structured_output import provider_json_schema
from hindsight_api.metrics import get_metrics_collector
from hindsight_api.worker.stage import set_stage

logger = logging.getLogger(__name__)

#: Cursor's docs now call the binary ``agent``, but that name is generic enough
#: to collide with other vendors' CLIs on the same PATH (grok ships one), so the
#: unambiguous ``cursor-agent`` wins when both exist.
_BINARY_CANDIDATES = ("cursor-agent", "agent")


@dataclass(frozen=True, slots=True)
class _CursorDirs:
    """The per-process scratch directories the spawned CLI is confined to."""

    workspace: str
    config_dir: str


#: Two throwaway directories, created once per process and reused by every call.
#:
#: ``workspace`` is the agent's cwd. ``cursor-agent`` is workspace-shaped — it reads
#: files around its working directory — and Hindsight's calls are pure text
#: completions, so an empty directory keeps the server's own files out of prompts.
#:
#: ``config_dir`` is ``CURSOR_CONFIG_DIR``, the same isolation ``claude_code_llm.py``
#: gets from ``CLAUDE_CONFIG_DIR``. Without it the CLI writes a session transcript
#: into the operator's ``~/.cursor/chats`` for every extraction, consolidation and
#: reflect turn, and reads that config tree — which on a machine with the
#: `hindsight-cursor-cli` integration installed carries a global ``hooks.json``
#: pointing ``beforeSubmitPrompt``/``stop`` at Hindsight's own recall and retain.
#: Headless ``-p`` mode does not run those hooks on the CLI version tested here
#: (2026.09.15 — verified: neither a user-level nor a project-level marker hook
#: fires), so the recursive-retain loop of issue #1751 does not reproduce today.
#: We isolate anyway: the session-store pollution is real and observed, and if a
#: later CLI starts honouring hooks headlessly this is exactly the loop that bit
#: the Claude Code provider. Auth is unaffected — it resolves from the OS
#: credential store, not the config directory (verified against a logged-in CLI
#: with ``CURSOR_CONFIG_DIR`` pointed at an empty temp dir).
#:
#: Guarded by a plain lock because the accessor is reachable from more than one
#: event loop; the critical section does no I/O beyond ``mkdtemp`` and never awaits.
_dirs_lock = threading.Lock()
_dirs: _CursorDirs | None = None

#: Every tool the agent advertises, denied. This is the counterpart of
#: ``claude_code_llm.py``'s ``tools=[]`` / ``allowed_tools=[]``: Hindsight wants a text
#: completion, not an agent, and the prompts it sends are built from arbitrary
#: user-retained content, so nothing in them should be able to reach a file or a shell.
#:
#: The CLI has no flag for this — ``--mode ask`` is documented as read-only but is NOT
#: a tool switch: with no deny rules a plain ``-p --mode ask`` run happily reads a file
#: out of its working directory (verified with a canary file). Only ``cli-config.json``
#: permissions stop it, and only as explicit per-tool rules — a single ``"*"`` entry is
#: silently ignored and the read succeeds (also verified). So the list is enumerated,
#: from what the agent reports when asked to name its tools on CLI 2026.09.15.
#:
#: The enumeration is the weakness: a tool added by a later CLI is allowed until it is
#: listed here. That is survivable because the agent is already confined to an empty
#: scratch workspace and its MCP servers are unapproved (the CLI refuses an MCP call
#: without ``--approve-mcps``, which we never pass), so a new tool starts with nothing
#: interesting in reach. Re-run the "name every tool you have" probe when bumping a
#: tested CLI version.
_DENIED_TOOLS = (
    "Shell",
    "AwaitShell",
    "Read",
    "Write",
    "Delete",
    "StrReplace",
    "EditNotebook",
    "Grep",
    "Glob",
    "ReadLints",
    "WebSearch",
    "WebFetch",
    "Task",
    "TodoWrite",
    "GetDynamicTools",
    "CallDynamicTool",
    "FetchMcpResource",
    "CreateGoal",
    "UpdateGoal",
    "GenerateImage",
)


def _get_dirs() -> _CursorDirs:
    global _dirs
    with _dirs_lock:
        if _dirs is None:
            dirs = _CursorDirs(
                workspace=tempfile.mkdtemp(prefix="hindsight-cursor-ws-"),
                config_dir=tempfile.mkdtemp(prefix="hindsight-cursor-cfg-"),
            )
            config = {
                "version": 1,
                "permissions": {"allow": [], "deny": [f"{tool}(*)" for tool in _DENIED_TOOLS]},
            }
            pathlib.Path(dirs.config_dir, "cli-config.json").write_text(json.dumps(config), encoding="utf-8")
            _dirs = dirs
            logger.debug(f"Cursor: isolated workspace={dirs.workspace} CURSOR_CONFIG_DIR={dirs.config_dir}")
        return _dirs


def _resolve_binary() -> str:
    for name in _BINARY_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError(
        "cursor-agent CLI not found on PATH.\n\n"
        "To set up the Cursor provider:\n"
        "1. Install the CLI: curl https://cursor.com/install -fsS | bash\n"
        "2. Log in: cursor-agent login (or set CURSOR_API_KEY / HINDSIGHT_API_LLM_API_KEY)\n"
        "3. Verify: cursor-agent --version"
    )


#: Cursor's upstream moderation rejects a prompt with
#: ``ActionRequiredError: Request blocked We are unable to complete this request
#: because it was blocked under the model provider's usage guidelines.`` Replaying
#: the same prompt earns the same block, so this gets the permanent error type the
#: retry loops re-raise untouched — the same treatment ``claude_code_llm.py`` gives
#: an AUP refusal (#3690). Observed live: reflect spent 113s on four identical
#: blocked attempts before moving on.
_POLICY_REFUSAL_MARKER = "blocked under the model provider's usage guidelines"


class _CliUsage(BaseModel):
    """Token counts as the CLI spells them (camelCase, and absent on some turns)."""

    input_tokens: int = Field(default=0, alias="inputTokens")
    output_tokens: int = Field(default=0, alias="outputTokens")


class _CliResult(BaseModel):
    """The single ``{"type": "result", ...}`` object a headless run emits."""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["result"]
    result: str = ""
    is_error: bool = False
    subtype: str = ""
    usage: _CliUsage = Field(default_factory=_CliUsage)


@dataclass(frozen=True, slots=True)
class _CliTurn:
    """One headless ``cursor-agent`` turn: its answer and the tokens it billed."""

    text: str
    input_tokens: int
    output_tokens: int


def _cli_error(text: str) -> Exception:
    """Classify a CLI failure as permanent (content policy) or retryable."""
    if _POLICY_REFUSAL_MARKER in text.lower():
        return ProviderContentPolicyError(text)
    return RuntimeError(text)


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    """Flatten a chat transcript into the single prompt the CLI accepts.

    Headless ``cursor-agent`` takes one prompt and has no system-message or
    multi-turn channel, so roles become labelled sections.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "system":
            parts.append(str(content))
        elif role == "assistant":
            parts.append(f"[Previous assistant response: {content}]")
        elif role == "tool":
            parts.append(f"[Tool result for {msg.get('tool_call_id', '')}: {content}]")
        else:
            parts.append(str(content))
    return "\n\n".join(parts)


def _extract_json(text: str) -> Any:
    """Parse JSON out of a free-form reply, tolerating markdown fences and prose."""
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # A chatty model wraps the object in a sentence; take the outermost
        # brace/bracket span rather than failing the whole call over prose.
        start = min((i for i in (candidate.find("{"), candidate.find("[")) if i != -1), default=-1)
        end = max(candidate.rfind("}"), candidate.rfind("]"))
        if start == -1 or end <= start:
            raise
        return json.loads(candidate[start : end + 1])


class CursorLLM(LLMInterface):
    """LLM provider backed by a Cursor subscription via the ``cursor-agent`` CLI."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        reasoning_effort: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(provider, api_key, base_url, model, reasoning_effort, timeout, **kwargs)
        self._warn_reasoning_effort_unsupported()
        self._binary = _resolve_binary()
        logger.info(f"Cursor: using CLI at {self._binary} (model={self.model})")

    async def _run(self, prompt: str, scope: str) -> _CliTurn:
        """Run one headless agent turn."""
        # --mode ask is Cursor's read-only Q&A mode: no edits, no shell. That is the
        # only tool restriction the CLI offers (there is no --tools/--max-turns), so
        # it carries the weight claude_code_llm.py spreads over tools=[] and
        # max_turns=1 — Hindsight's prompts contain arbitrary retained text, and
        # --trust suppresses the approval prompt that would otherwise gate an edit.
        args = [self._binary, "-p", "--output-format", "json", "--trust", "--mode", "ask"]
        if self.model:
            args += ["--model", self.model]

        dirs = _get_dirs()
        # The key goes in the environment, never in argv: every local user can read
        # another process's command line out of `ps`, and the CLI reads CURSOR_API_KEY
        # itself, so --api-key would leak the credential for no benefit.
        env = {**os.environ, "NO_COLOR": "1", "CURSOR_CONFIG_DIR": dirs.config_dir}
        if self.api_key:
            env["CURSOR_API_KEY"] = self.api_key

        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=dirs.workspace,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(prompt.encode()), timeout=self.timeout)
        except TimeoutError as e:
            proc.kill()
            await proc.wait()
            # asyncio's TimeoutError carries no message, so the retry loop would
            # log "Cursor error (attempt 1/4): " with nothing after it. An agent
            # CLI turn routinely takes 15-30s, which is longer than the 30s
            # reflect default leaves room for, so name the knob in the error.
            raise TimeoutError(
                f"cursor-agent did not answer within {self.timeout}s. Raise "
                "HINDSIGHT_API_LLM_TIMEOUT / HINDSIGHT_API_REFLECT_LLM_TIMEOUT: an agent CLI turn is "
                "slower than a chat-completions request."
            ) from e
        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()
            raise

        if proc.returncode != 0:
            detail = (stderr.decode(errors="replace").strip() or stdout.decode(errors="replace").strip())[-2000:]
            raise _cli_error(f"cursor-agent exited with {proc.returncode}: {detail}")

        # The CLI prints unrelated diagnostics ("cursor-retrieval: tracing to ...")
        # alongside its JSON, so scan for the single `type: "result"` object rather
        # than parsing the whole stream.
        result: _CliResult | None = None
        for line in stdout.decode(errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                result = _CliResult.model_validate_json(line)
            except ValidationError:
                # A `type` other than "result" (stream-json progress events) or a
                # truncated line — both are noise, not a failure.
                continue

        if result is None:
            raise RuntimeError(f"cursor-agent produced no result object: {stdout.decode(errors='replace')[:2000]}")
        if result.is_error:
            raise _cli_error(f"cursor-agent reported an error: {result.result or result.subtype}")

        logger.debug(f"Cursor: scope={scope} returned {len(result.result)} chars")
        return _CliTurn(
            text=result.result,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
        )

    def _record(self, scope: str, start: float, input_tokens: int, output_tokens: int) -> None:
        duration = time.time() - start
        get_metrics_collector().record_llm_call(
            provider=self.provider,
            model=self.model,
            scope=scope,
            duration=duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=True,
        )
        if duration > 10.0:
            logger.info(f"slow llm call: scope={scope}, model={self.provider}/{self.model}, time={duration:.3f}s")

    async def verify_connection(self) -> None:
        try:
            await self.call(messages=[{"role": "user", "content": "test"}], scope="verification", max_retries=0)
            logger.info("Cursor connection verified successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to verify Cursor connection: {e}") from e

    async def call(
        self,
        messages: list[dict[str, str]],
        response_format: Any | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        scope: str = "memory",
        max_retries: int = 10,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        skip_validation: bool = False,
        strict_schema: bool = False,
        attempt_context: Callable[[], AbstractAsyncContextManager[None]] | None = None,
    ) -> LLMCallResult:
        start = time.time()
        prompt = _messages_to_prompt(messages)
        if response_format is not None and hasattr(response_format, "model_json_schema"):
            schema = provider_json_schema(response_format)
            prompt += (
                "\n\nYou must respond with valid JSON matching this schema:\n"
                f"{json.dumps(schema, indent=2, ensure_ascii=False)}\n\n"
                "Respond with ONLY the JSON, no markdown formatting."
            )

        last_exception: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                async with attempt_context() if attempt_context is not None else nullcontext():
                    set_stage(f"llm.cursor.{scope}.attempt={attempt + 1}/{max_retries + 1}")
                    turn = await self._run(prompt, scope)
                text, in_tok, out_tok = turn.text, turn.input_tokens, turn.output_tokens

                stash_response_usage(LLMResponseUsage(input_tokens=in_tok, output_tokens=out_tok))

                if response_format is None:
                    result: Any = text
                else:
                    try:
                        data = _extract_json(text)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Cursor JSON parse error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        if attempt < max_retries:
                            last_exception = e
                            await asyncio.sleep(min(initial_backoff * (2**attempt), max_backoff))
                            continue
                        raise
                    result = data if skip_validation else response_format.model_validate(data)

                self._record(scope, start, in_tok, out_tok)
                return LLMCallResult(
                    content=result,
                    usage=TokenUsage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=in_tok + out_tok),
                )

            except ValidationError:
                # Replaying the same prompt earns the same schema violation (#1412).
                raise
            except ProviderContentPolicyError:
                # Permanent refusal: every replay earns the same block.
                raise
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(f"Cursor error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    await asyncio.sleep(min(initial_backoff * (2**attempt), max_backoff))
                    continue
                logger.error(f"Cursor error after {max_retries + 1} attempts: {e}")
                raise

        assert last_exception is not None
        raise last_exception

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        scope: str = "tools",
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        tool_choice: LLMToolChoice = LLM_TOOL_CHOICE_AUTO,
        attempt_context: Callable[[], AbstractAsyncContextManager[None]] | None = None,
    ) -> LLMToolCallResult:
        """Emulate tool calling by asking for a JSON envelope naming the calls.

        The headless CLI gives the caller no way to declare tools or read tool
        calls back, so the forced-synthetic-tool trick ``github_copilot_llm.py``
        and ``codex_llm.py`` use is unavailable. Describing the tools in the
        prompt and parsing the model's chosen call out of JSON is what is left.
        """
        start = time.time()
        prompt = _messages_to_prompt(messages)

        if tool_choice.mode is LLMToolChoiceMode.NONE or not tools:
            usable: list[dict[str, Any]] = []
        elif tool_choice.mode is LLMToolChoiceMode.NAMED:
            forced = tool_choice.selected_function_name
            usable = [t for t in tools if t.get("function", {}).get("name") == forced]
            if not usable:
                logger.warning(f"Cursor: forced tool '{forced}' not found in available tools")
                usable = tools
        else:
            usable = tools

        if usable:
            specs = [
                {
                    "name": t.get("function", {}).get("name", ""),
                    "description": t.get("function", {}).get("description", ""),
                    "parameters": t.get("function", {}).get("parameters", {}),
                }
                for t in usable
            ]
            if tool_choice.mode is LLMToolChoiceMode.NAMED:
                requirement = f"You MUST call the '{tool_choice.selected_function_name}' tool."
            elif tool_choice.mode is LLMToolChoiceMode.REQUIRED:
                requirement = "You MUST call at least one tool; an empty tool_calls list is not allowed."
            else:
                requirement = "Call tools only if they help; otherwise return an empty tool_calls list."
            prompt += (
                "\n\nYou have access to these tools:\n"
                f"{json.dumps(specs, indent=2, ensure_ascii=False)}\n\n"
                f"{requirement}\n"
                "Respond with ONLY a JSON object of this exact shape, no markdown formatting:\n"
                '{"content": "<your text answer, or null>", '
                '"tool_calls": [{"name": "<tool name>", "arguments": {<tool arguments>}}]}'
            )
        else:
            prompt += "\n\nDo not call any tools. Answer directly with text."

        last_exception: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                async with attempt_context() if attempt_context is not None else nullcontext():
                    set_stage(f"llm.cursor.tools.attempt={attempt + 1}/{max_retries + 1}")
                    turn = await self._run(prompt, scope)
                text, in_tok, out_tok = turn.text, turn.input_tokens, turn.output_tokens
                stash_response_usage(LLMResponseUsage(input_tokens=in_tok, output_tokens=out_tok))

                tool_calls: list[LLMToolCall] = []
                content: str | None = text or None
                if usable:
                    try:
                        data = _extract_json(text)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Cursor tool JSON parse error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        if attempt < max_retries:
                            last_exception = e
                            await asyncio.sleep(min(initial_backoff * (2**attempt), max_backoff))
                            continue
                        raise
                    if not isinstance(data, dict):
                        data = {"content": text}
                    raw_content = data.get("content")
                    content = raw_content if isinstance(raw_content, str) and raw_content else None
                    valid_names = {s["name"] for s in specs}
                    for entry in data.get("tool_calls") or []:
                        if not isinstance(entry, dict) or entry.get("name") not in valid_names:
                            continue
                        arguments = entry.get("arguments")
                        tool_calls.append(
                            LLMToolCall(
                                id=f"call_{uuid.uuid4().hex[:16]}",
                                name=entry["name"],
                                arguments=arguments if isinstance(arguments, dict) else {},
                            )
                        )

                self._record(scope, start, in_tok, out_tok)
                return LLMToolCallResult(
                    content=content,
                    tool_calls=tool_calls,
                    finish_reason="tool_calls" if tool_calls else "stop",
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                )

            except ProviderContentPolicyError:
                raise
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(f"Cursor tool call error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    await asyncio.sleep(min(initial_backoff * (2**attempt), max_backoff))
                    continue
                logger.error(f"Cursor tool call error after {max_retries + 1} attempts: {e}")
                raise

        assert last_exception is not None
        raise last_exception

    async def cleanup(self) -> None:
        """Nothing to close: each call is its own short-lived subprocess."""

    def supports_attempt_scoped_concurrency(self) -> bool:
        return True
