"""
GitHub Copilot LLM provider using the official Copilot SDK.

This provider uses the GitHub identity already authenticated by Copilot CLI, or
one of the SDK-supported GitHub token environment variables. It does not use
``HINDSIGHT_API_LLM_API_KEY``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Awaitable, Mapping
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from pydantic import ValidationError

from hindsight_api.engine.llm_interface import LLM_TOOL_CHOICE_AUTO, LLMInterface, LLMToolChoice, LLMToolChoiceMode
from hindsight_api.engine.llm_trace import LLMResponseUsage, stash_response_usage
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage
from hindsight_api.engine.structured_output import provider_json_schema
from hindsight_api.metrics import get_metrics_collector
from hindsight_api.worker.stage import set_stage

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from copilot import CopilotClient
    from copilot.session_events import AssistantMessageData, SessionEvent
    from copilot.tools import Tool

_STRUCTURED_TOOL_NAME = "structured_response"
_DEFAULT_TIMEOUT_SECONDS = 120.0
_RUNTIME_CLEANUP_TIMEOUT_SECONDS = 5.0
_SESSION_CLEANUP_TIMEOUT_SECONDS = 2.0
_TEMPLATE_OPENAI_BASE_URL = "https://api.openai.com/v1"
_SUPPORTED_REASONING_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


@dataclass
class _PromptParts:
    system_prompt: str
    user_prompt: str


@dataclass
class _TurnCapture:
    assistant_messages: list[AssistantMessageData] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    thoughts_tokens: int = 0
    finish_reason: str | None = None

    def handle_event(self, event: SessionEvent) -> None:
        from copilot.session_events import AssistantMessageData, AssistantUsageData

        data = event.data
        if isinstance(data, AssistantMessageData):
            self.assistant_messages.append(data)
        elif isinstance(data, AssistantUsageData):
            self.input_tokens += data.input_tokens or 0
            self.output_tokens += data.output_tokens or 0
            self.cached_tokens += data.cache_read_tokens or 0
            self.thoughts_tokens += data.reasoning_tokens or 0
            self.finish_reason = data.finish_reason or self.finish_reason

    def token_usage(self) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.input_tokens + self.output_tokens,
            cached_tokens=self.cached_tokens,
            thoughts_tokens=self.thoughts_tokens,
        )


@dataclass
class _InvocationResult:
    content: str
    tool_calls: list[LLMToolCall]
    usage: TokenUsage
    finish_reason: str | None


_TOKEN_ENV_VARS = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")


def _has_token_credentials() -> bool:
    """Whether the SDK can authenticate without a signed-in Copilot CLI account."""
    return any(os.environ.get(name) for name in _TOKEN_ENV_VARS)


def _sync_copilot_auth_metadata(target_home: Path, source_home: Path | None = None) -> None:
    """Copy only the account selector into the isolated runtime home.

    The normal Copilot home also contains user hooks, plugins, MCP servers,
    skills, and session state. Pointing Hindsight's internal runtime there made
    every internal LLM request run the Hindsight hooks again, creating an
    explosive recursion loop. ``config.json`` carries the signed-in account
    selection, so copying only that file preserves the account choice without
    importing executable config. The credential itself is never in this file:
    the runtime resolves it from the system keychain or from a `gh` CLI login.
    """
    target_home.mkdir(parents=True, exist_ok=True)
    source = (source_home or Path.home() / ".copilot") / "config.json"
    if not source.is_file():
        if _has_token_credentials():
            # A headless deployment (container, CI) authenticates from the token
            # environment and never runs Copilot CLI, so there is no signed-in
            # account on disk to carry over. An empty home is correct there.
            logger.info(
                "No Copilot account metadata at %s; authenticating from the token environment instead",
                source,
            )
            return
        raise RuntimeError(
            f"GitHub Copilot account metadata was not found at {source}. "
            "Start Copilot CLI and sign in, or set COPILOT_GITHUB_TOKEN, GH_TOKEN, or GITHUB_TOKEN."
        )
    shutil.copy2(source, target_home / "config.json")


class _SharedCopilotRuntime:
    """One Copilot runtime shared by every Hindsight LLM lane in this process."""

    def __init__(self, runtime_url: str) -> None:
        self.runtime_url = runtime_url
        self.ref_count = 0
        self._started = False
        self._lifecycle_lock = asyncio.Lock()
        self._copilot_home = None if runtime_url else Path(tempfile.mkdtemp(prefix="hindsight-github-copilot-"))
        self.client = self._new_client()

    def _new_client(self) -> CopilotClient:
        from copilot import CopilotClient, RuntimeConnection

        connection = RuntimeConnection.for_uri(self.runtime_url) if self.runtime_url else None
        base_directory = None
        if self._copilot_home is not None:
            _sync_copilot_auth_metadata(self._copilot_home)
            base_directory = str(self._copilot_home)
        return CopilotClient(
            connection=connection,
            working_directory=tempfile.gettempdir(),
            base_directory=base_directory,
            use_logged_in_user=True,
            session_idle_timeout_seconds=300,
            mode="copilot-cli",
        )

    async def ensure_started(self) -> None:
        if self._started:
            return
        async with self._lifecycle_lock:
            if self._started:
                return
            client = self.client
            try:
                await client.start()
                auth = await client.get_auth_status()
                if not auth.isAuthenticated:
                    raise RuntimeError(
                        "GitHub Copilot is not authenticated. Start Copilot CLI and sign in, "
                        "or set COPILOT_GITHUB_TOKEN, GH_TOKEN, or GITHUB_TOKEN."
                    )
                self._started = True
            except BaseException:
                self.client = self._new_client()
                self._started = False
                await self._force_stop_client(client)
                raise

    async def get_client(self) -> CopilotClient:
        await self.ensure_started()
        return self.client

    async def invalidate(self, expected_client: CopilotClient, reason: str) -> None:
        """Replace a dead or wedged runtime without disrupting a newer generation."""
        async with self._lifecycle_lock:
            if self.client is not expected_client:
                return
            logger.warning("Resetting shared GitHub Copilot runtime: %s", reason)
            self.client = self._new_client()
            self._started = False
            await self._force_stop_client(expected_client)

    @staticmethod
    async def _force_stop_client(client: CopilotClient) -> None:
        try:
            await asyncio.wait_for(client.force_stop(), timeout=_RUNTIME_CLEANUP_TIMEOUT_SECONDS)
        except Exception:
            logger.warning("Failed to force-stop GitHub Copilot runtime", exc_info=True)

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            if self._started:
                client = self.client
                self._started = False
                try:
                    await asyncio.wait_for(client.stop(), timeout=_RUNTIME_CLEANUP_TIMEOUT_SECONDS)
                except Exception:
                    logger.warning("GitHub Copilot runtime cleanup failed; forcing shutdown", exc_info=True)
                    await self._force_stop_client(client)
            if self._copilot_home is not None:
                try:
                    shutil.rmtree(self._copilot_home)
                except OSError:
                    logger.warning(
                        "Failed to remove isolated Copilot home %s",
                        self._copilot_home,
                        exc_info=True,
                    )


_runtime_registry_lock = threading.Lock()
_runtime_registry: dict[str, _SharedCopilotRuntime] = {}


def _acquire_runtime(runtime_url: str) -> _SharedCopilotRuntime:
    key = runtime_url.strip().rstrip("/")
    with _runtime_registry_lock:
        runtime = _runtime_registry.get(key)
        if runtime is None:
            runtime = _SharedCopilotRuntime(key)
            _runtime_registry[key] = runtime
        runtime.ref_count += 1
        return runtime


def _normalize_runtime_url(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    # Older generated .env files actively set the OpenAI default URL. Treat
    # that one known template value as unset so switching only provider/model
    # works; every other non-empty value remains explicit and is validated by
    # the Copilot SDK as a headless runtime URI.
    if value == _TEMPLATE_OPENAI_BASE_URL.rstrip("/"):
        logger.info("Ignoring the OpenAI template base URL for the github-copilot provider")
        return ""
    return value


async def _release_runtime(runtime: _SharedCopilotRuntime) -> None:
    should_stop = False
    with _runtime_registry_lock:
        if runtime.ref_count > 0:
            runtime.ref_count -= 1
        if runtime.ref_count == 0 and _runtime_registry.get(runtime.runtime_url) is runtime:
            del _runtime_registry[runtime.runtime_url]
            should_stop = True
    if should_stop:
        await runtime.stop()


def _build_prompt(messages: list[dict[str, Any]]) -> _PromptParts:
    system_parts: list[str] = []
    conversation: list[dict[str, Any]] = []

    for message in messages:
        role = str(message.get("role", "user"))
        if role == "system":
            content = message.get("content")
            if isinstance(content, str) and content:
                system_parts.append(content)
            continue
        conversation.append(message)

    system_prompt = "\n\n".join(system_parts).strip()
    if not system_prompt:
        system_prompt = "Act as a stateless language-model backend and follow the supplied conversation."

    if (
        len(conversation) == 1
        and conversation[0].get("role") == "user"
        and isinstance(conversation[0].get("content"), str)
        and set(conversation[0]).issubset({"role", "content"})
    ):
        user_prompt = conversation[0]["content"]
    else:
        serialized = json.dumps(conversation, ensure_ascii=False, default=str)
        user_prompt = (
            "Continue the conversation represented by the JSON messages below. "
            "Produce only the next assistant turn.\n\n"
            f"<conversation_messages>{serialized}</conversation_messages>"
        )

    return _PromptParts(system_prompt=system_prompt, user_prompt=user_prompt)


def _normalize_tool_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, Mapping):
        return dict(arguments)
    if isinstance(arguments, str):
        parsed = json.loads(arguments)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"GitHub Copilot returned non-object tool arguments: {type(arguments).__name__}")


def _is_authentication_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "not authenticated",
            "authentication failed",
            "unauthorized",
            "sign in",
            "login required",
            "invalid token",
        )
    )


def _is_quota_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in ("quota", "usage limit", "weekly limit", "ai credits"))


def _is_configuration_error(error: BaseException) -> bool:
    """Whether the runtime will reject this request identically on every attempt.

    These surface as JsonRpcError, which is otherwise indistinguishable from a
    transport failure. Treating one as a runtime failure costs a full Copilot
    CLI restart per attempt and can never succeed, so they are terminal.
    """
    text = str(error).lower()
    # Gate on "model" so a transient outage phrased as "service is not
    # available" stays retryable; only the model selection is terminal.
    if "model" not in text:
        return False
    return any(
        marker in text
        for marker in (
            "is not available",
            "unknown model",
            "not found",
            "invalid model",
            "unsupported model",
        )
    )


def _is_runtime_failure(error: BaseException) -> bool:
    if _is_configuration_error(error):
        return False
    if isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return True
    if type(error).__name__ in {"ProcessExitedError", "JsonRpcError"}:
        return True
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "client not connected",
            "process exited",
            "connection closed",
            "connection lost",
            "broken pipe",
            "transport closed",
            "unexpected eof",
        )
    )


class GitHubCopilotLLM(LLMInterface):
    """LLM provider backed by the authenticated GitHub Copilot runtime."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        reasoning_effort: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(provider, api_key, base_url, model, reasoning_effort, timeout=timeout, **kwargs)
        self._released = False

        if self.reasoning_effort == "none":
            self.reasoning_effort = None
        elif self.reasoning_effort is not None and self.reasoning_effort not in _SUPPORTED_REASONING_EFFORTS:
            raise ValueError(
                f"Unsupported GitHub Copilot reasoning effort {self.reasoning_effort!r}. "
                f"Use one of: {', '.join(sorted(_SUPPORTED_REASONING_EFFORTS))}, or leave it unset."
            )
        self.base_url = _normalize_runtime_url(base_url)
        self._runtime = _acquire_runtime(self.base_url)

    async def verify_connection(self) -> None:
        try:
            await self._runtime.ensure_started()
            await self.call(
                messages=[{"role": "user", "content": "Reply with exactly: ok"}],
                max_completion_tokens=10,
                scope="verification",
                max_retries=0,
            )
            logger.info("GitHub Copilot connection verified successfully")
        except Exception as error:
            raise RuntimeError(f"GitHub Copilot connection verification failed: {error}") from error

    async def _invoke(
        self,
        messages: list[dict[str, Any]],
        sdk_tools: list[Any],
        available_tools: list[str],
        system_suffix: str = "",
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> _InvocationResult:
        prompt = _build_prompt(messages)
        system_prompt = prompt.system_prompt
        if system_suffix:
            system_prompt = f"{system_prompt}\n\n{system_suffix}"

        capture = _TurnCapture()
        session = None
        session_id: str | None = None
        client = None
        runtime_invalidated = False
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds

        async def wait_with_deadline(factory: Callable[[], Awaitable[Any]]) -> Any:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f"GitHub Copilot attempt exceeded {timeout_seconds:g}s")
            return await asyncio.wait_for(factory(), timeout=remaining)

        try:
            client = await wait_with_deadline(self._runtime.get_client)
            session = await wait_with_deadline(
                lambda: client.create_session(
                    model=self.model,
                    reasoning_effort=self.reasoning_effort,
                    tools=sdk_tools,
                    available_tools=available_tools,
                    system_message={"mode": "replace", "content": system_prompt},
                    enable_experimental_mode=False,
                    enable_session_telemetry=False,
                    enable_file_change_tracking=False,
                    skip_custom_instructions=True,
                    custom_agents_local_only=True,
                    coauthor_enabled=False,
                    manage_schedule_enabled=False,
                    streaming=False,
                    include_sub_agent_streaming_events=False,
                    mcp_oauth_token_storage="in-memory",
                    embedding_cache_storage="in-memory",
                    enable_config_discovery=False,
                    skip_embedding_retrieval=True,
                    enable_on_demand_instruction_discovery=False,
                    enable_file_hooks=False,
                    enable_host_git_operations=False,
                    enable_session_store=False,
                    enable_skills=False,
                    memory={"enabled": False},
                    on_event=capture.handle_event,
                )
            )
            session_id = session.session_id
            response = await wait_with_deadline(
                lambda: session.send_and_wait(
                    prompt.user_prompt,
                    timeout=max(deadline - loop.time(), 0.001),
                )
            )

            if response is not None:
                from copilot.session_events import AssistantMessageData

                if isinstance(response.data, AssistantMessageData) and response.data not in capture.assistant_messages:
                    capture.assistant_messages.append(response.data)

            tool_calls: list[LLMToolCall] = []
            seen_call_ids: set[str] = set()
            content = ""
            for assistant_message in capture.assistant_messages:
                if assistant_message.content:
                    content = assistant_message.content
                for request in assistant_message.tool_requests or []:
                    if request.tool_call_id in seen_call_ids:
                        continue
                    seen_call_ids.add(request.tool_call_id)
                    tool_calls.append(
                        LLMToolCall(
                            id=request.tool_call_id,
                            name=request.name,
                            arguments=_normalize_tool_arguments(request.arguments),
                        )
                    )

            return _InvocationResult(
                content=content,
                tool_calls=tool_calls,
                usage=capture.token_usage(),
                finish_reason=capture.finish_reason,
            )
        except asyncio.CancelledError:
            if client is not None:
                await self._runtime.invalidate(client, "attempt was cancelled")
                runtime_invalidated = True
            raise
        except Exception as error:
            if client is not None and _is_runtime_failure(error):
                await self._runtime.invalidate(client, str(error))
                runtime_invalidated = True
            raise
        finally:
            if session is not None and client is not None and not runtime_invalidated:
                cleanup_ok = await self._cleanup_session(
                    client=client,
                    session=session,
                    session_id=session_id,
                    deadline=deadline,
                )
                if not cleanup_ok:
                    await self._runtime.invalidate(client, "transient session cleanup timed out or failed")

    def _timeout_seconds(self) -> float:
        return self.timeout if self.timeout is not None else _DEFAULT_TIMEOUT_SECONDS

    @staticmethod
    async def _cleanup_session(
        client: CopilotClient,
        session: Any,
        session_id: str | None,
        deadline: float,
    ) -> bool:
        loop = asyncio.get_running_loop()

        async def cleanup_step(factory: Callable[[], Awaitable[Any]]) -> None:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError("GitHub Copilot attempt deadline expired before session cleanup")
            await asyncio.wait_for(
                factory(),
                timeout=min(remaining, _SESSION_CLEANUP_TIMEOUT_SECONDS),
            )

        try:
            await cleanup_step(session.disconnect)
            if session_id is not None:
                await cleanup_step(lambda: client.delete_session(session_id))
            return True
        except Exception:
            logger.warning("Failed to clean up transient Copilot session %s", session_id, exc_info=True)
            return False

    @staticmethod
    def _terminal_tool(name: str, description: str, parameters: dict[str, Any]) -> Tool:
        from copilot.tools import Tool, ToolInvocation, ToolResult

        async def capture_tool(_invocation: ToolInvocation) -> ToolResult:
            # The assistant.message event carries the full tool request. Marking
            # this tool terminal prevents Copilot's own agent loop from feeding
            # this placeholder result back to the model; Hindsight executes the
            # real tool in its outer reflect loop.
            return ToolResult(text_result_for_llm="Tool call captured.", result_type="success")

        return Tool(
            name=name,
            description=description,
            parameters=parameters,
            handler=capture_tool,
            skip_permission=True,
            defer="never",
            is_terminal=True,
        )

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
        return_usage: bool = False,
        cached_prefix: str | None = None,
        attempt_context: Callable[[], AbstractAsyncContextManager[None]] | None = None,
    ) -> Any:
        start_time = time.time()

        for attempt in range(max_retries + 1):
            try:
                sdk_tools: list[Any] = []
                available_tools: list[str] = []
                system_suffix = ""

                if response_format is not None:
                    schema = provider_json_schema(response_format)
                    sdk_tools = [
                        self._terminal_tool(
                            _STRUCTURED_TOOL_NAME,
                            "Return the structured response matching the required schema.",
                            schema,
                        )
                    ]
                    available_tools = [f"custom:{_STRUCTURED_TOOL_NAME}"]
                    system_suffix = (
                        f"You MUST call the {_STRUCTURED_TOOL_NAME!r} tool exactly once. Do not answer with prose."
                    )

                async with attempt_context() if attempt_context is not None else nullcontext():
                    set_stage(f"llm.github_copilot.{scope}.attempt={attempt + 1}/{max_retries + 1}")
                    invocation = await self._invoke(
                        messages=list(messages),
                        sdk_tools=sdk_tools,
                        available_tools=available_tools,
                        system_suffix=system_suffix,
                        timeout_seconds=self._timeout_seconds(),
                    )

                stash_response_usage(
                    LLMResponseUsage(
                        input_tokens=invocation.usage.input_tokens,
                        output_tokens=invocation.usage.output_tokens,
                        cached_tokens=invocation.usage.cached_tokens,
                    )
                )

                if response_format is not None:
                    structured_call = next(
                        (call for call in invocation.tool_calls if call.name == _STRUCTURED_TOOL_NAME),
                        None,
                    )
                    if structured_call is None:
                        raise RuntimeError("GitHub Copilot did not return the required structured_response tool call")
                    result = (
                        structured_call.arguments
                        if skip_validation
                        else response_format.model_validate(structured_call.arguments)
                    )
                else:
                    if not invocation.content:
                        raise RuntimeError("GitHub Copilot returned an empty response")
                    result = invocation.content

                duration = time.time() - start_time
                self._record_success(
                    messages=messages,
                    result=result,
                    invocation=invocation,
                    scope=scope,
                    duration=duration,
                )
                if return_usage:
                    return result, invocation.usage
                return result
            except ValidationError:
                raise
            except Exception as error:
                if _is_configuration_error(error):
                    raise RuntimeError(f"GitHub Copilot rejected the request configuration: {error}") from error
                if _is_authentication_error(error):
                    raise RuntimeError(
                        "GitHub Copilot authentication failed. Start Copilot CLI and sign in, "
                        "or set COPILOT_GITHUB_TOKEN, GH_TOKEN, or GITHUB_TOKEN. "
                        "HINDSIGHT_API_LLM_API_KEY is not used by this provider."
                    ) from error
                if _is_quota_error(error):
                    raise RuntimeError(f"GitHub Copilot usage limit reached: {error}") from error
                if attempt >= max_retries:
                    raise
                backoff = min(initial_backoff * (2**attempt), max_backoff)
                logger.warning(
                    "GitHub Copilot error (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries + 1,
                    error,
                )
                await asyncio.sleep(backoff)

        raise RuntimeError("GitHub Copilot call failed after all retries")

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
        cached_prefix: str | None = None,
        cached_prefix_message_count: int = 0,
        attempt_context: Callable[[], AbstractAsyncContextManager[None]] | None = None,
    ) -> LLMToolCallResult:
        start_time = time.time()

        for attempt in range(max_retries + 1):
            try:
                selected_tools = tools
                system_suffix = ""
                if tool_choice.mode is LLMToolChoiceMode.NONE:
                    selected_tools = []
                elif tool_choice.mode is LLMToolChoiceMode.NAMED:
                    selected_name = tool_choice.selected_function_name
                    selected_tools = [tool for tool in tools if tool.get("function", {}).get("name") == selected_name]
                    if not selected_tools:
                        raise ValueError(f"Requested GitHub Copilot tool {selected_name!r} is not available")
                    system_suffix = f"You MUST call the {selected_name!r} tool. Do not answer with prose."
                elif tool_choice.mode is LLMToolChoiceMode.REQUIRED:
                    system_suffix = "You MUST call at least one available tool. Do not answer with prose."

                sdk_tools = []
                available_tools = []
                for tool in selected_tools:
                    function = tool.get("function", {})
                    name = function.get("name", "")
                    if not name:
                        continue
                    sdk_tools.append(
                        self._terminal_tool(
                            name=name,
                            description=function.get("description", ""),
                            parameters=function.get("parameters", {}),
                        )
                    )
                    available_tools.append(f"custom:{name}")

                async with attempt_context() if attempt_context is not None else nullcontext():
                    set_stage(f"llm.github_copilot.tools.attempt={attempt + 1}/{max_retries + 1}")
                    invocation = await self._invoke(
                        messages=messages,
                        sdk_tools=sdk_tools,
                        available_tools=available_tools,
                        system_suffix=system_suffix,
                        timeout_seconds=self._timeout_seconds(),
                    )

                stash_response_usage(
                    LLMResponseUsage(
                        input_tokens=invocation.usage.input_tokens,
                        output_tokens=invocation.usage.output_tokens,
                        cached_tokens=invocation.usage.cached_tokens,
                    )
                )

                if tool_choice.mode is LLMToolChoiceMode.NAMED:
                    selected_name = tool_choice.selected_function_name
                    if not any(call.name == selected_name for call in invocation.tool_calls):
                        raise RuntimeError(f"GitHub Copilot did not call the required {selected_name!r} tool")
                elif tool_choice.mode is LLMToolChoiceMode.REQUIRED and not invocation.tool_calls:
                    raise RuntimeError("GitHub Copilot did not call any tool when a tool call was required")

                duration = time.time() - start_time
                result = LLMToolCallResult(
                    content=invocation.content or None,
                    tool_calls=invocation.tool_calls,
                    finish_reason="tool_calls" if invocation.tool_calls else invocation.finish_reason or "stop",
                    input_tokens=invocation.usage.input_tokens,
                    output_tokens=invocation.usage.output_tokens,
                    cached_tokens=invocation.usage.cached_tokens,
                    thoughts_tokens=invocation.usage.thoughts_tokens,
                )
                self._record_success(
                    messages=messages,
                    result=result,
                    invocation=invocation,
                    scope=scope,
                    duration=duration,
                )
                return result
            except Exception as error:
                if _is_configuration_error(error):
                    raise RuntimeError(f"GitHub Copilot rejected the request configuration: {error}") from error
                if _is_authentication_error(error):
                    raise RuntimeError(
                        "GitHub Copilot authentication failed. Start Copilot CLI and sign in, "
                        "or set COPILOT_GITHUB_TOKEN, GH_TOKEN, or GITHUB_TOKEN."
                    ) from error
                if _is_quota_error(error):
                    raise RuntimeError(f"GitHub Copilot usage limit reached: {error}") from error
                if attempt >= max_retries:
                    raise
                backoff = min(initial_backoff * (2**attempt), max_backoff)
                logger.warning(
                    "GitHub Copilot tool-call error (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries + 1,
                    error,
                )
                await asyncio.sleep(backoff)

        raise RuntimeError("GitHub Copilot tool call failed after all retries")

    def _record_success(
        self,
        messages: list[dict[str, Any]],
        result: Any,
        invocation: _InvocationResult,
        scope: str,
        duration: float,
    ) -> None:
        usage = invocation.usage
        metrics = get_metrics_collector()
        metrics.record_llm_call(
            provider=self.provider,
            model=self.model,
            scope=scope,
            duration=duration,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=usage.cached_tokens,
            thoughts_tokens=usage.thoughts_tokens,
            success=True,
        )

        try:
            from hindsight_api.tracing import _serialize_for_span, get_span_recorder

            get_span_recorder().record_llm_call(
                provider=self.provider,
                model=self.model,
                scope=scope,
                messages=messages,
                response_content=_serialize_for_span(result),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_tokens=usage.cached_tokens,
                duration=duration,
                finish_reason=invocation.finish_reason,
                error=None,
                tool_calls=[
                    {"id": call.id, "name": call.name, "arguments": call.arguments} for call in invocation.tool_calls
                ]
                or None,
            )
        except Exception:
            logger.debug("GitHub Copilot span recording failed", exc_info=True)

        if duration > 10.0:
            logger.info(
                "slow llm call: scope=%s, model=%s/%s, time=%.3fs",
                scope,
                self.provider,
                self.model,
                duration,
            )

    async def cleanup(self) -> None:
        if self._released:
            return
        self._released = True
        await _release_runtime(self._runtime)

    def supports_attempt_scoped_concurrency(self) -> bool:
        return True
