from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from hindsight_api.config import PROVIDER_DEFAULT_MODELS
from hindsight_api.engine.llm_interface import LLM_TOOL_CHOICE_REQUIRED, LLMToolChoice
from hindsight_api.engine.llm_wrapper import create_llm_provider, requires_api_key
from hindsight_api.engine.providers import github_copilot_llm as provider_module
from hindsight_api.engine.providers.github_copilot_llm import GitHubCopilotLLM


class _StructuredAnswer(BaseModel):
    answer: str


class _FakeSession:
    def __init__(self, on_event, events, *, error: Exception | None = None) -> None:
        self.session_id = "copilot-session"
        self._on_event = on_event
        self._events = events
        self._error = error
        self.disconnected = False
        self.aborted = False

    async def send_and_wait(self, _prompt: str, *, timeout: float):
        if self._error is not None:
            raise self._error
        for event in self._events:
            self._on_event(event)
        assistant_events = [event for event in self._events if event.data.__class__.__name__ == "AssistantMessageData"]
        return assistant_events[-1] if assistant_events else None

    async def disconnect(self) -> None:
        self.disconnected = True

    async def abort(self) -> None:
        self.aborted = True


class _FakeClient:
    def __init__(self, events, *, error: Exception | None = None) -> None:
        self.events = events
        self.error = error
        self.create_kwargs = None
        self.create_count = 0
        self.session = None
        self.deleted_sessions: list[str] = []

    async def create_session(self, **kwargs):
        self.create_count += 1
        self.create_kwargs = kwargs
        self.session = _FakeSession(kwargs["on_event"], self.events, error=self.error)
        return self.session

    async def delete_session(self, session_id: str) -> None:
        self.deleted_sessions.append(session_id)


class _FakeRuntime:
    def __init__(self, client: _FakeClient) -> None:
        self.client = client
        self.runtime_url = ""
        self.ref_count = 1
        self.started = False
        self.invalidations: list[str] = []

    async def ensure_started(self) -> None:
        self.started = True

    async def get_client(self):
        self.started = True
        return self.client

    async def invalidate(self, _expected_client, reason: str) -> None:
        self.invalidations.append(reason)

    async def stop(self) -> None:
        self.started = False


def _assistant_event(content: str, tool_requests=None):
    from copilot.session_events import AssistantMessageData

    return SimpleNamespace(
        data=AssistantMessageData(
            content=content,
            message_id="message-1",
            tool_requests=tool_requests,
        )
    )


def _usage_event():
    from copilot.session_events import AssistantUsageData

    return SimpleNamespace(
        data=AssistantUsageData(
            model="gpt-5.6-terra",
            input_tokens=120,
            output_tokens=30,
            cache_read_tokens=20,
            reasoning_tokens=10,
            finish_reason="stop",
        )
    )


def _provider(monkeypatch, events, *, error: Exception | None = None) -> tuple[GitHubCopilotLLM, _FakeClient]:
    client = _FakeClient(events, error=error)
    runtime = _FakeRuntime(client)
    monkeypatch.setattr(provider_module, "_acquire_runtime", lambda _url: runtime)
    provider = GitHubCopilotLLM(
        provider="github-copilot",
        api_key="",
        base_url="",
        model="gpt-5.6-terra",
        timeout=15,
    )
    return provider, client


def test_provider_registration_and_default_model(monkeypatch):
    assert requires_api_key("github-copilot") is False
    assert PROVIDER_DEFAULT_MODELS["github-copilot"] == "gpt-5.6-terra"

    runtime = _FakeRuntime(_FakeClient([]))
    monkeypatch.setattr(provider_module, "_acquire_runtime", lambda _url: runtime)
    provider = create_llm_provider(
        provider="github-copilot",
        api_key="",
        base_url="",
        model="gpt-5.6-terra",
        reasoning_effort=None,
    )

    assert isinstance(provider, GitHubCopilotLLM)


def test_openai_template_base_url_is_ignored(monkeypatch):
    runtime = _FakeRuntime(_FakeClient([]))
    acquired_urls: list[str] = []

    def acquire(url: str):
        acquired_urls.append(url)
        return runtime

    monkeypatch.setattr(provider_module, "_acquire_runtime", acquire)
    provider = GitHubCopilotLLM(
        provider="github-copilot",
        api_key="",
        base_url="https://api.openai.com/v1",
        model="gpt-5.6-terra",
    )

    assert provider.base_url == ""
    assert acquired_urls == [""]


def test_isolated_copilot_home_copies_only_account_metadata(tmp_path):
    source_home = tmp_path / "source"
    target_home = tmp_path / "target"
    source_home.mkdir()
    (source_home / "config.json").write_text(
        '{"lastLoggedInUser":{"host":"https://github.com","login":"user"}}',
        encoding="utf-8",
    )
    (source_home / "hooks").mkdir()
    (source_home / "hooks" / "dangerous.json").write_text("{}", encoding="utf-8")

    provider_module._sync_copilot_auth_metadata(target_home, source_home)

    assert (target_home / "config.json").read_text(encoding="utf-8") == (source_home / "config.json").read_text(
        encoding="utf-8"
    )
    assert not (target_home / "hooks").exists()


def test_invalid_reasoning_effort_does_not_acquire_runtime(monkeypatch):
    acquire = MagicMock()
    monkeypatch.setattr(provider_module, "_acquire_runtime", acquire)

    with pytest.raises(ValueError, match="Unsupported GitHub Copilot reasoning effort"):
        GitHubCopilotLLM(
            provider="github-copilot",
            api_key="",
            base_url="",
            model="gpt-5.6-terra",
            reasoning_effort="extreme",
        )

    acquire.assert_not_called()


@pytest.mark.asyncio
async def test_plain_call_isolates_session_and_reports_usage(monkeypatch):
    provider, client = _provider(monkeypatch, [_assistant_event("ok"), _usage_event()])

    result, usage = await provider.call(
        messages=[
            {"role": "system", "content": "Reply concisely."},
            {"role": "user", "content": "Say ok."},
        ],
        return_usage=True,
        max_retries=0,
    )

    assert result == "ok"
    assert usage.input_tokens == 120
    assert usage.output_tokens == 30
    assert usage.cached_tokens == 20
    assert usage.thoughts_tokens == 10
    assert client.create_kwargs["model"] == "gpt-5.6-terra"
    assert client.create_kwargs["available_tools"] == []
    assert client.create_kwargs["system_message"] == {
        "mode": "replace",
        "content": "Reply concisely.",
    }
    assert client.create_kwargs["enable_file_hooks"] is False
    assert client.create_kwargs["enable_config_discovery"] is False
    assert client.create_kwargs["skip_custom_instructions"] is True
    assert client.create_kwargs["enable_session_store"] is False
    assert client.create_kwargs["enable_skills"] is False
    assert client.create_kwargs["memory"] == {"enabled": False}
    assert client.session.disconnected is True
    assert client.deleted_sessions == ["copilot-session"]


@pytest.mark.asyncio
async def test_structured_call_uses_terminal_schema_tool(monkeypatch):
    from copilot.session_events import AssistantMessageToolRequest

    request = AssistantMessageToolRequest(
        name="structured_response",
        tool_call_id="structured-1",
        arguments={"answer": "captured"},
    )
    provider, client = _provider(
        monkeypatch,
        [_assistant_event("", [request]), _usage_event()],
    )

    result = await provider.call(
        messages=[{"role": "user", "content": "Return an answer."}],
        response_format=_StructuredAnswer,
        max_retries=0,
    )

    assert result == _StructuredAnswer(answer="captured")
    tool = client.create_kwargs["tools"][0]
    assert tool.name == "structured_response"
    assert tool.parameters == _StructuredAnswer.model_json_schema()
    assert tool.is_terminal is True
    assert tool.skip_permission is True
    assert client.create_kwargs["available_tools"] == ["custom:structured_response"]


@pytest.mark.asyncio
async def test_named_tool_choice_exposes_only_requested_tool(monkeypatch):
    from copilot.session_events import AssistantMessageToolRequest

    request = AssistantMessageToolRequest(
        name="recall",
        tool_call_id="recall-1",
        arguments={"query": "project decisions"},
    )
    provider, client = _provider(
        monkeypatch,
        [_assistant_event("", [request]), _usage_event()],
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "recall",
                "description": "Recall memories",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "done",
                "description": "Finish",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    result = await provider.call_with_tools(
        messages=[{"role": "user", "content": "Find project decisions."}],
        tools=tools,
        tool_choice=LLMToolChoice.named("recall"),
        max_retries=0,
    )

    assert result.finish_reason == "tool_calls"
    assert result.tool_calls[0].name == "recall"
    assert result.tool_calls[0].arguments == {"query": "project decisions"}
    assert [tool.name for tool in client.create_kwargs["tools"]] == ["recall"]
    assert client.create_kwargs["available_tools"] == ["custom:recall"]
    assert "MUST call the 'recall' tool" in client.create_kwargs["system_message"]["content"]


@pytest.mark.asyncio
async def test_named_tool_choice_retries_then_rejects_prose(monkeypatch):
    provider, client = _provider(monkeypatch, [_assistant_event("I will answer directly."), _usage_event()])
    tools = [
        {
            "type": "function",
            "function": {
                "name": "recall",
                "description": "Recall memories",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    with pytest.raises(RuntimeError, match="did not call the required 'recall' tool"):
        await provider.call_with_tools(
            messages=[{"role": "user", "content": "Recall first."}],
            tools=tools,
            tool_choice=LLMToolChoice.named("recall"),
            max_retries=1,
            initial_backoff=0,
        )

    assert client.create_count == 2


@pytest.mark.asyncio
async def test_required_tool_choice_rejects_prose(monkeypatch):
    provider, _client = _provider(monkeypatch, [_assistant_event("No tool needed."), _usage_event()])
    tools = [
        {
            "type": "function",
            "function": {
                "name": "recall",
                "description": "Recall memories",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    with pytest.raises(RuntimeError, match="did not call any tool"):
        await provider.call_with_tools(
            messages=[{"role": "user", "content": "Use a tool."}],
            tools=tools,
            tool_choice=LLM_TOOL_CHOICE_REQUIRED,
            max_retries=0,
        )


@pytest.mark.asyncio
async def test_timeout_aborts_and_deletes_session(monkeypatch):
    provider, client = _provider(monkeypatch, [], error=TimeoutError("slow"))

    with pytest.raises(TimeoutError, match="slow"):
        await provider.call(
            messages=[{"role": "user", "content": "wait"}],
            max_retries=0,
        )

    assert provider._runtime.invalidations == ["slow"]
    assert client.session.disconnected is False
    assert client.deleted_sessions == []


@pytest.mark.asyncio
async def test_attempt_timeout_includes_runtime_startup(monkeypatch):
    client = _FakeClient([])

    class _SlowRuntime(_FakeRuntime):
        async def get_client(self):
            await asyncio.sleep(1)
            return self.client

    runtime = _SlowRuntime(client)
    monkeypatch.setattr(provider_module, "_acquire_runtime", lambda _url: runtime)
    provider = GitHubCopilotLLM(
        provider="github-copilot",
        api_key="",
        base_url="",
        model="gpt-5.6-terra",
        timeout=0.01,
    )

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await provider.call(
            messages=[{"role": "user", "content": "wait"}],
            max_retries=0,
        )

    assert time.monotonic() - started < 0.2


@pytest.mark.asyncio
async def test_runtime_failure_invalidates_shared_client(monkeypatch):
    provider, _client = _provider(monkeypatch, [], error=RuntimeError("Client not connected"))

    with pytest.raises(RuntimeError, match="Client not connected"):
        await provider.call(
            messages=[{"role": "user", "content": "hello"}],
            max_retries=0,
        )

    assert provider._runtime.invalidations == ["Client not connected"]


@pytest.mark.asyncio
async def test_session_cleanup_timeout_invalidates_runtime(monkeypatch):
    client = _FakeClient([_assistant_event("ok"), _usage_event()])

    class _SlowCleanupSession(_FakeSession):
        async def disconnect(self) -> None:
            await asyncio.sleep(1)

    async def create_session(**kwargs):
        client.create_count += 1
        client.create_kwargs = kwargs
        client.session = _SlowCleanupSession(kwargs["on_event"], client.events)
        return client.session

    client.create_session = create_session
    runtime = _FakeRuntime(client)
    monkeypatch.setattr(provider_module, "_acquire_runtime", lambda _url: runtime)
    provider = GitHubCopilotLLM(
        provider="github-copilot",
        api_key="",
        base_url="",
        model="gpt-5.6-terra",
        timeout=0.05,
    )

    started = time.monotonic()
    assert (
        await provider.call(
            messages=[{"role": "user", "content": "say ok"}],
            max_retries=0,
        )
        == "ok"
    )

    assert time.monotonic() - started < 0.2
    assert runtime.invalidations == ["transient session cleanup timed out or failed"]


@pytest.mark.asyncio
async def test_shared_runtime_invalidation_replaces_dead_client(monkeypatch, tmp_path):
    old_client = MagicMock()
    old_client.force_stop = AsyncMock()
    new_client = MagicMock()
    clients = iter([old_client, new_client])
    monkeypatch.setattr(
        provider_module.tempfile,
        "mkdtemp",
        lambda **_kwargs: str(tmp_path / "runtime-home"),
    )
    monkeypatch.setattr(
        provider_module._SharedCopilotRuntime,
        "_new_client",
        lambda _self: next(clients),
    )

    runtime = provider_module._SharedCopilotRuntime("")
    runtime._started = True
    await runtime.invalidate(old_client, "connection closed")

    assert runtime.client is new_client
    assert runtime._started is False
    old_client.force_stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_shared_runtime_stops_only_after_last_release(monkeypatch, tmp_path):
    with provider_module._runtime_registry_lock:
        provider_module._runtime_registry.clear()

    monkeypatch.setattr(
        provider_module.tempfile,
        "mkdtemp",
        lambda **_kwargs: str(tmp_path / "runtime-home"),
    )
    monkeypatch.setattr(
        provider_module._SharedCopilotRuntime,
        "_new_client",
        lambda _self: MagicMock(),
    )
    runtime_a = provider_module._acquire_runtime("")
    runtime_b = provider_module._acquire_runtime("")
    runtime_a.stop = AsyncMock()

    assert runtime_a is runtime_b
    assert runtime_a.ref_count == 2

    await provider_module._release_runtime(runtime_a)
    runtime_a.stop.assert_not_awaited()

    await provider_module._release_runtime(runtime_b)
    runtime_a.stop.assert_awaited_once()


def test_prompt_serializes_tool_history():
    prompt = provider_module._build_prompt(
        [
            {"role": "system", "content": "Use memory tools."},
            {"role": "user", "content": "What changed?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "recall", "arguments": '{"query":"changes"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": '{"memories":["A"]}'},
        ]
    )

    assert prompt.system_prompt == "Use memory tools."
    assert '"tool_call_id": "call-1"' in prompt.user_prompt
    assert '"name": "recall"' in prompt.user_prompt


class JsonRpcError(RuntimeError):
    """Stand-in for the SDK error class, which is matched by name."""


@pytest.mark.asyncio
async def test_configuration_error_is_terminal_and_spares_the_runtime(monkeypatch):
    """An unavailable model is rejected identically forever.

    It arrives as JsonRpcError, which _is_runtime_failure would otherwise treat
    as a dead transport — restarting the Copilot CLI once per attempt for an
    error that can never succeed.
    """
    error = JsonRpcError('Request session.create failed with message: Model "gpt-9" is not available.')
    provider, client = _provider(monkeypatch, [], error=error)

    with pytest.raises(RuntimeError, match="rejected the request configuration"):
        await provider.call(messages=[{"role": "user", "content": "hi"}], max_retries=3, initial_backoff=0)

    assert client.create_count == 1
    assert provider._runtime.invalidations == []


@pytest.mark.asyncio
async def test_configuration_error_is_terminal_for_tool_calls(monkeypatch):
    error = JsonRpcError("Model is not available.")
    provider, client = _provider(monkeypatch, [], error=error)

    with pytest.raises(RuntimeError, match="rejected the request configuration"):
        await provider.call_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "noop", "parameters": {}}}],
            max_retries=3,
            initial_backoff=0,
        )

    assert client.create_count == 1
    assert provider._runtime.invalidations == []


def test_missing_account_metadata_falls_back_to_token_credentials(monkeypatch, tmp_path):
    """A container that never ran Copilot CLI authenticates from the token env."""
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_token")
    target = tmp_path / "isolated-home"

    provider_module._sync_copilot_auth_metadata(target, source_home=tmp_path / "absent")

    assert target.is_dir()
    assert not (target / "config.json").exists()


def test_missing_account_metadata_without_credentials_raises(monkeypatch, tmp_path):
    for name in provider_module._TOKEN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="COPILOT_GITHUB_TOKEN"):
        provider_module._sync_copilot_auth_metadata(tmp_path / "isolated-home", source_home=tmp_path / "absent")


def test_transient_service_outage_is_not_mistaken_for_a_config_error():
    """ "is not available" alone must stay retryable — only the model is terminal."""
    assert provider_module._is_configuration_error(JsonRpcError("Copilot service is not available.")) is False
    assert provider_module._is_runtime_failure(JsonRpcError("Copilot service is not available.")) is True
    assert provider_module._is_configuration_error(JsonRpcError('Model "gpt-9" is not available.')) is True
