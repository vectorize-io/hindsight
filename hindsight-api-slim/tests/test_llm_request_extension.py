"""
Tests for the LLM request extension: per-call credentials chosen server-side.

Covers what the extension is asked (bank, schema, operation), that the chosen key
and headers reach the wire for each supporting provider, that an unsupporting
provider refuses instead of falling back to the server key, and that the key never
lands in a log line. The end-to-end story (every operation, API and worker) lives
in hindsight-system-tests.

All deterministic — no network: the SDK clients run over httpx.MockTransport.
"""

import json
import logging

import httpx
import pytest
from aiohttp import web
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from hindsight_api.config import LLM_STRATEGY_FAILOVER, LLMMemberConfig, LLMStrategyConfig
from hindsight_api.engine.llm_interface import PromptCachePrefix
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.multi_llm import MultiLLMProvider
from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM
from hindsight_api.engine.providers.llm_debug import _serialize_config
from hindsight_api.extensions.llm_request import (
    LLMRequestExtension,
    LLMRequestOverrides,
    LLMRequestRefused,
    LLMRequestTarget,
    bind_llm_request,
    install_llm_request_extension,
)
from hindsight_api.schema_context import _current_schema
from tests.aiohttp_stub import stub_server


class RecordingExtension(LLMRequestExtension):
    """Answers with a fixed result and records every target it was asked about."""

    def __init__(self, answer: LLMRequestOverrides | None) -> None:
        super().__init__({})
        self.answer = answer
        self.targets: list[LLMRequestTarget] = []

    async def resolve(self, target: LLMRequestTarget) -> LLMRequestOverrides | None:
        self.targets.append(target)
        return self.answer


@pytest.fixture
def install():
    def _install(answer: LLMRequestOverrides | None) -> RecordingExtension:
        ext = RecordingExtension(answer)
        install_llm_request_extension(ext)
        return ext

    yield _install
    install_llm_request_extension(None)


def _openai_provider(seen: list[httpx.Request]) -> LLMProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    llm = LLMProvider(provider="openai", api_key="server-key", base_url="http://gateway.test/v1", model="m")
    llm._provider_impl._client = AsyncOpenAI(
        api_key="server-key",
        base_url="http://gateway.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return llm


def _anthropic_provider(seen: list[httpx.Request]) -> LLMProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "model": "m",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    llm = LLMProvider(provider="anthropic", api_key="server-key", base_url="", model="m")
    llm._provider_impl._client = AsyncAnthropic(
        api_key="server-key",
        max_retries=0,
        timeout=30.0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return llm


def _bank_config() -> object:
    class _Config:
        llm_gemini_safety_settings = None

    return _Config()


async def _call_for_bank(llm: LLMProvider, bank_id: str, operation: str = "retain") -> None:
    await _call_for_bank_result(llm, bank_id, operation)


async def _call_for_bank_result(llm: LLMProvider, bank_id: str, operation: str = "retain"):
    configured = llm.with_config(_bank_config(), bank_id=bank_id, operation=operation)
    return await configured.call(messages=[{"role": "user", "content": "hi"}], max_retries=0)


# ── what the extension is asked ───────────────────────────────────────────────


async def test_extension_is_asked_with_bank_schema_and_operation(install):
    ext = install(None)
    token = _current_schema.set("tenant_a")
    try:
        await _call_for_bank(_openai_provider([]), "alice", operation="consolidation")
    finally:
        _current_schema.reset(token)

    assert ext.targets == [
        LLMRequestTarget(schema="tenant_a", bank_id="alice", operation="consolidation", provider="openai", model="m")
    ]


async def test_no_extension_means_server_key_and_no_lookup():
    seen: list[httpx.Request] = []
    await _call_for_bank(_openai_provider(seen), "alice")
    assert seen[0].headers["authorization"] == "Bearer server-key"


# ── the chosen credentials reach the wire ─────────────────────────────────────


async def test_openai_compatible_sends_the_bank_key_and_headers(install):
    install(LLMRequestOverrides(api_key="alice-key", headers={"x-gateway-customer": "alice"}))
    seen: list[httpx.Request] = []
    await _call_for_bank(_openai_provider(seen), "alice")

    assert seen[0].headers["authorization"] == "Bearer alice-key"
    assert seen[0].headers["x-gateway-customer"] == "alice"
    # The key rides the client, not the body: a body dump cannot carry it.
    assert "alice-key" not in seen[0].content.decode()


async def test_none_answer_keeps_the_server_key(install):
    install(None)
    seen: list[httpx.Request] = []
    await _call_for_bank(_openai_provider(seen), "alice")
    assert seen[0].headers["authorization"] == "Bearer server-key"


async def test_anthropic_sends_the_bank_key(install):
    install(LLMRequestOverrides(api_key="alice-key"))
    seen: list[httpx.Request] = []
    await _call_for_bank(_anthropic_provider(seen), "alice")
    assert seen[0].headers["x-api-key"] == "alice-key"


async def test_override_does_not_leak_into_the_next_call(install):
    ext = install(LLMRequestOverrides(api_key="alice-key"))
    seen: list[httpx.Request] = []
    llm = _openai_provider(seen)
    await _call_for_bank(llm, "alice")
    ext.answer = None
    await _call_for_bank(llm, "bob")
    assert [r.headers["authorization"] for r in seen] == ["Bearer alice-key", "Bearer server-key"]


async def test_litellm_puts_the_key_in_its_call_kwargs(install):
    install(LLMRequestOverrides(api_key="alice-key", headers={"x-customer": "alice"}))
    llm = LLMProvider(provider="litellm", api_key="server-key", base_url="", model="openai/m")
    impl = llm._provider_impl
    assert isinstance(impl, LiteLLMLLM)
    bound = await bind_llm_request(llm)
    try:
        kwargs = impl._build_common_kwargs(messages=[{"role": "user", "content": "hi"}])
    finally:
        bound.reset()
    assert kwargs["api_key"] == "alice-key"
    assert kwargs["extra_headers"]["x-customer"] == "alice"
    # And it is gone once the call's binding is.
    assert impl._build_common_kwargs(messages=[])["api_key"] == "server-key"


# ── a whole different LLM ─────────────────────────────────────────────────────


def _chat_completion(model: str) -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


async def test_answer_can_send_the_call_to_another_llm(install):
    """Provider, endpoint, model and key all come from the answer; the server's LLM
    receives nothing."""
    received: list[web.Request] = []
    bodies: list[dict] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        received.append(request)
        bodies.append(await request.json())
        return web.json_response(_chat_completion("bank-model"))

    server_seen: list[httpx.Request] = []
    async with stub_server(handler) as base_url:
        install(
            LLMRequestOverrides(
                llm=LLMMemberConfig(
                    provider="openai", api_key="bank-key", model="bank-model", base_url=f"{base_url}/v1"
                )
            )
        )
        result = await _call_for_bank_result(_openai_provider(server_seen), "alice")

    assert result.content == "ok"
    assert server_seen == []
    assert [r.headers["Authorization"] for r in received] == ["Bearer bank-key"]
    assert bodies[0]["model"] == "bank-model"


async def test_tool_calls_go_to_the_chosen_llm_too(install):
    received: list[str] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        received.append((await request.json())["model"])
        return web.json_response(_chat_completion("bank-model"))

    server_seen: list[httpx.Request] = []
    async with stub_server(handler) as base_url:
        install(
            LLMRequestOverrides(
                llm=LLMMemberConfig(
                    provider="openai", api_key="bank-key", model="bank-model", base_url=f"{base_url}/v1"
                )
            )
        )
        configured = _openai_provider(server_seen).with_config(_bank_config(), bank_id="alice", operation="reflect")
        await configured.call_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "done", "parameters": {"type": "object"}}}],
            max_retries=0,
        )

    assert received == ["bank-model"]
    assert server_seen == []


async def test_the_chosen_llm_is_built_once_and_reused(install):
    from hindsight_api.extensions import llm_request

    member = LLMMemberConfig(provider="mock", api_key=None, model="bank-model")
    install(LLMRequestOverrides(llm=member))
    llm = LLMProvider(provider="mock", api_key="", base_url="", model="m")

    first = await bind_llm_request(llm)
    second = await bind_llm_request(llm)
    first.reset()
    second.reset()

    assert first.llm is not None and first.llm is second.llm
    assert first.llm.model == "bank-model"
    assert len(llm_request._chosen_llms) <= llm_request._MAX_CHOSEN_LLMS


async def test_the_chosen_llm_cache_is_bounded(install, monkeypatch):
    from hindsight_api.extensions import llm_request

    monkeypatch.setattr(llm_request, "_MAX_CHOSEN_LLMS", 2)
    monkeypatch.setattr(llm_request, "_chosen_llms", type(llm_request._chosen_llms)())
    llm = LLMProvider(provider="mock", api_key="", base_url="", model="m")
    ext = install(None)
    for i in range(5):
        ext.answer = LLMRequestOverrides(llm=LLMMemberConfig(provider="mock", api_key=None, model=f"m{i}"))
        (await bind_llm_request(llm)).reset()

    assert [built.model for built in llm_request._chosen_llms.values()] == ["m3", "m4"]


async def test_credentials_for_a_chosen_llm_that_cannot_take_them_are_refused(install):
    install(LLMRequestOverrides(llm=LLMMemberConfig(provider="mock", api_key="own", model="x"), api_key="k"))
    with pytest.raises(LLMRequestRefused, match="provider 'mock'"):
        await _call_for_bank(_openai_provider([]), "alice")


def test_answer_rejects_an_llm_that_is_not_a_member_config():
    with pytest.raises(ValueError, match="LLMMemberConfig"):
        LLMRequestOverrides(llm={"provider": "openai"})  # type: ignore[arg-type]


def test_overrides_repr_hides_the_chosen_llm():
    member = LLMMemberConfig(provider="openai", api_key="sk-secret", model="m")
    assert "sk-secret" not in repr(LLMRequestOverrides(llm=member))


async def test_a_chosen_llm_without_a_key_takes_the_answers_key(install):
    received: list[str] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        received.append(request.headers["Authorization"])
        return web.json_response(_chat_completion("bank-model"))

    async with stub_server(handler) as base_url:
        member = LLMMemberConfig(provider="openai", api_key=None, model="bank-model", base_url=f"{base_url}/v1")
        install(LLMRequestOverrides(llm=member, api_key="bank-key"))
        await _call_for_bank(_openai_provider([]), "alice")

    assert received == ["Bearer bank-key"]


async def test_the_answers_key_fills_a_chosen_llm_that_cannot_take_per_call_keys(install):
    # The key is built into the chosen LLM, not sent per call, so nothing is refused.
    install(LLMRequestOverrides(llm=LLMMemberConfig(provider="mock", api_key=None, model="keyed"), api_key="k"))
    bound = await bind_llm_request(LLMProvider(provider="mock", api_key="", base_url="", model="m"))
    bound.reset()
    assert bound.llm is not None and bound.llm.api_key == "k"


async def test_a_chosen_llm_inherits_none_of_the_servers_credentials(install, monkeypatch):
    import dataclasses

    from hindsight_api import config as config_module

    server_config = dataclasses.replace(
        config_module._get_raw_config(),
        llm_default_headers={"x-portkey-api-key": "server-gateway-key"},
        llm_vertexai_service_account_key="/server/sa.json",
        llm_codex_home="/server/codex",
    )
    monkeypatch.setattr(config_module, "_get_raw_config", lambda: server_config)
    from hindsight_api.engine import memory_engine

    built_from = []
    build = memory_engine._member_to_llm
    monkeypatch.setattr(memory_engine, "_member_to_llm", lambda m, c, d: built_from.append(c) or build(m, c, d))
    install(LLMRequestOverrides(llm=LLMMemberConfig(provider="mock", api_key=None, model="inherits-nothing")))

    bound = await bind_llm_request(LLMProvider(provider="mock", api_key="", base_url="", model="m"))
    bound.reset()

    assert bound.llm is not None and not bound.llm.default_headers and not bound.llm.codex_home
    (config,) = built_from
    assert config.llm_vertexai_service_account_key is None
    assert config.llm_litellmrouter_config is None


async def test_prompt_cache_goes_to_the_chosen_llm(install):
    """The prefix says what is cacheable; the chosen LLM builds its own cache from it."""
    received: list[str] = []

    async def handler(request: web.Request) -> web.StreamResponse:
        received.append((await request.json())["model"])
        return web.json_response(_chat_completion("bank-model"))

    server_seen: list[httpx.Request] = []
    async with stub_server(handler) as base_url:
        member = LLMMemberConfig(provider="openai", api_key="bank-key", model="bank-model", base_url=f"{base_url}/v1")
        install(LLMRequestOverrides(llm=member))
        configured = _openai_provider(server_seen).with_config(_bank_config(), bank_id="alice", operation="retain")
        result = await configured.call(
            messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            prompt_cache=PromptCachePrefix(system_instruction="sys"),
            max_retries=0,
        )

    assert result.content == "ok"
    assert received == ["bank-model"]
    assert server_seen == []


@pytest.mark.parametrize("make_provider", [_openai_provider, _anthropic_provider], ids=["openai", "anthropic"])
async def test_tool_calls_send_the_bank_key(install, make_provider):
    install(LLMRequestOverrides(api_key="alice-key"))
    seen: list[httpx.Request] = []
    configured = make_provider(seen).with_config(_bank_config(), bank_id="alice", operation="reflect")
    await configured.call_with_tools(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "done", "parameters": {"type": "object"}}}],
        max_retries=0,
    )
    auth = seen[0].headers.get("x-api-key") or seen[0].headers["authorization"].removeprefix("Bearer ")
    assert auth == "alice-key"


async def test_each_member_of_a_chain_is_asked_separately(install):
    """A failover chain asks once per member, naming that member's provider and model,
    so an extension can keep a key meant for one of them away from the others."""
    failing: list[httpx.Request] = []
    primary = _openai_provider(failing)

    def fail(request: httpx.Request) -> httpx.Response:
        failing.append(request)
        return httpx.Response(500, json={"error": {"message": "down"}})

    primary._provider_impl._client = AsyncOpenAI(
        api_key="server-key",
        base_url="http://gateway.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(fail)),
    )
    fallback_seen: list[httpx.Request] = []
    fallback = _anthropic_provider(fallback_seen)
    fallback.model = "fallback-model"
    ext = install(None)

    chain = MultiLLMProvider([primary, fallback], LLMStrategyConfig(mode=LLM_STRATEGY_FAILOVER))
    await chain.with_config(_bank_config(), bank_id="alice", operation="retain").call(
        messages=[{"role": "user", "content": "hi"}], max_retries=0
    )

    assert [(t.provider, t.model) for t in ext.targets] == [("openai", "m"), ("anthropic", "fallback-model")]
    assert len(fallback_seen) == 1


# ── fail closed ───────────────────────────────────────────────────────────────


async def test_unsupporting_provider_refuses_instead_of_using_the_server_key(install):
    install(LLMRequestOverrides(api_key="alice-key"))
    llm = LLMProvider(provider="mock", api_key="", base_url="", model="m")
    with pytest.raises(LLMRequestRefused, match="cannot send a call with per-request credentials"):
        await _call_for_bank(llm, "alice")


async def test_unsupporting_provider_still_works_when_the_extension_says_server_key(install):
    install(None)
    llm = LLMProvider(provider="mock", api_key="", base_url="", model="m")
    await _call_for_bank(llm, "alice")


async def test_extension_error_fails_the_call(install):
    class Refusing(LLMRequestExtension):
        async def resolve(self, target: LLMRequestTarget) -> LLMRequestOverrides | None:
            raise LLMRequestRefused(f"no key for {target.bank_id}")

    install_llm_request_extension(Refusing({}))
    seen: list[httpx.Request] = []
    with pytest.raises(LLMRequestRefused, match="no key for mallory"):
        await _call_for_bank(_openai_provider(seen), "mallory")
    assert seen == []


async def test_a_server_credential_header_cannot_override_the_bank_key(install):
    """The SDKs apply default headers after their own auth header: a gateway key among
    them would win over the bank's key, so the call is refused instead."""
    install(LLMRequestOverrides(api_key="alice-key"))
    llm = _openai_provider([])
    llm.default_headers = {"x-portkey-api-key": "server-gateway-key", "x-trace": "t"}
    with pytest.raises(LLMRequestRefused, match="x-portkey-api-key") as refused:
        await _call_for_bank(llm, "alice")
    assert "server-gateway-key" not in str(refused.value)


async def test_a_credential_header_the_answer_overrides_is_fine(install):
    install(LLMRequestOverrides(api_key="alice-key", headers={"x-portkey-api-key": "alice-gateway-key"}))
    seen: list[httpx.Request] = []
    llm = _openai_provider(seen)
    llm.default_headers = {"x-portkey-api-key": "server-gateway-key"}
    await _call_for_bank(llm, "alice")
    assert seen[0].headers["x-portkey-api-key"] == "alice-gateway-key"


@pytest.mark.parametrize(
    ("model", "supported"),
    [
        ("openai/m", True),
        ("anthropic/claude-x", True),
        ("vertex_ai/gemini-x", False),
        ("bedrock/anthropic.claude-x", False),
        ("sagemaker/endpoint", False),
    ],
)
def test_litellm_takes_a_key_only_for_keyed_backends(model, supported):
    """Vertex, Bedrock, SageMaker ... ignore ``api_key`` and use the server's cloud credentials."""
    llm = LLMProvider(provider="litellm", api_key="server-key", base_url="", model=model)
    assert llm._provider_impl.supports_request_overrides() is supported


# Which class answers supports_request_overrides for each provider class. A new
# provider must be added here — and if it inherits True, its calls must go through
# the client the override is applied to.
_OVERRIDE_ANSWERED_BY = {
    "AnthropicLLM": "AnthropicLLM",
    "ChatGPTLLM": "LLMInterface",
    "ClaudeCodeLLM": "LLMInterface",
    "CodexLLM": "LLMInterface",
    "CursorLLM": "LLMInterface",
    "FireworksLLM": "OpenAICompatibleLLM",
    "GeminiLLM": "LLMInterface",
    "GitHubCopilotLLM": "LLMInterface",
    "LiteLLMLLM": "LiteLLMLLM",
    "LiteLLMRouterLLM": "LiteLLMRouterLLM",
    "LlamaCppLLM": "LLMInterface",
    "MockLLM": "LLMInterface",
    "NoneLLM": "LLMInterface",
    "NousLLM": "NousLLM",
    "OpenAICompatibleLLM": "OpenAICompatibleLLM",
    "OpenAIResponsesLLM": "LLMInterface",
    "XaiOAuthLLM": "LLMInterface",
}


def test_every_provider_declares_whether_it_takes_per_call_credentials():
    import importlib
    import pkgutil

    import hindsight_api.engine.providers as providers
    from hindsight_api.engine.llm_interface import LLMInterface

    for module in pkgutil.iter_modules(providers.__path__):
        importlib.import_module(f"{providers.__name__}.{module.name}")

    def subclasses(cls: type) -> list[type]:
        return [sub for child in cls.__subclasses__() for sub in (child, *subclasses(child))]

    found = {
        cls.__name__: next(c.__name__ for c in cls.__mro__ if "supports_request_overrides" in vars(c))
        for cls in subclasses(LLMInterface)
        if cls.__module__.startswith(providers.__name__)
    }
    assert found == _OVERRIDE_ANSWERED_BY


def test_providers_that_take_credentials_send_through_the_overridden_client():
    """Every SDK request in those providers goes through ``_request_client()``; the
    only direct uses of ``self._client`` are the batch APIs, which the extension
    disables at startup, and ``close``."""
    import re
    from pathlib import Path

    import hindsight_api.engine.providers as providers

    direct = {
        name: re.findall(r"self\._client\.([\w.]+)\(", (Path(providers.__path__[0]) / f"{name}.py").read_text())
        for name in ("openai_compatible_llm", "anthropic_llm", "fireworks_llm", "nous_llm")
    }
    assert {name: sorted(set(calls)) for name, calls in direct.items()} == {
        "openai_compatible_llm": ["batches.create", "batches.retrieve", "close", "files.content", "files.create"],
        "anthropic_llm": ["close", "messages.batches.create", "messages.batches.results", "messages.batches.retrieve"],
        "fireworks_llm": [],
        "nous_llm": [],
    }


async def test_wrong_return_type_is_refused(install):
    install("alice-key")  # type: ignore[arg-type]
    with pytest.raises(LLMRequestRefused, match="expected LLMRequestOverrides"):
        await _call_for_bank(_openai_provider([]), "alice")


# ── never logged ──────────────────────────────────────────────────────────────


def test_overrides_repr_hides_credentials():
    overrides = LLMRequestOverrides(api_key="sk-secret", headers={"authorization": "Bearer sk-secret"})
    assert "sk-secret" not in repr(overrides)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"api_key": ""},
        {"api_key": "   "},
        {"headers": {"x-key": "a\r\nx-injected: 1"}},
        {"headers": {"x-key\n": "a"}},
    ],
)
def test_overrides_reject_empty_keys_and_header_injection(kwargs):
    with pytest.raises(ValueError):
        LLMRequestOverrides(**kwargs)


def test_overrides_headers_are_frozen_copies():
    headers = {"x-key": "a"}
    overrides = LLMRequestOverrides(headers=headers)
    headers["x-key"] = "b"
    assert overrides.headers["x-key"] == "a"
    with pytest.raises(TypeError):
        overrides.headers["x-key"] = "c"  # type: ignore[index]


def test_4xx_dump_redacts_credentials(caplog):
    dumped = json.loads(
        _serialize_config(
            {"model": "m", "api_key": "sk-secret", "extra_headers": {"authorization": "Bearer sk-secret"}}
        )
    )
    assert dumped == {"model": "m", "api_key": "[redacted]", "extra_headers": "[redacted]"}
    assert not [r for r in caplog.records if "sk-secret" in r.getMessage()]


def test_logging_an_override_does_not_print_the_key(caplog):
    with caplog.at_level(logging.INFO):
        logging.getLogger("t").info("overrides=%s", LLMRequestOverrides(api_key="sk-secret"))
    assert "sk-secret" not in caplog.text


# ── engine wiring ─────────────────────────────────────────────────────────────


def _engine_env(**extra: str) -> dict[str, str]:
    return {
        "HINDSIGHT_API_LLM_PROVIDER": "none",
        "HINDSIGHT_API_LLM_MODEL": "none",
        "HINDSIGHT_API_LLM_API_KEY": "test-key",
        "HINDSIGHT_API_LLM_REQUEST_EXTENSION": f"{__name__}:_LoadableExtension",
        **extra,
    }


class _LoadableExtension(LLMRequestExtension):
    async def resolve(self, target: LLMRequestTarget) -> LLMRequestOverrides | None:
        return None


def _build_engine(env: dict[str, str]):
    import os
    from unittest.mock import MagicMock, patch

    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.memory_engine import MemoryEngine

    embeddings = MagicMock()
    embeddings.dimension = 384
    try:
        with patch.dict(os.environ, env, clear=False):
            clear_config_cache()
            return MemoryEngine(db_url="postgresql://localhost/hindsight_test", embeddings=embeddings)
    finally:
        # Keep the patched config from bleeding into other tests on this worker.
        clear_config_cache()


def test_engine_installs_the_configured_extension():
    from hindsight_api.extensions.llm_request import installed_llm_request_extension

    try:
        engine = _build_engine(_engine_env())
        assert isinstance(installed_llm_request_extension(), _LoadableExtension)
        assert installed_llm_request_extension() is engine._llm_request_extension
    finally:
        install_llm_request_extension(None)


def test_engine_refuses_batch_retain_with_the_extension():
    """Batch jobs are submitted once, on the server's account: there is no per-call
    request to put a bank's key on, so the combination must not boot."""
    from hindsight_api.extensions.llm_request import installed_llm_request_extension

    with pytest.raises(RuntimeError, match="not compatible with HINDSIGHT_API_RETAIN_BATCH_ENABLED"):
        _build_engine(_engine_env(HINDSIGHT_API_RETAIN_BATCH_ENABLED="true"))
    assert installed_llm_request_extension() is None
