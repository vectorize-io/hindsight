"""Tests for backend prompt-cache affinity on the OpenAI-compatible provider family.

Covers the `cache_affinity` knob end to end: mode parsing and `auto` host
resolution, the affinity id (operation trace identity, first-message hash
fallback), injection into both the plain and tool-calling call paths with
user-wins semantics, and the config plumbing that carries the setting from env
to the wire.

The decisive test here is `test_end_to_end_*`: direct-construction tests cannot
catch a break between config and the provider, which is exactly how a first
version of this feature compiled, tested green, and sent nothing on the reflect
lane.

All deterministic — no network, mocked clients only.
"""

import re
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.engine.cache_affinity import (
    OPENAI_PROMPT_CACHE_KEY_PARAM,
    XAI_CONV_ID_HEADER,
    CacheAffinityMode,
    apply_cache_affinity,
    cache_affinity_id,
    parse_cache_affinity,
    resolve_cache_affinity,
)
from hindsight_api.engine.llm_trace import LLMTraceContext, reset_trace_context, set_trace_context
from hindsight_api.engine.llm_wrapper import LLMProvider, create_llm_provider
from hindsight_api.engine.providers.nous_auth import NousAuthManager
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM

_HEX32 = re.compile(r"\A[0-9a-f]{32}\Z")


# ── helpers ───────────────────────────────────────────────────────────────────


def _llm(cache_affinity: str | None = None, **kwargs) -> OpenAICompatibleLLM:
    kwargs.setdefault("base_url", "https://example.test/v1")
    kwargs.setdefault("provider", "openai")
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("model", "gpt-4o-mini")
    return OpenAICompatibleLLM(cache_affinity=cache_affinity, **kwargs)


class _FakeNousAuth:
    """Minimal `NousAuthManager` stand-in: a fresh token, refresh never needed.

    Mirrors the fake in `test_nous_provider.py` but trimmed to what construction
    and a no-401 `call()` touch — `access_token`, `base_url`, `_token_is_stale()`.
    """

    def __init__(self, token: str = "tok-1", base_url: str = "https://inference-api.nousresearch.com/v1"):
        self.access_token = token
        self.base_url = base_url

    def _token_is_stale(self) -> bool:
        return False


def _nous_llm(cache_affinity: str | None = None, **kwargs):
    """Construct a NousLLM through the real `create_llm_provider` factory — the
    exact path that silently dropped `cache_affinity`/`default_headers` — with
    Nous's ~/.hermes/auth.json read stubbed out so the test needs no real login."""
    kwargs.setdefault("base_url", "https://inference-api.nousresearch.com/v1")
    kwargs.setdefault("api_key", "ignored")
    kwargs.setdefault("model", "deepseek/deepseek-v4-flash")
    kwargs.setdefault("reasoning_effort", "low")
    with patch.object(NousAuthManager, "from_file", return_value=_FakeNousAuth()):
        return create_llm_provider(provider="nous", cache_affinity=cache_affinity, **kwargs)


def _chat_response(content: str = "hello"):
    choice = SimpleNamespace(
        finish_reason="stop",
        message=SimpleNamespace(content=content, tool_calls=None, refusal=None, reasoning_content=None),
    )
    return SimpleNamespace(choices=[choice], usage=None, error=None)


async def _call(llm: OpenAICompatibleLLM, create: AsyncMock, **kwargs):
    """Drive `call()` against a mocked client, returning the mock for assertions."""
    llm._client.chat.completions.create = create
    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        await llm.call(messages=[{"role": "user", "content": "ping"}], max_retries=0, **kwargs)
    return create


async def _call_with_tools(llm: OpenAICompatibleLLM, create: AsyncMock, **kwargs):
    """Drive `call_with_tools()` against a mocked client."""
    llm._client.chat.completions.create = create
    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        await llm.call_with_tools(
            messages=[{"role": "user", "content": "ping"}],
            tools=[{"type": "function", "function": {"name": "noop", "parameters": {}}}],
            max_retries=0,
            **kwargs,
        )
    return create


@contextmanager
def _bound_trace(trace_id: str):
    """Bind an operation trace context, as `ConfiguredLLMProvider` does per call."""
    token = set_trace_context(LLMTraceContext(bank_id="bank-1", operation="reflect", trace_id=trace_id))
    try:
        yield
    finally:
        reset_trace_context(token)


# ── AC1: xai_conv_id header ───────────────────────────────────────────────────


async def test_xai_header_is_32_lowercase_hex():
    llm = _llm("xai_conv_id")
    create = await _call(llm, AsyncMock(return_value=_chat_response()))
    header = create.call_args.kwargs["extra_headers"][XAI_CONV_ID_HEADER]
    assert _HEX32.match(header)


async def test_xai_header_is_stable_within_one_trace_context():
    """Every LLM call of one reflect/retain run must pin to the same backend."""
    llm = _llm("xai_conv_id")
    create = AsyncMock(return_value=_chat_response())
    with _bound_trace("11111111-1111-1111-1111-111111111111"):
        await _call(llm, create)
        first = create.call_args.kwargs["extra_headers"][XAI_CONV_ID_HEADER]
        # Different message content: the id comes from operation identity, not payload.
        llm._client.chat.completions.create = create
        with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
            await llm.call(messages=[{"role": "user", "content": "a different prompt"}], max_retries=0)
        second = create.call_args.kwargs["extra_headers"][XAI_CONV_ID_HEADER]
    assert first == second


async def test_xai_header_differs_across_trace_contexts():
    llm = _llm("xai_conv_id")
    create = AsyncMock(return_value=_chat_response())
    with _bound_trace("11111111-1111-1111-1111-111111111111"):
        await _call(llm, create)
        first = create.call_args.kwargs["extra_headers"][XAI_CONV_ID_HEADER]
    with _bound_trace("22222222-2222-2222-2222-222222222222"):
        await _call(llm, create)
        second = create.call_args.kwargs["extra_headers"][XAI_CONV_ID_HEADER]
    assert first != second


# ── AC2: openai_prompt_cache_key ──────────────────────────────────────────────


async def test_openai_prompt_cache_key_is_sent():
    """`prompt_cache_key` is a first-class named param on chat.completions.create,
    so it is sent at the top level rather than through extra_body."""
    llm = _llm("openai_prompt_cache_key")
    create = await _call(llm, AsyncMock(return_value=_chat_response()))
    assert _HEX32.match(create.call_args.kwargs[OPENAI_PROMPT_CACHE_KEY_PARAM])


async def test_openai_prompt_cache_key_coexists_with_extra_body():
    """The provider's own extra_body defaults and the operator's configured
    extra_body must both survive — the affinity hint merges, never replaces."""
    llm = _llm(
        "openai_prompt_cache_key",
        provider="minimax",
        base_url="https://api.minimax.io/v1",
        model="MiniMax-M3",
        extra_body={"top_p": 0.9},
    )
    create = await _call(llm, AsyncMock(return_value=_chat_response()))
    kwargs = create.call_args.kwargs
    assert _HEX32.match(kwargs[OPENAI_PROMPT_CACHE_KEY_PARAM])
    assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}  # minimax default
    assert kwargs["extra_body"]["top_p"] == 0.9  # operator config


# ── AC3: default / none is byte-compatible with a pre-affinity request ────────


@pytest.mark.parametrize("mode", [None, "none"])
async def test_no_affinity_key_by_default(mode):
    llm = _llm(mode)
    create = await _call(llm, AsyncMock(return_value=_chat_response()))
    kwargs = create.call_args.kwargs
    assert "extra_headers" not in kwargs
    assert OPENAI_PROMPT_CACHE_KEY_PARAM not in kwargs
    assert XAI_CONV_ID_HEADER not in str(kwargs)


def test_invalid_mode_raises():
    """The setting has no visible effect in the response, so a typo that silently
    disabled it would be indistinguishable from it working."""
    with pytest.raises(ValueError, match="Invalid cache_affinity"):
        _llm("xai-conv-id")


# ── AC4: auto resolution table ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("provider", "base_url", "expected"),
    [
        ("openai", "https://api.x.ai/v1", CacheAffinityMode.XAI_CONV_ID),
        ("openai", "https://cli-chat-proxy.grok.com/v1", CacheAffinityMode.XAI_CONV_ID),
        ("openai", None, CacheAffinityMode.OPENAI_PROMPT_CACHE_KEY),
        ("openai", "", CacheAffinityMode.OPENAI_PROMPT_CACHE_KEY),
        ("openai", "https://api.openai.com/v1", CacheAffinityMode.OPENAI_PROMPT_CACHE_KEY),
        ("openai", "https://my-res.openai.azure.com/openai/deployments/x", CacheAffinityMode.NONE),
        ("fireworks", "https://api.fireworks.ai/inference/v1", CacheAffinityMode.NONE),
        ("fireworks", "", CacheAffinityMode.NONE),
        ("openai", "https://llm.internal.example/v1", CacheAffinityMode.NONE),
    ],
)
def test_auto_resolution_table(provider, base_url, expected):
    assert resolve_cache_affinity(CacheAffinityMode.AUTO, provider, base_url) is expected


@pytest.mark.parametrize("base_url", ["https://api.vertex.ai/v1", "https://x.ai.evil.example/v1"])
def test_auto_does_not_substring_match_xai(base_url):
    """`"x.ai" in base_url` would false-match both of these; the host is parsed."""
    assert resolve_cache_affinity(CacheAffinityMode.AUTO, "openai", base_url) is CacheAffinityMode.NONE


def test_auto_resolves_once_at_construction():
    llm = _llm("auto", base_url="https://api.x.ai/v1")
    assert llm._cache_affinity is CacheAffinityMode.XAI_CONV_ID


@pytest.mark.parametrize("mode", ["none", "xai_conv_id", "openai_prompt_cache_key"])
def test_explicit_modes_are_not_re_resolved(mode):
    parsed = parse_cache_affinity(mode)
    assert resolve_cache_affinity(parsed, "openai", "https://api.x.ai/v1") is parsed


# ── AC5: no-context fallback (first-message hash) ─────────────────────────────


def test_fallback_hashes_the_first_message():
    messages = [{"role": "system", "content": "you are helpful"}, {"role": "user", "content": "hi"}]
    assert _HEX32.match(cache_affinity_id(messages))


def test_fallback_is_stable_as_the_message_list_grows():
    """The agent loop appends turns; the pin must not rotate mid-conversation."""
    base = [{"role": "system", "content": "you are helpful"}]
    grown = [*base, {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert cache_affinity_id(base) == cache_affinity_id(grown)


def test_fallback_differs_for_a_different_first_message():
    a = cache_affinity_id([{"role": "system", "content": "prompt A"}])
    b = cache_affinity_id([{"role": "system", "content": "prompt B"}])
    assert a != b


@pytest.mark.parametrize("messages", [None, [], "bogus", ["not a dict"], [42]])
def test_fallback_returns_none_for_malformed_messages(messages):
    """A bare string would index to its first character and mint an id from
    garbage; anything unexpected sends no hint at all."""
    assert cache_affinity_id(messages) is None


async def test_no_header_when_messages_are_malformed():
    """Fail-open all the way to the wire: no id means no header, not an error."""
    request = {"messages": "bogus"}
    apply_cache_affinity(request, CacheAffinityMode.XAI_CONV_ID)
    assert request == {"messages": "bogus"}


async def test_call_uses_the_fallback_hash_without_a_trace_context():
    llm = _llm("xai_conv_id")
    create = await _call(llm, AsyncMock(return_value=_chat_response()))
    sent = create.call_args.kwargs["extra_headers"][XAI_CONV_ID_HEADER]
    assert sent == cache_affinity_id(create.call_args.kwargs["messages"])


# ── AC6: call_with_tools parity ───────────────────────────────────────────────


async def test_tools_path_sends_xai_header():
    llm = _llm("xai_conv_id")
    create = await _call_with_tools(llm, AsyncMock(return_value=_chat_response("done")))
    assert _HEX32.match(create.call_args.kwargs["extra_headers"][XAI_CONV_ID_HEADER])


async def test_tools_path_sends_prompt_cache_key():
    llm = _llm("openai_prompt_cache_key")
    create = await _call_with_tools(llm, AsyncMock(return_value=_chat_response("done")))
    assert _HEX32.match(create.call_args.kwargs[OPENAI_PROMPT_CACHE_KEY_PARAM])


async def test_tools_path_sends_nothing_by_default():
    llm = _llm()
    create = await _call_with_tools(llm, AsyncMock(return_value=_chat_response("done")))
    kwargs = create.call_args.kwargs
    assert "extra_headers" not in kwargs
    assert OPENAI_PROMPT_CACHE_KEY_PARAM not in kwargs


# ── AC13: fallback hash on the tools path ─────────────────────────────────────


async def test_tools_path_fallback_hash_is_stable():
    """`call_with_tools` builds its message list differently; its fallback id must
    still derive from the first message and survive a growing conversation."""
    llm = _llm("xai_conv_id")
    create = AsyncMock(return_value=_chat_response("done"))
    llm._client.chat.completions.create = create
    tools = [{"type": "function", "function": {"name": "noop", "parameters": {}}}]
    system = {"role": "system", "content": "you are helpful"}
    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        await llm.call_with_tools(messages=[system], tools=tools, max_retries=0)
        first = create.call_args.kwargs["extra_headers"][XAI_CONV_ID_HEADER]
        await llm.call_with_tools(
            messages=[system, {"role": "user", "content": "and now?"}], tools=tools, max_retries=0
        )
        second = create.call_args.kwargs["extra_headers"][XAI_CONV_ID_HEADER]
    assert _HEX32.match(first)
    assert first == second


# ── AC7: adjacent fix — default_headers reaches the SDK client ────────────────


def test_default_headers_reach_the_openai_compatible_client():
    """`HINDSIGHT_API_LLM_DEFAULT_HEADERS` used to no-op for all 15 providers on
    this branch: the factory accepted the field but never forwarded it."""
    llm = create_llm_provider(
        provider="openai",
        api_key="test-key",
        base_url="https://example.test/v1",
        model="gpt-4o-mini",
        reasoning_effort="low",
        default_headers={"x-test": "1"},
    )
    assert llm._client.default_headers["x-test"] == "1"


def test_default_headers_reach_the_fireworks_client():
    """Fireworks subclasses OpenAICompatibleLLM and had the same gap."""
    llm = create_llm_provider(
        provider="fireworks",
        api_key="test-key",
        base_url="https://api.fireworks.ai/inference/v1",
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        reasoning_effort="low",
        default_headers={"x-test": "2"},
    )
    assert llm._client.default_headers["x-test"] == "2"


def test_fireworks_branch_forwards_cache_affinity():
    llm = create_llm_provider(
        provider="fireworks",
        api_key="test-key",
        base_url="https://api.fireworks.ai/inference/v1",
        model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        reasoning_effort="low",
        cache_affinity="xai_conv_id",
    )
    assert llm._cache_affinity is CacheAffinityMode.XAI_CONV_ID


def test_default_headers_reach_the_nous_client():
    """NousLLM subclasses OpenAICompatibleLLM and had the same gap: the `nous`
    factory branch forwarded neither `default_headers` nor `cache_affinity`,
    even though `NousLLM.__init__` already passes both through **kwargs."""
    llm = _nous_llm(default_headers={"x-test": "3"})
    assert llm._client.default_headers["x-test"] == "3"


async def test_nous_branch_forwards_cache_affinity():
    """AC7-style factory-branch check, taken all the way to the wire (AC1 style)
    rather than stopping at `_cache_affinity`, since the missing kwarg here is a
    silent no-op — the same failure shape AC10's end-to-end test exists for."""
    llm = _nous_llm(cache_affinity="xai_conv_id")
    create = await _call(llm, AsyncMock(return_value=_chat_response()))
    header = create.call_args.kwargs["extra_headers"][XAI_CONV_ID_HEADER]
    assert _HEX32.match(header)


# ── AC8: config plumbing (env round-trip) ─────────────────────────────────────


@pytest.fixture
def clean_llm_env(monkeypatch):
    """Strip all HINDSIGHT_API_*LLM* env so each test sets only what it needs."""
    import os

    from hindsight_api.config import clear_config_cache

    for key in list(os.environ):
        if key.startswith("HINDSIGHT_API_") and "LLM" in key:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HINDSIGHT_API_LLM_PROVIDER", "openai")
    monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "sk-primary")
    monkeypatch.setenv("HINDSIGHT_API_SKIP_LLM_VERIFICATION", "true")
    clear_config_cache()
    yield monkeypatch
    clear_config_cache()


def test_config_reads_global_and_per_operation_affinity(clean_llm_env):
    from hindsight_api.config import HindsightConfig

    clean_llm_env.setenv("HINDSIGHT_API_LLM_CACHE_AFFINITY", "auto")
    clean_llm_env.setenv("HINDSIGHT_API_REFLECT_LLM_CACHE_AFFINITY", "xai_conv_id")
    config = HindsightConfig.from_env()

    assert config.llm_cache_affinity == "auto"
    assert config.reflect_llm_cache_affinity == "xai_conv_id"
    assert config.retain_llm_cache_affinity is None
    assert config.consolidation_llm_cache_affinity is None


def test_config_affinity_defaults_to_auto(clean_llm_env):
    """Unset means "auto", not "off".

    "auto" only emits a hint for hosts documented to accept one (see
    test_auto_default_sends_nothing_to_an_unrecognized_backend), so defaulting
    it on costs unknown backends nothing while every xAI/OpenAI deployment gets
    the cache hit it was otherwise silently losing.
    """
    from hindsight_api.config import HindsightConfig

    assert HindsightConfig.from_env().llm_cache_affinity == "auto"


def test_llm_provider_from_env_defaults_to_auto(clean_llm_env):
    """The two env entry points must agree; they resolved differently before."""
    clean_llm_env.setenv("HINDSIGHT_API_LLM_PROVIDER", "openai")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_API_KEY", "sk-test")

    assert LLMProvider.from_env().cache_affinity == "auto"


@pytest.mark.parametrize(
    ("provider", "base_url"),
    [
        ("openai", "https://my-proxy.internal/v1"),
        ("openai", "http://localhost:8000/v1"),
        ("ollama", "http://localhost:11434/v1"),
        ("groq", "https://api.groq.com/openai/v1"),
        ("deepseek", "https://api.deepseek.com"),
        ("openrouter", "https://openrouter.ai/api/v1"),
        ("lmstudio", "http://localhost:1234/v1"),
        ("", "https://vllm.internal/v1"),
    ],
)
def test_auto_default_sends_nothing_to_an_unrecognized_backend(provider, base_url):
    """The safety property that makes "auto" viable as a default.

    An OpenAI-compatible backend that never documented either mechanism must
    receive a byte-identical request. "auto" is an allowlist, so anything off it
    resolves to none rather than being probed with an unfamiliar field.
    """
    assert resolve_cache_affinity(CacheAffinityMode.AUTO, provider, base_url) is CacheAffinityMode.NONE


@pytest.mark.parametrize(
    ("provider", "base_url", "expected"),
    [
        ("openai", "https://api.x.ai/v1", CacheAffinityMode.XAI_CONV_ID),
        ("openai", "https://grok.com/v1", CacheAffinityMode.XAI_CONV_ID),
        ("openai", None, CacheAffinityMode.OPENAI_PROMPT_CACHE_KEY),
        ("openai", "https://api.openai.com/v1", CacheAffinityMode.OPENAI_PROMPT_CACHE_KEY),
        ("openai", "https://myco.openai.azure.com/", CacheAffinityMode.NONE),
    ],
)
def test_auto_default_sends_a_hint_only_to_documented_hosts(provider, base_url, expected):
    assert resolve_cache_affinity(CacheAffinityMode.AUTO, provider, base_url) is expected


def test_indexed_member_carries_affinity(clean_llm_env):
    from hindsight_api.config import _parse_llm_members

    clean_llm_env.setenv("HINDSIGHT_API_REFLECT_LLM_1_PROVIDER", "openai")
    clean_llm_env.setenv("HINDSIGHT_API_REFLECT_LLM_1_API_KEY", "sk-member")
    clean_llm_env.setenv("HINDSIGHT_API_REFLECT_LLM_1_CACHE_AFFINITY", "xai_conv_id")

    members = _parse_llm_members("REFLECT_")
    assert [m.cache_affinity for m in members] == ["xai_conv_id"]
    assert _parse_llm_members("RETAIN_") == []


def test_llm_provider_from_env_carries_affinity(clean_llm_env):
    clean_llm_env.setenv("HINDSIGHT_API_LLM_CACHE_AFFINITY", "xai_conv_id")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_BASE_URL", "https://api.x.ai/v1")
    llm = LLMProvider.from_env()
    assert llm._provider_impl._cache_affinity is CacheAffinityMode.XAI_CONV_ID


# ── AC9: bank attribution is unaffected ───────────────────────────────────────


async def test_bank_attribution_and_affinity_coexist(clean_llm_env):
    from hindsight_api.config import clear_config_cache
    from hindsight_api.engine.memory_engine import _current_bank_id

    clean_llm_env.setenv("HINDSIGHT_API_LLM_SEND_BANK_AS_USER", "true")
    clear_config_cache()

    llm = _llm("xai_conv_id")
    create = AsyncMock(return_value=_chat_response())
    token = _current_bank_id.set("user-9")
    try:
        await _call(llm, create)
    finally:
        _current_bank_id.reset(token)

    assert create.call_args.kwargs["user"] == "user-9"
    assert _HEX32.match(create.call_args.kwargs["extra_headers"][XAI_CONV_ID_HEADER])


# ── AC11: user-configured values win ──────────────────────────────────────────


async def test_configured_prompt_cache_key_survives():
    """The operator's escape hatch: an explicit prompt_cache_key in extra_body
    reaches the same wire field, so ours is suppressed rather than duplicated."""
    llm = _llm("openai_prompt_cache_key", extra_body={"prompt_cache_key": "operator-value"})
    create = await _call(llm, AsyncMock(return_value=_chat_response()))
    kwargs = create.call_args.kwargs
    assert kwargs["extra_body"]["prompt_cache_key"] == "operator-value"
    assert OPENAI_PROMPT_CACHE_KEY_PARAM not in kwargs


def test_preset_conv_id_header_is_not_clobbered():
    """Mirrors `apply_bank_attribution`'s "never override an explicit value" rule."""
    request = {
        "messages": [{"role": "user", "content": "ping"}],
        "extra_headers": {XAI_CONV_ID_HEADER: "caller-pinned"},
    }
    apply_cache_affinity(request, CacheAffinityMode.XAI_CONV_ID)
    assert request["extra_headers"][XAI_CONV_ID_HEADER] == "caller-pinned"


def test_other_extra_headers_are_preserved():
    request = {"messages": [{"role": "user", "content": "ping"}], "extra_headers": {"x-other": "keep"}}
    apply_cache_affinity(request, CacheAffinityMode.XAI_CONV_ID)
    assert request["extra_headers"]["x-other"] == "keep"
    assert _HEX32.match(request["extra_headers"][XAI_CONV_ID_HEADER])


# ── AC12: the hint survives a retry ───────────────────────────────────────────


async def test_affinity_persists_across_a_retry():
    """`call_params` is built once before the retry loop; attempt 2 must carry it."""
    llm = _llm("xai_conv_id")
    # Empty content raises a retryable ProviderResponseError on attempt 1.
    create = AsyncMock(side_effect=[_chat_response(content=""), _chat_response()])
    llm._client.chat.completions.create = create
    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        await llm.call(
            messages=[{"role": "user", "content": "ping"}],
            max_retries=1,
            initial_backoff=0.0,
        )
    assert create.await_count == 2
    for call in create.await_args_list:
        assert _HEX32.match(call.kwargs["extra_headers"][XAI_CONV_ID_HEADER])


# ── AC10: END-TO-END anti-inert ───────────────────────────────────────────────


async def test_end_to_end_reflect_lane_sends_the_affinity_header(clean_llm_env):
    """The decisive test. Builds the reflect provider through the PRODUCTION path
    — env -> HindsightConfig -> MemoryEngine's reflect base build -> LLMProvider
    -> create_llm_provider -> OpenAICompatibleLLM -> with_config() -> the wire —
    and asserts the header lands on the request.

    A direct-construction test passes even when nothing connects config to the
    provider, which is exactly how the first version of this feature shipped
    inert for the reflect lane. This one fails in that state.
    """
    from hindsight_api import MemoryEngine

    clean_llm_env.setenv("HINDSIGHT_API_LLM_MODEL", "grok-4.5")
    clean_llm_env.setenv("HINDSIGHT_API_REFLECT_LLM_BASE_URL", "https://api.x.ai/v1")
    clean_llm_env.setenv("HINDSIGHT_API_REFLECT_LLM_CACHE_AFFINITY", "xai_conv_id")

    engine = MemoryEngine(skip_llm_verification=True)
    reflect_llm = engine._reflect_llm_config
    create = AsyncMock(return_value=_chat_response())
    reflect_llm._provider_impl._client.chat.completions.create = create

    configured = reflect_llm.with_config(
        SimpleNamespace(llm_gemini_safety_settings=None), bank_id="bank-e2e", operation="reflect"
    )
    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        await configured.call(messages=[{"role": "user", "content": "ping"}], max_retries=0)

    assert _HEX32.match(create.call_args.kwargs["extra_headers"][XAI_CONV_ID_HEADER])


async def test_end_to_end_failover_member_sends_the_affinity_header(clean_llm_env):
    """Same production path for an indexed chain member, which is built by
    `_member_to_llm` rather than the per-operation base build. Without that leg,
    a failover member would silently drop the pin the primary carries."""
    from hindsight_api import MemoryEngine
    from hindsight_api.engine.multi_llm import MultiLLMProvider

    clean_llm_env.setenv("HINDSIGHT_API_LLM_MODEL", "grok-4.5")
    clean_llm_env.setenv("HINDSIGHT_API_LLM_CACHE_AFFINITY", "xai_conv_id")
    clean_llm_env.setenv("HINDSIGHT_API_REFLECT_LLM_1_PROVIDER", "openai")
    clean_llm_env.setenv("HINDSIGHT_API_REFLECT_LLM_1_API_KEY", "sk-member")
    clean_llm_env.setenv("HINDSIGHT_API_REFLECT_LLM_1_MODEL", "grok-4.5-fallback")
    clean_llm_env.setenv("HINDSIGHT_API_REFLECT_LLM_1_BASE_URL", "https://api.x.ai/v1")
    clean_llm_env.setenv("HINDSIGHT_API_REFLECT_LLM_STRATEGY", '{"mode": "failover"}')

    engine = MemoryEngine(skip_llm_verification=True)
    chain = engine._reflect_llm_config
    assert isinstance(chain, MultiLLMProvider)
    member = chain._members[1]
    assert member.model == "grok-4.5-fallback"

    create = AsyncMock(return_value=_chat_response())
    member._provider_impl._client.chat.completions.create = create
    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        await member.call(messages=[{"role": "user", "content": "ping"}], max_retries=0)

    assert _HEX32.match(create.call_args.kwargs["extra_headers"][XAI_CONV_ID_HEADER])
