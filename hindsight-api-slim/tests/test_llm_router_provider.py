"""
Tests for the LiteLLM Router LLM provider — chain config parsing, factory
dispatch, and the Router-backed call paths (plain text, structured output,
tool calls, retry on transient failure).
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.config import (
    ENV_CONSOLIDATION_LLM_LITELLMROUTER_CHAIN,
    ENV_LLM_LITELLMROUTER_CHAIN,
    ENV_LLM_PROVIDER,
    ENV_REFLECT_LLM_LITELLMROUTER_CHAIN,
    ENV_RETAIN_LLM_LITELLMROUTER_CHAIN,
    HindsightConfig,
    _parse_llm_router_chain,
)
from hindsight_api.engine.llm_wrapper import create_llm_provider
from hindsight_api.engine.providers.litellm_router_llm import (
    LiteLLMRouterLLM,
    _build_fallbacks,
    _build_litellm_model,
    _build_model_list,
)


@pytest.fixture
def two_step_chain() -> list[dict[str, Any]]:
    return [
        {
            "provider": "openai",
            "model": "MiniMax-M2.7",
            "api_key": "sk-primary",
            "base_url": "https://api.minimax.io/v1",
        },
        {"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-fallback", "base_url": None},
    ]


@pytest.fixture
def mock_router_response() -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = "ok"
    choice.message.tool_calls = None
    choice.finish_reason = "stop"
    response.choices = [choice]
    response.usage.prompt_tokens = 12
    response.usage.completion_tokens = 3
    response._hidden_params = {"model": "openai/gpt-4o-mini"}
    return response


# --- chain parsing -----------------------------------------------------------


class TestParseLLMRouterChain:
    def test_unset_returns_none(self, monkeypatch):
        monkeypatch.delenv(ENV_LLM_LITELLMROUTER_CHAIN, raising=False)
        assert _parse_llm_router_chain(ENV_LLM_LITELLMROUTER_CHAIN) is None

    def test_empty_string_returns_none(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_LITELLMROUTER_CHAIN, "  ")
        assert _parse_llm_router_chain(ENV_LLM_LITELLMROUTER_CHAIN) is None

    def test_valid_chain(self, monkeypatch):
        chain = [
            {"provider": "openai", "model": "gpt-4o-mini", "api_key": "k1"},
            {"provider": "anthropic", "model": "claude-sonnet-4-5", "api_key": "k2"},
        ]
        monkeypatch.setenv(ENV_LLM_LITELLMROUTER_CHAIN, json.dumps(chain))
        assert _parse_llm_router_chain(ENV_LLM_LITELLMROUTER_CHAIN) == chain

    def test_invalid_json(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_LITELLMROUTER_CHAIN, "{not json")
        with pytest.raises(ValueError, match="invalid JSON"):
            _parse_llm_router_chain(ENV_LLM_LITELLMROUTER_CHAIN)

    def test_not_a_list(self, monkeypatch):
        monkeypatch.setenv(
            ENV_LLM_LITELLMROUTER_CHAIN,
            json.dumps({"provider": "openai", "model": "x"}),
        )
        with pytest.raises(ValueError, match="non-empty JSON list"):
            _parse_llm_router_chain(ENV_LLM_LITELLMROUTER_CHAIN)

    def test_empty_list(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_LITELLMROUTER_CHAIN, "[]")
        with pytest.raises(ValueError, match="non-empty JSON list"):
            _parse_llm_router_chain(ENV_LLM_LITELLMROUTER_CHAIN)

    def test_missing_required_keys(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_LITELLMROUTER_CHAIN, json.dumps([{"provider": "openai"}]))
        with pytest.raises(ValueError, match="'provider' and 'model' are required"):
            _parse_llm_router_chain(ENV_LLM_LITELLMROUTER_CHAIN)

    def test_unknown_keys_rejected(self, monkeypatch):
        monkeypatch.setenv(
            ENV_LLM_LITELLMROUTER_CHAIN,
            json.dumps([{"provider": "openai", "model": "gpt-4o", "weight": 5}]),
        )
        with pytest.raises(ValueError, match="unknown keys"):
            _parse_llm_router_chain(ENV_LLM_LITELLMROUTER_CHAIN)


class TestFromEnvLoadsChains:
    def test_chain_loaded_when_provider_is_litellmrouter(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_PROVIDER, "litellmrouter")
        monkeypatch.setenv(
            ENV_LLM_LITELLMROUTER_CHAIN,
            json.dumps(
                [
                    {"provider": "openai", "model": "gpt-4o-mini", "api_key": "k1"},
                    {"provider": "openai", "model": "gpt-4o", "api_key": "k2"},
                ]
            ),
        )
        cfg = HindsightConfig.from_env()
        # Provider name is whatever the user set — no auto-promotion.
        assert cfg.llm_provider == "litellmrouter"
        assert cfg.llm_litellmrouter_chain is not None
        assert len(cfg.llm_litellmrouter_chain) == 2

    def test_chain_unset_keeps_default_provider(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_PROVIDER, "openai")
        monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "sk-primary")
        monkeypatch.delenv(ENV_LLM_LITELLMROUTER_CHAIN, raising=False)
        cfg = HindsightConfig.from_env()
        assert cfg.llm_provider == "openai"
        assert cfg.llm_litellmrouter_chain is None

    def test_per_op_chains_independent(self, monkeypatch):
        """Per-op chain env vars populate per-op fields without affecting the default."""
        monkeypatch.setenv(ENV_LLM_PROVIDER, "openai")
        monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "sk-primary")
        monkeypatch.setenv(
            ENV_RETAIN_LLM_LITELLMROUTER_CHAIN,
            json.dumps([{"provider": "openai", "model": "retain-1", "api_key": "rk"}]),
        )
        monkeypatch.setenv(
            ENV_REFLECT_LLM_LITELLMROUTER_CHAIN,
            json.dumps([{"provider": "anthropic", "model": "claude", "api_key": "ak"}]),
        )
        monkeypatch.setenv(
            ENV_CONSOLIDATION_LLM_LITELLMROUTER_CHAIN,
            json.dumps([{"provider": "openai", "model": "consol-1", "api_key": "ck"}]),
        )
        cfg = HindsightConfig.from_env()
        assert cfg.llm_litellmrouter_chain is None
        assert cfg.retain_llm_litellmrouter_chain == [{"provider": "openai", "model": "retain-1", "api_key": "rk"}]
        assert cfg.reflect_llm_litellmrouter_chain == [{"provider": "anthropic", "model": "claude", "api_key": "ak"}]
        assert cfg.consolidation_llm_litellmrouter_chain == [
            {"provider": "openai", "model": "consol-1", "api_key": "ck"}
        ]


# --- helper translation ------------------------------------------------------


class TestModelTranslation:
    def test_known_provider_prefixes(self):
        assert _build_litellm_model("openai", "gpt-4o-mini") == "openai/gpt-4o-mini"
        assert _build_litellm_model("anthropic", "claude-sonnet-4-5") == "anthropic/claude-sonnet-4-5"
        assert _build_litellm_model("gemini", "gemini-2.5-flash") == "gemini/gemini-2.5-flash"
        assert _build_litellm_model("vertexai", "gemini-2.5-flash") == "vertex_ai/gemini-2.5-flash"

    def test_unknown_provider_falls_back_to_openai(self):
        # OpenAI-compatible providers (lmstudio, custom, etc.) route via openai/ + base_url
        assert _build_litellm_model("lmstudio", "qwen3") == "openai/qwen3"
        assert _build_litellm_model("minimax", "MiniMax-M2.7") == "openai/MiniMax-M2.7"

    def test_pre_qualified_model_passes_through(self):
        assert _build_litellm_model("openai", "anthropic/claude-sonnet-4-5") == "anthropic/claude-sonnet-4-5"

    def test_litellm_provider_uses_raw_model(self):
        assert _build_litellm_model("litellm", "bedrock/anthropic.claude-3-5-sonnet") == (
            "bedrock/anthropic.claude-3-5-sonnet"
        )


class TestModelListBuilder:
    def test_builds_one_group_per_chain_entry(self, two_step_chain):
        ml = _build_model_list(two_step_chain, timeout=30.0)
        assert [d["model_name"] for d in ml] == ["hindsight-chain-0", "hindsight-chain-1"]
        assert ml[0]["litellm_params"]["model"] == "openai/MiniMax-M2.7"
        assert ml[0]["litellm_params"]["api_key"] == "sk-primary"
        assert ml[0]["litellm_params"]["api_base"] == "https://api.minimax.io/v1"
        assert ml[1]["litellm_params"]["model"] == "openai/gpt-4o-mini"
        assert "api_base" not in ml[1]["litellm_params"]  # base_url=None should be omitted

    def test_fallbacks_wired_in_order(self):
        assert _build_fallbacks(1) == []
        assert _build_fallbacks(2) == [{"hindsight-chain-0": ["hindsight-chain-1"]}]
        assert _build_fallbacks(3) == [{"hindsight-chain-0": ["hindsight-chain-1", "hindsight-chain-2"]}]


# --- factory dispatch --------------------------------------------------------


class TestFactoryDispatch:
    def test_router_provider_requires_chain(self):
        with pytest.raises(ValueError, match="non-empty chain"):
            create_llm_provider(
                provider="litellmrouter",
                api_key="",
                base_url="",
                model="unused",
                reasoning_effort="low",
                chain=None,
            )

    def test_router_provider_returns_router_impl(self, two_step_chain):
        with patch.dict("sys.modules", {"litellm": MagicMock(), "litellm.Router": MagicMock()}):
            with patch(
                "hindsight_api.engine.providers.litellm_router_llm.LiteLLMRouterLLM.__init__",
                return_value=None,
            ) as mock_init:
                impl = create_llm_provider(
                    provider="litellmrouter",
                    api_key="",
                    base_url="",
                    model="unused",
                    reasoning_effort="low",
                    chain=two_step_chain,
                )
                assert isinstance(impl, LiteLLMRouterLLM)
                mock_init.assert_called_once()
                _, kwargs = mock_init.call_args
                assert kwargs["chain"] == two_step_chain


# --- Router-backed call paths ------------------------------------------------


def _make_router_provider(chain: list[dict[str, Any]], mock_router: Any) -> LiteLLMRouterLLM:
    """Construct a LiteLLMRouterLLM with the inner Router replaced by a mock."""
    fake_litellm = MagicMock()
    fake_router_cls = MagicMock(return_value=mock_router)
    fake_litellm.Router = fake_router_cls
    with patch.dict("sys.modules", {"litellm": fake_litellm}):
        with patch(
            "hindsight_api.engine.providers.litellm_router_llm.Router",
            fake_router_cls,
            create=True,
        ):
            # Bypass the import inside __init__ by injecting directly via attribute swap.
            provider = LiteLLMRouterLLM.__new__(LiteLLMRouterLLM)
            # Initialise the LLMInterface base attrs without invoking the heavy ctor.
            provider.provider = "litellmrouter"
            provider.api_key = ""
            provider.base_url = ""
            provider.model = "unused"
            provider.reasoning_effort = "low"
            provider.timeout = 300.0
            provider.chain = chain
            provider._litellm = fake_litellm
            provider._router = mock_router
            return provider


class TestRouterCall:
    @pytest.mark.asyncio
    async def test_plain_text_call(self, two_step_chain, mock_router_response):
        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=mock_router_response)
        provider = _make_router_provider(two_step_chain, mock_router)

        result = await provider.call(
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=50,
            max_retries=0,
        )
        assert result == "ok"
        # Always invoked against the primary group; Router handles fallback internally.
        kwargs = mock_router.acompletion.await_args.kwargs
        assert kwargs["model"] == "hindsight-chain-0"

    @pytest.mark.asyncio
    async def test_structured_output(self, two_step_chain):
        class MySchema(BaseModel):
            answer: str

        response = MagicMock()
        choice = MagicMock()
        choice.message.content = '{"answer": "42"}'
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        response.choices = [choice]
        response.usage.prompt_tokens = 5
        response.usage.completion_tokens = 5
        response._hidden_params = {"model": "openai/gpt-4o-mini"}

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=response)
        provider = _make_router_provider(two_step_chain, mock_router)

        result = await provider.call(
            messages=[{"role": "user", "content": "q"}],
            response_format=MySchema,
            max_retries=0,
        )
        assert isinstance(result, MySchema)
        assert result.answer == "42"

    @pytest.mark.asyncio
    async def test_retry_on_transient_then_success(self, two_step_chain, mock_router_response):
        mock_router = MagicMock()
        # First call raises a 503-style error, second call returns ok.
        mock_router.acompletion = AsyncMock(side_effect=[Exception("503 Service Unavailable"), mock_router_response])
        provider = _make_router_provider(two_step_chain, mock_router)

        result = await provider.call(
            messages=[{"role": "user", "content": "hi"}],
            max_retries=2,
            initial_backoff=0.0,  # no real sleep in tests
            max_backoff=0.0,
        )
        assert result == "ok"
        assert mock_router.acompletion.await_count == 2

    @pytest.mark.asyncio
    async def test_auth_error_does_not_retry(self, two_step_chain):
        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(side_effect=Exception("401 Unauthorized: bad key"))
        provider = _make_router_provider(two_step_chain, mock_router)

        with pytest.raises(Exception, match="401"):
            await provider.call(
                messages=[{"role": "user", "content": "hi"}],
                max_retries=5,
                initial_backoff=0.0,
            )
        assert mock_router.acompletion.await_count == 1

    @pytest.mark.asyncio
    async def test_call_with_tools(self, two_step_chain):
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = None
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "lookup"
        tool_call.function.arguments = '{"q": "x"}'
        choice.message.tool_calls = [tool_call]
        choice.finish_reason = "tool_calls"
        response.choices = [choice]
        response.usage.prompt_tokens = 5
        response.usage.completion_tokens = 2
        response._hidden_params = {"model": "openai/gpt-4o-mini"}

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=response)
        provider = _make_router_provider(two_step_chain, mock_router)

        result = await provider.call_with_tools(
            messages=[{"role": "user", "content": "use tool"}],
            tools=[{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
            max_retries=0,
        )
        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "lookup"
        assert result.tool_calls[0].arguments == {"q": "x"}
