"""
Tests for the LiteLLM Router LLM provider — config parsing, factory dispatch,
and the Router-backed call paths (plain text, structured output, tool calls,
retry on transient failure).

The provider is a thin pass-through to ``litellm.Router``. The chain config
shape mirrors LiteLLM's API; we don't translate model names or impose
fallbacks. See https://docs.litellm.ai/docs/routing.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.config import (
    ENV_CONSOLIDATION_LLM_LITELLMROUTER_CONFIG,
    ENV_LLM_LITELLMROUTER_CONFIG,
    ENV_LLM_PROVIDER,
    ENV_REFLECT_LLM_LITELLMROUTER_CONFIG,
    ENV_RETAIN_LLM_LITELLMROUTER_CONFIG,
    HindsightConfig,
    _parse_llm_router_config,
)
from hindsight_api.engine.llm_wrapper import create_llm_provider
from hindsight_api.engine.providers.litellm_router_llm import LiteLLMRouterLLM


@pytest.fixture
def two_step_config() -> dict[str, Any]:
    """Raw LiteLLM Router config: two deployments wired for ordered fallback.

    Hindsight always issues completions against ``model_name="default"``;
    additional groups become fallback / load-balance pool members per the
    user's ``fallbacks`` / ``routing_strategy`` settings.
    """
    return {
        "model_list": [
            {
                "model_name": "default",
                "litellm_params": {
                    "model": "openai/MiniMax-M2.7",
                    "api_key": "sk-primary",
                    "api_base": "https://api.minimax.io/v1",
                },
            },
            {
                "model_name": "fallback",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-fallback"},
            },
        ],
        "fallbacks": [{"default": ["fallback"]}],
        "num_retries": 0,
    }


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


# --- config parsing ----------------------------------------------------------


class TestParseRouterConfig:
    def test_unset_returns_none(self, monkeypatch):
        monkeypatch.delenv(ENV_LLM_LITELLMROUTER_CONFIG, raising=False)
        assert _parse_llm_router_config(ENV_LLM_LITELLMROUTER_CONFIG) is None

    def test_empty_string_returns_none(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_LITELLMROUTER_CONFIG, "  ")
        assert _parse_llm_router_config(ENV_LLM_LITELLMROUTER_CONFIG) is None

    def test_valid_config_passes_through(self, monkeypatch, two_step_config):
        """Whatever the user provides round-trips verbatim — no translation."""
        monkeypatch.setenv(ENV_LLM_LITELLMROUTER_CONFIG, json.dumps(two_step_config))
        assert _parse_llm_router_config(ENV_LLM_LITELLMROUTER_CONFIG) == two_step_config

    def test_invalid_json(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_LITELLMROUTER_CONFIG, "{not json")
        with pytest.raises(ValueError, match="invalid JSON"):
            _parse_llm_router_config(ENV_LLM_LITELLMROUTER_CONFIG)

    def test_no_shape_validation(self, monkeypatch):
        """We don't validate the shape — anything that parses as JSON gets passed through.

        LiteLLM Router is authoritative for shape errors; we let them surface at
        Router construction time rather than pre-validating.
        """
        # A list, a string, an object with junk keys — all accepted by the parser.
        monkeypatch.setenv(ENV_LLM_LITELLMROUTER_CONFIG, json.dumps([{"hello": "world"}]))
        assert _parse_llm_router_config(ENV_LLM_LITELLMROUTER_CONFIG) == [{"hello": "world"}]
        monkeypatch.setenv(ENV_LLM_LITELLMROUTER_CONFIG, json.dumps({"only": "garbage"}))
        assert _parse_llm_router_config(ENV_LLM_LITELLMROUTER_CONFIG) == {"only": "garbage"}


class TestFromEnvLoadsConfig:
    def test_loaded_when_provider_is_litellmrouter(self, monkeypatch, two_step_config):
        monkeypatch.setenv(ENV_LLM_PROVIDER, "litellmrouter")
        monkeypatch.setenv(ENV_LLM_LITELLMROUTER_CONFIG, json.dumps(two_step_config))
        cfg = HindsightConfig.from_env()
        assert cfg.llm_provider == "litellmrouter"
        assert cfg.llm_litellmrouter_config == two_step_config

    def test_unset_keeps_default_provider(self, monkeypatch):
        monkeypatch.setenv(ENV_LLM_PROVIDER, "openai")
        monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "sk-primary")
        monkeypatch.delenv(ENV_LLM_LITELLMROUTER_CONFIG, raising=False)
        cfg = HindsightConfig.from_env()
        assert cfg.llm_provider == "openai"
        assert cfg.llm_litellmrouter_config is None

    def test_per_op_configs_independent(self, monkeypatch):
        """Per-op env vars populate per-op fields without touching the default."""
        retain_config = {
            "model_list": [{"model_name": "r", "litellm_params": {"model": "openai/retain", "api_key": "rk"}}]
        }
        reflect_config = {
            "model_list": [{"model_name": "f", "litellm_params": {"model": "anthropic/claude", "api_key": "ak"}}]
        }
        consol_config = {
            "model_list": [{"model_name": "c", "litellm_params": {"model": "openai/consol", "api_key": "ck"}}]
        }
        monkeypatch.setenv(ENV_LLM_PROVIDER, "openai")
        monkeypatch.setenv("HINDSIGHT_API_LLM_API_KEY", "sk-primary")
        monkeypatch.setenv(ENV_RETAIN_LLM_LITELLMROUTER_CONFIG, json.dumps(retain_config))
        monkeypatch.setenv(ENV_REFLECT_LLM_LITELLMROUTER_CONFIG, json.dumps(reflect_config))
        monkeypatch.setenv(ENV_CONSOLIDATION_LLM_LITELLMROUTER_CONFIG, json.dumps(consol_config))
        cfg = HindsightConfig.from_env()
        assert cfg.llm_litellmrouter_config is None
        assert cfg.retain_llm_litellmrouter_config == retain_config
        assert cfg.reflect_llm_litellmrouter_config == reflect_config
        assert cfg.consolidation_llm_litellmrouter_config == consol_config


# --- factory dispatch --------------------------------------------------------


class TestFactoryDispatch:
    def test_router_provider_requires_config(self):
        with pytest.raises(ValueError, match="config object"):
            create_llm_provider(
                provider="litellmrouter",
                api_key="",
                base_url="",
                model="unused",
                reasoning_effort="low",
                litellmrouter_config=None,
            )

    def test_router_provider_returns_router_impl(self, two_step_config):
        with patch.dict("sys.modules", {"litellm": MagicMock()}):
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
                    litellmrouter_config=two_step_config,
                )
                assert isinstance(impl, LiteLLMRouterLLM)
                _, kwargs = mock_init.call_args
                assert kwargs["config"] == two_step_config


# --- Router-backed call paths ------------------------------------------------


def _make_router_provider(config: dict[str, Any], mock_router: Any) -> LiteLLMRouterLLM:
    """Construct a LiteLLMRouterLLM with the inner Router replaced by a mock."""
    fake_litellm = MagicMock()
    fake_litellm.Router = MagicMock(return_value=mock_router)
    with patch.dict("sys.modules", {"litellm": fake_litellm}):
        # Bypass the heavy ctor chain by injecting state directly.
        provider = LiteLLMRouterLLM.__new__(LiteLLMRouterLLM)
        provider.provider = "litellmrouter"
        provider.api_key = ""
        provider.base_url = ""
        provider.model = "unused"
        provider.reasoning_effort = "low"
        provider.timeout = 300.0
        provider.config = config
        provider._litellm = fake_litellm
        provider._router = mock_router
        provider._router_output_cap = None  # tests that exercise the cap override this directly
        return provider


class TestRouterCall:
    @pytest.mark.asyncio
    async def test_plain_text_call_targets_default_entrypoint(self, two_step_config, mock_router_response):
        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=mock_router_response)
        provider = _make_router_provider(two_step_config, mock_router)

        result = await provider.call(
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=50,
            max_retries=0,
        )
        assert result == "ok"
        # Hindsight always issues against model_name="default"; Router handles fallback,
        # load-balancing, and routing strategy from there.
        kwargs = mock_router.acompletion.await_args.kwargs
        assert kwargs["model"] == "default"

    @pytest.mark.asyncio
    async def test_structured_output(self, two_step_config):
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
        provider = _make_router_provider(two_step_config, mock_router)

        result = await provider.call(
            messages=[{"role": "user", "content": "q"}],
            response_format=MySchema,
            max_retries=0,
        )
        assert isinstance(result, MySchema)
        assert result.answer == "42"

    @pytest.mark.asyncio
    async def test_retry_on_transient_then_success(self, two_step_config, mock_router_response):
        mock_router = MagicMock()
        # First call raises a 503-style error, second call returns ok.
        mock_router.acompletion = AsyncMock(side_effect=[Exception("503 Service Unavailable"), mock_router_response])
        provider = _make_router_provider(two_step_config, mock_router)

        result = await provider.call(
            messages=[{"role": "user", "content": "hi"}],
            max_retries=2,
            initial_backoff=0.0,
            max_backoff=0.0,
        )
        assert result == "ok"
        assert mock_router.acompletion.await_count == 2

    @pytest.mark.asyncio
    async def test_auth_error_does_not_retry(self, two_step_config):
        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(side_effect=Exception("401 Unauthorized: bad key"))
        provider = _make_router_provider(two_step_config, mock_router)

        with pytest.raises(Exception, match="401"):
            await provider.call(
                messages=[{"role": "user", "content": "hi"}],
                max_retries=5,
                initial_backoff=0.0,
            )
        assert mock_router.acompletion.await_count == 1

    @pytest.mark.asyncio
    async def test_caps_max_completion_tokens_to_litellm_registry(self, two_step_config, mock_router_response):
        """Cap max_completion_tokens to the most conservative deployment limit.

        Hindsight's defaults (e.g. retain_max_completion_tokens=64000) target
        high-capacity models. When a configured deployment has a smaller cap
        (gpt-4.1-nano = 32768), the call would otherwise be rejected — apply
        the cap silently using LiteLLM's per-model registry.
        """
        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=mock_router_response)
        provider = _make_router_provider(two_step_config, mock_router)
        provider._router_output_cap = 32768  # what _compute_router_output_cap would yield

        await provider.call(
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=64000,  # over the cap
            max_retries=0,
        )
        kwargs = mock_router.acompletion.await_args.kwargs
        assert kwargs["max_completion_tokens"] == 32768

    @pytest.mark.asyncio
    async def test_no_cap_when_litellm_registry_has_no_data(self, two_step_config, mock_router_response):
        """If LiteLLM doesn't know any of the deployment models, pass the requested value through."""
        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(return_value=mock_router_response)
        provider = _make_router_provider(two_step_config, mock_router)
        provider._router_output_cap = None

        await provider.call(
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=64000,
            max_retries=0,
        )
        kwargs = mock_router.acompletion.await_args.kwargs
        assert kwargs["max_completion_tokens"] == 64000

    @pytest.mark.asyncio
    async def test_call_with_tools(self, two_step_config):
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
        provider = _make_router_provider(two_step_config, mock_router)

        result = await provider.call_with_tools(
            messages=[{"role": "user", "content": "use tool"}],
            tools=[{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
            max_retries=0,
        )
        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "lookup"
        assert result.tool_calls[0].arguments == {"q": "x"}
