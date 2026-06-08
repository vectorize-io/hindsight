"""
Tests for HINDSIGHT_API_LLM_STRICT_SCHEMA / config.llm_strict_schema.

The flag asks every provider for its strongest structured-output mode so weaker
self-hosted instruction-followers can't wedge retain/consolidation by emitting
prose preambles, markdown ```json fences, or invalid JSON that fails parsing.

It is resolved once in ``LLMProvider.call`` (OR-ed with the per-call
``strict_schema`` argument) and passed down, so each provider honours it through
its existing ``strict_schema`` handling:

- OpenAI-compatible / LiteLLM: ``response_format`` ``json_schema`` with ``strict: true``
- Gemini: already grammar-enforces its native ``response_schema`` (flag is a no-op)
- Providers without a strict mode ignore it.

The batch retain path builds its request body directly (bypassing ``call``), so
it reads the config flag itself.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.config import ENV_LLM_STRICT_SCHEMA, HindsightConfig
from hindsight_api.engine.llm_wrapper import LLMProvider
from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM


class _Resp(BaseModel):
    ok: bool


def _config_with(strict: bool) -> object:
    """A config proxy that overrides only llm_strict_schema (avoids recursion)."""
    from hindsight_api.config import get_config

    real = get_config()

    class _Cfg:
        llm_strict_schema = strict

        def __getattr__(self, name):
            return getattr(real, name)

    return _Cfg()


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_env_var_enables_strict_schema(monkeypatch):
    monkeypatch.setenv(ENV_LLM_STRICT_SCHEMA, "true")
    assert HindsightConfig.from_env().llm_strict_schema is True


def test_strict_schema_defaults_off(monkeypatch):
    monkeypatch.delenv(ENV_LLM_STRICT_SCHEMA, raising=False)
    assert HindsightConfig.from_env().llm_strict_schema is False


# --------------------------------------------------------------------------- #
# wrapper: resolves the flag for every provider
# --------------------------------------------------------------------------- #


async def _strict_passed_to_provider(*, config_flag: bool, call_arg: bool) -> bool:
    """Return the strict_schema value the wrapper forwards to the provider impl."""
    llm = LLMProvider(provider="anthropic", api_key="test-key", base_url="", model="claude-x")
    impl = SimpleNamespace(call=AsyncMock(return_value=_Resp(ok=True)))
    llm._provider_impl = impl

    cfg = _config_with(config_flag)  # build before patching to avoid get_config recursion
    with patch("hindsight_api.config.get_config", lambda: cfg):
        await llm.call(
            messages=[{"role": "user", "content": "hi"}],
            response_format=_Resp,
            strict_schema=call_arg,
            max_retries=0,
        )
    return impl.call.call_args.kwargs["strict_schema"]


@pytest.mark.asyncio
async def test_wrapper_ors_in_config_flag():
    # Config flag on, caller didn't ask → provider still gets strict.
    assert await _strict_passed_to_provider(config_flag=True, call_arg=False) is True


@pytest.mark.asyncio
async def test_wrapper_off_by_default():
    assert await _strict_passed_to_provider(config_flag=False, call_arg=False) is False


@pytest.mark.asyncio
async def test_wrapper_per_call_override_still_works_with_flag_off():
    assert await _strict_passed_to_provider(config_flag=False, call_arg=True) is True


# --------------------------------------------------------------------------- #
# openai-compatible: strict_schema -> json_schema, else json_object
# --------------------------------------------------------------------------- #


def _openai_response(content: str = '{"ok": true}'):
    choice = SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content=content, tool_calls=None, refusal=None)
    )
    return SimpleNamespace(error=None, usage=None, choices=[choice])


async def _openai_response_format(*, strict: bool):
    llm = OpenAICompatibleLLM(
        provider="openai", api_key="test-key", base_url="https://example.test/v1", model="gpt-4o-mini"
    )
    create = AsyncMock(return_value=_openai_response())
    llm._client.chat.completions.create = create
    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        await llm.call(
            messages=[{"role": "user", "content": "Return whether this worked."}],
            response_format=_Resp,
            strict_schema=strict,
            max_retries=0,
        )
    return create.call_args.kwargs.get("response_format")


@pytest.mark.asyncio
async def test_openai_strict_uses_json_schema():
    rf = await _openai_response_format(strict=True)
    assert rf is not None and rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True


@pytest.mark.asyncio
async def test_openai_soft_uses_json_object():
    rf = await _openai_response_format(strict=False)
    assert rf is not None and rf["type"] == "json_object"


# --------------------------------------------------------------------------- #
# litellm: strict_schema -> response_format strict flag
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.asyncio
async def test_litellm_response_format_strict_follows_arg(strict):
    from hindsight_api.engine.providers.litellm_llm import LiteLLMLLM

    llm = LiteLLMLLM(provider="litellm", api_key="test-key", base_url="", model="gpt-4o-mini")
    response = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content='{"ok": true}'))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )
    acompletion = AsyncMock(return_value=response)
    with (
        patch.object(llm, "_acompletion", acompletion),
        patch("hindsight_api.engine.providers.litellm_llm.get_metrics_collector"),
    ):
        await llm.call(
            messages=[{"role": "user", "content": "Return whether this worked."}],
            response_format=_Resp,
            strict_schema=strict,
            max_retries=0,
        )
    rf = acompletion.call_args.kwargs["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is strict


# --------------------------------------------------------------------------- #
# batch retain path: reads the config flag directly
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strict", [True, False])
def test_batch_request_body_strict_follows_config(strict):
    from hindsight_api.engine.retain.fact_extraction import _build_request_body

    llm_config = SimpleNamespace(model="gpt-4o-mini", provider="openai", _provider_impl=SimpleNamespace())
    config = SimpleNamespace(retain_max_completion_tokens=None, llm_strict_schema=strict)
    # provider != "openai" service-tier branch skipped via _provider_impl without attr
    llm_config._provider_impl.openai_service_tier = None

    body = _build_request_body(llm_config, config, "system prompt", "user message", _Resp)
    assert body["response_format"]["json_schema"]["strict"] is strict
