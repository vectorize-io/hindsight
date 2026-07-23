"""
Tests for HINDSIGHT_API_LLM_STRICT_SCHEMA / config.llm_strict_schema.

The flag asks every provider for its strongest structured-output mode so weaker
self-hosted instruction-followers can't wedge retain/consolidation by emitting
prose preambles, markdown ```json fences, or invalid JSON that fails parsing.

It is resolved once in ``LLMProvider.call``: an explicit per-call ``strict_schema``
wins in both directions, and ``None`` (the default) inherits the global flag. The
value is then passed down, so each provider honours it through its existing
``strict_schema`` handling:

- OpenAI-compatible / LiteLLM: ``response_format`` ``json_schema`` with ``strict: true``
- Gemini: already grammar-enforces its native ``response_schema`` (flag is a no-op)
- Providers without a strict mode ignore it.

Each operation (retain / reflect / consolidation) resolves its own per-operation
flag -- ``HINDSIGHT_API_LLM_STRICT_SCHEMA_<OP>``, falling back to the global env
var -- and passes it explicitly at the call site.

The batch retain path builds its request body directly (bypassing ``call``), so it
reads the same retain-scoped field rather than the global flag, keeping the batch
and streaming paths in agreement.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.config import (
    ENV_LLM_STRICT_SCHEMA,
    ENV_LLM_STRICT_SCHEMA_CONSOLIDATION,
    ENV_LLM_STRICT_SCHEMA_REFLECT,
    ENV_LLM_STRICT_SCHEMA_RETAIN,
    HindsightConfig,
)
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
# config: per-operation overrides
# --------------------------------------------------------------------------- #

# (per-operation env var, resolved config field) for every operation that makes
# structured-output LLM calls.
_OPERATIONS = [
    (ENV_LLM_STRICT_SCHEMA_RETAIN, "llm_strict_schema_retain"),
    (ENV_LLM_STRICT_SCHEMA_REFLECT, "llm_strict_schema_reflect"),
    (ENV_LLM_STRICT_SCHEMA_CONSOLIDATION, "llm_strict_schema_consolidation"),
]


@pytest.fixture
def clean_strict_env(monkeypatch):
    """Clear the global and every per-operation strict-schema env var."""
    monkeypatch.delenv(ENV_LLM_STRICT_SCHEMA, raising=False)
    for env, _ in _OPERATIONS:
        monkeypatch.delenv(env, raising=False)
    return monkeypatch


@pytest.mark.parametrize("env,field", _OPERATIONS)
def test_operation_strict_schema_inherits_global(clean_strict_env, env, field):
    clean_strict_env.setenv(ENV_LLM_STRICT_SCHEMA, "true")
    assert getattr(HindsightConfig.from_env(), field) is True


@pytest.mark.parametrize("env,field", _OPERATIONS)
def test_operation_strict_schema_defaults_off(clean_strict_env, env, field):
    assert getattr(HindsightConfig.from_env(), field) is False


@pytest.mark.parametrize("env,field", _OPERATIONS)
def test_operation_strict_schema_enables_without_global(clean_strict_env, env, field):
    """Turn strict schema on for one operation alone, leaving the others soft."""
    clean_strict_env.setenv(env, "true")
    config = HindsightConfig.from_env()
    assert getattr(config, field) is True
    assert config.llm_strict_schema is False
    # The other operations must not be dragged along.
    for other_env, other_field in _OPERATIONS:
        if other_env != env:
            assert getattr(config, other_field) is False


@pytest.mark.parametrize("env,field", _OPERATIONS)
def test_operation_strict_schema_opts_out_of_global(clean_strict_env, env, field):
    """The per-operation flag can also disable strict schema when the global one is on."""
    clean_strict_env.setenv(ENV_LLM_STRICT_SCHEMA, "true")
    clean_strict_env.setenv(env, "false")
    config = HindsightConfig.from_env()
    assert getattr(config, field) is False
    assert config.llm_strict_schema is True
    # ...without opting the other operations out too.
    for other_env, other_field in _OPERATIONS:
        if other_env != env:
            assert getattr(config, other_field) is True


# --------------------------------------------------------------------------- #
# wrapper: resolves the flag for every provider
# --------------------------------------------------------------------------- #


async def _strict_passed_to_provider(*, config_flag: bool, call_arg: bool | None) -> bool:
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
async def test_wrapper_inherits_config_flag_when_caller_has_no_preference():
    # Config flag on, caller passed nothing → provider still gets strict.
    assert await _strict_passed_to_provider(config_flag=True, call_arg=None) is True


@pytest.mark.asyncio
async def test_wrapper_off_by_default():
    assert await _strict_passed_to_provider(config_flag=False, call_arg=None) is False


@pytest.mark.asyncio
async def test_wrapper_per_call_override_still_works_with_flag_off():
    assert await _strict_passed_to_provider(config_flag=False, call_arg=True) is True


@pytest.mark.asyncio
async def test_wrapper_per_call_false_overrides_config_flag():
    """An explicit False opts a scope out even when the global flag is on.

    Regression: resolving with ``or`` made a per-call False indistinguishable from
    "unset", so a caller opting out was silently ignored.
    """
    assert await _strict_passed_to_provider(config_flag=True, call_arg=False) is False


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
# batch retain path: reads the retain-scoped config field directly
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strict", [True, False])
def test_batch_request_body_strict_follows_retain_config(strict):
    """The batch path must track llm_strict_schema_retain, not the global flag.

    Reading the global flag here would let the batch and streaming retain paths
    disagree whenever the retain override diverges from it.
    """
    from hindsight_api.engine.retain.fact_extraction import _build_request_body

    llm_config = SimpleNamespace(model="gpt-4o-mini", provider="openai", _provider_impl=SimpleNamespace())
    config = SimpleNamespace(
        retain_max_completion_tokens=None,
        # Global deliberately set opposite to the retain override: if the batch path
        # regressed to reading llm_strict_schema, this test would catch it.
        llm_strict_schema=not strict,
        llm_strict_schema_retain=strict,
        llm_temperature_retain=None,
    )
    # provider != "openai" service-tier branch skipped via _provider_impl without attr
    llm_config._provider_impl.openai_service_tier = None

    body = _build_request_body(llm_config, config, "system prompt", "user message", _Resp)
    assert body["response_format"]["json_schema"]["strict"] is strict


# --------------------------------------------------------------------------- #
# call sites: each operation threads its own resolved flag
# --------------------------------------------------------------------------- #


def _retain_config(strict_retain: bool):
    """Minimal resolved retain config (mirrors test_fact_extraction_retry._make_config).

    A bank-resolved config, not the global one -- retain_extraction_mode and friends
    are bank-configurable and raise if read off global config.
    """
    from hindsight_api.config import HindsightConfig

    cfg = MagicMock(spec=HindsightConfig)
    cfg.retain_llm_max_retries = 1
    cfg.llm_max_retries = 1
    cfg.retain_llm_initial_backoff = 0.0
    cfg.llm_initial_backoff = 0.0
    cfg.retain_llm_max_backoff = 0.0
    cfg.llm_max_backoff = 0.0
    cfg.retain_max_completion_tokens = 8192
    cfg.retain_extraction_mode = "concise"
    cfg.retain_extract_causal_links = False
    cfg.retain_mission = None
    cfg.llm_temperature_retain = 0.1
    cfg.llm_strict_schema_retain = strict_retain
    return cfg


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [True, False])
async def test_retain_chunk_extraction_threads_retain_flag(strict):
    """The streaming retain path passes llm_strict_schema_retain to call().

    Asserted for both values: passing it only when True would let the global flag
    win on the False side, defeating the per-operation opt-out.
    """
    from hindsight_api.engine.llm_wrapper import LLMProvider
    from hindsight_api.engine.retain.fact_extraction import _extract_facts_from_chunk

    llm_config = MagicMock(spec=LLMProvider)
    llm_config.provider = "mock"
    llm_config._provider_impl = SimpleNamespace(supports_prompt_caching=lambda: False)
    token_usage = MagicMock()
    token_usage.__add__ = lambda self, other: self
    llm_config.call = AsyncMock(return_value=({"facts": []}, token_usage))

    await _extract_facts_from_chunk(
        chunk="some text",
        chunk_index=0,
        total_chunks=1,
        event_date=None,
        context="",
        llm_config=llm_config,
        config=_retain_config(strict_retain=strict),
    )
    assert llm_config.call.call_args.kwargs["strict_schema"] is strict


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [True, False])
async def test_reflect_structured_output_threads_reflect_flag(strict):
    """The reflect structured-output call passes llm_strict_schema_reflect to call()."""
    from hindsight_api.engine.reflect.agent import _generate_structured_output

    llm_config = MagicMock(spec=LLMProvider)
    llm_config.provider = "mock"
    llm_config.call = AsyncMock(return_value=({"answer": "x"}, MagicMock()))

    with patch(
        "hindsight_api.engine.reflect.agent.get_config",
        lambda: SimpleNamespace(llm_strict_schema_reflect=strict),
    ):
        await _generate_structured_output(
            answer="the answer",
            response_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
            llm_config=llm_config,
            reflect_id="r1",
        )
    assert llm_config.call.call_args.kwargs["strict_schema"] is strict
