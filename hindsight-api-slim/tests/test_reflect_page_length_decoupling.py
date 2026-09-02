"""Reflect/mental-model ``max_tokens`` is a page-length target, not a transport cap (#3365).

On thinking models the provider's output budget is consumed by reasoning tokens,
so passing the page ``max_tokens`` straight through as ``max_completion_tokens``
truncates the visible answer mid-word. These tests pin the decoupled contract:

- ``build_final_prompt`` communicates the length as a prompt directive.
- ``reflect_max_completion_tokens`` config is uncapped (None) by default and
  overridable via env.
- The forced synthesis passes the *config* cap (None by default), never the page
  budget.
- ``GeminiLLM`` logs a warning instead of silently returning MAX_TOKENS-truncated
  text.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.config import HindsightConfig, clear_config_cache
from hindsight_api.engine.reflect.agent import run_reflect_agent
from hindsight_api.engine.reflect.prompts import build_final_prompt

BANK = {"name": "TestBank", "mission": "Testing"}


# --------------------------------------------------------------------------- #
# Prompt directive
# --------------------------------------------------------------------------- #
def test_build_final_prompt_includes_length_directive():
    prompt = build_final_prompt("q", [], BANK, max_tokens=3072)
    assert "## Length" in prompt
    assert "approximately 3072 tokens" in prompt
    # The whole point: steer toward a clean stop, never a hard mid-word cut.
    assert "NEVER stop" in prompt


def test_build_final_prompt_omits_length_directive_when_unset():
    prompt = build_final_prompt("q", [], BANK)
    assert "## Length" not in prompt


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_reflect_max_completion_tokens_defaults_to_none(monkeypatch):
    monkeypatch.delenv("HINDSIGHT_API_REFLECT_MAX_COMPLETION_TOKENS", raising=False)
    config = HindsightConfig.from_env()
    assert config.reflect_max_completion_tokens is None


def test_reflect_max_completion_tokens_env_override(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_REFLECT_MAX_COMPLETION_TOKENS", "16000")
    config = HindsightConfig.from_env()
    assert config.reflect_max_completion_tokens == 16000


# --------------------------------------------------------------------------- #
# Forced synthesis honors the config cap, not the page budget
# --------------------------------------------------------------------------- #
def _mock_llm(final_answer: str = "Synthesized final answer."):
    from hindsight_api.engine.response_models import TokenUsage

    llm = MagicMock()
    llm.provider = "gemini"
    llm.model = "gemini-2.5-flash"
    llm.call = AsyncMock(return_value=(final_answer, TokenUsage(input_tokens=40, output_tokens=12, total_tokens=52)))
    return llm


def _mock_functions():
    return {
        "search_mental_models_fn": AsyncMock(
            return_value={"mental_models": [{"id": "mm-1", "name": "Prefs", "content": "Fresh.", "is_stale": False}]}
        ),
        "search_observations_fn": AsyncMock(return_value={"observations": []}),
        "recall_fn": AsyncMock(return_value={"memories": []}),
        "expand_fn": AsyncMock(return_value={"memories": []}),
    }


def _stop_after_evidence(llm):
    """Turn 0 tool-calls, turn 1 stops with plain text -> forced final synthesis."""
    from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult

    llm.call_with_tools = AsyncMock(
        side_effect=[
            LLMToolCallResult(
                tool_calls=[LLMToolCall(id="1", name="search_mental_models", arguments={"query": "q"})],
                finish_reason="tool_calls",
            ),
            LLMToolCallResult(tool_calls=[], content="I have enough.", finish_reason="stop"),
        ]
    )


@pytest.mark.asyncio
async def test_forced_synthesis_uncapped_by_default(monkeypatch):
    monkeypatch.delenv("HINDSIGHT_API_REFLECT_MAX_COMPLETION_TOKENS", raising=False)
    clear_config_cache()
    try:
        llm = _mock_llm()
        _stop_after_evidence(llm)
        result = await run_reflect_agent(
            llm_config=llm,
            bank_id="b",
            query="q",
            bank_profile=BANK,
            has_mental_models=True,
            budget="low",
            max_tokens=64,
            **_mock_functions(),
        )
        assert result.text == "Synthesized final answer."
        # Uncapped transport; page length reached the model via the prompt.
        assert llm.call.await_args.kwargs["max_completion_tokens"] is None
        assert "approximately 64 tokens" in llm.call.await_args.kwargs["messages"][1]["content"]
    finally:
        clear_config_cache()


@pytest.mark.asyncio
async def test_forced_synthesis_uses_config_cap_when_set(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_REFLECT_MAX_COMPLETION_TOKENS", "12345")
    clear_config_cache()
    try:
        llm = _mock_llm()
        _stop_after_evidence(llm)
        await run_reflect_agent(
            llm_config=llm,
            bank_id="b",
            query="q",
            bank_profile=BANK,
            has_mental_models=True,
            budget="low",
            max_tokens=64,
            **_mock_functions(),
        )
        # The transport cap is the operator-set ceiling, still independent of the
        # page budget (64).
        assert llm.call.await_args.kwargs["max_completion_tokens"] == 12345
    finally:
        clear_config_cache()


# --------------------------------------------------------------------------- #
# Gemini surfaces MAX_TOKENS truncation instead of a silent success
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_gemini_warns_on_max_tokens_truncation(caplog):
    pytest.importorskip("google.genai")
    from hindsight_api.engine.llm_wrapper import LLMConfig

    with patch("google.genai.Client", return_value=MagicMock()):
        llm = LLMConfig(provider="gemini", api_key="fake-key", base_url="", model="gemini-2.5-flash")

    response = MagicMock()
    response.text = "A page that was cut off mid-wor"
    candidate = MagicMock()
    candidate.finish_reason = "FinishReason.MAX_TOKENS"
    response.candidates = [candidate]
    response.usage_metadata = MagicMock(
        prompt_token_count=10, candidates_token_count=20, cached_content_token_count=0, thoughts_token_count=480
    )
    llm._provider_impl._client.aio.models.generate_content = AsyncMock(return_value=response)

    with caplog.at_level(logging.WARNING):
        out = await llm._provider_impl.call([{"role": "user", "content": "write a page"}], max_completion_tokens=500)

    assert out == "A page that was cut off mid-wor"
    assert any("truncated at max_output_tokens" in r.message for r in caplog.records)
