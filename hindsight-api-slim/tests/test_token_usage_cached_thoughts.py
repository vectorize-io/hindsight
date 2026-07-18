"""Tests for cached / thoughts token propagation through TokenUsage,
LLMToolCallResult, TokenUsageSummary, and RetainResult.

The Gemini 2.5+ family (and any future provider with prompt caching +
reasoning tokens) reports four distinct token counts on every response:
prompt, candidates (visible output), cached_content, and thoughts. The
last two are billed separately by the provider but were previously not
threaded through to downstream return contexts, so application-layer
metering had no way to attribute prompt-cache hit rate or reasoning cost
per operation.

These tests pin the propagation: when a provider populates cached or
thoughts on the way out, every accumulator and aggregate type carries
the value through unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM
from hindsight_api.engine.reflect.agent import _generate_structured_output
from hindsight_api.engine.reflect.models import StructuredOutputResult, TokenUsageSummary
from hindsight_api.engine.response_models import LLMToolCallResult, TokenUsage
from hindsight_api.extensions.operation_validator import RetainResult
from hindsight_api.metrics import MetricsCollector


class _OkModel(BaseModel):
    ok: bool


def _openai_llm() -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        provider="openai",
        api_key="test-key",
        base_url="https://example.test/v1",
        model="gpt-4o-mini",
    )


def test_token_usage_carries_cached_and_thoughts():
    """TokenUsage defaults both new fields to 0 and accepts non-zero values."""
    u = TokenUsage(input_tokens=1500, output_tokens=500, total_tokens=2000)
    assert u.cached_tokens == 0
    assert u.thoughts_tokens == 0

    u = TokenUsage(
        input_tokens=1500,
        output_tokens=500,
        total_tokens=2000,
        cached_tokens=200,
        thoughts_tokens=80,
    )
    assert u.cached_tokens == 200
    assert u.thoughts_tokens == 80


def test_token_usage_aggregates_thoughts_tokens():
    """TokenUsage.__add__ sums thoughts_tokens alongside the existing fields.

    Multi-iteration agentic loops accumulate per-call usage via ``+``. If
    thoughts_tokens isn't summed, the per-op total undercounts reasoning
    spend by a factor of N (the number of LLM sub-calls).
    """
    a = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15, cached_tokens=2, thoughts_tokens=7)
    b = TokenUsage(input_tokens=20, output_tokens=8, total_tokens=28, cached_tokens=3, thoughts_tokens=11)
    c = a + b
    assert c.input_tokens == 30
    assert c.output_tokens == 13
    assert c.total_tokens == 43
    assert c.cached_tokens == 5
    assert c.thoughts_tokens == 18


def test_llm_tool_call_result_carries_cached_and_thoughts():
    """call_with_tools returns LLMToolCallResult — both new fields default to 0
    and accept non-zero values from the provider."""
    r = LLMToolCallResult(content="ok", input_tokens=1234, output_tokens=56)
    assert r.cached_tokens == 0
    assert r.thoughts_tokens == 0

    r = LLMToolCallResult(
        content="ok",
        input_tokens=1234,
        output_tokens=56,
        cached_tokens=200,
        thoughts_tokens=78,
    )
    assert r.cached_tokens == 200
    assert r.thoughts_tokens == 78


def test_token_usage_summary_carries_cached_and_thoughts():
    """TokenUsageSummary is what reflect agent returns to its caller — needs
    to propagate the aggregate so per-op cost attribution works."""
    s = TokenUsageSummary(
        input_tokens=10000,
        output_tokens=200,
        total_tokens=10200,
        cached_tokens=3000,
        thoughts_tokens=150,
    )
    assert s.cached_tokens == 3000
    assert s.thoughts_tokens == 150


def test_token_usage_summary_defaults_cached_and_thoughts_to_zero():
    """Defaults preserve backward compatibility for callers built before the
    fields existed."""
    s = TokenUsageSummary(input_tokens=100, output_tokens=50, total_tokens=150)
    assert s.cached_tokens == 0
    assert s.thoughts_tokens == 0


def test_retain_result_carries_cached_input_and_thoughts():
    """RetainResult is the contract between the engine and any metering
    extension. The two new fields are optional (None) so older extensions
    that don't read them are unaffected; engines that DO populate them get
    end-to-end attribution into the metering hook."""

    class _Ctx:
        pass

    r = RetainResult(
        bank_id="b",
        contents=[],
        request_context=_Ctx(),
        document_id=None,
        fact_type_override=None,
        unit_ids=[],
        llm_input_tokens=1000,
        llm_output_tokens=50,
        llm_total_tokens=1050,
        llm_cached_input_tokens=300,
        llm_thoughts_tokens=25,
    )
    assert r.llm_cached_input_tokens == 300
    assert r.llm_thoughts_tokens == 25

    # Defaults stay None for engines that don't surface the data, so
    # downstream extensions can use ``or 0`` without breaking on a
    # core-only build.
    r2 = RetainResult(
        bank_id="b",
        contents=[],
        request_context=_Ctx(),
        document_id=None,
        fact_type_override=None,
        unit_ids=[],
    )
    assert r2.llm_cached_input_tokens is None
    assert r2.llm_thoughts_tokens is None


@pytest.mark.asyncio
async def test_generate_structured_output_returns_dataclass_on_no_fields():
    """_generate_structured_output returns a StructuredOutputResult, not a tuple.

    Regression guard: the function and all six call sites must agree on a single
    return type. A previous tuple-based contract drifted out of sync (the failure
    branch returned 3 values while callers unpacked 5), which would crash reflect
    with a ValueError on any structured-output failure. An empty schema exercises
    the no-LLM-call branch deterministically.
    """
    result = await _generate_structured_output(
        answer="anything",
        response_schema={},
        llm_config=None,
        reflect_id="test",
    )
    assert isinstance(result, StructuredOutputResult)
    assert result.structured_output is None
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.cached_tokens == 0
    assert result.thoughts_tokens == 0


# --- Provider-level extraction (follow-up to #2356) -------------------------
# The model-level tests above pin that the types CARRY the new fields. These
# pin that the OpenAI-compatible backend (the most-used provider: OpenAI
# o-series/gpt-5, groq, deepseek-r1, plus NousLLM/FireworksLLM subclasses)
# actually READS usage.completion_tokens_details.reasoning_tokens and surfaces
# it. Before this fix it never did, so thoughts_tokens was silently 0 for every
# OpenAI-compatible reasoning model and the #2356 cost-attribution field was
# wrong on the dominant code path.


def _usage(*, cached=0, reasoning=0, prompt=1500, completion=500, total=2000):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached) if cached is not None else None,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning) if reasoning is not None else None,
    )


def _response(*, usage, content='{"ok": true}', tool_calls=None):
    choice = SimpleNamespace(
        finish_reason="stop",
        message=SimpleNamespace(content=content, tool_calls=tool_calls, refusal=None),
    )
    return SimpleNamespace(error=None, usage=usage, choices=[choice])


@pytest.mark.asyncio
async def test_openai_compatible_call_extracts_reasoning_into_thoughts_tokens():
    llm = _openai_llm()
    llm._client.chat.completions.create = AsyncMock(return_value=_response(usage=_usage(cached=200, reasoning=80)))
    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        _, token_usage = await llm.call(
            messages=[{"role": "user", "content": "Return whether this worked."}],
            response_format=_OkModel,
            max_retries=0,
            return_usage=True,
        )
    # reasoning_tokens must flow into thoughts_tokens (was dropped -> 0 before).
    assert token_usage.thoughts_tokens == 80
    assert token_usage.cached_tokens == 200
    # OpenAI folds reasoning into completion_tokens; output_tokens/total_tokens
    # must stay visible-only so they don't double-count thoughts_tokens.
    assert token_usage.output_tokens == 500 - 80
    assert token_usage.total_tokens == 2000 - 80


@pytest.mark.asyncio
async def test_openai_compatible_call_with_tools_extracts_reasoning_and_cached():
    llm = _openai_llm()
    llm._client.chat.completions.create = AsyncMock(
        return_value=_response(usage=_usage(cached=64, reasoning=33), content="done")
    )
    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        result = await llm.call_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            max_retries=0,
        )
    # call_with_tools previously set neither field -> both defaulted to 0.
    assert result.thoughts_tokens == 33
    assert result.cached_tokens == 64
    # output_tokens is visible-only (completion_tokens minus reasoning).
    assert result.output_tokens == 500 - 33


@pytest.mark.asyncio
async def test_openai_compatible_call_no_token_details_keeps_thoughts_zero():
    """Non-reasoning models / Ollama report no *_details — the 0-safe getattr
    chain must leave thoughts_tokens (and cached_tokens) at 0, not raise."""
    llm = _openai_llm()
    llm._client.chat.completions.create = AsyncMock(
        return_value=_response(usage=_usage(cached=None, reasoning=None, prompt=10, completion=5, total=15))
    )
    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        _, token_usage = await llm.call(
            messages=[{"role": "user", "content": "x"}],
            response_format=_OkModel,
            max_retries=0,
            return_usage=True,
        )
    assert token_usage.thoughts_tokens == 0
    assert token_usage.cached_tokens == 0


@pytest.mark.asyncio
async def test_openai_compatible_output_tokens_exclude_thoughts_like_gemini():
    """Convention invariant: for an OpenAI-compatible reasoning model, the
    provider's ``completion_tokens`` already INCLUDES ``reasoning_tokens`` (real
    o4-mini sample: completion_tokens=83, reasoning_tokens=64). The TokenUsage
    contract — matching the Gemini provider — treats ``output_tokens`` as
    visible-only and surfaces reasoning separately in ``thoughts_tokens``, with
    ``total_tokens == input_tokens + output_tokens``. Pin that the two fields do
    not double-count: visible = completion - reasoning, and the three token
    counts reconcile."""
    llm = _openai_llm()
    llm._client.chat.completions.create = AsyncMock(
        return_value=_response(usage=_usage(reasoning=64, prompt=20, completion=83, total=103))
    )
    with patch("hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector"):
        _, token_usage = await llm.call(
            messages=[{"role": "user", "content": "17*23?"}],
            response_format=_OkModel,
            max_retries=0,
            return_usage=True,
        )
    assert token_usage.thoughts_tokens == 64
    assert token_usage.output_tokens == 83 - 64  # visible-only, no reasoning
    assert token_usage.input_tokens == 20
    assert token_usage.total_tokens == token_usage.input_tokens + token_usage.output_tokens
    # The reasoning tokens live in exactly one field, not both.
    assert token_usage.output_tokens + token_usage.thoughts_tokens == 83


def _recorded_llm_call(collector):
    """The kwargs of the single record_llm_call the provider made."""
    assert collector.record_llm_call.call_count == 1
    return collector.record_llm_call.call_args.kwargs


@pytest.mark.asyncio
async def test_openai_compatible_call_records_cached_and_thoughts_on_metrics():
    """call() must hand cached/thoughts to the metrics collector, not only to
    TokenUsage. The collector already accepts both kwargs and keeps a counter for
    each; the provider simply never passed them, so hindsight.llm.tokens.thoughts
    and hindsight.llm.tokens.cached_input stayed empty for every OpenAI-compatible
    reasoning model."""
    llm = _openai_llm()
    llm._client.chat.completions.create = AsyncMock(return_value=_response(usage=_usage(cached=200, reasoning=80)))
    collector = MagicMock(spec=MetricsCollector)
    with patch(
        "hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector",
        return_value=collector,
    ):
        await llm.call(
            messages=[{"role": "user", "content": "Return whether this worked."}],
            response_format=_OkModel,
            max_retries=0,
            return_usage=True,
        )
    recorded = _recorded_llm_call(collector)
    assert recorded["cached_input_tokens"] == 200
    assert recorded["thoughts_tokens"] == 80


@pytest.mark.asyncio
async def test_openai_compatible_call_with_tools_records_cached_and_thoughts_on_metrics():
    """call_with_tools() extracts both counts for its own return value, so it must
    record them too — the tool-calling path is where the agentic loop spends most
    of its reasoning tokens."""
    llm = _openai_llm()
    llm._client.chat.completions.create = AsyncMock(
        return_value=_response(usage=_usage(cached=64, reasoning=33), content="done")
    )
    collector = MagicMock(spec=MetricsCollector)
    with patch(
        "hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector",
        return_value=collector,
    ):
        await llm.call_with_tools(messages=[{"role": "user", "content": "hi"}], tools=[], max_retries=0)
    recorded = _recorded_llm_call(collector)
    assert recorded["cached_input_tokens"] == 64
    assert recorded["thoughts_tokens"] == 33


@pytest.mark.asyncio
async def test_openai_compatible_metrics_account_for_every_billed_output_token():
    """Accounting invariant: every token the provider bills as output lands on
    exactly one counter, never zero and never two.

    output_tokens is made visible-only just above the record_llm_call (reasoning
    subtracted so it doesn't double-count against thoughts_tokens). That subtraction
    only holds the books straight if thoughts_tokens is recorded as well: recorded
    output + recorded thoughts must add back up to the provider's completion_tokens.
    """
    completion, reasoning = 83, 64
    llm = _openai_llm()
    llm._client.chat.completions.create = AsyncMock(
        return_value=_response(usage=_usage(reasoning=reasoning, prompt=20, completion=completion, total=103))
    )
    collector = MagicMock(spec=MetricsCollector)
    with patch(
        "hindsight_api.engine.providers.openai_compatible_llm.get_metrics_collector",
        return_value=collector,
    ):
        await llm.call(
            messages=[{"role": "user", "content": "17*23?"}],
            response_format=_OkModel,
            max_retries=0,
            return_usage=True,
        )
    recorded = _recorded_llm_call(collector)
    assert recorded["output_tokens"] == completion - reasoning  # visible-only
    assert recorded["thoughts_tokens"] == reasoning
    assert recorded["output_tokens"] + recorded["thoughts_tokens"] == completion
