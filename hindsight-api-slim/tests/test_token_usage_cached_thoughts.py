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

import pytest

from hindsight_api.engine.reflect.agent import _generate_structured_output
from hindsight_api.engine.reflect.models import StructuredOutputResult, TokenUsageSummary
from hindsight_api.engine.response_models import LLMToolCallResult, TokenUsage
from hindsight_api.extensions.operation_validator import RetainResult


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
