"""Tests for the Gemini per-call audit log and thoughts_tokens plumbing.

The audit log is the diagnostic instrument we use to attribute LLM spend
back to a specific code path when the application-layer metering shows a
gap vs provider billing. These tests pin the audit-log shape so a refactor
doesn't silently drop fields that an external reconciliation pipeline is
relying on.
"""

from __future__ import annotations

import json
import logging

from hindsight_api.engine.providers.gemini_llm import (
    _emit_call_audit,
    _truncated_caller_stack,
)
from hindsight_api.engine.response_models import LLMToolCallResult, TokenUsage


def test_token_usage_aggregates_thoughts_tokens():
    """Aggregating two TokenUsage entries sums thoughts_tokens alongside the others.

    The reflect agent and retain orchestrator both accumulate per-call
    usage via ``+=``. If thoughts_tokens is not summed, the per-op total
    will undercount reasoning spend by a factor of N (where N is the
    number of LLM sub-calls per op).
    """
    a = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15, cached_tokens=2, thoughts_tokens=7)
    b = TokenUsage(input_tokens=20, output_tokens=8, total_tokens=28, cached_tokens=3, thoughts_tokens=11)
    c = a + b
    assert c.input_tokens == 30
    assert c.output_tokens == 13
    assert c.total_tokens == 43
    assert c.cached_tokens == 5
    assert c.thoughts_tokens == 18


def test_token_usage_thoughts_defaults_zero():
    """Existing callers that construct TokenUsage without thoughts still work."""
    u = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
    assert u.thoughts_tokens == 0


def test_llm_tool_call_result_carries_thoughts_and_cached():
    """call_with_tools returns LLMToolCallResult; downstream reflect-agent
    aggregation needs both new fields visible at this layer."""
    r = LLMToolCallResult(
        content="ok",
        input_tokens=1234,
        output_tokens=56,
        cached_tokens=200,
        thoughts_tokens=78,
    )
    assert r.cached_tokens == 200
    assert r.thoughts_tokens == 78


def test_truncated_caller_stack_filters_stdlib_and_packages():
    """The stack must only include project frames, capped at max_frames."""
    frames = _truncated_caller_stack(max_frames=4)
    # All returned entries follow ``path:lineno:func`` shape.
    for f in frames:
        parts = f.rsplit(":", 2)
        assert len(parts) == 3
        assert parts[1].isdigit()
    # No stdlib or 3p frames leaked through.
    for f in frames:
        assert "/site-packages/" not in f
        assert "/lib/python" not in f
    # Cap honored.
    assert len(frames) <= 4


def test_emit_call_audit_writes_valid_json_with_all_fields(caplog):
    """The audit log line must be JSON-parseable and contain every field
    the reconciliation script expects. If a future refactor renames or
    drops a field, this test fails before the rename ships."""
    caplog.set_level(logging.INFO, logger="hindsight.llm.gemini.calls")
    _emit_call_audit(
        provider="gemini",
        model="gemini-3.1-flash-lite",
        scope="test_scope",
        input_tokens=1500,
        cached_input_tokens=300,
        output_tokens=100,
        thoughts_tokens=50,
        duration_ms=420,
        finish_reason="stop",
    )

    audit_records = [r for r in caplog.records if r.name == "hindsight.llm.gemini.calls"]
    assert len(audit_records) == 1
    payload = json.loads(audit_records[0].message)

    expected_keys = {
        "provider",
        "model",
        "scope",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "thoughts_tokens",
        "duration_ms",
        "finish_reason",
        "caller_stack",
    }
    assert set(payload.keys()) == expected_keys
    assert payload["input_tokens"] == 1500
    assert payload["cached_input_tokens"] == 300
    assert payload["output_tokens"] == 100
    assert payload["thoughts_tokens"] == 50
    assert payload["duration_ms"] == 420
    assert isinstance(payload["caller_stack"], list)


def test_emit_call_audit_never_raises_on_internal_error(monkeypatch, caplog):
    """Audit emission must NEVER fail the request path. If the logger
    itself raises (e.g. log handler corruption), we swallow silently."""

    class _BoomLogger:
        def info(self, *_args, **_kwargs):
            raise RuntimeError("log backend down")

    monkeypatch.setattr(
        "hindsight_api.engine.providers.gemini_llm._call_audit_logger",
        _BoomLogger(),
    )

    # Must not raise.
    _emit_call_audit(
        provider="gemini",
        model="gemini-3.1-flash-lite",
        scope=None,
        input_tokens=1,
        cached_input_tokens=0,
        output_tokens=1,
        thoughts_tokens=0,
        duration_ms=1,
        finish_reason="stop",
    )
