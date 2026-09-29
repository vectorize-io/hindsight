"""The GenAI span carries the usage counts providers hand to the recorder.

``LLMSpanRecorder.record_llm_call`` absorbs unknown kwargs into ``**_extra`` so
recorders can diverge, which also means a count nobody declared as a parameter
is silently dropped. Reasoning tokens were dropped that way, so pin every usage
attribute the providers pass.
"""

from unittest.mock import MagicMock

from hindsight_api.tracing import LLMSpanRecorder


def _record(**usage: int) -> dict[str, object]:
    span = MagicMock()
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__.return_value = span
    LLMSpanRecorder(tracer).record_llm_call(
        provider="openai",
        model="qwen",
        scope="retain_extract_facts",
        messages=[{"role": "user", "content": "extract facts"}],
        response_content="fact",
        duration=0.1,
        **usage,
    )
    return {call.args[0]: call.args[1] for call in span.set_attribute.call_args_list}


def test_reported_usage_reaches_the_span():
    attrs = _record(input_tokens=100, output_tokens=20, cached_tokens=30, thoughts_tokens=60)

    assert attrs["gen_ai.usage.input_tokens"] == 100
    assert attrs["gen_ai.usage.output_tokens"] == 20
    assert attrs["gen_ai.usage.cached_tokens"] == 30
    assert attrs["gen_ai.usage.reasoning_tokens"] == 60


def test_unreported_usage_adds_no_attribute():
    attrs = _record(input_tokens=100, output_tokens=20)

    assert "gen_ai.usage.cached_tokens" not in attrs
    assert "gen_ai.usage.reasoning_tokens" not in attrs
