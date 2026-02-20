"""
Unit tests for OpenTelemetry tracing instrumentation.

Tests the tracing module's ability to record LLM calls with GenAI semantic conventions.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from hindsight_api.tracing import (
    PROVIDER_NAME_MAPPING,
    GenAIAttributes,
    LLMSpanRecorder,
    NoOpLLMSpanRecorder,
    _truncate_content,
    create_operation_span,
    initialize_tracing,
    is_tracing_enabled,
)


def test_provider_name_mapping():
    """Test that provider names are correctly mapped to GenAI conventions."""
    assert PROVIDER_NAME_MAPPING["openai"] == "openai"
    assert PROVIDER_NAME_MAPPING["anthropic"] == "anthropic"
    assert PROVIDER_NAME_MAPPING["gemini"] == "google"
    assert PROVIDER_NAME_MAPPING["vertexai"] == "google"
    assert PROVIDER_NAME_MAPPING["groq"] == "groq"
    assert PROVIDER_NAME_MAPPING["ollama"] == "ollama"
    assert PROVIDER_NAME_MAPPING["openai-codex"] == "openai"
    assert PROVIDER_NAME_MAPPING["claude-code"] == "anthropic"


def test_truncate_content_short():
    """Test that short content is not truncated."""
    content = "This is a short message"
    result = _truncate_content(content)
    assert result == content


def test_truncate_content_long():
    """Test that long content is truncated."""
    content = "x" * 150000  # Exceeds MAX_CONTENT_LENGTH
    result = _truncate_content(content)
    assert len(result) < len(content)
    assert "[TRUNCATED:" in result
    assert result.startswith("x" * 100)


def test_noop_span_recorder():
    """Test that NoOpLLMSpanRecorder doesn't raise errors."""
    recorder = NoOpLLMSpanRecorder()
    # Should not raise any errors
    recorder.record_llm_call(
        provider="openai",
        model="gpt-4",
        scope="test",
        messages=[{"role": "user", "content": "test"}],
        response_content="test response",
        input_tokens=10,
        output_tokens=5,
        duration=1.0,
    )


def test_llm_span_recorder_format_messages():
    """Test message formatting to GenAI convention."""
    mock_tracer = MagicMock()
    recorder = LLMSpanRecorder(mock_tracer)

    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
    ]

    result = recorder._format_messages(messages)
    parsed = json.loads(result)

    assert len(parsed) == 2
    assert parsed[0]["role"] == "system"
    assert parsed[0]["content"] == "You are helpful"
    assert parsed[1]["role"] == "user"
    assert parsed[1]["content"] == "Hello"


def test_llm_span_recorder_format_output():
    """Test output formatting to GenAI convention."""
    mock_tracer = MagicMock()
    recorder = LLMSpanRecorder(mock_tracer)

    result = recorder._format_output("Hello world", "stop")
    parsed = json.loads(result)

    assert len(parsed) == 1
    assert parsed[0]["role"] == "assistant"
    assert parsed[0]["content"] == "Hello world"


def test_llm_span_recorder_format_output_none():
    """Test output formatting with None content."""
    mock_tracer = MagicMock()
    recorder = LLMSpanRecorder(mock_tracer)

    result = recorder._format_output(None, None)
    parsed = json.loads(result)

    assert parsed == []


def test_llm_span_recorder_extract_system_instructions():
    """Test system instruction extraction."""
    mock_tracer = MagicMock()
    recorder = LLMSpanRecorder(mock_tracer)

    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
    ]

    result = recorder._extract_system_instructions(messages)
    assert result == "You are helpful"


def test_llm_span_recorder_extract_system_instructions_none():
    """Test system instruction extraction with no system message."""
    mock_tracer = MagicMock()
    recorder = LLMSpanRecorder(mock_tracer)

    messages = [
        {"role": "user", "content": "Hello"},
    ]

    result = recorder._extract_system_instructions(messages)
    assert result is None


@patch("hindsight_api.tracing.time")
def test_llm_span_recorder_record_success(mock_time):
    """Test successful LLM call recording."""
    # Mock time
    mock_time.time_ns.return_value = 1000000000000  # 1 second in nanoseconds

    # Create mock tracer and span
    mock_span = MagicMock()
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

    recorder = LLMSpanRecorder(mock_tracer)

    messages = [{"role": "user", "content": "Hello"}]
    response_content = "Hi there!"

    recorder.record_llm_call(
        provider="openai",
        model="gpt-4",
        scope="test",
        messages=messages,
        response_content=response_content,
        input_tokens=10,
        output_tokens=5,
        duration=1.5,
        finish_reason="stop",
        error=None,
    )

    # Verify span was created with correct name (hindsight.{scope})
    mock_tracer.start_as_current_span.assert_called_once()
    call_args = mock_tracer.start_as_current_span.call_args
    assert call_args[0][0] == "hindsight.test"

    # Verify attributes were set
    assert mock_span.set_attribute.called
    attribute_calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}

    assert attribute_calls[GenAIAttributes.OPERATION_NAME] == "chat"
    assert attribute_calls[GenAIAttributes.PROVIDER_NAME] == "openai"
    assert attribute_calls[GenAIAttributes.REQUEST_MODEL] == "gpt-4"
    assert attribute_calls[GenAIAttributes.RESPONSE_MODEL] == "gpt-4"
    assert attribute_calls[GenAIAttributes.USAGE_INPUT_TOKENS] == 10
    assert attribute_calls[GenAIAttributes.USAGE_OUTPUT_TOKENS] == 5
    assert attribute_calls["hindsight.scope"] == "test"

    # Verify event was added
    mock_span.add_event.assert_called_once()
    event_call = mock_span.add_event.call_args
    assert event_call[0][0] == "gen_ai.client.inference.operation.details"

    # Verify status was set to OK
    mock_span.set_status.assert_called()

    # Verify span was ended
    mock_span.end.assert_called_once()


@patch("hindsight_api.tracing.time")
def test_llm_span_recorder_record_error(mock_time):
    """Test error LLM call recording."""
    # Mock time
    mock_time.time_ns.return_value = 1000000000000

    # Create mock tracer and span
    mock_span = MagicMock()
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

    recorder = LLMSpanRecorder(mock_tracer)

    messages = [{"role": "user", "content": "Hello"}]
    error = ValueError("Test error")

    recorder.record_llm_call(
        provider="anthropic",
        model="claude-3",
        scope="test",
        messages=messages,
        response_content=None,
        input_tokens=10,
        output_tokens=0,
        duration=0.5,
        finish_reason=None,
        error=error,
    )

    # Verify error status was set
    mock_span.set_status.assert_called()
    status_call = mock_span.set_status.call_args[0][0]
    assert status_call.status_code.name == "ERROR"

    # Verify error type attribute was set
    attribute_calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}
    assert attribute_calls[GenAIAttributes.ERROR_TYPE] == "ValueError"

    # Verify exception was recorded
    mock_span.record_exception.assert_called_once_with(error)


@patch("hindsight_api.tracing.time")
def test_llm_span_recorder_provider_mapping(mock_time):
    """Test that provider names are mapped correctly."""
    mock_time.time_ns.return_value = 1000000000000

    mock_span = MagicMock()
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

    recorder = LLMSpanRecorder(mock_tracer)

    # Test gemini -> google mapping
    recorder.record_llm_call(
        provider="gemini",
        model="gemini-pro",
        scope="test",
        messages=[{"role": "user", "content": "test"}],
        response_content="test",
        input_tokens=5,
        output_tokens=3,
        duration=1.0,
    )

    attribute_calls = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}
    assert attribute_calls[GenAIAttributes.PROVIDER_NAME] == "google"


# ==================== Parent Span Tests ====================


@patch("hindsight_api.tracing._tracing_enabled", False)
def test_create_operation_span_disabled():
    """Test that create_operation_span returns no-op when tracing is disabled."""
    # Tracing should be disabled by default (explicitly patched for test isolation)
    assert not is_tracing_enabled()

    # Should return a no-op context manager
    span = create_operation_span("test_operation", "test_bank_id")

    # Should be usable as context manager without errors
    with span:
        pass


@patch("hindsight_api.tracing._tracer")
@patch("hindsight_api.tracing._tracing_enabled", True)
def test_create_operation_span_enabled(mock_tracer):
    """Test that create_operation_span creates a span when tracing is enabled."""
    # Mock the tracer
    mock_span = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_span

    # Create operation span
    span = create_operation_span("retain", "bank123")

    # Verify span was created with correct name
    mock_tracer.start_as_current_span.assert_called_once_with("hindsight.retain")

    # Verify attributes were set
    mock_span.set_attribute.assert_any_call("hindsight.operation", "retain")
    mock_span.set_attribute.assert_any_call("hindsight.bank_id", "bank123")


@patch("hindsight_api.tracing._tracer")
@patch("hindsight_api.tracing._tracing_enabled", True)
def test_create_operation_span_no_bank_id(mock_tracer):
    """Test that create_operation_span works without bank_id."""
    mock_span = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_span

    # Create operation span without bank_id
    span = create_operation_span("consolidation")

    # Verify span was created
    mock_tracer.start_as_current_span.assert_called_once_with("hindsight.consolidation")

    # Verify only operation attribute was set (not bank_id)
    assert mock_span.set_attribute.call_count == 1
    mock_span.set_attribute.assert_called_once_with("hindsight.operation", "consolidation")


@patch("hindsight_api.tracing._tracer")
@patch("hindsight_api.tracing._tracing_enabled", True)
def test_create_operation_span_all_operations(mock_tracer):
    """Test that all 4 operations can create parent spans."""
    mock_span = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_span

    operations = ["retain", "consolidation", "reflect", "mental_model_refresh"]

    for operation in operations:
        mock_tracer.reset_mock()
        mock_span.reset_mock()

        span = create_operation_span(operation, "test_bank")

        # Verify span was created with correct name
        mock_tracer.start_as_current_span.assert_called_once_with(f"hindsight.{operation}")

        # Verify attributes
        mock_span.set_attribute.assert_any_call("hindsight.operation", operation)
        mock_span.set_attribute.assert_any_call("hindsight.bank_id", "test_bank")


@patch("hindsight_api.tracing.time")
@patch("hindsight_api.tracing._tracer")
@patch("hindsight_api.tracing._tracing_enabled", True)
def test_parent_child_span_hierarchy(mock_tracer, mock_time):
    """Test that child LLM spans are created under parent operation spans."""
    mock_time.time_ns.return_value = 1000000000000

    # Create mock parent span
    mock_parent_span = MagicMock()
    mock_parent_span.__enter__ = MagicMock(return_value=mock_parent_span)
    mock_parent_span.__exit__ = MagicMock(return_value=False)

    # Create mock child span
    mock_child_span = MagicMock()

    # Mock tracer to return parent span first, then child span
    mock_tracer.start_as_current_span.side_effect = [
        mock_parent_span,  # Parent span
        MagicMock(__enter__=MagicMock(return_value=mock_child_span), __exit__=MagicMock(return_value=False)),  # Child
    ]

    # Create parent operation span
    with create_operation_span("retain", "bank123"):
        # Simulate creating a child LLM span
        recorder = LLMSpanRecorder(mock_tracer)
        recorder.record_llm_call(
            provider="openai",
            model="gpt-4",
            scope="retain_extract_facts",
            messages=[{"role": "user", "content": "test"}],
            response_content="response",
            input_tokens=10,
            output_tokens=5,
            duration=1.0,
        )

    # Verify both parent and child spans were created
    assert mock_tracer.start_as_current_span.call_count == 2

    # Verify parent span was created first
    first_call = mock_tracer.start_as_current_span.call_args_list[0]
    assert first_call[0][0] == "hindsight.retain"

    # Verify child span was created second (hindsight.{scope})
    second_call = mock_tracer.start_as_current_span.call_args_list[1]
    assert second_call[0][0] == "hindsight.retain_extract_facts"


@patch("hindsight_api.tracing._tracer")
@patch("hindsight_api.tracing._tracing_enabled", True)
def test_operation_span_context_manager(mock_tracer):
    """Test that operation spans work as context managers."""
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=False)
    mock_tracer.start_as_current_span.return_value = mock_span

    # Use span as context manager
    with create_operation_span("reflect", "bank456"):
        # Do some work
        pass

    # Verify span lifecycle
    mock_tracer.start_as_current_span.assert_called_once()
    mock_span.__enter__.assert_called_once()
    mock_span.__exit__.assert_called_once()
