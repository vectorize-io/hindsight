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
    assert PROVIDER_NAME_MAPPING["github-copilot"] == "github"


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


# ---------------------------------------------------------------------------
# Bootstrap helper shared by the API and worker entrypoints (issue #3614)
# ---------------------------------------------------------------------------


@pytest.fixture
def restore_tracing_globals():
    """Snapshot and restore tracing module state around a test.

    initialize_tracing_from_config starts a real BatchSpanProcessor thread, so
    every test that enables tracing has to tear it down again.
    """
    import hindsight_api.tracing as tracing_module

    saved = (
        tracing_module._tracer,
        tracing_module._tracing_enabled,
        tracing_module._provider,
        tracing_module._span_recorder,
    )
    try:
        yield tracing_module
    finally:
        tracing_module.shutdown_tracing()
        (
            tracing_module._tracer,
            tracing_module._tracing_enabled,
            tracing_module._provider,
            tracing_module._span_recorder,
        ) = saved


def _tracing_config(**overrides):
    """Build a raw HindsightConfig with tracing knobs set."""
    import dataclasses

    from hindsight_api.config import _get_raw_config

    defaults = {
        "otel_traces_enabled": True,
        "otel_exporter_otlp_endpoint": "http://localhost:4318",
        "otel_exporter_otlp_headers": None,
        "otel_service_name": "hindsight-api",
        "otel_deployment_environment": "test",
    }
    return dataclasses.replace(_get_raw_config(), **{**defaults, **overrides})


def test_initialize_tracing_from_config_disabled(restore_tracing_globals):
    """Tracing stays off when the feature flag is off."""
    tracing_module = restore_tracing_globals

    assert tracing_module.initialize_tracing_from_config(_tracing_config(otel_traces_enabled=False)) is False
    assert tracing_module.is_tracing_enabled() is False


def test_initialize_tracing_from_config_without_endpoint_warns(restore_tracing_globals, caplog):
    """Enabled but endpoint-less config is reported, not silently ignored."""
    tracing_module = restore_tracing_globals

    with caplog.at_level("WARNING", logger="hindsight_api.tracing"):
        result = tracing_module.initialize_tracing_from_config(_tracing_config(otel_exporter_otlp_endpoint=None))

    assert result is False
    assert tracing_module.is_tracing_enabled() is False
    assert "no endpoint configured" in caplog.text


def test_initialize_tracing_from_config_enables_tracing(restore_tracing_globals):
    """A fully configured process gets a live tracer and a registered LLM recorder."""
    tracing_module = restore_tracing_globals

    assert tracing_module.initialize_tracing_from_config(_tracing_config()) is True
    assert tracing_module.is_tracing_enabled() is True
    assert tracing_module._span_recorder is not None
    assert tracing_module._span_recorder in tracing_module.get_span_recorder()._recorders


def test_initialize_tracing_from_config_default_service_name(restore_tracing_globals, monkeypatch):
    """With HINDSIGHT_API_OTEL_SERVICE_NAME unset, the caller's default wins.

    This is what lets the worker report itself as "hindsight-worker" instead of
    inheriting the API's default service name (issue #3614).
    """
    from hindsight_api.config import ENV_OTEL_SERVICE_NAME

    tracing_module = restore_tracing_globals
    monkeypatch.delenv(ENV_OTEL_SERVICE_NAME, raising=False)

    assert (
        tracing_module.initialize_tracing_from_config(_tracing_config(), default_service_name="hindsight-worker")
        is True
    )
    assert tracing_module._provider.resource.attributes["service.name"] == "hindsight-worker"


def test_initialize_tracing_from_config_env_service_name_wins(restore_tracing_globals, monkeypatch):
    """An explicitly configured service name overrides the caller's default."""
    from hindsight_api.config import ENV_OTEL_SERVICE_NAME

    tracing_module = restore_tracing_globals
    monkeypatch.setenv(ENV_OTEL_SERVICE_NAME, "my-service")

    tracing_module.initialize_tracing_from_config(
        _tracing_config(otel_service_name="my-service"), default_service_name="hindsight-worker"
    )

    assert tracing_module._provider.resource.attributes["service.name"] == "my-service"


def test_initialize_tracing_from_config_never_raises(restore_tracing_globals, monkeypatch):
    """A broken exporter must not stop the process from starting."""
    tracing_module = restore_tracing_globals
    monkeypatch.setattr(
        tracing_module,
        "initialize_tracing",
        MagicMock(side_effect=RuntimeError("boom")),
    )

    assert tracing_module.initialize_tracing_from_config(_tracing_config()) is False
    assert tracing_module.is_tracing_enabled() is False


def test_span_resource_reports_the_package_version(restore_tracing_globals):
    """service.version must track the real package version, not a stale literal."""
    from hindsight_api import __version__

    tracing_module = restore_tracing_globals
    tracing_module.initialize_tracing_from_config(_tracing_config())

    assert tracing_module._provider.resource.attributes["service.version"] == __version__


def test_shutdown_tracing_flushes_and_resets(restore_tracing_globals):
    """Shutdown flushes the batch processor and leaves the module inert."""
    tracing_module = restore_tracing_globals
    tracing_module.initialize_tracing_from_config(_tracing_config())
    provider = tracing_module._provider
    recorder = tracing_module._span_recorder

    with patch.object(provider, "shutdown") as mock_shutdown:
        tracing_module.shutdown_tracing()

    mock_shutdown.assert_called_once()
    assert tracing_module.is_tracing_enabled() is False
    assert tracing_module._provider is None
    assert recorder not in tracing_module.get_span_recorder()._recorders


def test_shutdown_tracing_without_init_is_noop(restore_tracing_globals):
    """Never-initialized processes can call shutdown unconditionally."""
    tracing_module = restore_tracing_globals
    tracing_module._provider = None

    tracing_module.shutdown_tracing()  # must not raise


@pytest.mark.asyncio
async def test_api_lifespan_bootstraps_and_flushes_tracing():
    """The API entrypoint must bootstrap tracing on startup and flush it on shutdown.

    Parity guard with the worker entrypoint (see test_worker_main.py): both
    processes go through the same helper, and neither may silently stop calling
    it. Runs the lifespan headless — no DB, no poller — so it stays a fast unit
    test.
    """
    import dataclasses
    from unittest.mock import AsyncMock

    from hindsight_api.api.http import create_app
    from hindsight_api.config import _get_raw_config

    config = dataclasses.replace(
        _get_raw_config(),
        otel_traces_enabled=True,
        otel_exporter_otlp_endpoint="http://localhost:4318",
        worker_enabled=False,
    )
    memory = AsyncMock()
    memory._pool = None
    memory.audit_logger = None
    memory._backend.supports_worker_poller = False

    bootstrap_calls = []

    with (
        patch("hindsight_api.config.get_config", return_value=config),
        patch("hindsight_api.api.http.get_config", return_value=config),
        patch(
            "hindsight_api.tracing.initialize_tracing_from_config",
            side_effect=lambda cfg, **kwargs: bootstrap_calls.append(cfg) or False,
        ),
        patch("hindsight_api.tracing.shutdown_tracing") as mock_shutdown,
    ):
        app = create_app(memory, initialize_memory=False)
        async with app.router.lifespan_context(app):
            pass

    assert bootstrap_calls == [config]
    mock_shutdown.assert_called_once()
