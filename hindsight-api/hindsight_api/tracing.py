"""
OpenTelemetry distributed tracing instrumentation for Hindsight API.

This module provides tracing for:
- LLM API calls with full prompts/completions following GenAI semantic conventions
- Token usage and model information
- Error tracking and finish reasons

Tracing is conditional and disabled by default. When enabled, traces are exported
to Langfuse (or any OTLP-compatible backend) via OTLP HTTP protocol.
"""

import json
import logging
import time
from typing import Any, Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)


def _serialize_for_span(obj: Any) -> str:
    """Serialize an object for span recording, handling Pydantic models."""
    if isinstance(obj, str):
        return obj
    if hasattr(obj, "model_dump_json"):
        # Pydantic v2 model
        return obj.model_dump_json()
    if hasattr(obj, "json"):
        # Pydantic v1 model
        return obj.json()
    if hasattr(obj, "model_dump"):
        # Pydantic v2 model - convert to dict then json
        return json.dumps(obj.model_dump())
    if hasattr(obj, "dict"):
        # Pydantic v1 model - convert to dict then json
        return json.dumps(obj.dict())
    # Fallback to json.dumps for dicts and other types
    return json.dumps(obj)


# No-op tracer for when tracing is disabled
class NoOpTracer:
    """No-op tracer that provides the same interface as OpenTelemetry Tracer but does nothing."""

    def start_as_current_span(self, name: str, **kwargs):
        """Return a no-op context manager."""
        from contextlib import nullcontext

        return nullcontext()

    def start_span(self, name: str, **kwargs):
        """Return a no-op span."""
        return NoOpSpan()


class NoOpSpan:
    """No-op span that provides the same interface as OpenTelemetry Span but does nothing."""

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op."""
        pass

    def set_status(self, status: Any) -> None:
        """No-op."""
        pass

    def record_exception(self, exception: Exception) -> None:
        """No-op."""
        pass

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        """No-op."""
        pass

    def end(self, end_time: int | None = None) -> None:
        """No-op."""
        pass


# Global tracer instance
_tracer: trace.Tracer | NoOpTracer = NoOpTracer()
_tracing_enabled: bool = False


# GenAI semantic convention attribute names (based on v1.37 spec)
class GenAIAttributes:
    """GenAI semantic convention attribute names."""

    # Operation and provider
    OPERATION_NAME = "gen_ai.operation.name"
    PROVIDER_NAME = "gen_ai.provider.name"

    # Model information
    REQUEST_MODEL = "gen_ai.request.model"
    RESPONSE_MODEL = "gen_ai.response.model"

    # Token usage
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

    # Messages and prompts
    SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
    INPUT_MESSAGES = "gen_ai.input.messages"
    OUTPUT_MESSAGES = "gen_ai.output.messages"

    # Response metadata
    FINISH_REASONS = "gen_ai.response.finish_reasons"

    # Error tracking
    ERROR_TYPE = "error.type"


# Provider name mapping (Hindsight internal -> GenAI semantic convention)
PROVIDER_NAME_MAPPING = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "google",
    "vertexai": "google",
    "groq": "groq",
    "ollama": "ollama",
    "lmstudio": "lmstudio",
    "openai-codex": "openai",
    "claude-code": "anthropic",
    "mock": "mock",
}


def initialize_tracing(
    service_name: str,
    endpoint: str,
    headers: Optional[str] = None,
    deployment_environment: str = "development",
) -> None:
    """
    Initialize OpenTelemetry tracing with OTLP exporter.

    Args:
        service_name: Name of the service for resource attributes
        endpoint: OTLP endpoint URL (e.g., https://cloud.langfuse.com/api/public/otel)
        headers: Optional headers in format "key1=value1,key2=value2"
        deployment_environment: Deployment environment (e.g., development, staging, production)
    """
    global _tracer, _tracing_enabled

    # Create resource with service information
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "0.4.8",  # Could import from __version__
            "deployment.environment.name": deployment_environment,
        }
    )

    # Parse headers
    headers_dict = {}
    if headers:
        for pair in headers.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                headers_dict[key.strip()] = value.strip()

    # Create OTLP HTTP exporter
    # Note: Langfuse expects /v1/traces path appended to base endpoint
    otlp_endpoint = endpoint if endpoint.endswith("/v1/traces") else f"{endpoint}/v1/traces"
    otlp_exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        headers=headers_dict,
    )

    # Create tracer provider with batch processor
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Set global tracer provider
    trace.set_tracer_provider(provider)

    # Get tracer for this application
    _tracer = trace.get_tracer(__name__)
    _tracing_enabled = True

    logger.info(f"Tracing initialized: endpoint={otlp_endpoint}, service={service_name}")


def get_tracer() -> trace.Tracer | NoOpTracer:
    """
    Get the global tracer instance.

    Returns a no-op tracer if tracing is disabled, so callers don't need to check for None.
    This improves code readability by allowing direct use without null checks.
    """
    return _tracer


def create_operation_span(operation: str, bank_id: str | None = None):
    """
    Create a parent span for a Hindsight operation (retain, reflect, consolidation, etc.).

    This creates the span hierarchy:
    - hindsight.{operation} (parent)
      - chat {model} (child LLM calls)

    Args:
        operation: Operation name (retain, reflect, consolidation, mental_model_refresh)
        bank_id: Optional bank ID for context

    Returns:
        Span context manager
    """
    if not _tracing_enabled or _tracer is None:
        # Return a no-op context manager
        from contextlib import nullcontext

        return nullcontext()

    span_name = f"hindsight.{operation}"
    span = _tracer.start_as_current_span(span_name)

    # Add operation-specific attributes
    if span and hasattr(span, "set_attribute"):
        span.set_attribute("hindsight.operation", operation)
        if bank_id:
            span.set_attribute("hindsight.bank_id", bank_id)

    return span


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled."""
    return _tracing_enabled


# Maximum content length before truncation (to stay within span size limits)
MAX_CONTENT_LENGTH = 100_000  # characters


def _truncate_content(content: str) -> str:
    """Truncate content if too large for span."""
    if len(content) > MAX_CONTENT_LENGTH:
        return content[:MAX_CONTENT_LENGTH] + f"\n\n[TRUNCATED: {len(content) - MAX_CONTENT_LENGTH} chars omitted]"
    return content


class LLMSpanRecorder:
    """
    Records OpenTelemetry spans for LLM calls following GenAI semantic conventions.
    """

    def __init__(self, tracer: trace.Tracer):
        self.tracer = tracer

    def record_llm_call(
        self,
        provider: str,
        model: str,
        scope: str,
        messages: list[dict[str, str]],
        response_content: Optional[str],
        input_tokens: int,
        output_tokens: int,
        duration: float,
        finish_reason: Optional[str] = None,
        error: Optional[Exception] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """
        Record a completed LLM call as a span with GenAI semantic conventions.

        This creates a span AFTER the call completes, using timestamps to
        set the correct start/end times. This approach works better with
        the existing sync metrics recording pattern.

        Args:
            provider: Hindsight provider name
            model: Model name
            scope: Scope identifier (memory, reflect, consolidation, etc.)
            messages: Input messages (chat history)
            response_content: Response text from LLM
            input_tokens: Input token count
            output_tokens: Output token count
            duration: Call duration in seconds
            finish_reason: Reason the model stopped (stop, length, tool_calls, etc.)
            error: Exception if call failed
            tool_calls: List of tool calls made (for function calling)
        """
        try:
            # Map provider name to GenAI semantic convention
            genai_provider = PROVIDER_NAME_MAPPING.get(provider.lower(), provider.lower())

            # Determine operation name based on scope/context
            operation_name = "chat"  # Default for GenAI semantic conventions

            # Create span name: "hindsight.{scope}" for consistency with parent spans
            # Model info is available in span attributes (gen_ai.request.model)
            if scope:
                span_name = f"hindsight.{scope}"
            else:
                # Fallback to chat {model} if no scope provided
                span_name = f"{operation_name} {model}"

            # Calculate timestamps
            end_time_ns = time.time_ns()
            start_time_ns = end_time_ns - int(duration * 1_000_000_000)

            # Create span with explicit timestamps
            with self.tracer.start_as_current_span(
                span_name,
                start_time=start_time_ns,
                end_on_exit=False,  # We'll set end time manually
            ) as span:
                # Set required attributes
                span.set_attribute(GenAIAttributes.OPERATION_NAME, operation_name)
                span.set_attribute(GenAIAttributes.PROVIDER_NAME, genai_provider)
                span.set_attribute(GenAIAttributes.REQUEST_MODEL, model)
                span.set_attribute(GenAIAttributes.RESPONSE_MODEL, model)
                span.set_attribute(GenAIAttributes.USAGE_INPUT_TOKENS, input_tokens)
                span.set_attribute(GenAIAttributes.USAGE_OUTPUT_TOKENS, output_tokens)

                # Add custom attributes for Hindsight context
                span.set_attribute("hindsight.scope", scope)
                span.set_attribute("hindsight.provider.internal", provider)

                # Add tool call information if present
                if tool_calls:
                    span.set_attribute("gen_ai.tool_calls.count", len(tool_calls))
                    # Add tool names as comma-separated list
                    tool_names = [tc.get("name", "") for tc in tool_calls]
                    span.set_attribute("gen_ai.tool_calls.names", ",".join(tool_names))

                # Format messages for GenAI conventions (as JSON)
                input_messages_json = self._format_messages(messages)
                output_messages_json = self._format_output(response_content, finish_reason)

                # Extract system instructions if present
                system_instructions = self._extract_system_instructions(messages)

                # Add event with prompts/completions following v1.37 conventions
                event_attrs = {}
                if input_messages_json:
                    event_attrs[GenAIAttributes.INPUT_MESSAGES] = input_messages_json
                if output_messages_json:
                    event_attrs[GenAIAttributes.OUTPUT_MESSAGES] = output_messages_json
                if system_instructions:
                    event_attrs[GenAIAttributes.SYSTEM_INSTRUCTIONS] = system_instructions
                if finish_reason:
                    event_attrs[GenAIAttributes.FINISH_REASONS] = json.dumps([finish_reason])

                span.add_event(
                    "gen_ai.client.inference.operation.details",
                    attributes=event_attrs,
                )

                # Add individual tool call events with details
                if tool_calls:
                    for i, tc in enumerate(tool_calls):
                        tool_event_attrs = {
                            "tool.name": tc.get("name", ""),
                            "tool.id": tc.get("id", ""),
                            "tool.arguments": json.dumps(tc.get("arguments", {})),
                        }
                        span.add_event(f"gen_ai.tool_call.{i}", attributes=tool_event_attrs)

                # Handle errors
                if error:
                    span.set_status(Status(StatusCode.ERROR, str(error)))
                    span.set_attribute(GenAIAttributes.ERROR_TYPE, type(error).__name__)
                    span.record_exception(error)
                else:
                    span.set_status(Status(StatusCode.OK))

                # Set end time
                span.end(end_time=end_time_ns)

        except Exception as e:
            # Don't let tracing errors break LLM calls
            logger.error(f"Failed to record LLM span: {e}", exc_info=True)

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        """
        Format messages into GenAI semantic convention format (JSON array).

        Returns JSON string representation of message array.
        """
        try:
            formatted = []
            for msg in messages:
                content = msg.get("content", "")
                # Truncate if needed
                if isinstance(content, str):
                    content = _truncate_content(content)

                formatted.append(
                    {
                        "role": msg.get("role", "user"),
                        "content": content,
                    }
                )

            return json.dumps(formatted)
        except Exception as e:
            logger.warning(f"Failed to format input messages: {e}")
            return "[]"

    def _format_output(
        self,
        content: Optional[str],
        finish_reason: Optional[str],
    ) -> str:
        """Format output message into GenAI semantic convention format."""
        try:
            if content is None:
                return "[]"

            # Truncate if needed
            if isinstance(content, str):
                content = _truncate_content(content)

            return json.dumps(
                [
                    {
                        "role": "assistant",
                        "content": content,
                    }
                ]
            )
        except Exception as e:
            logger.warning(f"Failed to format output message: {e}")
            return "[]"

    def _extract_system_instructions(self, messages: list[dict[str, str]]) -> Optional[str]:
        """Extract system instructions from messages if present."""
        try:
            for msg in messages:
                if msg.get("role") == "system":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return _truncate_content(content)
                    return str(content)
        except Exception as e:
            logger.warning(f"Failed to extract system instructions: {e}")
        return None


class NoOpLLMSpanRecorder:
    """No-op span recorder for when tracing is disabled."""

    def record_llm_call(self, **kwargs) -> None:
        """No-op."""
        pass


# Global span recorder instance
_span_recorder: Optional[LLMSpanRecorder] = None


def get_span_recorder() -> LLMSpanRecorder | NoOpLLMSpanRecorder:
    """Get the global span recorder (NoOp if tracing disabled)."""
    if _span_recorder is None:
        return NoOpLLMSpanRecorder()
    return _span_recorder


def create_span_recorder() -> LLMSpanRecorder:
    """Create and set the global span recorder."""
    global _span_recorder
    tracer = get_tracer()
    if tracer is None:
        raise RuntimeError("Tracing not initialized. Call initialize_tracing() first.")
    _span_recorder = LLMSpanRecorder(tracer)
    return _span_recorder
