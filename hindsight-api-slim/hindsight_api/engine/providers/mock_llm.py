"""
Mock LLM provider for testing.

This provider allows tests to record LLM calls and return configurable mock responses
without making actual API calls to external LLM services.
"""

import logging
import re
from collections.abc import Callable
from typing import Any

from ..llm_interface import LLMInterface
from ..response_models import LLMToolCall, LLMToolCallResult, TokenUsage

logger = logging.getLogger(__name__)


class MockLLM(LLMInterface):
    """
    Mock LLM provider for testing.

    This provider records all calls and returns configurable mock responses,
    enabling tests to verify LLM interactions without making real API calls.

    Example:
        # Create mock provider
        mock_llm = MockLLM(provider="mock", api_key="", base_url="", model="mock-model")

        # Set mock response
        mock_llm.set_mock_response({"answer": "test"})

        # Make calls
        result = await mock_llm.call(
            messages=[{"role": "user", "content": "test"}],
            response_format=MyResponseModel
        )

        # Verify calls
        calls = mock_llm.get_mock_calls()
        assert len(calls) == 1
        assert calls[0]["scope"] == "memory"
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        reasoning_effort: str = "low",
        **kwargs: Any,
    ):
        """
        Initialize mock LLM provider.

        Args:
            provider: Provider name (should be "mock").
            api_key: Not used for mock provider.
            base_url: Not used for mock provider.
            model: Model name for tracking.
            reasoning_effort: Not used for mock provider.
            **kwargs: Additional parameters (not used).
        """
        super().__init__(provider, api_key, base_url, model, reasoning_effort, **kwargs)

        # Storage for test verification
        self._mock_calls: list[dict] = []
        self._mock_response: Any = None
        self._mock_exception: Exception | None = None
        self._response_callback: Callable[[list[dict], str], Any] | None = None

    async def verify_connection(self) -> None:
        """
        Verify mock provider (always succeeds).

        Mock provider doesn't need connection verification since it doesn't
        make real API calls.
        """
        logger.debug("Mock LLM: connection verification (always succeeds)")

    async def call(
        self,
        messages: list[dict[str, str]],
        response_format: Any | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        scope: str = "memory",
        max_retries: int = 10,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        skip_validation: bool = False,
        strict_schema: bool = False,
        return_usage: bool = False,
    ) -> Any:
        """
        Make a mock LLM API call.

        Records the call for test verification and returns the configured mock response.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            response_format: Optional Pydantic model for structured output.
            max_completion_tokens: Not used in mock.
            temperature: Not used in mock.
            scope: Scope identifier for tracking.
            max_retries: Not used in mock.
            initial_backoff: Not used in mock.
            max_backoff: Not used in mock.
            skip_validation: Return raw JSON without Pydantic validation.
            strict_schema: Not used in mock.
            return_usage: If True, return tuple (result, TokenUsage) instead of just result.

        Returns:
            If return_usage=False: Parsed response if response_format is provided, otherwise text content.
            If return_usage=True: Tuple of (result, TokenUsage) with mock token counts.
        """
        # Record the call for test verification
        call_record = {
            "provider": self.provider,
            "model": self.model,
            "messages": messages,
            "response_format": response_format.__name__
            if response_format and hasattr(response_format, "__name__")
            else str(response_format),
            "scope": scope,
        }
        self._mock_calls.append(call_record)
        logger.debug(f"Mock LLM call recorded: scope={scope}, model={self.model}")

        # Raise mock exception if configured
        if self._mock_exception is not None:
            raise self._mock_exception

        # Record trace span (minimal for mock provider)
        from hindsight_api.tracing import get_span_recorder

        span_recorder = get_span_recorder()
        span_recorder.record_llm_call(
            provider=self.provider,
            model=self.model,
            scope=scope,
            messages=messages,
            response_content="mock response",
            input_tokens=10,
            output_tokens=5,
            duration=0.001,  # Mock calls are instant
            finish_reason="stop",
            error=None,
        )

        # Return mock response
        if self._response_callback is not None:
            result = self._response_callback(messages, scope)
        elif self._mock_response is not None:
            result = self._mock_response
        elif scope == "retain_extract_facts":
            result = self._build_retain_extract_response(messages)
        elif response_format is not None:
            # Try to create a minimal valid instance of the response format
            try:
                # For Pydantic models, try to create with minimal valid data
                result = {"mock": True}
            except Exception:
                result = {"mock": True}
        else:
            result = "mock response"

        if return_usage:
            token_usage = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
            return result, token_usage
        return result

    @staticmethod
    def _extract_text_chunk(messages: list[dict[str, str]]) -> str:
        """Extract the source text payload from a retain prompt."""
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            marker = "\nText:\n"
            if marker in content:
                return content.split(marker, 1)[1].strip()
            return content.strip()
        return ""

    @staticmethod
    def _split_fact_sentences(text: str, max_facts: int = 50) -> list[str]:
        """Split source text into stable fact-sized sentences."""
        candidates = re.split(r"(?<=[.!?])\s+|\n+", text)
        facts = []
        for candidate in candidates:
            cleaned = candidate.strip().strip("-*").strip()
            if len(cleaned) < 3:
                continue
            facts.append(cleaned)
            if len(facts) >= max_facts:
                break
        if facts:
            return facts
        fallback = text.strip()
        return [fallback] if fallback else []

    @staticmethod
    def _extract_entities(sentence: str) -> list[str]:
        """Extract obvious named entities for test-mode fact generation."""
        entities = []
        seen = set()
        for match in re.finditer(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", sentence):
            entity = match.group(0)
            if entity not in seen:
                seen.add(entity)
                entities.append(entity)
        return entities

    @classmethod
    def _build_retain_extract_response(cls, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Generate deterministic fact extraction output from the input text."""
        text = cls._extract_text_chunk(messages)
        facts = []
        for sentence in cls._split_fact_sentences(text):
            fact: dict[str, Any] = {
                "what": sentence,
                "fact_type": "world",
                "fact_kind": "conversation",
            }
            entities = cls._extract_entities(sentence)
            if entities:
                fact["entities"] = entities
            facts.append(fact)
        return {"facts": facts}

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        scope: str = "tools",
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        tool_choice: str | dict[str, Any] = "auto",
    ) -> LLMToolCallResult:
        """
        Make a mock LLM API call with tool/function calling support.

        Records the call for test verification and returns the configured mock response.

        Args:
            messages: List of message dicts. Can include tool results with role='tool'.
            tools: List of tool definitions in OpenAI format.
            max_completion_tokens: Not used in mock.
            temperature: Not used in mock.
            scope: Scope identifier for tracking.
            max_retries: Not used in mock.
            initial_backoff: Not used in mock.
            max_backoff: Not used in mock.
            tool_choice: Not used in mock.

        Returns:
            LLMToolCallResult with content and/or tool_calls.
        """
        # Record the call for test verification
        call_record = {
            "provider": self.provider,
            "model": self.model,
            "messages": messages,
            "tools": [t.get("function", {}).get("name") for t in tools],
            "scope": scope,
        }
        self._mock_calls.append(call_record)

        # Raise mock exception if configured
        if self._mock_exception is not None:
            raise self._mock_exception

        # Record OpenTelemetry span
        from hindsight_api.tracing import get_span_recorder

        span_recorder = get_span_recorder()

        if self._response_callback is not None:
            cb_result = self._response_callback(messages, scope)
            if isinstance(cb_result, LLMToolCallResult):
                result = cb_result
            else:
                result = LLMToolCallResult(
                    content=str(cb_result) if cb_result is not None else "mock response", finish_reason="stop"
                )
        elif self._mock_response is not None:
            if isinstance(self._mock_response, LLMToolCallResult):
                result = self._mock_response
            elif isinstance(self._mock_response, list):
                # Allow setting just tool calls as a list
                result = LLMToolCallResult(
                    tool_calls=[
                        LLMToolCall(id=f"mock_{i}", name=tc["name"], arguments=tc.get("arguments", {}))
                        for i, tc in enumerate(self._mock_response)
                    ],
                    finish_reason="tool_calls",
                )
            else:
                result = LLMToolCallResult(content="mock response", finish_reason="stop")
        else:
            result = LLMToolCallResult(content="mock response", finish_reason="stop")

        # Record span with mock values
        # Convert LLMToolCall objects to dicts for span recording
        tool_calls_dict = (
            [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in result.tool_calls]
            if result.tool_calls
            else None
        )
        span_recorder.record_llm_call(
            provider=self.provider,
            model=self.model,
            scope=scope,
            messages=messages,
            response_content=result.content,
            input_tokens=10,  # Mock value
            output_tokens=5,  # Mock value
            duration=0.1,  # Mock value
            finish_reason=result.finish_reason,
            error=None,
            tool_calls=tool_calls_dict,
        )

        return result

    async def cleanup(self) -> None:
        """Clean up resources (no-op for mock provider)."""
        pass

    def set_response_callback(self, fn: Callable[[list[dict], str], Any]) -> None:
        """
        Set a callback invoked on each call() instead of _mock_response.

        The callback receives (messages, scope) and returns the response.
        Useful for returning different responses per call (e.g., cycling
        through a corpus in a benchmark).
        """
        self._response_callback = fn

    def set_mock_response(self, response: Any) -> None:
        """
        Set the response to return from mock calls.

        Args:
            response: The response to return. Can be:
                - A dict/Pydantic model for regular calls
                - An LLMToolCallResult for tool calls
                - A list of tool call dicts for tool calls
                - Any other value to return as-is
        """
        self._mock_response = response

    def set_mock_exception(self, exception: Exception) -> None:
        """
        Set an exception to raise from mock calls.

        Args:
            exception: The exception to raise on the next call.
                      After raising, the exception is cleared.
        """
        self._mock_exception = exception

    def get_mock_calls(self) -> list[dict]:
        """
        Get the list of recorded mock calls.

        Returns:
            List of call records, each containing:
                - provider: Provider name
                - model: Model name
                - messages: Messages sent
                - response_format/tools: Format or tools used
                - scope: Call scope
        """
        return self._mock_calls

    def clear_mock_calls(self) -> None:
        """Clear the recorded mock calls and any set exception."""
        self._mock_calls = []
        self._mock_exception = None
