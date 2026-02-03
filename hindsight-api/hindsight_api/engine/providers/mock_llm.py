"""
Mock LLM provider for testing.

This provider allows tests to record LLM calls and return configurable mock responses
without making actual API calls to external LLM services.
"""

import logging
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

        # Return mock response
        if self._mock_response is not None:
            result = self._mock_response
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

        if self._mock_response is not None:
            if isinstance(self._mock_response, LLMToolCallResult):
                return self._mock_response
            # Allow setting just tool calls as a list
            if isinstance(self._mock_response, list):
                return LLMToolCallResult(
                    tool_calls=[
                        LLMToolCall(id=f"mock_{i}", name=tc["name"], arguments=tc.get("arguments", {}))
                        for i, tc in enumerate(self._mock_response)
                    ],
                    finish_reason="tool_calls",
                )

        return LLMToolCallResult(content="mock response", finish_reason="stop")

    async def cleanup(self) -> None:
        """Clean up resources (no-op for mock provider)."""
        pass

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
        """Clear the recorded mock calls."""
        self._mock_calls = []
