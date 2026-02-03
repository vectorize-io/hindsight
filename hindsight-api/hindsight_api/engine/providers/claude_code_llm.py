"""
Claude Code LLM provider using Claude Agent SDK.

This provider enables using Claude Pro/Max subscriptions for API calls
via the Claude CLI authentication. It uses the Claude Agent SDK which
automatically handles authentication via `claude auth login` credentials.
"""

import asyncio
import json
import logging
import time
from typing import Any

from hindsight_api.engine.llm_interface import LLMInterface, OutputTooLongError
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage
from hindsight_api.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class ClaudeCodeLLM(LLMInterface):
    """
    LLM provider using Claude Code authentication.

    Authenticates using Claude Pro/Max credentials via `claude auth login`
    and makes API calls through the Claude Agent SDK.
    """

    def __init__(
        self,
        provider: str,
        api_key: str,  # Will be ignored, uses CLI auth
        base_url: str,
        model: str,
        reasoning_effort: str = "low",
        **kwargs: Any,
    ):
        """Initialize Claude Code LLM provider."""
        super().__init__(provider, api_key, base_url, model, reasoning_effort, **kwargs)

        # Verify Claude Agent SDK is available
        try:
            self._verify_claude_code_available()
            logger.info("Claude Code: Using Claude Agent SDK (authentication via claude auth login)")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Claude Code provider: {e}\n\n"
                "To set up Claude Code authentication:\n"
                "1. Install Claude Code CLI: npm install -g @anthropics/claude-code\n"
                "2. Login with your Pro/Max plan: claude auth login\n"
                "3. Verify authentication: claude --version\n\n"
                "Or use a different provider (anthropic, openai, gemini) with API keys."
            ) from e

        # Metrics collector is imported at module level

    def _verify_claude_code_available(self) -> None:
        """
        Verify that Claude Agent SDK can be imported and is properly configured.

        Raises:
            ImportError: If Claude Agent SDK is not installed.
            RuntimeError: If Claude Code is not authenticated.
        """
        try:
            # Import Claude Agent SDK
            # Reduce Claude Agent SDK logging verbosity
            import logging as sdk_logging

            from claude_agent_sdk import query  # noqa: F401

            sdk_logging.getLogger("claude_agent_sdk").setLevel(sdk_logging.WARNING)
            sdk_logging.getLogger("claude_agent_sdk._internal").setLevel(sdk_logging.WARNING)

            logger.debug("Claude Agent SDK imported successfully")
        except ImportError as e:
            raise ImportError(
                "Claude Agent SDK not installed. Run: uv add claude-agent-sdk or pip install claude-agent-sdk"
            ) from e

        # SDK will automatically check for authentication when first used
        # No need to verify here - let it fail gracefully on first call with helpful error

    async def verify_connection(self) -> None:
        """
        Verify that the Claude Code provider is configured correctly by making a simple test call.

        Raises:
            RuntimeError: If the connection test fails.
        """
        try:
            test_messages = [{"role": "user", "content": "test"}]
            await self.call(
                messages=test_messages,
                max_completion_tokens=10,
                temperature=0.0,
                scope="test",
                max_retries=0,
            )
            logger.info("Claude Code connection verified successfully")
        except Exception as e:
            logger.error(f"Claude Code connection verification failed: {e}")
            raise RuntimeError(f"Failed to verify Claude Code connection: {e}") from e

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
        Make an LLM API call with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            response_format: Optional Pydantic model for structured output.
            max_completion_tokens: Maximum tokens in response (ignored by Claude Agent SDK).
            temperature: Sampling temperature (ignored by Claude Agent SDK).
            scope: Scope identifier for tracking.
            max_retries: Maximum retry attempts.
            initial_backoff: Initial backoff time in seconds.
            max_backoff: Maximum backoff time in seconds.
            skip_validation: Return raw JSON without Pydantic validation.
            strict_schema: Use strict JSON schema enforcement (not supported).
            return_usage: If True, return tuple (result, TokenUsage) instead of just result.

        Returns:
            If return_usage=False: Parsed response if response_format is provided, otherwise text content.
            If return_usage=True: Tuple of (result, TokenUsage) with estimated token counts.

        Raises:
            OutputTooLongError: If output exceeds token limits (not supported by Claude Agent SDK).
            Exception: Re-raises API errors after retries exhausted.
        """
        from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

        start_time = time.time()

        # Build system prompt
        system_prompt = ""
        user_content = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_prompt += ("\n\n" + content) if system_prompt else content
            elif role == "user":
                user_content += ("\n\n" + content) if user_content else content
            elif role == "assistant":
                # Claude Agent SDK doesn't support multi-turn easily in query()
                # For now, prepend assistant messages to user content
                user_content += f"\n\n[Previous assistant response: {content}]"

        # Add JSON schema instruction if response_format is provided
        if response_format is not None and hasattr(response_format, "model_json_schema"):
            schema = response_format.model_json_schema()
            schema_instruction = (
                f"\n\nYou must respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}\n\n"
                "Respond with ONLY the JSON, no markdown formatting."
            )
            user_content += schema_instruction

        # Configure SDK options
        options = ClaudeAgentOptions(
            system_prompt=system_prompt if system_prompt else None,
            max_turns=1,  # Single-turn for API-style interactions
            allowed_tools=[],  # Disable tools for standard LLM calls
        )

        # Call Claude Agent SDK
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                # Collect streaming response
                full_text = ""

                async for message in query(prompt=user_content, options=options):
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                full_text += block.text

                # Handle structured output
                if response_format is not None:
                    # Models may wrap JSON in markdown
                    clean_text = full_text
                    if "```json" in full_text:
                        clean_text = full_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in full_text:
                        clean_text = full_text.split("```")[1].split("```")[0].strip()

                    try:
                        json_data = json.loads(clean_text)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Claude Code JSON parse error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        if attempt < max_retries:
                            backoff = min(initial_backoff * (2**attempt), max_backoff)
                            await asyncio.sleep(backoff)
                            last_exception = e
                            continue
                        raise

                    if skip_validation:
                        result = json_data
                    else:
                        result = response_format.model_validate(json_data)
                else:
                    result = full_text

                # Record metrics
                duration = time.time() - start_time
                metrics = get_metrics_collector()

                # Estimate token usage (Claude Agent SDK doesn't report exact counts)
                # Use character count / 4 as rough estimate (1 token ≈ 4 characters)
                estimated_input = sum(len(m.get("content", "")) for m in messages) // 4
                estimated_output = len(full_text) // 4

                metrics.record_llm_call(
                    provider=self.provider,
                    model=self.model,
                    scope=scope,
                    duration=duration,
                    input_tokens=estimated_input,
                    output_tokens=estimated_output,
                    success=True,
                )

                # Log slow calls
                if duration > 10.0:
                    logger.info(
                        f"slow llm call: scope={scope}, model={self.provider}/{self.model}, time={duration:.3f}s"
                    )

                if return_usage:
                    token_usage = TokenUsage(
                        input_tokens=estimated_input,
                        output_tokens=estimated_output,
                        total_tokens=estimated_input + estimated_output,
                    )
                    return result, token_usage

                return result

            except Exception as e:
                last_exception = e

                # Check for authentication errors
                error_str = str(e).lower()
                if "auth" in error_str or "login" in error_str or "credential" in error_str:
                    logger.error(f"Claude Code authentication error: {e}")
                    raise RuntimeError(
                        f"Claude Code authentication failed: {e}\n\n"
                        "Run 'claude auth login' to authenticate with Claude Pro/Max."
                    ) from e

                if attempt < max_retries:
                    backoff = min(initial_backoff * (2**attempt), max_backoff)
                    logger.warning(f"Claude Code error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    await asyncio.sleep(backoff)
                    continue
                else:
                    logger.error(f"Claude Code error after {max_retries + 1} attempts: {e}")
                    raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Claude Code call failed after all retries")

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
        Make an LLM API call with tool/function calling support.

        Note: This is a simplified implementation. Full tool support would require
        integrating with Claude Agent SDK's tool system.

        Args:
            messages: List of message dicts. Can include tool results with role='tool'.
            tools: List of tool definitions in OpenAI format.
            max_completion_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            scope: Scope identifier for tracking.
            max_retries: Maximum retry attempts.
            initial_backoff: Initial backoff time in seconds.
            max_backoff: Maximum backoff time in seconds.
            tool_choice: How to choose tools - "auto", "none", "required", or specific function.

        Returns:
            LLMToolCallResult with content and/or tool_calls.
        """
        # For now, use regular call without tools
        # Full implementation would require mapping OpenAI tool format to Claude Agent SDK tools
        logger.warning(
            "Claude Code provider does not fully support tool calling yet. Falling back to regular text completion."
        )

        result = await self.call(
            messages=messages,
            response_format=None,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            scope=scope,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            max_backoff=max_backoff,
            return_usage=True,
        )

        if isinstance(result, tuple):
            text, usage = result
            return LLMToolCallResult(
                content=text,
                tool_calls=[],
                finish_reason="stop",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )
        else:
            # Fallback if return_usage didn't work as expected
            return LLMToolCallResult(
                content=str(result),
                tool_calls=[],
                finish_reason="stop",
                input_tokens=0,
                output_tokens=0,
            )

    async def cleanup(self) -> None:
        """Clean up resources (no HTTP client to close for Claude Agent SDK)."""
        pass
