"""
OpenAI Codex LLM provider using ChatGPT Plus/Pro OAuth authentication.

This provider enables using ChatGPT Plus/Pro subscriptions for API calls
without separate OpenAI Platform API credits. It uses OAuth tokens from
~/.codex/auth.json and communicates with the ChatGPT backend API.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from hindsight_api.engine.llm_interface import LLMInterface, OutputTooLongError
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage
from hindsight_api.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class CodexLLM(LLMInterface):
    """
    LLM provider using OpenAI Codex OAuth authentication.

    Authenticates using ChatGPT Plus/Pro credentials stored in ~/.codex/auth.json
    and makes API calls to chatgpt.com/backend-api/codex/responses.
    """

    def __init__(
        self,
        provider: str,
        api_key: str,  # Will be ignored, reads from ~/.codex/auth.json
        base_url: str,
        model: str,
        reasoning_effort: str = "low",
        **kwargs: Any,
    ):
        """Initialize Codex LLM provider."""
        super().__init__(provider, api_key, base_url, model, reasoning_effort, **kwargs)

        # Load Codex OAuth credentials
        try:
            self.access_token, self.account_id = self._load_codex_auth()
            logger.info(f"Loaded Codex OAuth credentials for account: {self.account_id}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Codex OAuth credentials from ~/.codex/auth.json: {e}\n\n"
                "To set up Codex authentication:\n"
                "1. Install Codex CLI: npm install -g @openai/codex\n"
                "2. Login: codex auth login\n"
                "3. Verify: ls ~/.codex/auth.json\n\n"
                "Or use a different provider (openai, anthropic, gemini) with API keys."
            ) from e

        # Use ChatGPT backend API endpoint
        if not self.base_url:
            self.base_url = "https://chatgpt.com/backend-api"

        # Normalize model name (strip openai/ prefix if present)
        if self.model.startswith("openai/"):
            self.model = self.model[len("openai/") :]

        # Map reasoning effort to Codex reasoning summary format
        # Codex supports: "auto", "concise", "detailed"
        self.reasoning_summary = self._map_reasoning_effort(reasoning_effort)

        # HTTP client for SSE streaming
        self._client = httpx.AsyncClient(timeout=120.0)

    def _load_codex_auth(self) -> tuple[str, str]:
        """
        Load OAuth credentials from ~/.codex/auth.json.

        Returns:
            Tuple of (access_token, account_id).

        Raises:
            FileNotFoundError: If auth file doesn't exist.
            ValueError: If auth file is invalid.
        """
        auth_file = Path.home() / ".codex" / "auth.json"

        if not auth_file.exists():
            raise FileNotFoundError(
                f"Codex auth file not found: {auth_file}\nRun 'codex auth login' to authenticate with ChatGPT Plus/Pro."
            )

        with open(auth_file) as f:
            data = json.load(f)

        # Validate auth structure
        auth_mode = data.get("auth_mode")
        if auth_mode != "chatgpt":
            raise ValueError(f"Expected auth_mode='chatgpt', got: {auth_mode}")

        tokens = data.get("tokens", {})
        access_token = tokens.get("access_token")
        account_id = tokens.get("account_id")

        if not access_token:
            raise ValueError("No access_token found in Codex auth file. Run 'codex auth login' again.")

        return access_token, account_id

    def _map_reasoning_effort(self, effort: str) -> str:
        """
        Map standard reasoning effort to Codex reasoning summary format.

        Args:
            effort: Standard effort level ("low", "medium", "high", "xhigh").

        Returns:
            Codex reasoning summary: "concise", "detailed", or "auto".
        """
        mapping = {
            "low": "concise",
            "medium": "auto",
            "high": "detailed",
            "xhigh": "detailed",
        }
        return mapping.get(effort.lower(), "auto")

    async def verify_connection(self) -> None:
        """Verify Codex connection by making a simple test call."""
        try:
            logger.info(f"Verifying Codex LLM: model={self.model}, account={self.account_id}...")
            await self.call(
                messages=[{"role": "user", "content": "Say 'ok'"}],
                max_completion_tokens=10,
                max_retries=2,
                initial_backoff=0.5,
                max_backoff=2.0,
            )
            logger.info(f"Codex LLM verified: {self.model}")
        except Exception as e:
            raise RuntimeError(f"Codex LLM connection verification failed for {self.model}: {e}") from e

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
        """Make API call to Codex backend with SSE streaming."""
        start_time = time.time()

        # Prepare system instructions
        system_instruction = ""
        user_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction += ("\n\n" + content) if system_instruction else content
            else:
                user_messages.append(msg)

        # Add JSON schema instruction if response_format is provided
        if response_format is not None and hasattr(response_format, "model_json_schema"):
            schema = response_format.model_json_schema()
            schema_msg = f"\n\nYou must respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"
            system_instruction += schema_msg

        # gpt-5.2-codex only supports "detailed" reasoning summary
        reasoning_summary = "detailed" if "5.2" in self.model else self.reasoning_summary

        # Build Codex request payload
        payload = {
            "model": self.model,
            "instructions": system_instruction,
            "input": [
                {
                    "type": "message",
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                }
                for msg in user_messages
            ],
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "reasoning": {"summary": reasoning_summary},
            "store": False,  # Codex uses stateless mode
            "stream": True,  # SSE streaming
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": str(uuid.uuid4()),
        }

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "OpenAI-Account-ID": self.account_id,
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Origin": "https://chatgpt.com",
        }

        url = f"{self.base_url}/codex/responses"
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                response = await self._client.post(url, json=payload, headers=headers, timeout=120.0)
                response.raise_for_status()

                # Parse SSE stream
                content = await self._parse_sse_stream(response)

                # Handle structured output
                if response_format is not None:
                    # Models may wrap JSON in markdown
                    clean_content = content
                    if "```json" in content:
                        clean_content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        clean_content = content.split("```")[1].split("```")[0].strip()

                    try:
                        json_data = json.loads(clean_content)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Codex JSON parse error (attempt {attempt + 1}/{max_retries + 1}): {e}")
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
                    result = content

                # Record metrics
                duration = time.time() - start_time
                metrics = get_metrics_collector()
                metrics.record_llm_call(
                    provider=self.provider,
                    model=self.model,
                    scope=scope,
                    duration=duration,
                    input_tokens=0,  # Codex doesn't report token counts in SSE
                    output_tokens=0,
                    success=True,
                )

                if return_usage:
                    # Codex doesn't provide token counts, estimate based on content
                    estimated_input = sum(len(m.get("content", "")) for m in messages) // 4
                    estimated_output = len(content) // 4
                    token_usage = TokenUsage(
                        input_tokens=estimated_input,
                        output_tokens=estimated_output,
                        total_tokens=estimated_input + estimated_output,
                    )
                    return result, token_usage

                return result

            except httpx.HTTPStatusError as e:
                last_exception = e
                status_code = e.response.status_code

                # Fast fail on auth errors
                if status_code in (401, 403):
                    logger.error(f"Codex auth error (HTTP {status_code}): {e.response.text[:200]}")
                    raise RuntimeError(
                        "Codex authentication failed. Your OAuth token may have expired.\n"
                        "Run 'codex auth login' to re-authenticate."
                    ) from e

                # Log the actual error message from the API
                error_detail = e.response.text[:500] if hasattr(e.response, "text") else str(e)

                if attempt < max_retries:
                    backoff = min(initial_backoff * (2**attempt), max_backoff)
                    logger.warning(
                        f"Codex HTTP error {status_code} (attempt {attempt + 1}/{max_retries + 1}): {error_detail}"
                    )
                    await asyncio.sleep(backoff)
                    continue
                else:
                    logger.error(
                        f"Codex HTTP error after {max_retries + 1} attempts: Status {status_code}, Detail: {error_detail}"
                    )
                    raise

            except httpx.RequestError as e:
                last_exception = e
                if attempt < max_retries:
                    backoff = min(initial_backoff * (2**attempt), max_backoff)
                    logger.warning(f"Codex connection error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    await asyncio.sleep(backoff)
                    continue
                else:
                    logger.error(f"Codex connection error after {max_retries + 1} attempts: {e}")
                    raise

            except Exception as e:
                logger.error(f"Unexpected Codex error: {type(e).__name__}: {e}")
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Codex call failed after all retries")

    async def _parse_sse_stream(self, response: httpx.Response) -> str:
        """
        Parse Server-Sent Events (SSE) stream from Codex API.

        Args:
            response: HTTP response with SSE stream.

        Returns:
            Extracted text content from stream.
        """
        full_text = ""
        event_type = None

        async for line in response.aiter_lines():
            if not line:
                continue

            # Track event type
            if line.startswith("event: "):
                event_type = line[7:]

            # Parse data
            elif line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)

                    # Extract content based on event type
                    if event_type == "response.text.delta" and "delta" in data:
                        full_text += data["delta"]
                    elif event_type == "response.content_part.delta" and "delta" in data:
                        full_text += data["delta"]
                    # Check for item content
                    elif "item" in data:
                        item = data["item"]
                        if "content" in item:
                            content = item["content"]
                            if isinstance(content, list):
                                for part in content:
                                    if isinstance(part, dict) and "text" in part:
                                        full_text += part["text"]
                            elif isinstance(content, str):
                                full_text += content

                except json.JSONDecodeError:
                    # Skip malformed JSON events
                    pass

        return full_text

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
        Make API call with tool calling support.

        Parses Codex SSE stream to extract tool calls from response.output_item.done events.
        Tools are converted from OpenAI format to Codex format (flat structure at top level).

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
        start_time = time.time()

        # Prepare system instructions
        system_instruction = ""
        user_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction += ("\n\n" + content) if system_instruction else content
            elif role == "tool":
                # Handle tool results
                user_messages.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": f"Tool result: {content}",
                    }
                )
            else:
                user_messages.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": content,
                    }
                )

        # Convert tools to Codex format
        # Codex expects tools with type and name/description/parameters at top level
        codex_tools = []
        for tool in tools:
            func = tool.get("function", {})
            codex_tools.append(
                {
                    "type": "function",
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                }
            )

        # gpt-5.2-codex only supports "detailed" reasoning summary
        reasoning_summary = "detailed" if "5.2" in self.model else self.reasoning_summary

        payload = {
            "model": self.model,
            "instructions": system_instruction,
            "input": user_messages,
            "tools": codex_tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": True,
            "reasoning": {"summary": reasoning_summary},
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": str(uuid.uuid4()),
        }

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "OpenAI-Account-ID": self.account_id,
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Origin": "https://chatgpt.com",
        }

        url = f"{self.base_url}/codex/responses"

        # Debug logging for troubleshooting
        logger.debug(f"Codex tool call request: url={url}, model={payload['model']}, tools={len(codex_tools)}")

        try:
            response = await self._client.post(url, json=payload, headers=headers, timeout=120.0)

            # Log response details on error
            if response.status_code != 200:
                logger.error(f"Codex API error {response.status_code}: {response.text[:500]}")

            response.raise_for_status()

            # Parse SSE for tool calls and content
            content, tool_calls = await self._parse_sse_tool_stream(response)

            duration = time.time() - start_time
            metrics = get_metrics_collector()
            metrics.record_llm_call(
                provider=self.provider,
                model=self.model,
                scope=scope,
                duration=duration,
                input_tokens=0,
                output_tokens=0,
                success=True,
            )

            return LLMToolCallResult(
                content=content,
                tool_calls=tool_calls,
                finish_reason="tool_calls" if tool_calls else "stop",
                input_tokens=0,
                output_tokens=0,
            )

        except Exception as e:
            logger.error(f"Codex tool call error: {e}")
            raise

    async def _parse_sse_tool_stream(self, response: httpx.Response) -> tuple[str | None, list[LLMToolCall]]:
        """
        Parse SSE stream for tool calls and content.

        Returns:
            Tuple of (content, tool_calls).
        """
        content = ""
        tool_calls: list[LLMToolCall] = []
        event_type = None

        async for line in response.aiter_lines():
            if not line:
                continue

            if line.startswith("event: "):
                event_type = line[7:]

            elif line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)

                    # Extract text content
                    if event_type == "response.text.delta" and "delta" in data:
                        content += data["delta"]

                    # Extract completed tool calls from response.output_item.done
                    elif event_type == "response.output_item.done":
                        item = data.get("item", {})
                        if item.get("type") == "function_call" and item.get("status") == "completed":
                            tool_name = item.get("name", "")
                            arguments_str = item.get("arguments", "{}")
                            call_id = item.get("call_id", "")

                            try:
                                arguments = json.loads(arguments_str)
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse tool arguments: {arguments_str}")
                                arguments = {}

                            tool_calls.append(
                                LLMToolCall(
                                    id=call_id,
                                    name=tool_name,
                                    arguments=arguments,
                                )
                            )

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse SSE data: {e}, data_str: {data_str[:200]}")

        return content if content else None, tool_calls

    async def cleanup(self) -> None:
        """Clean up HTTP client."""
        await self._client.aclose()
