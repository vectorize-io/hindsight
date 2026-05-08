"""
LiteLLM Router LLM provider — ordered fallback across multiple deployments.

Wraps ``litellm.Router`` with a chain of deployments. Requests are tried against
the primary deployment first; on transient errors (rate-limit, timeout, 5xx) the
Router falls back to subsequent deployments in declared order. Chain entries are
modelled as distinct LiteLLM ``model_name`` groups so that the ``fallbacks`` map
expresses the linear chain explicitly — this gives deterministic fallback order
rather than the load-balancing behaviour you get from same-group deployments.

Auth-style errors (401/403) are not retried across deployments, since fallback
on a misconfigured key would just propagate the failure.

References:
    - LiteLLM Router: https://docs.litellm.ai/docs/routing
    - Router fallbacks: https://docs.litellm.ai/docs/routing#fallbacks

Chain entry shape (matches ``HINDSIGHT_API_LLM_LITELLMROUTER_CHAIN``)::

    [
      {"provider": "openai", "model": "gpt-4o-mini", "api_key": "sk-..."},
      {"provider": "anthropic", "model": "claude-sonnet-4-5", "api_key": "sk-ant-..."}
    ]
"""

import asyncio
import json
import logging
import time
from typing import Any

from hindsight_api.engine.llm_interface import LLMInterface, OutputTooLongError
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage
from hindsight_api.metrics import get_metrics_collector
from hindsight_api.worker.stage import set_stage

logger = logging.getLogger(__name__)


# Hindsight provider name → LiteLLM model prefix.
# Providers not listed are treated as OpenAI-compatible: the configured base_url
# carries the routing and the model is sent under the "openai/" prefix.
_LITELLM_PROVIDER_PREFIX = {
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "vertexai": "vertex_ai",
    "groq": "groq",
    "deepseek": "deepseek",
    "openrouter": "openrouter",
    "bedrock": "bedrock",
    "ollama": "ollama_chat",
    "litellm": None,  # caller already provided a fully-qualified model string
}

# Primary group name. All chain entries fall back to subsequent indices.
_PRIMARY_GROUP = "hindsight-chain-0"


def _build_litellm_model(provider: str, model: str) -> str:
    """Translate a Hindsight (provider, model) pair into a LiteLLM model string."""
    if "/" in model:
        return model  # caller pre-qualified the model, respect it
    prefix = _LITELLM_PROVIDER_PREFIX.get(provider.lower(), "openai")
    if prefix is None:
        return model
    return f"{prefix}/{model}"


def _build_model_list(chain: list[dict[str, Any]], timeout: float) -> list[dict[str, Any]]:
    """Translate a Hindsight chain into a LiteLLM Router model_list."""
    model_list: list[dict[str, Any]] = []
    for i, entry in enumerate(chain):
        provider = entry["provider"]
        model = entry["model"]
        litellm_params: dict[str, Any] = {
            "model": _build_litellm_model(provider, model),
            "timeout": timeout,
        }
        if entry.get("api_key"):
            litellm_params["api_key"] = entry["api_key"]
        if entry.get("base_url"):
            litellm_params["api_base"] = entry["base_url"]
        model_list.append(
            {
                "model_name": f"hindsight-chain-{i}",
                "litellm_params": litellm_params,
            }
        )
    return model_list


def _build_fallbacks(chain_length: int) -> list[dict[str, list[str]]]:
    """Map the primary group to every subsequent chain index in declared order."""
    if chain_length <= 1:
        return []
    return [{_PRIMARY_GROUP: [f"hindsight-chain-{i}" for i in range(1, chain_length)]}]


class LiteLLMRouterLLM(LLMInterface):
    """
    LLM provider backed by ``litellm.Router`` with ordered fallback.

    Each entry in the chain becomes a distinct LiteLLM model_group; the primary
    group is wired to fall back through the remaining groups in declared order.
    Requests are always issued against the primary group — Router internally
    re-issues against the fallback groups when the primary fails.
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        chain: list[dict[str, Any]],
        reasoning_effort: str = "low",
        timeout: float = 300.0,
        **kwargs: Any,
    ):
        super().__init__(provider, api_key, base_url, model, reasoning_effort, **kwargs)
        if not chain:
            raise ValueError("LiteLLMRouterLLM requires a non-empty chain")
        self.timeout = timeout
        self.chain = chain
        self._litellm: Any = None
        self._router: Any = None

        try:
            import litellm
            from litellm import Router

            self._litellm = litellm
            litellm.suppress_debug_info = True  # type: ignore[assignment]
            litellm.drop_params = True  # type: ignore[assignment]
            logging.getLogger("LiteLLM").setLevel(logging.WARNING)
            logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
        except ImportError as e:
            raise RuntimeError("LiteLLM SDK not installed. Run: uv add litellm or pip install litellm") from e

        model_list = _build_model_list(chain, timeout=timeout)
        fallbacks = _build_fallbacks(len(chain))
        self._router = Router(
            model_list=model_list,
            fallbacks=fallbacks,
            num_retries=0,  # outer loop in call()/call_with_tools() owns retries
            allowed_fails=1,  # cooldown a deployment after a single failure within a window
            cooldown_time=60,
        )

        chain_summary = ", ".join(f"{entry['provider']}/{entry['model']}" for entry in chain)
        logger.info(f"LiteLLM Router initialized with {len(chain)} deployment(s): [{chain_summary}]")

    async def verify_connection(self) -> None:
        try:
            await self.call(
                messages=[{"role": "user", "content": "test"}],
                max_completion_tokens=50,
                temperature=0.0,
                scope="verification",
                max_retries=0,
            )
            logger.info("LiteLLM Router connection verified successfully")
        except OutputTooLongError:
            logger.info("LiteLLM Router connection verified successfully (response truncated)")
        except Exception as e:
            logger.error(f"LiteLLM Router connection verification failed: {e}")
            raise RuntimeError(f"Failed to verify LiteLLM Router connection: {e}") from e

    def _build_call_kwargs(
        self,
        messages: list[dict[str, Any]],
        max_completion_tokens: int | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": _PRIMARY_GROUP,
            "messages": messages,
        }
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        return kwargs

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
        start_time = time.time()
        call_kwargs = self._build_call_kwargs(messages, max_completion_tokens, temperature)

        if response_format is not None and hasattr(response_format, "model_json_schema"):
            schema = response_format.model_json_schema()
            call_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": getattr(response_format, "__name__", "response"),
                    "schema": schema,
                    "strict": strict_schema,
                },
            }

        last_exception: Exception | None = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                set_stage(f"llm.litellmrouter.{scope}.attempt={attempt + 1}/{max_retries + 1}")
            try:
                response = await self._router.acompletion(**call_kwargs)

                content = response.choices[0].message.content or ""
                finish_reason = response.choices[0].finish_reason
                # Router exposes the deployment that actually answered via response._hidden_params.
                deployment_model = (
                    getattr(response, "_hidden_params", {}).get("model")
                    if hasattr(response, "_hidden_params")
                    else None
                ) or call_kwargs["model"]

                if finish_reason == "length":
                    raise OutputTooLongError("LiteLLM Router response was truncated due to token limit")

                if response_format is not None:
                    clean_content = content
                    if "```json" in content:
                        clean_content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        clean_content = content.split("```")[1].split("```")[0].strip()
                    try:
                        json_data = json.loads(clean_content)
                    except json.JSONDecodeError:
                        json_data = json.loads(content)
                    result = json_data if skip_validation else response_format.model_validate(json_data)
                else:
                    result = content

                input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
                total_tokens = input_tokens + output_tokens

                duration = time.time() - start_time
                metrics = get_metrics_collector()
                metrics.record_llm_call(
                    provider=self.provider,
                    model=deployment_model,
                    scope=scope,
                    duration=duration,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=True,
                )

                from hindsight_api.tracing import _serialize_for_span, get_span_recorder

                span_recorder = get_span_recorder()
                span_recorder.record_llm_call(
                    provider=self.provider,
                    model=deployment_model,
                    scope=scope,
                    messages=messages,
                    response_content=_serialize_for_span(result),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration=duration,
                    finish_reason=finish_reason,
                    error=None,
                )

                if duration > 10.0:
                    logger.info(
                        f"slow llm call: scope={scope}, model={self.provider}/{deployment_model}, "
                        f"input_tokens={input_tokens}, output_tokens={output_tokens}, "
                        f"time={duration:.3f}s"
                    )

                if return_usage:
                    return result, TokenUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                    )
                return result

            except OutputTooLongError:
                raise

            except json.JSONDecodeError as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning("LiteLLM Router returned invalid JSON, retrying...")
                    backoff = min(initial_backoff * (2**attempt), max_backoff)
                    await asyncio.sleep(backoff)
                    continue
                logger.error(f"LiteLLM Router returned invalid JSON after {max_retries + 1} attempts")
                raise

            except Exception as e:
                error_str = str(e).lower()
                if "401" in error_str or "403" in error_str or "unauthorized" in error_str:
                    logger.error(f"LiteLLM Router auth error, not retrying: {e}")
                    raise

                last_exception = e
                if attempt < max_retries:
                    is_retryable = any(
                        keyword in error_str
                        for keyword in ("rate", "limit", "timeout", "connection", "500", "502", "503", "529")
                    )
                    if is_retryable:
                        backoff = min(initial_backoff * (2**attempt), max_backoff)
                        jitter = backoff * 0.2 * (2 * (time.time() % 1) - 1)
                        await asyncio.sleep(backoff + jitter)
                        continue

                logger.error(f"LiteLLM Router API error after {attempt + 1} attempts: {e}")
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("LiteLLM Router call failed after all retries")

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
        start_time = time.time()
        call_kwargs = self._build_call_kwargs(messages, max_completion_tokens, temperature)
        call_kwargs["tools"] = tools
        call_kwargs["tool_choice"] = tool_choice

        last_exception: Exception | None = None
        for attempt in range(max_retries + 1):
            if attempt > 0:
                set_stage(f"llm.litellmrouter.tools.attempt={attempt + 1}/{max_retries + 1}")
            try:
                response = await self._router.acompletion(**call_kwargs)

                message = response.choices[0].message
                content = message.content
                finish_reason = response.choices[0].finish_reason
                deployment_model = (
                    getattr(response, "_hidden_params", {}).get("model")
                    if hasattr(response, "_hidden_params")
                    else None
                ) or call_kwargs["model"]

                tool_calls: list[LLMToolCall] = []
                if message.tool_calls:
                    for tc in message.tool_calls:
                        arguments = tc.function.arguments
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)
                        tool_calls.append(
                            LLMToolCall(
                                id=tc.id,
                                name=tc.function.name,
                                arguments=arguments,
                            )
                        )

                input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(response.usage, "completion_tokens", 0) or 0

                duration = time.time() - start_time
                metrics = get_metrics_collector()
                metrics.record_llm_call(
                    provider=self.provider,
                    model=deployment_model,
                    scope=scope,
                    duration=duration,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=True,
                )

                from hindsight_api.tracing import get_span_recorder

                span_recorder = get_span_recorder()
                tool_calls_dict = (
                    [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls]
                    if tool_calls
                    else None
                )
                span_recorder.record_llm_call(
                    provider=self.provider,
                    model=deployment_model,
                    scope=scope,
                    messages=messages,
                    response_content=content,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration=duration,
                    finish_reason=finish_reason,
                    error=None,
                    tool_calls=tool_calls_dict,
                )

                return LLMToolCallResult(
                    content=content,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason or ("tool_calls" if tool_calls else "stop"),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            except Exception as e:
                error_str = str(e).lower()
                if "401" in error_str or "403" in error_str or "unauthorized" in error_str:
                    raise

                last_exception = e
                if attempt < max_retries:
                    is_retryable = any(
                        keyword in error_str
                        for keyword in ("rate", "limit", "timeout", "connection", "500", "502", "503", "529")
                    )
                    if is_retryable:
                        await asyncio.sleep(min(initial_backoff * (2**attempt), max_backoff))
                        continue

                logger.error(f"LiteLLM Router tool call error after {attempt + 1} attempts: {e}")
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("LiteLLM Router tool call failed after all retries")

    async def cleanup(self) -> None:
        """Clean up resources."""
        pass
