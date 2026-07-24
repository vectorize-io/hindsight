"""
OpenAI-compatible LLM provider supporting OpenAI, Groq, Ollama, LMStudio, MiniMax, DeepSeek,
and Opencode Go.

This provider handles all OpenAI API-compatible models including:
- OpenAI: GPT-4, GPT-4o, GPT-5, o1, o3 (reasoning models)
- Groq: Fast inference with seed control and service tiers
- Ollama: Local models with native streaming API support
- LMStudio: Local models with OpenAI-compatible API
- MiniMax: MiniMax-M3 / MiniMax-M2.7 models with 1M context window
- DeepSeek: deepseek-v4-flash / deepseek-v4-pro / deepseek-chat / deepseek-reasoner via api.deepseek.com
- Opencode Go: deepseek-v4-flash via https://opencode.ai/zen/go/v1

Features:
- Reasoning models with extended thinking (o1, o3, GPT-5 families)
- Strict JSON schema enforcement (OpenAI)
- Provider-specific parameters (Groq seed, service tier)
- Native Ollama streaming for better structured output
- Automatic token limit handling per model family
"""

import asyncio
import io
import json
import logging
import os
import re
import time
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import parse_qs, urlparse, urlunparse

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, LengthFinishReasonError

from hindsight_api.config import DEFAULT_LLM_TIMEOUT, ENV_LLM_TIMEOUT
from hindsight_api.engine.bank_attribution import apply_bank_attribution
from hindsight_api.engine.llm_interface import (
    LLM_TOOL_CHOICE_AUTO,
    LLMInterface,
    LLMToolChoice,
    LLMToolChoiceMode,
    OutputTooLongError,
    ProviderRateLimitResetError,
)
from hindsight_api.engine.llm_trace import LLMResponseUsage, stash_response_usage
from hindsight_api.engine.providers.llm_debug import dump_request_on_4xx
from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult, TokenUsage
from hindsight_api.engine.structured_output import strict_json_schema
from hindsight_api.metrics import get_metrics_collector
from hindsight_api.worker.stage import set_stage

logger = logging.getLogger(__name__)

# Seed applied to every Groq request for deterministic behavior
DEFAULT_LLM_SEED = 4242
JSON_MODE_USER_HINT = "Return valid json only."
DEFAULT_VERIFICATION_MAX_COMPLETION_TOKENS = 512


def _validate_ollama_num_ctx(value: Any) -> int | None:
    """Validate a native Ollama context-window override."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"ollama_num_ctx must be a positive integer, got {value!r}")
    if value < 1:
        raise ValueError(f"ollama_num_ctx must be >= 1, got {value}")
    return value


# Provider implementations that advertise tool_choice="required"
# but silently ignore it: instead of forcing a tool call they return
# finish_reason "stop"/"tool_calls" with an EMPTY tool_calls array and no error.
# Reflect's agent loop then sees no tool call, runs synthesis with no retrieval,
# and answers "I don't have information" even when the bank holds the answer.
# See issues #1563 (LM Studio), #1179 (LM Studio + Qwen), #1877 (vLLM with
# --enable-auto-tool-choice). The generic OpenAI provider is intentionally not
# inferred from its URL: custom OpenAI-compatible endpoints can implement the
# required-tool contract, and silently downgrading them changes request semantics.
# llama-server (the "llamacpp" provider) honors "required" correctly and is
# intentionally excluded (#1179).
_TOOL_CHOICE_REQUIRED_UNSUPPORTED_PROVIDERS = frozenset({"lmstudio", "ollama"})

# Local providers whose OpenAI-compatible surface always lives under a `/v1`
# path (LM Studio: http://localhost:1234/v1, Ollama: http://localhost:11434/v1).
# For these we know the exact endpoint shape, so a bare host base URL can be
# normalized safely. Cloud/proxy endpoints are left untouched — their path is
# provider-specific and must be supplied verbatim.
_V1_PATH_LOCAL_PROVIDERS = frozenset({"lmstudio", "ollama"})


def _ensure_v1_base_url(base_url: str) -> str:
    """Append the OpenAI-compatible ``/v1`` prefix to a bare local base URL.

    LM Studio's server UI advertises its address as ``http://localhost:1234``,
    so users commonly set ``HINDSIGHT_API_LLM_BASE_URL`` to that bare host. The
    OpenAI SDK then POSTs to ``<host>/chat/completions`` and LM Studio rejects it
    with ``Unexpected endpoint or method`` — its OpenAI-compatible routes live
    under ``/v1``. Only a base URL with no meaningful path (bare host or a lone
    trailing slash) is rewritten; anything with an explicit path (e.g. a reverse
    proxy mount or an already-correct ``/v1``) is returned unchanged. See #2922.
    """
    parsed = urlparse(base_url)
    if parsed.path.strip("/"):
        return base_url
    return urlunparse(parsed._replace(path="/v1"))


class ProviderResponseError(RuntimeError):
    """Raised when a provider returns a success response without usable content."""

    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


def _is_json(text: str) -> bool:
    """True if ``text`` parses as a JSON value."""
    try:
        json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    return True


def _outer_json_span(content: str) -> str | None:
    """Return the outermost ``{...}`` / ``[...]`` span if it parses as JSON, else None.

    Fallback for responses where fences are partial/absent or the model wrapped
    the JSON in surrounding prose. Only returned when it is valid JSON so callers
    never receive a worse candidate than the raw content.
    """
    starts = [i for i in (content.find("{"), content.find("[")) if i >= 0]
    ends = [i for i in (content.rfind("}"), content.rfind("]")) if i >= 0]
    if not starts or not ends:
        return None
    start, end = min(starts), max(ends)
    if end <= start:
        return None
    candidate = content[start : end + 1].strip()
    return candidate if _is_json(candidate) else None


def _strip_code_fences(content: str) -> str:
    """Strip markdown code fences from LLM response if present.

    Many LLM providers (MiniMax, some Ollama models, Claude via proxies)
    wrap JSON responses in ```json ... ``` fences even when json_object
    response format is requested. Fences are detected by line (a closing
    ``` must sit alone on its line) so triple-backticks *inside* JSON string
    values do not truncate the payload. When the stripped candidate is not
    valid JSON (partial fence, prose-wrapped output, truncated response), fall
    back to the outermost parseable JSON span. Returns the original content
    unchanged if no better candidate is found.
    """
    candidate = content
    if "```" in content:
        lines = content.split("\n")
        # Find first line that starts a code fence (``` optionally followed by language)
        fence_start = next((i for i, line in enumerate(lines) if line.startswith("```")), None)
        if fence_start is not None:
            # Find matching closing fence (``` alone or with trailing whitespace)
            fence_end = next(
                (j for j in range(fence_start + 1, len(lines)) if lines[j].strip() == "```"),
                None,
            )
            if fence_end is not None:
                candidate = "\n".join(lines[fence_start + 1 : fence_end]).strip()

    if _is_json(candidate):
        return candidate

    # Fence stripping did not yield valid JSON — try to recover the outer JSON span.
    span = _outer_json_span(content)
    if span is not None:
        return span

    return candidate


# Reasoning/thinking tags emitted by extended-thinking models. Some providers
# (e.g. MiniMax-M3) leak the chain-of-thought wrapped in these tags into the
# response body instead of a separate reasoning_content field. Each entry is
# (open_tag, close_tag); the open tag also matches when the close tag is missing
# (truncated output) so a dangling block is removed to end-of-string.
_REASONING_TAG_PAIRS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<thought>", "</thought>"),
    ("<reasoning>", "</reasoning>"),
    ("|startthink|", "|endthink|"),
)


def _strip_reasoning_tags(text: str) -> str:
    """Strip extended-thinking/reasoning blocks from an LLM response.

    Removes the full set of tag styles emitted by reasoning models:
    ``<think>``, ``<thinking>``, ``<thought>``, ``<reasoning>`` and the
    ``|startthink|...|endthink|`` markers. Both the structured (JSON) path and
    the free-form path must call this — otherwise a non-structured response
    (e.g. a mental-model markdown blob from MiniMax-M3) leaks the raw
    ``<think>...</think>`` verbatim into stored memories.

    Handles two cases:
    1. Closed blocks: ``<think>...</think>`` removed wherever they appear.
    2. Unclosed blocks: a dangling ``<think>`` with no closing tag (model output
       truncated mid-thought) is removed from the open tag to end-of-string.

    Returns the input unchanged (modulo surrounding whitespace) when no tags are
    present.
    """
    if not text:
        return text
    for open_tag, close_tag in _REASONING_TAG_PAIRS:
        open_re = re.escape(open_tag)
        close_re = re.escape(close_tag)
        # Closed blocks first, then any remaining unclosed (truncated) block.
        text = re.sub(rf"{open_re}.*?{close_re}", "", text, flags=re.DOTALL)
        text = re.sub(rf"{open_re}.*", "", text, flags=re.DOTALL)
    return text.strip()


def _response_get(response: Any, key: str, default: Any = None) -> Any:
    if isinstance(response, dict):
        return response.get(key, default)
    return getattr(response, key, default)


def _response_to_dict(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        try:
            return response.model_dump(mode="json")
        except Exception:
            return {}
    return {}


def _summarize_provider_error_payload(error: Any, max_len: int = 400) -> str:
    if error is None:
        return "<none>"
    if isinstance(error, dict):
        message = error.get("message") or error.get("error") or error
        err_type = error.get("type")
        param = error.get("param")
        summary = str(message)
        details = [f"type={err_type}" if err_type else "", f"param={param}" if param else ""]
        details = [d for d in details if d]
        if details:
            summary = f"{summary} ({', '.join(details)})"
    else:
        summary = str(error)
    if len(summary) > max_len:
        summary = summary[:max_len] + "...TRUNCATED"
    return summary


def _finish_reason_for_choice(choice: Any) -> Any:
    return _response_get(choice, "finish_reason")


def _message_for_choice(choice: Any) -> Any:
    return _response_get(choice, "message")


def _message_content(message: Any) -> Any:
    return _response_get(message, "content")


def _message_tool_calls(message: Any) -> Any:
    return _response_get(message, "tool_calls")


def _message_refusal(message: Any) -> Any:
    return _response_get(message, "refusal")


def _first_choice_or_error(response: Any, *, provider: str, model: str, scope: str) -> Any:
    """Return the first choice or raise a clear error for malformed success responses."""

    data = _response_to_dict(response)
    error_payload = _response_get(response, "error") or data.get("error")
    if error_payload:
        raise ProviderResponseError(
            f"Provider returned error payload ({provider}/{model}, scope={scope}): "
            f"{_summarize_provider_error_payload(error_payload)}",
            retryable=False,
        )

    choices = _response_get(response, "choices")
    if not choices:
        raise ProviderResponseError(
            f"Provider returned no choices ({provider}/{model}, scope={scope})",
            retryable=True,
        )
    return choices[0]


def _content_or_error(response: Any, *, provider: str, model: str, scope: str) -> tuple[str, Any]:
    """Extract message.content while turning provider shape issues into useful errors."""

    choice = _first_choice_or_error(response, provider=provider, model=model, scope=scope)
    message = _message_for_choice(choice)
    finish_reason = _finish_reason_for_choice(choice)
    if message is None:
        raise ProviderResponseError(
            f"Provider returned a choice without message ({provider}/{model}, scope={scope}, "
            f"finish_reason={finish_reason})",
            retryable=True,
        )

    content = _message_content(message)
    if content is None or content == "":
        tool_calls = _message_tool_calls(message)
        refusal = _message_refusal(message)
        retryable = finish_reason not in {"content_filter"}
        raise ProviderResponseError(
            f"Provider returned empty message content ({provider}/{model}, scope={scope}, "
            f"finish_reason={finish_reason}, has_tool_calls={bool(tool_calls)}, "
            f"refusal={bool(refusal)})",
            retryable=retryable,
        )
    return content, choice


def _usage_from_openai_response(response: Any) -> LLMResponseUsage:
    """Extract prompt/completion/cached token counts from an OpenAI-shaped usage block."""
    usage = getattr(response, "usage", None)
    input_tokens = (usage.prompt_tokens or 0) if usage else 0
    output_tokens = (usage.completion_tokens or 0) if usage else 0
    cached_tokens = 0
    if usage and getattr(usage, "prompt_tokens_details", None):
        cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
    return LLMResponseUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
    )


def _ensure_json_word_in_user_message(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Some OpenAI-compatible gateways require 'json' in a user message for json_object mode."""

    normalized = [dict(message) for message in messages]
    user_indexes = [
        index
        for index, message in enumerate(normalized)
        if message.get("role") == "user" and isinstance(message.get("content"), str)
    ]
    if not user_indexes:
        normalized.append({"role": "user", "content": JSON_MODE_USER_HINT})
        return normalized

    if any("json" in normalized[index]["content"] for index in user_indexes):
        return normalized

    last_user = user_indexes[-1]
    normalized[last_user]["content"] = f"{JSON_MODE_USER_HINT}\n\n{normalized[last_user]['content']}"
    return normalized


def _summarize_status_error(e: APIStatusError, body_max: int = 400) -> str:
    """Render an APIStatusError with status code + truncated response body.

    Without this, retry loops only log "API error after N attempts" with the
    bare exception message — losing the provider's actual error payload, which
    is the only thing that explains *why* a request failed (rate limit reason,
    invalid tool schema, model overloaded, etc.).
    """
    body: Any = getattr(e, "body", None)
    if body is None:
        try:
            body = e.response.text
        except Exception:
            body = None
    if isinstance(body, (dict, list)):
        try:
            body_str = json.dumps(body, default=str, ensure_ascii=False)
        except Exception:
            body_str = str(body)
    else:
        body_str = str(body or "").strip()
    if len(body_str) > body_max:
        body_str = body_str[:body_max] + "...TRUNCATED"
    return f"HTTP {e.status_code}: {body_str or '<no body>'}"


_RATE_LIMIT_RESET_AT_RE = re.compile(
    r"\breset at\s+"
    r"(?P<reset_at>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\s*(?:Z|[+-]\d{2}:?\d{2}))?)",
    re.IGNORECASE,
)
_RATE_LIMIT_WINDOW_RE = re.compile(
    r"\b(?:for|in)\s+(?P<amount>\d+)\s*(?P<unit>second|minute|hour|day)s?\b",
    re.IGNORECASE,
)


def _status_error_body_text(e: APIStatusError) -> str:
    body: Any = getattr(e, "body", None)
    if body is None:
        try:
            body = e.response.text
        except Exception:
            body = None
    if isinstance(body, (dict, list)):
        try:
            return json.dumps(body, default=str, ensure_ascii=False)
        except Exception:
            return str(body)
    return str(body or "").strip()


def _parse_retry_after_header(value: str | None, now: datetime) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    try:
        seconds = float(raw)
    except ValueError:
        seconds = -1.0
    if seconds >= 0:
        return now + timedelta(seconds=seconds)

    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_reset_at_datetime(value: str) -> datetime | None:
    raw = value.strip().replace(" ", "T")
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    elif re.search(r"[+-]\d{4}$", raw):
        raw = f"{raw[:-2]}:{raw[-2:]}"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # Some providers (z.ai included) return a wall-clock reset timestamp
        # without a zone. Interpret it in the host's local zone so logs, status
        # pages, and the queued next_retry_at describe the same operator-facing
        # clock instead of silently shifting by UTC offset.
        parsed = parsed.astimezone()
    return parsed.astimezone(UTC)


def _rate_limit_retry_at(e: APIStatusError) -> datetime | None:
    now = datetime.now(UTC)
    response = getattr(e, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        retry_at = _parse_retry_after_header(headers.get("retry-after") or headers.get("Retry-After"), now)
        if retry_at is not None and retry_at > now:
            return retry_at

    body_text = _status_error_body_text(e)
    reset_match = _RATE_LIMIT_RESET_AT_RE.search(body_text)
    if reset_match:
        retry_at = _parse_reset_at_datetime(reset_match.group("reset_at"))
        if retry_at is not None and retry_at > now:
            return retry_at

    window_match = _RATE_LIMIT_WINDOW_RE.search(body_text)
    if not window_match:
        return None
    amount = int(window_match.group("amount"))
    unit = window_match.group("unit").lower()
    if unit == "second":
        seconds = amount
    elif unit == "minute":
        seconds = amount * 60
    elif unit == "hour":
        seconds = amount * 3600
    else:
        seconds = amount * 86400
    return now + timedelta(seconds=seconds)


def _raise_provider_quota_defer(
    e: APIStatusError, *, provider: str, model: str, scope: str, max_backoff: float
) -> None:
    if e.status_code != 429:
        return
    retry_at = _rate_limit_retry_at(e)
    if retry_at is None:
        return
    if (retry_at - datetime.now(UTC)).total_seconds() <= max_backoff:
        return
    summary = _summarize_status_error(e)
    raise ProviderRateLimitResetError(
        retry_at=retry_at,
        message=(
            f"Provider quota exhausted ({provider}/{model}, scope={scope}); retry at {retry_at.isoformat()}: {summary}"
        ),
    ) from e


class OpenAICompatibleLLM(LLMInterface):
    """
    LLM provider for OpenAI-compatible APIs.

    Supports:
    - OpenAI: Standard models (GPT-4, GPT-4o) and reasoning models (o1, o3, GPT-5)
    - Groq: Fast inference with seed control and service tiers
    - Ollama: Local models with native streaming API for better structured output
    - LMStudio: Local models with OpenAI-compatible API
    - MiniMax: MiniMax-M3 / MiniMax-M2.7 models via OpenAI-compatible API (https://api.minimax.io/v1)
    - DeepSeek: deepseek-v4-flash / deepseek-v4-pro / deepseek-chat / deepseek-reasoner via https://api.deepseek.com
    - opencode-go: deepseek-v4-flash via https://opencode.ai/zen/go/v1
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        reasoning_effort: str = "low",
        timeout: float | None = None,
        groq_service_tier: str | None = None,
        extra_body: dict[str, Any] | None = None,
        *,
        ollama_num_ctx: int | None = None,
        **kwargs: Any,
    ):
        """
        Initialize OpenAI-compatible LLM provider.

        Args:
            provider: Provider name ("openai", "groq", "ollama", "lmstudio", "opencode-go", etc.).
            api_key: API key (optional for ollama/lmstudio).
            base_url: Base URL for the API (uses defaults for groq/ollama/lmstudio if empty).
            model: Model name.
            reasoning_effort: Reasoning effort level for supported models ("low", "medium", "high").
            timeout: Request timeout in seconds (uses env var or 120s default).
            groq_service_tier: Groq service tier ("on_demand", "flex", "auto").
            extra_body: Extra body params merged into every API call.
            ollama_num_ctx: Native Ollama context window override. None lets Ollama use
                the model/server default.
            **kwargs: Additional provider-specific parameters.
        """
        super().__init__(provider, api_key, base_url, model, reasoning_effort, **kwargs)

        # Validate provider
        valid_providers = [
            "openai",
            "groq",
            "ollama",
            "ollama-cloud",
            "lmstudio",
            "llamacpp",
            "minimax",
            "deepseek",
            "volcano",
            "openrouter",
            "requesty",
            "zai",
            "opencode-go",
            "atlas",
            "fireworks",
        ]
        if self.provider not in valid_providers:
            raise ValueError(f"OpenAICompatibleLLM only supports: {', '.join(valid_providers)}. Got: {self.provider}")

        # Set default base URLs
        if not self.base_url:
            if self.provider == "groq":
                self.base_url = "https://api.groq.com/openai/v1"
            elif self.provider == "ollama":
                self.base_url = "http://localhost:11434/v1"
            elif self.provider == "ollama-cloud":
                self.base_url = "https://ollama.com/v1"
            elif self.provider == "lmstudio":
                self.base_url = "http://localhost:1234/v1"
            elif self.provider == "minimax":
                self.base_url = "https://api.minimax.io/v1"
            elif self.provider == "deepseek":
                self.base_url = "https://api.deepseek.com"
            elif self.provider == "openrouter":
                self.base_url = "https://openrouter.ai/api/v1"
            elif self.provider == "requesty":
                self.base_url = "https://router.requesty.ai/v1"
            elif self.provider == "zai":
                self.base_url = "https://api.z.ai/api/coding/paas/v4"
            elif self.provider == "opencode-go":
                self.base_url = "https://opencode.ai/zen/go/v1"
            elif self.provider == "atlas":
                self.base_url = "https://api.atlascloud.ai/v1"
            elif self.provider == "fireworks":
                # OpenAI-compatible inference host (online path). The batch API
                # lives on a separate control-plane host — see FireworksLLM.
                self.base_url = "https://api.fireworks.ai/inference/v1"

        # Normalize bare local base URLs (e.g. a user pasting the address shown
        # in the LM Studio UI) so the OpenAI SDK targets the `/v1` routes. See #2922.
        if self.provider in _V1_PATH_LOCAL_PROVIDERS and self.base_url:
            self.base_url = _ensure_v1_base_url(self.base_url)

        # For ollama/lmstudio, use dummy key if not provided
        if self.provider in ("ollama", "lmstudio") and not self.api_key:
            self.api_key = "local"

        # Validate API key for cloud providers
        if (
            self.provider
            in (
                "openai",
                "groq",
                "minimax",
                "deepseek",
                "openrouter",
                "requesty",
                "zai",
                "opencode-go",
                "atlas",
                "ollama-cloud",
            )
            and not self.api_key
        ):
            raise ValueError(f"API key is required for {self.provider}")

        # Service tier configuration (from config, not env vars)
        self.groq_service_tier = groq_service_tier
        self.openai_service_tier = kwargs.get("openai_service_tier")
        self.ollama_num_ctx = _validate_ollama_num_ctx(ollama_num_ctx)
        # User-configured extra body params (merged into every API call)
        self._config_extra_body = extra_body or {}

        # Get timeout config
        self.timeout = timeout or float(os.getenv(ENV_LLM_TIMEOUT, str(DEFAULT_LLM_TIMEOUT)))

        # Create OpenAI client — extract query params from base_url (e.g. Azure api-version)
        client_kwargs: dict[str, Any] = {"api_key": self.api_key, "max_retries": 0}
        if self.base_url:
            parsed = urlparse(self.base_url)
            if parsed.query:
                clean_url = urlunparse(parsed._replace(query=""))
                client_kwargs["base_url"] = clean_url
                default_query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                client_kwargs["default_query"] = default_query
                self.base_url = clean_url
            else:
                client_kwargs["base_url"] = self.base_url
        if self.timeout:
            client_kwargs["timeout"] = self.timeout

        self._client = AsyncOpenAI(**client_kwargs)
        logger.info(
            f"OpenAI-compatible client initialized: provider={self.provider}, model={self.model}, "
            f"base_url={self.base_url or 'default'}"
        )

    def _drops_tool_choice_required(self) -> bool:
        """Whether this endpoint silently ignores ``tool_choice="required"``.

        Only explicitly identified provider implementations are classified as
        unsupported. A custom base URL does not identify endpoint capabilities:
        an OpenAI-compatible endpoint may correctly enforce required tool calls,
        and replacing ``required`` with ``auto`` would violate the caller's named
        tool choice after the tools list has been narrowed.
        """
        return self.provider in _TOOL_CHOICE_REQUIRED_UNSUPPORTED_PROVIDERS

    def _verification_max_completion_tokens(self) -> int:
        """Return the startup verification budget for OpenAI-compatible gateways."""
        return DEFAULT_VERIFICATION_MAX_COMPLETION_TOKENS

    async def verify_connection(self) -> None:
        """
        Verify that the provider is configured correctly by making a simple test call.

        Raises:
            RuntimeError: If the connection test fails.
        """
        try:
            logger.info(f"Verifying connection: {self.provider}/{self.model}")
            await self.call(
                messages=[{"role": "user", "content": "Say 'ok'"}],
                max_completion_tokens=self._verification_max_completion_tokens(),
                max_retries=2,
                initial_backoff=0.5,
                max_backoff=2.0,
                scope="verification",
            )
            logger.info(f"Connection verified: {self.provider}/{self.model}")
        except Exception as e:
            raise RuntimeError(f"Connection verification failed for {self.provider}/{self.model}: {e}") from e

    def _supports_reasoning_model(self) -> bool:
        """Check if the current model is a reasoning model (o1, o3, GPT-5, DeepSeek)."""
        model_lower = self.model.lower()
        if "deepseek" in model_lower:
            # DeepSeek v4-flash is the non-thinking route. Treating every
            # DeepSeek model as a reasoning model injects reasoning_effort,
            # which conflicts with thinking-disabled flash calls.
            return any(x in model_lower for x in ["v4-pro", "reasoner", "r1", "thinking"])
        return any(x in model_lower for x in ["gpt-5", "o1", "o3"])

    def _get_max_reasoning_tokens(self) -> int | None:
        """Get max reasoning tokens for reasoning models."""
        model_lower = self.model.lower()

        # GPT-4 and GPT-4.1 models have different caps
        if any(x in model_lower for x in ["gpt-4.1", "gpt-4-"]):
            return 32000
        elif "gpt-4o" in model_lower:
            return 16384

        return None

    def _max_tokens_param_name(self) -> str:
        """Return the correct parameter name for limiting response tokens.

        Native OpenAI, Azure OpenAI, Groq, and llamacpp accept 'max_completion_tokens'.
        Mistral and other OpenAI-compatible endpoints that haven't adopted the newer
        parameter name require 'max_tokens', so when the openai provider is configured
        with a non-Azure custom base_url we fall back to the widely-supported
        'max_tokens'.

        Reasoning models (GPT-5, o1, o3) only accept 'max_completion_tokens' and reject
        'max_tokens' outright, so they always use the new parameter name regardless of
        base_url.
        """
        # Reasoning models (GPT-5, o1, o3, ...) only accept max_completion_tokens.
        # Azure OpenAI + GPT-5 is the canonical example: issue #978.
        if self._supports_reasoning_model():
            return "max_completion_tokens"
        # Native OpenAI (no custom base URL), Groq, and llamacpp use max_completion_tokens
        if self.provider in ("groq", "llamacpp"):
            return "max_completion_tokens"
        if self.provider == "openai" and not self.base_url:
            return "max_completion_tokens"
        # Azure OpenAI is fully OpenAI-API-compatible — detect it by hostname so users
        # can keep provider=openai + an Azure base_url (the documented setup).
        if self.provider == "openai" and self.base_url and ".openai.azure.com" in self.base_url:
            return "max_completion_tokens"
        # openai with custom base_url, ollama, lmstudio, minimax, volcano —
        # use the widely-supported max_tokens
        return "max_tokens"

    def _apply_provider_extra_body_defaults(self, extra_body: dict[str, Any]) -> None:
        """Apply provider-specific extra_body defaults while preserving user overrides."""
        if self.provider == "minimax":
            extra_body.setdefault("thinking", {"type": "disabled"})

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
            max_completion_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0-2.0).
            scope: Scope identifier for tracking.
            max_retries: Maximum retry attempts.
            initial_backoff: Initial backoff time in seconds.
            max_backoff: Maximum backoff time in seconds.
            skip_validation: Return raw JSON without Pydantic validation.
            strict_schema: Use strict json_schema (grammar-enforced) response_format instead of
                the soft json_object path. Supported by OpenAI and schema-capable self-hosted
                backends (llama.cpp, vLLM). Server-wide via HINDSIGHT_API_LLM_STRICT_SCHEMA.
            return_usage: If True, return tuple (result, TokenUsage) instead of just result.

        Returns:
            If return_usage=False: Parsed response if response_format is provided, otherwise text content.
            If return_usage=True: Tuple of (result, TokenUsage) with token counts.

        Raises:
            OutputTooLongError: If output exceeds token limits.
            Exception: Re-raises API errors after retries exhausted.
        """
        # Handle Ollama with native API for structured output (better schema enforcement)
        if self.provider == "ollama" and response_format is not None:
            return await self._call_ollama_native(
                messages=messages,
                response_format=response_format,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                skip_validation=skip_validation,
                scope=scope,
                return_usage=return_usage,
            )

        start_time = time.time()

        # Build call parameters
        call_params: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
        }

        # Check if model supports reasoning parameter
        is_reasoning_model = self._supports_reasoning_model()

        # Apply model-specific token limits
        if max_completion_tokens is not None:
            max_tokens_cap = self._get_max_reasoning_tokens()
            if max_tokens_cap and max_completion_tokens > max_tokens_cap:
                max_completion_tokens = max_tokens_cap
            # For reasoning models, enforce minimum to ensure space for reasoning + output
            if is_reasoning_model and max_completion_tokens < 16000:
                max_completion_tokens = 16000
            call_params[self._max_tokens_param_name()] = max_completion_tokens
        if temperature is not None and not is_reasoning_model:
            # MiniMax requires temperature in (0.0, 1.0] — clamp accordingly
            if self.provider == "minimax":
                temperature = max(0.01, min(temperature, 1.0))
            call_params["temperature"] = temperature

        # Set reasoning_effort for reasoning models
        if is_reasoning_model:
            call_params["reasoning_effort"] = self.reasoning_effort

        # Provider-specific parameters
        extra_body: dict[str, Any] = {**self._config_extra_body}
        self._apply_provider_extra_body_defaults(extra_body)
        if self.provider == "groq":
            call_params["seed"] = DEFAULT_LLM_SEED
            # Add service_tier if configured
            if self.groq_service_tier:
                extra_body["service_tier"] = self.groq_service_tier
            # Add reasoning parameters for reasoning models
            if is_reasoning_model:
                extra_body["include_reasoning"] = False
        if extra_body:
            call_params["extra_body"] = extra_body

        # Prepare response format ONCE before retry loop
        if response_format is not None:
            schema = None
            if hasattr(response_format, "model_json_schema"):
                schema = strict_json_schema(response_format) if strict_schema else response_format.model_json_schema()

            if strict_schema and schema is not None:
                # Use OpenAI's strict JSON schema enforcement
                call_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "strict": True,
                        "schema": schema,
                    },
                }
            else:
                # Soft enforcement: add schema to prompt and use json_object mode
                if schema is not None:
                    schema_msg = f"\n\nYou must respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2, ensure_ascii=False)}"

                    if call_params["messages"] and call_params["messages"][0].get("role") == "system":
                        first_msg = call_params["messages"][0]
                        if isinstance(first_msg, dict) and isinstance(first_msg.get("content"), str):
                            first_msg["content"] += schema_msg
                    elif call_params["messages"]:
                        first_msg = call_params["messages"][0]
                        if isinstance(first_msg, dict) and isinstance(first_msg.get("content"), str):
                            first_msg["content"] = schema_msg + "\n\n" + first_msg["content"]
                # Providers that skip json_object grammar enforcement
                skip_grammar = self.provider in ("lmstudio", "ollama", "volcano")
                if self.provider == "llamacpp":
                    from hindsight_api.config import get_config

                    skip_grammar = get_config().llamacpp_no_grammar
                if not skip_grammar:
                    call_params["messages"] = _ensure_json_word_in_user_message(call_params["messages"])
                    call_params["response_format"] = {"type": "json_object"}

        apply_bank_attribution(call_params)

        last_exception = None

        for attempt in range(max_retries + 1):
            # Surface attempt count in worker stage so JSON-schema retry loops
            # are visible from logs (small models on strict structured output
            # often loop here). Cheap no-op outside worker context.
            if attempt > 0:
                set_stage(f"llm.{self.provider}.{scope}.attempt={attempt + 1}/{max_retries + 1}")
            try:
                if response_format is not None:
                    response = await self._client.chat.completions.create(**call_params)
                    # Stash usage before parse/validate, which may raise locally
                    # even though the provider charged for these tokens (#2387).
                    stash_response_usage(_usage_from_openai_response(response))

                    content, first_choice = _content_or_error(
                        response,
                        provider=self.provider,
                        model=self.model,
                        scope=scope,
                    )

                    # Strip reasoning model thinking tags (closed and unclosed).
                    # Supports: <think>, <thinking>, <thought>, <reasoning>, |startthink|/|endthink|
                    original_len = len(content)
                    content = _strip_reasoning_tags(content)
                    if len(content) < original_len:
                        logger.debug(f"Stripped {original_len - len(content)} chars of reasoning tokens")

                    # Strip markdown code fences if present — any provider may
                    # produce these (confirmed with MiniMax, some Ollama models,
                    # Claude via proxies). No-op when content is already bare JSON.
                    clean_content = _strip_code_fences(content)
                    try:
                        json_data = json.loads(clean_content)
                    except json.JSONDecodeError:
                        # Fallback to parsing raw content in case stripping was wrong
                        try:
                            json_data = json.loads(content)
                        except json.JSONDecodeError as json_err:
                            # Truncate content for logging
                            content_preview = content[:500] if content else "<empty>"
                            if content and len(content) > 700:
                                content_preview = f"{content[:500]}...TRUNCATED...{content[-200:]}"
                            logger.warning(
                                f"JSON parse error from LLM response (attempt {attempt + 1}/{max_retries + 1}): {json_err}\n"
                                f"  Model: {self.provider}/{self.model}\n"
                                f"  Content length: {len(content) if content else 0} chars\n"
                                f"  Content preview: {content_preview!r}\n"
                                f"  Finish reason: {_finish_reason_for_choice(first_choice)}"
                            )
                            # Retry on JSON parse errors
                            if attempt < max_retries:
                                backoff = min(initial_backoff * (2**attempt), max_backoff)
                                await asyncio.sleep(backoff)
                                last_exception = json_err
                                continue
                            else:
                                logger.error(f"JSON parse error after {max_retries + 1} attempts, giving up")
                                raise

                    if skip_validation:
                        result = json_data
                    else:
                        result = response_format.model_validate(json_data)
                else:
                    response = await self._client.chat.completions.create(**call_params)
                    stash_response_usage(_usage_from_openai_response(response))
                    result, first_choice = _content_or_error(
                        response,
                        provider=self.provider,
                        model=self.model,
                        scope=scope,
                    )

                    # Free-form (non-structured) output also leaks reasoning tags:
                    # reasoning models like MiniMax-M3 wrap their chain-of-thought
                    # in <think>...</think> in the response body. Without this strip
                    # a mental-model markdown blob is stored verbatim with the raw
                    # thinking tags. Mirrors the structured-output path above.
                    result = _strip_reasoning_tags(result)

                # Record token usage metrics
                duration = time.time() - start_time
                usage = response.usage
                response_usage = _usage_from_openai_response(response)
                input_tokens = response_usage.input_tokens
                output_tokens = response_usage.output_tokens
                total_tokens = usage.total_tokens or 0 if usage else 0
                cached_tokens = response_usage.cached_tokens
                thoughts_tokens = 0
                if usage and getattr(usage, "completion_tokens_details", None):
                    thoughts_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
                # OpenAI-compatible providers fold reasoning tokens into
                # ``completion_tokens`` (and thus ``total_tokens``), but the
                # TokenUsage contract — and the Gemini provider — treat
                # ``output_tokens``/``total_tokens`` as visible-only, surfacing
                # reasoning separately in ``thoughts_tokens``. Subtract so the
                # two fields don't double-count reasoning (cost over-attribution).
                if thoughts_tokens:
                    output_tokens = max(0, output_tokens - thoughts_tokens)
                    total_tokens = max(0, total_tokens - thoughts_tokens)

                # Record LLM metrics. ``output_tokens`` is visible-only by now, so
                # ``thoughts_tokens`` has to be recorded alongside it or the reasoning
                # half of the billed output reaches no counter at all.
                metrics = get_metrics_collector()
                metrics.record_llm_call(
                    provider=self.provider,
                    model=self.model,
                    scope=scope,
                    duration=duration,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=True,
                    cached_input_tokens=cached_tokens,
                    thoughts_tokens=thoughts_tokens,
                )

                # Record trace span
                from hindsight_api.tracing import _serialize_for_span, get_span_recorder

                finish_reason = _finish_reason_for_choice(first_choice)
                span_recorder = get_span_recorder()
                span_recorder.record_llm_call(
                    provider=self.provider,
                    model=self.model,
                    scope=scope,
                    messages=call_params["messages"],
                    response_content=_serialize_for_span(result),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    duration=duration,
                    finish_reason=finish_reason,
                    error=None,
                    cached_tokens=cached_tokens,
                )

                # Log slow calls
                if duration > 10.0 and usage:
                    ratio = max(1, output_tokens) / max(1, input_tokens)
                    cache_info = f", cached_tokens={cached_tokens}" if cached_tokens > 0 else ""
                    logger.info(
                        f"slow llm call: scope={scope}, model={self.provider}/{self.model}, "
                        f"input_tokens={input_tokens}, output_tokens={output_tokens}, "
                        f"total_tokens={total_tokens}{cache_info}, time={duration:.3f}s, ratio out/in={ratio:.2f}"
                    )

                if return_usage:
                    token_usage = TokenUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        cached_tokens=cached_tokens,
                        thoughts_tokens=thoughts_tokens,
                    )
                    return result, token_usage
                return result

            except LengthFinishReasonError as e:
                logger.warning(f"LLM output exceeded token limits: {str(e)}")
                raise OutputTooLongError(
                    "LLM output exceeded token limits. Input may need to be split into smaller chunks."
                ) from e

            except APIConnectionError as e:
                last_exception = e
                status_code = getattr(e, "status_code", None) or getattr(
                    getattr(e, "response", None), "status_code", None
                )
                logger.warning(f"APIConnectionError (HTTP {status_code}), attempt {attempt + 1}: {str(e)[:200]}")
                if attempt < max_retries:
                    backoff = min(initial_backoff * (2**attempt), max_backoff)
                    await asyncio.sleep(backoff)
                    continue
                else:
                    logger.error(f"Connection error after {max_retries + 1} attempts: {str(e)}")
                    raise

            except APIStatusError as e:
                # Fast fail only on 401 (unauthorized) and 403 (forbidden)
                if e.status_code in (401, 403):
                    logger.error(f"Auth error (HTTP {e.status_code}), not retrying: {str(e)}")
                    raise

                # Diagnostic dump (opt-in) of the exact request behind any 4xx.
                dump_request_on_4xx(scope=scope, provider=self.provider, model=self.model, err=e, request=call_params)

                _raise_provider_quota_defer(
                    e, provider=self.provider, model=self.model, scope=scope, max_backoff=max_backoff
                )

                # Handle tool_use_failed error - model outputted in tool call format
                if e.status_code == 400 and response_format is not None:
                    try:
                        error_body = e.body if hasattr(e, "body") else {}
                        if isinstance(error_body, dict):
                            error_info: dict[str, Any] = error_body.get("error") or {}
                            if error_info.get("code") == "tool_use_failed":
                                failed_gen = error_info.get("failed_generation", "")
                                if failed_gen:
                                    # Parse tool call format and convert to expected format
                                    tool_call = json.loads(failed_gen)
                                    tool_name = tool_call.get("name", "")
                                    tool_args = tool_call.get("arguments", {})
                                    converted = {"actions": [{"tool": tool_name, **tool_args}]}
                                    if skip_validation:
                                        result = converted
                                    else:
                                        result = response_format.model_validate(converted)

                                    # Record metrics
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
                                    if return_usage:
                                        return result, TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)
                                    return result
                    except (json.JSONDecodeError, KeyError, TypeError):
                        pass  # Failed to parse tool_use_failed, continue with normal retry

                last_exception = e
                if attempt < max_retries:
                    logger.warning(
                        f"APIStatusError ({self.provider}/{self.model}, scope={scope}, "
                        f"attempt {attempt + 1}/{max_retries + 1}): {_summarize_status_error(e)}"
                    )
                    backoff = min(initial_backoff * (2**attempt), max_backoff)
                    jitter = backoff * 0.2 * (2 * (time.time() % 1) - 1)
                    sleep_time = backoff + jitter
                    await asyncio.sleep(sleep_time)
                else:
                    logger.error(
                        f"API error after {max_retries + 1} attempts ({self.provider}/{self.model}, "
                        f"scope={scope}): {_summarize_status_error(e)}"
                    )
                    raise
            except ProviderResponseError as e:
                last_exception = e
                if e.retryable and attempt < max_retries:
                    logger.warning(
                        f"Provider response error ({self.provider}/{self.model}, scope={scope}, "
                        f"attempt {attempt + 1}/{max_retries + 1}): {e}"
                    )
                    backoff = min(initial_backoff * (2**attempt), max_backoff)
                    await asyncio.sleep(backoff)
                    continue
                logger.error(
                    f"Provider response error after {attempt + 1} attempts "
                    f"({self.provider}/{self.model}, scope={scope}): {e}"
                )
                raise

            except Exception:
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("LLM call failed after all retries with no exception captured")

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
        tool_choice: LLMToolChoice = LLM_TOOL_CHOICE_AUTO,
    ) -> LLMToolCallResult:
        """
        Make an LLM API call with tool/function calling support.

        Args:
            messages: List of message dicts. Can include tool results with role='tool'.
            tools: List of tool definitions in OpenAI format.
            max_completion_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0-2.0).
            scope: Scope identifier for tracking.
            max_retries: Maximum retry attempts.
            initial_backoff: Initial backoff time in seconds.
            max_backoff: Maximum backoff time in seconds.
            tool_choice: Canonical tool-selection policy.

        Returns:
            LLMToolCallResult with content and/or tool_calls.
        """
        start_time = time.time()

        request_tool_choice: str | None
        if tool_choice.mode is LLMToolChoiceMode.NAMED:
            forced_name = tool_choice.selected_function_name
            filtered = [tool for tool in tools if tool.get("function", {}).get("name") == forced_name]
            if len(filtered) != 1:
                raise ValueError(
                    f"Named tool_choice must reference exactly one declared tool; "
                    f"found {len(filtered)} definitions for {forced_name!r}"
                )
            tools = filtered
            request_tool_choice = LLMToolChoiceMode.REQUIRED.value
        elif tool_choice.mode is LLMToolChoiceMode.AUTO:
            request_tool_choice = None
        else:
            request_tool_choice = tool_choice.mode.value

        # DeepSeek accepts tool calls but rejects explicit required/named
        # tool_choice values. The tools list has already been narrowed for
        # forced calls, so omitting tool_choice preserves the practical behavior.
        if "deepseek" in self.model.lower() and tool_choice.mode is not LLMToolChoiceMode.AUTO:
            request_tool_choice = None

        # LM Studio and Ollama silently drop tool_choice="required", returning an
        # empty tool_calls array instead of forcing a call (#1563/#1179).
        # Downgrade to auto (None) so the model still gets to call a tool. Named
        # tool_choice dicts were already normalized to "required" + a single
        # filtered tool above, so the call stays practically forced even under
        # auto. Generic OpenAI-compatible endpoints retain the canonical
        # ``required`` contract regardless of whether they use a custom base URL.
        if request_tool_choice == LLMToolChoiceMode.REQUIRED.value and self._drops_tool_choice_required():
            request_tool_choice = None

        # DeepSeek tool-call replies can carry provider-specific reasoning_content.
        # The normalized tool result does not retain it, but replaying assistant
        # tool_calls without the field can trigger a 400. DeepSeek accepts an
        # empty-string fallback, matching the provider's history-replay contract.
        if "deepseek" in self.model.lower():
            normalized_messages: list[dict[str, Any]] = []
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("tool_calls") and "reasoning_content" not in msg:
                    normalized_msg = dict(msg)
                    normalized_msg["reasoning_content"] = ""
                    normalized_messages.append(normalized_msg)
                else:
                    normalized_messages.append(msg)
            messages = normalized_messages

        # Build call parameters
        call_params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
        }
        if request_tool_choice is not None:
            call_params["tool_choice"] = request_tool_choice

        if max_completion_tokens is not None:
            call_params[self._max_tokens_param_name()] = max_completion_tokens
        if temperature is not None:
            # MiniMax requires temperature in (0.0, 1.0] — clamp accordingly
            if self.provider == "minimax":
                temperature = max(0.01, min(temperature, 1.0))
            call_params["temperature"] = temperature

        # Provider-specific parameters
        extra_body: dict[str, Any] = {**self._config_extra_body}
        self._apply_provider_extra_body_defaults(extra_body)
        if self.provider == "groq":
            call_params["seed"] = DEFAULT_LLM_SEED
        if extra_body:
            call_params["extra_body"] = extra_body

        apply_bank_attribution(call_params)

        last_exception = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                set_stage(f"llm.{self.provider}.tools.attempt={attempt + 1}/{max_retries + 1}")
            try:
                response = await self._client.chat.completions.create(**call_params)

                message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason

                # Extract tool calls if present
                tool_calls: list[LLMToolCall] = []
                if message.tool_calls:
                    for tc in message.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                        except json.JSONDecodeError:
                            args = {"_raw": tc.function.arguments}
                        tool_calls.append(LLMToolCall(id=tc.id, name=tc.function.name, arguments=args))

                content = message.content

                # Record metrics
                duration = time.time() - start_time
                usage = response.usage
                input_tokens = usage.prompt_tokens or 0 if usage else 0
                output_tokens = usage.completion_tokens or 0 if usage else 0
                cached_tokens = 0
                if usage and getattr(usage, "prompt_tokens_details", None):
                    cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
                thoughts_tokens = 0
                if usage and getattr(usage, "completion_tokens_details", None):
                    thoughts_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
                # See ``call()``: OpenAI-compatible ``completion_tokens`` includes
                # reasoning, so make ``output_tokens`` visible-only to avoid
                # double-counting it against ``thoughts_tokens``.
                if thoughts_tokens:
                    output_tokens = max(0, output_tokens - thoughts_tokens)

                # See ``call()``: record the reasoning and cached counts too, so no
                # billed token is dropped from the metrics counters.
                metrics = get_metrics_collector()
                metrics.record_llm_call(
                    provider=self.provider,
                    model=self.model,
                    scope=scope,
                    duration=duration,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=True,
                    cached_input_tokens=cached_tokens,
                    thoughts_tokens=thoughts_tokens,
                )

                # Record OpenTelemetry span
                from hindsight_api.tracing import get_span_recorder

                span_recorder = get_span_recorder()
                # Convert LLMToolCall objects to dicts for span recording
                tool_calls_dict = (
                    [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls]
                    if tool_calls
                    else None
                )
                span_recorder.record_llm_call(
                    provider=self.provider,
                    model=self.model,
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
                    finish_reason=finish_reason,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                    thoughts_tokens=thoughts_tokens,
                )

            except APIConnectionError as e:
                last_exception = e
                status_code = getattr(e, "status_code", None) or getattr(
                    getattr(e, "response", None), "status_code", None
                )
                if attempt < max_retries:
                    logger.warning(
                        f"APIConnectionError in tool call ({self.provider}/{self.model}, scope={scope}, "
                        f"attempt {attempt + 1}/{max_retries + 1}, HTTP {status_code}): {str(e)[:200]}"
                    )
                    await asyncio.sleep(min(initial_backoff * (2**attempt), max_backoff))
                    continue
                logger.error(
                    f"Connection error in tool call after {max_retries + 1} attempts "
                    f"({self.provider}/{self.model}, scope={scope}): {str(e)}"
                )
                raise

            except APIStatusError as e:
                if e.status_code in (401, 403):
                    logger.error(
                        f"Auth error in tool call (HTTP {e.status_code}, {self.provider}/{self.model}), "
                        f"not retrying: {_summarize_status_error(e)}"
                    )
                    raise

                # Diagnostic dump (opt-in) of the exact request behind any 4xx.
                dump_request_on_4xx(scope=scope, provider=self.provider, model=self.model, err=e, request=call_params)

                _raise_provider_quota_defer(
                    e, provider=self.provider, model=self.model, scope=scope, max_backoff=max_backoff
                )

                last_exception = e
                if attempt < max_retries:
                    logger.warning(
                        f"APIStatusError in tool call ({self.provider}/{self.model}, scope={scope}, "
                        f"attempt {attempt + 1}/{max_retries + 1}): {_summarize_status_error(e)}"
                    )
                    await asyncio.sleep(min(initial_backoff * (2**attempt), max_backoff))
                    continue
                logger.error(
                    f"API error in tool call after {max_retries + 1} attempts "
                    f"({self.provider}/{self.model}, scope={scope}): {_summarize_status_error(e)}"
                )
                raise
            except Exception:
                raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Tool call failed after all retries")

    async def _call_ollama_native(
        self,
        messages: list[dict[str, str]],
        response_format: Any,
        max_completion_tokens: int | None,
        temperature: float | None,
        max_retries: int,
        initial_backoff: float,
        max_backoff: float,
        skip_validation: bool,
        scope: str = "memory",
        return_usage: bool = False,
    ) -> Any:
        """
        Call Ollama using native API with JSON schema enforcement.

        Ollama's native API supports passing a full JSON schema in the 'format' parameter,
        which provides better structured output control than the OpenAI-compatible API.
        """
        start_time = time.time()

        # Get the JSON schema from the Pydantic model
        schema = response_format.model_json_schema() if hasattr(response_format, "model_json_schema") else None

        # Build the base URL for Ollama's native API
        # Default OpenAI-compatible URL is http://localhost:11434/v1
        # Native API is at http://localhost:11434/api/chat
        base_url = self.base_url or "http://localhost:11434/v1"
        if base_url.endswith("/v1"):
            native_url = base_url[:-3] + "/api/chat"
        else:
            native_url = base_url.rstrip("/") + "/api/chat"

        # Build request payload
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,  # Disable thinking for reasoning models (qwen3.5, etc.)
        }

        # Add schema as format parameter for structured output
        if schema:
            payload["format"] = schema

        # Add optional parameters with optimized defaults for Ollama
        options: dict[str, Any] = {
            "num_batch": 512,  # Optimal batch size for prompt processing
        }
        if self.ollama_num_ctx is not None:
            options["num_ctx"] = self.ollama_num_ctx
        if max_completion_tokens:
            options["num_predict"] = max_completion_tokens
        if temperature is not None:
            options["temperature"] = temperature
        payload["options"] = options

        last_exception = None

        # Pass API key as Bearer token for cloud Ollama endpoints
        headers: dict[str, str] = {}
        if self.api_key and self.api_key != "local":
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=300.0) as client:
            for attempt in range(max_retries + 1):
                if attempt > 0:
                    set_stage(f"llm.ollama_native.{scope}.attempt={attempt + 1}/{max_retries + 1}")
                try:
                    response = await client.post(native_url, json=payload, headers=headers)
                    response.raise_for_status()

                    result = response.json()
                    content = result.get("message", {}).get("content", "")

                    # Strip markdown code fences if present (safety net —
                    # Ollama with schema enforcement usually returns bare JSON,
                    # but some models may still wrap in fences)
                    clean_content = _strip_code_fences(content)
                    try:
                        json_data = json.loads(clean_content)
                    except json.JSONDecodeError:
                        # Fallback to raw content
                        try:
                            json_data = json.loads(content)
                        except json.JSONDecodeError as json_err:
                            content_preview = content[:500] if content else "<empty>"
                            if content and len(content) > 700:
                                content_preview = f"{content[:500]}...TRUNCATED...{content[-200:]}"
                            logger.warning(
                                f"Ollama JSON parse error (attempt {attempt + 1}/{max_retries + 1}): {json_err}\n"
                                f"  Model: ollama/{self.model}\n"
                                f"  Content length: {len(content) if content else 0} chars\n"
                                f"  Content preview: {content_preview!r}"
                            )
                            if attempt < max_retries:
                                backoff = min(initial_backoff * (2**attempt), max_backoff)
                                await asyncio.sleep(backoff)
                                last_exception = json_err
                                continue
                            else:
                                raise

                    # Extract token usage from Ollama response
                    duration = time.time() - start_time
                    input_tokens = result.get("prompt_eval_count", 0) or 0
                    output_tokens = result.get("eval_count", 0) or 0
                    total_tokens = input_tokens + output_tokens

                    # Record LLM metrics
                    metrics = get_metrics_collector()
                    metrics.record_llm_call(
                        provider=self.provider,
                        model=self.model,
                        scope=scope,
                        duration=duration,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        success=True,
                    )

                    # Validate against Pydantic model or return raw JSON
                    if skip_validation:
                        validated_result = json_data
                    else:
                        validated_result = response_format.model_validate(json_data)

                    if return_usage:
                        token_usage = TokenUsage(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            total_tokens=total_tokens,
                        )
                        return validated_result, token_usage
                    return validated_result

                except httpx.HTTPStatusError as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Ollama HTTP error (attempt {attempt + 1}/{max_retries + 1}): {e.response.status_code}"
                        )
                        backoff = min(initial_backoff * (2**attempt), max_backoff)
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        logger.error(f"Ollama HTTP error after {max_retries + 1} attempts: {e}")
                        raise

                except httpx.RequestError as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Ollama connection error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        backoff = min(initial_backoff * (2**attempt), max_backoff)
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        logger.error(f"Ollama connection error after {max_retries + 1} attempts: {e}")
                        raise

                except Exception as e:
                    logger.error(f"Unexpected error during Ollama call: {type(e).__name__}: {e}")
                    raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Ollama call failed after all retries")

    async def supports_batch_api(self) -> bool:
        """Check if this provider supports batch API operations."""
        # Only OpenAI and Groq support batch API
        return self.provider in ("openai", "groq")

    async def submit_batch(
        self,
        requests: list[dict[str, Any]],
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
    ) -> dict[str, Any]:
        """
        Submit a batch of requests to OpenAI/Groq Batch API.

        Args:
            requests: List of request dicts with custom_id, method, url, body
            endpoint: API endpoint (e.g., "/v1/chat/completions")
            completion_window: Completion window (e.g., "24h")

        Returns:
            Dict with batch metadata including batch_id

        Raises:
            NotImplementedError: If provider doesn't support batch API
        """
        if not await self.supports_batch_api():
            raise NotImplementedError(f"Batch API not supported for provider: {self.provider}")

        logger.info(f"Submitting batch with {len(requests)} requests to {self.provider}")

        # Format requests as JSONL
        jsonl_content = "\n".join(json.dumps(req, ensure_ascii=False) for req in requests)

        # Upload file to provider (wrap in BytesIO with filename)
        file_bytes = io.BytesIO(jsonl_content.encode("utf-8"))
        file_bytes.name = "batch_input.jsonl"  # OpenAI SDK needs a filename

        file_response = await self._client.files.create(
            file=file_bytes,
            purpose="batch",
        )

        logger.debug(f"Uploaded batch file: {file_response.id}")

        # Create batch
        batch_response = await self._client.batches.create(
            input_file_id=file_response.id,
            endpoint=endpoint,
            completion_window=completion_window,
        )

        logger.info(f"Batch submitted: {batch_response.id}, status={batch_response.status}")

        return {
            "batch_id": batch_response.id,
            "status": batch_response.status,
            "input_file_id": file_response.id,
            "created_at": batch_response.created_at,
            "request_count": len(requests),
        }

    async def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        """
        Get the status of a batch job.

        Args:
            batch_id: Batch identifier

        Returns:
            Dict with status info (batch_id, status, completed_at, etc.)
        """
        if not await self.supports_batch_api():
            raise NotImplementedError(f"Batch API not supported for provider: {self.provider}")

        batch = await self._client.batches.retrieve(batch_id)

        result = {
            "batch_id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "request_counts": {
                "total": batch.request_counts.total if batch.request_counts else 0,
                "completed": batch.request_counts.completed if batch.request_counts else 0,
                "failed": batch.request_counts.failed if batch.request_counts else 0,
            },
        }

        if batch.completed_at:
            result["completed_at"] = batch.completed_at
        if batch.output_file_id:
            result["output_file_id"] = batch.output_file_id
        if batch.error_file_id:
            result["error_file_id"] = batch.error_file_id
        if batch.errors:
            result["errors"] = batch.errors

        return result

    async def retrieve_batch_results(self, batch_id: str) -> list[dict[str, Any]]:
        """
        Retrieve completed batch results.

        Args:
            batch_id: Batch identifier

        Returns:
            List of result dicts (one per request, matched by custom_id)
        """
        if not await self.supports_batch_api():
            raise NotImplementedError(f"Batch API not supported for provider: {self.provider}")

        # Get batch status
        batch = await self._client.batches.retrieve(batch_id)

        if batch.status != "completed":
            raise ValueError(f"Batch {batch_id} is not completed yet (status: {batch.status})")

        if not batch.output_file_id:
            raise ValueError(f"Batch {batch_id} has no output file")

        # Download results file
        logger.debug(f"Downloading results for batch {batch_id} from file {batch.output_file_id}")
        file_content = await self._client.files.content(batch.output_file_id)

        # Parse JSONL results
        results = []
        for line in file_content.text.strip().split("\n"):
            if line:
                results.append(json.loads(line))

        logger.info(f"Retrieved {len(results)} results for batch {batch_id}")

        return results

    async def cleanup(self) -> None:
        """Clean up resources (close OpenAI client connections)."""
        if hasattr(self, "_client") and self._client:
            await self._client.close()
