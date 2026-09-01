"""
LLM wrapper for unified configuration across providers.
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, Any

from json_repair import repair_json
from pydantic import BaseModel

# Vertex AI imports (conditional - for LLMProvider to pass credentials to GeminiLLM)
try:
    from google.oauth2 import service_account

    VERTEXAI_AVAILABLE = True
except ImportError:
    VERTEXAI_AVAILABLE = False

from ..config import (
    DEFAULT_LLM_MAX_CONCURRENT,
    ENV_CONSOLIDATION_LLM_MAX_CONCURRENT,
    ENV_LLM_MAX_CONCURRENT,
    ENV_REFLECT_LLM_MAX_CONCURRENT,
    ENV_RETAIN_LLM_MAX_CONCURRENT,
)
from .cache_affinity import parse_cache_affinity
from .llm_interface import (
    LLM_TOOL_CHOICE_AUTO,
    LLMInterface,
    LLMToolChoice,
    LLMToolChoiceMode,
)
from .llm_interface import (
    OutputTooLongError as OutputTooLongError,
)

if TYPE_CHECKING:
    from .response_models import LLMToolCallResult

logger = logging.getLogger(__name__)

# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Global semaphore to limit concurrent LLM requests across all instances.
# Set HINDSIGHT_API_LLM_MAX_CONCURRENT=1 for local LLMs (LM Studio, Ollama).
_llm_max_concurrent = int(os.getenv(ENV_LLM_MAX_CONCURRENT, str(DEFAULT_LLM_MAX_CONCURRENT)))
_global_llm_semaphore = asyncio.Semaphore(_llm_max_concurrent)


def _build_per_op_semaphores() -> dict[str, asyncio.Semaphore]:
    """Build the per-operation semaphore registry from env vars.

    Each per-op cap is composed with — not a substitute for — the global cap:
    a call that matches a configured operation must acquire both its per-op
    semaphore and the global semaphore. This lets operators reserve headroom
    in the global pool by capping individual operations (e.g. cap retain at 2
    of 4 global slots so the live chat path always has 2 slots available).

    Operations without a configured env var are absent from the registry and
    therefore only constrained by the global cap.
    """
    semaphores: dict[str, asyncio.Semaphore] = {}
    for op, env_var in (
        ("retain", ENV_RETAIN_LLM_MAX_CONCURRENT),
        ("reflect", ENV_REFLECT_LLM_MAX_CONCURRENT),
        ("consolidation", ENV_CONSOLIDATION_LLM_MAX_CONCURRENT),
    ):
        raw = os.getenv(env_var)
        if raw is None or raw == "":
            continue
        value = int(raw)
        if value <= 0:
            raise ValueError(f"{env_var} must be a positive integer, got {raw!r}")
        semaphores[op] = asyncio.Semaphore(value)
    return semaphores


_per_op_llm_semaphores: dict[str, asyncio.Semaphore] = _build_per_op_semaphores()


def _scope_to_operation(scope: str) -> str | None:
    """Map a call scope to its per-operation concurrency bucket.

    Returns None for scopes that don't belong to a tracked operation
    (verification probes, bank_mission, memory_think, mental_model_delta_ops),
    which then run under the global cap only.
    """
    if scope.startswith("retain"):
        return "retain"
    if scope.startswith("reflect"):
        return "reflect"
    if scope.startswith("consolidation"):
        return "consolidation"
    return None


def _semaphores_for_scope(scope: str) -> list[asyncio.Semaphore]:
    """Return the semaphores a call with the given scope must acquire.

    Always includes the global semaphore; includes the per-op semaphore when
    one is configured for the scope's operation bucket.
    """
    op = _scope_to_operation(scope)
    per_op = _per_op_llm_semaphores.get(op) if op is not None else None
    if per_op is None:
        return [_global_llm_semaphore]
    # Per-op acquired first so contention queues on the narrower cap before
    # holding a global slot.
    return [per_op, _global_llm_semaphore]


@asynccontextmanager
async def _attempt_permits(scope: str):
    """Hold configured LLM concurrency permits for one upstream attempt."""
    from ..worker.stage import get_stage, set_stage

    async with AsyncExitStack() as stack:
        for sem in _semaphores_for_scope(scope):
            await stack.enter_async_context(sem)
        try:
            yield
        except BaseException:
            # A failed attempt exits here with its permits released while the
            # provider classifies the error and sleeps out its backoff. Suffix
            # the stage so `attempt=N` always means "permits held, request in
            # flight" (#3002); the next attempt re-stamps after re-acquiring.
            stage = get_stage()
            if stage is not None and not stage.endswith(".backoff"):
                set_stage(f"{stage}.backoff")
            raise


def _request_params(
    *,
    max_completion_tokens: int | None = None,
    temperature: float | None = None,
    scope: str | None = None,
    response_format: Any | None = None,
    tool_choice: LLMToolChoice | None = None,
) -> dict[str, Any] | None:
    """Build the requested-params bag for tracing — only values the caller set.

    Omitting unset values avoids the misleading nulls we used to record (e.g.
    consolidation, which passes no token cap), while surfacing the real cap for
    callers that do set one (e.g. retain's ``retain_max_completion_tokens``).
    """
    params: dict[str, Any] = {}
    if max_completion_tokens is not None:
        params["max_completion_tokens"] = max_completion_tokens
    if temperature is not None:
        params["temperature"] = temperature
    if response_format is not None:
        params["response_schema"] = getattr(response_format, "__name__", None) or "structured"
    if tool_choice is not None and tool_choice.mode is not LLMToolChoiceMode.AUTO:
        params["tool_choice"] = tool_choice.function_name or tool_choice.mode.value
    return params or None


_UNSAFE_TEXT_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff]")


def sanitize_text(text: str | None) -> str | None:
    """
    Sanitize text by removing characters that break downstream systems.

    Removes:
    - ASCII control characters (0x00-0x08, 0x0B-0x0C, 0x0E-0x1F, 0x7F): break
      json.loads and PostgreSQL UTF-8 encoding; tab (0x09), newline (0x0A), and
      carriage return (0x0D) are preserved as they are valid in text and JSON.
    - Unicode surrogates (U+D800-U+DFFF): Invalid in UTF-8, break LLM APIs

    Surrogate characters are used in UTF-16 encoding but cannot be encoded
    in UTF-8. They can appear in Python strings from improperly decoded data
    (e.g., from JavaScript or broken files): a client may serialize a half-emoji
    split at a boundary as a lone ``\\udXXX`` escape. Such input crashes the
    SentenceTransformers/cross-encoder Rust tokenizers and stdout logging, so
    user content is sanitized at the retain/recall/reflect ingress (see issue
    #1875). Control characters commonly appear in LLM output embedded inside
    JSON string values.
    """
    if text is None:
        return None
    if not text:
        return text
    return _UNSAFE_TEXT_RE.sub("", text)


# Back-compat alias: this helper was originally introduced to scrub LLM *output*;
# it now also scrubs user *input* at ingress, hence the broader name.
sanitize_llm_output = sanitize_text


def sanitize_llm_value(value: Any) -> Any:
    """
    Recursively strip UTF-8-hostile characters from every string an LLM produced.

    ``sanitize_text`` guards a single field. This guards a whole response — the
    text a provider returned, the dict a structured call parsed, the pydantic
    model it validated into, the ``(result, usage)`` tuple ``return_usage=True``
    hands back — so that *every* LLM call is covered at one boundary instead of
    each consumer remembering to scrub its own fields (see issue #3729).

    It matters beyond embeddings. A lone surrogate is legal in a Python ``str``
    but cannot be UTF-8 encoded, so it also breaks the cross-encoder's Rust
    tokenizer at rerank time, asyncpg on the way into a ``text`` column, and
    stdout logging. Sanitizing where the text enters the process means the
    downstream stages never have to care which field it landed in.

    Only strings are touched; ints, floats, datetimes and the like pass through
    as-is. Every container returns the *same object* when nothing inside it
    changed, so the overwhelmingly common clean response is not copied and
    object identity (a validated response model, an enum member) survives
    untouched.
    """
    if isinstance(value, str):
        # A str subclass (a StrEnum member, say) comes back as itself unless it
        # actually carries a hostile character — at which point a plain str is the
        # only safe answer, and the alternative was a crash.
        cleaned = _UNSAFE_TEXT_RE.sub("", value)
        return cleaned if cleaned != value else value

    if isinstance(value, BaseModel):
        updates = {}
        for name, field_value in value.__dict__.items():
            cleaned = sanitize_llm_value(field_value)
            if cleaned is not field_value:
                updates[name] = cleaned
        # ``model_copy`` skips validation, which is what we want: the values are
        # already sanitized, and re-validating could reject a model the provider
        # built with ``model_construct``.
        return value.model_copy(update=updates) if updates else value

    if isinstance(value, dict):
        cleaned_dict = {}
        changed = False
        for key, item in value.items():
            # Keys are sanitized too: a surrogate in a key is just as fatal once
            # the dict is rendered to text or bound to a query parameter.
            cleaned_key = sanitize_llm_value(key)
            cleaned_item = sanitize_llm_value(item)
            changed = changed or cleaned_key is not key or cleaned_item is not item
            cleaned_dict[cleaned_key] = cleaned_item
        return cleaned_dict if changed else value

    # ``tuple`` is here for the ``return_usage=True`` shape, ``(result, TokenUsage)``.
    if isinstance(value, (list, tuple)):
        cleaned_items = [sanitize_llm_value(item) for item in value]
        if all(cleaned is original for cleaned, original in zip(cleaned_items, value)):
            return value
        return cleaned_items if isinstance(value, list) else tuple(cleaned_items)

    return value


# ``OutputTooLongError`` is re-exported from ``llm_interface`` (the canonical
# definition the providers raise) so that ``fact_extraction`` and ``multi_llm``,
# which import it from here, catch/inspect the very same class. Do NOT redefine
# it locally: a shadow class silently breaks ``except OutputTooLongError`` on the
# real provider path (see issue #3172).


_JSON_CONTROL_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}


def _escape_control_chars_in_json(text: str) -> str:
    """Make raw control characters inside JSON string values parseable.

    ``json.loads`` rejects an unescaped control character in a string. Models
    hit this whenever they write a multi-line value (a markdown table, a list,
    a code fence) without escaping the line breaks.

    A line break, tab or form feed inside a string is *content*: it is escaped
    so it survives. Blanking it out instead welds a markdown table onto one
    line, and the damage is invisible downstream (#3361). Every other control
    character carries no meaning in text, so it keeps the historical treatment
    of becoming a space.

    The scan tracks string state because only characters inside a string need
    escaping; the same byte between tokens is junk either way.
    """
    out: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if in_string and escape:
            escape = False
            out.append(ch)
            continue
        if in_string and ch == "\\":
            escape = True
            out.append(ch)
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string and ch in _JSON_CONTROL_ESCAPES:
            out.append(_JSON_CONTROL_ESCAPES[ch])
            continue
        if ch <= "\x1f" or ch == "\x7f":
            out.append(" ")
            continue
        out.append(ch)
    return "".join(out)


def parse_llm_json(raw: str) -> Any:
    """
    Robustly parse JSON returned by an LLM.

    Handles common LLM output quirks:
    1. Markdown code fences (```json ... ```) — strip them before parsing.
    2. Embedded control characters (\\x00-\\x1f, \\x7f) — escape the ones inside
       string values (so a raw newline stays a line break), drop the ones
       between tokens, and retry if the initial parse fails.
    3. Structural malformation (trailing commas, unterminated strings, single
       quotes, invalid ``\\escape`` sequences) — repaired as a last resort via
       ``json_repair`` (#2547/#2544).

    The repair pass is purely *structural*: it fixes JSON that ``json.loads``
    cannot parse at all. It deliberately does NOT touch content semantics —
    degenerate-but-valid JSON (repetition loops or leaked scaffolding inside
    string values) parses fine here and is out of scope for this helper.

    Every successful parse is passed through ``sanitize_llm_value``. Decoding is
    where an un-encodable surrogate is *born*: a model that writes ``"\\ud83d"``
    emits six harmless ASCII characters, and only ``json.loads`` turns them into a
    lone surrogate no downstream stage can UTF-8 encode (#3729). Scrubbing the raw
    text beforehand cannot see it; scrubbing the parsed object can.

    Args:
        raw: Raw text returned by the LLM.

    Returns:
        Parsed Python object (dict, list, etc.), with model-authored strings
        scrubbed of surrogates and control characters.

    Raises:
        json.JSONDecodeError: If the text cannot be parsed even after cleanup
            and structural repair (e.g. repair yields an empty result).
    """
    text = raw.strip()

    # Strip markdown code fences (some models wrap JSON in ```json ... ```)
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return sanitize_llm_value(json.loads(text))
    except json.JSONDecodeError:
        # Some models (e.g. Gemini) embed raw control characters inside JSON
        # string values. Escape them rather than blank them out: a raw newline
        # in a string is the model writing a real line break, and replacing it
        # with a space silently welds a markdown table or list onto one line
        # (#3361). Control characters *outside* a string are noise and are
        # dropped, since nothing meaningful can sit between JSON tokens.
        cleaned = _escape_control_chars_in_json(text)

    try:
        return sanitize_llm_value(json.loads(cleaned))
    except json.JSONDecodeError:
        # Last resort: structural repair of malformed JSON. ``repair_json`` never
        # raises — unrecoverable input yields an empty result ("" / {} / []). Keep
        # failing loudly in that case rather than let an empty object masquerade
        # as a successful parse: callers (retry ladders, the #1833 fail-loud path)
        # rely on JSONDecodeError to retry or surface the failure.
        repaired = repair_json(cleaned, return_objects=True)
        if not repaired:
            raise
        return sanitize_llm_value(repaired)


_PROVIDERS_WITHOUT_API_KEY = frozenset(
    {
        "ollama",
        "lmstudio",
        "llamacpp",
        "openai-codex",
        "claude-code",
        "github-copilot",
        "mock",
        "none",
        "vertexai",
        "litellm",
        "litellmrouter",
        "bedrock",
        "nous",
        "xai-oauth",
    }
)


def requires_api_key(provider: str) -> bool:
    """Return True if the given provider requires an API key to operate."""
    return provider.lower() not in _PROVIDERS_WITHOUT_API_KEY


def _validate_ollama_num_ctx(value: Any) -> int | None:
    """Validate a native Ollama context-window override."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"ollama_num_ctx must be a positive integer, got {value!r}")
    if value < 1:
        raise ValueError(f"ollama_num_ctx must be >= 1, got {value}")
    return value


def create_llm_provider(
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    reasoning_effort: str | None,
    groq_service_tier: str | None = None,
    openai_service_tier: str | None = None,
    bedrock_service_tier: str | None = None,
    extra_body: dict[str, Any] | None = None,
    default_headers: dict[str, str] | None = None,
    vertexai_project_id: str | None = None,
    vertexai_region: str | None = None,
    vertexai_credentials: Any = None,
    gemini_safety_settings: list | None = None,
    prompt_cache_enabled: bool = False,
    litellmrouter_config: dict[str, Any] | None = None,
    gemini_service_tier: str | None = None,
    timeout: float | None = None,
    ollama_num_ctx: int | None = None,
    cache_affinity: str | None = None,
    structured_output_forced_tool: bool = False,
    # Appended rather than inserted: some callers still pass the older settings
    # positionally (guarded by the positional-compatibility tests in
    # tests/test_llm_wrapper.py), so a parameter added mid-list silently steals
    # another one's slot. New parameters go at the end.
    codex_home: str | None = None,
) -> Any:  # Returns LLMInterface
    """
    Factory function to create the appropriate LLM provider implementation.

    Args:
        provider: Provider name ("openai", "groq", "ollama", "gemini", "anthropic", etc.).
        api_key: API key (may be None for local providers or OAuth providers).
        base_url: Base URL for the API.
        model: Model name.
        reasoning_effort: Reasoning effort level for supported providers, or None when
            the operator configured none (providers then fall back to the default level
            and may skip the parameter entirely).
        groq_service_tier: Groq service tier (for Groq provider) - "on_demand", "flex", or "auto".
        openai_service_tier: OpenAI service tier (for OpenAI provider) - None (default) or "flex" (50% cheaper).
        bedrock_service_tier: Bedrock service tier (for Bedrock provider) - None (default), "flex", "priority", or "reserved".
        gemini_service_tier: Gemini service tier (for Gemini provider) - None (default) or "flex" (50% cheaper).
        ollama_num_ctx: Native Ollama context window override. None lets Ollama use the
            model/server default.
        extra_body: Extra request-body params merged into the provider's native
            call. Threaded into OpenAI-compatible, Fireworks, Anthropic, Gemini/
            VertexAI and LiteLLM providers (each merges them in its own parameter
            space). Keys must use each provider's native names (e.g. ``max_tokens``
            for OpenAI/Anthropic vs ``max_output_tokens`` for Gemini).
        default_headers: Custom headers passed to provider SDK clients (used by operators
            routing through proxies / request-tracing middleware). Wired into the Anthropic
            provider, the ``OpenAICompatibleLLM`` branch, ``fireworks``, ``nous`` and the
            Responses API (SDK ``default_headers``), and into the LiteLLM-backed providers —
            ``litellm``, ``litellmrouter`` and ``bedrock`` — as the LiteLLM ``extra_headers``
            completion kwarg; other providers may opt in as needed.
        cache_affinity: Backend prompt-cache pinning mode, forwarded to the
            ``OpenAICompatibleLLM`` branch, ``fireworks`` and ``nous`` (all three share the
            OpenAI-compatible wire format): "none" (default), "xai_conv_id",
            "openai_prompt_cache_key", or "auto". Providers on other branches do their own
            cache work or none at all. See ``engine/cache_affinity.py``.
        structured_output_forced_tool: Ask the LiteLLM-backed providers (``litellm``,
            ``litellmrouter``, ``bedrock``) for structured output via a forced tool call
            instead of ``response_format``. For backends that reject the response_format
            route — see ``HINDSIGHT_API_LLM_STRUCTURED_OUTPUT_FORCED_TOOL``. Other
            providers ignore it.
        vertexai_project_id: Vertex AI project ID (for VertexAI provider).
        vertexai_region: Vertex AI region (for VertexAI provider).
        vertexai_credentials: Vertex AI credentials object (for VertexAI provider).
        timeout: Per-request LLM timeout in seconds (resolved by the caller from the
            per-operation/global config). Threaded into the providers that honour a
            configurable request timeout (LiteLLM, LiteLLM Router, OpenAI-compatible,
            Nous). ``None`` lets each provider fall back to its own default
            (``HINDSIGHT_API_LLM_TIMEOUT`` / ``DEFAULT_LLM_TIMEOUT`` for those four;
            Anthropic and Gemini keep their provider-specific defaults).
        codex_home: Codex credentials directory (for the openai-codex provider); overrides
            the process-wide ``CODEX_HOME``.

    Returns:
        LLMInterface implementation for the specified provider.
    """
    ollama_num_ctx = _validate_ollama_num_ctx(ollama_num_ctx)

    from .providers import (
        AnthropicLLM,
        ClaudeCodeLLM,
        CodexLLM,
        FireworksLLM,
        GeminiLLM,
        GitHubCopilotLLM,
        LiteLLMLLM,
        LiteLLMRouterLLM,
        LlamaCppLLM,
        MockLLM,
        NoneLLM,
        OpenAICompatibleLLM,
        OpenAIResponsesLLM,
    )

    provider_lower = provider.lower()
    if provider_lower == "gemini":
        from ..config import parse_gemini_service_tier

        gemini_service_tier = parse_gemini_service_tier(gemini_service_tier)
    else:
        gemini_service_tier = None

    if provider_lower == "openai-codex":
        return CodexLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
            codex_home=codex_home,
            timeout=timeout,
        )

    elif provider_lower == "claude-code":
        return ClaudeCodeLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
        )

    elif provider_lower == "github-copilot":
        return GitHubCopilotLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )

    elif provider_lower == "mock":
        return MockLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
        )

    elif provider_lower == "none":
        return NoneLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
        )

    elif provider_lower in ("gemini", "vertexai"):
        return GeminiLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
            vertexai_project_id=vertexai_project_id,
            vertexai_region=vertexai_region,
            vertexai_credentials=vertexai_credentials,
            gemini_safety_settings=gemini_safety_settings,
            gemini_service_tier=gemini_service_tier,
            prompt_cache_enabled=prompt_cache_enabled,
            extra_body=extra_body,
        )

    elif provider_lower == "anthropic":
        return AnthropicLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            default_headers=default_headers,
            extra_body=extra_body,
            timeout=timeout,
        )

    elif provider_lower == "litellm":
        return LiteLLMLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
            default_headers=default_headers,
            timeout=timeout,
            structured_output_forced_tool=structured_output_forced_tool,
        )

    elif provider_lower == "litellmrouter":
        if not litellmrouter_config:
            raise ValueError(
                "Provider 'litellmrouter' requires a config object. "
                "Set HINDSIGHT_API_LLM_LITELLMROUTER_CONFIG (or the per-op variant) "
                "to a JSON object accepted by litellm.Router. "
                "See https://docs.litellm.ai/docs/routing."
            )
        return LiteLLMRouterLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            config=litellmrouter_config,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
            default_headers=default_headers,
            timeout=timeout,
            structured_output_forced_tool=structured_output_forced_tool,
        )

    elif provider_lower == "bedrock":
        # Bedrock is a first-class alias backed by LiteLLM with auto-prefixed model names
        bedrock_model = model if model.startswith("bedrock/") else f"bedrock/{model}"
        return LiteLLMLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=bedrock_model,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
            default_headers=default_headers,
            bedrock_service_tier=bedrock_service_tier,
            timeout=timeout,
            structured_output_forced_tool=structured_output_forced_tool,
        )

    elif provider_lower == "llamacpp":
        from ..config import get_config

        config = get_config()
        return LlamaCppLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
            timeout=timeout,
            model_path=config.llamacpp_model_path,
            gpu_layers=config.llamacpp_gpu_layers,
            context_size=config.llamacpp_context_size,
            chat_format=config.llamacpp_chat_format,
            no_grammar=config.llamacpp_no_grammar,
            extra_args=config.llamacpp_extra_args,
        )

    elif provider_lower == "fireworks":
        # Fireworks online inference is OpenAI-compatible; FireworksLLM adds the
        # native (non-OpenAI) batch API on top. The existing LiteLLM
        # ``fireworks_ai/...`` online path (provider="litellm") is untouched.
        return FireworksLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
            default_headers=default_headers,
            cache_affinity=cache_affinity,
            timeout=timeout,
        )

    elif provider_lower == "nous":
        # Nous Portal is OpenAI-compatible on the wire; NousLLM adds rotating
        # inference:invoke JWT auth read natively from ~/.hermes/auth.json
        # (no static api_key, no hermes_cli dependency — same shape as Codex).
        # default_headers/cache_affinity ride NousLLM's **kwargs passthrough to
        # OpenAICompatibleLLM.__init__ unchanged (see NousLLM.__init__).
        from hindsight_api.engine.providers.nous_llm import NousLLM

        return NousLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            extra_body=extra_body,
            default_headers=default_headers,
            cache_affinity=cache_affinity,
            timeout=timeout,
        )

    elif provider_lower == "xai-oauth":
        # SuperGrok subscription lane: api.x.ai spoken plainly, but the
        # credential is a device-code OAuth grant with proactive/reactive
        # refresh over a shared on-disk store, and xAI's 403 shapes need their
        # own classification — neither fits the OpenAI SDK client, hence its
        # own provider.
        from hindsight_api.engine.providers.xai_oauth_llm import XaiOAuthLLM

        return XaiOAuthLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout=timeout,
        )

    elif provider_lower == "openai-responses":
        # OpenAI Responses API (/v1/responses). Unlike chat/completions, it
        # supports reasoning + function tools together, so reflect's tool loop
        # can run with a real reasoning_effort. See OpenAIResponsesLLM.
        return OpenAIResponsesLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            openai_service_tier=openai_service_tier,
            extra_body=extra_body,
            default_headers=default_headers,
            timeout=timeout,
        )

    elif provider_lower in (
        "openai",
        "groq",
        "ollama",
        "ollama-cloud",
        "lmstudio",
        "minimax",
        "deepseek",
        "volcano",
        "openrouter",
        "requesty",
        "zai",
        "opencode-go",
        "atlas",
    ):
        return OpenAICompatibleLLM(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            groq_service_tier=groq_service_tier,
            openai_service_tier=openai_service_tier,
            extra_body=extra_body,
            default_headers=default_headers,
            cache_affinity=cache_affinity,
            ollama_num_ctx=ollama_num_ctx,
            timeout=timeout,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


class LLMProvider:
    """
    Unified LLM provider.

    Supports OpenAI, Groq, Ollama (OpenAI-compatible), and Gemini.
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        model: str,
        reasoning_effort: str | None = None,
        groq_service_tier: str | None = None,
        openai_service_tier: str | None = None,
        bedrock_service_tier: str | None = None,
        gemini_safety_settings: list | None = None,
        prompt_cache_enabled: bool = False,
        extra_body: dict[str, Any] | None = None,
        default_headers: dict[str, str] | None = None,
        litellmrouter_config: dict[str, Any] | None = None,
        gemini_service_tier: str | None = None,
        vertexai_project_id: str | None = None,
        vertexai_region: str | None = None,
        vertexai_service_account_key: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        initial_backoff: float | None = None,
        max_backoff: float | None = None,
        ollama_num_ctx: int | None = None,
        cache_affinity: str | None = None,
        structured_output_forced_tool: bool = False,
        # Appended rather than inserted — see the note on ``create_llm_provider``:
        # callers that pass these positionally would otherwise have one argument
        # land in the wrong slot.
        codex_home: str | None = None,
    ):
        """
        Initialize LLM provider.

        Args:
            provider: Provider name ("openai", "groq", "ollama", "gemini", "anthropic", "lmstudio").
            api_key: API key.
            base_url: Base URL for the API.
            model: Model name.
            reasoning_effort: Reasoning effort level for supported providers, or None
                when the operator configured none.
            groq_service_tier: Groq service tier ("on_demand", "flex", "auto") - from config.
            openai_service_tier: OpenAI service tier (None or "flex") - from config.
            bedrock_service_tier: Bedrock service tier (None, "flex", "priority", "reserved") - from config.
            gemini_service_tier: Gemini service tier (None or "flex") - from config.
            ollama_num_ctx: Native Ollama context window override. ``None`` lets Ollama
                use the model/server default.
            gemini_safety_settings: Safety settings for Gemini/VertexAI providers.
            extra_body: Extra request-body params merged into the provider's native call
                (OpenAI-compatible, Fireworks, Anthropic, Gemini/VertexAI, LiteLLM).
            default_headers: Custom headers passed as ``default_headers`` to provider SDK clients.
                Used by operators routing through proxies / request-tracing middleware.
            cache_affinity: Backend prompt-cache pinning mode for the OpenAI-compatible and
                Fireworks providers ("none", "xai_conv_id", "openai_prompt_cache_key",
                "auto"). Validated here for every provider so a typo never fails silently;
                providers on other factory branches ignore it. Used verbatim — callers
                resolve the per-operation/global fallback.
            litellmrouter_config: Provider-specific config for ``provider="litellmrouter"``.
                JSON object passed verbatim to ``litellm.Router(**config)`` — see
                https://docs.litellm.ai/docs/routing. Ignored unless ``provider == "litellmrouter"``.
            vertexai_project_id: Vertex AI project ID for ``provider="vertexai"`` (required for
                that provider).
            vertexai_region: Vertex AI region for ``provider="vertexai"`` (defaults to
                ``"us-central1"`` when ``None``).
            vertexai_service_account_key: Path to a Vertex AI service-account key file for
                ``provider="vertexai"`` (uses ADC when ``None``).
            timeout: Per-request LLM timeout in seconds. Resolved by the caller from the
                per-operation/global config (``retain_llm_timeout`` falling back to
                ``llm_timeout``, etc.). ``None`` lets each provider apply its own default.
            max_retries: Default retry-attempt budget for ``call`` / ``call_with_tools``
                when the per-call argument is omitted. Resolved by the caller from the
                per-operation/global config (``reflect_llm_max_retries`` falling back to
                ``llm_max_retries``, etc.). ``None`` keeps each method's own fallback.
            initial_backoff: Default initial retry backoff (seconds), same resolution as
                ``max_retries``. ``None`` keeps each method's own fallback.
            max_backoff: Default maximum retry backoff (seconds), same resolution as
                ``max_retries``. ``None`` keeps each method's own fallback.
            structured_output_forced_tool: Structured output via a forced tool call
                instead of ``response_format``, for the LiteLLM-backed providers - from
                config (``HINDSIGHT_API_LLM_STRUCTURED_OUTPUT_FORCED_TOOL``).
            codex_home: Codex credentials directory for ``provider="openai-codex"`` — the
                directory holding the ``auth.json`` this provider authenticates with. ``None``
                uses the process-wide ``CODEX_HOME`` (else ``~/.codex``). Set it per member of a
                multi-LLM chain to run two independently authorized ChatGPT profiles, so that
                failover away from a rate-limited profile actually reaches a different account.

        This constructor uses every argument as passed and does not read global
        ``HindsightConfig``: resolving the server-level default for a ``None`` argument is the
        caller's responsibility (see ``MemoryEngine``'s per-op builds, ``_member_to_llm``, and
        ``LLMProvider.from_env``). Keeping it config-free makes a provider's effective settings a
        pure function of its arguments — which is what lets each member of a multi-LLM chain be
        configured independently.
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.reasoning_effort = reasoning_effort
        # Per-request timeout (seconds). Used verbatim — the caller resolves the
        # per-operation/global fallback. ``None`` defers to the provider default.
        self.timeout = timeout
        # Default retry policy for call()/call_with_tools(). The caller resolves the
        # per-operation/global fallback; ``None`` keeps each method's own fallback so
        # providers built without a resolved config (from_env, tests) are unchanged.
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.litellmrouter_config = litellmrouter_config
        # Codex credentials directory (openai-codex only). Used verbatim — the caller
        # resolves the server-level default, like the fields around it.
        self.codex_home = codex_home
        # Service tiers from hierarchical config (not env vars)
        self.groq_service_tier = groq_service_tier
        self.openai_service_tier = openai_service_tier
        self.bedrock_service_tier = bedrock_service_tier
        self.gemini_service_tier = gemini_service_tier
        # Structured-output transport for the LiteLLM-backed providers. Used verbatim —
        # the caller resolves the server-level default, like the fields above.
        self.structured_output_forced_tool = structured_output_forced_tool
        self.ollama_num_ctx = _validate_ollama_num_ctx(ollama_num_ctx)
        # Gemini safety settings (instance default; can be overridden per-request via context var)
        self.gemini_safety_settings = gemini_safety_settings
        # Gemini prompt caching: when True, retain extraction (and any future
        # caller that opts in) will reuse a CachedContent prefix to cut
        # input-token cost. Off by default so the change is observable behind
        # a flip rather than a silent behaviour change on upgrade.
        self.prompt_cache_enabled = prompt_cache_enabled
        # Extra body params for OpenAI-compatible providers (e.g. chat_template_kwargs)
        self.extra_body = extra_body
        # Default headers passed to provider SDK clients (e.g. proxy auth, request tracing).
        # Used verbatim — callers resolve the global fallback (see _member_to_llm /
        # the per-op builds in MemoryEngine, and LLMProvider.from_env).
        self.default_headers = default_headers
        # Backend prompt-cache pinning mode. Validated here rather than only at the
        # provider so a typo fails for every provider, not just the ones that act on
        # it — the setting has no visible effect in the response, so a silent
        # fallback to "none" would be indistinguishable from it working.
        self.cache_affinity = parse_cache_affinity(cache_affinity).value

        # Validate provider
        valid_providers = [
            "openai",
            "openai-responses",
            "groq",
            "ollama",
            "ollama-cloud",
            "gemini",
            "anthropic",
            "lmstudio",
            "llamacpp",
            "vertexai",
            "openai-codex",
            "claude-code",
            "github-copilot",
            "mock",
            "none",
            "minimax",
            "deepseek",
            "litellm",
            "litellmrouter",
            "bedrock",
            "volcano",
            "openrouter",
            "requesty",
            "zai",
            "opencode-go",
            "atlas",
            "fireworks",
            "nous",
            "xai-oauth",
        ]
        if self.provider not in valid_providers:
            raise ValueError(f"Invalid LLM provider: {self.provider}. Must be one of: {', '.join(valid_providers)}")

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
            elif self.provider == "nous":
                self.base_url = "https://inference-api.nousresearch.com/v1"

        # Prepare Vertex AI config (if applicable). Values are used as passed; the
        # caller resolves the global-config fallback (MemoryEngine builds /
        # _member_to_llm / from_env). The region keeps a constant default here.
        vertexai_credentials = None

        if self.provider == "vertexai":
            if not vertexai_project_id:
                raise ValueError(
                    "HINDSIGHT_API_LLM_VERTEXAI_PROJECT_ID is required for Vertex AI provider. "
                    "Set it to your GCP project ID."
                )

            vertexai_region = vertexai_region or "us-central1"
            service_account_key = vertexai_service_account_key

            # Load explicit service account credentials if provided
            if service_account_key:
                if not VERTEXAI_AVAILABLE:
                    raise ValueError(
                        "Vertex AI service account auth requires 'google-auth' package. "
                        "Install with: pip install google-auth"
                    )
                vertexai_credentials = service_account.Credentials.from_service_account_file(
                    service_account_key,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                logger.info(f"Vertex AI: Using service account key: {service_account_key}")

            # Strip google/ prefix from model name — native SDK uses bare names
            if self.model.startswith("google/"):
                self.model = self.model[len("google/") :]

            logger.info(
                f"Vertex AI: project={vertexai_project_id}, region={vertexai_region}, "
                f"model={self.model}, auth={'service_account' if service_account_key else 'ADC'}"
            )

        # Normalize the Gemini service tier (pure: maps/validates the passed value,
        # no global config read). Non-Gemini providers never carry a tier. The
        # server-level default is resolved by the caller, like the other fields.
        if self.provider == "gemini":
            from ..config import parse_gemini_service_tier

            self.gemini_service_tier = parse_gemini_service_tier(self.gemini_service_tier)
        else:
            self.gemini_service_tier = None

        # gemini_safety_settings / prompt_cache_enabled / litellmrouter_config are
        # used as passed — the caller resolves the global-config fallback. Providers
        # that don't support prompt caching ignore the flag.
        router_config: dict[str, Any] | None = self.litellmrouter_config

        # Create provider implementation using factory
        self._provider_impl = create_llm_provider(
            provider=self.provider,
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            groq_service_tier=self.groq_service_tier,
            openai_service_tier=self.openai_service_tier,
            bedrock_service_tier=self.bedrock_service_tier,
            gemini_service_tier=self.gemini_service_tier,
            codex_home=self.codex_home,
            extra_body=self.extra_body,
            default_headers=self.default_headers,
            vertexai_project_id=vertexai_project_id,
            vertexai_region=vertexai_region,
            vertexai_credentials=vertexai_credentials,
            gemini_safety_settings=self.gemini_safety_settings,
            prompt_cache_enabled=self.prompt_cache_enabled,
            litellmrouter_config=router_config,
            ollama_num_ctx=self.ollama_num_ctx,
            timeout=self.timeout,
            cache_affinity=self.cache_affinity,
            structured_output_forced_tool=self.structured_output_forced_tool,
        )

        # Backward compatibility: Keep mock provider properties
        self._mock_calls: list[dict] = []
        self._mock_response: Any = None

    @property
    def _client(self) -> Any:
        """
        Get the OpenAI client for OpenAI-compatible providers.

        This property provides backward compatibility for code that directly accesses
        the _client attribute (e.g., benchmarks, memory_engine).

        Returns:
            AsyncOpenAI client instance for OpenAI-compatible providers, or None for other providers.
        """
        from .providers.openai_compatible_llm import OpenAICompatibleLLM

        if isinstance(self._provider_impl, OpenAICompatibleLLM):
            return self._provider_impl._client
        return None

    @property
    def _gemini_client(self) -> Any:
        """
        Get the Gemini client for Gemini/VertexAI providers.

        This property provides backward compatibility for code that directly accesses
        the _gemini_client attribute.

        Returns:
            genai.Client instance for Gemini/VertexAI providers, or None for other providers.
        """
        from .providers.gemini_llm import GeminiLLM

        if isinstance(self._provider_impl, GeminiLLM):
            return self._provider_impl._client
        return None

    async def verify_connection(self) -> None:
        """
        Verify that the LLM provider is configured correctly by making a simple test call.

        Raises:
            RuntimeError: If the connection test fails.
        """
        await self._provider_impl.verify_connection()

    async def supports_batch_api(self) -> bool:
        """Whether the underlying provider supports the OpenAI/Groq Batch API."""
        return await self._provider_impl.supports_batch_api()

    async def batch_provider_impl(self, account_key: str | None = None) -> LLMInterface | None:
        """The implementation serving batch, or ``None`` when it cannot serve one.

        Exists so the batch path (``extract_facts_from_contents_batch_api``) can
        target the implementation through the same interface as a multi-LLM chain:
        ``MultiLLMProvider`` returns the first batch-capable member's impl here,
        while a single provider returns its own. Returning ``None`` rather than a
        provider that would reject every batch call keeps the "can it serve one?"
        answer in one place — the caller raises on ``None``.

        ``account_key`` is a :attr:`LLMInterface.batch_account_key` persisted when
        an in-flight batch was submitted. Passing it restricts the answer to the
        account that actually owns that batch, so a same-provider lookalike is
        rejected (``None``) instead of being handed the wrong credentials.
        """
        if not await self._provider_impl.supports_batch_api():
            return None
        if account_key is not None and self._provider_impl.batch_account_key != account_key:
            return None
        return self._provider_impl

    async def call(
        self,
        messages: list[dict[str, str]],
        response_format: Any | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        scope: str = "memory",
        max_retries: int | None = None,
        initial_backoff: float | None = None,
        max_backoff: float | None = None,
        skip_validation: bool = False,
        strict_schema: bool | None = None,
        return_usage: bool = False,
        cached_prefix: str | None = None,
    ) -> Any:
        """
        Make an LLM API call with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            response_format: Optional Pydantic model for structured output.
            max_completion_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0-2.0).
            scope: Scope identifier for tracking.
            max_retries: Maximum retry attempts. ``None`` uses the provider's configured
                default (per-operation/global ``llm_max_retries``), else 10.
            initial_backoff: Initial backoff time in seconds. ``None`` uses the provider's
                configured default (``llm_initial_backoff``), else 1.0.
            max_backoff: Maximum backoff time in seconds. ``None`` uses the provider's
                configured default (``llm_max_backoff``), else 60.0.
            skip_validation: Return raw JSON without Pydantic validation.
            strict_schema: Per-call override requesting grammar-enforced (json_schema strict)
                structured output instead of the soft json_object path. None (the default)
                inherits the server-level HINDSIGHT_API_LLM_STRICT_SCHEMA flag; an explicit
                True or False wins over it, so a caller can force strict output on -- or off --
                for its own scope. Providers without a strict mode ignore it.
            return_usage: If True, return tuple (result, TokenUsage) instead of just result.

        Returns:
            If return_usage=False: Parsed response if response_format is provided, otherwise text content.
            If return_usage=True: Tuple of (result, TokenUsage) with token counts from the LLM call.

        Raises:
            OutputTooLongError: If output exceeds token limits.
            Exception: Re-raises API errors after retries exhausted.
        """
        # Stage breadcrumb so the worker log shows which LLM call a task is
        # currently inside; the stage_age field then reveals long JSON-schema
        # retry loops (e.g. a small model that can't satisfy strict_schema).
        # No-op outside a worker context.
        from ..worker.stage import set_stage

        structured = "+structured" if response_format is not None else ""
        # `.queued` until the concurrency permits are in hand — see the acquire
        # below. Without it, a call waiting on a saturated semaphore is
        # indistinguishable from one the provider is actively running, and the
        # label points at the provider (#3002: an operator lost an hour to
        # "llm.bedrock.*" for tasks that had never reached Bedrock).
        base_stage = f"llm.{self.provider}.{scope}{structured}"
        set_stage(f"{base_stage}.queued")

        # Resolve the retry policy: explicit per-call arg wins, else the provider's
        # configured per-operation/global default, else this method's own fallback.
        max_retries = (
            max_retries if max_retries is not None else (self.max_retries if self.max_retries is not None else 10)
        )
        initial_backoff = (
            initial_backoff
            if initial_backoff is not None
            else (self.initial_backoff if self.initial_backoff is not None else 1.0)
        )
        max_backoff = (
            max_backoff if max_backoff is not None else (self.max_backoff if self.max_backoff is not None else 60.0)
        )

        # Resolve strict-schema once, here, rather than in each provider: the
        # per-call argument, falling back to the server-level
        # HINDSIGHT_API_LLM_STRICT_SCHEMA flag when the caller expressed no
        # preference. Providers with a json_schema response_format (OpenAI-compatible,
        # LiteLLM) then grammar-enforce structured output instead of the fragile
        # soft json_object path; Gemini already enforces its native response_schema,
        # and providers without a strict mode simply ignore the flag.
        from ..config import get_config

        # An explicit per-call value wins in BOTH directions -- `or` would have made a
        # per-call False indistinguishable from "unset", silently ignoring any caller
        # that opts out while the global flag is on.
        strict_schema = strict_schema if strict_schema is not None else get_config().llm_strict_schema

        # LLM call observability flows through the OTel GenAI recorder
        # (tracing.get_span_recorder().record_llm_call). Provider implementations
        # record successful calls; we forward failures here since they don't.
        # The requested params are stashed in a contextvar (only what the caller
        # actually set) so the recorder can attach them to either path.
        from ..tracing import get_span_recorder
        from .llm_trace import (
            current_response_usage,
            reset_request_context,
            reset_response_usage,
            set_request_context,
            set_response_usage,
        )

        call_start = time.monotonic()
        request_token = set_request_context(
            _request_params(
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                scope=scope,
                response_format=response_format,
            )
        )
        # Cleared per call; the provider stashes real usage once a response is in
        # hand so the error path below can attach it if parsing/validation fails.
        usage_token = set_response_usage(None)
        try:
            # Providers that own retry loops acquire the shared permits for each
            # upstream attempt so backoff never occupies request capacity.
            attempt_gated = self._provider_impl.supports_attempt_scoped_concurrency()
            async with AsyncExitStack() as stack:
                if not attempt_gated:
                    for sem in _semaphores_for_scope(scope):
                        await stack.enter_async_context(sem)
                    # Permits in hand — only now leave `.queued`. Attempt-gated
                    # providers acquire permits per attempt instead, so they keep
                    # `.queued` until their first `attempt=N` stamp lands after
                    # the permit acquire inside attempt_context (#3002).
                    set_stage(base_stage)

                # cached_prefix is only set for providers that returned a handle
                # from get_or_create_cached_prefix() (e.g. Gemini); it's None for
                # the rest. Forward it only when present so providers that don't
                # implement caching keep their call() signature untouched.
                cache_kwarg = {"cached_prefix": cached_prefix} if cached_prefix is not None else {}
                try:
                    # Delegate to provider implementation
                    attempt_kwarg = {"attempt_context": lambda: _attempt_permits(scope)} if attempt_gated else {}
                    result = await self._provider_impl.call(
                        messages=messages,
                        response_format=response_format,
                        max_completion_tokens=max_completion_tokens,
                        temperature=temperature,
                        scope=scope,
                        max_retries=max_retries,
                        initial_backoff=initial_backoff,
                        max_backoff=max_backoff,
                        skip_validation=skip_validation,
                        strict_schema=strict_schema,
                        return_usage=return_usage,
                        **cache_kwarg,
                        **attempt_kwarg,
                    )
                except Exception as e:
                    # The provider call may have succeeded (and incurred token
                    # cost) before local parsing/validation raised; attach the
                    # provider-reported usage to the error trace when available.
                    usage = current_response_usage()
                    get_span_recorder().record_llm_call(
                        provider=self.provider,
                        model=self.model,
                        scope=scope,
                        messages=messages,
                        response_content=None,
                        input_tokens=usage.input_tokens if usage else 0,
                        output_tokens=usage.output_tokens if usage else 0,
                        cached_tokens=usage.cached_tokens if usage else 0,
                        duration=time.monotonic() - call_start,
                        error=e,
                    )
                    raise

                # Backward compatibility: Update mock call tracking for mock provider
                # This allows existing tests using LLMProvider._mock_calls to continue working
                if self.provider == "mock":
                    from .providers.mock_llm import MockLLM

                    if isinstance(self._provider_impl, MockLLM):
                        # Sync the mock calls from provider implementation to wrapper
                        self._mock_calls = self._provider_impl.get_mock_calls()
        finally:
            reset_request_context(request_token)
            reset_response_usage(usage_token)

        # Single scrub point for every structured/text LLM response in the engine:
        # a model can emit a lone `\udXXX` escape that JSON decoding turns into an
        # un-encodable surrogate, and the field it lands in is not knowable here
        # (#3729). Clean output is returned unchanged, object identity included.
        return sanitize_llm_value(result)

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
        scope: str = "tools",
        max_retries: int | None = None,
        initial_backoff: float | None = None,
        max_backoff: float | None = None,
        tool_choice: LLMToolChoice = LLM_TOOL_CHOICE_AUTO,
        cached_prefix: str | None = None,
        cached_prefix_message_count: int = 0,
    ) -> "LLMToolCallResult":
        """
        Make an LLM API call with tool/function calling support.

        Args:
            messages: List of message dicts. Can include tool results with role='tool'.
            tools: List of tool definitions in OpenAI format.
            max_completion_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0-2.0).
            scope: Scope identifier for tracking.
            max_retries: Maximum retry attempts. ``None`` uses the provider's configured
                default (per-operation/global ``llm_max_retries``), else 5.
            initial_backoff: Initial backoff time in seconds. ``None`` uses the provider's
                configured default (``llm_initial_backoff``), else 1.0.
            max_backoff: Maximum backoff time in seconds. ``None`` uses the provider's
                configured default (``llm_max_backoff``), else 30.0.
            tool_choice: Canonical tool-selection policy.

        Returns:
            LLMToolCallResult with content and/or tool_calls.
        """
        from ..worker.stage import set_stage

        # `.queued` until the permits are held — see the structured path above.
        base_stage = f"llm.{self.provider}.{scope}+tools"
        set_stage(f"{base_stage}.queued")

        # Resolve the retry policy: explicit per-call arg wins, else the provider's
        # configured per-operation/global default, else this method's own fallback.
        max_retries = (
            max_retries if max_retries is not None else (self.max_retries if self.max_retries is not None else 5)
        )
        initial_backoff = (
            initial_backoff
            if initial_backoff is not None
            else (self.initial_backoff if self.initial_backoff is not None else 1.0)
        )
        max_backoff = (
            max_backoff if max_backoff is not None else (self.max_backoff if self.max_backoff is not None else 30.0)
        )

        # Failures forwarded to the GenAI recorder; successes recorded by providers.
        from ..tracing import get_span_recorder
        from .llm_trace import (
            current_response_usage,
            reset_request_context,
            reset_response_usage,
            set_request_context,
            set_response_usage,
        )

        call_start = time.monotonic()
        request_token = set_request_context(
            _request_params(
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                scope=scope,
                tool_choice=tool_choice,
            )
        )
        # Cleared per call; the provider stashes real usage once a response is in
        # hand so the error path below can attach it if parsing/validation fails.
        usage_token = set_response_usage(None)
        try:
            attempt_gated = self._provider_impl.supports_attempt_scoped_concurrency()
            async with AsyncExitStack() as stack:
                if not attempt_gated:
                    for sem in _semaphores_for_scope(scope):
                        await stack.enter_async_context(sem)
                    # Permits in hand — only now leave `.queued`; attempt-gated
                    # providers stay `.queued` until their first post-acquire
                    # `attempt=N` stamp (see call() above, #3002).
                    set_stage(base_stage)

                # cached_prefix is only set for providers that returned a handle
                # from get_or_create_cached_prefix() / create_incremental_cache();
                # forward it (plus how many leading messages it covers) only when
                # present so non-caching providers keep their signature.
                cache_kwarg = (
                    {"cached_prefix": cached_prefix, "cached_prefix_message_count": cached_prefix_message_count}
                    if cached_prefix is not None
                    else {}
                )
                try:
                    # Delegate to provider implementation
                    attempt_kwarg = {"attempt_context": lambda: _attempt_permits(scope)} if attempt_gated else {}
                    result = await self._provider_impl.call_with_tools(
                        messages=messages,
                        tools=tools,
                        max_completion_tokens=max_completion_tokens,
                        temperature=temperature,
                        scope=scope,
                        max_retries=max_retries,
                        initial_backoff=initial_backoff,
                        max_backoff=max_backoff,
                        tool_choice=tool_choice,
                        **cache_kwarg,
                        **attempt_kwarg,
                    )
                except Exception as e:
                    # The provider call may have succeeded (and incurred token
                    # cost) before local parsing/validation raised; attach the
                    # provider-reported usage to the error trace when available.
                    usage = current_response_usage()
                    get_span_recorder().record_llm_call(
                        provider=self.provider,
                        model=self.model,
                        scope=scope,
                        messages=messages,
                        response_content=None,
                        input_tokens=usage.input_tokens if usage else 0,
                        output_tokens=usage.output_tokens if usage else 0,
                        cached_tokens=usage.cached_tokens if usage else 0,
                        duration=time.monotonic() - call_start,
                        error=e,
                    )
                    raise

                # Backward compatibility: Update mock call tracking for mock provider
                # This allows existing tests using LLMProvider._mock_calls to continue working
                if self.provider == "mock":
                    from .providers.mock_llm import MockLLM

                    if isinstance(self._provider_impl, MockLLM):
                        # Sync the mock calls from provider implementation to wrapper
                        self._mock_calls = self._provider_impl.get_mock_calls()
        finally:
            reset_request_context(request_token)
            reset_response_usage(usage_token)

        # Same scrub for the tool-calling path: the agent's text content and every
        # tool-call argument are model-authored and flow on to storage and reranking.
        return sanitize_llm_value(result)

    def set_response_callback(self, fn: Any) -> None:
        """Set a callback invoked on each call() instead of the fixed mock response."""
        if self.provider == "mock":
            from .providers.mock_llm import MockLLM

            if isinstance(self._provider_impl, MockLLM):
                self._provider_impl.set_response_callback(fn)

    def set_mock_response(self, response: Any) -> None:
        """Set the response to return from mock calls."""
        # Backward compatibility: Store in both wrapper and provider implementation
        self._mock_response = response
        if self.provider == "mock":
            from .providers.mock_llm import MockLLM

            if isinstance(self._provider_impl, MockLLM):
                self._provider_impl.set_mock_response(response)

    def get_mock_calls(self) -> list[dict]:
        """Get the list of recorded mock calls."""
        # Backward compatibility: Read from provider implementation if mock provider
        if self.provider == "mock":
            from .providers.mock_llm import MockLLM

            if isinstance(self._provider_impl, MockLLM):
                return self._provider_impl.get_mock_calls()
        return self._mock_calls

    def clear_mock_calls(self) -> None:
        """Clear the recorded mock calls."""
        # Backward compatibility: Clear in both wrapper and provider implementation
        self._mock_calls = []
        if self.provider == "mock":
            from .providers.mock_llm import MockLLM

            if isinstance(self._provider_impl, MockLLM):
                self._provider_impl.clear_mock_calls()

    def _load_codex_auth(self) -> tuple[str, str]:
        """
        Load OAuth credentials from the Codex ``auth.json``.

        Honors this provider's ``codex_home``, then ``CODEX_HOME`` (falling back
        to ``~/.codex``).

        Returns:
            Tuple of (access_token, account_id).

        Raises:
            FileNotFoundError: If auth file doesn't exist.
            ValueError: If auth file is invalid.
        """
        from .providers.codex_auth import default_codex_auth_file

        auth_file = default_codex_auth_file(self.codex_home)

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

            from claude_agent_sdk import query  # noqa: F401  # type: ignore[unresolved-import]

            sdk_logging.getLogger("claude_agent_sdk").setLevel(sdk_logging.WARNING)
            sdk_logging.getLogger("claude_agent_sdk._internal").setLevel(sdk_logging.WARNING)

            logger.debug("Claude Agent SDK imported successfully")
        except ImportError as e:
            raise ImportError(
                "Claude Agent SDK not installed. Run: uv add claude-agent-sdk or pip install claude-agent-sdk"
            ) from e

        # SDK will automatically check for authentication when first used
        # No need to verify here - let it fail gracefully on first call with helpful error

    def with_config(
        self,
        config: Any,
        *,
        bank_id: str | None = None,
        operation: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ConfiguredLLMProvider":
        """
        Return a configured wrapper for a specific bank operation.

        The wrapper applies per-bank overrides (e.g. Gemini safety settings)
        to every ``call()`` / ``call_with_tools()`` invocation without
        changing the underlying provider or its long-lived client connection.

        Args:
            config: Resolved ``HindsightConfig`` for the current bank/request.
            bank_id: Bank the operation runs for; attributed to LLM trace rows.
            operation: Logical operation label ("retain", "reflect", ...) for
                LLM trace rows.
            metadata: Optional extra caller metadata stored on trace rows.

        Returns:
            A ``ConfiguredLLMProvider`` that delegates to this provider with
            the supplied config applied.
        """
        trace_ctx = None
        if bank_id is not None or operation is not None or metadata:
            from .llm_trace import LLMTraceContext

            # One trace + operation span per with_config() call — i.e. per
            # operation invocation. Every LLM call made through this wrapper
            # shares them, so a reflect/retain/consolidation run groups its
            # calls as parent (operation) → children (LLM calls).
            trace_ctx = LLMTraceContext(
                bank_id=bank_id,
                operation=operation,
                metadata=dict(metadata or {}),
                trace_id=str(uuid.uuid4()),
                operation_span_id=str(uuid.uuid4()),
            )
        return ConfiguredLLMProvider(self, config.llm_gemini_safety_settings, trace_ctx)

    async def cleanup(self) -> None:
        """Clean up resources (e.g. stop llamacpp subprocess)."""
        if self._provider_impl:
            await self._provider_impl.cleanup()

    @classmethod
    def from_env(cls) -> "LLMProvider":
        """Create provider from environment variables using config.py constants."""
        # Read every field straight from the environment. The constructor no longer
        # resolves global-config fallbacks, so this factory must supply them — and it
        # does so without building the full HindsightConfig, keeping from_env() a
        # lightweight env-only loader (see test_llm_provider_from_env_keeps_lightweight_loader).
        from ..config import (
            DEFAULT_LLM_CACHE_AFFINITY,
            DEFAULT_LLM_GROQ_SERVICE_TIER,
            DEFAULT_LLM_OPENAI_SERVICE_TIER,
            DEFAULT_LLM_PROMPT_CACHE_ENABLED,
            DEFAULT_LLM_PROVIDER,
            DEFAULT_LLM_STRUCTURED_OUTPUT_FORCED_TOOL,
            DEFAULT_LLM_TIMEOUT,
            ENV_LLM_API_KEY,
            ENV_LLM_BASE_URL,
            ENV_LLM_BEDROCK_SERVICE_TIER,
            ENV_LLM_CACHE_AFFINITY,
            ENV_LLM_CODEX_HOME,
            ENV_LLM_DEFAULT_HEADERS,
            ENV_LLM_EXTRA_BODY,
            ENV_LLM_GEMINI_SAFETY_SETTINGS,
            ENV_LLM_GEMINI_SERVICE_TIER,
            ENV_LLM_GROQ_SERVICE_TIER,
            ENV_LLM_LITELLMROUTER_CONFIG,
            ENV_LLM_MODEL,
            ENV_LLM_OLLAMA_NUM_CTX,
            ENV_LLM_OPENAI_SERVICE_TIER,
            ENV_LLM_PROMPT_CACHE_ENABLED,
            ENV_LLM_PROVIDER,
            ENV_LLM_REASONING_EFFORT,
            ENV_LLM_STRUCTURED_OUTPUT_FORCED_TOOL,
            ENV_LLM_TIMEOUT,
            ENV_LLM_VERTEXAI_PROJECT_ID,
            ENV_LLM_VERTEXAI_REGION,
            ENV_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY,
            _get_default_model_for_provider,
            _parse_boolean_env,
            _parse_llm_router_config,
            _parse_optional_positive_int,
            parse_gemini_service_tier,
        )

        provider = os.getenv(ENV_LLM_PROVIDER, DEFAULT_LLM_PROVIDER)
        api_key = os.getenv(ENV_LLM_API_KEY, "")

        if not api_key and not requires_api_key(provider):
            pass  # Provider handles its own auth
        elif not api_key:
            raise ValueError(f"{ENV_LLM_API_KEY} environment variable is required for provider '{provider}'")

        base_url = os.getenv(ENV_LLM_BASE_URL, "")
        model = os.getenv(ENV_LLM_MODEL) or _get_default_model_for_provider(provider)
        extra_body = json.loads(os.getenv(ENV_LLM_EXTRA_BODY, "null"))
        default_headers = json.loads(os.getenv(ENV_LLM_DEFAULT_HEADERS, "null"))
        # Same default as HindsightConfig.from_env: this entry point must not
        # resolve to a different mode than the engine's own config path.
        cache_affinity = os.getenv(ENV_LLM_CACHE_AFFINITY, DEFAULT_LLM_CACHE_AFFINITY) or None
        prompt_cache_enabled = os.getenv(
            ENV_LLM_PROMPT_CACHE_ENABLED, str(DEFAULT_LLM_PROMPT_CACHE_ENABLED)
        ).lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=os.getenv(ENV_LLM_REASONING_EFFORT) or None,
            extra_body=extra_body,
            default_headers=default_headers,
            cache_affinity=cache_affinity,
            groq_service_tier=os.getenv(ENV_LLM_GROQ_SERVICE_TIER, DEFAULT_LLM_GROQ_SERVICE_TIER),
            openai_service_tier=os.getenv(ENV_LLM_OPENAI_SERVICE_TIER, DEFAULT_LLM_OPENAI_SERVICE_TIER),
            bedrock_service_tier=os.getenv(ENV_LLM_BEDROCK_SERVICE_TIER) or None,
            gemini_service_tier=(
                parse_gemini_service_tier(os.getenv(ENV_LLM_GEMINI_SERVICE_TIER))
                if provider.lower() == "gemini"
                else None
            ),
            gemini_safety_settings=json.loads(os.getenv(ENV_LLM_GEMINI_SAFETY_SETTINGS, "null")),
            prompt_cache_enabled=prompt_cache_enabled,
            ollama_num_ctx=_parse_optional_positive_int(ENV_LLM_OLLAMA_NUM_CTX, os.getenv(ENV_LLM_OLLAMA_NUM_CTX)),
            litellmrouter_config=_parse_llm_router_config(ENV_LLM_LITELLMROUTER_CONFIG),
            codex_home=os.getenv(ENV_LLM_CODEX_HOME) or None,
            vertexai_project_id=os.getenv(ENV_LLM_VERTEXAI_PROJECT_ID) or None,
            vertexai_region=os.getenv(ENV_LLM_VERTEXAI_REGION) or None,
            vertexai_service_account_key=os.getenv(ENV_LLM_VERTEXAI_SERVICE_ACCOUNT_KEY) or None,
            timeout=float(os.getenv(ENV_LLM_TIMEOUT, str(DEFAULT_LLM_TIMEOUT))),
            structured_output_forced_tool=_parse_boolean_env(
                ENV_LLM_STRUCTURED_OUTPUT_FORCED_TOOL,
                DEFAULT_LLM_STRUCTURED_OUTPUT_FORCED_TOOL,
            ),
        )


class ConfiguredLLMProvider:
    """
    Thin wrapper around LLMProvider that applies bank-specific config to every call.

    Obtained via ``LLMProvider.with_config(resolved_config)``.  The wrapper
    sets any provider-specific overrides (currently Gemini safety settings)
    immediately before each call using a ContextVar token, then resets it
    afterwards — so nesting is safe and the configuration cannot leak across
    operations.

    All attribute access falls through to the underlying provider so callers
    that read ``llm.provider``, ``llm.model``, etc. continue to work without
    any changes.
    """

    def __init__(
        self,
        provider: "LLMProvider",
        gemini_safety_settings: list | None,
        trace_ctx: Any | None = None,
    ) -> None:
        # Use object.__setattr__ to avoid triggering __getattr__
        object.__setattr__(self, "_provider", provider)
        object.__setattr__(self, "_gemini_safety_settings", gemini_safety_settings)
        object.__setattr__(self, "_trace_ctx", trace_ctx)

    # ── attribute passthrough ──────────────────────────────────────────────────

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_provider"), name)

    # ── overridden call methods ────────────────────────────────────────────────

    async def call(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        from .providers.gemini_llm import _safety_settings_ctx

        token = _safety_settings_ctx.set(object.__getattribute__(self, "_gemini_safety_settings"))
        trace_token = self._bind_trace_context()
        try:
            return await object.__getattribute__(self, "_provider").call(messages=messages, **kwargs)
        finally:
            _safety_settings_ctx.reset(token)
            self._reset_trace_context(trace_token)

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> "LLMToolCallResult":
        from .providers.gemini_llm import _safety_settings_ctx

        token = _safety_settings_ctx.set(object.__getattribute__(self, "_gemini_safety_settings"))
        trace_token = self._bind_trace_context()
        try:
            return await object.__getattribute__(self, "_provider").call_with_tools(
                messages=messages, tools=tools, **kwargs
            )
        finally:
            _safety_settings_ctx.reset(token)
            self._reset_trace_context(trace_token)

    def trace_context(self) -> Any | None:
        """The operation-level LLM trace context (or None when untraced).

        Lets the engine attach the operation's produced/consumed memory_ids to
        this run's trace rows once they're known (after the LLM calls).
        """
        return object.__getattribute__(self, "_trace_ctx")

    def _bind_trace_context(self) -> Any | None:
        """Bind bank/operation attribution for the duration of one call."""
        trace_ctx = object.__getattribute__(self, "_trace_ctx")
        if trace_ctx is None:
            return None
        from .llm_trace import set_trace_context

        return set_trace_context(trace_ctx)

    def _reset_trace_context(self, trace_token: Any | None) -> None:
        if trace_token is None:
            return
        from .llm_trace import reset_trace_context

        reset_trace_context(trace_token)


# Backwards compatibility alias
LLMConfig = LLMProvider
