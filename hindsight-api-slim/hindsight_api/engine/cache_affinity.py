"""Server-side prompt-cache affinity hints for OpenAI-compatible providers.

Prompt caching only pays off when the same conversation reaches the same backend
cache, and providers expose different mechanisms for that:

- xAI stores prompt-cache entries **per backend server** and routes requests
  carrying the same ``x-grok-conv-id`` to one server (docs.x.ai, "Maximizing
  Cache Hits"). Without it, consecutive calls of one agentic loop can each land
  on a cache-cold replica.
- OpenAI accepts a ``prompt_cache_key`` request field that improves its own
  cache routing.

Hindsight already does provider-specific cache work for its first-class
providers (``anthropic_llm`` sets ``cache_control`` breakpoints; ``gemini_llm``
runs an explicit ``CachedContent`` manager). This module is the equivalent for
the OpenAI-compatible family — ``OpenAICompatibleLLM`` and its ``fireworks``
and ``nous`` subclasses — which sent no affinity hint at all.

Default ``auto`` per member (``cache_affinity``). ``auto`` is an allowlist, not a
best-effort probe: it emits a hint only for hosts documented to accept one and
resolves to ``none`` for everything else, so an unknown OpenAI-compatible backend
never receives an unfamiliar field. Every helper here is fail-open — when no id
can be derived the request goes out byte-identical to before. Set ``none`` to
disable entirely.
"""

from __future__ import annotations

import hashlib
import json
import logging
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# xAI's documented cache-pinning header, and OpenAI's cache-routing field.
XAI_CONV_ID_HEADER = "x-grok-conv-id"
OPENAI_PROMPT_CACHE_KEY_PARAM = "prompt_cache_key"

# Hosts (exact or parent domain) whose backends implement the xAI header.
_XAI_DOMAINS = ("x.ai", "grok.com")
# Hosts (exact or parent domain) that accept OpenAI's prompt_cache_key field.
# Deliberately excludes openai.azure.com: Azure OpenAI itself accepts the field
# on GPT deployments, but the same *.openai.azure.com endpoint also fronts
# non-OpenAI Foundry models (DeepSeek, Llama, Mistral) that reject it with
# `unrecognized_request_argument` (#3518). The host says nothing about which
# model family the deployment serves, so `auto` stays off there and an Azure
# GPT operator opts in with openai_prompt_cache_key.
_OPENAI_DOMAINS = ("openai.com",)


class CacheAffinityMode(StrEnum):
    """How (and whether) to pin a request to a backend prompt cache."""

    NONE = "none"
    XAI_CONV_ID = "xai_conv_id"
    OPENAI_PROMPT_CACHE_KEY = "openai_prompt_cache_key"
    AUTO = "auto"


def parse_cache_affinity(value: str | None) -> CacheAffinityMode:
    """Validate a configured cache-affinity mode, defaulting to ``none``.

    Raises ``ValueError`` on an unrecognized value so a typo fails loudly at
    provider construction rather than silently disabling the feature — the whole
    point of the setting is that its effect is invisible in the response.
    """
    if not value:
        return CacheAffinityMode.NONE
    try:
        return CacheAffinityMode(value.strip().lower())
    except ValueError as e:
        valid = ", ".join(mode.value for mode in CacheAffinityMode)
        raise ValueError(f"Invalid cache_affinity {value!r}. Must be one of: {valid}.") from e


def _host_matches(hostname: str, domain: str) -> bool:
    """True when ``hostname`` is ``domain`` itself or a subdomain of it.

    Parsed-host suffix matching, never a substring test: a bare
    ``"x.ai" in base_url`` also matches ``vertex.ai`` and
    ``https://x.ai.evil.example``. The in-tree Azure check
    (``".openai.azure.com" in self.base_url``) gets away with a substring only
    because its needle is long and dotted; ``x.ai`` is four characters.
    """
    return hostname == domain or hostname.endswith(f".{domain}")


def resolve_cache_affinity(mode: CacheAffinityMode, provider: str, base_url: str | None) -> CacheAffinityMode:
    """Resolve ``auto`` to a concrete mode from the provider and base-URL host.

    Non-``auto`` modes are returned unchanged. ``auto`` resolves to
    ``xai_conv_id`` for an x.ai / grok.com host, ``openai_prompt_cache_key`` for
    native OpenAI (no base URL) or an openai.com host, and
    ``none`` for everything else — an unknown backend gets no unfamiliar field.

    The xAI check is host-only and deliberately provider-independent: the
    documented setup for an xAI endpoint is ``provider=openai`` plus an x.ai base
    URL, exactly like Azure OpenAI, so keying on the provider name would miss it.
    """
    if mode is not CacheAffinityMode.AUTO:
        return mode

    hostname = (urlparse(base_url).hostname or "") if base_url else ""
    if hostname and any(_host_matches(hostname, domain) for domain in _XAI_DOMAINS):
        return CacheAffinityMode.XAI_CONV_ID
    if provider.lower() == "openai":
        if not hostname:
            return CacheAffinityMode.OPENAI_PROMPT_CACHE_KEY
        if any(_host_matches(hostname, domain) for domain in _OPENAI_DOMAINS):
            return CacheAffinityMode.OPENAI_PROMPT_CACHE_KEY
    return CacheAffinityMode.NONE


def _first_message_fingerprint(messages: Any) -> str | None:
    """Hash the first message into a 32-hex id, or None if the shape is wrong.

    Used only when no trace context is bound (direct provider use, tests). The
    first message is the system prompt, so the id is stable as the message list
    grows through an agent loop — which is the property cache pinning needs —
    while differing across conversations whose first messages differ.

    Shape-checked rather than truthiness-checked: a bare string ``messages``
    would index to its first character and mint an id from garbage. Anything
    unexpected returns None and the request goes out with no affinity hint.
    """
    if not isinstance(messages, list) or not messages or not isinstance(messages[0], dict):
        return None
    try:
        canonical = json.dumps(messages[0], sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        logger.debug("Cache affinity: first message not serializable; sending no hint", exc_info=True)
        return None
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def cache_affinity_id(messages: Any) -> str | None:
    """Return the affinity id for the in-flight call, or None to send nothing.

    Primary source is the operation's ``trace_id`` — one uuid per
    retain/reflect/consolidation run, generated in ``LLMProvider.with_config``
    and bound around every underlying provider call, so every LLM call of one
    run shares it. That is engine identity rather than payload hashing: it stays
    constant across a run even when the first message changes mid-run.

    The value is always 32 lowercase hex characters, including for the trace_id
    path (hashed rather than passed through) so the wire format is uniform and
    carries no uuid semantics.
    """
    from .llm_trace import current_trace_context

    trace_ctx = current_trace_context()
    if trace_ctx is not None and trace_ctx.trace_id:
        return hashlib.sha256(str(trace_ctx.trace_id).encode("utf-8")).hexdigest()[:32]
    return _first_message_fingerprint(messages)


def apply_cache_affinity(request: dict[str, Any], mode: CacheAffinityMode) -> None:
    """Add this request's cache-affinity hint to ``request`` in place.

    ``mode`` must already be resolved (see :func:`resolve_cache_affinity`);
    ``none`` — and an unresolved ``auto`` — add nothing.

    User-wins semantics throughout, matching the file's ``setdefault`` precedent
    in ``_apply_provider_extra_body_defaults``: an ``x-grok-conv-id`` the caller
    already placed in ``extra_headers`` is kept, and a ``prompt_cache_key`` in
    the operator's configured ``extra_body`` (the escape hatch for a backend
    that wants its own value) suppresses ours entirely.

    Never raises: when no id can be derived the request is left byte-identical
    to a pre-affinity one.
    """
    affinity_id = cache_affinity_id(request.get("messages"))
    if affinity_id is None:
        return

    if mode is CacheAffinityMode.XAI_CONV_ID:
        extra_headers = request.setdefault("extra_headers", {})
        extra_headers.setdefault(XAI_CONV_ID_HEADER, affinity_id)
    elif mode is CacheAffinityMode.OPENAI_PROMPT_CACHE_KEY:
        # prompt_cache_key is a first-class named parameter on
        # chat.completions.create() in the resolved openai SDK, so it goes at the
        # top level rather than through extra_body. An operator value in
        # extra_body would still reach the same wire field, so honour it and
        # send nothing rather than sending both.
        extra_body = request.get("extra_body")
        if isinstance(extra_body, dict) and OPENAI_PROMPT_CACHE_KEY_PARAM in extra_body:
            return
        request.setdefault(OPENAI_PROMPT_CACHE_KEY_PARAM, affinity_id)
