"""Gemini context-cache manager.

Wraps the ``google-genai`` SDK's CachedContent API to let callers reuse a
stable system_instruction + response_schema prefix across many requests.

Cached input tokens are billed at ~10× lower than fresh input tokens
(check the current Gemini pricing for the exact ratio per model), so for
workloads that repeatedly send a large fixed prefix with a small variable
user message — fact extraction, structured tagging, classification — the
input-cost savings are substantial.

This module owns only the create/refresh/lookup lifecycle. It is up to
the caller to (a) decide that the prefix is stable enough to cache, and
(b) pass the returned cache name to ``GeminiLLM.call()``. When the
returned name is ``None`` (because Gemini rejected the create — most
commonly because the prefix is smaller than the model's minimum), the
caller MUST fall back to a non-cached call.

Cardinality
-----------
The intended cache count per process is small (≲100 entries). Each
entry corresponds to one combination of (model, system_instruction,
response_schema). If a caller sees the cache grow unboundedly it
indicates the system_instruction contains per-request data that should
move into the user message instead.

TTL
---
Gemini's CachedContent has a TTL bounded by the model (currently 1h
for most generally-available models). This manager refreshes proactively
at ``ttl_safety_margin`` before expiry. If a cached entry has expired
between refreshes the next call will recreate it transparently.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# Default TTL: 55 minutes. Gemini's hard max for CachedContent is 1 hour
# for most models; we refresh 5 minutes early so a request landing right
# at the boundary doesn't race against expiry.
_DEFAULT_TTL_SECONDS = 55 * 60
_DEFAULT_REFRESH_MARGIN_SECONDS = 5 * 60
# Cap on the cache-create network call. It runs while holding the manager lock, so
# a hung create would block every concurrent caller (e.g. all chunks of a 10-chunk
# retain batch waiting on the cold-start create). On timeout the create soft-fails
# to None and callers proceed uncached, rather than stalling the whole batch.
_DEFAULT_CREATE_TIMEOUT_SECONDS = 30.0

# TTL for the per-step reflect caches created by ``create_incremental``. These
# live only for the duration of one reflect (seconds), so the TTL is just a
# storage backstop in case the explicit ``delete_session`` at reflect end is
# missed (crash / event-loop teardown). Short so orphaned caches age out fast —
# storage is billed per token-hour, so a 5-minute cap keeps the cost of a leaked
# cache negligible.
_DEFAULT_INCREMENTAL_TTL_SECONDS = 5 * 60


@dataclass
class _CacheEntry:
    name: str  # The CachedContent resource name returned by Gemini.
    created_at: float
    ttl_seconds: int


class GeminiCacheManager:
    """Per-process map of (prefix fingerprint) → CachedContent name.

    Thread-safe across asyncio tasks via a single ``asyncio.Lock``. The
    create/refresh calls are serialised; this is fine because cache
    creation is a one-shot warm-up per fingerprint (subsequent reads are
    pure dict lookups outside the lock).

    Not shared across pods — each worker / api replica builds its own
    cache. The cost of cold-starting one extra full-price call per pod
    per fingerprint per hour is negligible compared to the steady-state
    savings.
    """

    def __init__(
        self,
        client: Any,
        *,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        refresh_margin_seconds: int = _DEFAULT_REFRESH_MARGIN_SECONDS,
        create_timeout_seconds: float = _DEFAULT_CREATE_TIMEOUT_SECONDS,
    ) -> None:
        self._client = client
        self._ttl_seconds = ttl_seconds
        self._refresh_margin_seconds = refresh_margin_seconds
        self._create_timeout_seconds = create_timeout_seconds
        self._entries: dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()
        # session_id -> CachedContent names created via ``create_incremental``.
        # A reflect creates a fresh rolling cache per step under one session id;
        # ``delete_session`` tears them all down when the reflect finishes.
        self._sessions: dict[str, list[str]] = {}

    @staticmethod
    def fingerprint(
        model: str,
        system_instruction: str,
        response_schema: Any | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Stable hash of the cacheable surface.

        ``response_schema`` may be a Pydantic class, a dict, or ``None``.
        Pydantic schemas are normalised by serialising via
        ``model_json_schema()`` and stripping the auto-generated
        ``"title"`` fields so two dynamically-built models with the same
        shape but different class names hash identically. This matters
        for callers (e.g. fact extraction) that rebuild the schema
        class on every request via a builder helper — without the
        normalisation the cache would never hit.

        ``tools`` is the OpenAI-style tools list (each entry has a
        ``"function"`` dict with name/description/parameters). When
        supplied, the tool definitions become part of the cache key so a
        loop that adds or renames a tool gets a fresh cache and doesn't
        silently use a stale schema. Tools are serialised with
        ``sort_keys=True`` to neutralise dict-ordering drift.
        """
        hasher = hashlib.sha256()
        hasher.update(model.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(system_instruction.encode("utf-8"))
        hasher.update(b"\x00")
        if response_schema is None:
            hasher.update(b"none")
        elif hasattr(response_schema, "model_json_schema"):
            try:
                schema = response_schema.model_json_schema()
                _strip_titles(schema)
                hasher.update(json.dumps(schema, sort_keys=True).encode("utf-8"))
            except Exception:
                # Fall back to class identity if the schema can't be serialised.
                hasher.update(repr(response_schema).encode("utf-8"))
        else:
            try:
                hasher.update(json.dumps(response_schema, sort_keys=True).encode("utf-8"))
            except (TypeError, ValueError):
                hasher.update(repr(response_schema).encode("utf-8"))
        hasher.update(b"\x00")
        if tools:
            try:
                hasher.update(json.dumps(tools, sort_keys=True).encode("utf-8"))
            except (TypeError, ValueError):
                hasher.update(repr(tools).encode("utf-8"))
        else:
            hasher.update(b"no-tools")
        return hasher.hexdigest()

    async def get_or_create(
        self,
        *,
        model: str,
        system_instruction: str,
        response_schema: Any | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str | None:
        """Return a CachedContent resource name for the given prefix, or
        ``None`` if Gemini rejects the create (prefix too small, model
        does not support caching, etc.).

        ``tools`` is the OpenAI-style tools list. When supplied, the tool
        definitions are baked into the CachedContent so the caller's
        ``call_with_tools`` doesn't need to resend them on every
        iteration. Pass ``None`` for non-tool calls.

        ``None`` return is a normal, expected value — the caller falls
        back to an uncached call and the system continues to work.
        """
        key = self.fingerprint(model, system_instruction, response_schema, tools)

        async with self._lock:
            entry = self._entries.get(key)
            if entry is not None and self._is_fresh(entry):
                return entry.name

            # Need to (re)create. Pop the stale entry first so a failed
            # create doesn't leave a name we'd return on the next call.
            self._entries.pop(key, None)

            try:
                cache_name = await self._create_cache(
                    model=model,
                    system_instruction=system_instruction,
                    tools=tools,
                )
            except _CacheNotEligible as e:
                logger.debug(
                    "GeminiCacheManager: prefix not eligible for caching (model=%s, reason=%s) — caller will fall back",
                    model,
                    e,
                )
                return None
            except Exception:
                logger.exception(
                    "GeminiCacheManager: failed to create cached content "
                    "(model=%s); caller will fall back to uncached call",
                    model,
                )
                return None

            if cache_name is None:
                return None

            self._entries[key] = _CacheEntry(
                name=cache_name,
                created_at=time.monotonic(),
                ttl_seconds=self._ttl_seconds,
            )
            return cache_name

    def _is_fresh(self, entry: _CacheEntry) -> bool:
        """An entry is fresh if it's young enough that the next request
        won't race against the TTL expiry."""
        age = time.monotonic() - entry.created_at
        return age < (entry.ttl_seconds - self._refresh_margin_seconds)

    def invalidate(self, name: str) -> None:
        """Forget a cache name that the server rejected (expired/deleted/invalid).

        Called by the provider when a generate request using this CachedContent
        fails, so the next ``get_or_create`` recreates it instead of handing back
        the dead name again. Best-effort and sync — drops the matching entry from
        the in-process map; the orphaned server-side cache (if any) ages out on
        its own TTL.
        """
        for key, entry in list(self._entries.items()):
            if entry.name == name:
                self._entries.pop(key, None)

    async def create_incremental(
        self,
        *,
        session_id: str,
        model: str,
        system_instruction: str,
        contents: list[Any],
        tools: list[dict[str, Any]] | None = None,
    ) -> str | None:
        """Create a fresh CachedContent holding ``system + tools + contents`` and
        track it under ``session_id`` for later teardown.

        Unlike ``get_or_create``, this does NOT deduplicate by fingerprint: each
        step of a reflect grows the conversation prefix, so every call is a
        distinct, single-use cache. The reflect loop creates one per step (each
        covering the previous step's full input) and reuses it for exactly the
        next model turn, then supersedes it. All caches for the session are
        deleted by ``delete_session`` when the reflect ends; the short TTL is
        only a backstop.

        Returns the cache resource name, or ``None`` when caching is disabled,
        the prefix is below the model minimum, or the create otherwise fails —
        callers MUST fall back to an uncached call in that case.
        """
        try:
            name = await self._create_cache(
                model=model,
                system_instruction=system_instruction,
                tools=tools,
                contents=contents,
                ttl_seconds=_DEFAULT_INCREMENTAL_TTL_SECONDS,
            )
        except _CacheNotEligible as e:
            logger.debug(
                "GeminiCacheManager: incremental prefix not eligible (model=%s, reason=%s) — caller falls back",
                model,
                e,
            )
            return None
        except Exception:
            logger.exception(
                "GeminiCacheManager: failed to create incremental cache (model=%s); caller falls back",
                model,
            )
            return None
        if name is not None:
            self._sessions.setdefault(session_id, []).append(name)
        return name

    async def delete(self, name: str) -> None:
        """Best-effort server-side delete of a single CachedContent.

        Swallows all errors: a failed delete just means the cache ages out on
        its TTL. Also drops any matching in-process entry.
        """
        self.invalidate(name)
        try:
            await self._client.aio.caches.delete(name=name)
        except Exception:
            logger.debug("GeminiCacheManager: delete of cache %s failed (will age out on TTL)", name, exc_info=True)

    async def delete_session(self, session_id: str) -> None:
        """Delete every CachedContent created for ``session_id`` (reflect teardown).

        Deletes concurrently and best-effort — a reflect must never fail because
        a cache couldn't be torn down; the short TTL is the backstop.
        """
        names = self._sessions.pop(session_id, [])
        if not names:
            return
        await asyncio.gather(*(self.delete(n) for n in names), return_exceptions=True)

    async def _create_cache(
        self,
        *,
        model: str,
        system_instruction: str,
        tools: list[dict[str, Any]] | None = None,
        contents: list[Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> str | None:
        """Wrap ``client.aio.caches.create`` with the config we want.

        The SDK surface differs slightly across google-genai versions;
        this implementation targets the >=1.0.0 line where caches live
        under ``client.aio.caches``.

        ``contents`` (already-converted ``genai_types.Content`` turns) is
        appended after the system_instruction/tools so the cache can hold a
        growing multi-turn conversation prefix, not just the static prefix —
        this is what the step-by-step reflect cache relies on. ``ttl_seconds``
        overrides the manager default (used to give per-step reflect caches a
        short backstop TTL).
        """
        # Lazy import so this module doesn't require the SDK at import time.
        from google.genai import types as genai_types

        # A CachedContent only holds reusable *input* — system_instruction,
        # contents, tools, ttl. ``response_schema``/``response_mime_type`` are
        # generation-time output constraints and the SDK rejects them here
        # (``CreateCachedContentConfig`` forbids those fields). They are applied
        # per-request on the GenerateContentConfig instead — see the call sites,
        # which set them alongside ``cached_content``. ``response_schema`` is
        # still part of the fingerprint so a schema change keys a fresh cache.
        config_kwargs: dict[str, Any] = {
            "system_instruction": system_instruction,
            "ttl": f"{ttl_seconds if ttl_seconds is not None else self._ttl_seconds}s",
        }
        if contents:
            config_kwargs["contents"] = contents
        if tools:
            # OpenAI-style {"function": {...}} entries must be converted to
            # Gemini's Tool/FunctionDeclaration shape before caching.
            gemini_tools = []
            for tool in tools:
                func = tool.get("function", {})
                gemini_tools.append(
                    genai_types.Tool(
                        function_declarations=[
                            genai_types.FunctionDeclaration(
                                name=func.get("name", ""),
                                description=func.get("description", ""),
                                parameters=func.get("parameters"),
                            )
                        ]
                    )
                )
            config_kwargs["tools"] = gemini_tools

        try:
            cached = await asyncio.wait_for(
                self._client.aio.caches.create(
                    model=model,
                    config=genai_types.CreateCachedContentConfig(**config_kwargs),
                ),
                timeout=self._create_timeout_seconds,
            )
        except Exception as e:
            # Gemini returns a 400 with a "minimum token count" message
            # when the prefix is too small. We treat this as a soft
            # "not eligible" signal rather than a real error so callers
            # silently fall back to non-cached.
            msg = str(e).lower()
            if "minimum" in msg or "too small" in msg or "too short" in msg:
                raise _CacheNotEligible(str(e)) from e
            raise

        return getattr(cached, "name", None)


class _CacheNotEligible(Exception):
    """Raised when Gemini rejects the cache create because the prefix
    is below the model's minimum cacheable size. Treated as a soft
    fallback by the caller, not an error."""


def _strip_titles(node: Any) -> None:
    """Recursively remove auto-generated ``"title"`` keys from a JSON
    Schema-like dict tree, in place. Pydantic seeds these from the
    Python class name, which means structurally-identical schemas built
    from differently-named classes look distinct to a naive hash."""
    if isinstance(node, dict):
        node.pop("title", None)
        for v in node.values():
            _strip_titles(v)
    elif isinstance(node, list):
        for item in node:
            _strip_titles(item)
