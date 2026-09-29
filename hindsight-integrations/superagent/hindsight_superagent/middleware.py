"""Safety middleware wrapping Hindsight memory operations with Superagent Guard and Redact."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from hindsight_client import Hindsight
from hindsight_client_api.models.recall_response import RecallResponse
from hindsight_client_api.models.reflect_response import ReflectResponse
from safety_agent import SafetyClient

from ._client import build_safety_client, resolve_hindsight_client, snapshot_safety_config
from .config import Budget, TagsMatch, get_config
from .errors import GuardBlockedError, HindsightError

logger = logging.getLogger(__name__)

# Per-item keys passed straight through from a retain_batch caller's dict to
# Hindsight.aretain_batch.  `content` is mandatory (handled separately).  Tags
# are merged with default tags, so they're handled separately too.
_BATCH_PASSTHROUGH_KEYS = (
    "timestamp",
    "context",
    "metadata",
    "document_id",
    "entities",
    "observation_scopes",
    "strategy",
    "update_mode",
)

GuardCallback = Callable[[str, Any], Awaitable[None] | None]
"""Optional callable: `on_guard(scope, result)`.

`scope` is one of "retain", "recall", "reflect", "retain_batch".  `result` is
the Superagent guard response (with `.classification`, `.reasoning`,
`.violation_types`, `.cwe_codes`).  Called for every guard verdict including
passes — callers use it for observability, near-miss logging, or analytics
without changing the core flow.  May be sync or async.
"""


def _kw(value: Any, config_value: Any, fallback: Any) -> Any:
    """Resolve kwarg precedence: explicit value > global config > fallback default.

    Uses `is not None` so explicit falsy values (empty list, 0, False) override
    a populated global config — previously `or`-chained precedence treated those
    the same as "unset" and silently fell through to global config.
    """
    if value is not None:
        return value
    if config_value is not None:
        return config_value
    return fallback


class SafeHindsight:
    """Hindsight client wrapper that applies Superagent safety checks to memory operations.

    Wraps retain, recall, and reflect with configurable Guard (prompt injection detection)
    and Redact (PII removal) middleware.

    Usage::

        from hindsight_superagent import SafeHindsight

        safe = SafeHindsight(
            bank_id="user-123",
            hindsight_api_url="http://localhost:8888",
        )

        # Content is guarded and redacted before storage
        await safe.retain("John's email is john@acme.com and he prefers dark mode")

        # Queries are guarded before recall
        results = await safe.recall("What are John's preferences?")

        # Always release pooled connections when done
        await safe.aclose()
    """

    def __init__(
        self,
        *,
        bank_id: str,
        hindsight_client: Hindsight | None = None,
        safety_client: SafetyClient | None = None,
        hindsight_api_url: str | None = None,
        api_key: str | None = None,
        superagent_api_key: str | None = None,
        budget: Budget | None = None,
        max_tokens: int | None = None,
        tags: list[str] | None = None,
        recall_tags: list[str] | None = None,
        recall_tags_match: TagsMatch | None = None,
        guard_model: str | None = None,
        redact_model: str | None = None,
        redact_entities: list[str] | None = None,
        redact_rewrite: bool | None = None,
        safety_concurrency: int | None = None,
        enable_guard_on_retain: bool | None = None,
        enable_guard_on_recall: bool | None = None,
        enable_guard_on_reflect: bool | None = None,
        enable_redact_on_retain: bool | None = None,
        enable_redact_on_recall: bool | None = None,
        enable_redact_on_reflect: bool | None = None,
        enable_fallback: bool | None = None,
        fallback_timeout: float | None = None,
        on_guard: GuardCallback | None = None,
    ) -> None:
        self._bank_id = bank_id
        self._hindsight = resolve_hindsight_client(hindsight_client, hindsight_api_url, api_key)
        self._on_guard = on_guard

        # Snapshot the safety client config eagerly so a later configure()
        # call can't silently change what the lazy-constructed client looks
        # like.  The actual create_client() is deferred to first guard/redact.
        # Explicit `safety_client` instance still wins and skips lazy build.
        self._safety: SafetyClient | None = safety_client
        self._safety_snapshot: dict[str, Any] | None = (
            None
            if safety_client is not None
            else snapshot_safety_config(superagent_api_key, enable_fallback, fallback_timeout)
        )

        # Also track whether we own the underlying Hindsight client.  If the
        # caller passed in their own, we don't aclose() it on our way out.
        self._owns_hindsight = hindsight_client is None
        self._owns_safety = safety_client is None
        self._closed = False

        config = get_config()
        self._budget = _kw(budget, config.budget if config else None, "mid")
        self._max_tokens = _kw(max_tokens, config.max_tokens if config else None, 4096)
        self._tags = _kw(tags, config.tags if config else None, None)
        self._recall_tags = _kw(recall_tags, config.recall_tags if config else None, None)
        self._recall_tags_match = _kw(recall_tags_match, config.recall_tags_match if config else None, "any")
        self._guard_model = _kw(guard_model, config.guard_model if config else None, None)
        self._redact_model = _kw(redact_model, config.redact_model if config else None, None)
        self._redact_entities = _kw(redact_entities, config.redact_entities if config else None, None)
        self._redact_rewrite = _kw(redact_rewrite, config.redact_rewrite if config else None, False)
        self._safety_concurrency = _kw(
            safety_concurrency,
            getattr(config, "safety_concurrency", None) if config else None,
            5,
        )
        if not isinstance(self._safety_concurrency, int) or self._safety_concurrency < 1:
            # asyncio.Semaphore(0) never admits work, so 0 (or anything <1)
            # would deadlock _redact_many and the guard-batching path in
            # retain_batch.  Fail fast at construction.
            raise ValueError(f"safety_concurrency must be a positive int, got {self._safety_concurrency!r}")
        self._enable_guard_on_retain = _kw(
            enable_guard_on_retain, config.enable_guard_on_retain if config else None, True
        )
        self._enable_guard_on_recall = _kw(
            enable_guard_on_recall, config.enable_guard_on_recall if config else None, True
        )
        self._enable_guard_on_reflect = _kw(
            enable_guard_on_reflect, config.enable_guard_on_reflect if config else None, True
        )
        self._enable_redact_on_retain = _kw(
            enable_redact_on_retain, config.enable_redact_on_retain if config else None, True
        )
        self._enable_redact_on_recall = _kw(
            enable_redact_on_recall, config.enable_redact_on_recall if config else None, False
        )
        self._enable_redact_on_reflect = _kw(
            enable_redact_on_reflect,
            getattr(config, "enable_redact_on_reflect", None) if config else None,
            False,
        )

    def _get_safety(self) -> SafetyClient:
        """Return the SafetyClient, building it from the snapshot on first access."""
        if self._safety is None:
            assert self._safety_snapshot is not None
            self._safety = build_safety_client(self._safety_snapshot)
        return self._safety

    async def _guard(self, text: str, *, scope: str) -> None:
        """Run Superagent Guard on text. Raises GuardBlockedError if blocked.

        Invokes the optional `on_guard(scope, result)` callback regardless of
        the verdict so callers can observe non-block decisions (allows,
        flags, near-misses) for logging or analytics without changing the
        core control flow.  Exceptions raised by the callback are caught and
        logged at WARNING — observability failures must never take down the
        memory operation being observed.
        """
        guard_kwargs: dict[str, Any] = {"input": text}
        if self._guard_model:
            guard_kwargs["model"] = self._guard_model
        result = await self._get_safety().guard(**guard_kwargs)

        if self._on_guard is not None:
            try:
                cb_result = self._on_guard(scope, result)
                if asyncio.iscoroutine(cb_result):
                    await cb_result
            except Exception as cb_exc:  # noqa: BLE001 — observability must not crash the op
                logger.warning(
                    "on_guard callback raised %s for scope=%r: %s",
                    type(cb_exc).__name__,
                    scope,
                    cb_exc,
                )

        if result.classification == "block":
            raise GuardBlockedError(
                reasoning=result.reasoning,
                violation_types=result.violation_types,
                cwe_codes=result.cwe_codes,
            )

    async def _redact(self, text: str) -> str:
        """Run Superagent Redact on text. Returns redacted text."""
        if not self._redact_model:
            raise HindsightError("Redact requires a model. Set redact_model in SafeHindsight() or configure().")
        redact_kwargs: dict[str, Any] = {
            "input": text,
            "model": self._redact_model,
            "rewrite": self._redact_rewrite,
        }
        if self._redact_entities:
            redact_kwargs["entities"] = self._redact_entities
        result = await self._get_safety().redact(**redact_kwargs)
        if result.findings:
            logger.info("Redacted %d PII entities", len(result.findings))
        return result.redacted

    async def _redact_many(self, texts: list[str]) -> list[str]:
        """Redact a batch of texts under a per-instance concurrency cap.

        Bounded concurrency keeps wide recall/reflect batches from stampeding
        the Superagent rate limit.  Order is preserved.  If any single redact
        fails, the gather raises and the whole operation fails — same
        contract as before, just under a semaphore.
        """
        if not texts:
            return []
        semaphore = asyncio.Semaphore(self._safety_concurrency)

        async def _one(t: str) -> str:
            async with semaphore:
                return await self._redact(t)

        return list(await asyncio.gather(*(_one(t) for t in texts)))

    def _merge_tags(self, call_tags: list[str] | None) -> list[str]:
        """Merge per-call tags with default tags, preserving order and deduping."""
        return list(dict.fromkeys((call_tags or []) + (self._tags or [])))

    async def retain(
        self,
        content: str,
        *,
        context: str | None = None,
        tags: list[str] | None = None,
        timestamp: str | None = None,
    ) -> str:
        """Store information to memory after applying safety checks.

        If guard is enabled, the content is checked for prompt injection.
        If redact is enabled, PII is removed before storage.

        Args:
            content: Text content to store.
            context: Optional context for the memory.
            tags: Optional tags (merged with default tags).
            timestamp: Optional ISO timestamp.

        Returns:
            Status message.

        Raises:
            GuardBlockedError: If content is blocked by Guard.
            HindsightError: If the operation fails.
        """
        try:
            if self._enable_guard_on_retain:
                await self._guard(content, scope="retain")

            safe_content = content
            if self._enable_redact_on_retain:
                safe_content = await self._redact(content)

            retain_kwargs: dict[str, Any] = {
                "bank_id": self._bank_id,
                "content": safe_content,
            }
            effective_tags = self._merge_tags(tags)
            if effective_tags:
                retain_kwargs["tags"] = effective_tags
            if context:
                retain_kwargs["context"] = context
            if timestamp:
                retain_kwargs["timestamp"] = timestamp

            await self._hindsight.aretain(**retain_kwargs)
            return "Memory stored successfully."
        except (GuardBlockedError, HindsightError):
            raise
        except Exception as e:
            logger.error("Retain failed: %s", e)
            raise HindsightError(f"Retain failed: {e}") from e

    async def retain_batch(
        self,
        items: list[dict[str, Any]],
        *,
        document_id: str | None = None,
        document_tags: list[str] | None = None,
        retain_async: bool = False,
    ) -> str:
        """Store a batch of memories, applying safety checks to each item.

        Each item is a dict matching the shape accepted by
        `Hindsight.aretain_batch` — `content` (required) plus any of
        `timestamp`, `context`, `metadata`, `document_id`, `entities`,
        `observation_scopes`, `strategy`, `update_mode`, `tags`.  Tags merge
        with the instance's default tags (preserving order, deduped); every
        other field is passed through verbatim.

        Guard and Redact run per-item under the configured concurrency cap
        (`safety_concurrency`); if any item's guard blocks, `GuardBlockedError`
        propagates and the whole batch aborts before any item is stored —
        matching the per-call semantics of `retain`.

        Args:
            items: List of memory items.  Each dict needs `content`; other
                keys are optional and forwarded to `aretain_batch`.
            document_id: Optional batch-level document ID applied to items
                that don't have their own.
            document_tags: Optional list of tags applied to all items in
                this batch (merged with per-item tags by Hindsight).
            retain_async: If True, Hindsight processes the batch
                asynchronously in the background after the safety pipeline
                has completed.  Guard + Redact still run synchronously
                before the call returns — this only defers the underlying
                store.  Defaults to False to match Hindsight's default.

        Returns:
            Status message.

        Raises:
            GuardBlockedError: If any item's content is blocked.
            HindsightError: If the operation fails.
        """
        if not items:
            return "Memory stored successfully."
        try:
            contents = [item["content"] for item in items]

            if self._enable_guard_on_retain:
                semaphore = asyncio.Semaphore(self._safety_concurrency)

                async def _one_guard(t: str) -> None:
                    async with semaphore:
                        await self._guard(t, scope="retain_batch")

                await asyncio.gather(*(_one_guard(c) for c in contents))

            if self._enable_redact_on_retain:
                redacted_contents = await self._redact_many(contents)
            else:
                redacted_contents = contents

            batch_items: list[dict[str, Any]] = []
            for src, safe_content in zip(items, redacted_contents):
                entry: dict[str, Any] = {"content": safe_content}
                effective_tags = self._merge_tags(src.get("tags"))
                if effective_tags:
                    entry["tags"] = effective_tags
                # Pass through every other supported field verbatim.
                for key in _BATCH_PASSTHROUGH_KEYS:
                    if src.get(key) is not None:
                        entry[key] = src[key]
                batch_items.append(entry)

            batch_kwargs: dict[str, Any] = {"bank_id": self._bank_id, "items": batch_items}
            if document_id is not None:
                batch_kwargs["document_id"] = document_id
            if document_tags is not None:
                batch_kwargs["document_tags"] = document_tags
            if retain_async:
                batch_kwargs["retain_async"] = True

            await self._hindsight.aretain_batch(**batch_kwargs)
            return "Memory stored successfully."
        except (GuardBlockedError, HindsightError):
            raise
        except Exception as e:
            logger.error("Retain batch failed: %s", e)
            raise HindsightError(f"Retain batch failed: {e}") from e

    async def recall(
        self,
        query: str,
        *,
        budget: Budget | None = None,
        max_tokens: int | None = None,
        tags: list[str] | None = None,
        tags_match: TagsMatch | None = None,
    ) -> RecallResponse:
        """Search memory after guarding the query, optionally redacting results.

        Args:
            query: Search query.
            budget: Recall budget override.
            max_tokens: Max tokens override.
            tags: Tags to filter results.
            tags_match: Tag matching mode.

        Returns:
            Recall response with results.  When `enable_redact_on_recall` is
            set (off by default — every result triggers its own redact call,
            bounded by `safety_concurrency`), each result's text field is
            passed through Redact before being returned, so PII stored from
            earlier sessions or other sources doesn't leak back to the caller.

        Raises:
            GuardBlockedError: If query is blocked by Guard.
            HindsightError: If the operation fails.
        """
        try:
            if self._enable_guard_on_recall:
                await self._guard(query, scope="recall")

            recall_kwargs: dict[str, Any] = {
                "bank_id": self._bank_id,
                "query": query,
                "budget": budget if budget is not None else self._budget,
                "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            }
            effective_tags = tags if tags is not None else self._recall_tags
            if effective_tags:
                recall_kwargs["tags"] = effective_tags
                recall_kwargs["tags_match"] = tags_match if tags_match is not None else self._recall_tags_match

            response = await self._hindsight.arecall(**recall_kwargs)

            if self._enable_redact_on_recall and response.results:
                redacted_texts = await self._redact_many([r.text for r in response.results])
                for result, redacted_text in zip(response.results, redacted_texts):
                    result.text = redacted_text

            return response
        except (GuardBlockedError, HindsightError):
            raise
        except Exception as e:
            logger.error("Recall failed: %s", e)
            raise HindsightError(f"Recall failed: {e}") from e

    async def reflect(
        self,
        query: str,
        *,
        budget: Budget | None = None,
        max_tokens: int | None = None,
    ) -> ReflectResponse:
        """Synthesize an answer from memory after guarding the query.

        Args:
            query: Reflection query.
            budget: Reflect budget override.
            max_tokens: Max tokens override.

        Returns:
            Reflect response.  When `enable_redact_on_reflect` is set (off by
            default — reflect's synthesised text can be derived from PII-laden
            memories), the response text is passed through Redact before
            being returned.

        Raises:
            GuardBlockedError: If query is blocked by Guard.
            HindsightError: If the operation fails.
        """
        try:
            if self._enable_guard_on_reflect:
                await self._guard(query, scope="reflect")

            reflect_kwargs: dict[str, Any] = {
                "bank_id": self._bank_id,
                "query": query,
                "budget": budget if budget is not None else self._budget,
                "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            }

            response = await self._hindsight.areflect(**reflect_kwargs)

            if self._enable_redact_on_reflect and response.text:
                response.text = await self._redact(response.text)

            return response
        except (GuardBlockedError, HindsightError):
            raise
        except Exception as e:
            logger.error("Reflect failed: %s", e)
            raise HindsightError(f"Reflect failed: {e}") from e

    async def aclose(self) -> None:
        """Release the underlying Hindsight (and SafetyClient if owned) connection pools.

        Safe to call multiple times.  Clients passed in by the caller via
        `hindsight_client=` or `safety_client=` are NOT closed — the caller
        retains ownership of those.
        """
        if self._closed:
            return
        self._closed = True

        if self._owns_hindsight and self._hindsight is not None:
            aclose = getattr(self._hindsight, "aclose", None)
            if aclose is not None:
                await aclose()
            else:
                close = getattr(self._hindsight, "close", None)
                if close is not None:
                    close()

        if self._owns_safety and self._safety is not None:
            aclose = getattr(self._safety, "aclose", None)
            if aclose is not None:
                await aclose()
            else:
                close = getattr(self._safety, "close", None)
                if close is not None:
                    close()

    async def __aenter__(self) -> SafeHindsight:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()
