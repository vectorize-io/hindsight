"""Safety middleware wrapping Hindsight memory operations with Superagent Guard and Redact."""

from __future__ import annotations

import logging
from typing import Any

from hindsight_client import Hindsight
from hindsight_client_api.models.recall_response import RecallResponse
from hindsight_client_api.models.reflect_response import ReflectResponse
from safety_agent import SafetyClient

from ._client import resolve_hindsight_client, resolve_safety_client
from .config import Budget, TagsMatch, get_config
from .errors import GuardBlockedError, HindsightError

logger = logging.getLogger(__name__)


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
        enable_guard_on_retain: bool | None = None,
        enable_guard_on_recall: bool | None = None,
        enable_guard_on_reflect: bool | None = None,
        enable_redact_on_retain: bool | None = None,
        enable_fallback: bool | None = None,
        fallback_timeout: float | None = None,
    ) -> None:
        self._bank_id = bank_id
        self._hindsight = resolve_hindsight_client(hindsight_client, hindsight_api_url, api_key)
        self._safety = resolve_safety_client(safety_client, superagent_api_key, enable_fallback, fallback_timeout)

        config = get_config()
        self._budget = budget or (config.budget if config else "mid")
        self._max_tokens = max_tokens or (config.max_tokens if config else 4096)
        self._tags = tags if tags is not None else (config.tags if config else None)
        self._recall_tags = recall_tags if recall_tags is not None else (config.recall_tags if config else None)
        self._recall_tags_match = recall_tags_match or (config.recall_tags_match if config else "any")
        self._guard_model = guard_model or (config.guard_model if config else None)
        self._redact_model = redact_model or (config.redact_model if config else None)
        self._redact_entities = redact_entities or (config.redact_entities if config else None)
        self._redact_rewrite = (
            redact_rewrite if redact_rewrite is not None else (config.redact_rewrite if config else False)
        )
        self._enable_guard_on_retain = (
            enable_guard_on_retain
            if enable_guard_on_retain is not None
            else (config.enable_guard_on_retain if config else True)
        )
        self._enable_guard_on_recall = (
            enable_guard_on_recall
            if enable_guard_on_recall is not None
            else (config.enable_guard_on_recall if config else True)
        )
        self._enable_guard_on_reflect = (
            enable_guard_on_reflect
            if enable_guard_on_reflect is not None
            else (config.enable_guard_on_reflect if config else True)
        )
        self._enable_redact_on_retain = (
            enable_redact_on_retain
            if enable_redact_on_retain is not None
            else (config.enable_redact_on_retain if config else True)
        )

    async def _guard(self, text: str) -> None:
        """Run Superagent Guard on text. Raises GuardBlockedError if blocked."""
        guard_kwargs: dict[str, Any] = {"input": text}
        if self._guard_model:
            guard_kwargs["model"] = self._guard_model
        result = await self._safety.guard(**guard_kwargs)
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
        result = await self._safety.redact(**redact_kwargs)
        if result.findings:
            logger.info("Redacted %d PII entities before retain", len(result.findings))
        return result.redacted

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
                await self._guard(content)

            safe_content = content
            if self._enable_redact_on_retain:
                safe_content = await self._redact(content)

            retain_kwargs: dict[str, Any] = {
                "bank_id": self._bank_id,
                "content": safe_content,
            }
            effective_tags = list(set((tags or []) + (self._tags or [])))
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

    async def recall(
        self,
        query: str,
        *,
        budget: Budget | None = None,
        max_tokens: int | None = None,
        tags: list[str] | None = None,
        tags_match: TagsMatch | None = None,
    ) -> RecallResponse:
        """Search memory after guarding the query.

        Args:
            query: Search query.
            budget: Recall budget override.
            max_tokens: Max tokens override.
            tags: Tags to filter results.
            tags_match: Tag matching mode.

        Returns:
            Recall response with results.

        Raises:
            GuardBlockedError: If query is blocked by Guard.
            HindsightError: If the operation fails.
        """
        try:
            if self._enable_guard_on_recall:
                await self._guard(query)

            recall_kwargs: dict[str, Any] = {
                "bank_id": self._bank_id,
                "query": query,
                "budget": budget or self._budget,
                "max_tokens": max_tokens or self._max_tokens,
            }
            effective_tags = tags or self._recall_tags
            if effective_tags:
                recall_kwargs["tags"] = effective_tags
                recall_kwargs["tags_match"] = tags_match or self._recall_tags_match

            return await self._hindsight.arecall(**recall_kwargs)
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
            Reflect response.

        Raises:
            GuardBlockedError: If query is blocked by Guard.
            HindsightError: If the operation fails.
        """
        try:
            if self._enable_guard_on_reflect:
                await self._guard(query)

            reflect_kwargs: dict[str, Any] = {
                "bank_id": self._bank_id,
                "query": query,
                "budget": budget or self._budget,
                "max_tokens": max_tokens or self._max_tokens,
            }

            return await self._hindsight.areflect(**reflect_kwargs)
        except (GuardBlockedError, HindsightError):
            raise
        except Exception as e:
            logger.error("Reflect failed: %s", e)
            raise HindsightError(f"Reflect failed: {e}") from e
