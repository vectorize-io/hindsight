"""CrewAI Storage backend powered by Hindsight.

Implements CrewAI's Storage interface (save/search/reset) using
Hindsight's retain/recall APIs for persistent agent memory.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from crewai.memory.storage.interface import Storage

from ._compat import call_sync
from .config import get_config
from .errors import HindsightError

logger = logging.getLogger(__name__)


class HindsightStorage(Storage):
    """CrewAI Storage backend that persists memories to Hindsight.

    Maps CrewAI's storage interface to Hindsight's memory API:
    - save(value, metadata, agent)  -> client.retain(bank_id, content)
    - search(query, limit)          -> client.recall(bank_id, query)
    - reset()                       -> client.delete_bank() + recreate

    Args:
        bank_id: The Hindsight memory bank ID for this crew.
        hindsight_api_url: Override the configured API URL.
        api_key: Override the configured API key.
        budget: Recall budget level (low/mid/high). Overrides config.
        max_tokens: Max recall tokens. Overrides config.
        tags: Tags for retain operations. Overrides config.
        recall_tags: Tags to filter recall. Overrides config.
        recall_tags_match: Tag matching mode. Overrides config.
        per_agent_banks: If True, each agent gets its own bank
            (bank_id is suffixed with sanitized agent role). Default False.
        bank_resolver: Custom callable (bank_id, agent) -> resolved_bank_id.
            Overrides per_agent_banks if provided.
        mission: If provided, creates/updates the bank with this mission.
        verbose: Enable verbose logging. Overrides config.

    Example::

        from hindsight_crewai import configure, HindsightStorage
        from crewai.memory.external.external_memory import ExternalMemory

        configure(hindsight_api_url="http://localhost:8888")
        storage = HindsightStorage(bank_id="my-crew")
        external_memory = ExternalMemory(storage=storage)
    """

    def __init__(
        self,
        bank_id: str,
        hindsight_api_url: str | None = None,
        api_key: str | None = None,
        budget: str | None = None,
        max_tokens: int | None = None,
        tags: list[str] | None = None,
        recall_tags: list[str] | None = None,
        recall_tags_match: str | None = None,
        per_agent_banks: bool = False,
        bank_resolver: Callable[[str, str | None], str] | None = None,
        mission: str | None = None,
        verbose: bool | None = None,
    ):
        self._bank_id = bank_id
        self._per_agent_banks = per_agent_banks
        self._bank_resolver = bank_resolver
        self._mission = mission
        self._local = threading.local()
        self._created_banks: set[str] = set()

        # Resolve settings: constructor args override global config
        config = get_config()
        self._api_url = hindsight_api_url or (
            config.hindsight_api_url if config else "http://localhost:8888"
        )
        self._api_key = api_key or (config.api_key if config else None)
        self._budget = budget or (config.budget if config else "mid")
        self._max_tokens = max_tokens or (config.max_tokens if config else 4096)
        self._tags = tags or (config.tags if config else None)
        self._recall_tags = recall_tags or (config.recall_tags if config else None)
        self._recall_tags_match = recall_tags_match or (
            config.recall_tags_match if config else "any"
        )
        self._verbose = (
            verbose if verbose is not None else (config.verbose if config else False)
        )

        # Eagerly create the default bank if mission is provided
        if mission:
            self._ensure_bank(self._bank_id)

    def _get_client(self) -> Any:
        """Get or create a thread-local Hindsight client.

        Each thread gets its own client so the underlying aiohttp
        session stays bound to that thread's event loop.
        """
        client = getattr(self._local, "client", None)
        if client is None:
            from hindsight_client import Hindsight

            client = Hindsight(
                base_url=self._api_url,
                api_key=self._api_key,
                timeout=30.0,
            )
            self._local.client = client
        return client

    def _resolve_bank_id(self, agent: str | None = None) -> str:
        """Resolve the effective bank_id for this operation.

        If bank_resolver is provided, delegates to it.
        If per_agent_banks=True and agent is provided, uses
        ``f"{bank_id}-{sanitized_agent}"``.
        Otherwise returns the base bank_id.
        """
        if self._bank_resolver:
            return self._bank_resolver(self._bank_id, agent)
        if self._per_agent_banks and agent:
            sanitized = agent.lower().replace(" ", "-")
            return f"{self._bank_id}-{sanitized}"
        return self._bank_id

    def _ensure_bank(self, bank_id: str) -> None:
        """Create bank if not already created in this session."""
        if bank_id in self._created_banks:
            return

        def _create() -> None:
            client = self._get_client()
            client.create_bank(
                bank_id=bank_id,
                name=bank_id,
                mission=self._mission,
            )

        try:
            call_sync(_create)
            self._created_banks.add(bank_id)
            if self._verbose:
                logger.info(f"Created/updated bank: {bank_id}")
        except Exception as e:
            # Bank may already exist — that's fine
            self._created_banks.add(bank_id)
            if self._verbose:
                logger.warning(f"Bank creation for {bank_id}: {e}")

    def save(
        self,
        value: str,
        metadata: dict[str, Any] | None = None,
        agent: str | None = None,
    ) -> None:
        """Store a memory to Hindsight.

        Called by CrewAI automatically after each task completes.

        Args:
            value: The task output text to store.
            metadata: Optional metadata dict from CrewAI.
            agent: Optional agent role/name that produced this output.

        Raises:
            HindsightError: If the retain operation fails.
        """
        bank_id = self._resolve_bank_id(agent)
        self._ensure_bank(bank_id)

        # Build retain metadata — Hindsight requires dict[str, str]
        retain_metadata: dict[str, str] = {"source": "crewai"}
        if agent:
            retain_metadata["agent"] = agent
        if metadata:
            for k, v in metadata.items():
                retain_metadata[k] = str(v)

        def _retain() -> None:
            self._get_client().retain(
                bank_id=bank_id,
                content=value,
                context=f"crewai:task_output:{agent or 'unknown'}",
                metadata=retain_metadata,
                tags=self._tags,
            )

        try:
            call_sync(_retain)
            if self._verbose:
                logger.info(
                    f"Stored memory to bank {bank_id} (agent={agent}, len={len(value)})"
                )
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise HindsightError(f"Failed to store memory: {e}") from e

    def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Search memories in Hindsight via recall.

        Called by CrewAI automatically at the start of each task.

        Args:
            query: The search query (constructed by CrewAI from task description).
            limit: Maximum results to return.
            score_threshold: Minimum relevance score (0-1).

        Returns:
            List of dicts with keys: context, score, metadata.

        Raises:
            HindsightError: If the recall operation fails.
        """
        bank_id = self._resolve_bank_id(agent=None)

        recall_kwargs: dict[str, Any] = {
            "bank_id": bank_id,
            "query": query,
            "budget": self._budget,
            "max_tokens": self._max_tokens,
        }
        if self._recall_tags:
            recall_kwargs["tags"] = self._recall_tags
            recall_kwargs["tags_match"] = self._recall_tags_match

        def _recall() -> Any:
            return self._get_client().recall(**recall_kwargs)

        try:
            response = call_sync(_recall)

            # Convert RecallResponse to CrewAI's expected list[dict] format.
            results: list[dict[str, Any]] = []
            recall_results = response.results if hasattr(response, "results") else []
            total = max(len(recall_results), 1)

            for i, r in enumerate(recall_results[:limit]):
                # Hindsight returns results ordered by relevance.
                # Assign descending synthetic scores.
                score = 1.0 - (i / total)

                if score < score_threshold:
                    break

                result_metadata: dict[str, Any] = {}
                if r.type:
                    result_metadata["type"] = r.type
                if r.context:
                    result_metadata["source_context"] = r.context
                if r.occurred_start:
                    result_metadata["occurred_start"] = r.occurred_start
                if r.document_id:
                    result_metadata["document_id"] = r.document_id
                if r.metadata:
                    result_metadata.update(r.metadata)
                if r.tags:
                    result_metadata["tags"] = r.tags

                results.append(
                    {
                        "context": r.text,
                        "score": round(score, 4),
                        "metadata": result_metadata,
                    }
                )

            if self._verbose:
                logger.info(
                    f"Recalled {len(results)} memories from bank {bank_id} for query: {query[:80]}"
                )

            return results

        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            raise HindsightError(f"Failed to search memories: {e}") from e

    def reset(self) -> None:
        """Clear all memories by deleting and recreating the bank.

        This removes all facts, entities, and mental models from the
        bank. The bank is recreated with its original mission if one
        was provided.
        """

        def _delete() -> None:
            self._get_client().delete_bank(self._bank_id)

        try:
            call_sync(_delete)
            self._created_banks.discard(self._bank_id)

            if self._verbose:
                logger.info(f"Reset bank: {self._bank_id}")

            # Recreate the bank if mission was set
            if self._mission:
                self._ensure_bank(self._bank_id)

        except Exception as e:
            logger.warning(f"Failed to reset bank: {e}")
            # Don't raise — reset is best-effort
