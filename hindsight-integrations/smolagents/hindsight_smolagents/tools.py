"""SmolAgents Tool subclasses for Hindsight memory operations.

Provides three ``Tool`` subclasses (retain, recall, reflect) that give
SmolAgents agents persistent long-term memory via the Hindsight API.
"""

from __future__ import annotations

import logging
from typing import Any

from hindsight_client import Hindsight
from smolagents import Tool

from .config import get_config
from .errors import HindsightError

logger = logging.getLogger(__name__)


def _resolve_client(
    client: Hindsight | None,
    hindsight_api_url: str | None,
    api_key: str | None,
) -> Hindsight:
    """Resolve a Hindsight client from explicit args or global config."""
    if client is not None:
        return client

    config = get_config()
    url = hindsight_api_url or (config.hindsight_api_url if config else None)
    key = api_key or (config.api_key if config else None)

    if url is None:
        raise HindsightError(
            "No Hindsight API URL configured. "
            "Pass client= or hindsight_api_url=, or call configure() first."
        )

    kwargs: dict[str, Any] = {"base_url": url, "timeout": 30.0}
    if key:
        kwargs["api_key"] = key
    return Hindsight(**kwargs)


class HindsightRetainTool(Tool):
    """Store information to long-term memory for later retrieval.

    Use this to save important facts, user preferences, decisions,
    or any information that should be remembered across conversations.
    """

    name = "hindsight_retain"
    description = (
        "Store information to long-term memory for later retrieval. "
        "Use this to save important facts, user preferences, decisions, "
        "or any information that should be remembered across conversations."
    )
    inputs = {
        "content": {
            "type": "string",
            "description": "The information to store in memory.",
        },
    }
    output_type = "string"

    def __init__(
        self,
        *,
        bank_id: str,
        client: Hindsight | None = None,
        hindsight_api_url: str | None = None,
        api_key: str | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._bank_id = bank_id
        self._client = _resolve_client(client, hindsight_api_url, api_key)
        self._created_banks: set[str] = set()

        config = get_config()
        self._tags = tags if tags is not None else (config.tags if config else None)

    def _ensure_bank(self, bank_id: str) -> None:
        """Create bank if not already created in this session."""
        if bank_id in self._created_banks:
            return
        try:
            self._client.create_bank(bank_id=bank_id, name=bank_id)
            self._created_banks.add(bank_id)
        except Exception:
            self._created_banks.add(bank_id)

    def forward(self, content: str) -> str:
        try:
            self._ensure_bank(self._bank_id)

            retain_kwargs: dict[str, Any] = {
                "bank_id": self._bank_id,
                "content": content,
            }
            if self._tags:
                retain_kwargs["tags"] = self._tags
            self._client.retain(**retain_kwargs)
            return "Memory stored successfully."
        except HindsightError:
            raise
        except Exception as e:
            logger.error(f"Retain failed: {e}")
            raise HindsightError(f"Retain failed: {e}") from e


class HindsightRecallTool(Tool):
    """Search long-term memory for relevant information.

    Use this to find previously stored facts, preferences, or context.
    Returns a numbered list of matching memories.
    """

    name = "hindsight_recall"
    description = (
        "Search long-term memory for relevant information. "
        "Use this to find previously stored facts, preferences, or context. "
        "Returns a numbered list of matching memories."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to find relevant memories.",
        },
    }
    output_type = "string"

    def __init__(
        self,
        *,
        bank_id: str,
        client: Hindsight | None = None,
        hindsight_api_url: str | None = None,
        api_key: str | None = None,
        budget: str = "mid",
        max_tokens: int = 4096,
        recall_tags: list[str] | None = None,
        recall_tags_match: str = "any",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._bank_id = bank_id
        self._client = _resolve_client(client, hindsight_api_url, api_key)

        config = get_config()
        self._budget = budget or (config.budget if config else "mid")
        self._max_tokens = max_tokens or (config.max_tokens if config else 4096)
        self._recall_tags = (
            recall_tags
            if recall_tags is not None
            else (config.recall_tags if config else None)
        )
        self._recall_tags_match = recall_tags_match or (
            config.recall_tags_match if config else "any"
        )

    def forward(self, query: str) -> str:
        try:
            recall_kwargs: dict[str, Any] = {
                "bank_id": self._bank_id,
                "query": query,
                "budget": self._budget,
                "max_tokens": self._max_tokens,
            }
            if self._recall_tags:
                recall_kwargs["tags"] = self._recall_tags
                recall_kwargs["tags_match"] = self._recall_tags_match
            response = self._client.recall(**recall_kwargs)
            if not response.results:
                return "No relevant memories found."
            lines = []
            for i, result in enumerate(response.results, 1):
                lines.append(f"{i}. {result.text}")
            return "\n".join(lines)
        except HindsightError:
            raise
        except Exception as e:
            logger.error(f"Recall failed: {e}")
            raise HindsightError(f"Recall failed: {e}") from e


class HindsightReflectTool(Tool):
    """Synthesize a thoughtful answer from long-term memories.

    Use this when you need a coherent summary or reasoned response
    about what you know, rather than raw memory facts.
    """

    name = "hindsight_reflect"
    description = (
        "Synthesize a thoughtful answer from long-term memories. "
        "Use this when you need a coherent summary or reasoned response "
        "about what you know, rather than raw memory facts."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The question to reflect on using stored memories.",
        },
    }
    output_type = "string"

    def __init__(
        self,
        *,
        bank_id: str,
        client: Hindsight | None = None,
        hindsight_api_url: str | None = None,
        api_key: str | None = None,
        budget: str = "mid",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._bank_id = bank_id
        self._client = _resolve_client(client, hindsight_api_url, api_key)

        config = get_config()
        self._budget = budget or (config.budget if config else "mid")

    def forward(self, query: str) -> str:
        try:
            reflect_kwargs: dict[str, Any] = {
                "bank_id": self._bank_id,
                "query": query,
                "budget": self._budget,
            }
            response = self._client.reflect(**reflect_kwargs)
            return response.text or "No relevant memories found."
        except HindsightError:
            raise
        except Exception as e:
            logger.error(f"Reflect failed: {e}")
            raise HindsightError(f"Reflect failed: {e}") from e


def create_hindsight_tools(
    *,
    bank_id: str,
    client: Hindsight | None = None,
    hindsight_api_url: str | None = None,
    api_key: str | None = None,
    budget: str = "mid",
    max_tokens: int = 4096,
    tags: list[str] | None = None,
    recall_tags: list[str] | None = None,
    recall_tags_match: str = "any",
    enable_retain: bool = True,
    enable_recall: bool = True,
    enable_reflect: bool = True,
) -> list[Tool]:
    """Create all Hindsight memory tools for use with SmolAgents.

    Returns a list of Tool instances that can be passed to a SmolAgents agent.

    Args:
        bank_id: The Hindsight memory bank to operate on.
        client: Pre-configured Hindsight client (preferred).
        hindsight_api_url: API URL (used if no client provided).
        api_key: API key (used if no client provided).
        budget: Recall/reflect budget level (low/mid/high).
        max_tokens: Maximum tokens for recall results.
        tags: Tags applied when storing memories via retain.
        recall_tags: Tags to filter when searching memories.
        recall_tags_match: Tag matching mode (any/all/any_strict/all_strict).
        enable_retain: Include the retain (store) tool.
        enable_recall: Include the recall (search) tool.
        enable_reflect: Include the reflect (synthesize) tool.

    Returns:
        A list of SmolAgents Tool instances.
    """
    resolved_client = _resolve_client(client, hindsight_api_url, api_key)

    tools: list[Tool] = []
    if enable_retain:
        tools.append(
            HindsightRetainTool(
                bank_id=bank_id,
                client=resolved_client,
                tags=tags,
            )
        )
    if enable_recall:
        tools.append(
            HindsightRecallTool(
                bank_id=bank_id,
                client=resolved_client,
                budget=budget,
                max_tokens=max_tokens,
                recall_tags=recall_tags,
                recall_tags_match=recall_tags_match,
            )
        )
    if enable_reflect:
        tools.append(
            HindsightReflectTool(
                bank_id=bank_id,
                client=resolved_client,
                budget=budget,
            )
        )
    return tools


def memory_instructions(
    *,
    bank_id: str,
    client: Hindsight | None = None,
    hindsight_api_url: str | None = None,
    api_key: str | None = None,
    query: str = "relevant context about the user",
    budget: str = "low",
    max_results: int = 5,
    max_tokens: int = 4096,
    prefix: str = "Relevant memories:\n",
    tags: list[str] | None = None,
    tags_match: str = "any",
) -> str:
    """Pre-recall memories for injection into agent system prompt.

    Performs a sync recall at construction time and returns a formatted
    string of memories. SmolAgents doesn't have auto-injection, so add
    the returned string to the agent's ``system_prompt`` parameter.

    Args:
        bank_id: The Hindsight memory bank to recall from.
        client: Pre-configured Hindsight client (preferred).
        hindsight_api_url: API URL (used if no client provided).
        api_key: API key (used if no client provided).
        query: The recall query to find relevant memories.
        budget: Recall budget level (low/mid/high).
        max_results: Maximum number of memories to include.
        max_tokens: Maximum tokens for recall results.
        prefix: Text prepended before the memory list.
        tags: Tags to filter recall results.
        tags_match: Tag matching mode (any/all/any_strict/all_strict).

    Returns:
        A formatted string of memories, or empty string if none found.

    Raises:
        HindsightError: If no client or API URL can be resolved.
    """
    resolved_client = _resolve_client(client, hindsight_api_url, api_key)

    try:
        recall_kwargs: dict[str, Any] = {
            "bank_id": bank_id,
            "query": query,
            "budget": budget,
            "max_tokens": max_tokens,
        }
        if tags:
            recall_kwargs["tags"] = tags
            recall_kwargs["tags_match"] = tags_match
        response = resolved_client.recall(**recall_kwargs)
        results = response.results[:max_results] if response.results else []
        if not results:
            return ""
        lines = [prefix]
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result.text}")
        return "\n".join(lines)
    except Exception:
        # Silently return empty — instructions failures shouldn't block the agent
        return ""
