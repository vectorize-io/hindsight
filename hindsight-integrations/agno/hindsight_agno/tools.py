"""Agno Toolkit for Hindsight memory operations.

Provides a ``Toolkit`` subclass that registers retain/recall/reflect
as agent-callable tools, following the same pattern as Agno's ``Mem0Tools``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from importlib import metadata
from typing import Any

from agno.run.base import RunContext
from agno.tools.toolkit import Toolkit
from hindsight_client import Hindsight

from .config import get_config
from .errors import HindsightError

logger = logging.getLogger(__name__)

try:
    _VERSION = metadata.version("hindsight-agno")
except metadata.PackageNotFoundError:
    _VERSION = "0.0.0"
_USER_AGENT = f"hindsight-agno/{_VERSION}"

_TOOL_INSTRUCTIONS = """\
You have access to long-term memory via Hindsight tools.

- Use `retain_memory` to save important facts, user preferences, decisions, \
or any information that should be remembered across conversations.
- Use `recall_memory` to search for previously stored facts, preferences, or context.
- Use `reflect_on_memory` to synthesize a thoughtful, reasoned answer from \
what you know, rather than raw memory facts.

Proactively store information the user shares that may be useful later. \
When answering questions, check memory first for relevant context.\
"""


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
            "No Hindsight API URL configured. Pass client= or hindsight_api_url=, or call configure() first."
        )

    kwargs: dict[str, Any] = {"base_url": url, "timeout": 30.0, "user_agent": _USER_AGENT}
    if key:
        kwargs["api_key"] = key
    return Hindsight(**kwargs)


class HindsightTools(Toolkit):
    """Agno Toolkit providing Hindsight memory tools.

    Registers retain, recall, and reflect as agent-callable tools
    following the same pattern as Agno's ``Mem0Tools``.

    Args:
        bank_id: Static memory bank ID.
        bank_resolver: Callable that resolves bank_id from RunContext.
        client: Pre-configured Hindsight client.
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
        **kwargs: Passed through to Toolkit (e.g. include_tools, exclude_tools).

    Example::

        from agno.agent import Agent
        from agno.models.openai import OpenAIChat
        from hindsight_agno import HindsightTools

        agent = Agent(
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=[HindsightTools(
                bank_id="user-123",
                hindsight_api_url="http://localhost:8888",
            )],
        )
        agent.print_response("Remember that I prefer dark mode")
    """

    def __init__(
        self,
        *,
        bank_id: str | None = None,
        bank_resolver: Callable[[RunContext], str] | None = None,
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
        **kwargs: Any,
    ):
        self._bank_id = bank_id
        self._bank_resolver = bank_resolver
        self._client = _resolve_client(client, hindsight_api_url, api_key)
        self._created_banks: set[str] = set()

        # Resolve defaults from global config
        config = get_config()
        self._budget = budget or (config.budget if config else "mid")
        self._max_tokens = max_tokens or (config.max_tokens if config else 4096)
        self._tags = tags if tags is not None else (config.tags if config else None)
        self._recall_tags = recall_tags if recall_tags is not None else (config.recall_tags if config else None)
        self._recall_tags_match = recall_tags_match or (config.recall_tags_match if config else "any")

        # Build list of tools to register based on enable flags
        tools: list[Callable[..., Any]] = []
        if enable_retain:
            tools.append(self.retain_memory)
        if enable_recall:
            tools.append(self.recall_memory)
        if enable_reflect:
            tools.append(self.reflect_on_memory)

        super().__init__(
            name="hindsight_tools",
            tools=tools,
            instructions=_TOOL_INSTRUCTIONS,
            **kwargs,
        )

    def _resolve_bank_id(self, run_context: RunContext) -> str:
        """Resolve the effective bank_id for this operation.

        Resolution order:
        1. bank_resolver(run_context) if set
        2. Static bank_id if set
        3. run_context.user_id if available
        4. Raise HindsightError
        """
        if self._bank_resolver is not None:
            return self._bank_resolver(run_context)

        if self._bank_id is not None:
            return self._bank_id

        user_id = getattr(run_context, "user_id", None)
        if user_id:
            return user_id

        raise HindsightError(
            "No bank_id available. Provide bank_id=, bank_resolver=, or ensure run_context.user_id is set."
        )

    def _ensure_bank(self, bank_id: str) -> None:
        """Create bank if not already created in this session."""
        if bank_id in self._created_banks:
            return

        try:
            self._client.create_bank(bank_id=bank_id, name=bank_id)
            self._created_banks.add(bank_id)
        except Exception:
            # Bank may already exist — that's fine
            self._created_banks.add(bank_id)

    def retain_memory(self, run_context: RunContext, content: str) -> str:
        """Store information to long-term memory for later retrieval.

        Use this to save important facts, user preferences, decisions,
        or any information that should be remembered across conversations.

        Args:
            run_context: Agno run context.
            content: The information to store in memory.

        Returns:
            A success message string.
        """
        try:
            bank_id = self._resolve_bank_id(run_context)
            self._ensure_bank(bank_id)

            retain_kwargs: dict[str, Any] = {"bank_id": bank_id, "content": content}
            if self._tags:
                retain_kwargs["tags"] = self._tags
            self._client.retain(**retain_kwargs)
            return "Memory stored successfully."
        except HindsightError:
            raise
        except Exception as e:
            logger.error(f"Retain failed: {e}")
            raise HindsightError(f"Retain failed: {e}") from e

    def recall_memory(self, run_context: RunContext, query: str) -> str:
        """Search long-term memory for relevant information.

        Use this to find previously stored facts, preferences, or context.
        Returns a numbered list of matching memories.

        Args:
            run_context: Agno run context.
            query: The search query to find relevant memories.

        Returns:
            A numbered list of matching memories, or a message if none found.
        """
        try:
            bank_id = self._resolve_bank_id(run_context)

            recall_kwargs: dict[str, Any] = {
                "bank_id": bank_id,
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

    def reflect_on_memory(self, run_context: RunContext, query: str) -> str:
        """Synthesize a thoughtful answer from long-term memories.

        Use this when you need a coherent summary or reasoned response
        about what you know, rather than raw memory facts.

        Args:
            run_context: Agno run context.
            query: The question to reflect on using stored memories.

        Returns:
            A synthesized response based on stored memories.
        """
        try:
            bank_id = self._resolve_bank_id(run_context)

            reflect_kwargs: dict[str, Any] = {
                "bank_id": bank_id,
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
    """Pre-recall memories for injection into Agent instructions.

    Performs a sync recall at construction time and returns a formatted
    string of memories. Use with ``Agent(instructions=[...])`` to inject
    relevant context into every run.

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
