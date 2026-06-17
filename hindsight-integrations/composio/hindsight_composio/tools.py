"""Composio custom tools for Hindsight memory operations.

Exposes Hindsight's retain/recall/reflect operations as Composio in-process
custom tools (``@composio.experimental.tool()``). Each tool resolves the
Hindsight bank from the Composio session's ``user_id`` at call time, so a single
registered tool set isolates memory per user automatically.

Usage::

    from composio import Composio
    from hindsight_composio import register_hindsight_tools

    composio = Composio()
    tools = register_hindsight_tools(
        composio,
        hindsight_api_url="https://api.hindsight.vectorize.io",
        api_key="hsk_...",
    )
    session = composio.create(
        user_id="user-123",  # becomes the Hindsight bank_id
        experimental={"custom_tools": tools},
    )
"""

import logging
from typing import Any

from hindsight_client import Hindsight
from pydantic import BaseModel, Field

from .config import Budget, TagsMatch, get_config
from .errors import HindsightError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input schemas (module-level so annotations resolve for the Composio decorator)
# ---------------------------------------------------------------------------


class RetainInput(BaseModel):
    content: str = Field(description="The information to store in long-term memory.")


class RecallInput(BaseModel):
    query: str = Field(description="The search query to find relevant memories.")


class ReflectInput(BaseModel):
    query: str = Field(description="The question to reflect on using stored memories.")


# ---------------------------------------------------------------------------
# Client resolution
# ---------------------------------------------------------------------------


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

    kwargs: dict[str, Any] = {"base_url": url, "timeout": 30.0}
    if key:
        kwargs["api_key"] = key
    return Hindsight(**kwargs)


def register_hindsight_tools(
    composio: Any,
    *,
    client: Hindsight | None = None,
    hindsight_api_url: str | None = None,
    api_key: str | None = None,
    default_bank: str | None = None,
    budget: Budget = "mid",
    max_tokens: int = 4096,
    tags: list[str] | None = None,
    recall_tags: list[str] | None = None,
    recall_tags_match: TagsMatch = "any",
    enable_retain: bool = True,
    enable_recall: bool = True,
    enable_reflect: bool = True,
) -> list[Any]:
    """Create Hindsight memory custom tools for a Composio session.

    Returns a list of Composio ``CustomTool`` objects (one each for retain,
    recall, and reflect, subject to the ``enable_*`` flags). Pass them to
    ``composio.create(user_id=..., experimental={"custom_tools": tools})``.

    The Hindsight bank for each call is the session's ``user_id``; if a call
    has no ``user_id``, ``default_bank`` (or the configured default) is used.

    Args:
        composio: The ``Composio`` instance (provides the ``experimental.tool``
            decorator).
        client: Pre-configured Hindsight client (preferred).
        hindsight_api_url: API URL (used if no client provided).
        api_key: API key (used if no client provided).
        default_bank: Bank used when a call has no Composio ``user_id``.
        budget: Recall/reflect budget level (low/mid/high).
        max_tokens: Maximum tokens for recall results.
        tags: Tags applied when storing memories via retain.
        recall_tags: Tags to filter when searching memories.
        recall_tags_match: Tag matching mode (any/all/any_strict/all_strict).
        enable_retain: Include the retain (store) tool.
        enable_recall: Include the recall (search) tool.
        enable_reflect: Include the reflect (synthesize) tool.

    Returns:
        A list of Composio custom tools ready to register on a session.
    """
    resolved = _resolve_client(client, hindsight_api_url, api_key)

    config = get_config()
    default_bank = default_bank if default_bank is not None else (config.default_bank if config else None)
    budget = budget or (config.budget if config else "mid")
    max_tokens = max_tokens or (config.max_tokens if config else 4096)
    tags = tags if tags is not None else (config.tags if config else None)
    recall_tags = recall_tags if recall_tags is not None else (config.recall_tags if config else None)
    recall_tags_match = recall_tags_match or (config.recall_tags_match if config else "any")

    created_banks: set[str] = set()

    def _bank(ctx: Any) -> str:
        bank = getattr(ctx, "user_id", None) or default_bank
        if not bank:
            raise HindsightError(
                "No Hindsight bank for this call: set a Composio user_id on the session, "
                "or pass default_bank= to register_hindsight_tools()."
            )
        return bank

    def _ensure_bank(bank: str) -> None:
        if bank in created_banks:
            return
        try:
            resolved.create_bank(bank_id=bank, name=bank)
        except Exception as e:
            # Bank likely already exists; treat as created either way. Logged at
            # debug so a real auth/network failure is visible here rather than
            # only surfacing later on the retain call.
            logger.debug(f"create_bank({bank!r}) failed (assuming it exists): {e}")
        created_banks.add(bank)

    def hindsight_retain(input: RetainInput, ctx: Any) -> dict:
        """Store information to long-term memory for later retrieval.

        Use this to save important facts, user preferences, decisions, or any
        information that should be remembered across conversations.
        """
        try:
            bank = _bank(ctx)
            _ensure_bank(bank)
            retain_kwargs: dict[str, Any] = {"bank_id": bank, "content": input.content}
            if tags:
                retain_kwargs["tags"] = tags
            resolved.retain(**retain_kwargs)
            return {"status": "stored", "bank": bank}
        except HindsightError:
            raise
        except Exception as e:
            logger.error(f"Retain failed: {e}")
            raise HindsightError(f"Retain failed: {e}") from e

    def hindsight_recall(input: RecallInput, ctx: Any) -> dict:
        """Search long-term memory for relevant information.

        Use this to find previously stored facts, preferences, or context.
        Returns a list of matching memories (empty if none are found).
        """
        try:
            bank = _bank(ctx)
            recall_kwargs: dict[str, Any] = {
                "bank_id": bank,
                "query": input.query,
                "budget": budget,
                "max_tokens": max_tokens,
            }
            if recall_tags:
                recall_kwargs["tags"] = recall_tags
                recall_kwargs["tags_match"] = recall_tags_match
            response = resolved.recall(**recall_kwargs)
            results = response.results or []
            memories = [r.text for r in results]
            return {"memories": memories, "count": len(memories)}
        except HindsightError:
            raise
        except Exception as e:
            logger.error(f"Recall failed: {e}")
            raise HindsightError(f"Recall failed: {e}") from e

    def hindsight_reflect(input: ReflectInput, ctx: Any) -> dict:
        """Synthesize a thoughtful answer from long-term memories.

        Use this when you need a coherent summary or reasoned response about
        what you know, rather than raw memory facts.
        """
        try:
            bank = _bank(ctx)
            response = resolved.reflect(bank_id=bank, query=input.query, budget=budget)
            return {"answer": response.text or "No relevant memories found."}
        except HindsightError:
            raise
        except Exception as e:
            logger.error(f"Reflect failed: {e}")
            raise HindsightError(f"Reflect failed: {e}") from e

    decorate = composio.experimental.tool

    tools: list[Any] = []
    if enable_retain:
        tools.append(decorate(hindsight_retain))
    if enable_recall:
        tools.append(decorate(hindsight_recall))
    if enable_reflect:
        tools.append(decorate(hindsight_reflect))
    return tools


def memory_instructions(
    *,
    bank_id: str,
    client: Hindsight | None = None,
    hindsight_api_url: str | None = None,
    api_key: str | None = None,
    query: str = "relevant context about the user",
    budget: Budget = "low",
    max_results: int = 5,
    max_tokens: int = 4096,
    prefix: str = "Relevant memories:\n",
    tags: list[str] | None = None,
    tags_match: TagsMatch = "any",
) -> str:
    """Pre-recall memories for injection into an agent's system prompt.

    Performs a sync recall and returns a formatted string of memories. Composio
    doesn't auto-inject context, so add the returned string to your agent's
    system prompt yourself.

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
    resolved = _resolve_client(client, hindsight_api_url, api_key)

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
        response = resolved.recall(**recall_kwargs)
        results = response.results[:max_results] if response.results else []
        if not results:
            return ""
        lines = [prefix]
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result.text}")
        return "\n".join(lines)
    except Exception:
        # Silently return empty — instructions failures shouldn't block the agent.
        return ""
