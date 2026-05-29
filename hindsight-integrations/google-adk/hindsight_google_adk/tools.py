"""Google ADK tool wrappers for Hindsight memory operations.

Factory that returns a list of ADK ``FunctionTool`` instances backed by
Hindsight's retain / recall / reflect APIs. Pass the result directly to an
``LlmAgent(tools=[...])`` to let the model decide when to call them.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from google.adk.tools import FunctionTool
from hindsight_client import Hindsight

from ._client import resolve_client
from .config import get_config
from .errors import HindsightError

logger = logging.getLogger(__name__)


def create_hindsight_tools(
    *,
    bank_id: str,
    client: Optional[Hindsight] = None,
    hindsight_api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    budget: Optional[str] = None,
    max_tokens: Optional[int] = None,
    tags: Optional[list[str]] = None,
    recall_tags: Optional[list[str]] = None,
    recall_tags_match: Optional[str] = None,
    retain_metadata: Optional[dict[str, str]] = None,
    retain_document_id: Optional[str] = None,
    recall_types: Optional[list[str]] = None,
    recall_include_entities: bool = False,
    reflect_context: Optional[str] = None,
    reflect_max_tokens: Optional[int] = None,
    reflect_response_schema: Optional[dict[str, Any]] = None,
    reflect_tags: Optional[list[str]] = None,
    reflect_tags_match: Optional[str] = None,
    include_retain: bool = True,
    include_recall: bool = True,
    include_reflect: bool = True,
) -> list[FunctionTool]:
    """Create Hindsight memory tools for a Google ADK agent.

    Args:
        bank_id: Hindsight memory bank these tools operate on.
        client: Pre-built Hindsight client (preferred).
        hindsight_api_url: API URL (used if no client provided).
        api_key: API key (used if no client provided).
        budget: Recall/reflect budget level (``low``/``mid``/``high``).
        max_tokens: Max tokens for recall results.
        tags: Tags applied when storing memories via retain.
        recall_tags: Tags to filter when searching memories.
        recall_tags_match: Tag matching mode (``any``/``all``/``any_strict``/``all_strict``).
        retain_metadata: Default metadata dict for retain operations.
        retain_document_id: Default document_id for retain (groups/upserts memories).
        recall_types: Fact types to filter (``world``, ``experience``, ``opinion``, ``observation``).
        recall_include_entities: Include entity information in recall results.
        reflect_context: Additional context for reflect operations.
        reflect_max_tokens: Max tokens for reflect results (defaults to ``max_tokens``).
        reflect_response_schema: JSON schema to constrain reflect output format.
        reflect_tags: Tags to filter memories used in reflect (defaults to ``recall_tags``).
        reflect_tags_match: Tag matching for reflect (defaults to ``recall_tags_match``).
        include_retain: Include the ``hindsight_retain`` tool.
        include_recall: Include the ``hindsight_recall`` tool.
        include_reflect: Include the ``hindsight_reflect`` tool.

    Returns:
        List of ADK ``FunctionTool`` instances.

    Raises:
        HindsightError: If no client can be resolved from args + global config.
    """
    resolved_client = resolve_client(client, hindsight_api_url, api_key)
    config = get_config()

    effective_tags = tags if tags is not None else (config.tags if config else None)
    effective_recall_tags = recall_tags if recall_tags is not None else (config.recall_tags if config else None)
    effective_recall_tags_match = recall_tags_match or (config.recall_tags_match if config else "any")
    effective_budget = budget or (config.budget if config else "mid")
    effective_max_tokens = max_tokens if max_tokens is not None else (config.max_tokens if config else 4096)

    tools: list[FunctionTool] = []

    if include_retain:

        async def hindsight_retain(content: str) -> str:
            """Store information to long-term memory for later retrieval.

            Use this to save important facts, user preferences, decisions,
            or anything that should be remembered across conversations.

            Args:
                content: The information to store in memory.
            """
            try:
                retain_kwargs: dict[str, Any] = {"bank_id": bank_id, "content": content}
                if effective_tags:
                    retain_kwargs["tags"] = effective_tags
                if retain_metadata:
                    retain_kwargs["metadata"] = retain_metadata
                if retain_document_id:
                    retain_kwargs["document_id"] = retain_document_id
                await resolved_client.aretain(**retain_kwargs)
                return "Memory stored successfully."
            except HindsightError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.error("Retain failed: %s", e)
                raise HindsightError(f"Retain failed: {e}") from e

        tools.append(FunctionTool(hindsight_retain))

    if include_recall:

        async def hindsight_recall(query: str) -> str:
            """Search long-term memory for relevant information.

            Use this to find previously stored facts, preferences, or context.
            Returns a numbered list of matching memories.

            Args:
                query: What to search for in memory.
            """
            try:
                recall_kwargs: dict[str, Any] = {
                    "bank_id": bank_id,
                    "query": query,
                    "budget": effective_budget,
                    "max_tokens": effective_max_tokens,
                }
                if effective_recall_tags:
                    recall_kwargs["tags"] = effective_recall_tags
                    recall_kwargs["tags_match"] = effective_recall_tags_match
                if recall_types:
                    recall_kwargs["types"] = recall_types
                if recall_include_entities:
                    recall_kwargs["include_entities"] = True
                response = await resolved_client.arecall(**recall_kwargs)
                if not response.results:
                    return "No relevant memories found."
                lines = [f"{i}. {r.text}" for i, r in enumerate(response.results, 1)]
                return "\n".join(lines)
            except HindsightError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.error("Recall failed: %s", e)
                raise HindsightError(f"Recall failed: {e}") from e

        tools.append(FunctionTool(hindsight_recall))

    if include_reflect:

        async def hindsight_reflect(query: str) -> str:
            """Synthesize a thoughtful answer from long-term memories.

            Use this when you need a coherent summary or reasoned response
            about what you know, rather than raw memory facts.

            Args:
                query: The question to reflect on using stored memories.
            """
            try:
                reflect_kwargs: dict[str, Any] = {
                    "bank_id": bank_id,
                    "query": query,
                    "budget": effective_budget,
                }
                if reflect_context:
                    reflect_kwargs["context"] = reflect_context
                effective_reflect_max = reflect_max_tokens or effective_max_tokens
                if effective_reflect_max:
                    reflect_kwargs["max_tokens"] = effective_reflect_max
                if reflect_response_schema:
                    reflect_kwargs["response_schema"] = reflect_response_schema
                effective_reflect_tags = reflect_tags if reflect_tags is not None else effective_recall_tags
                effective_reflect_tags_match = reflect_tags_match or effective_recall_tags_match
                if effective_reflect_tags:
                    reflect_kwargs["tags"] = effective_reflect_tags
                    reflect_kwargs["tags_match"] = effective_reflect_tags_match
                response = await resolved_client.areflect(**reflect_kwargs)
                return response.text or "No relevant memories found."
            except HindsightError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.error("Reflect failed: %s", e)
                raise HindsightError(f"Reflect failed: {e}") from e

        tools.append(FunctionTool(hindsight_reflect))

    return tools
