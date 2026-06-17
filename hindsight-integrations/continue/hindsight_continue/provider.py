"""Core context-provider logic: turn a Continue HTTP request into memory.

Continue's built-in ``http`` context provider POSTs a JSON body shaped like::

    {"query": str, "fullInput": str, "options": {...}, "workspacePath": str}

and expects back either a single context item or a list of them, each shaped::

    {"name": str, "description": str, "content": str}

:func:`build_context_items` performs that translation by recalling from
Hindsight. It is transport-agnostic (no HTTP types) so it can be unit-tested
directly against Continue's exact request contract.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from hindsight_client import Hindsight

from ._client import resolve_client
from .config import HindsightContinueConfig, get_config
from .errors import HindsightError

logger = logging.getLogger(__name__)


@dataclass
class ContextItem:
    """A Continue context item (``name``/``description``/``content``)."""

    name: str
    description: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description, "content": self.content}


def _resolve_query(payload: dict[str, Any]) -> str:
    """Pick the search text from Continue's request.

    Continue sends ``query`` (the text typed after ``@hindsight``) and
    ``fullInput`` (the whole message). Prefer the explicit query; fall back to
    the full message so a bare ``@hindsight`` still recalls against context.
    """
    query = (payload.get("query") or "").strip()
    if query:
        return query
    return (payload.get("fullInput") or "").strip()


def _resolve_bank_id(payload: dict[str, Any], config: HindsightContinueConfig) -> str | None:
    """Resolve the bank id, letting the request override the configured default."""
    options = payload.get("options")
    if isinstance(options, dict):
        override = options.get("bankId") or options.get("bank_id")
        if override:
            return str(override)
    return config.bank_id


def build_context_items(
    payload: dict[str, Any],
    *,
    client: Hindsight | None = None,
    config: HindsightContinueConfig | None = None,
) -> list[ContextItem]:
    """Recall memory for a Continue HTTP context request.

    Args:
        payload: The JSON body Continue POSTed (``query``/``fullInput``/
            ``options``/``workspacePath``).
        client: Pre-resolved Hindsight client (resolved from config if omitted).
        config: Configuration override (global config if omitted).

    Returns:
        A list with a single combined :class:`ContextItem` holding the recalled
        memories, or an empty list when there is nothing to search for or no
        memories match.

    Raises:
        HindsightError: If no bank id is configured or the recall call fails.
    """
    config = config or get_config()
    search = _resolve_query(payload)
    if not search:
        return []

    bank_id = _resolve_bank_id(payload, config)
    if not bank_id:
        raise HindsightError(
            "No Hindsight bank id configured. Set HINDSIGHT_CONTINUE_BANK_ID, "
            "call configure(bank_id=...), or pass options.bankId in the request."
        )

    resolved_client = resolve_client(client)

    recall_kwargs: dict[str, Any] = {
        "bank_id": bank_id,
        "query": search,
        "budget": config.budget,
        "max_tokens": config.max_tokens,
    }
    if config.recall_types:
        recall_kwargs["types"] = config.recall_types
    if config.recall_tags:
        recall_kwargs["tags"] = config.recall_tags
        recall_kwargs["tags_match"] = config.recall_tags_match

    try:
        response = resolved_client.recall(**recall_kwargs)
    except HindsightError:
        raise
    except Exception as e:
        # hindsight_client raises an ApiException carrying the HTTP ``status`` on
        # auth/connection failures; surface it so a 401/403 (bad token) is
        # distinguishable from a benign "no results" in the adapter logs.
        status = getattr(e, "status", None)
        if status is not None:
            logger.error("Recall failed for bank %s (HTTP %s): %s", bank_id, status, e)
            raise HindsightError(f"Recall failed (HTTP {status}) — check HINDSIGHT_API_KEY / HINDSIGHT_API_URL") from e
        logger.error("Recall failed for bank %s: %s", bank_id, e)
        raise HindsightError(f"Recall failed: {e}") from e

    results = getattr(response, "results", None) or []
    if not results:
        return []

    lines = [f"{i}. {result.text}" for i, result in enumerate(results, 1)]
    content = f"{config.preamble}\n\n" + "\n".join(lines)
    description = f"Memory recalled for: {search[:80]}"
    return [ContextItem(name=config.item_name, description=description, content=content)]


def serialize(items: list[ContextItem]) -> list[dict[str, str]]:
    """Serialize context items to the JSON shape Continue expects."""
    return [item.to_dict() for item in items]
