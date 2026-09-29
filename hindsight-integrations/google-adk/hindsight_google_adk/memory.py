"""Hindsight memory adapter for Google ADK.

Implements ``google.adk.memory.BaseMemoryService`` so any ``Runner`` configured
with ``HindsightMemoryService`` gets persistent long-term memory backed by
Hindsight. Sessions are retained on ``add_session_to_memory`` and recalled on
``search_memory``; both per-event deltas and explicit ``MemoryEntry`` writes
are supported.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

from google.adk.memory import BaseMemoryService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types
from hindsight_client import Hindsight

from ._client import resolve_client
from .config import DEFAULT_BANK_ID_TEMPLATE, get_config
from .errors import HindsightError

if TYPE_CHECKING:
    from google.adk.events.event import Event
    from google.adk.sessions.session import Session

logger = logging.getLogger(__name__)


class HindsightMemoryService(BaseMemoryService):
    """Hindsight-backed implementation of ADK's ``BaseMemoryService``.

    Bank IDs are derived from ``(app_name, user_id)`` via ``bank_id_template``
    (default ``"{app_name}::{user_id}"``) so memory is scoped per user. Pass
    a different template to share banks across users or apps.

    The service never raises on Hindsight failures during ``add_*`` or
    ``search_memory`` — errors are logged and the Runner continues.

    Args:
        client: Pre-built Hindsight client. If omitted, falls back to global
            config set via ``configure(...)``.
        bank_id_template: Format string with ``{app_name}`` / ``{user_id}``
            placeholders.
        context: Source label attached to retained content (``hindsight``'s
            provenance field).
        budget: Recall budget passed to Hindsight (``low``/``mid``/``high``).
        max_tokens: Max tokens for recall results.
        tags: Default tags added to every retain.
        recall_tags: Default tags appended to every recall query filter.
        recall_tags_match: Tag match mode for recall (``any``/``all``/``any_strict``/``all_strict``).
        mission: Optional bank mission. When set, the bank is created on first
            use (idempotent).
    """

    def __init__(
        self,
        *,
        client: Optional[Hindsight] = None,
        bank_id_template: Optional[str] = None,
        context: Optional[str] = None,
        budget: Optional[str] = None,
        max_tokens: Optional[int] = None,
        tags: Optional[list[str]] = None,
        recall_tags: Optional[list[str]] = None,
        recall_tags_match: Optional[str] = None,
        mission: Optional[str] = None,
    ) -> None:
        if client is None:
            raise HindsightError(
                "HindsightMemoryService requires a Hindsight client. "
                "Use HindsightMemoryService.from_url(...) or pass client=..."
            )
        cfg = get_config()
        self._client = client
        self._bank_id_template = bank_id_template or (cfg.bank_id_template if cfg else DEFAULT_BANK_ID_TEMPLATE)
        self._context = context or (cfg.context if cfg else "google-adk")
        self._budget = budget or (cfg.budget if cfg else "mid")
        self._max_tokens = max_tokens if max_tokens is not None else (cfg.max_tokens if cfg else 4096)
        self._tags = tags if tags is not None else (cfg.tags if cfg else None)
        self._recall_tags = recall_tags if recall_tags is not None else (cfg.recall_tags if cfg else None)
        self._recall_tags_match = recall_tags_match or (cfg.recall_tags_match if cfg else "any")
        self._mission = mission if mission is not None else (cfg.mission if cfg else None)
        self._banks_with_mission_set: set[str] = set()

    @classmethod
    def from_client(cls, client: Hindsight, **kwargs: Any) -> "HindsightMemoryService":
        """Construct from a pre-built Hindsight client."""
        return cls(client=client, **kwargs)

    @classmethod
    def from_url(
        cls,
        hindsight_api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> "HindsightMemoryService":
        """Construct by resolving a Hindsight client from URL / api key / global config."""
        client = resolve_client(None, hindsight_api_url, api_key)
        return cls(client=client, **kwargs)

    # ---- bank id and content helpers ---------------------------------------

    def _bank_id(self, app_name: str, user_id: str) -> str:
        return self._bank_id_template.format(app_name=app_name, user_id=user_id)

    def _base_tags(self, app_name: str, user_id: str) -> list[str]:
        tags = [f"app:{app_name}", f"user:{user_id}"]
        if self._tags:
            tags.extend(self._tags)
        return tags

    def _base_metadata(self, app_name: str, user_id: str) -> dict[str, str]:
        return {
            "app_name": app_name,
            "user_id": user_id,
            "source": "google-adk",
        }

    @staticmethod
    def _extract_event_text(event: "Event") -> str:
        """Concatenate textual parts of an event; empty string if none."""
        content = getattr(event, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            return ""
        return " ".join(p.text for p in parts if getattr(p, "text", None))

    @staticmethod
    def _extract_content_text(content: Optional[types.Content]) -> str:
        if content is None or not getattr(content, "parts", None):
            return ""
        return " ".join(p.text for p in content.parts if getattr(p, "text", None))

    def _events_to_document(self, events: Sequence["Event"]) -> str:
        lines: list[str] = []
        for event in events:
            text = self._extract_event_text(event)
            if not text:
                continue
            author = getattr(event, "author", None) or "unknown"
            lines.append(f"{author}: {text}")
        return "\n".join(lines)

    async def _ensure_bank(self, bank_id: str) -> None:
        if not self._mission or bank_id in self._banks_with_mission_set:
            return
        try:
            await self._client.acreate_bank(bank_id=bank_id, mission=self._mission)
        except Exception as exc:  # noqa: BLE001 — never raise to caller
            logger.error("hindsight create_bank failed for %s: %s", bank_id, exc)
        finally:
            self._banks_with_mission_set.add(bank_id)

    # ---- BaseMemoryService implementation ----------------------------------

    async def add_session_to_memory(self, session: "Session") -> None:
        events = getattr(session, "events", None) or []
        content = self._events_to_document(events)
        if not content:
            return
        app_name = getattr(session, "app_name", "")
        user_id = getattr(session, "user_id", "")
        bank_id = self._bank_id(app_name, user_id)
        await self._ensure_bank(bank_id)
        try:
            await self._client.aretain(
                bank_id=bank_id,
                content=content,
                context=self._context,
                document_id=session.id,
                tags=self._base_tags(app_name, user_id),
                metadata=self._base_metadata(app_name, user_id),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("hindsight retain failed for session %s: %s", session.id, exc)

    async def add_events_to_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence["Event"],
        session_id: Optional[str] = None,
        custom_metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        if not events:
            return
        content = self._events_to_document(events)
        if not content:
            return
        bank_id = self._bank_id(app_name, user_id)
        await self._ensure_bank(bank_id)
        document_id = f"{session_id}-{uuid4().hex[:8]}" if session_id else f"events-{uuid4().hex[:12]}"
        tags = self._base_tags(app_name, user_id)
        if session_id:
            tags.append(f"session:{session_id}")
        metadata = self._base_metadata(app_name, user_id)
        if custom_metadata:
            metadata.update({k: str(v) for k, v in custom_metadata.items()})
        try:
            await self._client.aretain(
                bank_id=bank_id,
                content=content,
                context=self._context,
                document_id=document_id,
                tags=tags,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("hindsight retain (events) failed for %s/%s: %s", app_name, user_id, exc)

    async def add_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        memories: Sequence[MemoryEntry],
        custom_metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        if not memories:
            return
        bank_id = self._bank_id(app_name, user_id)
        await self._ensure_bank(bank_id)
        for memory in memories:
            text = self._extract_content_text(memory.content)
            if not text:
                continue
            metadata = self._base_metadata(app_name, user_id)
            if memory.author:
                metadata["author"] = memory.author
            if memory.custom_metadata:
                metadata.update({k: str(v) for k, v in memory.custom_metadata.items()})
            if custom_metadata:
                metadata.update({k: str(v) for k, v in custom_metadata.items()})
            try:
                await self._client.aretain(
                    bank_id=bank_id,
                    content=text,
                    context=self._context,
                    document_id=memory.id,
                    tags=self._base_tags(app_name, user_id),
                    metadata=metadata,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("hindsight retain (memory) failed: %s", exc)

    async def search_memory(self, *, app_name: str, user_id: str, query: str) -> SearchMemoryResponse:
        bank_id = self._bank_id(app_name, user_id)
        recall_tags = [f"user:{user_id}"]
        if self._recall_tags:
            recall_tags.extend(self._recall_tags)
        try:
            response = await self._client.arecall(
                bank_id=bank_id,
                query=query,
                budget=self._budget,
                max_tokens=self._max_tokens,
                tags=recall_tags,
                tags_match=self._recall_tags_match,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("hindsight recall failed for %s/%s: %s", app_name, user_id, exc)
            return SearchMemoryResponse()

        entries: list[MemoryEntry] = []
        for result in getattr(response, "results", []) or []:
            text = getattr(result, "text", None)
            if not text:
                continue
            timestamp = getattr(result, "occurred_start", None)
            entries.append(
                MemoryEntry(
                    content=types.Content(parts=[types.Part(text=text)]),
                    author="hindsight",
                    id=getattr(result, "id", None),
                    timestamp=timestamp,
                )
            )
        return SearchMemoryResponse(memories=entries)
