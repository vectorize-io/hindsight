"""Knowledge banks: ingest files and texts, search their passages. No LLM anywhere.

Knowledge banks ingest and search, as memory banks retain and recall. An ingest stores the
original files, records the documents as ``pending`` and queues one ``knowledge_ingest``
operation for all of them (or, sync, processes them in the request). The batch splits each file
into passages (see ``media``, or the ingestion strategy the item chose: ``ingest_strategy``),
embeds them with the server's multimodal embedder in rounds across documents, stores each
passage's media next to the original, and swaps every document's passages in at once. Search
embeds the query into the same space and returns the nearest passages.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
import time
import uuid
import weakref
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import asyncpg

from ..config import get_config
from ..engine import bank_aliases
from ..engine.db_utils import acquire_with_retry
from ..engine.response_models import KnowledgeDocumentHit
from ..engine.retain import bank_utils
from ..engine.schema import fq_table
from ..engine.storage import bank_storage_prefix, key_segment
from ..extensions.operation_validator import (
    BankListScope,
    CreateKnowledgeBankContext,
    KnowledgeBankListContext,
    KnowledgeBankReadContext,
    KnowledgeBankReadOperation,
    KnowledgeBankWriteContext,
    KnowledgeBankWriteOperation,
    OperationValidationError,
)
from . import ingest_strategy, media
from .embedders import (
    EmbedInput,
    KnowledgeEmbedder,
    UnsupportedModalityError,
    create_knowledge_embedder,
)
from .language import detect as detect_language
from .store import (
    BankStats,
    DocumentPage,
    DocumentStatus,
    KnowledgeDocument,
    KnowledgePassage,
    KnowledgeStore,
    PassageReplacement,
    RoleVector,
    SearchFilters,
    SearchHit,
    StatusUpdate,
    TimeBucket,
    load_knowledge_store,
)

if TYPE_CHECKING:
    from ..engine.memory_engine import MemoryEngine
    from ..models import RequestContext

logger = logging.getLogger(__name__)

TASK_TYPE = "knowledge_ingest"
KIND = "knowledge"
# Bumped whenever splitting changes what a file becomes, so rewriting an unchanged file re-splits it.
_PIPELINE_VERSION = 9  # 9: PDF page units pack whole lines (as benchmarked)
# How many passages are embedded and stored per round, so a 2h video never sits in memory whole.
_EMBED_ROUND = 16
#: Files an ingest saves at once.
_FILE_SAVE_CONCURRENCY = 16


#: Passages reflect's recall tool reads from a memory bank's linked knowledge banks. Fixed: a per-bank
#: setting was dropped before launch (nothing tuned it).
LINKED_RECALL_TOP_K = 5


class KnowledgeBankError(Exception):
    """A request that cannot be served, with the HTTP status that says why."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class _Embedding:
    role: str | None  # None: the passage's primary vector
    n: int
    input: EmbedInput


@dataclass
class KnowledgeBankInfo:
    bank_id: str
    created_at: datetime | None
    stats: BankStats


@dataclass
class KnowledgeBankPage:
    banks: list[KnowledgeBankInfo]
    #: Knowledge banks matching the search, before paging.
    total: int


@dataclass
class IngestItem:
    """One document to ingest: a file's bytes, or a text."""

    data: bytes
    filename: str | None
    content_type: str
    document_id: str | None = None
    tags: list[str] = field(default_factory=list)
    #: The ingestion strategy for this item; None falls back to the server's default.
    strategy: str | None = None


@dataclass
class IngestResult:
    document_ids: list[str]
    #: Per document, its queued operation; None when nothing was queued (unchanged, or processed in place).
    operation_ids: list[str | None]
    #: Documents identical to what is stored: neither queued nor processed again.
    unchanged_document_ids: list[str] = field(default_factory=list)


@dataclass
class SearchRequest:
    query: str | None
    query_image: bytes | None
    top_k: int
    filters: SearchFilters
    # Text hits come back with this many passages on each side as `context`; hits whose windows
    # overlap in one document are merged into the best one (CHUNKING.md: ±1 was the step that paid).
    neighbors: int = 1
    # Exact text (case-insensitive; a POSIX regular expression when `regex`): passages containing it
    # rank first. Alone, without a query, it ranks documents by how many of their passages match.
    pattern: str | None = None
    regex: bool = False


@dataclass
class SearchResultItem:
    hit: SearchHit
    context: str | None = None
    context_first: int | None = None
    context_last: int | None = None


@dataclass
class KnowledgeConfig:
    """The server's knowledge embedder and the limits a write is held to."""

    provider: str
    model: str
    dimension: int
    media_types: list[str]
    max_file_bytes: int
    max_media_seconds: int
    max_pdf_pages: int
    audio_window_seconds: int
    video_window_seconds: int
    media_overlap_seconds: int
    pdf_page_pixels: int
    text_passage_tokens: int


@dataclass
class DocumentDetail:
    document: KnowledgeDocument
    passages: list[KnowledgePassage]


@dataclass
class OpenedFile:
    stream: AsyncIterator[bytes]
    content_type: str
    filename: str | None = None


@dataclass
class _Plan:
    """How a checked file becomes passages: the built-in splitter for its kind, or a strategy."""

    kind: media.Kind | None
    #: The chosen strategy's name, and its entry for this file's type (None: we chunk the file).
    strategy: str | None = None
    handler: ingest_strategy.Handler | None = None


@dataclass
class _Entry:
    """One checked file of an ingest, under the document id it is stored as."""

    document_id: str
    item: IngestItem
    plan: _Plan
    #: The document an ingestion strategy split this one out of.
    source_document_id: str | None = None


@dataclass
class _Saved:
    storage_key: str
    #: It was a ready document an ingestion strategy had split into files.
    was_split: bool = False


@dataclass
class _Work:
    """One stored document to process: what the ingest decided, carried in the task payload."""

    document_id: str
    storage_key: str
    strategy: str | None = None
    handler: ingest_strategy.Handler | None = None
    was_split: bool = False

    def to_payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"document_id": self.document_id, "storage_key": self.storage_key}
        if self.strategy is not None:
            # Resolved now, so the worker runs what this ingest chose even if the config changes meanwhile.
            out["strategy"] = self.strategy
        if self.handler is not None:
            out["handler"] = {"extension": self.handler.extension, "settings": self.handler.settings}
        if self.was_split:
            out["was_split"] = True
        return out

    @classmethod
    def from_payload(cls, raw: dict[str, Any]) -> _Work:
        handler = raw.get("handler")
        return cls(
            document_id=raw["document_id"],
            storage_key=raw["storage_key"],
            strategy=raw.get("strategy"),
            handler=ingest_strategy.Handler(handler["extension"], handler["settings"]) if handler else None,
            was_split=bool(raw.get("was_split")),
        )


def _work(entry: _Entry, saved: _Saved) -> _Work:
    return _Work(entry.document_id, saved.storage_key, entry.plan.strategy, entry.plan.handler, saved.was_split)


@dataclass
class _InFlight:
    """A document while its batch is processed: the passages embedded so far, or why it failed."""

    work: _Work
    document: KnowledgeDocument
    title: str | None
    media_prefix: str
    passages: list[KnowledgePassage] = field(default_factory=list)
    media_keys: list[str] = field(default_factory=list)
    error: Exception | None = None
    split: bool = False


@dataclass
class _Waiting:
    """A passage waiting in the current embed round."""

    doc: _InFlight
    draft: media.PassageDraft


@dataclass
class LinkedHit:
    bank_id: str
    hit: SearchHit


# The splitter a media type's typical file goes through (a PDF for "document").
_SPLIT_OF: dict[str, media.Kind] = {
    "document": "pdf",
    "image": "image",
    "audio": "audio",
    "video": "video",
    "text": "text",
}
# What the embedder must embed for a kind of file. A PDF takes either: its page images, or its text
# layer alone with a text-only model (see _vectors_to_embed).
_NEEDS: dict[media.Kind, frozenset[str]] = {
    "pdf": frozenset({"image", "text"}),
    "image": frozenset({"image"}),
    "audio": frozenset({"audio"}),
    "video": frozenset({"video"}),
    "text": frozenset({"text"}),
}


class KnowledgeService:
    def __init__(self, engine: MemoryEngine) -> None:
        self._engine = engine
        self._embedder: KnowledgeEmbedder | None = None
        self._store: KnowledgeStore | None = None

    # --- wiring -----------------------------------------------------------------------------

    @property
    def embedder(self) -> KnowledgeEmbedder:
        if self._embedder is None:
            self._embedder = create_knowledge_embedder(get_config(), self._engine.embeddings)
        return self._embedder

    @property
    def store(self) -> KnowledgeStore:
        if self._store is None:
            self._store = load_knowledge_store(self._engine._get_backend, get_config().knowledge_store_extension)
        return self._store

    async def _ready(self, request_context: RequestContext | None) -> None:
        if request_context is not None:
            await self._engine._authenticate_tenant(request_context)
        if not self.embedder.dimension:
            # A remote model's size is known only from its first answer.
            await self.embedder.embed_query("dimension probe")
        await self.store.ensure_ready(self.embedder.dimension)

    async def describe(self) -> KnowledgeConfig:
        if not self.embedder.dimension:
            await self.embedder.embed_query("dimension probe")
        config = get_config()
        e = self.embedder
        return KnowledgeConfig(
            provider=e.provider,
            model=e.model,
            dimension=e.dimension,
            media_types=[
                t
                for t in media.MEDIA_TYPES
                if _NEEDS[_SPLIT_OF[t]] & e.modalities and t not in config.knowledge_disabled_media_types
            ],
            max_file_bytes=config.knowledge_max_file_mb * 1024 * 1024,
            max_media_seconds=config.knowledge_max_media_seconds,
            max_pdf_pages=config.knowledge_max_pdf_pages,
            audio_window_seconds=config.knowledge_audio_window_seconds,
            video_window_seconds=config.knowledge_video_window_seconds,
            media_overlap_seconds=config.knowledge_media_overlap_seconds,
            pdf_page_pixels=config.knowledge_pdf_page_pixels,
            text_passage_tokens=config.knowledge_text_passage_tokens,
        )

    # --- banks ------------------------------------------------------------------------------

    async def _kind_of(self, bank_id: str) -> str | None:
        async with acquire_with_retry(await self._engine._get_backend()) as conn:
            return await conn.fetchval(f"SELECT kind FROM {fq_table('banks')} WHERE bank_id = $1", bank_id)

    async def _require_bank(self, bank_id: str) -> None:
        kind = await self._kind_of(bank_id)
        if kind != KIND:
            raise KnowledgeBankError(f"knowledge bank '{bank_id}' not found", 404)

    async def _authorize(
        self,
        bank_id: str,
        operation: KnowledgeBankReadOperation | KnowledgeBankWriteOperation,
        request_context: RequestContext,
    ) -> None:
        """The operation validator's knowledge-bank hook for one operation (the memory-bank hooks never run
        for a knowledge bank). Run before the bank's existence is checked, so a caller who may not touch
        a bank cannot learn whether it exists from a 404."""
        validator = self._engine._operation_validator
        if validator is None:
            return
        if isinstance(operation, KnowledgeBankReadOperation):
            verdict = validator.validate_knowledge_bank_read(
                KnowledgeBankReadContext(bank_id=bank_id, operation=operation, request_context=request_context)
            )
        else:
            verdict = validator.validate_knowledge_bank_write(
                KnowledgeBankWriteContext(bank_id=bank_id, operation=operation, request_context=request_context)
            )
        await self._engine._validate_operation(verdict)

    async def _open_bank(
        self,
        bank_id: str,
        operation: KnowledgeBankReadOperation | KnowledgeBankWriteOperation,
        request_context: RequestContext,
    ) -> None:
        await self._ready(request_context)
        await self._authorize(bank_id, operation, request_context)
        await self._require_bank(bank_id)

    async def create_bank(self, bank_id: str, request_context: RequestContext) -> KnowledgeBankInfo:
        await self._ready(request_context)
        kind = await self._kind_of(bank_id)
        if kind == "memory":
            raise KnowledgeBankError(f"'{bank_id}' is already a memory bank", 409)
        if kind is None:
            bank_utils.validate_new_bank_id(bank_id)
            validator = self._engine._operation_validator
            if validator:
                await self._engine._validate_operation(
                    validator.validate_create_knowledge_bank(
                        CreateKnowledgeBankContext(bank_id=bank_id, request_context=request_context)
                    )
                )
            async with acquire_with_retry(await self._engine._get_backend()) as conn, conn.transaction():
                # A bank must not be born under a name that is already an alias: resolution prefers a real
                # bank, so the alias would silently start pointing here (see bank_utils' create path).
                await bank_aliases.refuse_alias_as_bank_id_on_conn(conn, bank_id)
                await conn.execute(
                    f"""
                    INSERT INTO {fq_table("banks")} (bank_id, name, disposition, mission, internal_id, kind)
                    VALUES ($1, $2, $3::jsonb, '', $4, $5)
                    ON CONFLICT (bank_id) DO NOTHING
                    """,
                    bank_id,
                    bank_id,
                    json.dumps(bank_utils.DEFAULT_DISPOSITION),
                    uuid.uuid4(),
                    KIND,
                )
        return await self.get_bank(bank_id, request_context)

    async def list_banks(
        self, request_context: RequestContext, *, limit: int = 100, offset: int = 0, q: str | None = None
    ) -> KnowledgeBankPage:
        """Newest first; ``q`` matches part of the bank id, case-insensitively. Counts are read for
        the page only. Scoped by the operation validator's knowledge-bank list hooks: a declared scope
        is applied in SQL, otherwise every match goes through ``filter_knowledge_bank_list``."""
        await self._ready(request_context)
        limit, offset = max(limit, 0), max(offset, 0)
        validator = self._engine._operation_validator
        scope = await validator.knowledge_bank_list_scope(request_context) if validator else BankListScope()
        backend = await self._engine._get_backend()
        allowed = await bank_aliases.resolve_many(backend, scope.bank_ids) if scope and scope.bank_ids else None
        if scope is not None and scope.bank_ids is not None and not allowed:
            return KnowledgeBankPage(banks=[], total=0)
        where = (
            "WHERE kind = $1 AND ($2::text IS NULL OR strpos(lower(bank_id), lower($2)) > 0) "
            "AND ($3::text[] IS NULL OR bank_id = ANY($3))"
        )
        async with acquire_with_retry(backend) as conn:
            if scope is None:
                # The validator may drop any bank, so it sees every match and the page is cut after.
                rows = await conn.fetch(
                    f"SELECT bank_id, created_at FROM {fq_table('banks')} {where} ORDER BY created_at DESC, bank_id",
                    KIND,
                    q or None,
                    None,
                )
                assert validator is not None
                kept = await validator.filter_knowledge_bank_list(
                    KnowledgeBankListContext(
                        banks=[{"bank_id": r["bank_id"], "created_at": r["created_at"]} for r in rows],
                        request_context=request_context,
                    )
                )
                page_rows = [(b["bank_id"], b["created_at"]) for b in kept.banks[offset : offset + limit]]
                total = len(kept.banks)
            else:
                rows = await conn.fetch(
                    f"SELECT bank_id, created_at, count(*) OVER () AS total FROM {fq_table('banks')} {where} "
                    "ORDER BY created_at DESC, bank_id LIMIT $4 OFFSET $5",
                    KIND,
                    q or None,
                    allowed,
                    limit,
                    offset,
                )
                page_rows = [(r["bank_id"], r["created_at"]) for r in rows]
                # An offset past the end returns no rows, and with them no window total.
                total = (
                    int(rows[0]["total"])
                    if rows
                    else await conn.fetchval(
                        f"SELECT count(*) FROM {fq_table('banks')} {where}", KIND, q or None, allowed
                    )
                )
        stats = await self.store.bank_stats([bank_id for bank_id, _ in page_rows])
        return KnowledgeBankPage(
            banks=[KnowledgeBankInfo(bank_id, created_at, stats[bank_id]) for bank_id, created_at in page_rows],
            total=total,
        )

    async def get_bank(self, bank_id: str, request_context: RequestContext) -> KnowledgeBankInfo:
        await self._ready(request_context)
        await self._authorize(bank_id, KnowledgeBankReadOperation.GET_KNOWLEDGE_BANK, request_context)
        async with acquire_with_retry(await self._engine._get_backend()) as conn:
            row = await conn.fetchrow(
                f"SELECT bank_id, created_at FROM {fq_table('banks')} WHERE bank_id = $1 AND kind = $2",
                bank_id,
                KIND,
            )
        if not row:
            raise KnowledgeBankError(f"knowledge bank '{bank_id}' not found", 404)
        stats = await self.store.bank_stats([bank_id])
        return KnowledgeBankInfo(row["bank_id"], row["created_at"], stats[bank_id])

    async def delete_bank(self, bank_id: str, request_context: RequestContext) -> None:
        await self._open_bank(bank_id, KnowledgeBankWriteOperation.DELETE_KNOWLEDGE_BANK, request_context)
        # The banks row cascades to kb_documents -> kb_passages, and to its async operations.
        await self._engine.delete_bank(bank_id, request_context=request_context, _knowledge_bank=True)
        await self._engine._file_storage.delete_prefix(f"{bank_storage_prefix(bank_id)}kb/")

    # --- ingest -----------------------------------------------------------------------------

    def _strategy(self, name: str | None) -> ingest_strategy.IngestStrategy | None:
        """The ingestion strategy named for an item, else the server's default; None when neither is set."""
        config = get_config()
        name = name or config.ingest_default_strategy
        if name is None:
            return None
        strategies = ingest_strategy.parse_strategies(config.ingest_strategies)
        if name not in strategies:
            raise KnowledgeBankError(
                f"unknown ingestion strategy '{name}' (this server has: {', '.join(sorted(strategies)) or 'none'})"
            )
        return strategies[name]

    def _check_item(self, item: IngestItem, *, depth: int = 0) -> _Plan:
        """How the file will be split; refuses what cannot be ingested."""
        limit = get_config().knowledge_max_file_mb * 1024 * 1024
        if len(item.data) > limit:
            raise KnowledgeBankError(
                f"'{item.filename}' is {len(item.data) / 1048576:.1f}MB; the limit is "
                f"{get_config().knowledge_max_file_mb}MB (HINDSIGHT_API_KNOWLEDGE_MAX_FILE_MB)",
                413,
            )
        if not item.data:
            raise KnowledgeBankError(f"'{item.filename}' is empty")
        kind = media.classify(item.content_type, item.filename)
        strategy = self._strategy(item.strategy)
        # A file inside a file inside ... stops at MAX_DEPTH: deeper ones are chunked as they are.
        handler = (
            strategy.handler_for(item.content_type, item.filename)
            if strategy is not None and depth < ingest_strategy.MAX_DEPTH
            else None
        )
        if strategy is not None and handler is not None:
            # The extension decides what the file becomes, so a type we cannot chunk is fine here. It is
            # loaded now so a misconfigured class fails this ingest, not the worker later.
            try:
                ingest_strategy.load_extension(handler)
            except Exception as exc:
                raise KnowledgeBankError(f"ingestion strategy '{strategy.name}': {exc}", 500) from exc
            return _Plan(kind=kind, strategy=strategy.name, handler=handler)
        self._check_kind(kind, item)
        return _Plan(kind=kind, strategy=strategy.name if strategy else None)

    def _check_kind(self, kind: media.Kind | None, item: IngestItem | KnowledgeDocument) -> media.Kind:
        if kind is None:
            raise KnowledgeBankError(
                f"unsupported file type for '{item.filename}' ({item.content_type}). Knowledge banks take PDF, "
                "images, audio, video, text, Markdown, JSON, CSV, HTML and Office documents.",
                415,
            )
        media_type = media.media_type_of(kind, item.filename)
        if media_type in get_config().knowledge_disabled_media_types:
            raise KnowledgeBankError(
                f"'{item.filename}' is {media_type}, which this server does not ingest "
                "(HINDSIGHT_API_KNOWLEDGE_DISABLED_MEDIA_TYPES)",
                415,
            )
        if not _NEEDS[kind] & self.embedder.modalities:
            raise KnowledgeBankError(
                f"'{item.filename}' is {media_type}, but the knowledge embedder ({self.embedder.provider}) only embeds "
                f"{', '.join(sorted(self.embedder.modalities))}. Configure HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_PROVIDER "
                "with a model that embeds it.",
                415,
            )
        return kind

    def _check_passage_modality(self, modality: str) -> None:
        """A passage an ingest extension emitted: something this server embeds and ingests."""
        if modality not in self.embedder.modalities:
            raise media.MediaError(f"the ingest extension emitted {modality} passages, which the embedder cannot embed")
        if modality in get_config().knowledge_disabled_media_types:
            raise media.MediaError(
                f"the ingest extension emitted {modality} passages, which this server does not ingest"
            )

    def _content_hash(self, item: IngestItem, plan: _Plan) -> str:
        # The splitting settings are part of the hash: rewriting the same file after they change
        # must re-split it, not be skipped as unchanged. So is a strategy, its settings and version.
        c = get_config()
        how: str = plan.kind or ""
        if plan.handler is not None:
            version = ingest_strategy.load_extension(plan.handler).version
            how = f"strategy:{plan.strategy}:{plan.handler.key}:{version}"
        settings = (
            f"v{_PIPELINE_VERSION}|{how}|{self.embedder.provider}:{self.embedder.model}:{self.embedder.dimension}|{c.knowledge_pdf_page_pixels}"
            f"|{c.knowledge_image_tokens}|{c.knowledge_audio_window_seconds}|{c.knowledge_video_window_seconds}"
            f"|{c.knowledge_media_overlap_seconds}|{c.knowledge_pdf_text_weight}|{c.knowledge_pdf_bands}|{c.knowledge_text_passage_tokens}"
        )
        return hashlib.sha256(item.data + settings.encode()).hexdigest()

    async def ingest(
        self, bank_id: str, items: list[IngestItem], request_context: RequestContext, *, wait: bool = False
    ) -> IngestResult:
        """Store each item and queue one operation that processes them all; an unchanged item is skipped.

        ``wait`` processes them here instead and returns once all are searchable (no operation); the
        files a strategy splits one into are still queued, as their own operation. If any failed it
        raises, after the others committed, as with a retain."""
        await self._open_bank(bank_id, KnowledgeBankWriteOperation.INGEST_KNOWLEDGE_BANK, request_context)
        # All-or-nothing: check every item before storing any.
        entries = [_Entry(item.document_id or str(uuid.uuid4()), item, self._check_item(item)) for item in items]
        seen: set[str] = set()
        for e in entries:
            if e.document_id in seen:
                raise KnowledgeBankError(f"document '{e.document_id}' appears more than once in this ingest")
            seen.add(e.document_id)
        caller: dict[str, Any] = {}
        if request_context.tenant_id:
            caller["_tenant_id"] = request_context.tenant_id
        if request_context.api_key_id:
            caller["_api_key_id"] = request_context.api_key_id
        saved = await self._save(bank_id, entries)
        work = [_work(e, s) for e, s in zip(entries, saved, strict=True) if s is not None]
        operation_id: str | None = None
        if work and wait:
            await self.process_batch(bank_id, work, caller=caller)
        elif work:
            operation_id = await self._queue(bank_id, work, caller)
        return IngestResult(
            document_ids=[e.document_id for e in entries],
            operation_ids=[operation_id if s is not None else None for s in saved],
            unchanged_document_ids=[e.document_id for e, s in zip(entries, saved, strict=True) if s is None],
        )

    async def _queue(self, bank_id: str, work: list[_Work], caller: dict[str, Any], depth: int = 0) -> str:
        """Queue one operation that processes every stored document of ``work``; returns its id. One task
        row and one operation row however many documents, like an async retain of a batch."""
        payload: dict[str, Any] = {"documents": [w.to_payload() for w in work], **caller}
        if depth:
            payload["depth"] = depth
        op = await self._engine._submit_async_operation(
            bank_id=bank_id,
            operation_type=TASK_TYPE,
            task_type=TASK_TYPE,
            task_payload=payload,
            result_metadata={"document_ids": [w.document_id for w in work]},
        )
        return str(op["operation_id"])

    async def _save(self, bank_id: str, entries: list[_Entry]) -> list[_Saved | None]:
        """Store checked files as pending documents; None for one that is unchanged. One read and one write
        of the store for the whole batch. The files are saved first, concurrently, and only then the rows,
        so a document row never names a file that is not there yet."""
        if not entries:
            return []
        existing = await self.store.get_documents(bank_id, [e.document_id for e in entries])
        saved: list[_Saved | None] = []
        rows: list[KnowledgeDocument] = []
        files: list[tuple[str, _Entry]] = []
        for e in entries:
            item = e.item
            item.filename = item.filename or f"{e.document_id}.md"
            content_hash = self._content_hash(item, e.plan)
            before = existing.get(e.document_id)
            if (
                before
                and before.content_hash == content_hash
                and before.status == "ready"
                and before.tags == item.tags
                # The name is embedded as each passage's title, and the type decides how it is served.
                and before.filename == item.filename
                and before.content_type == item.content_type
            ):
                saved.append(None)  # unchanged: nothing to do
                continue
            storage_key = f"{bank_storage_prefix(bank_id)}kb/files/{uuid.uuid4()}/{key_segment(item.filename)}"
            files.append((storage_key, e))
            # A ready document without passages is one a strategy split into files (anything else ready
            # has at least one): only those need their split-out documents looked up later.
            saved.append(
                _Saved(storage_key, was_split=bool(before and before.status == "ready" and not before.passage_count))
            )
            rows.append(
                KnowledgeDocument(
                    document_id=e.document_id,
                    filename=item.filename,
                    content_type=item.content_type,
                    # A type only a strategy reads is filed as a document until it is split.
                    media_type=media.media_type_of(e.plan.kind, item.filename) if e.plan.kind else "document",
                    size_bytes=len(item.data),
                    storage_key=storage_key,
                    content_hash=content_hash,
                    status="pending",
                    error=None,
                    passage_count=before.passage_count if before else 0,
                    tags=item.tags,
                    source_document_id=e.source_document_id,
                )
            )
        # Saved one after another, these were most of an ingest's time (a commit each on Postgres, a few
        # calls each on an object store). Bounded so a 100-file ingest does not open 100 connections.
        limit = asyncio.Semaphore(_FILE_SAVE_CONCURRENCY)

        async def store(key: str, e: _Entry) -> None:
            async with limit:
                await self._engine._file_storage.store(
                    file_data=e.item.data,
                    key=key,
                    metadata={"content_type": e.item.content_type, "bank_id": bank_id, "document_id": e.document_id},
                )

        await asyncio.gather(*(store(key, e) for key, e in files))
        previous = await self.store.put_documents(bank_id, rows)
        await self._delete_keys([k for d in rows if (k := previous.get(d.document_id)) and k != d.storage_key])
        return saved

    async def _delete_keys(self, keys: list[str]) -> None:
        for key in keys:
            try:
                await self._engine._file_storage.delete(key)
            except Exception as exc:  # an orphaned blob is harmless; a failed write over it is not
                logger.warning(f"Could not delete knowledge file {key}: {exc}")

    # --- ingest (worker) --------------------------------------------------------------------

    def _drafts(self, document: KnowledgeDocument, data: bytes) -> Iterator[media.PassageDraft]:
        c = get_config()
        kind = media.classify(document.content_type, document.filename)
        if kind == "pdf":
            return media.split_pdf(
                data,
                page_pixels=c.knowledge_pdf_page_pixels,
                max_pages=c.knowledge_max_pdf_pages,
                bands=c.knowledge_pdf_bands,
            )
        if kind == "image":
            return iter(media.split_image(data))
        if kind in ("audio", "video"):
            window = c.knowledge_audio_window_seconds if kind == "audio" else c.knowledge_video_window_seconds
            return media.split_media(
                data,
                kind,
                filename=document.filename,
                window_seconds=window,
                overlap_seconds=c.knowledge_media_overlap_seconds,
                max_seconds=c.knowledge_max_media_seconds,
            )
        text = media.read_text(data, document.filename)
        return iter(media.split_text(text, tokens=c.knowledge_text_passage_tokens))

    def _vectors_to_embed(
        self, draft: media.PassageDraft, document: KnowledgeDocument, title: str | None
    ) -> list[_Embedding]:
        """The vectors one passage is stored as. A PDF page (PDF_GEMMA.md): the whole page image at 560
        vision tokens with the file name, three overlapping horizontal bands at 280, the page text and its
        128-token units, all titled with the file name. Anything else: one vector, with the asset's name and
        tags in the same input (GAME_ASSETS.md)."""
        if (
            draft.page is not None
            and document.media_type == "document"
            and draft.modality == "image"
            and "image" not in self.embedder.modalities
        ):
            # A text-only model: the page is found by its text layer alone (a scanned page by its title).
            page_text = draft.embed_text or draft.text or ""
            units = media.pack_lines(page_text, tokens=media.PDF_TEXT_UNIT_TOKENS) if page_text else []
            return [_Embedding(None, 0, EmbedInput(modality="text", text=page_text, title=title))] + [
                _Embedding("unit", n, EmbedInput(modality="text", text=u, title=title)) for n, u in enumerate(units)
            ]
        if draft.page is not None and document.media_type == "document" and draft.modality == "image":
            out = [
                _Embedding(
                    None,
                    0,
                    EmbedInput(
                        modality="image",
                        data=draft.data,
                        mime=draft.mime,
                        title=title,
                        image_tokens=media.PDF_PAGE_IMAGE_TOKENS,
                    ),
                )
            ]
            out += [
                _Embedding(
                    "band",
                    n,
                    EmbedInput(modality="image", data=band, mime="image/png", image_tokens=media.PDF_BAND_IMAGE_TOKENS),
                )
                for n, band in enumerate(draft.bands)
            ]
            if draft.embed_text and "text" in self.embedder.modalities:
                out.append(_Embedding("page_text", 0, EmbedInput(modality="text", text=draft.embed_text, title=title)))
                units = media.pack_lines(draft.embed_text, tokens=media.PDF_TEXT_UNIT_TOKENS)
                out += [
                    _Embedding("unit", n, EmbedInput(modality="text", text=u, title=title)) for n, u in enumerate(units)
                ]
            return out
        return [
            _Embedding(
                None,
                0,
                EmbedInput(
                    modality=draft.modality,
                    text=draft.text,
                    data=draft.data,
                    mime=draft.mime,
                    title=title,
                    tags=tuple(document.tags) if draft.modality != "text" else (),
                ),
            )
        ]

    async def process_batch(
        self,
        bank_id: str,
        work: list[_Work],
        *,
        retries_left: bool = False,
        depth: int = 0,
        caller: dict[str, Any] | None = None,
    ) -> None:
        """The worker half of an ingest (and the whole of a sync one): split stored documents into passages,
        embed them and commit them, as one batch. The store is read once, marked processing once and
        written once; passages are embedded in rounds of ``_EMBED_ROUND`` across documents, so a batch of
        short texts costs a few embed calls, not one per document.

        A permanent error (a ``ValueError``: a file that cannot be split) fails that document alone; the
        others still commit, then this raises a ``MediaError`` naming the failures. Any other error (the
        embedder down) commits nothing and propagates, for the worker to retry; a document this batch
        already committed is ``ready`` under the same storage key and is skipped on the retry."""
        await self._ready(None)
        rows = await self.store.get_documents(bank_id, [w.document_id for w in work])
        docs: list[_InFlight] = []
        for w in work:
            row = rows.get(w.document_id)
            if row is None or row.storage_key != w.storage_key or row.status == "ready":
                logger.info(f"Knowledge ingest skipped: {bank_id}/{w.document_id} was replaced, deleted or is done")
                continue
            media_prefix = f"{bank_storage_prefix(bank_id)}kb/media/{key_segment(w.document_id)}/{uuid.uuid4()}/"
            docs.append(_InFlight(w, row, _title_hint(row), media_prefix))
        if not docs:
            return
        await self.store.set_statuses(
            bank_id, [StatusUpdate(d.work.document_id, "processing", storage_key=d.work.storage_key) for d in docs]
        )
        started = time.perf_counter()
        committed: set[str] = set()
        try:
            waiting: list[_Waiting] = []
            for doc in docs:
                try:
                    drafts = await self._open(bank_id, doc, caller or {}, depth)
                except ValueError as exc:
                    doc.error = exc
                    continue
                if drafts is None:  # split into files, each its own document now, and itself ready
                    doc.split = True
                    committed.add(doc.work.document_id)
                    continue
                try:
                    # A document an embed round refused stops here: its remaining passages (a long
                    # video's windows) would only be cut to be thrown away.
                    while doc.error is None and (draft := await asyncio.to_thread(_next_draft, drafts)) is not None:
                        waiting.append(_Waiting(doc, draft))
                        if len(waiting) >= _EMBED_ROUND:
                            await self._embed_round(waiting)
                            waiting = []
                except ValueError as exc:
                    doc.error = exc
                finally:
                    # Close the splitter now, off the event loop, rather than whenever it is collected.
                    await asyncio.to_thread(_close_drafts, drafts)
            if waiting:
                await self._embed_round(waiting)
            for doc in docs:
                if doc.error is None and not doc.split and not doc.passages:
                    doc.error = media.MediaError("the file holds nothing to search")

            done = [d for d in docs if d.error is None and not d.split]
            # The store commits each one only while it still holds the file this batch read: one replaced
            # or deleted meanwhile is skipped (it used to be re-read here first, which left a gap between
            # the check and the write), and its passages belong to nobody.
            replaced = await self.store.replace_passages(
                bank_id,
                [
                    PassageReplacement(
                        d.work.document_id,
                        d.passages,
                        detect_language("\n".join(p.text or "" for p in d.passages)),
                        storage_key=d.work.storage_key,
                    )
                    for d in done
                ],
            )
            skipped = set(replaced.skipped_document_ids)
            for d in done:
                if d.work.document_id in skipped:
                    await self._delete_keys(d.media_keys)
            kept = [d for d in done if d.work.document_id not in skipped]
            committed |= {d.work.document_id for d in kept}
            await self._delete_keys(replaced.stale_media_keys)
            # Only a document that used to be split can have split-out documents to drop now.
            was_split = [d.work.document_id for d in kept if d.work.was_split]
            if was_split:
                derived = await self.store.derived_documents(bank_id, was_split)
                await self._delete_trees(bank_id, [c for children in derived.values() for c in children])
            failed = [d for d in docs if d.error is not None]
            for d in failed:
                await self._delete_keys(d.media_keys)
            if failed:
                await self.store.set_statuses(
                    bank_id,
                    [
                        StatusUpdate(d.work.document_id, "failed", str(d.error)[:2000], d.work.storage_key)
                        for d in failed
                    ],
                )
            logger.info(
                f"Knowledge ingest {bank_id}: {len(kept)} documents, {sum(len(d.passages) for d in kept)} passages, "
                f"{len(failed)} failed, in {time.perf_counter() - started:.1f}s"
            )
        except Exception as exc:
            # Transient: nothing of what is not committed is kept. Until the retries run out a document is
            # still on its way, so it stays pending with the error noted (it used to be marked failed at
            # once, and clients waiting for ready/failed gave up on it).
            pending = [d for d in docs if d.work.document_id not in committed]
            for d in pending:
                await self._delete_keys(d.media_keys)
            status: DocumentStatus = "pending" if retries_left else "failed"
            note = f"retrying after: {exc!r}" if retries_left else repr(exc)
            await self.store.set_statuses(
                bank_id, [StatusUpdate(d.work.document_id, status, note[:2000], d.work.storage_key) for d in pending]
            )
            raise
        if failed:
            first = failed[0]
            raise media.MediaError(
                f"{len(failed)} of {len(docs)} documents failed; '{first.work.document_id}': {first.error}"
            )

    async def _open(
        self, bank_id: str, doc: _InFlight, caller: dict[str, Any], depth: int
    ) -> Iterator[media.PassageDraft] | None:
        """One document's passages to embed, as a lazy iterator; None when its strategy split it into files."""
        document, w = doc.document, doc.work
        data = await self._engine._file_storage.retrieve(w.storage_key)
        if w.handler is None:
            # Splitting is blocking (pdfium, ffmpeg): it runs in a thread, one draft at a time.
            return await asyncio.to_thread(self._drafts, document, data)
        made = await ingest_strategy.load_extension(w.handler).ingest(
            ingest_strategy.IngestFile(
                bank_id=bank_id,
                document_id=w.document_id,
                filename=document.filename or w.document_id,
                content_type=document.content_type,
                data=data,
                tags=list(document.tags),
            ),
        )
        if made.files is not None:
            await self._write_derived(bank_id, document, w.storage_key, made.files, w.strategy, caller, depth)
            return None
        # A strategy that returned passages has done the splitting already.
        emitted = _drafts_of(made.passages or [])
        for d in emitted:
            self._check_passage_modality(d.modality)
        return iter(emitted)

    async def _embed_round(self, waiting: list[_Waiting]) -> None:
        """Embed one round of passages, from any documents, in one call, and hand each to its document. A
        permanent error is narrowed to the documents it comes from, embedding each one's share alone."""
        live = [w for w in waiting if w.doc.error is None]
        if not live:
            return
        plans = [self._vectors_to_embed(w.draft, w.doc.document, w.doc.title) for w in live]
        try:
            embedded = await self.embedder.embed_documents([e.input for plan in plans for e in plan])
        except ValueError as exc:
            if len({id(w.doc) for w in live}) == 1:
                live[0].doc.error = exc
                return
            # By identity: an _InFlight is a mutable dataclass, so it has no hash to dedupe on.
            for doc in {id(w.doc): w.doc for w in live}.values():
                await self._embed_round([w for w in live if w.doc is doc])
            return
        cursor = 0
        for w, plan in zip(live, plans, strict=True):
            mine = embedded[cursor : cursor + len(plan)]
            cursor += len(plan)
            doc, draft = w.doc, w.draft
            index = len(doc.passages)
            key = None
            if draft.data is not None:
                key = f"{doc.media_prefix}{index}.{draft.extension}"
                await self._engine._file_storage.store(
                    file_data=draft.data, key=key, metadata={"content_type": draft.mime or ""}
                )
                doc.media_keys.append(key)
            doc.passages.append(
                KnowledgePassage(
                    index=index,
                    modality=draft.modality,
                    text=draft.text,
                    page=draft.page,
                    start_ms=draft.start_ms,
                    end_ms=draft.end_ms,
                    media_key=key,
                    embedding=mine[0],
                    extra_vectors=[
                        RoleVector(e.role, e.n, v) for e, v in zip(plan[1:], mine[1:], strict=True) if e.role
                    ],
                )
            )

    async def _write_derived(
        self,
        bank_id: str,
        document: KnowledgeDocument,
        storage_key: str,
        files: list[ingest_strategy.OutputFile],
        strategy: str | None,
        caller: dict[str, Any],
        depth: int,
    ) -> None:
        """Ingest the raw files an extension split ``document`` into, each as its own document: chunked by
        us, or by the same strategy again when it maps their type. They are queued as one operation."""
        names = [f.name for f in files]
        if not files:
            raise media.MediaError("the ingest extension split the file into nothing")
        if not all(names) or len(set(names)) != len(names):
            raise media.MediaError("the ingest extension's file names must be unique and non-empty")
        children: list[_Entry] = []
        for f in files:
            item = IngestItem(
                data=f.data,
                filename=f.filename or f.name,
                content_type=f.content_type,
                tags=list(dict.fromkeys([*document.tags, *f.tags])),
                strategy=strategy,
            )
            try:
                plan = self._check_item(item, depth=depth + 1)
            except KnowledgeBankError as exc:  # the same file would be refused next time too
                raise media.MediaError(f"'{f.name}': {exc}") from exc
            children.append(_Entry(f"{document.document_id}/{f.name}", item, plan, document.document_id))
        current = (await self.store.get_documents(bank_id, [document.document_id])).get(document.document_id)
        if current is None or current.storage_key != storage_key:
            return  # replaced or deleted while the strategy ran
        saved = await self._save(bank_id, children)
        work = [_work(c, s) for c, s in zip(children, saved, strict=True) if s is not None]
        if work:
            await self._queue(bank_id, work, caller, depth + 1)
        derived = (await self.store.derived_documents(bank_id, [document.document_id]))[document.document_id]
        keep = {c.document_id for c in children}
        await self._delete_trees(bank_id, [c for c in derived if c not in keep])
        # The source itself holds no passages: it is ready once its files are queued.
        replaced = await self.store.replace_passages(
            bank_id, [PassageReplacement(document.document_id, [], storage_key=storage_key)]
        )
        await self._delete_keys(replaced.stale_media_keys)
        logger.info(f"Knowledge ingest {bank_id}/{document.document_id}: split into {len(children)} documents")

    async def _delete_trees(self, bank_id: str, document_ids: list[str]) -> list[KnowledgeDocument]:
        """Delete documents, the documents split out of them (to any depth), and their files: one store
        call per level of splitting to find them, then one each for their media and the delete."""
        everything: list[str] = []
        level = document_ids
        while level:
            everything += level
            derived = await self.store.derived_documents(bank_id, level)
            level = [c for children in derived.values() for c in children]
        if not everything:
            return []
        keys = await self.store.media_keys(bank_id, everything)
        deleted = await self.store.delete_documents(bank_id, everything)
        await self._delete_keys([*keys, *(d.storage_key for d in deleted if d.storage_key)])
        return deleted

    # --- read -------------------------------------------------------------------------------

    async def list_documents(
        self,
        bank_id: str,
        request_context: RequestContext,
        *,
        limit: int,
        offset: int,
        query: str | None,
        status: str | None,
        media_type: str | None,
        tags: list[str] | None,
    ) -> DocumentPage:
        await self._open_bank(bank_id, KnowledgeBankReadOperation.LIST_KNOWLEDGE_BANK_DOCUMENTS, request_context)
        return await self.store.list_documents(
            bank_id, limit=limit, offset=offset, query=query, status=status, media_type=media_type, tags=tags
        )

    async def get_document(self, bank_id: str, document_id: str, request_context: RequestContext) -> DocumentDetail:
        await self._open_bank(bank_id, KnowledgeBankReadOperation.GET_KNOWLEDGE_BANK_DOCUMENT, request_context)
        document = (await self.store.get_documents(bank_id, [document_id])).get(document_id)
        if document is None:
            raise KnowledgeBankError(f"document '{document_id}' not found", 404)
        passages = (await self.store.list_passages(bank_id, [document_id]))[document_id]
        return DocumentDetail(document=document, passages=passages)

    async def delete_document(self, bank_id: str, document_id: str, request_context: RequestContext) -> None:
        await self._open_bank(bank_id, KnowledgeBankWriteOperation.DELETE_KNOWLEDGE_BANK_DOCUMENT, request_context)
        if not any(d.document_id == document_id for d in await self._delete_trees(bank_id, [document_id])):
            raise KnowledgeBankError(f"document '{document_id}' not found", 404)

    async def open_file(self, bank_id: str, document_id: str, request_context: RequestContext) -> OpenedFile:
        """The original upload: its byte stream, content type and filename."""
        await self._open_bank(bank_id, KnowledgeBankReadOperation.GET_KNOWLEDGE_BANK_DOCUMENT, request_context)
        document = (await self.store.get_documents(bank_id, [document_id])).get(document_id)
        if document is None or not document.storage_key:
            raise KnowledgeBankError(f"document '{document_id}' not found", 404)
        stream = self._engine._file_storage.retrieve_stream(document.storage_key)
        return OpenedFile(stream=stream, content_type=document.content_type, filename=document.filename)

    async def open_passage_media(
        self, bank_id: str, document_id: str, index: int, request_context: RequestContext
    ) -> OpenedFile:
        await self._open_bank(bank_id, KnowledgeBankReadOperation.GET_KNOWLEDGE_BANK_DOCUMENT, request_context)
        found = (await self.store.passage_ranges(bank_id, [(document_id, index, index)]))[(document_id, index, index)]
        passage = found[0] if found else None
        if passage is None or not passage.media_key:
            raise KnowledgeBankError("this passage has no media", 404)
        ext = passage.media_key.rsplit(".", 1)[-1]
        content_type = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp", "m4a": "audio/mp4"}.get(
            ext, "video/mp4" if ext == "mp4" else "application/octet-stream"
        )
        return OpenedFile(
            stream=self._engine._file_storage.retrieve_stream(passage.media_key), content_type=content_type
        )

    async def timeseries(
        self, bank_id: str, period: str, trunc: str, request_context: RequestContext
    ) -> list[TimeBucket]:
        await self._open_bank(bank_id, KnowledgeBankReadOperation.GET_KNOWLEDGE_BANK, request_context)
        days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}[period]
        since = datetime.now(UTC) - timedelta(days=days)
        return await self.store.timeseries(bank_id, since=since, trunc=trunc)

    # --- search -----------------------------------------------------------------------------

    async def search(
        self, bank_id: str, request: SearchRequest, request_context: RequestContext
    ) -> list[SearchResultItem]:
        await self._open_bank(bank_id, KnowledgeBankReadOperation.SEARCH_KNOWLEDGE_BANK, request_context)
        return await self._with_context(bank_id, await self._search(bank_id, request), request.neighbors)

    async def _search(self, bank_id: str, request: SearchRequest) -> list[SearchHit]:
        pattern = request.pattern if (request.pattern or "").strip() else None
        has_query = bool((request.query or "").strip()) or request.query_image is not None
        if not has_query and pattern is None:
            raise KnowledgeBankError("search needs a query, a query_image or a pattern")
        try:
            if not has_query and pattern is not None:
                return await self.store.grep(
                    bank_id, pattern, top_k=request.top_k, filters=request.filters, regex=request.regex
                )
            try:
                vector = await self.embedder.embed_query(request.query, request.query_image)
            except UnsupportedModalityError as exc:
                raise KnowledgeBankError(str(exc), 415) from exc
            return await self.store.search(
                bank_id,
                vector,
                top_k=request.top_k,
                filters=request.filters,
                text_weight=get_config().knowledge_pdf_text_weight,
                query_text=request.query,
                keyword_weight=get_config().knowledge_keyword_weight,
                pattern=pattern,
                regex=request.regex,
            )
        except asyncpg.InvalidRegularExpressionError as exc:
            raise KnowledgeBankError(f"invalid regular expression: {exc}") from exc

    async def _with_context(self, bank_id: str, hits: list[SearchHit], neighbors: int) -> list[SearchResultItem]:
        """Expand text hits to their neighbouring passages, merging overlapping windows per document."""
        items: list[SearchResultItem] = []
        windows: dict[str, list[SearchResultItem]] = {}
        for hit in hits:
            if neighbors <= 0 or hit.passage.modality != "text":
                items.append(SearchResultItem(hit))
                continue
            first, last = max(0, hit.passage.index - neighbors), hit.passage.index + neighbors
            same_doc = windows.setdefault(hit.document.document_id, [])
            merged = next(
                (w for w in same_doc if first <= (w.context_last or 0) + 1 and last >= (w.context_first or 0) - 1),
                None,
            )
            if merged is not None:  # a better hit already covers (or touches) this one
                merged.context_first = min(merged.context_first or 0, first)
                merged.context_last = max(merged.context_last or 0, last)
                # Grown, it may now reach another window of the document: fold that one in too, or the
                # passages between them would come back twice.
                for other in [w for w in same_doc if w is not merged]:
                    if (other.context_first or 0) <= (merged.context_last or 0) + 1 and (other.context_last or 0) >= (
                        merged.context_first or 0
                    ) - 1:
                        merged.context_first = min(merged.context_first or 0, other.context_first or 0)
                        merged.context_last = max(merged.context_last or 0, other.context_last or 0)
                        same_doc.remove(other)
                        items.remove(other)
                continue
            item = SearchResultItem(hit, context_first=first, context_last=last)
            same_doc.append(item)
            items.append(item)
        windowed = [i for i in items if i.context_first is not None and i.context_last is not None]
        # Every window's passages in one store call, not one per hit.
        ranges = await self.store.passage_ranges(
            bank_id, [(i.hit.document.document_id, i.context_first or 0, i.context_last or 0) for i in windowed]
        )
        for item in windowed:
            passages = ranges[(item.hit.document.document_id, item.context_first or 0, item.context_last or 0)]
            item.context = "\n\n".join(p.text or "" for p in passages)
            item.context_first = passages[0].index if passages else item.context_first
            item.context_last = passages[-1].index if passages else item.context_last
        return items

    async def search_linked(
        self, bank_ids: list[str], query: str, top_k: int, request_context: RequestContext
    ) -> list[LinkedHit]:
        """Search the knowledge banks a memory bank links, best passages first across all of them."""
        await self._ready(request_context)
        hits: list[LinkedHit] = []
        vector: list[float] | None = None
        for bank_id in bank_ids:
            try:
                await self._authorize(bank_id, KnowledgeBankReadOperation.SEARCH_KNOWLEDGE_BANK, request_context)
            except OperationValidationError as exc:
                # The memory bank links it, but this caller may not read it: the reflect goes on without it.
                logger.info(f"Linked knowledge bank '{bank_id}' refused for this caller ({exc.reason}); skipped")
                continue
            if await self._kind_of(bank_id) != KIND:
                logger.warning(f"Linked knowledge bank '{bank_id}' does not exist; skipped")
                continue
            if vector is None:
                vector = await self.embedder.embed_query(query)
            for hit in await self.store.search(
                bank_id,
                vector,
                top_k=top_k,
                filters=SearchFilters(),
                # no keyword arm: fused scores are rescaled per bank and could not be ranked across banks
                text_weight=get_config().knowledge_pdf_text_weight,
            ):
                hits.append(LinkedHit(bank_id=bank_id, hit=hit))
        hits.sort(key=lambda linked: linked.hit.score, reverse=True)
        return hits[:top_k]


async def documents_for_memory_bank(
    engine: MemoryEngine, bank_id: str, query: str, request_context: RequestContext
) -> list[KnowledgeDocumentHit] | None:
    """The linked knowledge banks' best passages for reflect's recall tool, or None when none are linked.

    A failure here never fails the reflect: the memories are still the answer, without the docs.
    """
    if not get_config().enable_knowledge_banks:
        return None
    config = await engine._config_resolver.get_bank_config(bank_id, request_context)
    linked = [b for b in (config.get("knowledge_banks") or []) if b != bank_id]
    if not linked or not (query or "").strip():
        return None
    # Reflect's recall runs as an internal operation, but the linked banks are read on the caller's
    # behalf: the validator must see the caller, not a background job it would let through.
    caller = replace(request_context, internal=False)
    try:
        hits = await get_knowledge_service(engine).search_linked(linked, query, LINKED_RECALL_TOP_K, caller)
    except Exception as exc:
        logger.warning(f"Knowledge search for memory bank '{bank_id}' failed: {exc!r}")
        return []
    from .routes import search_result

    out = []
    for linked in hits:
        r = search_result(linked.bank_id, linked.hit)
        out.append(
            KnowledgeDocumentHit(
                bank_id=linked.bank_id,
                document_id=r.document_id,
                passage_index=r.index,
                filename=r.filename,
                media_type=r.media_type,
                text=r.text,
                page=r.page,
                start_ms=r.start_ms,
                end_ms=r.end_ms,
                score=r.score,
                passage_url=r.passage_url,
                document_url=r.document_url,
                tags=r.tags,
            )
        )
    return out


def _title_hint(document: KnowledgeDocument) -> str | None:
    """The file name without its extension: the embedder's "title: ..." prompt slot. None for the name
    a text write gets when it has none (``<document_id>.md``): an id, often a UUID, is not a title. A
    file named like its id (``report.pdf`` as ``report``) keeps it: the title is worth 2-10 nDCG@10."""
    if not document.filename or document.filename == f"{document.document_id}.md":
        return None
    stem = Path(document.filename).stem
    return re.sub(r"[_\-]+", " ", stem).strip()  # "footstep_gravel_03" reads as words


def _drafts_of(passages: list[ingest_strategy.OutputPassage]) -> list[media.PassageDraft]:
    drafts = []
    for n, p in enumerate(passages):
        if (p.text is None) == (p.data is None):
            raise media.MediaError(f"the ingest extension's passage {n} needs exactly one of text or data")
        drafts.append(
            media.PassageDraft(
                modality=p.modality,
                text=p.text,
                data=p.data,
                mime=p.mime,
                page=p.page,
                start_ms=p.start_ms,
                end_ms=p.end_ms,
            )
        )
    return drafts


def _close_drafts(drafts: Iterator[media.PassageDraft]) -> None:
    close = getattr(drafts, "close", None)  # generators release the PDF / media they hold
    if close is not None:
        close()


def _next_draft(drafts: Iterator[media.PassageDraft]) -> media.PassageDraft | None:
    return next(drafts, None)


_services: weakref.WeakKeyDictionary[MemoryEngine, KnowledgeService] = weakref.WeakKeyDictionary()


# A threading lock, not an asyncio one: the map is process-wide and the critical section never awaits.
_services_lock = threading.Lock()


def get_knowledge_service(engine: MemoryEngine) -> KnowledgeService:
    """One service per engine (one embedder, loaded once), created on first use."""
    with _services_lock:
        service = _services.get(engine)
        if service is None:
            service = _services[engine] = KnowledgeService(engine)
        return service


async def run_ingest_task(engine: MemoryEngine, task: dict[str, Any]) -> None:
    retries_left = task.get("_retry_count", 0) < get_config().worker_max_retries  # see execute_task's retry
    await get_knowledge_service(engine).process_batch(
        task["bank_id"],
        [_Work.from_payload(d) for d in task["documents"]],
        retries_left=retries_left,
        depth=task.get("depth", 0),
        # The files a strategy splits these into are queued on behalf of the same caller.
        caller={k: task[k] for k in ("_tenant_id", "_api_key_id") if task.get(k)},
    )
