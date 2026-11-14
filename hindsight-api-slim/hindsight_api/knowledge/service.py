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
import itertools
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from ..config import get_config
from ..engine import bank_aliases
from ..engine.db_utils import acquire_with_retry
from ..engine.operation_metadata import KnowledgeIngestChildMetadata, KnowledgeIngestParentMetadata
from ..engine.response_models import KnowledgeDocumentHit
from ..engine.retain import bank_utils
from ..engine.retain import timing as _timing
from ..engine.schema import fq_table
from ..engine.storage import bank_storage_prefix, key_segment
from ..engine.token_encoding import count_tokens
from ..extensions.operation_validator import (
    BankListScope,
    CreateKnowledgeBankContext,
    KnowledgeBankListContext,
    KnowledgeBankReadContext,
    KnowledgeBankReadOperation,
    KnowledgeBankWriteContext,
    KnowledgeBankWriteOperation,
)
from ..tracing import inject_task_trace_context
from . import bank_config, ingest_strategy, media
from .bank_config import KnowledgeBankOverrides, KnowledgeBankSettings
from .embedders import (
    EmbedInput,
    KnowledgeEmbedder,
    UnsupportedModalityError,
    create_knowledge_embedder,
)
from .store import (
    BankStats,
    DocumentPage,
    DocumentStatus,
    KnowledgeDocument,
    KnowledgePassage,
    KnowledgeStore,
    PassageReplacement,
    PassageStage,
    RoleVector,
    SearchFilters,
    SearchHit,
    StatusUpdate,
    StoreSearchTrace,
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
_PIPELINE_VERSION = 10  # 10: PDF pages also get sentence vectors; 9: units pack whole lines
# How many passages are embedded and stored per round, so a 2h video never sits in memory whole.
_EMBED_ROUND = 16
#: Files an ingest saves at once.
_FILE_SAVE_CONCURRENCY = 16
#: Passage files (page images, clips) an ingest writes at once, in the background of its embed rounds.
#: Lower than the upload's: these run alongside the rounds' vector commits, and 16 at once stalled a
#: local Postgres long enough for its other queries to time out.
_PASSAGE_FILE_CONCURRENCY = 8
#: Embed rounds in flight at once while the next passages are cut. One round at a time left the
#: splitter, the embedder and the file writes taking turns: a 1,110-page PDF batch spent ~40 s
#: rendering, ~40 s writing page images one by one and the embed calls, end to end, never at once.
_EMBED_ROUNDS_IN_FLIGHT = 4
#: A document with at least this many passages commits on its own as soon as they are embedded.
_EARLY_COMMIT_PASSAGES = 2 * _EMBED_ROUND
#: Passages that commit together: a batch's finished documents are written in groups of about this many
#: as they finish, not all at the end, so what a batch holds does not grow with how many documents it has.
_COMMIT_GROUP_PASSAGES = 256
#: What one vector is estimated to cost before it is embedded (768 dimensions as a Python list of floats,
#: ~25 KB); corrected to its real size once it is.
_VECTOR_BYTES_ESTIMATE = 25_000
#: Vectors a passage with a file (a PDF page, a clip) is estimated to carry before it is embedded: a page is
#: its image, bands, text units and sentences.
_FILE_PASSAGE_VECTORS_ESTIMATE = 24
#: Documents in one child of an async ingest, however small: a child is one store read, status write
#: and passage write, and many tiny documents would make those writes the batch.
_CHILD_DOCUMENTS = 100


#: Documents a bank delete removes per transaction, with their passages and vectors (about 7 and 30
#: each on a web corpus), so no single statement nears the command timeout.
DELETE_BATCH_DOCUMENTS = 500
#: How long a bank delete waits on documents a running ingest still holds before the banks row's
#: cascade takes what is left.
DELETE_WAIT_SECONDS = 60


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


class _FileGone(Exception):
    """The document's stored original is not there: replaced by a newer upload, or lost."""


@dataclass
class SearchRequest:
    query: str | None
    query_image: bytes | None
    filters: SearchFilters
    # What the hits may weigh in all, the text around them included: see ``_within_budget``.
    max_tokens: int = 4096
    # How deep the search ranks before the token budget picks from it (``SEARCH_CANDIDATES``).
    budget: SearchBudget = "mid"
    # An MP4 clip to search with, alone or with the text and the image: all of them make one vector.
    query_video: bytes | None = None


SearchBudget = Literal["low", "mid", "high"]
#: Candidates a search ranks before ``max_tokens`` picks from them, by ``budget``, as memory-bank recall
#: names its effort: low is fastest, mid the old ``top_k`` ceiling, high reaches deepest (past the HNSW
#: ``ef_search`` of 200, so its candidate scan iterates and is slower).
SEARCH_CANDIDATES: dict[SearchBudget, int] = {"low": 50, "mid": 200, "high": 500}
#: What a hit with no text of its own (an image, a page's picture, a clip) costs the budget, on top of
#: any text it carries: its media is a URL and a few fields to whoever reads the results.
MEDIA_HIT_TOKENS = 64
#: The most passages one text result holds: the hit and two on each side. Leftover budget widens results
#: up to it, once every hit that fits has its own. Results used to also grow to take in a nearby lower
#: hit, without limit: in a small document result #1 became the whole document, and later hits came
#: back with no text around them.
MAX_CONTEXT_PASSAGES = 5


@dataclass
class _Window:
    """A text hit's returned run of passages, ``first..last`` of its document."""

    item: SearchResultItem
    document_id: str
    first: int
    last: int
    #: A side stops growing at the document's edge or at another hit's window.
    grows_before: bool = True
    grows_after: bool = True


@dataclass
class SearchResultItem:
    hit: SearchHit
    #: A text hit's run of passages, which the API returns as the hit's ``text``.
    context: str | None = None
    #: Which passages that run spans. Not in the API (its ``index`` names the matched passage); kept so
    #: the budget's windows can be checked directly.
    context_first: int | None = None
    context_last: int | None = None


@dataclass
class TracedHit:
    """A ranked candidate and what the token budget did with it."""

    hit: SearchHit
    #: "returned": a result of its own; "merged": inside a better hit's passages (``merged_into``);
    #: "over_budget": nothing of it fit in what was left of ``max_tokens``.
    outcome: Literal["returned", "merged", "over_budget"]
    merged_into: SearchHit | None = None


BudgetAction = Literal["window", "alone", "media", "merged", "widened", "skipped"]


@dataclass
class BudgetStep:
    """One step of filling ``max_tokens``, in the order the budget took it."""

    #: window: a text hit with its free neighbours (a passage each side no result holds yet); alone: a text hit with no room for neighbours;
    #: media: an image, page or clip; merged: a hit already inside a better hit's result (nothing grows);
    #: widened: leftover budget, one passage added to a result's side (up to MAX_CONTEXT_PASSAGES);
    #: skipped: nothing of the hit fit.
    action: BudgetAction
    hit: SearchHit
    document_id: str
    first: int
    last: int
    tokens: int
    total: int
    #: For merged: the hit whose result holds it.
    into: SearchHit | None = None


@dataclass
class SearchTrace:
    """What one search did, for a caller that asks (the API's ``trace: true``)."""

    store: StoreSearchTrace = field(default_factory=StoreSearchTrace)
    #: Every ranked candidate, best first, with its outcome.
    hits: list[TracedHit] = field(default_factory=list)
    tokens_used: int = 0
    budget_steps: list[BudgetStep] = field(default_factory=list)
    #: Each phase's wall time (open_bank, embed, store.search, context), in seconds.
    seconds: dict[str, float] = field(default_factory=dict)


@dataclass
class KnowledgeBankConfig:
    """A bank's ingest settings as it splits files with them, and which of them it overrides."""

    bank_id: str
    settings: KnowledgeBankSettings
    overrides: KnowledgeBankOverrides


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
    #: The stored file's size, which sizes the child operations an async ingest is split into. Not in
    #: the payload: a worker has the file itself.
    size: int = 0

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
    return _Work(
        entry.document_id,
        saved.storage_key,
        entry.plan.strategy,
        entry.plan.handler,
        saved.was_split,
        len(entry.item.data),
    )


def _vector_bytes(vector: list[float]) -> int:
    return 56 + 32 * len(vector)  # the list, plus a pointer and a float object per dimension


def _passage_estimate(draft: media.PassageDraft) -> int:
    vectors = _FILE_PASSAGE_VECTORS_ESTIMATE if draft.data is not None else 1
    return len(draft.data or b"") + len(draft.text or "") + vectors * _VECTOR_BYTES_ESTIMATE


def _passage_bytes(draft: media.PassageDraft, vectors: list[list[float]]) -> int:
    return len(draft.data or b"") + len(draft.text or "") + sum(_vector_bytes(v) for v in vectors)


class _MemoryBudget:
    """The bytes of cut-but-uncommitted passages (their text, file and vectors) one batch may hold, as
    retain bounds its extracted-but-unwritten facts: the splitter reserves a passage's cost before queueing
    it, and a document's commit gives its share back. Under the budget nothing waits; at it the splitter
    does, which is the batch throttling itself rather than the worker being killed for its memory.

    It cannot deadlock: before waiting, the splitter starts what can free room (the queued round, the
    finished documents' commit), and a reservation is admitted whatever its size when everything held is
    the document being cut, since that document must be held whole to commit (its passages are replaced
    in one write). A limit of 0 turns the budget off."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.held = 0
        self.peak = 0
        self._freed = asyncio.Event()
        self._failed: BaseException | None = None

    def _fits(self, n: int, own: int) -> bool:
        return self.limit <= 0 or self.held + n <= self.limit or self.held <= own

    async def reserve(self, n: int, own: int, make_room: Callable[[], Awaitable[None]]) -> None:
        if not self._fits(n, own):
            await make_room()
            async with _timing.timed("split.wait_memory"):
                while not self._fits(n, own):
                    if self._failed is not None:
                        raise self._failed
                    self._freed.clear()
                    await self._freed.wait()
        self.adjust(n)

    def fail(self, exc: BaseException) -> None:
        """A round failed: what it held will not be committed, so a splitter waiting for room stops."""
        self._failed = exc
        self._freed.set()

    def adjust(self, delta: int) -> None:
        self.held += delta
        self.peak = max(self.peak, self.held)
        if delta < 0:
            self._freed.set()


def _children(work: list[_Work], budget: int) -> list[list[_Work]]:
    """Pack an async ingest's documents, in order, into children of about ``budget`` bytes and at most
    ``_CHILD_DOCUMENTS`` documents. A document is never split across children, and one over the budget
    is a child on its own: it is processed in one batch either way."""
    out: list[list[_Work]] = []
    size = 0
    for w in work:
        if not out or size + w.size > budget or len(out[-1]) >= _CHILD_DOCUMENTS:
            out.append([])
            size = 0
        out[-1].append(w)
        size += w.size
    return out


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
    #: Passages are queued in order and embedded by rounds that may finish out of order: each is
    #: given its index when queued and lands here, and ``passages`` is put together once all are in.
    queued: int = 0
    done: dict[int, KnowledgePassage] = field(default_factory=dict)
    #: The embed rounds holding this document's passages, and whether its commit has started.
    rounds: list[asyncio.Task[None]] = field(default_factory=list)
    committing: bool = False
    #: Its files' background writes (see ``_embed_round``).
    saves: list[asyncio.Task[None]] = field(default_factory=list)
    #: Bytes it holds against the batch's memory budget, given back once it is committed; and how many
    #: passages it committed, kept for the batch's log line after its passages are dropped.
    reserved: int = 0
    committed_passages: int = 0
    #: Passages already staged for its next version (a document too big to hold whole goes in pieces), and
    #: whether the store skipped it as replaced meanwhile.
    staged: int = 0
    replaced: bool = False


@dataclass
class _Waiting:
    """A passage waiting in the current embed round."""

    doc: _InFlight
    draft: media.PassageDraft
    index: int
    #: What it reserved against the memory budget when queued: an estimate, corrected once embedded.
    cost: int = 0


#: Writes one passage's file for a document in the background (see ``KnowledgeService._embed_round``).
_SaveFile = Callable[[_InFlight, str, media.PassageDraft], Awaitable[None]]


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
        *,
        needs_vectors: bool = True,
    ) -> None:
        # ``needs_vectors=False`` for an operation that never reads or writes a vector (a delete): the
        # store's first-use setup asks the embedder for its size, and an unreachable embedder must not
        # be what stops a bank from being deleted.
        if needs_vectors:
            await self._ready(request_context)
        else:
            await self._engine._authenticate_tenant(request_context)
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
        await self._open_bank(
            bank_id, KnowledgeBankWriteOperation.DELETE_KNOWLEDGE_BANK, request_context, needs_vectors=False
        )
        # Its queued ingests are cancelled first (the worker no longer claims them), then its documents
        # go a batch per transaction: the banks row cascades to kb_documents -> kb_passages -> vectors,
        # and on a 100k-document bank that one cascading statement ran past the command timeout
        # (BrowseComp-Plus, ~680k passages) while ingests kept writing to the same rows. An ingest
        # already running does not check for cancellation: it finishes its current write, and the
        # documents it holds are skipped until their locks free, for up to DELETE_WAIT_SECONDS; what
        # is left then goes with the banks row.
        async with acquire_with_retry(await self._engine._get_backend()) as conn:
            await conn.execute(
                f"UPDATE {fq_table('async_operations')} SET status = 'cancelled', updated_at = now() "
                "WHERE bank_id = $1 AND status IN ('pending', 'processing')",
                bank_id,
            )
        deadline = time.monotonic() + DELETE_WAIT_SECONDS
        while time.monotonic() < deadline:
            if await self.store.delete_some_documents(bank_id, DELETE_BATCH_DOCUMENTS):
                continue
            if not (await self.store.list_documents(bank_id, limit=1, offset=0)).documents:
                break
            await asyncio.sleep(1)
        await self._engine.delete_bank(bank_id, request_context=request_context, _knowledge_bank=True)
        await self._engine._file_storage.delete_prefix(f"{bank_storage_prefix(bank_id)}kb/")

    # --- per-bank settings ------------------------------------------------------------------

    async def _overrides(self, bank_id: str) -> KnowledgeBankOverrides:
        async with acquire_with_retry(await self._engine._get_backend()) as conn:
            raw = await conn.fetchval(
                f"SELECT config FROM {fq_table('banks')} WHERE bank_id = $1 AND kind = $2", bank_id, KIND
            )
        return bank_config.stored_overrides(raw)

    async def _bank_settings(self, bank_id: str) -> KnowledgeBankSettings:
        """The settings an ingest splits this bank's files with: read once per request or batch."""
        return bank_config.resolve(await self._overrides(bank_id))

    def _bank_config(self, bank_id: str, overrides: KnowledgeBankOverrides) -> KnowledgeBankConfig:
        return KnowledgeBankConfig(bank_id, bank_config.resolve(overrides), overrides)

    async def get_bank_config(self, bank_id: str, request_context: RequestContext) -> KnowledgeBankConfig:
        await self._open_bank(bank_id, KnowledgeBankReadOperation.GET_KNOWLEDGE_BANK_CONFIG, request_context)
        return self._bank_config(bank_id, await self._overrides(bank_id))

    async def update_bank_config(
        self, bank_id: str, updates: dict[str, Any], request_context: RequestContext
    ) -> KnowledgeBankConfig:
        """Merge ``updates`` into the bank's overrides (None clears one). The row is locked while the
        merged overrides are checked and written, so two concurrent edits cannot each pass the
        overlap-below-window check against a value the other is replacing."""
        await self._open_bank(bank_id, KnowledgeBankWriteOperation.UPDATE_KNOWLEDGE_BANK_CONFIG, request_context)
        async with acquire_with_retry(await self._engine._get_backend()) as conn, conn.transaction():
            raw = await conn.fetchval(
                f"SELECT config FROM {fq_table('banks')} WHERE bank_id = $1 AND kind = $2 FOR UPDATE", bank_id, KIND
            )
            try:
                overrides = bank_config.validate_overrides(bank_config.stored_overrides(raw), updates)
            except ValueError as exc:
                raise KnowledgeBankError(str(exc)) from exc
            await self._write_overrides(conn, bank_id, overrides)
        return self._bank_config(bank_id, overrides)

    async def reset_bank_config(self, bank_id: str, request_context: RequestContext) -> KnowledgeBankConfig:
        await self._open_bank(bank_id, KnowledgeBankWriteOperation.RESET_KNOWLEDGE_BANK_CONFIG, request_context)
        overrides = KnowledgeBankOverrides()
        async with acquire_with_retry(await self._engine._get_backend()) as conn:
            await self._write_overrides(conn, bank_id, overrides)
        return self._bank_config(bank_id, overrides)

    async def _write_overrides(self, conn: Any, bank_id: str, overrides: KnowledgeBankOverrides) -> None:
        # Only this bank's knowledge keys are replaced: anything else in the column is left as it is.
        await conn.execute(
            f"""
            UPDATE {fq_table("banks")}
            SET config = (COALESCE(config, '{{}}'::jsonb) - $1::text[]) || $2::jsonb, updated_at = now()
            WHERE bank_id = $3 AND kind = $4
            """,
            [bank_config.STORED_PREFIX + f for f in bank_config.OVERRIDABLE_FIELDS],
            json.dumps(bank_config.to_stored(overrides)),
            bank_id,
            KIND,
        )

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

    def _check_item(self, item: IngestItem, settings: KnowledgeBankSettings, *, depth: int = 0) -> _Plan:
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
        self._check_kind(kind, item, settings)
        return _Plan(kind=kind, strategy=strategy.name if strategy else None)

    def _check_kind(
        self, kind: media.Kind | None, item: IngestItem | KnowledgeDocument, settings: KnowledgeBankSettings
    ) -> media.Kind:
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
        if media_type in settings.disabled_media_types:
            raise KnowledgeBankError(
                f"'{item.filename}' is {media_type}, which this bank does not ingest (its disabled_media_types)",
                415,
            )
        if not _NEEDS[kind] & self.embedder.modalities:
            raise KnowledgeBankError(
                f"'{item.filename}' is {media_type}, but the knowledge embedder only embeds "
                f"{', '.join(sorted(self.embedder.modalities))}. Configure HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_PROVIDER "
                "with a model that embeds it.",
                415,
            )
        return kind

    def _check_passage_modality(self, modality: str, settings: KnowledgeBankSettings) -> None:
        """A passage an ingest extension emitted: something this server embeds and ingests."""
        if modality not in self.embedder.modalities:
            raise media.MediaError(f"the ingest extension emitted {modality} passages, which the embedder cannot embed")
        if modality in settings.disabled_media_types:
            raise media.MediaError(f"the ingest extension emitted {modality} passages, which this bank does not ingest")

    def _content_hash(self, item: IngestItem, plan: _Plan, s: KnowledgeBankSettings) -> str:
        # The splitting settings (the bank's) are part of the hash: rewriting the same file after they
        # change must re-split it, not be skipped as unchanged. So is a strategy, its settings and version.
        c = get_config()
        how: str = plan.kind or ""
        if plan.handler is not None:
            version = ingest_strategy.load_extension(plan.handler).version
            how = f"strategy:{plan.strategy}:{plan.handler.key}:{version}"
        settings = (
            f"v{_PIPELINE_VERSION}|{how}|{self.embedder.provider}:{self.embedder.model}:{self.embedder.dimension}|{s.pdf_page_pixels}"
            f"|{s.image_tokens}|{s.audio_window_seconds}|{s.video_window_seconds}"
            f"|{s.media_overlap_seconds}|{c.knowledge_pdf_text_weight}|{s.pdf_bands}|{s.text_passage_tokens}"
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
        settings = await self._bank_settings(bank_id)
        # All-or-nothing: check every item before storing any.
        entries = [
            _Entry(item.document_id or str(uuid.uuid4()), item, self._check_item(item, settings)) for item in items
        ]
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
        saved = await self._save(bank_id, entries, settings)
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
        """Queue an async ingest as one parent operation and its children, and return the parent's id.

        Always a parent, even for one child, so an async ingest has one shape to track: the parent has no
        task of its own and completes (or fails) once every child has, like a batch retain's. Each child
        is a worker task over a slice of the documents (see ``_children``), so a big upload spreads across
        free slots, and a retry redoes one child, not the whole upload."""
        children = _children(work, get_config().knowledge_ingest_child_bytes)
        parent_id = uuid.uuid4()
        payloads: list[dict[str, Any]] = []
        backend = await self._engine._get_backend()
        table = fq_table("async_operations")
        # Parent and children in one transaction, as a batch retain's are: a parent written without its
        # children would be pending forever (no task runs it, no child ever completes it).
        async with acquire_with_retry(backend) as conn:
            async with conn.transaction():
                if await conn.fetchval(f"SELECT 1 FROM {fq_table('banks')} WHERE bank_id = $1", bank_id) is None:
                    raise KnowledgeBankError(f"knowledge bank '{bank_id}' not found", status_code=404)
                await conn.execute(
                    f"INSERT INTO {table} (operation_id, bank_id, operation_type, result_metadata, status) "
                    "VALUES ($1, $2, $3, $4, $5)",
                    parent_id,
                    bank_id,
                    TASK_TYPE,
                    json.dumps(
                        KnowledgeIngestParentMetadata(
                            document_ids=[w.document_id for w in work], num_sub_batches=len(children)
                        ).to_dict()
                    ),
                    "pending",
                )
                for i, child in enumerate(children, 1):
                    child_id = uuid.uuid4()
                    payload: dict[str, Any] = {
                        "type": TASK_TYPE,
                        "operation_id": str(child_id),
                        "bank_id": bank_id,
                        "documents": [w.to_payload() for w in child],
                        **caller,
                    }
                    if depth:
                        payload["depth"] = depth
                    inject_task_trace_context(payload)
                    meta = KnowledgeIngestChildMetadata(
                        parent_operation_id=str(parent_id),
                        sub_batch_index=i,
                        total_sub_batches=len(children),
                        document_ids=[w.document_id for w in child],
                    )
                    await conn.execute(
                        f"INSERT INTO {table} (operation_id, bank_id, operation_type, result_metadata, status, "
                        "task_payload) VALUES ($1, $2, $3, $4, $5, $6::jsonb)",
                        child_id,
                        bank_id,
                        TASK_TYPE,
                        json.dumps(meta.to_dict()),
                        "pending",
                        json.dumps(payload),
                    )
                    payloads.append(payload)
        if len(children) > 1:
            logger.info(
                f"Knowledge ingest {bank_id}: {len(work)} documents queued as {len(children)} child operations "
                f"of {parent_id}"
            )
        # After the commit: a synchronous task backend (tests) runs each task right here, against rows
        # it must be able to see.
        for payload in payloads:
            await self._engine._task_backend.submit_task(payload)
        return str(parent_id)

    async def _save(self, bank_id: str, entries: list[_Entry], settings: KnowledgeBankSettings) -> list[_Saved | None]:
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
            content_hash = self._content_hash(item, e.plan, settings)
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
        if not keys:
            return
        async with _timing.timed("file.delete"):
            for key in keys:
                try:
                    await self._engine._file_storage.delete(key)
                except Exception as exc:  # an orphaned blob is harmless; a failed write over it is not
                    logger.warning(f"Could not delete knowledge file {key}: {exc}")
        _timing.record("file.delete", 0.0, calls=len(keys) - 1)  # one round trip per key

    # --- ingest (worker) --------------------------------------------------------------------

    def _drafts(
        self, document: KnowledgeDocument, data: bytes, s: KnowledgeBankSettings
    ) -> Iterator[media.PassageDraft]:
        kind = media.classify(document.content_type, document.filename)
        if kind == "pdf":
            return media.split_pdf(
                data,
                page_pixels=s.pdf_page_pixels,
                max_pages=s.max_pdf_pages,
                bands=s.pdf_bands,
            )
        if kind == "image":
            return iter(media.split_image(data))
        if kind in ("audio", "video"):
            window = s.audio_window_seconds if kind == "audio" else s.video_window_seconds
            return media.split_media(
                data,
                kind,
                filename=document.filename,
                window_seconds=window,
                overlap_seconds=s.media_overlap_seconds,
                max_seconds=s.max_media_seconds,
            )
        text = media.read_text(data, document.filename)
        return iter(media.split_text(text, tokens=s.text_passage_tokens))

    def _vectors_to_embed(
        self,
        draft: media.PassageDraft,
        document: KnowledgeDocument,
        title: str | None,
        image_tokens: int | None = None,
    ) -> list[_Embedding]:
        """The vectors one passage is stored as. A PDF page (PDF_GEMMA.md): the whole page image at 560
        vision tokens with the file name, three overlapping horizontal bands at 280, the page text's 128-token
        units and its sentences, all titled with the file name (a text-only model also gets the whole page
        text, as the primary vector). Anything else: one vector, with the asset's name and tags in the same
        input (GAME_ASSETS.md)."""
        if (
            draft.page is not None
            and document.media_type == "document"
            and draft.modality == "image"
            and "image" not in self.embedder.modalities
        ):
            # A text-only model: the page is found by its text layer alone (a scanned page by its title).
            page_text = draft.embed_text or draft.text or ""
            units = media.pack_lines(page_text, tokens=media.PDF_TEXT_UNIT_TOKENS) if page_text else []
            return (
                [_Embedding(None, 0, EmbedInput(modality="text", text=page_text, title=title))]
                + [_Embedding("unit", n, EmbedInput(modality="text", text=u, title=title)) for n, u in enumerate(units)]
                + (_sentence_vectors(page_text, title) if page_text else [])
            )
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
                # No whole-page text vector next to the sentences: its best match is nearly always a unit's
                # or a sentence's (ViDoRe hr -0.24, computer_science +0.06 nDCG@10 without it), and it was
                # half the text a page embedded, so dropping it pays for most of the sentences.
                units = media.pack_lines(draft.embed_text, tokens=media.PDF_TEXT_UNIT_TOKENS)
                out += [
                    _Embedding("unit", n, EmbedInput(modality="text", text=u, title=title)) for n, u in enumerate(units)
                ]
                out += _sentence_vectors(draft.embed_text, title)
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
                    image_tokens=image_tokens if draft.modality == "image" else None,
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
        # Every phase of the batch is attributed (seconds and round trips) and emitted as one
        # `[knowledge_ingest-timing]` line plus the phase metrics, as a retain's are: a slow batch is
        # otherwise only attributable by guessing which of storage, splitting, the embedder or the
        # knowledge store it was. Rounds, file writes and early commits run concurrently, so phases
        # overlap: read each against the batch's `wall`, not against the others.
        async with _timing.retain_timing(
            op="knowledge_ingest", bank_id=bank_id, store=type(self.store).__name__, documents=len(work)
        ):
            await self._process_batch(bank_id, work, retries_left=retries_left, depth=depth, caller=caller)

    async def _process_batch(
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
        settings = await self._bank_settings(bank_id)
        # An image's vision tokens, sent only when the bank overrides the server's: None keeps each
        # embedder's own default, which for a remote one is not necessarily the server setting.
        image_tokens = settings.image_tokens if settings.image_tokens != get_config().knowledge_image_tokens else None
        async with _timing.timed("store.read"):
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
        async with _timing.timed("store.status"):
            await self.store.set_statuses(
                bank_id, [StatusUpdate(d.work.document_id, "processing", storage_key=d.work.storage_key) for d in docs]
            )
        started = time.perf_counter()
        committed: set[str] = set()
        rounds: list[asyncio.Task[None]] = []
        slots = asyncio.Semaphore(_EMBED_ROUNDS_IN_FLIGHT)
        budget = _MemoryBudget(get_config().knowledge_ingest_memory_budget_mb * 1024 * 1024)

        async def run_round(batch: list[_Waiting]) -> None:
            try:
                await self._embed_round(batch, save, budget.adjust, image_tokens)
            except BaseException as exc:
                budget.fail(exc)
                raise
            finally:
                slots.release()

        async def launch(batch: list[_Waiting]) -> None:
            # at most _EMBED_ROUNDS_IN_FLIGHT; the splitter waits for a free one -- timed, because a
            # splitter that spends its time here is a batch bound by the embedder, not by splitting.
            async with _timing.timed("split.wait_embed"):
                await slots.acquire()
            task = asyncio.create_task(run_round(batch))
            rounds.append(task)
            for doc in {id(w.doc): w.doc for w in batch}.values():
                doc.rounds.append(task)

        early: list[asyncio.Task[None]] = []
        file_slots = asyncio.Semaphore(_PASSAGE_FILE_CONCURRENCY)

        async def write(key: str, data: bytes, mime: str | None) -> None:
            try:
                async with _timing.timed("media.write"):
                    await self._engine._file_storage.store(
                        file_data=data, key=key, metadata={"content_type": mime or ""}
                    )
            finally:
                file_slots.release()

        async def save(doc: _InFlight, key: str, draft: media.PassageDraft) -> None:
            """Write a passage's file in the background; its key is noted first, so a failed write's
            siblings are still cleaned up. The slot is taken before the write is started, so writes that
            fall behind hold the round back instead of piling their files up in memory."""
            assert draft.data is not None
            doc.media_keys.append(key)
            async with _timing.timed("media.wait_slot"):
                await file_slots.acquire()
            doc.saves.append(asyncio.create_task(write(key, draft.data, draft.mime)))

        def saves_of(group: list[_InFlight]) -> list[asyncio.Task[None]]:
            return [t for d in group for t in d.saves]

        async def commit(group: list[_InFlight]) -> None:
            """Write these documents' passages in one go; each is marked ready, or failed when it held
            nothing. The store commits each one only while it still holds the file this batch read: one
            replaced or deleted meanwhile is skipped (it used to be re-read here first, which left a gap
            between the check and the write), and its passages belong to nobody."""
            await asyncio.gather(*saves_of(group))  # a document commits once its files are all written
            for d in group:
                d.committing = True
                d.passages = [d.done[i] for i in sorted(d.done)]
                if d.error is None and not d.split and not d.replaced and not d.passages and not d.staged:
                    d.error = media.MediaError("the file holds nothing to search")
            # A version replaced meanwhile has nothing of its own to commit: its newer task writes the document.
            # What it already wrote (media of pieces staged before the store turned it away) is dropped here,
            # as the store's skip below used to drop it.
            done = [d for d in group if d.error is None and not d.split and not d.replaced]
            for d in group:
                if d.replaced:
                    await self._delete_keys(d.media_keys)
                    d.media_keys = []  # the transient cleanup must not delete them a second time
            try:
                if done:
                    await write_passages(done)
            finally:
                # Committed (or failed): its passages are not needed any more, and nor is their memory.
                for d in group:
                    d.committed_passages = d.staged + len(d.passages)
                    d.passages, d.done = [], {}
                    budget.adjust(-d.reserved)
                    d.reserved = 0

        async def write_passages(done: list[_InFlight]) -> None:
            async with _timing.timed("store.write"):
                replaced = await self.store.replace_passages(
                    bank_id,
                    [
                        PassageReplacement(
                            d.work.document_id, d.passages, storage_key=d.work.storage_key, staged=d.staged
                        )
                        for d in done
                    ],
                )
            skipped = set(replaced.skipped_document_ids)
            for d in done:
                if d.work.document_id in skipped:
                    await self._delete_keys(d.media_keys)
            kept = [d for d in done if d.work.document_id not in skipped]
            committed.update(d.work.document_id for d in kept)
            await self._delete_keys(replaced.stale_media_keys)
            # Only a document that used to be split can have split-out documents to drop now.
            was_split = [d.work.document_id for d in kept if d.work.was_split]
            if was_split:
                derived = await self.store.derived_documents(bank_id, was_split)
                await self._delete_trees(bank_id, [c for children in derived.values() for c in children])

        async def commit_when_embedded(group: list[_InFlight]) -> None:
            await asyncio.gather(*{id(t): t for d in group for t in d.rounds}.values())
            await commit(group)

        waiting: list[_Waiting] = []
        finished: list[_InFlight] = []  # cut and queued, not yet committing

        async def flush_round() -> None:
            if waiting:
                batch = list(waiting)
                waiting.clear()
                await launch(batch)

        async def stage(doc: _InFlight) -> None:
            """Write what a document has embedded so far out of sight, as a piece of its next version, and
            give back its memory: a document too big for the budget is held one piece at a time, not
            whole, and commits once it is all cut (``PassageReplacement(staged=n)``)."""
            await flush_round()
            await asyncio.gather(*doc.rounds)  # every passage queued so far is embedded
            await asyncio.gather(*doc.saves)
            if doc.error is not None:
                return
            piece = [doc.done[i] for i in sorted(doc.done)]
            async with _timing.timed("store.stage"):
                skipped = await self.store.stage_passages(
                    bank_id,
                    [PassageStage(doc.work.document_id, piece, doc.work.storage_key, doc.staged)],
                )
            if skipped:  # replaced or deleted meanwhile: what is left of it would belong to nobody
                doc.replaced = True
            doc.staged += len(piece)
            doc.done = {}
            budget.adjust(-doc.reserved)
            doc.reserved = 0

        async def commit_finished() -> None:
            """Start the commit of every document cut so far: it runs once their rounds are embedded,
            while the next documents are cut."""
            await flush_round()
            if finished:
                group = list(finished)
                finished.clear()
                for d in group:
                    d.committing = True  # now, so the batch's final commit never takes it too
                early.append(asyncio.create_task(commit_when_embedded(group)))

        async def settle() -> None:
            """Wait for every round; re-raise the first transient error once all have stopped, so the
            cleanup below never deletes a file a round is still writing."""
            results = await asyncio.gather(*rounds, return_exceptions=True)
            rounds.clear()
            errors = [r for r in results if isinstance(r, BaseException)]
            if errors:
                raise errors[0]

        try:
            for doc in docs:
                try:
                    drafts = await self._open(bank_id, doc, caller or {}, depth, settings)
                except ValueError as exc:
                    doc.error = exc
                    continue
                except _FileGone as exc:
                    # A newer upload can replace the file between the check above and this read, and it
                    # deletes the old one: this version is then superseded, and its newer task ingests it.
                    # Retrying (what this did before) only read the missing file again until retries ran
                    # out. Gone under the current version, the file cannot come back: the document fails.
                    current = (await self.store.get_documents(bank_id, [doc.work.document_id])).get(
                        doc.work.document_id
                    )
                    if current is None or current.storage_key != doc.work.storage_key:
                        doc.replaced = True
                        logger.info(f"Knowledge ingest skipped: {bank_id}/{doc.work.document_id} was replaced")
                    else:
                        doc.error = exc
                    continue
                if drafts is None:  # split into files, each its own document now, and itself ready
                    doc.split = True
                    committed.add(doc.work.document_id)
                    continue
                try:
                    # A document an embed round refused stops here: its remaining passages (a long
                    # video's windows) would only be cut to be thrown away.
                    while (
                        doc.error is None
                        and not doc.replaced
                        and (draft := await _split(_next_draft, drafts)) is not None
                    ):
                        cost = _passage_estimate(draft)
                        if budget.limit > 0 and doc.reserved + cost > budget.limit and doc.queued > doc.staged:
                            await stage(doc)
                        await budget.reserve(cost, doc.reserved, commit_finished)
                        doc.reserved += cost
                        waiting.append(_Waiting(doc, draft, doc.queued, cost))
                        doc.queued += 1
                        if len(waiting) >= _EMBED_ROUND:
                            await flush_round()
                except ValueError as exc:
                    doc.error = exc
                finally:
                    # Close the splitter now, off the event loop, rather than whenever it is collected.
                    await _split(_close_drafts, drafts)
                finished.append(doc)
                # Finished documents commit as soon as they are embedded, while the next ones are cut, in
                # groups: a big document (a long PDF, a video) on its own, small ones together. Writing a
                # batch's every vector at the end held them all at once, and was a third of a PDF batch's
                # ingest with nothing else running.
                if (doc.error is None and doc.queued >= _EARLY_COMMIT_PASSAGES) or sum(
                    d.queued for d in finished
                ) >= _COMMIT_GROUP_PASSAGES:
                    await commit_finished()
            await flush_round()
            await settle()
            # Whatever did not commit early commits here, together.
            await commit([d for d in docs if not d.committing])
            await asyncio.gather(*early)  # an early commit's error propagates like any other
            kept = [d for d in docs if d.work.document_id in committed]
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
                f"Knowledge ingest {bank_id}: {len(kept)} documents, "
                f"{sum(d.committed_passages for d in kept)} passages, "
                f"{len(failed)} failed, in {time.perf_counter() - started:.1f}s, holding at most "
                f"{budget.peak / 2**20:.0f} MB of passages"
            )
        except Exception as exc:
            # No round, file write or early commit may still be running when the cleanup deletes files.
            await asyncio.gather(*rounds, *early, return_exceptions=True)
            await asyncio.gather(*saves_of(docs), return_exceptions=True)
            # Transient: nothing of what is not committed is kept. Until the retries run out a document is
            # still on its way, so it stays pending with the error noted (it used to be marked failed at
            # once, and clients waiting for ready/failed gave up on it).
            pending = [d for d in docs if d.work.document_id not in committed]
            for d in pending:
                await self._delete_keys(d.media_keys)
            # An embedder still rate-limiting is deferred by the worker, not failed (worker/backpressure.py),
            # so its documents stay on their way whatever the retries left.
            from ..worker.backpressure import is_store_backpressure, is_upstream_rate_limit

            # A ValueError is permanent (the engine marks the operation failed without a retry, e.g. a
            # write at another vector size than the bank's): left pending, its documents would wait for
            # a retry that never comes.
            permanent = isinstance(exc, ValueError)
            on_its_way = not permanent and (retries_left or is_upstream_rate_limit(exc) or is_store_backpressure(exc))
            status: DocumentStatus = "pending" if on_its_way else "failed"
            note = f"retrying after: {exc!r}" if on_its_way else repr(exc)
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
        self, bank_id: str, doc: _InFlight, caller: dict[str, Any], depth: int, settings: KnowledgeBankSettings
    ) -> Iterator[media.PassageDraft] | None:
        """One document's passages to embed, as a lazy iterator; None when its strategy split it into files."""
        document, w = doc.document, doc.work
        async with _timing.timed("file.read"):
            try:
                data = await self._engine._file_storage.retrieve(w.storage_key)
            except FileNotFoundError as exc:
                # Only the stored original being gone means "superseded or lost" to the batch; a missing
                # file further in (a splitter's helper, a temp file) stays an ordinary, retried error.
                raise _FileGone(str(exc)) from exc
        if w.handler is None:
            # Splitting is blocking (pdfium, ffmpeg): it runs in a thread, one draft at a time.
            return await _split(self._drafts, document, data, settings)
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
            await self._write_derived(bank_id, document, w.storage_key, made.files, w.strategy, caller, depth, settings)
            return None
        # A strategy that returned passages has done the splitting already.
        emitted = _drafts_of(made.passages or [])
        for d in emitted:
            self._check_passage_modality(d.modality, settings)
        return iter(emitted)

    async def _embed_round(
        self,
        waiting: list[_Waiting],
        save: _SaveFile,
        adjust: Callable[[int], None] = lambda _: None,
        image_tokens: int | None = None,
    ) -> None:
        """Embed one round of passages, from any documents, in one call, and hand each to its document. A
        permanent error is narrowed to the documents it comes from, embedding each one's share alone.
        Each passage's file is handed to ``save``, which writes it in the background: the round does not
        wait for it (a round that waited held its slot through the file writes and left the embedder idle
        most of the time); the document's commit does."""
        live = [w for w in waiting if w.doc.error is None]
        if not live:
            return
        plans = [self._vectors_to_embed(w.draft, w.doc.document, w.doc.title, image_tokens) for w in live]
        try:
            async with _timing.timed("embed"):
                embedded = await self.embedder.embed_documents([e.input for plan in plans for e in plan])
        except ValueError as exc:
            if len({id(w.doc) for w in live}) == 1:
                live[0].doc.error = exc
                return
            # By identity: an _InFlight is a mutable dataclass, so it has no hash to dedupe on.
            for doc in {id(w.doc): w.doc for w in live}.values():
                await self._embed_round([w for w in live if w.doc is doc], save, adjust, image_tokens)
            return
        keys = {id(w): f"{w.doc.media_prefix}{w.index}.{w.draft.extension}" for w in live if w.draft.data is not None}
        for w in live:
            if id(w) in keys:
                await save(w.doc, keys[id(w)], w.draft)
        cursor = 0
        for w, plan in zip(live, plans, strict=True):
            mine = embedded[cursor : cursor + len(plan)]
            cursor += len(plan)
            doc, draft, index = w.doc, w.draft, w.index
            key = keys.get(id(w))
            # The estimate it reserved, corrected to what its vectors really weigh.
            actual = _passage_bytes(draft, mine)
            adjust(actual - w.cost)
            doc.reserved += actual - w.cost
            doc.done[index] = KnowledgePassage(
                index=index,
                modality=draft.modality,
                text=draft.text,
                page=draft.page,
                start_ms=draft.start_ms,
                end_ms=draft.end_ms,
                media_key=key,
                embedding=mine[0],
                extra_vectors=[RoleVector(e.role, e.n, v) for e, v in zip(plan[1:], mine[1:], strict=True) if e.role],
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
        settings: KnowledgeBankSettings,
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
                plan = self._check_item(item, settings, depth=depth + 1)
            except KnowledgeBankError as exc:  # the same file would be refused next time too
                raise media.MediaError(f"'{f.name}': {exc}") from exc
            children.append(_Entry(f"{document.document_id}/{f.name}", item, plan, document.document_id))
        current = (await self.store.get_documents(bank_id, [document.document_id])).get(document.document_id)
        if current is None or current.storage_key != storage_key:
            return  # replaced or deleted while the strategy ran
        saved = await self._save(bank_id, children, settings)
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
        self,
        bank_id: str,
        period: str,
        trunc: str,
        request_context: RequestContext,
        *,
        media_type: str | None = None,
    ) -> list[TimeBucket]:
        await self._open_bank(bank_id, KnowledgeBankReadOperation.GET_KNOWLEDGE_BANK, request_context)
        days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}[period]
        since = datetime.now(UTC) - timedelta(days=days)
        return await self.store.timeseries(bank_id, since=since, trunc=trunc, media_type=media_type)

    # --- search -----------------------------------------------------------------------------

    async def search(
        self,
        bank_id: str,
        request: SearchRequest,
        request_context: RequestContext,
        trace: SearchTrace | None = None,
    ) -> list[SearchResultItem]:
        # One `[knowledge_search-timing]` line per search, phase by phase, as an ingest batch has: a
        # slow search is otherwise only attributable by guessing which of the bank checks, the
        # embedder, the store or the budget's context reads it was.
        async with _timing.retain_timing(
            op="knowledge_search", bank_id=bank_id, store=type(self.store).__name__
        ) as timing:
            has_query = (
                bool((request.query or "").strip())
                or request.query_image is not None
                or request.query_video is not None
            )
            if not has_query:
                raise KnowledgeBankError("search needs a query, a query_image or a query_video")
            # The query is embedded while the bank is opened, not after: the embedding reads nothing
            # of the bank, and the two are each a network round trip. The store is not touched until
            # the bank is open, so a caller refused by the checks still learns nothing about it.
            embedding = asyncio.ensure_future(self._embed_query(request))
            try:
                async with _timing.timed("open_bank"):
                    await self._open_bank(bank_id, KnowledgeBankReadOperation.SEARCH_KNOWLEDGE_BANK, request_context)
            except BaseException:
                embedding.cancel()
                raise
            vector = await embedding
            async with _timing.timed("store.search"):
                hits = await self._search(bank_id, request, vector, trace)
            async with _timing.timed("context"):
                items = await self._within_budget(bank_id, hits, request.max_tokens, trace)
            if trace is not None:
                trace.seconds = dict(timing.seconds)
            return items

    async def _embed_query(self, request: SearchRequest) -> list[float]:
        async with _timing.timed("embed"):
            try:
                return await self.embedder.embed_query(request.query, request.query_image, request.query_video)
            except UnsupportedModalityError as exc:
                raise KnowledgeBankError(str(exc), 415) from exc

    async def _search(
        self, bank_id: str, request: SearchRequest, vector: list[float], trace: SearchTrace | None = None
    ) -> list[SearchHit]:
        return await self.store.search(
            bank_id,
            vector,
            top_k=SEARCH_CANDIDATES[request.budget],
            filters=request.filters,
            text_weight=get_config().knowledge_pdf_text_weight,
            query_text=request.query,
            keyword_weight=get_config().knowledge_keyword_weight,
            trace=trace.store if trace is not None else None,
        )

    async def _within_budget(
        self, bank_id: str, hits: list[SearchHit], max_tokens: int, trace: SearchTrace | None = None
    ) -> list[SearchResultItem]:
        """The ranked hits that fit in ``max_tokens``, each text hit with the passages around it as its
        ``context``. In rank order, a text hit comes with one passage on each side (the query-time step that
        paid, CHUNKING.md), never repeating a passage another result holds, or alone if only that fits; a hit
        another result already holds comes back there (merged), and that result does not grow. What is
        left then widens the best results, one passage a side at a time, up to ``MAX_CONTEXT_PASSAGES``. A hit with no text of its own costs
        ``MEDIA_HIT_TOKENS`` plus its text. This replaced ``top_k`` and ``neighbors``: a caller holds a token
        budget, and a count of hits said nothing about how much text came back."""
        text_hits = [h for h in hits if h.passage.modality == "text"]
        passages: dict[tuple[str, int], KnowledgePassage] = {}

        async def load(ranges: list[tuple[str, int, int]]) -> None:
            wanted = [r for r in ranges if any((r[0], i) not in passages for i in range(r[1], r[2] + 1))]
            if not wanted:
                return
            found = await self.store.passage_ranges(bank_id, wanted)
            for (document_id, _first, _last), run in found.items():
                for p in run:
                    passages[(document_id, p.index)] = p

        tokens: dict[tuple[str, int], int] = {}

        def cost(document_id: str, index: int) -> int:
            key = (document_id, index)
            if key not in tokens:
                tokens[key] = count_tokens(passages[key].text or "") if key in passages else 0
            return tokens[key]

        await load([(h.document.document_id, max(0, h.passage.index - 1), h.passage.index + 1) for h in text_hits])
        items: list[SearchResultItem] = []
        windows: list[_Window] = []
        held: dict[str, set[int]] = {}  # per document, the passages some window already returns
        used = 0

        def step(
            action: BudgetAction, hit: SearchHit, first: int, last: int, tokens: int, into: SearchHit | None = None
        ) -> None:
            if trace is not None:
                trace.budget_steps.append(
                    BudgetStep(action, hit, hit.document.document_id, first, last, tokens, used, into)
                )

        for hit in hits:
            document_id, index = hit.document.document_id, hit.passage.index
            if hit.passage.modality != "text":
                weight = MEDIA_HIT_TOKENS + count_tokens(hit.passage.text or "")
                if used + weight <= max_tokens:
                    items.append(SearchResultItem(hit))
                    used += weight
                    step("media", hit, index, index, weight)
                else:
                    step("skipped", hit, index, index, 0)
                continue
            mine = held.setdefault(document_id, set())
            if index in mine:
                # Already inside a better hit's result: it comes back there, and that result does not grow.
                owner = next(w for w in windows if w.document_id == document_id and w.first <= index <= w.last)
                step("merged", hit, owner.first, owner.last, 0, into=owner.item.hit)
                continue
            # The hit with the neighbours no result holds yet (a run: only its edges can be held).
            around = [i for i in (index - 1, index, index + 1) if (document_id, i) in passages and i not in mine]
            placed = False
            for span in (around, [index]):  # with its free neighbours, or alone
                weight = sum(cost(document_id, i) for i in span)
                if used + weight > max_tokens:
                    continue
                used += weight
                mine.update(span)
                placed = True
                item = SearchResultItem(hit)
                items.append(item)
                windows.append(_Window(item, document_id, min(span), max(span)))
                step("window" if len(span) > 1 else "alone", hit, min(span), max(span), weight)
                break
            if not placed:
                step("skipped", hit, index, index, 0)
        for _ in range(MAX_CONTEXT_PASSAGES):
            growing = [
                w for w in windows if (w.grows_before or w.grows_after) and w.last - w.first + 1 < MAX_CONTEXT_PASSAGES
            ]
            if not growing:
                break
            await load(
                [(w.document_id, w.first - 1, w.first - 1) for w in growing if w.grows_before and w.first > 0]
                + [(w.document_id, w.last + 1, w.last + 1) for w in growing if w.grows_after]
            )
            for w in growing:  # rank order: the best hits widen first
                for side in ("before", "after"):
                    if not (w.grows_before if side == "before" else w.grows_after):
                        continue
                    i = w.first - 1 if side == "before" else w.last + 1
                    mine = held[w.document_id]
                    full = w.last - w.first + 1 >= MAX_CONTEXT_PASSAGES
                    blocked = full or i < 0 or (w.document_id, i) not in passages or i in mine
                    if not blocked and used + cost(w.document_id, i) <= max_tokens:
                        used += cost(w.document_id, i)
                        mine.add(i)
                        if side == "before":
                            w.first = i
                        else:
                            w.last = i
                        step("widened", w.item.hit, w.first, w.last, cost(w.document_id, i))
                    elif side == "before":
                        w.grows_before = False
                    else:
                        w.grows_after = False
        for w in windows:
            w.item.context_first, w.item.context_last = w.first, w.last
            w.item.context = "\n\n".join(
                passages[(w.document_id, i)].text or ""
                for i in range(w.first, w.last + 1)
                if (w.document_id, i) in passages
            )
        if trace is not None:
            trace.tokens_used = used
            trace.hits = [_traced(hit, items, windows) for hit in hits]
        return items


def _traced(hit: SearchHit, items: list[SearchResultItem], windows: list[_Window]) -> TracedHit:
    """What the budget did with one ranked hit, read off the results it built."""
    if any(item.hit is hit for item in items):
        return TracedHit(hit, "returned")
    document_id, index = hit.document.document_id, hit.passage.index
    for w in windows:
        if w.document_id == document_id and w.first <= index <= w.last:
            return TracedHit(hit, "merged", merged_into=w.item.hit)
    return TracedHit(hit, "over_budget")


async def documents_for_memory_bank(
    engine: MemoryEngine, bank_id: str, linked: list[str], query: str, max_tokens: int, request_context: RequestContext
) -> list[KnowledgeDocumentHit] | None:
    """The linked knowledge banks' hits for reflect's recall tool, or None when none are linked.

    The same search as ``POST /knowledge-banks/{id}/search``, keywords and meaning, each text hit with
    the passages around it, within ``max_tokens``; the banks are searched at once and their hits taken in
    turn, best of each first (fused scores are per bank, so they cannot be ranked against each other).
    It used to be its own search: meaning only, the 5 best passages across banks, no text around them,
    so reflect read less than the same question showed in the UI.

    A failure here never fails the reflect: the memories are still the answer, without the docs.
    """
    if not get_config().enable_knowledge_banks:
        return None
    linked = [b for b in linked if b != bank_id]
    if not linked or not (query or "").strip():
        return None
    # Reflect's recall runs as an internal operation, but the linked banks are read on the caller's
    # behalf: the validator must see the caller, not a background job it would let through.
    caller = replace(request_context, internal=False)
    try:
        return await _linked_documents(engine, bank_id, linked, query, max_tokens, caller)
    except Exception as exc:
        logger.warning(f"Knowledge search for memory bank '{bank_id}' failed: {exc!r}")
        return []


async def _linked_documents(
    engine: MemoryEngine, bank_id: str, linked: list[str], query: str, max_tokens: int, caller: RequestContext
) -> list[KnowledgeDocumentHit]:
    service = get_knowledge_service(engine)
    request = SearchRequest(query=query, query_image=None, filters=SearchFilters(), max_tokens=max_tokens)
    found = await asyncio.gather(*(service.search(b, request, caller) for b in linked), return_exceptions=True)
    ranked: list[list[tuple[str, SearchResultItem]]] = []
    for linked_bank, items in zip(linked, found, strict=True):
        if isinstance(items, BaseException):
            # A bank this caller may not read, or that no longer exists, is skipped; reflect goes on.
            logger.warning(f"Linked knowledge bank '{linked_bank}' of '{bank_id}' skipped: {items!r}")
            continue
        ranked.append([(linked_bank, item) for item in items])
    from .routes import item_result

    out: list[KnowledgeDocumentHit] = []
    spent = 0
    for linked_bank, item in (pair for turn in itertools.zip_longest(*ranked) for pair in turn if pair):
        r = item_result(linked_bank, item)
        cost = count_tokens(r.text or "") + (0 if item.hit.passage.modality == "text" else MEDIA_HIT_TOKENS)
        if spent + cost > max_tokens:
            continue
        spent += cost
        out.append(
            KnowledgeDocumentHit(
                bank_id=linked_bank,
                document_id=r.document_id,
                passage_index=r.index,
                filename=r.filename,
                media_type=r.media_type,
                text=r.text,
                page=r.page,
                start_ms=r.start_ms,
                end_ms=r.end_ms,
                score=item.hit.score,
                passage_url=r.passage_url,
                document_url=r.document_url,
                tags=r.tags,
            )
        )
    return out


#: Letter runs in a file name that say nothing about its content: what cameras, phones, screen
#: recorders and exporters name files (``IMG_4821``, ``DSC01234``, ``video7020``, ``scan_003``).
_GENERIC_NAME_WORDS = frozenset(
    "img image images dsc dscn dscf pxl mvimg vid video videos mov clip photo pic picture frame scan "
    "screenshot screen recording rec audio sound track file untitled copy final new export output".split()
)


def _sentence_vectors(text: str, title: str | None) -> list[_Embedding]:
    """A PDF page's sentences, one vector each. A fact is one sentence, and a 128-token unit dilutes it
    with its neighbours: on ViDoRe hr and computer_science (Gemini Embedding 2) adding the page's best
    sentence gained +1.3 and +1.0 nDCG@10. Pairs of neighbouring sentences added +0.55 and +0.06 more
    but embedded every sentence twice again (hr ingest +53% instead of about +25%), so they were dropped."""
    return [
        _Embedding("sentence", n, EmbedInput(modality="text", text=s, title=title))
        for n, s in enumerate(media.sentences(text))
    ]


def _title_hint(document: KnowledgeDocument) -> str | None:
    """The file name without its extension: the embedder's "title: ..." prompt slot. None for the name
    a text write gets when it has none (``<document_id>.md``): an id, often a UUID, is not a title. A
    file named like its id (``report.pdf`` as ``report``) keeps it: the title is worth 2-10 nDCG@10.

    None too when the name has no real word in it (a number, a hash, a camera name): on Flickr30k,
    photos named ``1007129816.jpg`` embedded as "title: 1007129816" scored R@1 79.8; the image alone
    scored 93.3. A named asset (``footstep_gravel_03``) still gets its title."""
    if not document.filename or document.filename == f"{document.document_id}.md":
        return None
    stem = Path(document.filename).stem
    words = [
        w
        for w in re.findall(r"[^\W\d_]{3,}", stem)
        if w.lower() not in _GENERIC_NAME_WORDS
        and not re.fullmatch(r"[0-9a-fA-F]{8,}", w)  # a hash, not "face" or "bed"
    ]
    if not words:
        return None
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


_T = TypeVar("_T")


async def _split(fn: Callable[..., _T], *args: Any) -> _T:
    """Run a blocking step of splitting (opening a file, cutting its next passage) off the event
    loop, timed as the batch's ``split`` phase."""
    async with _timing.timed("split"):
        return await asyncio.to_thread(fn, *args)


def _close_drafts(drafts: Iterator[media.PassageDraft]) -> None:
    close = getattr(drafts, "close", None)  # generators release the PDF / media they hold
    if close is not None:
        close()


def _next_draft(drafts: Iterator[media.PassageDraft]) -> media.PassageDraft | None:
    return next(drafts, None)


# A threading lock, not an asyncio one: engines are reached from more than one thread and the critical
# section never awaits.
_services_lock = threading.Lock()


def get_knowledge_service(engine: MemoryEngine) -> KnowledgeService:
    """One service per engine (one embedder, loaded once), created on first use and kept on the engine,
    so it lives exactly as long as the engine does. It used to sit in a process-wide WeakKeyDictionary
    keyed by the engine, but the service holds its engine, and a value that holds its own key keeps the
    entry (and the engine, with its pools and caches) alive for ever: every engine a test created leaked,
    until an api-slim CI shard ran out of memory and its runner was killed."""
    with _services_lock:
        service = engine._knowledge_service
        if service is None:
            service = engine._knowledge_service = KnowledgeService(engine)
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
