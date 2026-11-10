"""Where knowledge-bank documents and passages live.

``KnowledgeStore`` is the seam an external backend plugs into
through ``HINDSIGHT_API_KNOWLEDGE_STORE_EXTENSION=module.path:ClassName``. The bank row itself
always stays in the ``banks`` table, because async operations, audit and the delete cascade key
off it. ``PostgresKnowledgeStore`` is the default: ``kb_documents`` + ``kb_passages`` with a
pgvector HNSW index.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, cast

from ..engine.db_utils import acquire_with_retry
from ..engine.schema import fq_table
from ..schema_context import get_current_schema
from .language import detect as detect_language
from .language import text_search_config

logger = logging.getLogger(__name__)

# pgvector indexes `vector` up to 2000 dimensions and `halfvec` up to 4000.
_VECTOR_INDEX_MAX = 2000
_HALFVEC_INDEX_MAX = 4000


Modality = Literal["text", "image", "audio", "video"]
MediaType = Literal["document", "image", "audio", "video", "text"]
DocumentStatus = Literal["pending", "processing", "ready", "failed"]


@dataclass
class KnowledgeDocument:
    document_id: str
    filename: str | None
    content_type: str
    media_type: MediaType
    size_bytes: int
    storage_key: str | None
    content_hash: str
    status: DocumentStatus
    error: str | None
    passage_count: int
    tags: list[str]
    created_at: datetime | None = None
    language: str | None = None  # detected from the text at ingest; gates keyword search
    updated_at: datetime | None = None
    #: The document an ingestion strategy split this one out of (see ingest_strategy.py).
    source_document_id: str | None = None


@dataclass
class RoleVector:
    """An extra vector of a passage: a PDF page's band crop, its whole text, or one of its text units."""

    role: str  # band | page_text | unit
    n: int
    embedding: list[float]


@dataclass
class KnowledgePassage:
    index: int
    modality: Modality
    text: str | None = None
    page: int | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    media_key: str | None = None
    embedding: list[float] | None = None
    extra_vectors: list[RoleVector] = field(default_factory=list)


@dataclass
class StatusUpdate:
    document_id: str
    status: DocumentStatus
    error: str | None = None
    #: Only while the document still holds this file: an ingest of an older version must not overwrite
    #: the status of the newer one that replaced it meanwhile. None updates whatever is stored.
    storage_key: str | None = None


@dataclass
class PassageReplacement:
    document_id: str
    passages: list[KnowledgePassage]
    language: str | None = None
    #: Only while the document still holds this file (see ``StatusUpdate``); otherwise it is skipped.
    storage_key: str | None = None


@dataclass
class PassagesReplaced:
    #: Media keys of the old passages no new passage reuses: the caller deletes those files.
    stale_media_keys: list[str] = field(default_factory=list)
    #: Items left out because their document no longer holds their ``storage_key`` (or is gone):
    #: their new passages belong to nobody, so the caller deletes their media too.
    skipped_document_ids: list[str] = field(default_factory=list)


@dataclass
class SearchHit:
    passage: KnowledgePassage
    document: KnowledgeDocument
    score: float


@dataclass
class SearchFilters:
    tags: list[str] | None = None
    tags_match: str = "any"  # any | all
    document_ids: list[str] | None = None
    media_types: list[str] | None = None


@dataclass
class DocumentPage:
    documents: list[KnowledgeDocument]
    total: int


@dataclass
class TimeBucket:
    time: datetime
    documents: int
    passages: int


@dataclass
class BankStats:
    document_count: int = 0
    passage_count: int = 0
    ready_count: int = 0
    failed_count: int = 0
    pending_count: int = 0
    total_bytes: int = 0
    by_media_type: dict[str, int] = field(default_factory=dict)


class KnowledgeStore(ABC):
    """Documents and passages of knowledge banks. Every method is scoped to one bank."""

    @abstractmethod
    async def ensure_ready(self, dimension: int) -> None:
        """Prepare storage for vectors of ``dimension`` (idempotent, called before every use)."""

    @abstractmethod
    async def get_documents(self, bank_id: str, document_ids: list[str]) -> dict[str, KnowledgeDocument]:
        """The documents found among ``document_ids``, by id (a missing one is simply absent)."""

    @abstractmethod
    async def put_documents(self, bank_id: str, documents: list[KnowledgeDocument]) -> dict[str, str | None]:
        """Insert or replace each document's row (passages untouched). Returns each one's previous storage
        key, None for a new document. Ids must be unique within the call."""

    @abstractmethod
    async def list_documents(
        self,
        bank_id: str,
        *,
        limit: int,
        offset: int,
        query: str | None = None,
        status: str | None = None,
        media_type: str | None = None,
        tags: list[str] | None = None,
    ) -> DocumentPage: ...

    @abstractmethod
    async def set_statuses(self, bank_id: str, updates: list[StatusUpdate]) -> None: ...

    @abstractmethod
    async def replace_passages(self, bank_id: str, items: list[PassageReplacement]) -> PassagesReplaced:
        """Swap each document's passages for the given ones and mark it ready; an item whose
        ``storage_key`` the document no longer holds is skipped, and reported as such."""

    @abstractmethod
    async def list_passages(self, bank_id: str, document_ids: list[str]) -> dict[str, list[KnowledgePassage]]:
        """Every passage of each document, in order; a document with none maps to an empty list."""

    @abstractmethod
    async def passage_ranges(
        self, bank_id: str, ranges: list[tuple[str, int, int]]
    ) -> dict[tuple[str, int, int], list[KnowledgePassage]]:
        """For each ``(document_id, first, last)``, passages ``first..last`` (inclusive) in order."""

    @abstractmethod
    async def delete_documents(self, bank_id: str, document_ids: list[str]) -> list[KnowledgeDocument]:
        """Delete these documents and their passages; returns what was deleted, for its files."""

    @abstractmethod
    async def media_keys(self, bank_id: str, document_ids: list[str]) -> list[str]:
        """The media keys of every passage of these documents."""

    @abstractmethod
    async def derived_documents(self, bank_id: str, document_ids: list[str]) -> dict[str, list[str]]:
        """For each document, the ids of the documents an ingestion strategy split it into."""

    @abstractmethod
    async def search(
        self,
        bank_id: str,
        vector: list[float],
        *,
        top_k: int,
        filters: SearchFilters,
        text_weight: float = 0.6,
        query_text: str | None = None,
        keyword_weight: float = 0.2,
        pattern: str | None = None,
        regex: bool = False,
    ) -> list[SearchHit]:
        """Best passages, each scored from all its vectors (``page_score``), keyword matches fused in.
        With ``pattern``, passages containing it are candidates too and rank above every passage that
        does not (``EXACT_MATCH_BONUS``)."""

    @abstractmethod
    async def grep(
        self,
        bank_id: str,
        pattern: str,
        *,
        top_k: int,
        filters: SearchFilters,
        regex: bool = False,
    ) -> list[SearchHit]:
        """Search by ``pattern`` alone: passages whose text contains it (case-insensitive; a POSIX regular
        expression when ``regex``). Documents with the most matching passages first, each in passage
        order; a hit's score is how many of its document's passages match."""

    @abstractmethod
    async def bank_stats(self, bank_ids: list[str]) -> dict[str, BankStats]: ...

    @abstractmethod
    async def timeseries(self, bank_id: str, *, since: datetime, trunc: str) -> list[TimeBucket]:
        """Documents created per ``trunc`` ('hour' or 'day') since ``since``, with their passage counts."""


def _document(row: Any) -> KnowledgeDocument:
    return KnowledgeDocument(
        document_id=row["document_id"],
        filename=row["filename"],
        content_type=row["content_type"],
        media_type=cast(MediaType, row["media_type"]),
        size_bytes=int(row["size_bytes"] or 0),
        storage_key=row["storage_key"],
        content_hash=row["content_hash"],
        status=cast(DocumentStatus, row["status"]),
        error=row["error"],
        passage_count=int(row["passage_count"] or 0),
        tags=list(row["tags"] or []),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        language=row["language"],
        source_document_id=row["source_document_id"],
    )


def _passage(row: Any) -> KnowledgePassage:
    return KnowledgePassage(
        index=row["idx"],
        modality=cast(Modality, row["modality"]),
        text=row["text"],
        page=row["page"],
        start_ms=row["start_ms"],
        end_ms=row["end_ms"],
        media_key=row["media_key"],
    )


_DOC_COLUMNS = (
    "document_id, filename, content_type, media_type, size_bytes, storage_key, content_hash, status, error, "
    "passage_count, tags, created_at, updated_at, language, source_document_id"
)
_PASSAGE_COLUMNS = "idx, modality, text, page, start_ms, end_ms, media_key"


class PostgresKnowledgeStore(KnowledgeStore):
    def __init__(self, backend_provider: Any) -> None:
        # The memory engine's own pool, fetched lazily: it exists only after the engine initialized.
        self._backend_provider = backend_provider
        self._ready: dict[str, int] = {}
        self._dimension = 0
        self._iterative_scan = False

    async def _pool(self) -> Any:
        return await self._backend_provider()

    async def ensure_ready(self, dimension: int) -> None:
        schema = get_current_schema() or "public"
        if self._ready.get(schema) == dimension:
            return
        async with acquire_with_retry(await self._pool()) as conn:
            for table in ("kb_passages", "kb_passage_vectors"):
                await self._pin_dimension(conn, schema, table, dimension)
            version = await conn.fetchval("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        self._ready[schema] = dimension
        self._dimension = dimension
        self._iterative_scan = _version_at_least(version, (0, 8))

    async def _pin_dimension(self, conn: Any, schema: str, table_name: str, dimension: int) -> None:
        passages = fq_table(table_name)
        async with conn.transaction():
            # One process at a time pins the column and builds the index, per schema: the table
            # lock (not an advisory lock, which poolers break) serializes a concurrent first use,
            # and the loser then finds the column already pinned. Writers wait only this once.
            await conn.execute(f"LOCK TABLE {passages} IN SHARE ROW EXCLUSIVE MODE")
            current = await conn.fetchval(
                """
                SELECT a.atttypmod FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = $1 AND c.relname = $2 AND a.attname = 'embedding'
                """,
                schema,
                table_name,
            )
            if current is None:
                raise RuntimeError(f"{table_name} is missing in schema {schema}; migrations have not run")
            if current != dimension:
                rows = await conn.fetchval(f"SELECT COUNT(*) FROM {passages}")
                if rows:
                    raise RuntimeError(
                        f"Knowledge banks hold {rows} passages embedded at {current} dimensions, but the "
                        f"configured knowledge embedder makes {dimension}. Use a model with {current} "
                        f"dimensions (HINDSIGHT_API_KNOWLEDGE_EMBEDDINGS_DIMENSIONS) or delete the knowledge "
                        f"banks and write them again."
                    )
                await conn.execute(f"DROP INDEX IF EXISTS {fq_table(f'idx_{table_name}_embedding')}")
                await conn.execute(f"ALTER TABLE {passages} ALTER COLUMN embedding TYPE vector({dimension})")
            if dimension <= _VECTOR_INDEX_MAX:
                using = "hnsw (embedding vector_cosine_ops)"
            elif dimension <= _HALFVEC_INDEX_MAX:
                using = f"hnsw ((embedding::halfvec({dimension})) halfvec_cosine_ops)"
            else:
                using = None
                logger.warning(f"Knowledge passages at {dimension} dims cannot be indexed; search scans")
            if using:
                # ponytail: one global HNSW index; per-bank partial indexes if one tenant holds many big banks.
                await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding ON {passages} USING {using}")

    def _distance(self, placeholder: str, alias: str = "p") -> str:
        dimension = self._dimension
        if _VECTOR_INDEX_MAX < dimension <= _HALFVEC_INDEX_MAX:
            # Same expression as the index, or the planner cannot use it.
            return f"({alias}.embedding::halfvec({dimension}) <=> {placeholder}::halfvec({dimension}))"
        return f"({alias}.embedding <=> {placeholder}::vector)"

    async def get_documents(self, bank_id: str, document_ids: list[str]) -> dict[str, KnowledgeDocument]:
        if not document_ids:
            return {}
        async with acquire_with_retry(await self._pool()) as conn:
            rows = await conn.fetch(
                f"SELECT {_DOC_COLUMNS} FROM {fq_table('kb_documents')} "
                "WHERE bank_id = $1 AND document_id = ANY($2::text[])",
                bank_id,
                document_ids,
            )
        return {r["document_id"]: _document(r) for r in rows}

    async def put_documents(self, bank_id: str, documents: list[KnowledgeDocument]) -> dict[str, str | None]:
        if not documents:
            return {}
        table = fq_table("kb_documents")
        # One statement: the CTE reads the previous keys from the snapshot taken before the upsert. The rows
        # go as one JSON array because asyncpg cannot pass an array of rows holding a text[] (the tags).
        rows = json.dumps(
            [
                {
                    "document_id": d.document_id,
                    "filename": d.filename,
                    "content_type": d.content_type,
                    "media_type": d.media_type,
                    "size_bytes": d.size_bytes,
                    "storage_key": d.storage_key,
                    "content_hash": d.content_hash,
                    "status": d.status,
                    "tags": d.tags,
                    "source_document_id": d.source_document_id,
                }
                # In document_id order, the order every batch write locks rows in (see set_statuses).
                for d in sorted(documents, key=lambda d: d.document_id)
            ]
        )
        async with acquire_with_retry(await self._pool()) as conn:
            previous = await conn.fetch(
                f"""
                WITH input AS (
                    SELECT * FROM jsonb_to_recordset($2::text::jsonb) AS x(document_id text, filename text,
                        content_type text, media_type text, size_bytes bigint, storage_key text,
                        content_hash text, status text, tags jsonb, source_document_id text)
                ),
                previous AS (
                    SELECT d.document_id, d.storage_key FROM {table} d
                    WHERE d.bank_id = $1 AND d.document_id IN (SELECT document_id FROM input)
                ),
                upsert AS (
                    INSERT INTO {table} (bank_id, document_id, filename, content_type, media_type, size_bytes,
                        storage_key, content_hash, status, error, tags, source_document_id)
                    SELECT $1, document_id, filename, content_type, media_type, size_bytes, storage_key,
                        content_hash, status, NULL, ARRAY(SELECT jsonb_array_elements_text(tags)),
                        source_document_id
                    FROM input
                    ON CONFLICT (bank_id, document_id) DO UPDATE SET
                        filename = EXCLUDED.filename,
                        content_type = EXCLUDED.content_type, media_type = EXCLUDED.media_type,
                        size_bytes = EXCLUDED.size_bytes, storage_key = EXCLUDED.storage_key,
                        content_hash = EXCLUDED.content_hash, status = EXCLUDED.status, error = NULL,
                        tags = EXCLUDED.tags, source_document_id = EXCLUDED.source_document_id,
                        updated_at = now()
                )
                SELECT document_id, storage_key FROM previous
                """,
                bank_id,
                rows,
            )
        found = {r["document_id"]: r["storage_key"] for r in previous}
        return {d.document_id: found.get(d.document_id) for d in documents}

    async def list_documents(
        self,
        bank_id: str,
        *,
        limit: int,
        offset: int,
        query: str | None = None,
        status: str | None = None,
        media_type: str | None = None,
        tags: list[str] | None = None,
    ) -> DocumentPage:
        params: list[Any] = [bank_id]
        where = "bank_id = $1"
        if query:
            params.append(f"%{query}%")
            where += f" AND (document_id ILIKE ${len(params)} OR filename ILIKE ${len(params)})"  # fmt: skip
        if status:
            params.append(status)
            where += f" AND status = ${len(params)}"
        if media_type:
            params.append(media_type)
            where += f" AND media_type = ${len(params)}"
        if tags:
            params.append(tags)
            where += f" AND tags && ${len(params)}::text[]"
        table = fq_table("kb_documents")
        async with acquire_with_retry(await self._pool()) as conn:
            total = await conn.fetchval(f"SELECT COUNT(*) FROM {table} WHERE {where}", *params)
            rows = await conn.fetch(
                f"SELECT {_DOC_COLUMNS} FROM {table} WHERE {where} "
                f"ORDER BY updated_at DESC, document_id LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}",
                *params,
                limit,
                offset,
            )
        return DocumentPage(documents=[_document(r) for r in rows], total=int(total or 0))

    async def set_statuses(self, bank_id: str, updates: list[StatusUpdate]) -> None:
        if not updates:
            return
        async with acquire_with_retry(await self._pool()) as conn:
            table = fq_table("kb_documents")
            # The rows are locked in document_id order first (an UPDATE ... FROM locks in join order), the
            # order replace_passages and put_documents use, so overlapping batches cannot deadlock.
            await conn.execute(
                f"""
                WITH locked AS (
                    SELECT document_id FROM {table}
                    WHERE bank_id = $1 AND document_id = ANY($2::text[]) ORDER BY document_id FOR UPDATE
                )
                UPDATE {table} d SET status = u.status, error = u.error, updated_at = now()
                FROM unnest($2::text[], $3::text[], $4::text[], $5::text[]) AS u(document_id, status, error, key)
                WHERE d.bank_id = $1 AND d.document_id = u.document_id
                    AND d.document_id IN (SELECT document_id FROM locked)
                    AND (u.key IS NULL OR d.storage_key = u.key)
                """,
                bank_id,
                [u.document_id for u in updates],
                [u.status for u in updates],
                [u.error for u in updates],
                [u.storage_key for u in updates],
            )

    async def replace_passages(self, bank_id: str, items: list[PassageReplacement]) -> PassagesReplaced:
        if not items:
            return PassagesReplaced()
        table = fq_table("kb_passages")
        # A fixed five statements in one transaction, however many documents: the old passages must be
        # gone before the new ones (same primary keys) go in, which one statement with data-modifying
        # CTEs cannot order. The inserts are executemany, pipelined in one round trip each.
        async with acquire_with_retry(await self._pool()) as conn:
            async with conn.transaction():
                # Locked, so an ingest of a newer version cannot swap the file between this check and the
                # writes below; an item for a file the document no longer holds is left out.
                held = {
                    r["document_id"]: r["storage_key"]
                    for r in await conn.fetch(
                        f"SELECT document_id, storage_key FROM {fq_table('kb_documents')} "
                        # Locked in document_id order, as set_statuses and put_documents take them, so two
                        # overlapping batches cannot deadlock on each other's rows.
                        "WHERE bank_id = $1 AND document_id = ANY($2::text[]) ORDER BY document_id FOR UPDATE",
                        bank_id,
                        [i.document_id for i in items],
                    )
                }

                def holds(i: PassageReplacement) -> bool:
                    return i.document_id in held and (i.storage_key is None or held[i.document_id] == i.storage_key)

                skipped = [i.document_id for i in items if not holds(i)]
                items = [i for i in items if holds(i)]
                if not items:
                    return PassagesReplaced(skipped_document_ids=skipped)
                document_ids = [i.document_id for i in items]
                old = await conn.fetch(
                    f"DELETE FROM {table} WHERE bank_id = $1 AND document_id = ANY($2::text[]) RETURNING media_key",
                    bank_id,
                    document_ids,
                )
                rows = [
                    (
                        bank_id,
                        i.document_id,
                        p.index,
                        p.modality,
                        p.text,
                        p.page,
                        p.start_ms,
                        p.end_ms,
                        p.media_key,
                        str(p.embedding),
                        text_search_config(i.language),
                    )
                    for i in items
                    for p in i.passages
                ]
                if rows:
                    await conn.executemany(
                        f"""
                        INSERT INTO {table} (bank_id, document_id, idx, modality, text, page, start_ms, end_ms,
                            media_key, embedding, search_vector)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::vector,
                                to_tsvector($11::regconfig, coalesce($5, '')))
                        """,
                        rows,
                    )
                extras = [
                    (bank_id, i.document_id, p.index, v.role, v.n, str(v.embedding))
                    for i in items
                    for p in i.passages
                    for v in p.extra_vectors
                ]
                if extras:
                    await conn.executemany(
                        f"INSERT INTO {fq_table('kb_passage_vectors')} (bank_id, document_id, idx, role, n, "
                        "embedding) VALUES ($1, $2, $3, $4, $5, $6::vector)",
                        extras,
                    )
                await conn.execute(
                    f"UPDATE {fq_table('kb_documents')} d SET status = 'ready', error = NULL, "
                    "passage_count = u.passage_count, language = u.language, updated_at = now() "
                    "FROM unnest($2::text[], $3::int[], $4::text[]) AS u(document_id, passage_count, language) "
                    "WHERE d.bank_id = $1 AND d.document_id = u.document_id",
                    bank_id,
                    document_ids,
                    [len(i.passages) for i in items],
                    [i.language for i in items],
                )
        new_keys = {p.media_key for i in items for p in i.passages}
        return PassagesReplaced(
            stale_media_keys=[r["media_key"] for r in old if r["media_key"] and r["media_key"] not in new_keys],
            skipped_document_ids=skipped,
        )

    async def list_passages(self, bank_id: str, document_ids: list[str]) -> dict[str, list[KnowledgePassage]]:
        out: dict[str, list[KnowledgePassage]] = {d: [] for d in document_ids}
        if not document_ids:
            return out
        async with acquire_with_retry(await self._pool()) as conn:
            rows = await conn.fetch(
                f"SELECT document_id, {_PASSAGE_COLUMNS} FROM {fq_table('kb_passages')} "
                "WHERE bank_id = $1 AND document_id = ANY($2::text[]) ORDER BY document_id, idx",
                bank_id,
                document_ids,
            )
        for r in rows:
            out[r["document_id"]].append(_passage(r))
        return out

    async def passage_ranges(
        self, bank_id: str, ranges: list[tuple[str, int, int]]
    ) -> dict[tuple[str, int, int], list[KnowledgePassage]]:
        ranges = list(dict.fromkeys(ranges))  # a repeated range would join its passages twice
        out: dict[tuple[str, int, int], list[KnowledgePassage]] = {r: [] for r in ranges}
        if not ranges:
            return out
        async with acquire_with_retry(await self._pool()) as conn:
            rows = await conn.fetch(
                f"SELECT r.document_id AS range_document_id, r.first, r.last, {_PASSAGE_COLUMNS} "
                "FROM unnest($2::text[], $3::int[], $4::int[]) AS r(document_id, first, last) "
                f"JOIN {fq_table('kb_passages')} p ON p.bank_id = $1 AND p.document_id = r.document_id "
                "AND p.idx BETWEEN r.first AND r.last ORDER BY p.idx",
                bank_id,
                [r[0] for r in ranges],
                [r[1] for r in ranges],
                [r[2] for r in ranges],
            )
        for r in rows:
            out[(r["range_document_id"], r["first"], r["last"])].append(_passage(r))
        return out

    async def media_keys(self, bank_id: str, document_ids: list[str]) -> list[str]:
        if not document_ids:
            return []
        async with acquire_with_retry(await self._pool()) as conn:
            rows = await conn.fetch(
                f"SELECT media_key FROM {fq_table('kb_passages')} "
                "WHERE bank_id = $1 AND document_id = ANY($2::text[]) AND media_key IS NOT NULL",
                bank_id,
                document_ids,
            )
        return [r["media_key"] for r in rows]

    async def derived_documents(self, bank_id: str, document_ids: list[str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {d: [] for d in document_ids}
        if not document_ids:
            return out
        async with acquire_with_retry(await self._pool()) as conn:
            rows = await conn.fetch(
                f"SELECT source_document_id, document_id FROM {fq_table('kb_documents')} "
                "WHERE bank_id = $1 AND source_document_id = ANY($2::text[]) ORDER BY document_id",
                bank_id,
                document_ids,
            )
        for r in rows:
            out[r["source_document_id"]].append(r["document_id"])
        return out

    async def delete_documents(self, bank_id: str, document_ids: list[str]) -> list[KnowledgeDocument]:
        if not document_ids:
            return []
        async with acquire_with_retry(await self._pool()) as conn:
            table = fq_table("kb_documents")
            # Locked in document_id order first, like every other batch write (see set_statuses).
            rows = await conn.fetch(
                f"DELETE FROM {table} WHERE bank_id = $1 AND document_id IN ("
                f"SELECT document_id FROM {table} WHERE bank_id = $1 AND document_id = ANY($2::text[]) "
                f"ORDER BY document_id FOR UPDATE) RETURNING {_DOC_COLUMNS}",
                bank_id,
                document_ids,
            )
        return [_document(r) for r in rows]

    async def search(
        self,
        bank_id: str,
        vector: list[float],
        *,
        top_k: int,
        filters: SearchFilters,
        text_weight: float = 0.6,
        query_text: str | None = None,
        keyword_weight: float = 0.2,
        pattern: str | None = None,
        regex: bool = False,
    ) -> list[SearchHit]:
        """Two stages. (1) Candidate passages: the nearest primary vectors, the nearest extra vectors
        (a page's bands, text, text units), and — when the query's language is known — the best keyword
        matches among documents in that language. (2) Every candidate is scored from all its vectors
        with ``page_score``; keyword matches, when any, are fused in as ``(1 - w) * dense + w * keyword``
        after min-max scaling over the candidates (PDF_GEMMA.md)."""
        params: list[Any] = [str(vector), bank_id]
        where = ""
        if filters.document_ids:
            params.append(filters.document_ids)
            where += f" AND x.document_id = ANY(${len(params)}::text[])"
        if filters.media_types:
            params.append(filters.media_types)
            where += f" AND d.media_type = ANY(${len(params)}::text[])"
        if filters.tags:
            params.append(filters.tags)
            op = "@>" if filters.tags_match == "all" else "&&"
            where += f" AND d.tags {op} ${len(params)}::text[]"
        # ponytail: fixed candidate pool; widen it (or move stage 1 to a token-level index) at 10M+ rows.
        params.append(max(100, top_k * 10))
        pool = f"${len(params)}"
        language = detect_language(query_text) if query_text and keyword_weight > 0 else None
        if language:
            params.append(text_search_config(language))
            cfg = f"${len(params)}::regconfig"
            params.append(language)
            lang = f"${len(params)}"
            params.append(query_text)
            tsquery = _or_tsquery(cfg, f"${len(params)}")
        docs = fq_table("kb_documents")

        def nearest(table: str) -> str:
            return f"""
                SELECT x.document_id, x.idx FROM {fq_table(table)} x
                JOIN {docs} d ON d.bank_id = x.bank_id AND d.document_id = x.document_id
                WHERE x.bank_id = $2{where}
                ORDER BY {self._distance("$1", "x")} LIMIT {pool}
            """

        arms = [nearest("kb_passages"), nearest("kb_passage_vectors")]
        if pattern:
            # Exact text is its own candidate arm: a code or id that meaning search ranks low (or not
            # at all) still reaches stage 2, where it wins. This replaced a separate /grep endpoint.
            params.append(_match_value(pattern, regex))
            arms.append(f"""
                SELECT x.document_id, x.idx FROM {fq_table("kb_passages")} x
                JOIN {docs} d ON d.bank_id = x.bank_id AND d.document_id = x.document_id
                WHERE x.bank_id = $2{where} AND {_text_match("x.text", f"${len(params)}", regex)} LIMIT {pool}
            """)
        if language:
            arms.append(f"""
                SELECT x.document_id, x.idx FROM {fq_table("kb_passages")} x
                JOIN {docs} d ON d.bank_id = x.bank_id AND d.document_id = x.document_id
                WHERE x.bank_id = $2{where} AND d.language = {lang} AND x.search_vector @@ {tsquery}
                ORDER BY ts_rank_cd(x.search_vector, {tsquery}) DESC LIMIT {pool}
            """)
        async with acquire_with_retry(await self._pool()) as conn:
            async with conn.transaction():
                # A filtered HNSW scan otherwise stops at ef_search candidates and returns too few rows.
                await conn.execute("SET LOCAL hnsw.ef_search = 200")
                if self._iterative_scan:
                    await conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
                candidates = await conn.fetch(
                    "SELECT DISTINCT document_id, idx FROM (" + " UNION ALL ".join(f"({a})" for a in arms) + ") c",
                    *params,
                )
            if not candidates:
                return []
            stage_params: list[Any] = [
                str(vector),
                bank_id,
                [r["document_id"] for r in candidates],
                [r["idx"] for r in candidates],
            ]
            keyword = "0"
            if language:
                # BM25's IDF: a query word scores ln(1 + (N - df + 0.5) / (df + 0.5)) over the bank's
                # passages in that language, so rare words (names, codes) count and common ones barely do.
                # ts_rank_cd alone has no IDF and lost 0.2 nDCG@10 on ViDoRe hr where this gains 0.4.
                # ponytail: one df count per query word (GIN, ~20 ms at 1K pages); cache dfs per bank or
                # read them from pg_stats (engine/search/bm25_term_selection.py) past ~1M passages.
                terms = await conn.fetch(
                    f"""
                    WITH w AS (SELECT DISTINCT unnest(tsvector_to_array(to_tsvector($2::regconfig, $3))) AS w),
                         n AS (SELECT count(*)::float8 AS n FROM {fq_table("kb_passages")} x
                               JOIN {docs} d ON d.bank_id = x.bank_id AND d.document_id = x.document_id
                               WHERE x.bank_id = $1 AND d.language = $4)
                    SELECT w.w, ln(1 + (n.n - df.df + 0.5) / (df.df + 0.5)) AS idf FROM w, n,
                      LATERAL (SELECT count(*)::float8 AS df FROM {fq_table("kb_passages")} x
                               JOIN {docs} d ON d.bank_id = x.bank_id AND d.document_id = x.document_id
                               WHERE x.bank_id = $1 AND d.language = $4
                                 AND x.search_vector @@ to_tsquery('simple', quote_literal(w.w))) df
                    """,
                    bank_id,
                    text_search_config(language),
                    query_text,
                    language,
                )
                stage_params += [language, [t["w"] for t in terms], [float(t["idf"]) for t in terms]]
                keyword = (
                    "CASE WHEN d.language = $5 THEN (SELECT coalesce(sum(t.idf), 0) FROM unnest($6::text[], "
                    "$7::float8[]) t(w, idf) WHERE p.search_vector @@ to_tsquery('simple', quote_literal(t.w))) "
                    "ELSE 0 END"
                )
            matched = "false"
            if pattern:
                stage_params.append(_match_value(pattern, regex))
                matched = _text_match("p.text", f"${len(stage_params)}", regex)
            rows = await conn.fetch(
                f"""
                WITH c AS (SELECT * FROM unnest($3::text[], $4::int[]) AS c(document_id, idx))
                SELECT {", ".join(f"p.{col.strip()}" for col in _PASSAGE_COLUMNS.split(","))},
                       {", ".join(f"d.{col.strip()}" for col in _DOC_COLUMNS.split(",") if col.strip() != "document_id")},
                       p.document_id, 1 - {self._distance("$1")} AS score, {keyword} AS keyword,
                       coalesce({matched}, false) AS matched,
                       (SELECT array_agg(v.role || ':' || (1 - {self._distance("$1", "v")})::text)
                        FROM {fq_table("kb_passage_vectors")} v
                        WHERE v.bank_id = p.bank_id AND v.document_id = p.document_id AND v.idx = p.idx) AS extra
                FROM c
                JOIN {fq_table("kb_passages")} p ON p.bank_id = $2 AND p.document_id = c.document_id AND p.idx = c.idx
                JOIN {docs} d ON d.bank_id = p.bank_id AND d.document_id = p.document_id
                """,
                *stage_params,
            )
        dense = with_document_prior(
            [r["document_id"] for r in rows],
            [
                page_score(float(r["score"]), _role_scores(r["extra"]), page_text_weight(r["text"], text_weight))
                for r in rows
            ],
        )
        keywords = [float(r["keyword"] or 0.0) for r in rows]
        scores = [
            score + (EXACT_MATCH_BONUS if r["matched"] else 0.0)
            for r, score in zip(rows, fuse_keyword(dense, keywords, keyword_weight), strict=True)
        ]
        hits = [
            SearchHit(passage=_passage(r), document=_document(r), score=sc) for r, sc in zip(rows, scores, strict=True)
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    async def grep(
        self,
        bank_id: str,
        pattern: str,
        *,
        top_k: int,
        filters: SearchFilters,
        regex: bool = False,
    ) -> list[SearchHit]:
        match = _text_match("p.text", "$2", regex)
        params: list[Any] = [bank_id, _match_value(pattern, regex), top_k]
        where = ""
        if filters.document_ids:
            params.append(filters.document_ids)
            where += f" AND p.document_id = ANY(${len(params)}::text[])"
        if filters.media_types:
            params.append(filters.media_types)
            where += f" AND d.media_type = ANY(${len(params)}::text[])"
        if filters.tags:
            params.append(filters.tags)
            op = "@>" if filters.tags_match == "all" else "&&"
            where += f" AND d.tags {op} ${len(params)}::text[]"
        # ponytail: a sequential ILIKE scan; add a pg_trgm GIN index on text when banks grow past ~1M passages.
        async with acquire_with_retry(await self._pool()) as conn:
            rows = await conn.fetch(
                f"""
                SELECT * FROM (
                    SELECT {", ".join(f"p.{col.strip()}" for col in _PASSAGE_COLUMNS.split(","))},
                           {", ".join(f"d.{col.strip()}" for col in _DOC_COLUMNS.split(",") if col.strip() != "document_id")},
                           p.document_id, count(*) OVER (PARTITION BY p.document_id) AS matches
                    FROM {fq_table("kb_passages")} p
                    JOIN {fq_table("kb_documents")} d ON d.bank_id = p.bank_id AND d.document_id = p.document_id
                    WHERE p.bank_id = $1 AND {match}{where}
                ) m
                ORDER BY matches DESC, document_id, idx LIMIT $3
                """,
                *params,
            )
        return [SearchHit(passage=_passage(r), document=_document(r), score=float(r["matches"])) for r in rows]

    async def timeseries(self, bank_id: str, *, since: datetime, trunc: str) -> list[TimeBucket]:
        if trunc not in ("hour", "day"):
            raise ValueError("trunc must be 'hour' or 'day'")
        async with acquire_with_retry(await self._pool()) as conn:
            rows = await conn.fetch(
                f"""
                SELECT date_trunc('{trunc}', created_at) AS t, COUNT(*) AS documents,
                       COALESCE(SUM(passage_count), 0) AS passages
                FROM {fq_table("kb_documents")} WHERE bank_id = $1 AND created_at >= $2
                GROUP BY t ORDER BY t
                """,
                bank_id,
                since,
            )
        return [TimeBucket(time=r["t"], documents=int(r["documents"]), passages=int(r["passages"])) for r in rows]

    async def bank_stats(self, bank_ids: list[str]) -> dict[str, BankStats]:
        if not bank_ids:
            return {}
        async with acquire_with_retry(await self._pool()) as conn:
            rows = await conn.fetch(
                f"""
                SELECT bank_id, media_type, status, COUNT(*) AS documents, COALESCE(SUM(passage_count), 0) AS passages,
                       COALESCE(SUM(size_bytes), 0) AS bytes
                FROM {fq_table("kb_documents")} WHERE bank_id = ANY($1::text[])
                GROUP BY bank_id, media_type, status
                """,
                bank_ids,
            )
        stats = {b: BankStats() for b in bank_ids}
        for r in rows:
            s = stats[r["bank_id"]]
            n = int(r["documents"])
            s.document_count += n
            s.passage_count += int(r["passages"])
            s.total_bytes += int(r["bytes"])
            s.by_media_type[r["media_type"]] = s.by_media_type.get(r["media_type"], 0) + n
            if r["status"] == "ready":
                s.ready_count += n
            elif r["status"] == "failed":
                s.failed_count += n
            else:
                s.pending_count += n
        return stats


def _version_at_least(version: str | None, minimum: tuple[int, int]) -> bool:
    try:
        major, minor = (int(x) for x in (version or "0.0").split(".")[:2])
    except ValueError:
        return False
    return (major, minor) >= minimum


def _role_scores(raw: list[str] | None) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for item in raw or []:
        role, _, value = item.partition(":")
        out.setdefault(role, []).append(float(value))
    return out


def page_score(primary: float, roles: dict[str, list[float]], text_weight: float) -> float:
    """A passage's score from all its vectors (PDF_GEMMA.md, EmbeddingGemma 2 on ViDoRe V3).

    A PDF page: image = mean(whole page, best band); text = best of (whole page text, best 128-token
    unit); score = (1 - w) * image + w * text, w = 0.6. That beat one blended page vector by ~7 nDCG@10.
    Every other passage has only its primary vector, so its score is that vector's similarity.
    """
    image = 0.5 * primary + 0.5 * max(roles["band"]) if roles.get("band") else primary
    texts = roles.get("page_text", []) + roles.get("unit", [])
    if not texts:
        return image
    return (1 - text_weight) * image + text_weight * max(texts)


#: A PDF page whose text layer is mostly not words (equations, tables, figure labels) leans on its image.
_POOR_TEXT_QUALITY = 0.5
_POOR_TEXT_WEIGHT = 0.3
#: Each passage gains this share of its document's best candidate score: the right PDF first, then the
#: page. +0.4 nDCG@10 over 8 ViDoRe V3 tasks, +1.0 on physics (PDF_GEMMA.md round 7; flat for 0.5-0.7).
DOCUMENT_PRIOR = 0.5


def text_quality(text: str | None) -> float:
    """Share of a page's characters inside real words (3+ letters); 0 below 200 characters."""
    chars = len(re.sub(r"\s", "", text or ""))
    if chars < 200:
        return 0.0
    return sum(len(w) for w in re.findall(r"[^\W\d_]{3,}", text or "")) / chars


def page_text_weight(text: str | None, text_weight: float) -> float:
    """The text side's weight for one page: lower when its text layer is junk (round 7: +0.3 nDCG@10)."""
    return _POOR_TEXT_WEIGHT if text_quality(text) < _POOR_TEXT_QUALITY else text_weight


def with_document_prior(document_ids: list[str], scores: list[float]) -> list[float]:
    best: dict[str, float] = {}
    for document_id, score in zip(document_ids, scores, strict=True):
        best[document_id] = max(score, best.get(document_id, score))
    # Divided by (1 + prior) so a one-passage document (an image, a sound) keeps its plain similarity.
    return [
        (score + DOCUMENT_PRIOR * best[d]) / (1 + DOCUMENT_PRIOR) for d, score in zip(document_ids, scores, strict=True)
    ]


def _or_tsquery(config: str, text: str) -> str:
    """An OR over the query's stemmed words, like BM25: the AND of a long question matches nothing."""
    return (
        f"(SELECT to_tsquery('simple', string_agg(quote_literal(w), ' | ')) "
        f"FROM unnest(tsvector_to_array(to_tsvector({config}, {text}))) w)"
    )


#: Added to the score of a passage that contains the search's ``pattern``. Fused scores stay within
#: [0, 1], so every exact match ranks above every passage that is only similar in meaning.
EXACT_MATCH_BONUS = 1.0


def _match_value(pattern: str, regex: bool) -> str:
    """The SQL parameter for ``pattern``: as is for a regex, else escaped and wrapped for ILIKE."""
    if regex:
        return pattern
    return "%" + pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _text_match(column: str, placeholder: str, regex: bool) -> str:
    return f"{column} ~* {placeholder}" if regex else f"{column} ILIKE {placeholder}"


def fuse_keyword(dense: list[float], keyword: list[float], weight: float) -> list[float]:
    """``(1 - w) * minmax(dense) + w * minmax(keyword)`` over the candidates; dense alone when no keyword hit."""
    if weight <= 0 or not any(k > 0 for k in keyword):
        return dense

    def scaled(values: list[float]) -> list[float]:
        lo, hi = min(values), max(values)
        return [(v - lo) / (hi - lo) if hi > lo else 0.0 for v in values]

    return [(1 - weight) * d + weight * k for d, k in zip(scaled(dense), scaled(keyword), strict=True)]


def load_knowledge_store(backend_provider: Any, extension_path: str | None) -> KnowledgeStore:
    """The configured store: an extension class when one is named, else Postgres."""
    if not extension_path:
        return PostgresKnowledgeStore(backend_provider)
    import importlib
    import os

    if ":" not in extension_path:
        raise ValueError(
            f"Invalid HINDSIGHT_API_KNOWLEDGE_STORE_EXTENSION '{extension_path}'. Expected 'module.path:ClassName'."
        )
    module_path, class_name = extension_path.rsplit(":", 1)
    cls = getattr(importlib.import_module(module_path), class_name, None)
    if not (isinstance(cls, type) and issubclass(cls, KnowledgeStore)):
        raise ValueError(f"Knowledge store extension '{class_name}' must inherit from KnowledgeStore.")
    prefix = "HINDSIGHT_API_KNOWLEDGE_STORE_"
    config = {
        k[len(prefix) :].lower(): v
        for k, v in os.environ.items()
        if k.startswith(prefix) and k != "HINDSIGHT_API_KNOWLEDGE_STORE_EXTENSION"
    }
    return cls(config)
