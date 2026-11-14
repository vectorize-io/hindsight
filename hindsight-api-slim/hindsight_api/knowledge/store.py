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
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, cast

from ..engine.db_utils import acquire_with_retry
from ..engine.schema import fq_table
from ..schema_context import get_current_schema

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
    #: Only while the document still holds this file (see ``StatusUpdate``); otherwise it is skipped.
    storage_key: str | None = None
    #: Passages already staged for this version (see ``PassageStage``), committed with these: the document
    #: ends up with ``staged + len(passages)``, numbered on from ``staged``. 0 when nothing was staged.
    staged: int = 0


@dataclass
class PassageStage:
    """A piece of a document's next version, written out of sight: a document too big to hold whole is
    cut, embedded and staged in pieces, then committed with ``PassageReplacement(staged=n)``. Until that
    commit, searches and reads see the version it replaces, whole."""

    document_id: str
    passages: list[KnowledgePassage]
    #: Only while the document still holds this file; otherwise the piece is skipped.
    storage_key: str | None = None
    #: Passages staged before this piece. 0 starts the version over: anything staged before for the
    #: document (left by an ingest that never committed) is dropped first.
    staged: int = 0


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
    #: How ``score`` was reached, step by step; filled by the Postgres store, for a search's trace.
    explain: ScoreExplain | None = None


@dataclass
class ScoreExplain:
    """One hit's score, step by step, as ``search`` builds it."""

    #: Which candidate searches found it: "vector" (its main vector), "extra_vectors" (a PDF page's
    #: bands, text or text units), "keyword".
    found_by: list[str]
    #: Cosine similarity of the query and the passage's main vector.
    similarity: float
    #: The best similarity of each extra vector role (band, page_text, unit, sentence).
    roles: dict[str, float]
    #: ``page_score``: the similarity with the extra vectors mixed in.
    passage_score: float
    #: After the document prior: the average of the passage's score and its document's best.
    with_document: float
    #: The BM25-style keyword score (sum of the matched query words' IDF); 0 when no word matched.
    keyword: float
    #: A page read as a reference list, pushed down by ``REFERENCE_PENALTY``.
    reference_page: bool


@dataclass
class StoreSearchTrace:
    """What one store search did, filled in when the caller asks for a trace."""

    candidates: int = 0
    keyword_weight: float = 0.0
    #: Each query word the keyword search used, with its IDF in this bank.
    keyword_terms: dict[str, float] = field(default_factory=dict)
    #: Each candidate search's own list, best first: (document_id, index, its score in that search).
    arms: dict[ArmName, list[ArmHit]] = field(default_factory=dict)
    #: Every candidate scored, best first, before the depth cut keeps the top ``top_k``.
    scored: list[SearchHit] = field(default_factory=list)


#: The candidate searches: the passages' main vectors, their extra vectors, and keywords.
ArmName = Literal["vector", "extra_vectors", "keyword"]


@dataclass
class ArmHit:
    document_id: str
    index: int
    #: Cosine similarity for the vector searches, ts_rank_cd for the keyword one.
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
    #: When a document was last added or changed; None for an empty bank.
    last_document_at: datetime | None = None


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
    async def set_statuses(self, bank_id: str, updates: list[StatusUpdate]) -> None:
        """Set each document's status and error (see ``StatusUpdate``). A document set to "failed" also
        drops whatever it had staged (see ``PassageStage``): that version will never commit it."""

    @abstractmethod
    async def stage_passages(self, bank_id: str, items: list[PassageStage]) -> list[str]:
        """Write each piece out of sight, for the document's next version (see ``PassageStage``). Returns
        the documents skipped because they no longer hold the piece's ``storage_key`` (or are gone)."""

    @abstractmethod
    async def replace_passages(self, bank_id: str, items: list[PassageReplacement]) -> PassagesReplaced:
        """Swap each document's passages for the given ones, plus the ``staged`` ones staged before, and
        mark it ready; an item whose ``storage_key`` the document no longer holds is skipped, and reported
        as such. Whatever was staged for an item's document and not committed by it is dropped."""

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
    async def delete_some_documents(self, bank_id: str, limit: int) -> int:
        """Delete up to ``limit`` of the bank's documents with their passages, skipping any another
        transaction holds; returns how many went (0 with documents left means they are all held)."""

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
        trace: StoreSearchTrace | None = None,
    ) -> list[SearchHit]:
        """Best passages, each scored from all its vectors (``page_score``), keyword matches fused in."""

    @abstractmethod
    async def bank_stats(self, bank_ids: list[str]) -> dict[str, BankStats]: ...

    @abstractmethod
    async def timeseries(
        self, bank_id: str, *, since: datetime, trunc: str, media_type: str | None = None
    ) -> list[TimeBucket]:
        """Documents created per ``trunc`` ('hour' or 'day') since ``since``, with their passage counts;
        only documents of ``media_type`` when given."""


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
    "passage_count, tags, created_at, updated_at, source_document_id"
)
_PASSAGE_COLUMNS = "idx, modality, text, page, start_ms, end_ms, media_key"
#: The extra vectors that find candidates (and so sit in the HNSW index); the rest only rescore them.
RESCORE_ONLY_ROLES = ("sentence",)
#: How many of the best pages so far are rescored with their sentence vectors (stage 3 of ``search``).
#: 50 kept the gain of 100 within 0.05 nDCG@10 on ViDoRe hr and computer_science, for half the work.
RESCORE_POOL = 50
#: Most candidates one arm of stage 1 returns (see ``search``); never fewer than ``top_k``.
#: 300 kept ViDoRe hr nDCG@10 (65.68 vs 65.59 uncapped) and halved its search time (Gemini Embedding 2).
CANDIDATE_POOL_CAP = 300
CANDIDATE_ROLES_SQL = "role NOT IN (" + ", ".join(f"'{r}'" for r in RESCORE_ONLY_ROLES) + ")"


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
                await conn.execute(f"DROP INDEX IF EXISTS {fq_table(f'idx_{table_name}_candidates')}")
                await conn.execute(f"ALTER TABLE {passages} ALTER COLUMN embedding TYPE vector({dimension})")
            if dimension <= _VECTOR_INDEX_MAX:
                using = "hnsw (embedding vector_cosine_ops)"
            elif dimension <= _HALFVEC_INDEX_MAX:
                using = f"hnsw ((embedding::halfvec({dimension})) halfvec_cosine_ops)"
            else:
                using = None
                logger.warning(f"Knowledge passages at {dimension} dims cannot be indexed; search scans")
            if using and table_name == "kb_passage_vectors":
                # Sentence vectors only rescore pages that the other vectors found, so they stay out of the
                # HNSW graph: four times the rows in it made a ViDoRe hr ingest 2.7x slower and search 2x.
                # The search's candidate arm repeats this predicate, or the planner cannot use the index.
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table_name}_candidates ON {passages} USING {using} "
                    f"WHERE {CANDIDATE_ROLES_SQL}"
                )
                # Until _PIPELINE_VERSION became 10 one full index covered every vector; it was dropped then,
                # as it would still pay every sentence insert.
                await conn.execute(f"DROP INDEX IF EXISTS {fq_table(f'idx_{table_name}_embedding')}")
            elif using:
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
            async with conn.transaction():
                # The rows are locked in document_id order first (an UPDATE ... FROM locks in join order), the
                # order replace_passages and put_documents use, so overlapping batches cannot deadlock.
                rows = await conn.fetch(
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
                    RETURNING d.document_id, d.status
                    """,
                    bank_id,
                    [u.document_id for u in updates],
                    [u.status for u in updates],
                    [u.error for u in updates],
                    [u.storage_key for u in updates],
                )
                # A document that failed for good will not commit what it staged, so the pieces go now
                # (their vectors cascade). They used to stay until the document's next ingest started over,
                # and a failed document that was never ingested again kept them for ever. Only the rows the
                # update reached: a failure of an older version must not drop what a newer one is staging.
                # One case is left: the same task running twice at once (a worker's claim expired while
                # the first copy still runs) — one copy failing drops the other's pieces. Restaging after
                # a restart could always do that; the copy that commits then holds a short document.
                failed = [r["document_id"] for r in rows if r["status"] == "failed"]
                if failed:
                    await conn.execute(
                        f"DELETE FROM {fq_table('kb_staged_passages')} "
                        "WHERE bank_id = $1 AND document_id = ANY($2::text[])",
                        bank_id,
                        failed,
                    )

    async def _held(self, conn: Any, bank_id: str, document_ids: list[str]) -> dict[str, str | None]:
        """Each document's storage key, its row locked: an ingest of a newer version cannot swap the file
        between this check and the writes after it. Locked in document_id order, as set_statuses and
        put_documents take them, so two overlapping batches cannot deadlock on each other's rows."""
        return {
            r["document_id"]: r["storage_key"]
            for r in await conn.fetch(
                f"SELECT document_id, storage_key FROM {fq_table('kb_documents')} "
                "WHERE bank_id = $1 AND document_id = ANY($2::text[]) ORDER BY document_id FOR UPDATE",
                bank_id,
                document_ids,
            )
        }

    @staticmethod
    async def _insert_passages(
        conn: Any,
        bank_id: str,
        items: Sequence[PassageStage | PassageReplacement],
        table: str,
        vectors_table: str,
        keyword_index: bool,
    ) -> None:
        """The passages of ``items`` and their extra vectors, pipelined (executemany). Vectors go as binary
        float arrays cast to vector, not as their text: formatting and parsing 768 floats as text cost
        0.29 ms a vector against 0.08 (a ViDoRe hr ingest writes ~23k of them). ``keyword_index``: also
        write the keyword search's plain-word index (a staged piece gets it when its version commits)."""
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
                p.embedding,
            )
            for i in items
            for p in i.passages
        ]
        if rows:
            columns = "bank_id, document_id, idx, modality, text, page, start_ms, end_ms, media_key, embedding"
            values = "$1, $2, $3, $4, $5, $6, $7, $8, $9, $10::real[]::vector"
            if keyword_index:
                columns += ", search_vector"
                values += ", to_tsvector('simple', coalesce($5, ''))"
            await conn.executemany(f"INSERT INTO {table} ({columns}) VALUES ({values})", rows)
        extras = [
            (bank_id, i.document_id, p.index, v.role, v.n, v.embedding)
            for i in items
            for p in i.passages
            for v in p.extra_vectors
        ]
        if extras:
            await conn.executemany(
                f"INSERT INTO {vectors_table} (bank_id, document_id, idx, role, n, embedding) "
                "VALUES ($1, $2, $3, $4, $5, $6::real[]::vector)",
                extras,
            )

    async def stage_passages(self, bank_id: str, items: list[PassageStage]) -> list[str]:
        if not items:
            return []
        staged = fq_table("kb_staged_passages")
        async with acquire_with_retry(await self._pool()) as conn:
            async with conn.transaction():
                held = await self._held(conn, bank_id, [i.document_id for i in items])

                def holds(i: PassageStage) -> bool:
                    return i.document_id in held and (i.storage_key is None or held[i.document_id] == i.storage_key)

                skipped = [i.document_id for i in items if not holds(i)]
                items = [i for i in items if holds(i)]
                restart = [i.document_id for i in items if i.staged == 0]
                if restart:  # a version starting over drops what an ingest that never committed left
                    await conn.execute(
                        f"DELETE FROM {staged} WHERE bank_id = $1 AND document_id = ANY($2::text[])", bank_id, restart
                    )
                await self._insert_passages(
                    conn, bank_id, items, staged, fq_table("kb_staged_passage_vectors"), keyword_index=False
                )
        return skipped

    async def replace_passages(self, bank_id: str, items: list[PassageReplacement]) -> PassagesReplaced:
        if not items:
            return PassagesReplaced()
        table = fq_table("kb_passages")
        staged_table = fq_table("kb_staged_passages")
        # A fixed handful of statements in one transaction, however many documents: the old passages must
        # be gone before the new ones (same primary keys) go in, which one statement with data-modifying
        # CTEs cannot order. The inserts are executemany, pipelined in one round trip each.
        async with acquire_with_retry(await self._pool()) as conn:
            async with conn.transaction():
                # An item for a file the document no longer holds is left out.
                held = await self._held(conn, bank_id, [i.document_id for i in items])

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
                # The pieces staged for these versions move in, with their keyword index made as they do;
                # then whatever is left staged for these documents is dropped.
                with_staged = [i for i in items if i.staged]
                moved: list[Any] = []
                if with_staged:
                    moved = await conn.fetch(
                        f"""
                        INSERT INTO {table} (bank_id, document_id, idx, modality, text, page, start_ms, end_ms,
                            media_key, embedding, search_vector)
                        SELECT s.bank_id, s.document_id, s.idx, s.modality, s.text, s.page, s.start_ms, s.end_ms,
                            s.media_key, s.embedding, to_tsvector('simple', coalesce(s.text, ''))
                        FROM {staged_table} s
                        JOIN unnest($2::text[], $3::int[]) AS u(document_id, staged)
                          ON s.document_id = u.document_id AND s.idx < u.staged
                        WHERE s.bank_id = $1
                        RETURNING document_id, media_key
                        """,
                        bank_id,
                        [i.document_id for i in with_staged],
                        [i.staged for i in with_staged],
                    )
                    await conn.execute(
                        f"""
                        INSERT INTO {fq_table("kb_passage_vectors")} (bank_id, document_id, idx, role, n, embedding)
                        SELECT v.bank_id, v.document_id, v.idx, v.role, v.n, v.embedding
                        FROM {fq_table("kb_staged_passage_vectors")} v
                        JOIN unnest($2::text[], $3::int[]) AS u(document_id, staged)
                          ON v.document_id = u.document_id AND v.idx < u.staged
                        WHERE v.bank_id = $1
                        """,
                        bank_id,
                        [i.document_id for i in with_staged],
                        [i.staged for i in with_staged],
                    )
                await conn.execute(
                    f"DELETE FROM {staged_table} WHERE bank_id = $1 AND document_id = ANY($2::text[])",
                    bank_id,
                    document_ids,
                )
                await self._insert_passages(
                    conn, bank_id, items, table, fq_table("kb_passage_vectors"), keyword_index=True
                )
                moved_count: dict[str, int] = {}
                for r in moved:
                    moved_count[r["document_id"]] = moved_count.get(r["document_id"], 0) + 1
                await conn.execute(
                    f"UPDATE {fq_table('kb_documents')} d SET status = 'ready', error = NULL, "
                    "passage_count = u.passage_count, updated_at = now() "
                    "FROM unnest($2::text[], $3::int[]) AS u(document_id, passage_count) "
                    "WHERE d.bank_id = $1 AND d.document_id = u.document_id",
                    bank_id,
                    document_ids,
                    [moved_count.get(i.document_id, 0) + len(i.passages) for i in items],
                )
        new_keys = {p.media_key for i in items for p in i.passages} | {r["media_key"] for r in moved}
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

    async def delete_some_documents(self, bank_id: str, limit: int) -> int:
        table = fq_table("kb_documents")
        async with acquire_with_retry(await self._pool()) as conn:
            # SKIP LOCKED: a document an ingest is still writing is left for a later batch, rather than
            # this statement waiting on that write and running past the command timeout. The batch is
            # picked once (MATERIALIZED) and joined: as `IN (SELECT ... LIMIT ... SKIP LOCKED)` Postgres
            # re-ran the pick per row, each run took the next rows, and one batch deleted the whole bank.
            return await conn.fetchval(
                f"WITH picked AS MATERIALIZED (SELECT document_id FROM {table} WHERE bank_id = $1 "
                f"ORDER BY document_id LIMIT $2 FOR UPDATE SKIP LOCKED), "
                f"gone AS (DELETE FROM {table} t USING picked p WHERE t.bank_id = $1 "
                f"AND t.document_id = p.document_id RETURNING 1) SELECT count(*) FROM gone",
                bank_id,
                limit,
            )

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
        trace: StoreSearchTrace | None = None,
    ) -> list[SearchHit]:
        """Two stages. (1) Candidate passages: the nearest primary vectors, the nearest extra vectors
        (a page's bands, text, text units), and the best keyword matches of the query's rarer words.
        (2) Every candidate is scored from all its vectors with ``page_score``; keyword matches, when
        any, are fused in as ``(1 - w) * dense + w * keyword`` after min-max scaling over the
        candidates (PDF_GEMMA.md), a passage with no text by its dense score alone."""
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
        # Capped: past the HNSW scan width a bigger LIMIT makes the iterative scan walk on, and at 1000 per
        # arm (top_k 100) stage 1 took 266 ms of a 410 ms ViDoRe hr search; at 300 it takes 10 ms.
        params.append(max(100, min(top_k * 10, CANDIDATE_POOL_CAP), top_k))
        pool = f"${len(params)}"
        # Keyword words first: their rarity decides which take part. Plain words ('simple': no
        # language, no stemming), in every document. It was gated on a detected language (stopwords
        # of 6 languages, stemming per language): any other language never got keywords, and a name
        # or code never matched a document in another language than its query.
        terms = await self._keyword_terms(bank_id, query_text) if query_text and keyword_weight > 0 else {}
        if trace is not None:
            trace.keyword_terms = terms
        if terms:
            params.append(list(terms))
            tsquery = f"(SELECT to_tsquery('simple', string_agg(quote_literal(w), ' | ')) FROM unnest(${len(params)}::text[]) w)"
        docs = fq_table("kb_documents")

        def nearest(table: str) -> str:
            roles = f" AND {CANDIDATE_ROLES_SQL.replace('role', 'x.role')}" if table == "kb_passage_vectors" else ""
            arm = "vector" if table == "kb_passages" else "extra_vectors"
            return f"""
                SELECT x.document_id, x.idx, '{arm}' AS arm, 1 - {self._distance("$1", "x")} AS arm_score
                FROM {fq_table(table)} x
                JOIN {docs} d ON d.bank_id = x.bank_id AND d.document_id = x.document_id
                WHERE x.bank_id = $2{where}{roles}
                ORDER BY {self._distance("$1", "x")} LIMIT {pool}
            """

        arms = [nearest("kb_passages"), nearest("kb_passage_vectors")]
        if terms:
            arms.append(f"""
                SELECT x.document_id, x.idx, 'keyword' AS arm,
                       ts_rank_cd(x.search_vector, {tsquery}) AS arm_score FROM {fq_table("kb_passages")} x
                JOIN {docs} d ON d.bank_id = x.bank_id AND d.document_id = x.document_id
                WHERE x.bank_id = $2{where} AND x.search_vector @@ {tsquery}
                ORDER BY ts_rank_cd(x.search_vector, {tsquery}) DESC LIMIT {pool}
            """)
        async with acquire_with_retry(await self._pool()) as conn:
            async with conn.transaction():
                # A filtered HNSW scan otherwise stops at ef_search candidates and returns too few rows.
                await conn.execute("SET LOCAL hnsw.ef_search = 200")
                if self._iterative_scan:
                    await conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
                # Every search's own rows: the candidates are their union, and the trace shows each list.
                arm_rows = await conn.fetch(" UNION ALL ".join(f"({a})" for a in arms), *params)
            found_by: dict[tuple[str, int], list[str]] = {}
            for r in arm_rows:
                found_by.setdefault((r["document_id"], r["idx"]), []).append(r["arm"])
            candidates = list(found_by)
            if trace is not None:
                trace.candidates, trace.keyword_weight = len(candidates), keyword_weight
                for r in arm_rows:
                    trace.arms.setdefault(r["arm"], []).append(
                        ArmHit(r["document_id"], r["idx"], float(r["arm_score"] or 0.0))
                    )
                for arm_hits in trace.arms.values():
                    arm_hits.sort(key=lambda a: a.score, reverse=True)
            if not candidates:
                return []
            stage_params: list[Any] = [
                str(vector),
                bank_id,
                [document_id for document_id, _ in candidates],
                [idx for _, idx in candidates],
            ]
            keyword = "0"
            if terms:
                # BM25's IDF per word: rare words (names, codes) count, common ones barely (_keyword_terms).
                stage_params += [list(terms), list(terms.values())]
                keyword = (
                    "(SELECT coalesce(sum(t.idf), 0) FROM unnest($5::text[], $6::float8[]) t(w, idf) "
                    "WHERE p.search_vector @@ to_tsquery('simple', quote_literal(t.w)))"
                )
            rows = await conn.fetch(
                f"""
                WITH c AS (SELECT * FROM unnest($3::text[], $4::int[]) AS c(document_id, idx))
                SELECT {", ".join(f"p.{col.strip()}" for col in _PASSAGE_COLUMNS.split(","))},
                       {", ".join(f"d.{col.strip()}" for col in _DOC_COLUMNS.split(",") if col.strip() != "document_id")},
                       p.document_id, 1 - {self._distance("$1")} AS score, {keyword} AS keyword,
                       (SELECT array_agg(v.role || ':' || (1 - {self._distance("$1", "v")})::text)
                        FROM {fq_table("kb_passage_vectors")} v
                        WHERE v.bank_id = p.bank_id AND v.document_id = p.document_id AND v.idx = p.idx
                          AND {CANDIDATE_ROLES_SQL.replace("role", "v.role")}) AS extra
                FROM c
                JOIN {fq_table("kb_passages")} p ON p.bank_id = $2 AND p.document_id = c.document_id AND p.idx = c.idx
                JOIN {docs} d ON d.bank_id = p.bank_id AND d.document_id = p.document_id
                """,
                *stage_params,
            )
            roles = [_role_scores(r["extra"]) for r in rows]
            first = [
                page_score(float(r["score"]), ro, page_text_weight(r["text"], text_weight))
                for r, ro in zip(rows, roles, strict=True)
            ]
            # Stage 3: the best pages so far gain their sentence vectors. Scoring every candidate with
            # ~30 sentence vectors each cost a ViDoRe hr search 1.8x; the top RESCORE_POOL hold the gold.
            top = sorted(range(len(rows)), key=lambda i: first[i], reverse=True)[: max(top_k, RESCORE_POOL)]
            top = [i for i in top if rows[i]["page"] is not None]
            if top:
                found = await conn.fetch(
                    f"""
                    SELECT v.document_id, v.idx, max(1 - {self._distance("$1", "v")}) AS best
                    FROM {fq_table("kb_passage_vectors")} v
                    JOIN unnest($3::text[], $4::int[]) AS c(document_id, idx)
                      ON v.document_id = c.document_id AND v.idx = c.idx
                    WHERE v.bank_id = $2 AND v.role IN ({", ".join(f"'{r}'" for r in RESCORE_ONLY_ROLES)})
                    GROUP BY v.document_id, v.idx
                    """,
                    str(vector),
                    bank_id,
                    [rows[i]["document_id"] for i in top],
                    [rows[i]["idx"] for i in top],
                )
                best = {(f["document_id"], f["idx"]): float(f["best"]) for f in found}
                for i in top:
                    sentence = best.get((rows[i]["document_id"], rows[i]["idx"]))
                    if sentence is not None:
                        roles[i]["sentence"] = [sentence]
                        first[i] = page_score(
                            float(rows[i]["score"]), roles[i], page_text_weight(rows[i]["text"], text_weight)
                        )
        dense = with_document_prior([r["document_id"] for r in rows], first)
        keywords = [float(r["keyword"] or 0.0) for r in rows]
        reference = [r["page"] is not None and is_reference_page(r["text"]) for r in rows]
        has_text = [bool((r["text"] or "").strip()) for r in rows]
        scores = final_scores(dense, keywords, keyword_weight, reference=reference, has_text=has_text)
        hits = [
            SearchHit(
                passage=_passage(r),
                document=_document(r),
                score=scores[i],
                explain=ScoreExplain(
                    found_by=sorted(set(found_by.get((r["document_id"], r["idx"]), []))),
                    similarity=float(r["score"]),
                    roles={role: max(values) for role, values in roles[i].items() if values},
                    passage_score=first[i],
                    with_document=dense[i],
                    keyword=keywords[i],
                    reference_page=reference[i],
                ),
            )
            for i, r in enumerate(rows)
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        if trace is not None:
            trace.scored = hits
        return hits[:top_k]

    async def _keyword_terms(self, bank_id: str, query_text: str) -> dict[str, float]:
        """The query's words that take part in keyword search, each with its BM25 IDF in this bank:
        ln(1 + (N - df + 0.5) / (df + 0.5)) over the bank's passages, so rare words (names, codes) count
        and common ones barely. A word less than half as rare as the query's rarest one (IDF under
        ``KEYWORD_IDF_SHARE`` of the best) is left out: next to a name or a code, "the" and "of" say
        nothing about which passage answers, and matching them would pull in most of the bank. That
        drops them in any language and at any bank size, where a stopword list knew 6 languages.
        N comes from the documents' passage counts, not from counting passages: that count grew with the
        bank. Each df stops at ``_DF_CAP`` matches, so the GIN count of a common word stays bounded too."""
        docs = fq_table("kb_documents")
        async with acquire_with_retry(await self._pool()) as conn:
            rows = await conn.fetch(
                f"""
                WITH w AS (SELECT DISTINCT unnest(tsvector_to_array(to_tsvector('simple', $2))) AS w),
                     n AS (SELECT coalesce(sum(passage_count), 0)::float8 AS n FROM {docs} WHERE bank_id = $1)
                SELECT w.w, df.df, n.n FROM w, n,
                  LATERAL (SELECT count(*)::float8 AS df FROM (
                           SELECT 1 FROM {fq_table("kb_passages")} x
                           WHERE x.bank_id = $1 AND x.search_vector @@ to_tsquery('simple', quote_literal(w.w))
                           LIMIT {_DF_CAP}) m) df
                """,
                bank_id,
                query_text,
            )
        idf = {
            r["w"]: math.log(
                1 + max(float(r["n"]) - float(r["df"]), 0) / (float(r["df"]) + 0.5) + 0.5 / (float(r["df"]) + 0.5)
            )
            for r in rows
            if r["df"] > 0
        }
        best = max(idf.values(), default=0.0)
        return {w: v for w, v in idf.items() if v >= KEYWORD_IDF_SHARE * best}

    async def timeseries(
        self, bank_id: str, *, since: datetime, trunc: str, media_type: str | None = None
    ) -> list[TimeBucket]:
        if trunc not in ("hour", "day"):
            raise ValueError("trunc must be 'hour' or 'day'")
        async with acquire_with_retry(await self._pool()) as conn:
            rows = await conn.fetch(
                f"""
                SELECT date_trunc('{trunc}', created_at) AS t, COUNT(*) AS documents,
                       COALESCE(SUM(passage_count), 0) AS passages
                FROM {fq_table("kb_documents")} WHERE bank_id = $1 AND created_at >= $2
                  AND ($3::text IS NULL OR media_type = $3)
                GROUP BY t ORDER BY t
                """,
                bank_id,
                since,
                media_type,
            )
        return [TimeBucket(time=r["t"], documents=int(r["documents"]), passages=int(r["passages"])) for r in rows]

    async def bank_stats(self, bank_ids: list[str]) -> dict[str, BankStats]:
        if not bank_ids:
            return {}
        async with acquire_with_retry(await self._pool()) as conn:
            rows = await conn.fetch(
                f"""
                SELECT bank_id, media_type, status, COUNT(*) AS documents, COALESCE(SUM(passage_count), 0) AS passages,
                       COALESCE(SUM(size_bytes), 0) AS bytes, MAX(updated_at) AS last_at
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
            if r["last_at"] is not None and (s.last_document_at is None or r["last_at"] > s.last_document_at):
                s.last_document_at = r["last_at"]
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

    A PDF page: image = mean(whole page, best band); text = best of (its 128-token units, its sentences
    once the page is in the rescored top, and a whole-page text vector where an older ingest stored one);
    score = (1 - w) * image + w * text, w = ``knowledge_pdf_text_weight`` (default 0.7). That beat one
    blended page vector by ~7 nDCG@10.
    Every other passage has only its primary vector, so its score is that vector's similarity.
    """
    image = 0.5 * primary + 0.5 * max(roles["band"]) if roles.get("band") else primary
    texts = roles.get("page_text", []) + roles.get("unit", []) + roles.get("sentence", [])
    if not texts:
        return image
    return (1 - text_weight) * image + text_weight * max(texts)


#: A PDF page whose text layer is mostly not words (equations, tables, figure labels) leans on its image.
_POOR_TEXT_QUALITY = 0.5
_POOR_TEXT_WEIGHT = 0.3
#: Each passage gains this share of its document's best candidate score: the right PDF first, then the
#: page. +0.4 nDCG@10 over 8 ViDoRe V3 tasks, +1.0 on physics (PDF_GEMMA.md round 7; flat for 0.5-0.7).
#: 0.75 with sentence vectors: +0.1 on ViDoRe hr and computer_science over 0.5 (Gemini Embedding 2). 1.0, the
#: page and its document's best counting equally, ties 0.75 on hr for both EmbeddingGemma 2 and Gemini, so
#: it is the plain average with nothing left to tune per model.
DOCUMENT_PRIOR = 1.0


def text_quality(text: str | None) -> float:
    """Share of a page's characters inside real words (3+ letters); 0 below 200 characters."""
    chars = len(re.sub(r"\s", "", text or ""))
    if chars < 200:
        return 0.0
    return sum(len(w) for w in re.findall(r"[^\W\d_]{3,}", text or "")) / chars


_CITATION = re.compile(r"\((19|20)\d\d[a-z]?\)|\bvol\.|\bpp\.|\bdoi\b|https?://|\bet al\b|\bISBN\b|\bISSN\b", re.I)
#: A page this share of whose lines are citations is a reference list: on ViDoRe hr 5% of the candidates
#: and 0.1% of the gold pages. Its titles match every query on their topic, and it answers none.
_REFERENCE_LINES = 0.3
REFERENCE_PENALTY = 1.0


def is_reference_page(text: str | None) -> bool:
    lines = [line for line in (text or "").splitlines() if line.strip()]
    return len(lines) >= 5 and sum(bool(_CITATION.search(line)) for line in lines) / len(lines) > _REFERENCE_LINES


def page_text_weight(text: str | None, text_weight: float) -> float:
    """The text side's weight for one page: lower when its text layer is junk (round 7: +0.3 nDCG@10)."""
    return page_text_weight_for_quality(text_quality(text), text_weight)


def page_text_weight_for_quality(quality: float, text_weight: float) -> float:
    """``page_text_weight`` from a page's ``text_quality``, for a store that works it out once, when the
    page is written, rather than with a regex over every candidate's text at search time. An extension
    point: this repo's Postgres store scores at search time and reaches it through ``page_text_weight``;
    a store loaded as an extension (``load_knowledge_store``) can call it directly."""
    return _POOR_TEXT_WEIGHT if quality < _POOR_TEXT_QUALITY else text_weight


def with_document_prior(document_ids: list[str], scores: list[float]) -> list[float]:
    best: dict[str, float] = {}
    for document_id, score in zip(document_ids, scores, strict=True):
        best[document_id] = max(score, best.get(document_id, score))
    # Divided by (1 + prior) so a one-passage document (an image, a sound) keeps its plain similarity.
    return [
        (score + DOCUMENT_PRIOR * best[d]) / (1 + DOCUMENT_PRIOR) for d, score in zip(document_ids, scores, strict=True)
    ]


#: A query word takes part in keyword search when its IDF is at least this share of its rarest word's.
KEYWORD_IDF_SHARE = 0.5
#: Where a query word's document frequency stops being counted (see the IDF in ``search``).
_DF_CAP = 50_000


def final_scores(
    dense: list[float], keyword: list[float], weight: float, *, reference: list[bool], has_text: list[bool]
) -> list[float]:
    """Dense fused with keywords, then the reference-list demotion. The demotion comes after the fusion:
    inside it, a page at -1 stretched the min-max range of the dense scores, so keywords alone ordered
    the rest (ViDoRe hr English: 58.8 nDCG@10 instead of 67.2). An exact-text bonus (`pattern`) and a
    grep-only search lived here too; no benchmark used them and they were dropped."""
    return [
        score - (REFERENCE_PENALTY if ref else 0.0)
        for score, ref in zip(fuse_keyword(dense, keyword, weight, has_text), reference, strict=True)
    ]


def fuse_keyword(dense: list[float], keyword: list[float], weight: float, has_text: list[bool]) -> list[float]:
    """``(1 - w) * minmax(dense) + w * minmax(keyword)`` over the candidates; dense alone when no keyword hit.

    A passage with no text (an image, a clip with no transcript) cannot match a keyword: it scores
    ``minmax(dense)`` alone, by meaning. With a keyword of 0 it topped out at ``1 - w``, so in a bank of
    text and media every image ranked below text of the same meaning that matched a word."""
    if weight <= 0 or not any(k > 0 for k in keyword):
        return dense

    def scaled(values: list[float]) -> list[float]:
        # All equal (one candidate, or a tie): all equally the best. 0.0 here scored a one-page bank's
        # page 0, once keywords ran for every query rather than for a detected language only.
        lo, hi = min(values), max(values)
        return [(v - lo) / (hi - lo) if hi > lo else 1.0 for v in values]

    return [
        (1 - weight) * d + weight * k if text else d
        for d, k, text in zip(scaled(dense), scaled(keyword), has_text, strict=True)
    ]


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
