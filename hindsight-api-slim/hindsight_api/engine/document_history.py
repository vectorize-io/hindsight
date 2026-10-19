"""Per-document change history, rebuilt from the operations log (#5638).

Observations are derived data, rebuilt whenever a fact under them changes. The rebuild deletes
the old ones outright — keeping every version would cost far more storage than the knowledge it
explains — so without this nothing records that they ever existed, and a rebuild that merged two
unrelated events cannot be found afterwards.

No table of its own: each change is written onto the async operation that carried it, as a
``document_events`` list in ``result_metadata`` — ids and counts only, never text, every id list
capped at :data:`MAX_IDS`. A retain or append lands on its own operation, a consolidation run on
its own, and a fact edit, invalidation or retag on the consolidation operation it queues (the one
that rebuilds what it removed). The history of a document is then a read over the bank's
operations. That bounds it by design: it reaches back as far as operations are kept
(``HINDSIGHT_API_OPERATION_RETENTION_DAYS``), and a change that ran without an operation — a
synchronous retain, or an edit with auto-consolidation off — is not in it.

Recording has two halves. The write sites deep in the pipeline (the stale-observation sweep, and
consolidation's create/update/delete) note what they touched on the :class:`ObservationChanges`
bound by :func:`track_observation_changes`; a no-op when nothing is bound. The operation that bound
it knows which document it was about and appends the events with :func:`append_operation_events`,
after it commits and best-effort: history must never fail the change it describes.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from .db_utils import acquire_with_retry
from .schema import fq_table

logger = logging.getLogger(__name__)

#: Ids kept per list on one event. The count is always exact; past this only the first ids are kept.
MAX_IDS = 100


class DocumentEventKind(StrEnum):
    """What a change was. A named enum, not a Literal: inline enums are named after their field by
    the generated Rust client, and ``kind`` already names the knowledge-base node enum there."""

    RETAIN = "retain"
    APPEND = "append"
    REPROCESS = "reprocess"
    RETAG = "retag"
    FACT_EDITED = "fact_edited"
    FACT_INVALIDATED = "fact_invalidated"
    FACT_RESTORED = "fact_restored"
    FACT_DELETED = "fact_deleted"
    OBSERVATIONS_CLEARED = "observations_cleared"
    CONSOLIDATION = "consolidation"


class ObservationIdSample(BaseModel):
    """How many observations an event touched, and (up to ``MAX_IDS``) which."""

    count: int = 0
    ids: list[str] = Field(default_factory=list)

    @classmethod
    def of(cls, ids: Iterable[str]) -> ObservationIdSample:
        unique = sorted(set(ids))
        return cls(count=len(unique), ids=unique[:MAX_IDS])


class DocumentEvent(BaseModel):
    """One change to a document, and its effect on the facts and observations built from it."""

    document_id: str
    kind: DocumentEventKind
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    operation_id: str | None = Field(
        default=None, description="The operation this event was recorded on (filled in when read)."
    )
    fact_id: str | None = Field(default=None, description="The fact a fact_* event changed.")
    facts_added: int = 0
    facts_removed: int = 0
    observations_created: ObservationIdSample = Field(default_factory=ObservationIdSample)
    observations_updated: ObservationIdSample = Field(default_factory=ObservationIdSample)
    observations_removed: ObservationIdSample = Field(default_factory=ObservationIdSample)


class DocumentHistoryResponse(BaseModel):
    """A document's change history, newest first. ``total``, ``limit`` and ``offset`` count operations:
    one operation can carry several events for the same document (an oversized retain in slices)."""

    document_id: str
    total: int
    limit: int
    offset: int
    items: list[DocumentEvent]


@dataclass
class ObservationChanges:
    """What the write sites did while an operation was tracked: observation id -> its source facts."""

    created: dict[str, list[str]] = field(default_factory=dict)
    updated: dict[str, list[str]] = field(default_factory=dict)
    removed: dict[str, list[str]] = field(default_factory=dict)
    # Facts handed to the stale-observation sweep: for a retain, the facts it retired.
    swept_facts: set[str] = field(default_factory=set)

    def source_fact_ids(self) -> set[str]:
        return {s for d in (self.created, self.updated, self.removed) for srcs in d.values() for s in srcs}

    def by_document(self, fact_document: dict[str, str]) -> dict[str, DocumentEvent]:
        """One consolidation event per document an observation stands on, given fact id -> document id."""
        per_doc: dict[str, ObservationChanges] = {}
        for obs_id, sources in self.created.items():
            for doc_id in {fact_document[s] for s in sources if s in fact_document}:
                per_doc.setdefault(doc_id, ObservationChanges()).created[obs_id] = sources
        for obs_id, sources in self.updated.items():
            for doc_id in {fact_document[s] for s in sources if s in fact_document}:
                per_doc.setdefault(doc_id, ObservationChanges()).updated[obs_id] = sources
        for obs_id, sources in self.removed.items():
            for doc_id in {fact_document[s] for s in sources if s in fact_document}:
                per_doc.setdefault(doc_id, ObservationChanges()).removed[obs_id] = sources
        return {
            doc_id: DocumentEvent(
                document_id=doc_id,
                kind=DocumentEventKind.CONSOLIDATION,
                observations_created=ObservationIdSample.of(changes.created),
                observations_updated=ObservationIdSample.of(changes.updated),
                observations_removed=ObservationIdSample.of(changes.removed),
            )
            for doc_id, changes in per_doc.items()
        }


def deletion_events(deleted: dict[str, str | None], changes: ObservationChanges) -> list[DocumentEvent]:
    """One ``fact_deleted`` event per document, given the deleted facts (fact id -> document id) and
    what the sweep removed: each document gets the observations that stood on its own facts."""
    by_doc: dict[str, set[str]] = {}
    for fact_id, document_id in deleted.items():
        if document_id:
            by_doc.setdefault(document_id, set()).add(fact_id)
    return [
        DocumentEvent(
            document_id=document_id,
            kind=DocumentEventKind.FACT_DELETED,
            fact_id=next(iter(facts)) if len(facts) == 1 else None,
            facts_removed=len(facts),
            observations_removed=ObservationIdSample.of(
                obs_id for obs_id, sources in changes.removed.items() if facts.intersection(sources)
            ),
        )
        for document_id, facts in by_doc.items()
    ]


# Child tasks copy the context, so the object bound here is the one every parallel batch notes on.
_changes: ContextVar[ObservationChanges | None] = ContextVar("document_history_changes", default=None)


@contextmanager
def track_observation_changes() -> Iterator[ObservationChanges]:
    """Collect the observation writes made inside the block (including in tasks it spawns)."""
    changes = ObservationChanges()
    token = _changes.set(changes)
    try:
        yield changes
    finally:
        _changes.reset(token)


def observation_changes_tracked() -> bool:
    """Whether a write site's notes would be kept — for the one note that costs a read."""
    return _changes.get() is not None


def note_observations_removed(removed: dict[str, list[str]], swept_facts: Iterable[str | uuid.UUID] = ()) -> None:
    """Record observations deleted (id -> source facts), and the facts whose change caused it."""
    changes = _changes.get()
    if changes is not None:
        changes.removed.update(removed)
        changes.swept_facts.update(str(f) for f in swept_facts)


def note_observation_written(observation_id: str, source_fact_ids: Iterable[str], *, created: bool) -> None:
    changes = _changes.get()
    if changes is not None:
        target = changes.created if created else changes.updated
        target[str(observation_id)] = [str(s) for s in source_fact_ids]


async def append_operation_events(pool: Any, operation_id: str | None, events: list[DocumentEvent]) -> None:
    """Add ``events`` to the operation's ``result_metadata.document_events``. Best-effort: logged, never raised.

    One path for both dialects — lock the row, merge in Python, write it back — because the
    Oracle rewriter cannot translate PostgreSQL's in-place jsonb append, and an operation row is
    written rarely enough that the lock costs nothing.
    """
    if not operation_id or not events:
        return
    table = fq_table("async_operations")
    try:
        async with acquire_with_retry(pool) as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    f"SELECT result_metadata FROM {table} WHERE operation_id = $1 FOR UPDATE", uuid.UUID(operation_id)
                )
                if row is None:
                    return
                metadata = conn.parse_json(row["result_metadata"]) or {}
                metadata["document_events"] = [
                    *(metadata.get("document_events") or []),
                    *(e.model_dump(mode="json", exclude={"operation_id"}) for e in events),
                ]
                await conn.execute(
                    f"UPDATE {table} SET result_metadata = $1::jsonb WHERE operation_id = $2",
                    json.dumps(metadata),
                    uuid.UUID(operation_id),
                )
    except Exception:
        logger.warning(f"Failed to record document history on operation {operation_id}", exc_info=True)


async def record_consolidation_events(
    pool: Any, bank_id: str, changes: ObservationChanges, operation_id: str | None
) -> None:
    """File a consolidation job's observation writes under the documents their sources came from."""
    fact_ids = changes.source_fact_ids()
    if not operation_id or not fact_ids:
        return
    from .memories import get_memories

    try:
        async with acquire_with_retry(pool) as conn:
            facts = await get_memories().get_memories(
                conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=sorted(fact_ids)
            )
    except Exception:
        logger.warning(f"Failed to resolve documents for consolidation history in bank {bank_id}", exc_info=True)
        return
    events = changes.by_document({m.unit_id: m.document_id for m in facts if m.document_id})
    await append_operation_events(pool, operation_id, list(events.values()))


async def list_document_events(
    pool: Any, bank_id: str, document_id: str, *, limit: int, offset: int
) -> DocumentHistoryResponse:
    """Read a document's events back out of the bank's operations, newest first.

    ponytail: no index serves the containment test, so this scans the bank's operations — bounded
    by operation retention; add an expression index on document ids if a bank's log gets huge.
    """
    table = fq_table("async_operations")
    async with acquire_with_retry(pool) as conn:
        if conn.backend_type == "oracle":
            # The rewriter turns any `result_metadata @> :N` into a parent_operation_id lookup, so
            # Oracle gets the JSON path filter spelled out.
            match = "JSON_EXISTS(result_metadata, '$.document_events[*]?(@.document_id == $d)' PASSING $2 AS \"d\")"
            needle: str = document_id
        else:
            match = "result_metadata->'document_events' @> $2::jsonb"
            needle = json.dumps([{"document_id": document_id}])
        total = await conn.fetchval(f"SELECT COUNT(*) FROM {table} WHERE bank_id = $1 AND {match}", bank_id, needle)
        rows = await conn.fetch(
            f"""
            SELECT operation_id, result_metadata FROM {table}
            WHERE bank_id = $1 AND {match}
            ORDER BY created_at DESC
            LIMIT $3 OFFSET $4
            """,
            bank_id,
            needle,
            limit,
            offset,
        )
        items = [
            DocumentEvent.model_validate({**event, "operation_id": str(row["operation_id"])})
            for row in rows
            for event in (conn.parse_json(row["result_metadata"]) or {}).get("document_events") or []
            if event.get("document_id") == document_id
        ]
    # Events on one operation are not in time order across operations (a consolidation operation
    # carries the edit that queued it), so order by when each change happened.
    items.sort(key=lambda e: e.created_at, reverse=True)
    return DocumentHistoryResponse(
        document_id=document_id, total=int(total or 0), limit=limit, offset=offset, items=items
    )
