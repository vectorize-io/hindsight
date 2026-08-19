"""Extension interface for the *memories* slice of storage.

`memory_units` and the link tables around it (`memory_links`, `unit_entities`)
are the one part of the schema that is a search index as much as a table: every
recall arm — semantic, BM25, graph, temporal — is a query over them. This module
carves that slice out from behind the raw SQL so a different engine can own it,
without touching how documents, chunks, banks, operations or the entity registry
are stored.

The default :class:`~hindsight_api.engine.memories.postgres.PostgresMemories`
keeps everything exactly where it has always been: rows in `memory_units`, links
in `memory_links` and `unit_entities`, retrieval as SQL. It is what runs unless
an extension is configured, and it is the implementation the test suite
exercises.

An alternative implementation is loaded like any other Hindsight extension::

    HINDSIGHT_API_MEMORIES_EXTENSION=mypackage.memories:MyMemories
    HINDSIGHT_API_MEMORIES_SOME_SETTING=value

Such an implementation is the **sole store** for memories: no memory- or
link-shaped row reaches Postgres at all. Unit ids are minted by
:meth:`MemoriesExtension.allocate_unit_ids` rather than by an INSERT's RETURNING
clause, facts carry their entity ids and causal edges inline instead of becoming
join rows, and recall results come back fully populated with no Postgres
hydration. Everything else — documents, chunks, banks, the `entities` registry —
stays in Postgres either way.

Most operations are a method here, so the call chains route through the interface
rather than reimplement it per store; where the two differ, they usually differ by
what the method does — the Postgres implementation writes join rows and reprocesses
links, one that owns the store no-ops those passes and does its own thing. A handful
of call sites still branch on the two capability flags (``writes_memory_rows_in_sql``
for the inline-SQL fast paths, ``owns_document_store`` for the document/chunk bodies)
where the shapes are genuinely different; those are the seams, not accidental leaks.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ...extensions.base import Extension

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..search.retrieval import GraphRetriever


class MemoryTxn:
    """Opaque token for a cross-store write-group transaction, threaded from
    :meth:`MemoriesExtension.begin_txn` through the write calls to
    :meth:`MemoriesExtension.decide_txn`.

    A store that keeps memories in the same database as the caller's transaction has nothing
    to coordinate and returns ``None`` from ``begin_txn`` — the write methods then receive
    ``txn=None`` and behave exactly as before. A store that writes memories to a *separate*
    system subclasses this to carry whatever it needs to defer the writes'
    visibility until the caller's transaction is known to have committed. The base is
    deliberately empty: only the store that minted a handle interprets it."""


class StoreWriteUnavailable(RuntimeError):
    """The store cannot accept writes for this bank *right now*, but will shortly.

    Distinct from a failure: nothing is wrong, the bank is briefly closed to writes — a store
    migrating a bank between backends holds it for a few seconds while it takes the final delta
    and flips. The caller should retry rather than surface an error, which is why the API maps
    this to 503 with a `Retry-After` rather than a 5xx that reads as a bug.

    Raised from :meth:`MemoriesExtension.assert_writable` and from bank-scoped write methods.
    """

    #: Seconds a caller should wait before retrying. A cutover freeze is drain + a reconcile.
    retry_after: int = 30


# Keys used in an implementation's opaque metadata bag for the `memory_units`
# columns it has no first-class model of. These round-trip verbatim: they are
# stored without interpretation and returned on every hit, which is what lets
# recall rebuild a full result row without touching Postgres.
#
# Nothing here is queryable — an implementation cannot filter or sort on these. A
# column that retrieval must *filter* on has to be modelled properly instead.
META_CONTEXT = "context"
META_DOCUMENT_ID = "document_id"
META_CHUNK_ID = "chunk_id"
META_METADATA_JSON = "metadata_json"
META_OBSERVATION_SCOPES = "observation_scopes"
META_TEXT_SIGNALS = "text_signals"
META_CREATED_AT = "created_at"
#: When the memory last changed, and the contract every write path owes it (#3490):
#: a write that changes what the memory *is* — text, context, dates, fact_type, tags,
#: metadata, embedding, an observation's sources — stamps ``updated_at``, so a consumer
#: chasing ``WHERE updated_at > watermark`` sees the change. Those consumers are
#: incremental export, cache invalidation, the mental-model staleness check
#: (:meth:`any_memory_updated_since`) and its delta refresh — and recall's own
#: ``created_after`` / ``created_before`` window, which despite the name filters on this
#: column, so what stamps it also decides what a date-bounded recall returns.
#:
#: The consolidation *scheduler* is the one deliberate exception: when a pass records
#: that it folded a fact (or requeues one whose observation went away) it writes only
#: ``consolidated_at`` / ``consolidation_failed_at``, which are scheduler state rather
#: than the memory. Stamping there would make every pass look like an edit to every fact
#: it folded — re-flagging mental models stale and re-feeding unchanged facts to a delta
#: refresh. :meth:`MemoriesExtension.mark_consolidated` and the requeue sites that clear
#: the markers inline therefore leave the column alone.
#:
#: The exemption is that *situation*, not the two columns: a write that clears the markers
#: as part of a real change to the memory still stamps — :meth:`restore_memory` brings an
#: archived memory back and resets it for re-consolidation in one statement, and that is an
#: edit. A store that owns memories itself is expected to keep the same contract.
#:
#: No timestamp can report a hard delete; a consumer that must catch those needs a
#: content fingerprint, not a watermark.
META_UPDATED_AT = "updated_at"
# Observation bookkeeping. `source_memory_ids` is a JSON list: an implementation
# with no edge relation carries an observation's sources denormalised.
META_SOURCE_MEMORY_IDS = "source_memory_ids"
META_CONSOLIDATED_AT = "consolidated_at"
# A *positive* flag mirroring META_CONSOLIDATED_AT, because a metadata predicate
# can only match equality — there is no "key is absent". Consolidation's candidate
# query is "not yet consolidated", so it needs a value to match on: every memory is
# written with "0" and flipped to "1" once folded into an observation.
META_CONSOLIDATED_FLAG = "consolidated"
CONSOLIDATED_NO = "0"
CONSOLIDATED_YES = "1"

#: Prefix for the per-source metadata key an observation carries, one per source.
#: The forward list (:data:`META_SOURCE_MEMORY_IDS`) reads an observation's
#: sources; these read the other direction — "observations built on this fact" —
#: as an equality predicate rather than a corpus walk.
META_SOURCE_KEY_PREFIX = "src:"


def source_key(unit_id: str) -> str:
    """The metadata key marking an observation as built on ``unit_id``."""
    return f"{META_SOURCE_KEY_PREFIX}{unit_id}"


@dataclass
class CausalEdgeRecord:
    """A causal edge, resolved to the target's unit id."""

    target_unit_id: str
    relation_type: str  # "caused_by" for retain; legacy types on transfer import
    weight: float = 1.0


@dataclass
class StoredMemory:
    """A memory read by address rather than by ranking.

    What comes back from a get-by-id or a scan: no arm scores, because nothing
    ranked it. Shaped like a `memory_units` row so the callers that render one
    (the curation UI, export) need no second shape.
    """

    unit_id: str
    text: str
    fact_type: str
    context: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict | None = None
    proof_count: int = 1
    event_date: datetime | None = None
    occurred_start: datetime | None = None
    occurred_end: datetime | None = None
    mentioned_at: datetime | None = None
    created_at: datetime | None = None
    # Which observation scopes a memory is routed to. Consolidation reads it off
    # its candidates to decide which observation each one belongs in, so it has
    # to survive the round trip through the store.
    observation_scopes: list | None = None
    entity_ids: list[str] = field(default_factory=list)
    source_memory_ids: list[str] = field(default_factory=list)
    consolidated_at: datetime | None = None
    # Derived kNN edges `(target_unit_id, weight)`, populated only when the read
    # asked for them — the ranking path never does.
    semantic_edges: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class MemoryPatch:
    """A partial update to one memory. Unset fields are left alone.

    ``proof_count_delta`` is relative; everything else is an absolute set.
    ``metadata`` merges into the existing bag rather than replacing it.
    """

    unit_id: str
    text: str | None = None
    # Either a float list or the pgvector literal '[0.1,0.2,...]' — Hindsight
    # carries embeddings in both forms depending on the call site.
    embedding: list[float] | str | None = None
    tags: list[str] | None = None
    event_date: datetime | None = None
    occurred_start: datetime | None = None
    occurred_end: datetime | None = None
    mentioned_at: datetime | None = None
    metadata: dict[str, str] | None = None
    proof_count_delta: int = 0


@dataclass
class DeletePredicate:
    """Which memories a predicate-delete removes: type AND metadata AND tags.

    An empty predicate is refused unless ``delete_all`` — a stray empty filter
    must not be able to wipe a bank.
    """

    fact_types: list[str] | None = None
    metadata_equals: dict[str, str] | None = None
    tags: list[str] | None = None
    tags_match: str = "any"
    delete_all: bool = False

    def is_empty(self) -> bool:
        # A fact_type restriction is a real constraint, so a predicate carrying only
        # ``fact_types`` is NOT empty — it scopes the delete to those types (e.g. clearing
        # just a bank's observations), and must not be refused as a stray empty filter.
        return not self.metadata_equals and not self.tags and not self.fact_types


@dataclass
class ScanPage:
    """One page of a scan, plus the cursor for the next.

    ``next_page_token`` is empty when the walk is exhausted. It is a *position*,
    not a snapshot: concurrent writes can shift later pages, so a scan is
    eventually-complete browsing rather than a consistent iterator.
    """

    memories: list[StoredMemory] = field(default_factory=list)
    next_page_token: str = ""


@dataclass
class FactRecord:
    """One memory unit, as an implementation that owns the store needs to see it.

    There is no row behind this — it is the *whole* record — so it carries every
    column recall returns, plus the edges that would otherwise have become
    `memory_links` and `unit_entities` rows.
    """

    unit_id: str  # UUID string
    text: str
    # A float list, or the pgvector literal '[0.1,...]' — Hindsight produces both.
    embedding: list[float] | str
    fact_type: str
    tags: list[str] = field(default_factory=list)
    proof_count: int = 1
    context: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None
    metadata: dict | None = None
    observation_scopes: list | str | None = None
    # Entity names + spelled-out date tokens Hindsight folds into its BM25 document.
    text_signals: str | None = None
    event_date: datetime | None = None
    occurred_start: datetime | None = None
    occurred_end: datetime | None = None
    mentioned_at: datetime | None = None
    created_at: datetime | None = None
    # What would have become `unit_entities` rows: the entity registry stays in
    # Postgres, but the unit→entity posting travels with the memory.
    entity_ids: list[str] = field(default_factory=list)
    # What would have become causal `memory_links` rows.
    causal_edges: list[CausalEdgeRecord] = field(default_factory=list)
    # Observations only: the facts this observation was consolidated from.
    source_memory_ids: list[str] = field(default_factory=list)
    # When this memory was folded into an observation (sources only).
    consolidated_at: datetime | None = None

    def metadata_bag(self) -> dict[str, str]:
        """Render the non-modelled columns as an opaque str→str bag."""
        bag: dict[str, str] = {}
        if self.context:
            bag[META_CONTEXT] = self.context
        if self.document_id:
            bag[META_DOCUMENT_ID] = self.document_id
        if self.chunk_id:
            bag[META_CHUNK_ID] = self.chunk_id
        if self.metadata:
            bag[META_METADATA_JSON] = json.dumps(self.metadata)
        if self.observation_scopes is not None:
            bag[META_OBSERVATION_SCOPES] = json.dumps(self.observation_scopes)
        if self.text_signals:
            bag[META_TEXT_SIGNALS] = self.text_signals
        if self.created_at is not None:
            bag[META_CREATED_AT] = self.created_at.isoformat()
        # Hindsight filters recall's created_after/created_before window on
        # updated_at. A freshly written fact has updated_at == created_at.
        stamp = self.created_at
        if stamp is not None:
            bag[META_UPDATED_AT] = stamp.isoformat()
        if self.source_memory_ids:
            # Forward direction: the list, for reading an observation's sources back.
            bag[META_SOURCE_MEMORY_IDS] = json.dumps(self.source_memory_ids)
            # Backward direction: one key per source, so "observations built on
            # this fact" is an equality predicate rather than a corpus walk.
            for source_id in self.source_memory_ids:
                bag[source_key(source_id)] = "1"
        if self.consolidated_at is not None:
            bag[META_CONSOLIDATED_AT] = self.consolidated_at.isoformat()
        # Observations are not themselves consolidated, so only sources carry the flag.
        if self.fact_type != "observation":
            bag[META_CONSOLIDATED_FLAG] = CONSOLIDATED_YES if self.consolidated_at else CONSOLIDATED_NO
        return bag


def build_text_signals(fact) -> str | None:
    """Entity names + spelled-out dates — the enrichment Hindsight folds into BM25.

    Mirrors the signal construction the `memory_units` INSERT performs, so an
    implementation that owns the store produces the same searchable document the
    SQL path does.
    """
    parts: list[str] = []
    if fact.entities:
        parts.extend(e.name for e in fact.entities)
    stamps = [fact.occurred_start]
    if fact.occurred_end and fact.occurred_end != fact.occurred_start:
        stamps.append(fact.occurred_end)
    for stamp in stamps:
        if stamp is None:
            continue
        try:
            parts.append(stamp.strftime("%B %d %Y").lstrip("0").replace(" 0", " "))
        except (ValueError, AttributeError):
            pass
    return " ".join(parts) if parts else None


def build_fact_records(
    unit_ids: list[str],
    facts: list,
    document_id: str | None = None,
    unit_entity_ids: dict[str, list[str]] | None = None,
) -> list[FactRecord]:
    """Turn the retain pipeline's facts into records, edges resolved.

    ``unit_entity_ids`` is the unit→entity posting that would otherwise become
    `unit_entities` rows; causal relations become the memory's causal edges. Both
    travel with the memory, which is why a store that owns them writes once rather
    than inserting and then linking.

    Only called by implementations that own the store — the Postgres one already
    wrote all of this and never builds a record.
    """
    now = datetime.now(timezone.utc)
    records: list[FactRecord] = []
    for index, (unit_id, fact) in enumerate(zip(unit_ids, facts)):
        entity_ids = (unit_entity_ids or {}).get(str(unit_id))
        if entity_ids is None:
            entity_ids = [str(e.entity_id) for e in (fact.entities or []) if e.entity_id is not None]

        causal_edges = []
        for relation in fact.causal_relations or []:
            target = relation.target_fact_index
            # Targets are indices into this batch; a stale index would otherwise
            # produce an edge pointing at the wrong memory.
            if not isinstance(target, int) or not 0 <= target < len(unit_ids) or target == index:
                continue
            causal_edges.append(
                CausalEdgeRecord(target_unit_id=str(unit_ids[target]), relation_type=relation.relation_type)
            )

        records.append(
            FactRecord(
                unit_id=str(unit_id),
                text=fact.fact_text,
                embedding=fact.embedding,
                fact_type=fact.fact_type,
                tags=fact.tags or [],
                context=fact.context,
                document_id=fact.document_id or document_id,
                chunk_id=fact.chunk_id,
                metadata=fact.metadata,
                observation_scopes=fact.observation_scopes,
                text_signals=build_text_signals(fact),
                event_date=fact.occurred_start if fact.occurred_start is not None else fact.mentioned_at,
                occurred_start=fact.occurred_start,
                occurred_end=fact.occurred_end,
                mentioned_at=fact.mentioned_at,
                created_at=now,
                entity_ids=entity_ids,
                causal_edges=causal_edges,
            )
        )
    return records


@dataclass
class RelinkPassResult:
    """What one relink drain got through.

    ``queue_exhausted`` is False when the pass stopped on its deadline (or the
    runaway-iteration cap) with rows still queued — not a failure, since every
    batch commits before the next is claimed, but the caller needs to know the
    queue is not empty so it can arrange for the rest to be picked up.
    """

    units_processed: int = 0
    links_added: int = 0
    queue_exhausted: bool = True


@dataclass
class EntityPrunePassResult:
    """What one entity-prune drain got through.

    ``entities_examined`` counts candidates claimed, not rows deleted: most
    candidates turn out to be alive and are kept, which is the pass working as
    intended rather than wasted effort.
    """

    entities_examined: int = 0
    orphan_entities_pruned: int = 0
    stale_cooccurrences_pruned: int = 0
    queue_exhausted: bool = True


@dataclass
class RecallArms:
    """One fact_type's per-arm candidate lists from :meth:`MemoriesExtension.recall_unified`.

    Each list holds ``RetrievalResult`` items, unfused — RRF/rerank happen downstream.
    ``temporal`` is empty unless a window was given; ``graph`` is empty when that arm is off.
    """

    semantic: list = field(default_factory=list)
    bm25: list = field(default_factory=list)
    graph: list = field(default_factory=list)
    temporal: list = field(default_factory=list)


class MemoriesExtension(Extension, ABC):
    """Storage + retrieval for memory units and their links, behind one interface.

    Loaded with the ``MEMORIES`` prefix; see the module docstring. Subclasses get
    ``self.config`` (the ``HINDSIGHT_API_MEMORIES_*`` environment) and
    ``self.context`` from :class:`~hindsight_api.extensions.base.Extension`.

    Methods are grouped by what calls them: the retain write path, the recall
    arms, addressed reads for curation/export, and the maintenance passes. The
    Postgres implementation is the reference for what each one must mean.
    """

    @property
    def name(self) -> str:
        """Name for logs and the startup banner. Subclasses set a class-level ``name``
        (``PostgresMemories`` is ``"postgres"``); one that forgets reports its own class
        name rather than masquerading as another store in the banner."""
        return type(self).__name__

    #: Whether memories live as rows in the SQL ``memory_units`` table. True for the SQL stores
    #: (Postgres/Oracle), whose ``upsert_observation`` / ``delete_facts`` are no-ops because the
    #: consolidator writes those rows inline. A store that keeps memories elsewhere sets this
    #: False so the consolidator skips the inline SQL and routes the write through the store —
    #: then all of an observation's state lives wherever the store keeps it, not in Postgres.
    writes_memory_rows_in_sql: bool = True

    #: Whether this store owns the document/chunk BODIES — a document's extracted text, its chunk
    #: texts, and its original uploaded file. Default False: Postgres keeps ``documents.original_text``
    #: / ``chunks.chunk_text`` and the file goes through ``file_storage``. A store that sets this True
    #: owns a dedicated document store, so the retain and read paths route document/chunk
    #: bodies through the ``put_document`` / ``get_document_record`` / ``get_chunk_text`` /
    #: ``list_chunk_texts`` / ``count_chunks`` / ``document_content_hash`` methods below instead of
    #: the inline SQL. Cold, never-searched, key-based — see docs/documents-chunks.md.
    owns_document_store: bool = False

    def writes_memory_rows_in_sql_for(self, bank_id: str) -> bool:
        """Per-bank form of :attr:`writes_memory_rows_in_sql`. Defaults to the class attribute, so a
        single-store extension needs no override. A store that keeps different banks in different
        backends (some in SQL, some not) overrides this to answer PER BANK; every *bank-scoped* call
        site consults this instead of the class attribute, so mixed banks each take the correct path.
        (The few process-level gates — e.g. "is cross-store txn recovery relevant at all" — keep
        reading the class attribute.)"""
        return self.writes_memory_rows_in_sql

    def owns_document_store_for(self, bank_id: str) -> bool:
        """Per-bank form of :attr:`owns_document_store`. Defaults to the class attribute; a store
        that keeps some banks in a separate backend overrides it. See :meth:`writes_memory_rows_in_sql_for`."""
        return self.owns_document_store

    async def assert_writable(self, bank_id: str) -> None:
        """Refuse the operation if the store cannot take writes for this bank right now.

        Called at the entry to a *multi-store* operation — retain, which writes documents, chunks
        and entities through paths that are not this interface at all. Every write that does go
        through a store method is already covered by the method itself; this exists for the ones
        that are not, so a store can close a bank completely rather than only partly.

        The default is a no-op, so no existing store needs a change. A store that migrates banks
        between backends raises :class:`StoreWriteUnavailable` while a bank is mid-cutover: the
        window is seconds, and a retain that started before it and writes after it would land in
        the store that is about to stop being authoritative.
        """
        return None

    # ------------------------------------------------------------------ lifecycle

    async def initialize(self) -> None:
        """Open connections/channels. Called once during engine startup.

        Separate from :meth:`Extension.on_startup` because the memories store has
        to be live before the engine finishes booting, not alongside the HTTP app.
        """

    async def shutdown(self) -> None:
        """Release resources. Called during engine shutdown."""

    async def ensure_bank_storage(self, bank_id: str) -> None:
        """Ensure per-bank storage exists. Idempotent."""

    def allocate_unit_ids(self, count: int) -> list[str]:
        """Mint unit ids for a batch about to be written.

        The Postgres path never calls this — its ids come back from the INSERT's
        RETURNING clause — so this is what an implementation that owns the store
        uses to name memories before writing them.
        """
        return [str(uuid.uuid4()) for _ in range(count)]

    # ------------------------------------------------------------------ writes

    async def begin_txn(self, *, conn, fq_table, bank_id: str, mutating: bool) -> "MemoryTxn | None":
        """Open a cross-store write-group transaction around a unit of work, or ``None``.

        Called INSIDE the caller's database transaction, before the writes that belong to it.
        The returned handle is threaded (as ``txn=``) into every write of the unit and finally
        into :meth:`decide_txn` once the caller's transaction has settled.

        Default is ``None``: a store whose memories live in the caller's own database needs no
        cross-store coordination — its writes are already covered by that transaction, and the
        ``txn`` kwarg is ignored everywhere. A store that writes to a *separate* system returns
        a handle so those writes can be held invisible until the transaction is known to have
        committed. ``mutating`` distinguishes a unit that only creates new memories (safe to
        write plainly and compensate on abort) from one that changes or removes existing ones
        (whose previous value only deferred visibility can preserve)."""
        return None

    async def decide_txn(self, txn: "MemoryTxn | None", *, commit: bool) -> None:
        """Resolve a handle from :meth:`begin_txn` after its transaction settled.

        ``commit=True`` once the caller's transaction has COMMITTED, ``commit=False`` if it
        aborted. A no-op for ``None``. For a separate-store implementation this is where the
        held writes are made visible (commit) or discarded/compensated (abort)."""
        return None

    async def mint_txn(self, *, bank_id: str, mutating: bool) -> "MemoryTxn | None":
        """Mint a write-group handle WITHOUT opening a database transaction — the split form of
        :meth:`begin_txn` for a unit of work (consolidation) that runs slow work between its
        writes and must not hold a transaction across it. Tag the writes with the handle, then
        :meth:`write_txn_witness` + commit in one short transaction at the end, then
        :meth:`decide_txn`. Default ``None`` (no cross-store coordination)."""
        return None

    async def write_txn_witness(self, txn: "MemoryTxn | None", *, conn, fq_table) -> None:
        """Record a :meth:`mint_txn` handle's commit witness in the caller's transaction, just
        before it commits. No-op for ``None``."""
        return None

    async def recover_pending_txns(
        self,
        *,
        conn,
        fq_table,
        bank_ids: list[str],
        first_seen: dict[str, float],
        now: float,
        grace_seconds: float = 300.0,
        witness_ttl_seconds: float = 3600.0,
    ) -> int:
        """Backstop for a crashed writer: resolve each bank's undecided write-group txns against
        the witness table. Only a store that keeps memory rows outside SQL has cross-store txns to
        recover; the default (Postgres) has none, and the maintenance loop skips it. Returns the
        number of txns decided."""
        return 0

    @abstractmethod
    async def insert_facts(
        self,
        *,
        conn,
        ops,
        bank_id: str,
        facts: list,
        document_id: str | None = None,
        defer_index: bool = False,
        txn: "MemoryTxn | None" = None,
    ) -> list[str]:
        """Store a batch of extracted facts and return their unit ids, in order.

        ``defer_index`` asks for ids *without* the write, because the retain
        orchestrator can only supply entity ids and causal edges after Phase-1
        placeholders have been remapped onto real unit ids; it then calls
        :meth:`index_facts` with the complete picture. An implementation whose
        write is the row insert itself ignores the flag.

        ``conn`` and ``ops`` are the live Postgres connection and dialect ops,
        used only by an implementation that keeps its rows there.
        """

    async def index_facts(
        self,
        bank_id: str,
        unit_ids: list[str],
        facts: list,
        document_id: str | None = None,
        unit_entity_ids: dict[str, list[str]] | None = None,
    ) -> None:
        """Index facts whose ids came from a deferred :meth:`insert_facts`.

        A no-op by default: for Postgres the row *is* the index entry, so there is
        nothing left to do, and nothing is built. :func:`build_fact_records` turns
        the arguments into records for implementations that need them.
        """

    @abstractmethod
    async def delete_facts(self, bank_id: str, unit_ids: list[str], *, txn: "MemoryTxn | None" = None) -> None:
        """Remove units. Safe to call for ids that were never written."""

    async def delete_where(self, bank_id: str, predicate: DeletePredicate, txn=None) -> int:
        """Remove every memory matching ``predicate``. Returns the count when known.

        May be implemented lazily (recording the delete and materializing it
        later), in which case the returned count is 0 rather than a scan.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_document(
        self, *, conn, fq_table, bank_id: str, document_id: str, txn: "MemoryTxn | None" = None
    ) -> None:
        """Remove every memory belonging to ``document_id``.

        Called when a document is replaced, so it races the replacement's writes:
        an implementation must remove only what was written *before* this call,
        never the facts arriving moments later.
        """

    # ------------------------------------------------------ document/chunk bodies
    #
    # Only relevant when :attr:`owns_document_store` is True: a store that keeps document/chunk
    # BODIES (extracted text, chunk texts, original file) in its own dedicated store rather than in
    # ``documents.original_text`` / ``chunks.chunk_text`` / ``file_storage``. The retain and read
    # paths branch on ``owns_document_store`` and call these instead of the inline SQL. All bodies
    # are cold and never-searched; the document is passed whole (text + ordered chunk texts + file)
    # so the store can pack and dedup it — see docs/documents-chunks.md.

    async def put_document(
        self,
        *,
        bank_id: str,
        document_id: str,
        content_hash: str,
        original_text: "str | None",
        chunk_texts: list[str],
        tags: "list[str] | None" = None,
        metadata: "dict | None" = None,
        file_bytes: "bytes | None" = None,
        file_content_type: str = "",
        file_original_name: str = "",
        txn: "MemoryTxn | None" = None,
    ) -> None:
        """Store (or replace) a document's bodies: its extracted text, its ordered chunk texts, and
        optionally the original uploaded file. Idempotent by content — re-ingest re-uploads only
        what changed. Under a ``txn`` the record commits atomically with the retain's facts."""
        raise NotImplementedError

    async def document_content_hash(self, *, bank_id: str, document_id: str) -> "str | None":
        """The stored document's content hash, for the idempotent-skip check; ``None`` if absent."""
        raise NotImplementedError

    async def get_document_record(self, *, bank_id: str, document_id: str, include_text: bool = False) -> "dict | None":
        """A document's metadata (and, if asked, its extracted ``original_text``), or ``None``."""
        raise NotImplementedError

    async def get_chunk_text(self, *, bank_id: str, document_id: str, chunk_index: int) -> "str | None":
        """One chunk's text by position, or ``None`` if the document/index does not exist."""
        raise NotImplementedError

    async def list_chunk_texts(self, *, bank_id: str, document_id: str) -> "list[str] | None":
        """Every chunk's text in order, or ``None`` if the document does not exist."""
        raise NotImplementedError

    async def count_chunks(self, *, bank_id: str, document_id: str) -> int:
        """How many chunks a document has (0 if it does not exist)."""
        raise NotImplementedError

    async def delete_document_record(self, *, bank_id: str, document_id: str, txn: "MemoryTxn | None" = None) -> None:
        """Delete a document's RECORD and bodies from the document store — an EXPLICIT document
        deletion, distinct from :meth:`delete_document` (which drops only the document's facts on
        re-ingest and must not touch the record, since the replacement's ``put_document`` overwrites
        it). No-op for a store that does not own the document store."""
        raise NotImplementedError

    async def drop_bank_storage(self, bank_id: str) -> None:
        """Drop a bank's entire storage. Irreversible.

        A no-op for Postgres, where deleting the bank cascades to its rows.
        """

    async def delete_observations(self, *, conn, fq_table, bank_id: str, txn=None) -> None:
        """Remove every observation in a bank, leaving the facts behind it."""
        raise NotImplementedError

    async def update_memories(self, bank_id: str, patches: list[MemoryPatch], txn=None) -> None:
        """Apply partial updates. Only the fields set on each patch change."""
        raise NotImplementedError

    # ------------------------------------------------------------------ recall

    @abstractmethod
    async def recall_unified(
        self,
        *,
        conn,
        bank_id: str,
        fact_types: list[str],
        query_embedding: str,
        query_text: str,
        limit: int,
        temporal_window: "tuple[datetime, datetime] | None" = None,
        temporal_semantic_threshold: float = 0.1,
        tags: list[str] | None = None,
        tags_match: str = "any",
        tag_groups: list | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        min_semantic: float | None = None,
        min_keyword: float | None = None,
        enable_graph: bool = True,
    ) -> "dict[str, RecallArms]":
        """Run ALL retrieval arms for every fact_type — the whole recall interface, in one call.

        Returns ``{fact_type: RecallArms(semantic, bm25, graph, temporal)}`` of
        ``RetrievalResult``: the four per-arm candidate lists, unfused (RRF/rerank happen
        downstream, unchanged). ``temporal`` is empty unless ``temporal_window`` is given;
        ``graph`` is empty when ``enable_graph`` is False.

        This is the ONE method recall goes through — how a store answers the arms is entirely its
        own business. Postgres runs the split per-arm SQL orchestration behind this (a dense+BM25
        UNION query, a graph retriever per type, a temporal query); a store that owns its index
        answers every arm from a single query with no per-arm round-trips. Either way the caller
        sees only this method and its per-arm result.

        ``conn`` is the store's connection handle for the call. Postgres treats it as the pool it
        acquires its own connections from and runs the graph arm on; a store that reaches its index
        another way (e.g. over the network) ignores it.
        """

    def graph_retriever(self) -> "GraphRetriever | None":
        """The retriever backing the graph arm, or ``None`` to use the configured one.

        ``None`` means the links are in Postgres and ``config.graph_retriever``
        chooses among the SQL retrievers, as it always has. An implementation that
        owns the links returns its own, because the SQL retrievers would walk
        tables it never wrote to.
        """
        return None

    # ------------------------------------------------------------------ addressed reads
    #
    # Not retrieval: these serve the curation UI, export, consolidation and stats.
    # Every one has a `memory_units` query behind it in the Postgres implementation.

    @abstractmethod
    async def get_memories(self, *, conn, fq_table, bank_id: str, unit_ids: list[str]) -> list[StoredMemory]:
        """Fetch memories by id. Missing or deleted ids are simply absent."""

    @abstractmethod
    async def scan_memories(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        fact_types: list[str] | None = None,
        limit: int = 100,
        page_token: str = "",
        tags: list[str] | None = None,
        tags_match: str = "any",
        tag_groups: list | None = None,
        document_id: str | None = None,
        metadata_equals: dict[str, str] | None = None,
        skip: int = 0,
        include_edges: bool = False,
    ) -> ScanPage:
        """Page through stored memories.

        A full walk by construction — cost grows with the corpus — so this is for
        browsing and export, never for retrieval.

        ``document_id`` is its own filter rather than an entry in
        ``metadata_equals`` because it is not metadata everywhere: Postgres has a
        real column for it, and a store that keeps it in an opaque bag must still
        be asked the same question.

        ``tags_match`` selects a flat tag mode; ``tag_groups`` is the compound form
        (a list of AND/OR/NOT trees, AND-ed together) for conditions a flat filter
        cannot express, the same shape ``search`` takes. Both are AND-ed with
        ``metadata_equals``; a scan walks every member, so they filter what a page
        returns rather than what it reads.
        """

    @abstractmethod
    async def count_memories(self, *, conn, fq_table, bank_id: str) -> dict[str, int]:
        """Live memory count per fact_type."""

    @abstractmethod
    async def list_tags(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        pattern: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """One page of a bank's tag histogram, filtered/sorted/paged by the store.

        Returns ``{"items": [{"tag", "count"}], "total", "limit", "offset"}``.
        ``pattern`` is a case-insensitive wildcard (``*``); ordering is count
        descending then tag ascending. The store applies all three so a large
        histogram is never shipped whole for the caller to trim."""

    @abstractmethod
    async def find_unconsolidated(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        fact_types: list[str],
        limit: int,
        scope_tags: list[str] | None = None,
    ) -> list[StoredMemory]:
        """Memories not yet folded into an observation, oldest first.

        ``scope_tags`` restricts to memories carrying *every* one of them, the
        same containment the SQL ``tags @> scope`` expresses.
        """

    async def count_unconsolidated(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        fact_types: list[str],
        scopes: list[list[str] | None],
        limit: int,
    ) -> int:
        """How many unconsolidated candidates match *any* of ``scopes`` (deduped), capped at ``limit``.

        Drives the "is there work?" gate and the progress denominator, so a floor at ``limit`` on a
        huge backlog is harmless — but pulling whole rows just to count them is not, which is why
        this is its own method. This default dedupes :meth:`find_unconsolidated` across scopes and
        is correct for any store; a SQL store overrides it with a bounded ``COUNT(*)`` that never
        ships a row. ``scopes`` is ``[None]`` for the unscoped case, else one entry per scope.
        """
        seen: set[str] = set()
        for scope in scopes:
            for m in await self.find_unconsolidated(
                conn=conn, fq_table=fq_table, bank_id=bank_id, fact_types=fact_types, limit=limit, scope_tags=scope
            ):
                seen.add(m.unit_id)
                if len(seen) >= limit:
                    return limit
        return len(seen)

    async def find_failed_consolidation(self, *, conn, fq_table, bank_id: str) -> list[StoredMemory]:
        """Source memories the consolidator marked as permanently failed, for retry to requeue.

        Gated like ``find_unconsolidated``: a SQL store keeps the failure marker in a column and
        answers the retry inline, so this default is empty; a store that keeps memories outside
        SQL overrides it. Returns experience/world memories only (observations are never
        consolidated) — the caller clears them with ``mark_consolidated(when=None)``.
        """
        return []

    @abstractmethod
    async def mark_consolidated(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        unit_ids: list[str],
        when: datetime | None,
        failed: bool = False,
        txn: "MemoryTxn | None" = None,
    ) -> None:
        """Stamp (or clear, with ``when=None``) the consolidated marker on sources.

        ``failed`` stamps the failure marker instead, so a memory the LLM could
        not consolidate is not retried forever.

        This is scheduler state, not an edit: it must leave the memory's
        ``updated_at`` alone (see :data:`META_UPDATED_AT`).
        """

    @abstractmethod
    async def entity_memory_counts(
        self, *, conn, fq_table, bank_id: str, entity_ids: list[str] | None = None
    ) -> dict[str, int]:
        """Live memory count per entity id.

        Entities with no live memories are absent, so an id passed in and not
        returned is an orphan.
        """

    @abstractmethod
    async def entities_for_units(self, *, conn, fq_table, bank_id: str, unit_ids: list[str]) -> dict[str, list[str]]:
        """The entity ids each unit carries, keyed by unit id."""

    @abstractmethod
    async def entity_map_for_units(
        self, *, conn, fq_table, bank_id: str, unit_ids: list[str]
    ) -> dict[str, list[dict[str, str]]]:
        """``{unit_id: [{entity_id, canonical_name}]}`` — the named form recall renders.

        Like :meth:`entities_for_units` but carrying each entity's label, because
        recall shows the name on the fact. An observation with no direct postings
        inherits its source memories' entities, so a hit reads the same either way.
        """

    @abstractmethod
    async def resolve_entity_names(self, *, conn, fq_table, bank_id: str, entity_ids: list[str]) -> dict[str, str]:
        """``{entity_id: canonical_name}`` for the given ids, from the ``entities`` registry.

        The label half of :meth:`entity_map_for_units`, split out so a backend that
        already carries a unit's entity ids on the recalled result can turn those ids
        into names without re-fetching the memories — recall then builds the entity map
        from the result's ids plus this one lookup. Bank-scoped, and ids with no registry
        row are simply absent from the result. The concrete SQL is the store's, next to
        :meth:`entity_map_for_units`, because the query dialect belongs to the backend,
        not this interface.
        """

    @abstractmethod
    async def any_memory_updated_since(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        since: datetime,
        fact_types: list[str] | None = None,
        tags: list[str] | None = None,
        tags_match: str = "any",
        tag_groups: list | None = None,
    ) -> bool:
        """Whether any memory in the given scope was written after ``since``.

        Backs the mental-model staleness check, so it must be cheap: a bounded
        existence test, never a count. The scope is the mental model's — its flat
        tags or compound ``tag_groups``, plus an optional ``fact_types`` filter —
        so the same scope that gates a refresh decides whether one is due.
        """

    # ------------------------------------------------------------------ count surfaces
    #
    # The stats/admin views that aggregate memories by a key: consolidation
    # freshness, per-document counts, ingestion over time, observation scopes. For
    # Postgres each is one GROUP BY; a store without a queryable index over these
    # keys answers them by walking, so cost is O(matching) — acceptable for
    # admin/stats surfaces, and the reason these are their own methods rather than
    # uses of `count_memories`.

    async def consolidation_freshness(self, *, conn, fq_table, bank_id: str) -> dict[str, Any]:
        """``{"last_consolidated_at", "last_memory_write_at", "pending", "failed"}`` for a bank.

        ``pending`` / ``failed`` count the world/experience facts not yet folded
        into an observation, and those the LLM gave up on. Backs
        ``get_bank_freshness``, which reflect() calls often, so keep it cheap.

        ``last_memory_write_at`` is the newest write time (``updated_at``) across
        the bank's memories, or None for an empty bank. It is the bank-wide
        counterpart of :meth:`any_memory_updated_since`: a mental model whose
        ``last_memory_seen_at`` is at or after it cannot be stale, whatever its
        scope — which is how the stats and knowledge-tree surfaces answer "is
        this up to date" for many models without a scoped scan each.
        """
        raise NotImplementedError

    async def document_memory_counts(self, *, conn, fq_table, bank_id: str, document_ids: list[str]) -> dict[str, int]:
        """Live memory count per document id, for the documents named. Absent = 0."""
        raise NotImplementedError

    async def link_counts(self, *, conn, fq_table, bank_id: str) -> dict[str, int]:
        """``{link_type: count}`` of live links in a bank, for the stats page's link total.

        Keyed by link type (the caller sums the values); an absent type is zero. A store
        must answer from its own link representation — Postgres counts ``memory_links`` rows
        plus entity-derived edges; a store that keeps links inside the memory counts those —
        so the stats page never disagrees with the graph view about whether links exist.
        """
        raise NotImplementedError

    async def memories_timeseries(
        self, *, conn, fq_table, bank_id: str, time_field: str, trunc: str, since: datetime
    ) -> list[dict[str, Any]]:
        """``[{"bucket": datetime, "fact_type": str, "count": int}]`` since ``since``.

        Memories bucketed by ``time_field`` truncated to ``trunc`` (minute / hour /
        day) on UTC boundaries, broken down by fact_type — the caller fills the
        empty buckets. ``time_field`` is one of created_at / mentioned_at /
        occurred_start (the event-time fields fall back to created_at per memory).
        """
        raise NotImplementedError

    async def observation_scope_counts(self, *, conn, fq_table, bank_id: str) -> list[dict[str, Any]]:
        """``[{"tags": list[str], "count": int}]`` — observations grouped by scope.

        A scope is the sorted set of tags an observation was consolidated with;
        ``[]`` is the global (untagged) scope. Most-populous first.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ curation reads
    #
    # These back the curation UI and the bank/entity views. They page and filter,
    # which is why they are their own methods rather than uses of `scan_memories`:
    # a scan walks the corpus, and these must not.

    @abstractmethod
    async def list_memory_units(
        self,
        *,
        conn,
        ops,
        fq_table,
        bank_id: str,
        fact_type: str | list[str] | None = None,
        search_query: str | None = None,
        consolidation_state: str | None = None,
        state: str | None = None,
        document_id: str | None = None,
        entity_id: str | None = None,
        tags: list[str] | None = None,
        tags_match: str = "any",
        created_before: "datetime | None" = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """One page of the curation list: ``{"items": [...], "total": int}``.

        ``total`` is the count matching the filters, not the page size, because
        the UI pages on it.
        """

    @abstractmethod
    async def get_memory_unit(self, *, conn, ops, fq_table, bank_id: str, unit_id: str) -> dict[str, Any] | None:
        """One memory rendered for the curation detail view, or ``None``."""

    # ------------------------------------------------------------------ curation archive
    #
    # Invalidation is *structural*, not a flag: a memory the curator rejects is
    # moved out of every recall surface into an archive it can be restored from,
    # so recall / consolidation / graph never need a "valid?" predicate. The two
    # implementations realize the archive differently — Postgres moves the row to
    # a sibling table, a store that owns its memories moves it to a sibling
    # namespace — but the lifecycle is the same, so it lives behind these methods.

    @abstractmethod
    async def get_archived_memory(self, *, conn, fq_table, bank_id: str, unit_id: str) -> StoredMemory | None:
        """An *invalidated* memory read from the archive, or ``None``.

        Only invalidated memories are in the archive, so a live or missing id
        returns ``None`` — which is how a caller tells "invalidated" from "live"
        without a state column.
        """

    @abstractmethod
    async def invalidate_memory(
        self, *, conn, fq_table, bank_id: str, unit_id: str, reason: str | None, txn=None
    ) -> bool:
        """Move a live memory into the archive, out of every recall surface.

        Returns ``True`` if it was live and is now archived, ``False`` if there was
        no live memory with that id. The memory stays retrievable via
        :meth:`get_archived_memory` and restorable via :meth:`restore_memory`;
        ``reason`` is recorded alongside it.
        """

    @abstractmethod
    async def set_invalidation_reason(self, *, conn, fq_table, bank_id: str, unit_id: str, reason: str | None) -> None:
        """Update the recorded reason on a memory that is already archived."""

    @abstractmethod
    async def restore_memory(self, *, conn, fq_table, bank_id: str, unit_id: str, txn=None) -> StoredMemory | None:
        """Move an archived memory back to the live set, restoring its entity postings.

        Returns the restored memory (so the caller can recompute its embedding —
        the archive need not keep one), or ``None`` if it was not archived.

        Bringing a memory back is an edit, so this stamps ``updated_at`` even though
        it also resets the consolidation markers (see :data:`META_UPDATED_AT`).
        """

    @abstractmethod
    async def set_memory_embedding(self, *, conn, fq_table, bank_id: str, unit_id: str, embedding, txn=None) -> None:
        """Write a memory's embedding, recomputed by the caller.

        Its own method because the general :meth:`update_memories` is a no-op for
        the store whose write is the row itself — reverting or editing a memory has
        to put a freshly computed vector back on it, so this is a real write for
        both. ``embedding`` is a float list or the pgvector literal.

        The vector is part of the memory, so this stamps ``updated_at`` itself rather
        than leaning on the edit statement its in-tree callers happen to pair it with
        (see :data:`META_UPDATED_AT`).
        """

    async def clear_unit_entities(self, *, conn, fq_table, bank_id: str, unit_id: str) -> None:
        """Drop a unit's entity postings, ahead of an edit re-resolving them.

        A no-op for a store that keeps entity ids on the memory itself — the edit's
        rewrite replaces the whole set, so there is nothing to clear first.
        """

    async def apply_edit(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        unit_id: str,
        text: str,
        context: str | None,
        fact_type: str,
        occurred_start,
        occurred_end,
        event_date,
        mentioned_at,
        entity_ids: list[str] | None,
        txn=None,
    ) -> None:
        """Apply a curation field edit to a live memory.

        Writes the new text / context / fact_type / occurred window, resets the
        consolidation markers (the memory re-consolidates) and stamps the edit
        time, and drops the memory's derived links (they are recomputed). The
        embedding is *not* written here — the caller re-embeds from the new fields
        and calls :meth:`set_memory_embedding` after.

        ``entity_ids`` is the resolved entity set the memory should now carry; a
        store that keeps them on the memory writes them here, one that keeps them
        in a join table has already re-linked them and ignores this. ``None`` means
        the entity set was not part of this edit.
        """
        raise NotImplementedError

    @abstractmethod
    async def list_entities(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Entities in a bank with their ``mention_count``, paged and ordered by it.

        ``search`` is an optional case-insensitive substring match on the canonical
        name. Returns ``{items, total, limit, offset}``."""

    @abstractmethod
    async def graph_units(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        fact_type: str | None = None,
        search_query: str | None = None,
        document_id: str | None = None,
        chunk_id: str | None = None,
        tags: list[str] | None = None,
        tags_match: str = "all_strict",
        limit: int = 1000,
    ) -> dict[str, Any]:
        """Memory nodes for the graph view, plus the total matching count.

        Returns ``{"units": [...], "total": int}``: the page of nodes (newest
        first, capped at ``limit``) and how many match the filters. ``document_id``
        / ``chunk_id`` also match an observation whose sources carry them.
        """

    @abstractmethod
    async def graph_entity_rows(self, *, conn, fq_table, bank_id: str, unit_ids: list[str]) -> list[dict[str, Any]]:
        """``(unit_id, entity_id, canonical_name)`` rows for the graph view's entity edges."""

    @abstractmethod
    async def graph_direct_links(self, *, conn, fq_table, bank_id: str, unit_ids: list[str]) -> list[dict[str, Any]]:
        """Memory-to-memory edges among ``unit_ids`` for the graph view."""

    # ------------------------------------------------------------------ observations

    async def upsert_observation(
        self, *, conn, bank_id: str, record: FactRecord, txn: "MemoryTxn | None" = None
    ) -> None:
        """Write an observation, replacing any earlier one with the same id."""
        raise NotImplementedError

    @abstractmethod
    async def observations_for_sources(
        self, *, conn, ops, fq_table, bank_id: str, unit_ids: list[str]
    ) -> list[StoredMemory]:
        """Observations consolidated from any of ``unit_ids``."""

    @abstractmethod
    async def delete_stale_observations(self, *, conn, ops, fq_table, bank_id: str, fact_ids: list) -> int:
        """Delete observations built on ``fact_ids`` and requeue surviving sources.

        Returns how many observations were removed. Called whenever facts are
        replaced or deleted, so an observation never outlives the facts it
        summarises; sources that survive go back in the consolidation queue.
        """

    # ------------------------------------------------------------------ maintenance
    #
    # The graph-maintenance job orchestrates these; each pass asks the store to do
    # the part it owns. A store whose links are inline has nothing to relink and no
    # join table to sweep, so those passes are no-ops for it.

    async def record_unit_entities(
        self,
        *,
        conn,
        ops,
        fq_table,
        bank_id: str | None = None,
        unit_ids: list[Any],
        entity_ids: list[Any],
        txn: "MemoryTxn | None" = None,
    ) -> None:
        """Record the unit→entity postings for a batch of memories.

        ``unit_ids`` and ``entity_ids`` are parallel: a unit that mentions three
        entities appears three times. The `entities` registry itself stays in
        Postgres regardless; this is the join from a memory to the entities it
        mentions. ``bank_id`` is passed because a store that keeps the posting on
        the memory (rather than in a global join table) needs to know which
        namespace the units live in — the Postgres join is keyed by global unit id
        and ignores it.

        ``txn`` is the caller's write-group handle. For a store that keeps the
        posting ON the memory this call is a re-write of rows the same write-group
        already created, so it belongs to that group: passing the handle keeps the
        two writes atomic together and — for a store that records what its groups
        wrote — keeps this write inside the group's accounting. Ignored by the
        Postgres store, whose posting is an ordinary row in the caller's own
        transaction.
        """

    async def enqueue_relink_victims(
        self, *, conn, fq_table, bank_id: str, affected_unit_ids: list, include_affected_units: bool = False
    ) -> int:
        """Queue memories that lost a link when ``affected_unit_ids`` changed.

        Zero for a store with no link table to dangle: nothing can point at a
        deleted memory if the pointers travel inside the memories themselves.
        ``include_affected_units`` (also enqueue the affected units themselves, for
        edits that leave them live) is honoured only by a store with a link table.
        """
        return 0

    async def relink_pass(
        self, *, backend, fq_table, bank_id: str, config, deadline: float | None = None
    ) -> "RelinkPassResult":
        """Top up links for queued victims. All-zero when there is nothing to relink."""
        return RelinkPassResult()

    async def enqueue_entity_prune_candidates(self, *, conn, fq_table, bank_id: str, affected_unit_ids: list) -> int:
        """Queue the entities ``affected_unit_ids`` reference as prune candidates.

        Zero for a store that never wrote `unit_entities`: it has no entity
        postings to lose, so nothing can become an orphan.
        """
        return 0

    async def entity_prune_pass(
        self, *, backend, fq_table, bank_id: str, deadline: float | None = None
    ) -> "EntityPrunePassResult":
        """Prune queued candidate entities and the co-occurrences they stranded.

        All-zero when the store keeps no entity postings and so queues nothing.
        """
        return EntityPrunePassResult()


__all__ = [
    "CONSOLIDATED_NO",
    "CONSOLIDATED_YES",
    "META_CHUNK_ID",
    "META_CONSOLIDATED_AT",
    "META_CONSOLIDATED_FLAG",
    "META_CONTEXT",
    "META_CREATED_AT",
    "META_DOCUMENT_ID",
    "META_METADATA_JSON",
    "META_OBSERVATION_SCOPES",
    "META_SOURCE_KEY_PREFIX",
    "META_SOURCE_MEMORY_IDS",
    "META_TEXT_SIGNALS",
    "META_UPDATED_AT",
    "CausalEdgeRecord",
    "DeletePredicate",
    "EntityPrunePassResult",
    "FactRecord",
    "MemoriesExtension",
    "MemoryPatch",
    "MemoryTxn",
    "RelinkPassResult",
    "ScanPage",
    "StoredMemory",
    "build_fact_records",
    "build_text_signals",
    "source_key",
]
