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
of call sites still branch on the one capability flag, ``store_owned`` — whether the
store owns its writes or the caller issues them as SQL — where the shapes are
genuinely different; those are the seams, not accidental leaks.
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


class StoreWriteConflict(RuntimeError):
    """A conditional write lost its race: the state it was based on moved before it committed.

    Raised by a store whose write carried a precondition — the read-modify-write case, where the
    caller read something, derived a new value from it, and asked the store to accept that value
    only if nothing had changed underneath. Nothing was written.

    Distinct from :class:`StoreWriteUnavailable`: that one means "not now, try again shortly" and
    the same write will succeed unchanged. This one means the write is STALE — retrying it as-is
    would re-apply a decision made on an old base. The caller has to re-read and redo the work,
    which is what makes concurrent appends to one document safe rather than last-writer-wins.
    """


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
    # Write time, as opposed to the four content times above: when the memory was last
    # written, which is the watermark a caller compares against to detect a change. Distinct
    # from `created_at`, which never moves after the first write.
    updated_at: datetime | None = None
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
    # Intrinsic causal edges the memory was written with, same shape the write model
    # carries. Populated only when the read asked for edges. A store that keeps memories
    # outside SQL has no `memory_links` table to reconstruct these from, so without them
    # on the read model an export of such a bank silently loses every causal relation.
    causal_edges: list[CausalEdgeRecord] = field(default_factory=list)


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
                # Unpacked here on purpose: `FactRecord` is the contract every store
                # implementation reads, and retain's packed `array("f")` is an internal
                # carrying format (#3756). The list exists only for the record.
                embedding=list(fact.embedding),
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


@dataclass(frozen=True)
class MemoryScopeWatermark:
    """One "has this scope changed?" question, for the batched staleness check.

    ``key`` is opaque to the store — it is whatever the caller wants the answer
    reported under (a mental-model id, a knowledge-page id) — and the scope
    fields are the same ones :meth:`MemoriesExtension.any_memory_updated_since`
    takes for a single model, so the two surfaces cannot drift apart.
    """

    key: str
    since: datetime
    fact_types: list[str] | None = None
    tags: list[str] | None = None
    tags_match: str = "any"
    tag_groups: list | None = None


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


@dataclass
class FullRecallRequest:
    """Everything a store needs to answer a whole recall — see
    :meth:`MemoriesExtension.full_recall`.

    This is deliberately the ENGINE's resolved values, not the caller's raw request: the budget
    has already been resolved from the bank config, the arm toggles and the reranker mode have
    been decided, and ``now`` is whatever the request said to score recency against. A store
    receiving this needs no access to configuration, which is what keeps product policy out of a
    store release — change the bank config and the next call carries the new values.

    Everything after :attr:`enable_graph` is a stage the engine would otherwise run itself.
    """

    # ---- what to retrieve (mirrors `recall_unified`) ----
    bank_id: str
    fact_types: "list[str]"
    query_embedding: str
    #: The user's question. Used BOTH for the full-text arm and as the reranker's query — a store
    #: with text search off still needs it for the second.
    query_text: str
    limit: int
    temporal_window: "tuple[datetime, datetime] | None" = None
    temporal_semantic_threshold: float = 0.1
    tags: "list[str] | None" = None
    tags_match: str = "any"
    tag_groups: "list | None" = None
    created_after: "datetime | None" = None
    created_before: "datetime | None" = None
    min_semantic: "float | None" = None
    min_keyword: "float | None" = None
    enable_text_search: bool = True
    enable_graph: bool = True

    # ---- how to rank ----
    #: ``"rrf"``, ``"interleave"`` or ``"cross_encoder"`` — the engine's ``reranking`` mode. The
    #: first two are passthrough (the fusion order stands); only the third calls a reranker.
    reranking: str = "rrf"
    reranker_max_candidates: int = 0
    per_source_cap: int = 0
    #: Arm name -> priority level, from ``HINDSIGHT_API_RECALL_STRATEGY_BOOSTS``.
    strategy_boosts: "dict[str, str] | None" = None
    recency_decay_function: str = "linear"
    recency_decay_linear_window_days: float = 365.0
    recency_decay_halflife_days: float = 90.0
    #: What recency is measured from — ``question_date`` when the caller gave one, else now.
    now: "datetime | None" = None
    min_reranker: "float | None" = None
    min_final: "float | None" = None

    # ---- what to return ----
    #: Results are truncated to this before the token budget is spent.
    truncate_to: int = 0
    max_tokens: int = 4096
    #: The BPE vocabulary token counts are computed against. Travels because the count decides
    #: which results come back, so the store must count with the same table the engine does.
    tokenizer_encoding: str = "o200k_base"
    include_entities: bool = False
    include_chunks: bool = False
    max_chunk_tokens: int = 4096

    # ---- shapes a store may not implement ----
    #: Set when the caller asked for something beyond the stages above. A store that sees any of
    #: these and does not implement them must decline (return ``None``) rather than silently
    #: answering a different question.
    prefer_observations: bool = False
    include_source_facts: bool = False


@dataclass
class KnowledgePageEntry:
    """One knowledge page as the store indexes it.

    Only what a search needs. The page's own row — its name, body, folder, trigger, history —
    stays in Postgres, which remains the authority; this is the derived half.
    """

    #: The mental model's id, and the id every match comes back under.
    page_id: str
    #: What full-text search matches on. The page name and body joined; never returned.
    index_text: str
    #: The page's embedding. ``None`` indexes it for text search only.
    embedding: list[float] | None = None
    #: The page's visibility tags, so a scoped search filters inside the store rather than
    #: over-fetching and discarding — a discarded hit has already cost a top-k slot.
    tags: list[str] = field(default_factory=list)
    #: When the page's row last changed. Read back by :meth:`MemoriesExtension.list_knowledge_pages`
    #: so a reconcile can spot a stale index entry without reading the page.
    updated_at: datetime | None = None


@dataclass
class KnowledgePageRef:
    """A page as the reconcile pass sees it: what the store holds and how old it thinks it is."""

    page_id: str
    updated_at: datetime | None = None


@dataclass
class KnowledgePageMatch:
    """One search result. ``score`` is comparable within a result set, not across stores."""

    page_id: str
    score: float


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

    #: Whether this store OWNS ITS WRITES, rather than the caller writing them as SQL.
    #:
    #: One question, because in practice there has only ever been one: a store either keeps
    #: everything itself — the memory rows, the document/chunk bodies, and the whole retain — or it
    #: keeps none of them and the caller issues the SQL inside its own transaction. This used to be
    #: three separate flags (``writes_memory_rows_in_sql``, ``owns_document_store``,
    #: ``store_owned_retain``) asking that same question in three places, two of them in the
    #: opposite polarity to the third, and no store ever set a mixed combination.
    #:
    #: False (the default) is the SQL stores, Postgres and Oracle, and nothing about how they work
    #: changes: memories are rows in ``memory_units``, a document's text is
    #: ``documents.original_text`` and its chunks are ``chunks.chunk_text``, the retain runs its
    #: Phase-1 entity resolution in SQL, and this extension's write methods are no-ops because the
    #: caller already wrote them — inside a transaction that makes the whole re-ingest atomic.
    #:
    #: True is a store that keeps memories elsewhere. It owns a dedicated document store (bodies go
    #: through ``put_document`` / ``get_document_record`` / ``get_chunk_text`` / ``list_chunk_texts``
    #: / ``count_chunks`` / ``document_content_hash``), resolves entity NAMES itself, and commits the
    #: entire retain — resolution, upserts and the document replace — as ONE atomic server-side call
    #: (``retain``). The orchestrator then needs no Postgres connection phase for it.
    #:
    #: Bank-scoped via :meth:`store_owned_for`, for a router whose banks live in different backends.
    store_owned: bool = False

    def store_owned_for(self, bank_id: str) -> bool:
        """Per-bank form of :attr:`store_owned`. Defaults to the class attribute, so a
        single-backend extension needs no override. A router whose banks live in different backends
        overrides this to answer PER BANK; every bank-scoped call site consults it rather than the
        attribute."""
        return self.store_owned

    # -- the knowledge-page index -------------------------------------------
    #
    # Knowledge pages are the one place where a store holds a DERIVED copy rather than the
    # authority. The page's row stays in Postgres; the store keeps only an index over its text and
    # embedding, so a lost or diverged entry is repaired by indexing it again and the two never need
    # a transaction between them.
    #
    # That is why this is its own flag rather than part of `store_owned`. `store_owned` collapsed
    # three flags into one because no store ever wanted a mixed combination — this is the mixed
    # combination: the ownership split runs the opposite way (Postgres keeps the row, the store
    # keeps the index), and a store can own every memory in a bank while its pages are still
    # searched in SQL. The rollout needs exactly that state.

    #: Whether this store indexes knowledge pages for search.
    #:
    #: False (the default) leaves `search_knowledge_pages` and the reflect tool on their SQL
    #: queries against `mental_models` — the ANN index over `mental_models.embedding` and the BM25
    #: index over its name + content. Nothing changes.
    #:
    #: True moves BOTH of those reads to this store, and the page write paths additionally call
    #: :meth:`index_knowledge_pages` / :meth:`delete_knowledge_pages`. The Postgres row is still
    #: written first and is still what a hit is hydrated from.
    owns_knowledge_index: bool = False

    def owns_knowledge_index_for(self, bank_id: str) -> bool:
        """Per-bank form of :attr:`owns_knowledge_index`. Defaults to the class attribute; a router
        whose banks live in different backends overrides it. Every bank-scoped call site consults
        this rather than the attribute."""
        return self.owns_knowledge_index

    async def index_knowledge_pages(self, bank_id: str, entries: list["KnowledgePageEntry"]) -> None:
        """Upsert pages into the store's index. A ``page_id`` already present is replaced.

        Called after the Postgres row is committed, so an entry here always describes a row that
        exists. The reverse — a committed row whose indexing failed — is the expected failure and is
        repaired by the reconcile pass, not by a transaction.

        Replacing rather than merging is deliberate: a derived index has nothing worth preserving
        across a rewrite, and it makes re-indexing everything idempotent, which is what lets the
        reconcile pass be "put them all again"."""
        raise NotImplementedError("this store does not index knowledge pages")

    async def delete_knowledge_pages(self, bank_id: str, page_ids: list[str]) -> None:
        """Remove pages from the index. Deleting one the store does not hold must be a no-op, so a
        reconcile can reap an entry it believes is stale without first proving it is there."""
        raise NotImplementedError("this store does not index knowledge pages")

    async def search_knowledge_pages(
        self,
        bank_id: str,
        *,
        embedding: list[float] | None,
        text: str,
        limit: int,
        tags: list[str] | None = None,
        tags_match: str = "any",
        tag_groups: list | None = None,
    ) -> list["KnowledgePageMatch"]:
        """Hybrid search over the bank's pages: text and (when given) embedding, fused BY THE STORE.

        Fusion is the store's business because only it knows what its arms produce — one returning
        ranks and another returning distances cannot share a caller-side formula. ``score`` is
        therefore comparable within one result set and meaningless across stores; callers order by
        it and do not otherwise interpret it.

        Returns ids only. The caller joins them back to Postgres for name, snippet and folder — and
        that join is also what filters out ids that are not pages, so the store never needs to know
        the difference between a page and a pinned mental model."""
        raise NotImplementedError("this store does not index knowledge pages")

    async def search_knowledge_pages_semantic(
        self,
        bank_id: str,
        *,
        embedding: list[float],
        limit: int,
        tags: list[str] | None = None,
        tags_match: str = "any",
        tag_groups: list | None = None,
        exclude_ids: list[str] | None = None,
    ) -> list["KnowledgePageMatch"]:
        """Pure vector search over the bank's pages, for the reflect tool.

        Separate from :meth:`search_knowledge_pages` because the two want different answers, not
        different tunings of one: this one reports ``score`` as a **similarity in [0, 1]**, which
        the agent surfaces as a relevance figure, and a fused hybrid score cannot stand in for it.

        ``exclude_ids`` drops pages from the result — a refresh must not retrieve the very page it
        is regenerating."""
        raise NotImplementedError("this store does not index knowledge pages")

    async def list_knowledge_pages(self, bank_id: str) -> list["KnowledgePageRef"]:
        """Every page the store currently indexes for this bank.

        The read half of the reconcile: diff it against the bank's `mental_models` rows to find both
        what is missing from the index and what is left over in it. Bounded by the bank's page
        count, which is a curated set rather than its corpus."""
        raise NotImplementedError("this store does not index knowledge pages")

    async def retain(
        self,
        bank_id: str,
        unit_ids: list[str],
        facts: list,
        *,
        document_id: str | None = None,
        unit_entity_names: dict[str, list[str]] | None = None,
        replace_document_id: str = "",
        replace_chunk_ids: list[str] | None = None,
        replace_keep_chunk_ids: list[str] | None = None,
        resolve_threshold: float = 0.0,
        enable_text_search: bool = True,
        enable_graph_retrieval: bool = True,
    ):
        """Commit an entire retain in one server-side call — resolve/mint the ``unit_entity_names``
        against the store's own registry, write the memories with the resulting entity ids, and
        (when ``replace_document_id`` is set) tombstone the document's prior version — all atomically.
        Only a store advertising :attr:`store_owned` implements this; the orchestrator calls it
        exactly when :meth:`store_owned_for` is true, so the default never runs. It exists on
        the interface so a routing extension delegates it automatically (see RoutingMemories).

        ``replace_chunk_ids`` narrows the replace to named chunks of the document — the DELTA case,
        where every chunk not named keeps its facts. Pass the chunks whose facts must go: the ones
        that changed AND the ones that were removed, since a removed chunk has no replacement upsert
        to supersede it. Without this a re-ingest can only replace wholesale, which means
        re-extracting the entire document to change one paragraph.

        ``replace_keep_chunk_ids`` states the same scope from the other side — the survivors — and
        exists because a store may cap how many values a scope can name. A re-ingest that rewrites
        most of a large document cannot name the changed chunks under such a cap, but naming the few
        that survive is the same replace. Pass whichever side is smaller; passing both is an error.

        A store that cannot scope a replace must ignore these rather than silently widening to a
        wholesale one: the chunks the caller did not name are exactly the ones it is trying to
        keep.

        ``enable_text_search`` / ``enable_graph_retrieval`` are the bank's recall toggles, passed
        here because for a store that owns its index they are not only read-time settings: an arm
        the bank has switched off needs no index built for it, and building one is work and bytes
        spent on a query that will not run. They are the bank's CURRENT values on every retain, so
        a store that acts on them tracks a bank that changes its mind without an out-of-band call.

        A store that indexes everything regardless ignores them, which is what the default does —
        and what Postgres does, where the columns behind both arms are maintained by the insert
        itself and there is nothing separable to skip."""
        raise NotImplementedError("this store does not support a store-owned retain")

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
    async def delete_facts(self, bank_id: str, unit_ids: list[str]) -> None:
        """Remove units. Safe to call for ids that were never written."""

    async def delete_where(self, bank_id: str, predicate: DeletePredicate) -> int:
        """Remove every memory matching ``predicate``. Returns the count when known.

        May be implemented lazily (recording the delete and materializing it
        later), in which case the returned count is 0 rather than a scan.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_document(self, *, conn, fq_table, bank_id: str, document_id: str) -> None:
        """Remove every memory belonging to ``document_id``.

        Called when a document is replaced, so it races the replacement's writes:
        an implementation must remove only what was written *before* this call,
        never the facts arriving moments later.
        """

    # ------------------------------------------------------ document/chunk bodies
    #
    # Only relevant when :attr:`store_owned` is True: a store that keeps document/chunk
    # BODIES (extracted text, chunk texts, original file) in its own dedicated store rather than in
    # ``documents.original_text`` / ``chunks.chunk_text`` / ``file_storage``. The retain and read
    # paths branch on ``store_owned`` and call these instead of the inline SQL. All bodies
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
        expect_watermark: "int | None" = None,
    ) -> None:
        """Store (or replace) a document's bodies: its extracted text, its ordered chunk texts, and
        optionally the original uploaded file. Idempotent by content — re-ingest re-uploads only
        what changed.

        ``chunk_texts`` REPLACES the document's chunk list, so a caller holding only part of a
        document (a retain sub-batch) must send the whole list, not its slice — see
        ``_store_document_bodies``, which restores the prefix before calling this.

        ``expect_watermark`` makes this a compare-and-set: the write is applied only if the store's
        state is still the one the caller read (the ``watermark`` from
        :meth:`get_document_record`), and otherwise raises :class:`StoreWriteConflict` having
        written nothing. This is what makes a read-modify-write — appending onto the stored body —
        safe against a concurrent one, which without it silently erases the other's turn. ``None``
        writes unconditionally. A store with no notion of a watermark ignores it, and is expected
        to serialize such writes some other way."""
        raise NotImplementedError

    async def document_content_hash(self, *, bank_id: str, document_id: str) -> "str | None":
        """The stored document's content hash, for the idempotent-skip check; ``None`` if absent."""
        raise NotImplementedError

    async def get_document_record(self, *, bank_id: str, document_id: str, include_text: bool = False) -> "dict | None":
        """A document's metadata (and, if asked, its extracted ``original_text``), or ``None``.

        A returned record may carry a ``watermark``: an opaque token for the store state this read
        observed, to hand back as ``put_document(expect_watermark=...)`` when the write is derived
        from what was just read. Absent for a store that does not support conditional writes."""
        raise NotImplementedError

    async def list_documents(
        self,
        *,
        bank_id: str,
        search_query: "str | None" = None,
        tags: "list[str] | None" = None,
        tags_match: str = "any_strict",
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        """Page this bank's documents from the store's OWN registry — the ``{items, total, limit,
        offset}`` shape the documents browser expects. Only a store that owns its document metadata
        overrides this (a Postgres-backed store lists from the SQL ``documents`` table instead, so
        the engine only calls this for a ``store_owned`` store). Default raises so a
        mis-routed call is loud rather than silently empty.

        ``tags``/``tags_match`` filter by the documents' tags with the same modes and meanings as
        anywhere else, and ``total`` must count what MATCHES — a page filtered after the fact would
        report the unfiltered total and drop every match past the window."""
        raise NotImplementedError

    async def count_documents(self, *, bank_id: str) -> int:
        """This bank's document count, from the store's own registry — the bank-stats document
        total. Only a ``store_owned`` store overrides this (a Postgres store counts the
        SQL ``documents`` table instead); the engine only calls it for a store that owns its docs."""
        raise NotImplementedError

    async def get_entity_graph(self, *, bank_id: str, limit: int = 1000, min_count: int = 1) -> dict:
        """The entity co-occurrence graph (``{nodes, edges, ...}``) from the store's OWN aggregate.
        Only a store that owns its entities overrides this (a Postgres store reads its
        ``entity_cooccurrences`` table); the engine calls it only for a store-owned bank, whose SQL
        table is empty."""
        raise NotImplementedError

    async def get_chunk_text(self, *, bank_id: str, document_id: str, chunk_index: int) -> "str | None":
        """One chunk's text by position, or ``None`` if the document/index does not exist."""
        raise NotImplementedError

    async def hydrate_results(self, *, bank_id: str, results: "list") -> None:
        """Fill in the payload for retrieval results a store returned without one, IN PLACE.

        Default: nothing to do. A store that returns fully-populated results from retrieval — the
        Postgres one does — is already hydrated, and this costs it a single ``return``.

        It exists because ranking does not need payloads. Fusion orders candidates by id and arm
        score, and only the few that survive are ever read, so a store CAN return scores for the
        wide arms and materialize the rest afterwards. A store that does so must populate at least
        ``text`` here, and should also restore ``entity_ids`` and, for observations,
        ``source_memory_ids`` — each field left ``None`` sends recall down a fallback that re-fetches
        the very memories this just fetched (``entity_map_for_units`` for the first, the
        ``prefer_observations`` and ``include_chunks`` reads for the second).

        Declared here rather than probed for, because a routing store generates its delegators from
        this interface: a method that exists only on a concrete store is unreachable in a cloud
        deployment, and the call silently does nothing. That has cost three optimisations already.
        """
        return None

    async def count_memories_many(self, *, bank_ids: "list[str]", strong: bool = False) -> "dict[str, dict[str, int]]":
        """Per-bank fact counts for MANY banks — ``{bank_id: {fact_type: count}}``.

        A bank absent from the result has nothing to count, so one unknown bank cannot fail a page.

        Declared here for the same reason as :meth:`get_chunk_texts`: a bank list wants a count for
        every bank on the page, and the per-bank shape makes that a round-trip per bank. A store
        that can answer them together overrides this; the default is the per-bank loop, which is
        correct everywhere and merely saves nothing.

        ``strong`` asks for read-your-writes. A store whose counts lag (because they come from a
        periodically-refreshed index rather than a live read) may answer the default form from that
        lagging view; the loop below ignores the flag because a per-bank count is already live.
        """
        out: dict[str, dict[str, int]] = {}
        for bank_id in bank_ids:
            counts = await self.count_memories(conn=None, fq_table=None, bank_id=bank_id)
            if counts:
                out[bank_id] = counts
        return out

    async def last_write_at_many(self, *, bank_ids: "list[str]") -> "dict[str, datetime]":
        """When each bank was last written — ``{bank_id: datetime}``, for many banks at once.

        The bank list is ORDERED by this. Postgres derives it with ``MAX`` over live
        ``documents`` / ``memory_units`` rows, which a store owning those rows leaves empty, so
        without an answer here such a bank's ordering silently degenerates to created-at.

        Empty by default: the SQL path already has the columns and does not need this, and a store
        that cannot answer should say nothing rather than guess. **A bank absent from the result
        keeps whatever the caller already had — absent means "unknown", never "the epoch".**
        """
        return {}

    async def last_document_at_many(self, *, bank_ids: "list[str]") -> "dict[str, datetime]":
        """When each bank last INGESTED a new document — ``{bank_id: datetime}``.

        Not the same question as :meth:`last_write_at_many`: re-retaining an existing document is a
        write but not a new document, and the bank list shows the two separately. Empty by default,
        with the same rule — a bank absent from the result keeps whatever the caller had.
        """
        return {}

    async def get_chunk_texts(self, *, bank_id: str, refs: "list[tuple[str, int]]") -> "list[str | None]":
        """Many chunks' text at once — ``refs`` is ``(document_id, chunk_index)``.

        Returns one entry per ref, in the SAME order, ``None`` where the chunk does not exist.

        Declared here, not left to the store, because a chunk-hydrated recall wants one chunk from
        each of many documents and the per-chunk shape makes that a round-trip per document. A store
        that can fetch them together overrides this; the default below is correct for every store and
        simply does not save anything.

        **It has to be on this interface to be reachable.** A routing store generates its delegators
        from the methods declared here, so a fetch-many that exists only on a concrete store is
        invisible through the router — the call silently falls back and the optimisation is dead code
        in exactly the deployment it was written for. That has now happened twice; see
        ``store_owned_for``.
        """
        return [await self.get_chunk_text(bank_id=bank_id, document_id=doc_id, chunk_index=idx) for doc_id, idx in refs]

    async def list_chunk_texts(self, *, bank_id: str, document_id: str) -> "list[str] | None":
        """Every chunk's text in order, or ``None`` if the document does not exist."""
        raise NotImplementedError

    async def count_chunks(self, *, bank_id: str, document_id: str) -> int:
        """How many chunks a document has (0 if it does not exist)."""
        raise NotImplementedError

    async def set_document_tags(self, *, bank_id: str, document_id: str, tags: "list[str]") -> None:
        """Replace a document RECORD's tags, leaving its bodies alone.

        Only a ``store_owned`` store implements this; a Postgres store updates its own
        ``documents`` row instead, so the engine calls it only for a store-owned bank. It exists
        because re-tagging must not mean re-uploading: the record already carries every body's
        content hash, so a store can rewrite the record with new tags and move no bytes.

        Without it, `update_document(tags=...)` changed the memories' tags and left the document
        itself showing the old ones, which is the sort of half-applied edit that only surfaces in
        the browser a week later."""
        raise NotImplementedError

    async def delete_document_record(self, *, bank_id: str, document_id: str) -> None:
        """Delete a document's RECORD and bodies from the document store — an EXPLICIT document
        deletion, distinct from :meth:`delete_document` (which drops only the document's facts on
        re-ingest and must not touch the record, since the replacement's ``put_document`` overwrites
        it). No-op for a store that does not own the document store."""
        raise NotImplementedError

    async def drop_bank_storage(self, bank_id: str) -> None:
        """Drop a bank's entire storage. Irreversible.

        A no-op for Postgres, where deleting the bank cascades to its rows.
        """

    async def delete_observations(self, *, conn, fq_table, bank_id: str) -> None:
        """Remove every observation in a bank, leaving the facts behind it."""
        raise NotImplementedError

    async def update_memories(self, bank_id: str, patches: list[MemoryPatch]) -> None:
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
        enable_text_search: bool = True,
        enable_graph: bool = True,
    ) -> "dict[str, RecallArms]":
        """Run ALL retrieval arms for every fact_type — the whole recall interface, in one call.

        Returns ``{fact_type: RecallArms(semantic, bm25, graph, temporal)}`` of
        ``RetrievalResult``: the four per-arm candidate lists, unfused (RRF/rerank happen
        downstream, unchanged). ``temporal`` is empty unless ``temporal_window`` is given;
        ``bm25`` is empty when ``enable_text_search`` is False, and ``graph`` when
        ``enable_graph`` is False.

        This is the ONE method recall goes through — how a store answers the arms is entirely its
        own business. Postgres runs the split per-arm SQL orchestration behind this (a dense+BM25
        UNION query, a graph retriever per type, a temporal query); a store that owns its index
        answers every arm from a single query with no per-arm round-trips. Either way the caller
        sees only this method and its per-arm result.

        ``conn`` is the store's connection handle for the call. Postgres treats it as the pool it
        acquires its own connections from and runs the graph arm on; a store that reaches its index
        another way (e.g. over the network) ignores it.
        """

    async def full_recall(self, request: "FullRecallRequest") -> "RecallResult | None":
        """Answer a WHOLE recall inside the store, or decline by returning ``None``.

        The default is ``None``: a store that does not implement this simply never claims a
        recall, and the engine runs its own pipeline exactly as before. That is what makes this
        safe to add — it is an opt-in shortcut, not a fork in the interface.

        A store that DOES claim one is taking on everything the engine does after
        :meth:`recall_unified`: fusion, the candidate trim, reranking, the recency / temporal /
        strategy boosts, the ``min_scores`` floors, the ``max_tokens`` budget, and the entity and
        chunk enrichment. It must return the same shape the engine would have — the results in
        rank order, with their scores — because the caller cannot tell which path produced its
        answer and must not need to.

        **Declining is normal and must stay cheap.** A store should return ``None`` for any
        request shape it does not implement rather than approximating it: a recall answered
        almost-right is far worse than one answered by the path that has always answered it. The
        engine falls through with no extra round-trip, since nothing has been read yet.

        The reason this exists is round-trips. On a store that lives across the network, the
        engine's pipeline is four calls — the arms, then hydration, then chunks, then entities —
        and it moves every candidate's text out of the store to decide which handful to keep.
        A store that owns its index can do all of it where the data already is.
        """
        return None

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

    async def any_memory_updated_since_batch(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        scopes: list[MemoryScopeWatermark],
    ) -> dict[str, bool]:
        """:meth:`any_memory_updated_since` for many scopes at once, keyed by ``scope.key``.

        The knowledge tree and the mental-model list both need the answer for
        every model in a bank on one read, and asking one at a time makes the
        round-trips, not the scans, the cost. A store that can answer them
        together should override this; the default is the honest loop, so a
        store only has to implement the single-scope method to work correctly.

        Duplicate keys are not meaningful — the caller owns the keyspace, and a
        repeat simply overwrites. An empty ``scopes`` list returns ``{}`` without
        touching the connection.
        """
        return {
            scope.key: await self.any_memory_updated_since(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                since=scope.since,
                fact_types=scope.fact_types,
                tags=scope.tags,
                tags_match=scope.tags_match,
                tag_groups=scope.tag_groups,
            )
            for scope in scopes
        }

    async def live_memory_ids(self, *, conn, fq_table, bank_id: str, unit_ids: list[Any]) -> set[str]:
        """Which of ``unit_ids`` still exist among the bank's live memories.

        Backs the retraction check behind the mental-model refresh: a document's
        grounding is a set of ids on ``reflect_response.based_on``, and one that no
        longer answers here has been retracted. Invalidated, deleted, and swept-as-
        stale are deliberately not distinguished — from the document's point of view
        all three mean the same thing, so this asks only whether the row is live.

        Ids that are not memory ids (or do not parse) read as absent rather than
        raising, so callers may pass a mixed set.
        """
        raise NotImplementedError

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
    async def invalidate_memory(self, *, conn, fq_table, bank_id: str, unit_id: str, reason: str | None) -> bool:
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
    async def restore_memory(self, *, conn, fq_table, bank_id: str, unit_id: str) -> StoredMemory | None:
        """Move an archived memory back to the live set, restoring its entity postings.

        Returns the restored memory (so the caller can recompute its embedding —
        the archive need not keep one), or ``None`` if it was not archived.

        Bringing a memory back is an edit, so this stamps ``updated_at`` even though
        it also resets the consolidation markers (see :data:`META_UPDATED_AT`).
        """

    @abstractmethod
    async def set_memory_embedding(self, *, conn, fq_table, bank_id: str, unit_id: str, embedding) -> None:
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
        entity_names: list[str] | None = None,
    ) -> None:
        """Apply a curation field edit to a live memory.

        Writes the new text / context / fact_type / occurred window, resets the
        consolidation markers (the memory re-consolidates) and stamps the edit
        time, and drops the memory's derived links (they are recomputed). The
        embedding is *not* written here — the caller re-embeds from the new fields
        and calls :meth:`set_memory_embedding` after.

        The new entity set for the memory is supplied one of two ways, and a store
        uses whichever fits how it keeps its registry:

        * ``entity_names`` — the raw names the edit resolved to. A store that owns
          its entity registry resolves + mints these against its OWN registry
          (exactly as its :meth:`retain` does) and rewrites the memory's entity
          ids from the result, so a brand-new entity created by an edit lands in
          that registry. When it is not ``None`` it is the authoritative set and
          ``entity_ids`` is ignored.
        * ``entity_ids`` — the already-resolved set, for a store whose registry is
          the host's SQL (the host minted them and, for a join-table store, has
          already re-linked them, so it ignores this).

        Both ``None`` means the entity set was not part of this edit.
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

    async def upsert_observation(self, *, conn, bank_id: str, record: FactRecord) -> None:
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
    ) -> None:
        """Record the unit→entity postings for a batch of memories.

        ``unit_ids`` and ``entity_ids`` are parallel: a unit that mentions three
        entities appears three times. The `entities` registry itself stays in
        Postgres regardless; this is the join from a memory to the entities it
        mentions. ``bank_id`` is passed because a store that keeps the posting on
        the memory (rather than in a global join table) needs to know which
        namespace the units live in — the Postgres join is keyed by global unit id
        and ignores it.

        For a store that keeps the posting ON the memory this call re-writes rows
        an earlier :meth:`insert_facts` created, so it must be idempotent: it is
        reached again whenever a retain re-resolves the same batch's entities.
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
    "RelinkPassResult",
    "ScanPage",
    "StoredMemory",
    "build_fact_records",
    "build_text_signals",
    "source_key",
]
