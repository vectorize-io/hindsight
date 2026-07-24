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

Every operation the engine needs is a method here, so no call site branches on
which implementation is installed. Where the two differ, they differ by what the
method does: the Postgres implementation writes join rows and reprocesses links;
one that owns the store no-ops those passes and does its own thing.
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

# Hindsight's fact_type strings <-> a numeric memory type. An implementation that
# indexes per type (results are never fused across types) lines up exactly with
# how recall queries one arm set per fact_type.
FACT_TYPE_TO_MEMORY_TYPE: dict[str, int] = {
    "world": 1,
    "experience": 2,
    "observation": 3,
}
MEMORY_TYPE_TO_FACT_TYPE: dict[int, str] = {v: k for k, v in FACT_TYPE_TO_MEMORY_TYPE.items()}

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
        return not self.metadata_equals and not self.tags


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


class MemoriesExtension(Extension, ABC):
    """Storage + retrieval for memory units and their links, behind one interface.

    Loaded with the ``MEMORIES`` prefix; see the module docstring. Subclasses get
    ``self.config`` (the ``HINDSIGHT_API_MEMORIES_*`` environment) and
    ``self.context`` from :class:`~hindsight_api.extensions.base.Extension`.

    Methods are grouped by what calls them: the retain write path, the recall
    arms, addressed reads for curation/export, and the maintenance passes. The
    Postgres implementation is the reference for what each one must mean.
    """

    #: Name for logs and the startup banner.
    name: str = "postgres"

    #: Whether memories live as rows in the SQL ``memory_units`` table. True for the SQL stores
    #: (Postgres/Oracle), whose ``upsert_observation`` / ``delete_facts`` are no-ops because the
    #: consolidator writes those rows inline. A store that keeps memories elsewhere sets this
    #: False so the consolidator skips the inline SQL and routes the write through the store —
    #: then all of an observation's state lives wherever the store keeps it, not in Postgres.
    writes_memory_rows_in_sql: bool = True

    # ------------------------------------------------------------------ lifecycle

    async def initialize(self) -> None:
        """Open connections/channels. Called once during engine startup.

        Separate from :meth:`Extension.on_startup` because the memories store has
        to be live before the engine finishes booting, not alongside the HTTP app.
        """

    async def shutdown(self) -> None:
        """Release resources. Called during engine shutdown."""

    async def ensure_namespace(self, bank_id: str) -> None:
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

    async def delete_namespace(self, bank_id: str) -> None:
        """Drop a bank's entire storage. Irreversible.

        A no-op for Postgres, where deleting the bank cascades to its rows.
        """

    async def delete_observations(self, *, conn, fq_table, bank_id: str) -> None:
        """Remove every observation in a bank, leaving the facts behind it."""
        raise NotImplementedError

    async def update_memories(self, bank_id: str, patches: list[MemoryPatch]) -> None:
        """Apply partial updates. Only the fields set on each patch change."""
        raise NotImplementedError

    # ------------------------------------------------------------------ recall arms

    @abstractmethod
    async def search(
        self,
        *,
        conn,
        bank_id: str,
        fact_types: list[str],
        query_embedding: str,
        query_text: str,
        limit: int,
        tags: list[str] | None = None,
        tags_match: str = "any",
        tag_groups: list | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        min_semantic: float | None = None,
        min_keyword: float | None = None,
    ) -> dict[str, tuple[list, list]]:
        """Run the semantic + BM25 arms.

        Returns ``{fact_type: (semantic_results, bm25_results)}`` of
        ``RetrievalResult`` — the contract ``retrieve_semantic_bm25_combined`` has
        always had, so callers and the downstream fusion are unchanged.
        """

    @abstractmethod
    async def temporal_search(
        self,
        *,
        conn,
        bank_id: str,
        fact_types: list[str],
        query_embedding: str,
        start_date: datetime,
        end_date: datetime,
        limit: int,
        semantic_threshold: float = 0.1,
        tags: list[str] | None = None,
        tags_match: str = "any",
        tag_groups: list | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> dict[str, list]:
        """Run the temporal arm over ``[start_date, end_date]``.

        Returns ``{fact_type: [RetrievalResult]}``: entry points whose effective
        time — ``COALESCE(occurred_start, mentioned_at, occurred_end)`` — falls in
        the window, spread one hop and scored by proximity to it.
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
    async def list_tags(self, *, conn, fq_table, bank_id: str) -> dict[str, int]:
        """Distinct tags in a bank and how many live memories carry each."""

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
        """``{"last_consolidated_at", "pending", "failed"}`` for a bank.

        ``pending`` / ``failed`` count the world/experience facts not yet folded
        into an observation, and those the LLM gave up on. Backs
        ``get_bank_freshness``, which reflect() calls often, so keep it cheap.
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
        fact_type: str | None = None,
        search_query: str | None = None,
        consolidation_state: str | None = None,
        state: str | None = None,
        document_id: str | None = None,
        tags: list[str] | None = None,
        tags_match: str = "any",
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
        """

    @abstractmethod
    async def set_memory_embedding(self, *, conn, fq_table, bank_id: str, unit_id: str, embedding) -> None:
        """Write a memory's embedding, recomputed by the caller.

        Its own method because the general :meth:`update_memories` is a no-op for
        the store whose write is the row itself — reverting or editing a memory has
        to put a freshly computed vector back on it, so this is a real write for
        both. ``embedding`` is a float list or the pgvector literal.
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
        """Entities in a bank with their live memory counts, paged."""

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
        self, *, conn, ops, fq_table, bank_id: str | None = None, unit_ids: list[Any], entity_ids: list[Any]
    ) -> None:
        """Record the unit→entity postings for a batch of memories.

        ``unit_ids`` and ``entity_ids`` are parallel: a unit that mentions three
        entities appears three times. The `entities` registry itself stays in
        Postgres regardless; this is the join from a memory to the entities it
        mentions. ``bank_id`` is passed because a store that keeps the posting on
        the memory (rather than in a global join table) needs to know which
        namespace the units live in — the Postgres join is keyed by global unit id
        and ignores it.
        """

    async def enqueue_relink_victims(self, *, conn, fq_table, bank_id: str, deleted_unit_ids: list) -> int:
        """Queue memories that lost a link when ``deleted_unit_ids`` went away.

        Zero for a store with no link table to dangle: nothing can point at a
        deleted memory if the pointers travel inside the memories themselves.
        """
        return 0

    async def relink_pass(self, *, backend, fq_table, bank_id: str, config) -> dict:
        """Top up links for queued victims. ``{}`` when there is nothing to relink."""
        return {}

    async def prune_orphan_entities(self, *, conn, fq_table, bank_id: str) -> int:
        """Delete `entities` rows no live memory references. Returns the count."""
        return 0

    async def prune_stale_cooccurrences(self, *, conn, fq_table, bank_id: str) -> int:
        """Delete co-occurrence rows whose witnessing memories are all gone."""
        return 0


__all__ = [
    "CONSOLIDATED_NO",
    "CONSOLIDATED_YES",
    "FACT_TYPE_TO_MEMORY_TYPE",
    "MEMORY_TYPE_TO_FACT_TYPE",
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
    "FactRecord",
    "MemoriesExtension",
    "MemoryPatch",
    "ScanPage",
    "StoredMemory",
    "build_fact_records",
    "build_text_signals",
    "source_key",
]
