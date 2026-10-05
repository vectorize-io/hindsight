"""Extension interface for the *memories* slice of storage.

The memories store owns every table a memory touches: `memory_units`, `memory_links`,
`unit_entities`, `documents`, `chunks`, `entities`, `entity_cooccurrences` and
`invalidated_memory_units`. Every recall arm — semantic, BM25, graph, temporal — is a
query over them, so this module carves them out from behind the raw SQL and lets a
different engine own them, without touching how banks, operations, mental models or
the rest of the schema are stored.

The default :class:`~hindsight_api.engine.memories.postgres.PostgresMemories` keeps
everything exactly where it has always been: rows in those tables, retrieval as SQL. It
is what runs unless an extension is configured, and it is the implementation the test
suite exercises. It is also the only code allowed to name those tables: it resolves
them through :func:`~hindsight_api.engine.schema.fq_store_table`, while ``fq_table``
refuses them, and ``tests/test_store_table_boundary.py`` fails on SQL text naming them,
or on an import of the Postgres store's modules, anywhere else.

An alternative implementation is loaded like any other Hindsight extension::

    HINDSIGHT_API_MEMORIES_EXTENSION=mypackage.memories:MyMemories
    HINDSIGHT_API_MEMORIES_SOME_SETTING=value

Such an implementation is the **sole store** for a bank it owns: Postgres holds none of
those tables' rows for it. Unit ids are minted by :meth:`MemoriesExtension.allocate_unit_ids`
rather than by an INSERT's RETURNING clause, facts carry their entity ids and causal edges
inline instead of becoming join rows, and recall results come back fully populated with no
Postgres hydration. Banks, operations and the other engine tables stay in Postgres either way.

Every operation on those tables is a method here, and each default is the answer for a store
that owns its bank; Postgres overrides it with the SQL. The remaining ``store_owned_for``
branches in the engine do not pick SQL — they pick a different flow where the shapes are
genuinely different (the retain session that hands a whole document to the store, the transfer
import that writes one), so they are the seams, not leaks.
"""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol, cast

from ...extensions.base import Extension

# The five tag-matching modes, as the HTTP layer already validates them. Declared here
# rather than `str` so a store implementing this seam is checked against the modes that
# actually exist -- the SQL builders below take this exact type, and a bare `str` made
# every hop between them unverifiable.
from ..search.tags import TagGroup, TagsMatch, tag_filter_active
from ..search.types import RetrievalResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..consolidation.consolidator import _TemporalBounds
    from ..embeddings import Embeddings
    from ..response_models import MemoryFact
    from ..retain.types import CausalRelation, EmbeddingLike, EntityResolutionResult, ProcessedFact
    from ..search.retrieval import GraphRetriever
    from ..transfer.export import _LoadedExport, _UnitLocation
    from ..transfer.importer import FactLifecycle, _ObservationOutcome
    from ..transfer.schema import TransferObservation

logger = logging.getLogger(__name__)


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
#: The attachments a fact was drawn from, as a JSON list of short ids — the per-fact
#: provenance the extractor records. Carried on the memory so read surfaces can resolve
#: it from the rows the store already returned, without a second lookup.
META_ATTACHMENT_IDS = "attachment_ids"

# Keys in a store-owned DOCUMENT record's metadata map (string -> string, so structured values
# travel as one JSON string each). Unlike the memory bag above these describe the document, and
# the same record read serves every one of them.
#: The document's replayable retain parameters, as one JSON object.
DOC_META_RETAIN_PARAMS = "retain_params"
#: The names the caller gave this document's attachments, as a JSON object of short id ->
#: filename. On the document rather than the fact because a filename describes the reference,
#: not the bytes: the same image can be "diagram.png" in one document and "fig-2.png" in another.
#: The authority is `attachments.filename`, written at the ingress on every backend; this is the
#: store's own copy, which the paths that replay stored text restate their names from.
DOC_META_ATTACHMENT_FILENAMES = "attachment_filenames"
#: Where the document's original upload lives in Hindsight's ``file_storage`` — the key
#: ``documents.file_storage_key`` holds for a bank whose documents live in SQL. The upload's name
#: and content type ride on the record's own ``file_original_name`` / ``file_content_type``.
DOC_META_FILE_STORAGE_KEY = "file_storage_key"
CONSOLIDATED_NO = "0"
CONSOLIDATED_YES = "1"


def document_record_metadata(
    retain_params: "dict | None", attachment_filenames: "Mapping[str, str] | None" = None
) -> dict[str, str]:
    """The metadata map a store-owned document record is written with.

    One builder for every write path, because a path that forgets a key does not fail -- it
    writes a record without it, and the key reads back as absent until the next full re-ingest.
    The map REPLACES the record's previous one: a write carries every name the document should
    keep, which is why the paths that re-send stored text (append, reprocess, an edit re-sending
    placeholders) carry the names they read back with it.
    """
    out: dict[str, str] = {}
    if retain_params:
        out[DOC_META_RETAIN_PARAMS] = json.dumps(retain_params)
    names = {str(k): str(v) for k, v in (attachment_filenames or {}).items() if k and v}
    if names:
        out[DOC_META_ATTACHMENT_FILENAMES] = json.dumps(names, sort_keys=True)
    return out


def document_attachment_filenames(record: "Mapping | None") -> dict[str, str]:
    """A store-owned document record's attachment names (short id -> filename); ``{}`` if none.

    Tolerant by design: a record written before the key existed, or one whose value does not
    parse, has no names -- which reads back as a null ``filename``, exactly what it read before.
    """
    raw = ((record or {}).get("metadata") or {}).get(DOC_META_ATTACHMENT_FILENAMES)
    if not raw:
        return {}
    try:
        decoded = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return {}
    if not isinstance(decoded, dict):
        return {}
    return {str(k): str(v) for k, v in decoded.items() if k and v}


# ------------------------------------------------------------------ document and chunk types


@dataclass(frozen=True)
class AttachmentRef:
    """Attachment short ids named by one document: the filename lives on the document edge."""

    document_id: str | None
    attachment_ids: list[str]


@dataclass
class ObservationChunkIds:
    """The chunk ids of each observation's sources, for recall ``include_chunks``.

    ``chunk_ids_by_observation`` is in observation-rank order, each list in source order
    and not yet deduplicated across observations. ``sources_by_observation`` is set only
    when answering had to read the observations for their sources, so the caller can put
    them back on its results rather than read them a second time.
    """

    chunk_ids_by_observation: dict[str, list[str]]
    sources_by_observation: dict[str, list[str]] | None = None


@dataclass
class DocumentSourceUnits:
    """A document's experience/world memory ids and its total memory count, read before a delete."""

    unit_ids: list[str]
    units_count: int


@dataclass
class DeletedDocument:
    """What deleting a document found: whether it existed, and its uploaded file's storage key."""

    deleted: bool
    file_storage_key: str | None


@dataclass
class DocumentTags:
    """A document's current tags. ``tags`` is ``None`` when they could not be read — the
    document is absent, or its record does not carry them — which is never "no tags"."""

    found: bool
    tags: list[str] | None


def _epoch_ms_to_datetime(value: Any) -> datetime | None:
    """Epoch milliseconds -> aware datetime, for records read from a store rather than from SQL.

    A store returns timestamps as integers; the SQL rows these records stand in for come back as
    datetimes, and the response builders call ``.isoformat()`` on them. Normalising here keeps
    those builders unaware of where the record came from. ``0`` means unset, not the epoch.
    """
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromtimestamp(int(value) / 1000.0, tz=timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


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
    # When a curation edit last changed the memory; ``None`` for one still exactly as extracted.
    # The curation views report it, so a store has to hand it back for them to say so.
    edited_at: datetime | None = None
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
    # Short ids of the attachments this fact was drawn from (see META_ATTACHMENT_IDS).
    # Carried so the list and detail views resolve them from this read alone.
    attachment_ids: list[str] = field(default_factory=list)


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
    tags_match: TagsMatch = "any"
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
class BankWriteTime:
    """When one bank was last written, as the store records it."""

    bank_id: str
    last_write_at: datetime


@dataclass
class BankWritePage:
    """One page of :meth:`MemoriesExtension.list_banks_by_write`, newest-written first.

    Same shape as :class:`ScanPage`: ``next_page_token`` is a position and is empty exactly when
    the walk is exhausted.
    """

    banks: list[BankWriteTime] = field(default_factory=list)
    next_page_token: str = ""


@dataclass
class RetainDocumentPart:
    """One consumer batch's worth of a document: its chunk texts and the facts extracted from them.

    The unit the engine already produces. It is deliberately NOT "a whole document": a document's
    chunks arrive across sub-batches and across extraction completions, and requiring completeness
    here would either serialise extraction behind it or force the engine to decide when a document
    is done — a judgement the streaming pipeline is specifically built not to need.

    A part may carry chunk texts, facts, or both, and the engine sends BOTH KINDS SEPARATELY for
    the same document. That is not a convenience: the streaming producer frees each chunk string as
    soon as it has been extracted (`all_pre_chunks[i] = ""`), so the texts are only live at the
    point they are produced, while the facts do not exist until extraction completes. A contract
    that demanded them together would either pin the whole document in memory for the retain or
    read back blanked strings. The store merges them per document.

    `chunk_texts` are the texts for `chunk_offset .. chunk_offset + len(chunk_texts)`. The OFFSET is
    per document, not per call: it is what makes a document's chunk identity independent of which
    other documents shared the retain, and a store that derives chunk ids from anything else will
    give the same document different ids depending on its neighbours — silently breaking dedup and
    delta. See `document_body` for the rest of that contract.
    """

    document_id: str
    #: The whole document's text, for the stored body. Every part of one document carries the same
    #: value, so whichever part is written first can supply it.
    document_body: str | None
    content_hash: str
    chunk_offset: int
    #: Empty on a facts-only part. A part never carries a PARTIAL chunk list for its offset.
    chunk_texts: list[str] = field(default_factory=list)
    facts: list["FactRecord"] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    #: Entity NAMES per unit_id, unresolved. A store that owns an entity registry resolves them
    #: itself; one that does not resolves them before calling.
    entity_names: dict[str, list[str]] = field(default_factory=dict)
    #: The subset of ``entity_names`` per unit_id the caller opted out of resolution
    #: (``resolve_entities=False``). A store that resolves names itself must match these on the
    #: name alone (case-insensitive) and mint a new entity otherwise — never fuzzy-merge them onto
    #: a similar name, which is how "Alice Smyth" ended up as "Alice Smith" (#5050).
    exact_entity_names: dict[str, list[str]] = field(default_factory=dict)
    #: What this part replaces, if anything: `None` replaces nothing, an empty list replaces the
    #: WHOLE document, and a non-empty list names the chunk ids whose facts go. Only the first part
    #: of a document may carry it — a later one would tombstone its own siblings.
    replace_chunk_ids: list[str] | None = None


@dataclass
class RetainResult:
    """What a committed retain produced, per document."""

    #: unit_id lists keyed by document_id, in the order the facts were added.
    unit_ids: dict[str, list[str]] = field(default_factory=dict)
    #: Entities minted (not resolved to existing ones) across the whole retain.
    new_entities: int = 0


class RetainSession:
    """An in-flight retain. The engine streams parts in; the store decides when to write.

    A session rather than one call because the engine's pipeline overlaps extraction with writes —
    LLM extraction is the dominant cost when it runs, and a contract of "hand me everything, then I
    persist" would serialise it away. `add` is called exactly where the engine writes today, so the
    overlap is unchanged; what moves is WHEN the write happens, which becomes the store's choice.

    That choice is the point. A bulk ingest that runs no LLM should commit ONCE — one WAL entry,
    one bump of the namespace head, which is what a sustained ingest actually serialises on. A long
    LLM extraction should not, because committing only at the end means a crash loses all of it.
    Neither is right in general, so the engine must not decide it.

    Two rules a flush policy has to honour, whatever it decides:

    * **Never memories without their bodies.** A flush must not commit facts whose document record
      has not landed, or an interrupted retain leaves memories citing chunks that do not exist.
    * **Bound what an interruption loses.** "Only at the end" is unbounded for a long retain; some
      progress has to become durable as it accumulates.
    """

    async def add(self, part: RetainDocumentPart) -> None:
        """Take one part. May write, may buffer — the caller must not assume either."""
        raise NotImplementedError

    async def commit(self) -> RetainResult:
        """Write whatever is still buffered and return what the retain produced."""
        raise NotImplementedError

    async def abort(self) -> None:
        """Give up. Anything already flushed STAYS: the store has no cross-entry rollback, and
        pretending otherwise would be a transaction it cannot honour. Callers get at-least-what-was-
        flushed, which is the same guarantee an interrupted retain has always had."""
        return None


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
    # What would have become `unit_entities` rows: the unit→entity posting travels with the
    # memory, and the ids point into the store's own entity registry.
    entity_ids: list[str] = field(default_factory=list)
    # What would have become causal `memory_links` rows.
    causal_edges: list[CausalEdgeRecord] = field(default_factory=list)
    # Observations only: the facts this observation was consolidated from.
    source_memory_ids: list[str] = field(default_factory=list)
    # When this memory was folded into an observation (sources only).
    consolidated_at: datetime | None = None
    # Short ids of the attachments this fact was drawn from — what Postgres keeps in
    # `memory_units.attachment_ids`. Empty for a fact stated in plain text.
    attachment_ids: list[str] = field(default_factory=list)

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
        if self.attachment_ids:
            # Deduplicated in first-seen order, the same normalisation the SQL write applies.
            bag[META_ATTACHMENT_IDS] = json.dumps(list(dict.fromkeys(self.attachment_ids)))
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
                attachment_ids=list(getattr(fact, "attachment_ids", None) or []),
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
    tags_match: TagsMatch = "any"
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


#: How many semantic hits seed the graph arm's link expansion — its entry points into the graph.
GRAPH_SEED_LIMIT = 20


@dataclass
class SemanticBm25Result:
    """One fact_type's dense + keyword candidates, plus the graph arm's seeds.

    What a store's combined semantic/BM25 read hands back (the Postgres store's ``search``). Declared
    here rather than in the Postgres store's modules so a store implementing the same read does not
    have to import them.
    """

    semantic: list[RetrievalResult]
    bm25: list[RetrievalResult]
    graph_seeds: list[RetrievalResult] | None


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
    tags_match: TagsMatch = "any"
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
    # ---- derived memories (observations) and their sources ----
    #
    # An observation is consolidated FROM source facts and carries their ids, so a store that holds
    # that list can answer all three of these itself. Each was previously a decline.
    #: Drop raw facts that a returned observation was consolidated from, and backfill the freed
    #: slots. No-op unless both observations and a raw type were requested.
    prefer_observations: bool = False
    #: Return each returned observation's source facts, under the budgets below.
    include_source_facts: bool = False
    #: Total token budget for source facts. Negative means unlimited.
    max_source_facts_tokens: int = 4096
    #: Per-observation budget. When >= 0 this REPLACES the total, so one observation with many
    #: sources cannot spend every other observation's provenance.
    max_source_facts_tokens_per_observation: int = -1


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


@dataclass
class MemoryLocation:
    """Where one memory lives and what it is: the answer to the lookups that precede a delete
    or a history read. ``source_memory_ids`` is filled only by :meth:`MemoriesExtension.observation_head`."""

    unit_id: str
    bank_id: str
    fact_type: str
    source_memory_ids: list[str] = field(default_factory=list)


@dataclass
class TypedMemoryScope:
    """A bank's memories of one fact_type, as a typed bank clear needs them: the ids (source
    types only, for the stale-observation sweep) and the count it reports."""

    unit_ids: list[str] = field(default_factory=list)
    count: int = 0


@dataclass
class BankContentCounts:
    """What a whole-bank delete reports having removed."""

    memory_units: int = 0
    entities: int = 0
    documents: int = 0


# What the retain path's document reads return.


@dataclass
class DocumentBase:
    """The stored document an append builds on: its body and the version it was read at.

    ``watermark`` is the token a conditional write compare-and-sets against (see
    :meth:`MemoriesExtension.put_document`); ``None`` for a store that serializes appends
    on a row lock instead.
    """

    original_text: str | None
    content_hash: str | None
    attachment_filenames: dict[str, str] = field(default_factory=dict)
    watermark: int | None = None


@dataclass
class ExistingChunk:
    """A chunk already stored for a document: its id, position and content hash."""

    chunk_id: str
    chunk_index: int
    content_hash: str | None


@dataclass
class DocumentChunkState:
    """A stored document's version, body (when asked for) and chunks, as a delta retain diffs them.

    ``watermark`` is the store-state token the delta's write compare-and-sets against (see
    :meth:`MemoriesExtension.put_document`); ``None`` where the write serializes on a row lock."""

    content_hash: str | None
    original_text: str | None
    chunks: list[ExistingChunk] = field(default_factory=list)
    watermark: int | None = None


def document_chunk_state(
    record: "Mapping | None", *, bank_id: str, document_id: str, include_text: bool
) -> DocumentChunkState:
    """A store-owned document record read as a :class:`DocumentChunkState`; empty if ``None``.

    The record's own chunk hashes, not a download of every chunk's text: they were computed with
    the same ``sha256(chunk.encode()).hexdigest()`` retain compares with. A chunk_id is
    ``build_chunk_id(bank_id, document_id, index)`` by construction, so the record carries none.
    All of it comes from the ONE record: un-pairing the watermark from the hash is what the
    compare-and-set exists to prevent.
    """
    from ..chunk_ids import build_chunk_id

    rec = record or {}
    return DocumentChunkState(
        content_hash=rec.get("content_hash"),
        original_text=rec.get("original_text") if include_text else None,
        chunks=[
            ExistingChunk(
                chunk_id=build_chunk_id(bank_id, document_id, index),
                chunk_index=index,
                content_hash=chunk_hash,
            )
            for index, chunk_hash in enumerate(rec.get("chunk_hashes") or [])
        ],
        watermark=rec.get("watermark"),
    )


@dataclass
class RelabelResult:
    """What relabelling a document's memories did: how many it updated, and which of them
    (``experience``/``world`` only) changed tags or observation scoping — the ones whose
    observations are no longer valid."""

    updated: int
    rescoped_unit_ids: list[str] = field(default_factory=list)


class EntityResolverHandle(Protocol):
    """What the engine holds as its entity resolver: the Postgres store's SQL resolver, built once at
    startup through ``sql_memories()`` and handed to :meth:`MemoriesExtension.resolve_entities`.

    A Protocol rather than the Postgres ``EntityResolver`` class: this module may not import the
    Postgres store's modules (``tests/test_store_table_boundary.py``). The two per-task stats
    hooks are what the retain and import paths call on every batch whatever store owns the bank;
    the memory edit calls the reassert / link pair after :meth:`MemoriesExtension.resolve_entities`
    resolved something.
    """

    def discard_pending_stats(self) -> None: ...

    async def flush_pending_stats(self) -> None: ...

    async def reassert_entities_batch(self, bank_id: str, resolved_entities: list[Any], conn) -> None: ...

    async def link_units_to_entities_batch(
        self,
        unit_entity_pairs: list[tuple[str, str]] | list[tuple[str, str, datetime | None]],
        conn=None,
        bank_id: str | None = None,
    ) -> None: ...


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
    #: / ``document_content_hash``), resolves entity NAMES itself, and commits the
    #: entire retain — resolution, upserts and the document replace — as ONE atomic server-side call
    #: (``retain``). The orchestrator then needs no Postgres connection phase for it.
    #:
    #: It also selects the KNOWLEDGE-PAGE index: a store that owns its memories serves
    #: `search_knowledge_pages` and the reflect tool from its own index, and the page write paths
    #: call :meth:`index_knowledge_pages` / :meth:`delete_knowledge_pages`. The Postgres row is
    #: still written first and is still what a hit is hydrated from — the store holds a DERIVED
    #: copy, so a divergence is repaired by indexing again rather than restored from a backup.
    #:
    #: This is also what selects the retain SESSION (:meth:`begin_retain`): a store that owns its
    #: memories owns the persistence half of a retain, and the engine hands it the whole of it —
    #: how many round trips it takes, how chunk identity is derived, and what commits atomically
    #: with what. There is no separate flag for that, deliberately. There was, and a router that
    #: forwarded `store_owned_for` but not the second probe silently answered "no session" for
    #: every bank: nothing failed, retain just fell back to writing per consumer batch and ran at
    #: half the throughput. One capability, one flag, one thing for a router to forward.
    #:
    #: Bank-scoped via :meth:`store_owned_for`, for a router whose banks live in different backends.
    store_owned: bool = False

    def store_owned_for(self, bank_id: str) -> bool:
        """Per-bank form of :attr:`store_owned`. Defaults to the class attribute, so a
        single-backend extension needs no override. A router whose banks live in different backends
        overrides this to answer PER BANK; every bank-scoped call site consults it rather than the
        attribute."""
        return self.store_owned

    def backend_name_for(self, bank_id: str) -> str:
        """Which store serves this bank, as the ``memories_backend`` metric label. Empty by default.

        Empty means no label at all, so a deployment that never overrides this keeps exactly the
        series it has today: adding a label to an existing series starts a new one and orphans its
        history. A router whose banks live in different backends overrides this to name the store
        it routes a bank to -- typically only the non-default one, leaving the default store's
        series unlabelled and continuous. Per-backend latency otherwise needs the per-tenant label,
        which is too high-cardinality to leave on. Must be a short, bounded name.
        """
        return ""

    async def put_documents(self, *, bank_id: str, documents: list[dict], expect_watermark: int | None = None) -> None:
        """Store (or replace) several documents in one call.

        Default is a loop over :meth:`put_document`, so a store gains nothing by not implementing
        it and no caller has to ask whether it exists. A store whose write is a network round trip
        overrides this to send one, which is where the saving is.

        Declared HERE rather than left on the provider: a public provider method the seam does not
        declare is one the engine can never call, and that has shipped twice.
        """
        for d in documents:
            await self.put_document(bank_id=bank_id, expect_watermark=expect_watermark, **d)

    async def get_document_records(self, *, bank_id: str, document_ids: list[str]) -> dict[str, dict]:
        """Several documents' metadata in one read, keyed by document_id; absent ones omitted.

        Default is a loop over :meth:`get_document_record`, so a store gains nothing by not
        implementing it and no caller has to ask whether it exists. A store whose read is a network
        round trip overrides this to send one.
        """
        out: dict[str, dict] = {}
        for did in document_ids:
            rec = await self.get_document_record(bank_id=bank_id, document_id=did)
            if rec is not None:
                out[did] = rec
        return out

    async def begin_retain(self, *, bank_id: str, config: Any) -> "RetainSession":
        """Open a retain session. Only a store advertising :attr:`store_owned`
        implements this; the orchestrator calls it instead of driving the writes itself."""
        raise NotImplementedError

    #: True when the store derives semantic links itself, so retain must not run the SQL pass.
    #:
    #: The end-of-retain ANN pass reads every committed unit's embedding out of `memory_units` and
    #: writes links back. For a store that owns its memories those rows are not in SQL at all: the
    #: read returns nothing, the pass derives nothing, and all it costs is a connection acquire and
    #: a query per retain -- against an empty table, on the hot write path.
    #:
    #: Declared here rather than read off the provider with `getattr`, because an attribute only
    #: the provider knows about is one the engine silently never consults -- which is exactly what
    #: happened: a provider set this and nothing honoured it.
    derives_semantic_links_internally: bool = False

    def derives_semantic_links_internally_for(self, bank_id: str) -> bool:
        """Per-bank form of :attr:`derives_semantic_links_internally`, for a router whose banks
        live in different backends. Defaults to the class attribute."""
        return self.derives_semantic_links_internally

    # -- the knowledge-page index -------------------------------------------
    #
    # Knowledge pages are the one place where a store holds a DERIVED copy rather than the
    # authority. The page's row stays in Postgres; the store keeps only an index over its text and
    # embedding, so a lost or diverged entry is repaired by indexing it again and the two never need
    # a transaction between them.
    #
    # It follows `store_owned` rather than carrying a flag of its own. A separate flag would only
    # earn its place if some store wanted the mixed state — owning every memory in a bank while its
    # pages were still searched in SQL — and none does: a store that owns the bank's memories is
    # already the thing answering its searches, so splitting the two only creates a combination
    # nothing sets and every call site has to keep handling.

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
        tags_match: TagsMatch = "any",
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
        tags_match: TagsMatch = "any",
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
        unit_exact_entity_names: dict[str, list[str]] | None = None,
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

        ``unit_exact_entity_names`` is the subset of ``unit_entity_names`` the caller opted out of
        resolution (``resolve_entities=False``): match those on the name alone and mint otherwise,
        never fuzzy-merge them. Only passed when non-empty.

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
        itself and there is nothing separable to skip.

        Returns a **mapping** describing the commit: ``seq`` (the store's write coordinate for this
        retain), ``unit_ids`` (the ids actually written, echoed back) and ``new_entities`` (how many
        entities the resolve minted). Read it with ``resp["seq"]`` / ``resp.get(...)``, never as
        attributes — this is a plain mapping, not a response object, and callers that reached for
        ``resp.seq`` raised ``AttributeError`` from inside a log line and failed the whole write.
        Stated here because the return value was previously undeclared, which is what let the two
        sides disagree without either being obviously wrong."""
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
        tags_match: TagsMatch = "any_strict",
        tag_groups: "list[TagGroup] | None" = None,
        time_field: str | None = None,
        start_date: "datetime | None" = None,
        end_date: "datetime | None" = None,
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
        report the unfiltered total and drop every match past the window.

        ``time_field`` (``created_at`` / ``updated_at``) with ``start_date``/``end_date`` is the
        same window the SQL branch applies: half-open ``[start, end)`` on the named axis, which also
        becomes the ordering. ``total`` counts the window, on the same terms as tags above."""
        raise NotImplementedError

    async def count_documents(self, *, bank_id: str) -> int:
        """This bank's document count, from the store's own registry — the bank-stats document
        total. Only a ``store_owned`` store overrides this (a Postgres store counts the
        SQL ``documents`` table instead); the engine only calls it for a store that owns its docs."""
        raise NotImplementedError

    async def get_entity_graph(
        self,
        *,
        bank_id: str,
        limit: int = 1000,
        min_count: int = 1,
        tags: "list[str] | None" = None,
        tags_match: TagsMatch = "any",
        tag_groups: "list | None" = None,
    ) -> dict:
        """The entity co-occurrence graph (``{nodes, edges, ...}``) from the store's OWN aggregate.
        Only a store that owns its entities overrides this (a Postgres store reads its
        ``entity_cooccurrences`` table); the engine calls it only for a store-owned bank, whose SQL
        table is empty.

        ``tags``/``tags_match`` and ``tag_groups`` (AND-ed, fuzzy leaves already resolved)
        restrict the graph to the memories that match (same modes as anywhere else): an edge
        counts only matching memories naming both entities, and a node's ``mentionCount`` only
        matching memories naming it. A store that cannot filter must raise,
        never return the unfiltered graph — that would hand a scoped reader every other scope's
        entities (#5031)."""
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

    async def list_banks_by_write(self, *, limit: int = 100, page_token: str = "") -> "BankWritePage":
        """One page of this store's banks, most recently written first, plus the next cursor.

        The ORDER over a bank list, which is the part a store owning the memories has to supply.
        ``last_write_at_many`` answers the same question one bank at a time, and a list ordered by
        it therefore has to ask about EVERY bank before it can cut page 1 — O(total banks) per
        request, which on a large tenant is the entire cost of the endpoint. This hands back the
        page already ordered, so the caller hydrates the rows it will show rather than ranking the
        whole tenant to find them.

        It is an ordering, not a listing: names, settings and everything else about a bank stay
        wherever they live, and the caller joins them onto this page.

        **Scoped to the calling tenant.** This is the only method here that LISTS bank ids rather
        than being handed them — every other one (:meth:`last_write_at_many`,
        :meth:`count_memories_many`, …) is tenant-scoped by construction, through the ids its
        caller passes in. An id from outside the tenant has no row in that tenant's ``banks``
        table, so it consumes a slot of the page and hands the caller a short one.

        An empty ``page_token`` starts at the most recently written bank; the returned
        ``next_page_token`` is empty exactly when the walk is exhausted.

        Empty by default, and that is a meaningful answer: a store with no ordering of its own
        leaves the caller's SQL ordering in place rather than replacing it with a worse one.
        """
        return BankWritePage(banks=[], next_page_token="")

    async def run_store_migration(self, *, migration_id: str) -> "tuple[int, int]":
        """Apply a named migration of the STORE's own derived state to this tenant.

        Returns ``(started, total)`` — how many units of the store's work this call set going, out
        of how many it found. Both are the store's own units and mean nothing to the caller beyond
        progress; :meth:`store_migration_status` is what says when it is finished.

        A store that owns the memory rows also owns state derived from them — orderings, counts,
        auxiliary indexes — and a release that adds such state leaves every PRE-EXISTING bank
        without it. Writing a bank is usually what builds it, so a bank nobody writes to would
        never acquire it at all. This is the hook that reaches those banks, called once per tenant
        from a migration, next to the schema changes the same release makes to Postgres.

        **Not required to do the work before returning, and usually should not.** A store free to
        make this a trigger keeps the call O(1) in the tenant's size, which is what allows it in a
        pre-upgrade hook at all: a migration whose duration grows with the largest tenant turns
        "slow" into "failed release". The caller polls instead.

        **Migrations are NAMED, not numbered.** A store handed an id it does not have must raise,
        not return quietly: that turns a deploy which rolled this side ahead of the store into a
        loud failure rather than one that reports success and migrates nothing. Numbering would
        make the same mistake a silent no-op, which is why it is not the interface.

        Expected to be idempotent, and to detect per unit what is already done — a caller may
        retry freely, and recovers a partial run by calling again rather than by recording how far
        it got. **This side keeps the record of what has run**, in the migration that calls it; a
        second watermark in the store would only disagree with it.

        ``(0, 0)`` by default: a store with no derived state of its own has nothing to migrate.
        """
        return (0, 0)

    async def store_migration_status(self, *, migration_id: str) -> "tuple[int, int]":
        """How far a store migration has got in this tenant — ``(remaining, stuck)``.

        Done when ``remaining`` is 0. ``stuck`` counts units the store has tried repeatedly without
        finishing; it is not a failure the store gives up on, but without it one poison unit would
        make a caller poll forever.

        ``(0, 0)`` by default, which reads as "finished" — correct for a store that never started
        anything.
        """
        return (0, 0)

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

    async def set_document_file(
        self,
        *,
        bank_id: str,
        document_id: str,
        storage_key: str,
        original_name: str,
        content_type: str,
    ) -> bool:
        """Record on a document RECORD the uploaded file it was converted from, leaving its bodies
        alone. Returns ``False`` when the document does not exist.

        Only a ``store_owned`` store implements this; a Postgres store keeps the reference on its
        own ``documents`` row, so the engine calls it only for a store-owned bank. It is a separate
        write because a file-convert retain learns the reference in its own task, after the retain
        that wrote the record; without it the reference had nowhere to go and was silently dropped.

        The storage key is a pointer into Hindsight's ``file_storage``, not bytes the store holds; it
        goes in the record's metadata under :data:`DOC_META_FILE_STORAGE_KEY`. A later write that replaces the
        document's content replaces the record, and with it the reference: the new content was not
        converted from that file."""
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
        tags_match: TagsMatch = "any",
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
        """The retriever backing the graph arm.

        Postgres returns the SQL retriever ``config.graph_retriever`` chooses, as it
        always has. An implementation that owns the links returns its own, because
        the SQL retrievers would walk tables it never wrote to. ``None`` (the default)
        is for a store whose ``recall_unified`` never asks for one; asking then raises.
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
        tags_match: TagsMatch = "any",
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
        tag_groups: "list[TagGroup] | None" = None,
    ) -> dict[str, Any]:
        """One page of a bank's tag histogram, filtered/sorted/paged by the store.

        ``tag_groups`` (a caller's forced tag scope) limits the histogram to the memories it admits.

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
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        entity_ids: list[str] | None = None,
        tags: list[str] | None = None,
        tags_match: TagsMatch = "any",
        tag_groups: list | None = None,
    ) -> dict[str, int]:
        """Live memory count per entity id.

        Entities with no live memories are absent, so an id passed in and not
        returned is an orphan.

        ``tags``/``tags_match``/``tag_groups`` count only the memories that match (same modes as
        anywhere else), so an entity no matching memory mentions is absent too —
        that is how a tag-scoped entity read hides it (#5031).
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
        tags_match: TagsMatch = "any",
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

    @abstractmethod
    async def newest_memory_updated_at(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        until: datetime,
        since: datetime | None = None,
        fact_types: list[str] | None = None,
        tags: list[str] | None = None,
        tags_match: TagsMatch = "any",
        tag_groups: list | None = None,
    ) -> datetime | None:
        """The newest ``updated_at`` in the given scope within ``(since, until]``, or None.

        Backs the mental-model refresh, which reads it for two answers: whether the
        scope holds anything to reflect over at all (None means no), and the
        watermark the refresh persists — the newest memory it could have read, so the
        next staleness check asks about writes after it. ``since`` is the delta
        window's lower bound (None for a full refresh), which is also what lets a
        store bound the read to the writes since the last refresh.

        The scope is the same as :meth:`any_memory_updated_since`'s. Abstract rather
        than defaulted: a store that answered None here would leave every refresh
        with nothing to read, which is silent rather than wrong-looking.
        """

    async def latest_memory_write_at(self, *, conn, fq_table, bank_id: str) -> datetime | None:
        """The newest ``updated_at`` across the bank's memories, or None if it has none.

        The bank-wide counterpart of :meth:`any_memory_updated_since`, and the
        shortcut in front of it: a mental model whose watermark is at or past this
        cannot be stale whatever its scope, so every staleness surface asks this
        once and only then asks the scoped question for the models it cannot rule
        out. That is worth a method of its own because the scoped check is the
        expensive one — it is bounded by the writes since a model's watermark, and
        a model whose own scope has been quiet pays for all of them.

        None means the bank has no memories, never "unknown": a store that cannot
        answer cheaply should leave the default in place rather than return None,
        which callers read as an empty bank and act on.

        The default is the value ``consolidation_freshness`` already computes, so a
        store works without implementing this; override it when the aggregate costs
        more than the single value does (Postgres reads it off the
        ``(bank_id, updated_at)`` index instead of scanning to count).
        """
        fresh = await self.consolidation_freshness(conn=conn, fq_table=fq_table, bank_id=bank_id)
        return fresh.get("last_memory_write_at")

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
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        time_field: str,
        trunc: str,
        since: datetime,
        tag_groups: "list[TagGroup] | None" = None,
    ) -> list[dict[str, Any]]:
        """``[{"bucket": datetime, "fact_type": str, "count": int}]`` since ``since``.

        Memories bucketed by ``time_field`` truncated to ``trunc`` (minute / hour /
        day) on UTC boundaries, broken down by fact_type — the caller fills the
        empty buckets. ``time_field`` is one of created_at / mentioned_at /
        occurred_start (the event-time fields fall back to created_at per memory).
        """
        raise NotImplementedError

    async def observation_scope_counts(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        limit: int = 100,
        offset: int = 0,
        tag_groups: "list[TagGroup] | None" = None,
    ) -> dict[str, Any]:
        """One page of the observation scope histogram, paged by the store.

        Returns ``{"scopes": [{"tags": list[str], "count": int}], "total",
        "limit", "offset"}``. A scope is the sorted set of tags an observation
        was consolidated with; ``[]`` is the global (untagged) scope.
        Most-populous first, then scope ascending. ``total`` counts every
        distinct scope, not just the page — a bank has as many scopes as it has
        distinct tag sets, so the store must not ship them all for the caller
        to trim.
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
        tags_match: TagsMatch = "any",
        tag_groups: "list[TagGroup] | None" = None,
        created_before: "datetime | None" = None,
        time_field: str | None = None,
        start_date: "datetime | None" = None,
        end_date: "datetime | None" = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """One page of the curation list: ``{"items": [...], "total": int}``.

        ``total`` is the count matching the filters, not the page size, because
        the UI pages on it.

        ``tag_groups`` (AND-ed with ``tags``) carries a caller's forced tag scope.

        ``time_field`` / ``start_date`` / ``end_date`` are one time window: the
        named axis filters AND orders, and memories with no value on it are left
        out — so ``total`` can legitimately be 0 on a bank full of memories none
        of which carry that timestamp. See :mod:`hindsight_api.engine.time_filter`
        for the full contract a store must honour.

        A store that owns its rows puts each memory's attachment ids on its item as
        ``"attachment_ids": list[str]`` (see :data:`META_ATTACHMENT_IDS`). The HTTP
        layer takes the key off and resolves the ids; it cannot read them back from
        ``memory_units``, which holds none of a store-owned bank's memories. An item
        without the key shows no attachments.
        """

    @abstractmethod
    async def get_memory_unit(self, *, conn, ops, fq_table, bank_id: str, unit_id: str) -> dict[str, Any] | None:
        """One memory rendered for the curation detail view, or ``None``.

        Carries ``"attachment_ids"`` on the same terms as :meth:`list_memory_units`.
        """

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
        """Write a memory's embedding, recomputed by the caller, leaving its fields as they are.

        Its own method because the general :meth:`update_memories` is a no-op for
        the store whose write is the row itself — restoring an invalidated memory has
        to put a freshly computed vector back on it, so this is a real write.
        ``embedding`` is a float list or the pgvector literal.

        A curation edit no longer arrives here. It used to call this straight after
        :meth:`apply_edit` — a second write of the row that call had just written, which
        for a store whose write is a durable append is the whole cost of the edit again —
        so the vector now rides the edit itself as ``apply_edit(embedding=...)``. What is
        left for this method is writing a vector when no field is changing alongside it.

        The vector is part of the memory, so this stamps ``updated_at`` itself rather
        than leaning on a statement a caller happens to pair it with
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
        embedding=None,
        current_fact_type: str | None = None,
        exact_entity_names: bool = False,
    ) -> None:
        """Apply a curation field edit to a live memory.

        Writes the new text / context / fact_type / occurred window, resets the
        consolidation markers (the memory re-consolidates) and stamps the edit
        time, and drops the memory's derived links (they are recomputed).

        ``embedding`` is the vector the caller re-embedded from the new fields, and
        writing it is **part of applying the edit** — an implementation writes it
        alongside the fields above rather than leaving it for a following
        :meth:`set_memory_embedding`. Where a write is a durable append rather than
        a row update, a separate call is a second write of the row this one just
        wrote and doubles what an edit costs. ``None`` leaves the stored vector
        alone. (:meth:`set_memory_embedding` remains for the paths that write a
        vector without editing fields, such as restoring an invalidated memory.)

        ``current_fact_type`` is the memory's fact_type BEFORE this edit, which the
        caller has just read under this transaction. A fact-type change is the one
        part of an edit that some stores cannot apply as a partial update, and
        discovering it here would cost a read the caller has already paid for. It
        may be ``None``, from a caller that does not have it.

        The new entity set for the memory is supplied one of two ways, and a store
        uses whichever fits how it keeps its registry:

        * ``entity_names`` — the raw names the edit resolved to. A store that owns
          its entity registry resolves + mints these against its OWN registry
          (exactly as its :meth:`retain` does) and rewrites the memory's entity
          ids from the result, so a brand-new entity created by an edit lands in
          that registry. When it is not ``None`` it is the authoritative set and
          ``entity_ids`` is ignored. ``exact_entity_names`` set means the caller
          opted out of resolution (``resolve_entities=False``): match the names
          exactly and mint otherwise, never fuzzy-merge them.
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
        tags: list[str] | None = None,
        tags_match: TagsMatch = "any",
        tag_groups: list | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Entities in a bank with their ``mention_count``, paged and ordered by it.

        ``search`` is an optional case-insensitive substring match on the canonical
        name. Returns ``{items, total, limit, offset}``.

        ``tags``/``tags_match`` and ``tag_groups`` (AND-ed, fuzzy leaves already
        resolved) filter on the memories that mention each entity (same modes as
        anywhere else). An entity is listed only when a matching memory
        mentions it, and its ``mention_count``, ``first_seen``, ``last_seen`` and
        ``total`` cover matching memories only — the stored totals would reveal how
        much out-of-scope memories talk about it (#5031). A store that cannot filter
        must raise rather than ignore the filter."""

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
        tags_match: TagsMatch = "all_strict",
        tag_groups: "list[TagGroup] | None" = None,
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

    async def graph_view(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        fact_type: "str | list[str] | None" = None,
        search_query: str | None = None,
        document_id: str | None = None,
        chunk_id: str | None = None,
        tags: list[str] | None = None,
        tags_match: TagsMatch = "all_strict",
        tag_groups: "list[TagGroup] | None" = None,
        limit: int = 1000,
    ) -> dict[str, Any]:
        """Everything one graph render reads, in one pass:
        ``{"units", "total", "links", "entity_rows"}``.

        The three questions are about ONE set of memories — which are visible, how they link, and
        what they mention — and against a store that keeps memories outside SQL they are all
        answered by the same records. Asked separately, the visible set is read, then read again to
        pick its edges off, then a third time for its entity ids. Asked together, once.

        This default composes the parts, which is right where they really are different queries
        (SQL joins different tables per question). A store that would re-read overrides it.

        ``entity_rows`` covers the units AND their source memories, because an observation borrows
        its sources' entities for the render.
        """
        page = await self.graph_units(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            # `graph_view` accepts a list of fact types; `graph_units`, which stores override,
            # declares a single one. Widening `graph_units` would land on every implementer, so
            # the mismatch is stated here -- a list caller reaches a store that cannot take one.
            fact_type=cast("str | None", fact_type),
            search_query=search_query,
            document_id=document_id,
            chunk_id=chunk_id,
            tags=tags,
            tags_match=tags_match,
            tag_groups=tag_groups,
            limit=limit,
        )
        units = page["units"]
        ids = [str(row["id"]) for row in units]
        sources = sorted({str(s) for row in units for s in (row["source_memory_ids"] or [])})
        links, entity_rows = await self.graph_links_and_entities(
            conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=ids + sources
        )
        return {"units": units, "total": page["total"], "links": links, "entity_rows": entity_rows}

    async def graph_links_and_entities(
        self, *, conn, fq_table, bank_id: str, unit_ids: list[str]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """``(direct_links, entity_rows)`` for the same ``unit_ids``, in one pass.

        The graph view asks both questions about exactly one set of memories, and a store that
        keeps memories outside SQL answers both from the same records — so asking separately reads
        the whole set twice for a single render. This default keeps that shape for stores where the
        two really are different queries (SQL joins different tables); a store that would re-read
        overrides it and reads once.
        """
        links = await self.graph_direct_links(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids)
        entities = await self.graph_entity_rows(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids)
        return links, entities

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
        entities appears three times. This is the join from a memory to the
        entities it mentions; the registry rows themselves are the store's too.
        ``bank_id`` is passed because a store that keeps the posting on the memory
        (rather than in a global join table) needs to know which namespace the
        units live in — the Postgres join is keyed by global unit id
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
        """Queue the entities ``affected_unit_ids`` reference as prune candidates,
        and give back the ``mention_count`` their postings contributed.

        Zero for a store that never wrote `unit_entities`: it has no entity
        postings to lose, so nothing can become an orphan and no mention count
        can drift.
        """
        return 0

    async def entity_prune_pass(
        self, *, backend, fq_table, bank_id: str, deadline: float | None = None
    ) -> "EntityPrunePassResult":
        """Prune queued candidate entities and the co-occurrences they stranded.

        All-zero when the store keeps no entity postings and so queues nothing.
        """
        return EntityPrunePassResult()

    # ------------------------------------------------------------------ documents and chunks
    #
    # The document/chunk routes, the attachment lookups and the recall enrichment. Each default
    # is what a store that owns its memories answers, through the methods above; the Postgres
    # store overrides it with the SQL the engine used to run inline.

    async def record_document_file(
        self,
        *,
        backend,
        fq_table,
        bank_id: str,
        document_id: str,
        storage_key: str,
        original_name: str,
        content_type: str,
    ) -> bool:
        """Record on a document the uploaded file it was converted from. ``False`` if it is absent.

        Default: :meth:`set_document_file` on the store's own record. Postgres updates the
        `documents` row on a connection it acquires from ``backend``."""
        return await self.set_document_file(
            bank_id=bank_id,
            document_id=document_id,
            storage_key=storage_key,
            original_name=original_name,
            content_type=content_type,
        )

    async def memory_attachment_refs(
        self, *, conn, fq_table, bank_id: str, unit_ids: list[str]
    ) -> dict[str, AttachmentRef]:
        """Each memory's own attachment ids, keyed by unit id; memories with none are omitted.

        Default: the ids the store carries on each row (``StoredMemory.attachment_ids``).
        Postgres reads `memory_units.attachment_ids`."""
        rows = await self.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=list(unit_ids))
        return {m.unit_id: AttachmentRef(m.document_id, list(m.attachment_ids)) for m in rows if m.attachment_ids}

    async def recall_observation_chunk_ids(
        self,
        *,
        backend,
        ops,
        fq_table,
        bank_id: str,
        observation_ids: list[uuid.UUID],
        carried_sources: Mapping[str, list[str] | None],
    ) -> ObservationChunkIds:
        """The chunk ids of each observation's sources, for recall ``include_chunks``.

        ``carried_sources`` is the sources each observation already carries on its recall
        result (``None`` where it carries none). Default: resolve each observation's sources —
        free when every one was carried, one addressed read otherwise — then read those source
        memories for their chunk_ids, walked in observation-rank order. Postgres answers with one
        join on a connection it acquires from ``backend``, and ignores ``carried_sources``."""
        # The first half is free when the results carry their sources: hydration already fetched
        # these observations whole, so re-fetching them to read one list back off is an addressed
        # read that buys nothing. The SECOND read stays either way — the sources are memories
        # recall never retrieved, and their chunk_ids are genuinely new.
        sources_read: dict[str, list[str]] | None = None
        if all(carried_sources.get(str(o)) is not None for o in observation_ids):
            sources_by_obs = {str(o): (carried_sources[str(o)] or []) for o in observation_ids}
        else:
            obs_units = await self.get_memories(
                conn=None, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(o) for o in observation_ids]
            )
            sources_by_obs = {u.unit_id: [str(s) for s in (u.source_memory_ids or [])] for u in obs_units}
            sources_read = sources_by_obs
        src_ids = [sid for sids in sources_by_obs.values() for sid in sids]
        srcs = await self.get_memories(
            conn=None, fq_table=fq_table, bank_id=bank_id, unit_ids=list(dict.fromkeys(src_ids))
        )
        src_chunk = {s.unit_id: s.chunk_id for s in srcs}
        by_obs: dict[str, list[str]] = {}
        for obs_uuid in observation_ids:
            obs_sources = sources_by_obs.get(str(obs_uuid))
            if obs_sources is None:
                continue
            for sid in obs_sources:
                cid = src_chunk.get(sid)
                if cid:
                    by_obs.setdefault(str(obs_uuid), []).append(cid)
        return ObservationChunkIds(chunk_ids_by_observation=by_obs, sources_by_observation=sources_read)

    async def recall_chunks(self, *, backend, fq_table, bank_id: str, chunk_ids: list[str]) -> dict[str, Any]:
        """The recall ``include_chunks`` candidates by chunk_id: rows with ``chunk_text``,
        ``chunk_index`` and ``document_id`` (the last decides whether a tag-scoped reader may see
        the chunk, #5030). Absent chunks are omitted.

        Default: a store that owns the document store keeps no `chunks` rows, so the metadata is
        synthesized from the chunk ids themselves — the id carries the document and the index,
        and ``bank_id`` is known (see ``engine/chunk_ids.py``) — and the text comes from
        :meth:`get_chunk_texts`. Postgres reads the `chunks` rows on a connection it acquires from
        ``backend``."""
        from ..chunk_ids import resolve_chunk_id_in

        chunks_rows = []
        for _cid in chunk_ids:
            _ref = resolve_chunk_id_in(_cid, bank_id)
            if _ref is None:
                continue
            chunks_rows.append(
                {
                    "chunk_id": _cid,
                    "chunk_text": "",
                    "chunk_index": _ref.chunk_index,
                    "document_id": _ref.document_id,
                }
            )
        # Rows are mutable dicts so the fetch below can write into them.
        chunks_lookup = {row["chunk_id"]: dict(row) for row in chunks_rows}
        if chunks_lookup:
            rows_by_doc: dict[str, list[dict]] = {}
            for row in chunks_lookup.values():
                rows_by_doc.setdefault(row["document_id"], []).append(row)

            # Two things decide the cost here, and counting round-trips alone gets
            # both wrong.
            #
            # 1. `list_chunk_texts` downloads a document's WHOLE packed chunk blob.
            #    When a document contributes a single chunk to this recall — the
            #    common case, since hits are spread across documents — fetching that
            #    one chunk is strictly less data. So pick per document rather than
            #    using one call shape for everything.
            # 2. These were awaited in a loop, which serialises one store round-trip
            #    per document. That, not the number of calls, was the cost: measured
            #    end to end, a 30-hit recall spent ~20.6s here, ~690ms per document,
            #    against ~735ms for the entire un-hydrated recall.
            # 3. Concurrency only hides round-trips, it does not remove them. A
            #    recall's hits are spread thin across documents — measured, 88
            #    chunks over 76 documents — so per-document fetching was ~76 round
            #    trips, and at a bounded concurrency that is still several waves.
            #    `get_chunk_texts` asks for all of them at once; its default
            #    implementation on the interface is the per-chunk loop, so this is
            #    correct for every store and merely cheaper for one that batches.
            _rows = [r for rows in rows_by_doc.values() for r in rows]
            _refs = [(r["document_id"], r["chunk_index"]) for r in _rows]
            if _refs:
                try:
                    _texts = await self.get_chunk_texts(bank_id=bank_id, refs=_refs)
                    for _row, _text in zip(_rows, _texts):
                        if _text is not None:
                            _row["chunk_text"] = _text
                except Exception as _err:
                    # Returning the hits without text beats failing a whole recall
                    # over chunk bodies, which is how the per-document path behaves
                    # too — one failure there costs one document, not the request.
                    logger.warning(
                        "batched chunk hydration failed; returning %d chunk(s) without text: %s",
                        len(_rows),
                        _err,
                    )
        return chunks_lookup

    async def recall_observation_sources(
        self, *, conn, fq_table, bank_id: str, observation_ids: list[uuid.UUID]
    ) -> dict[str, Any]:
        """Each observation's source memory ids, keyed by observation id in observation-rank order.

        The token budget recall fills from these is filled in this order, so an unordered read
        would let a low-ranked observation spend the budget the top-ranked one needs (#3221).
        Default: one addressed read of the observations. Postgres reads the two columns it needs."""
        obs_by_id = {
            m.unit_id: m
            for m in await self.get_memories(
                conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(o) for o in observation_ids]
            )
            if m.fact_type == "observation"
        }
        return {
            m.unit_id: m.source_memory_ids for m in (obs_by_id.get(str(o)) for o in observation_ids) if m is not None
        }

    async def recall_source_facts(
        self, *, conn, fq_table, bank_id: str, unit_ids: list[str]
    ) -> dict[str, StoredMemory]:
        """The source facts recall shows under its observations, keyed by unit id.

        Only the display fields need be set (text, fact_type, context, the three content times,
        document_id, chunk_id, tags, metadata). Default: :meth:`get_memories`. Postgres selects
        just those columns."""
        return {
            m.unit_id: m
            for m in await self.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids)
        }

    async def get_document_with_counts(
        self, *, conn, ops, fq_table, bank_id: str, document_id: str
    ) -> Mapping[str, Any] | None:
        """A document with its memory counts, or ``None``: ``id, bank_id, original_text,
        content_hash, created_at, updated_at, tags, retain_params, unit_count, world_count,
        experience_count, observation_count``.

        Default: the store's record (with its text), counted by scanning the document's memories
        and the observations built on them; it also carries ``attachment_filenames``, an internal
        carrier the get-document route takes off. Postgres reads the `documents` row and counts
        in one statement."""
        _rec = await self.get_document_record(bank_id=bank_id, document_id=document_id, include_text=True)
        if _rec is None:
            return None
        doc: dict[str, Any] = {
            "id": _rec.get("document_id") or document_id,
            "bank_id": bank_id,
            "original_text": _rec.get("original_text"),
            "content_hash": _rec.get("content_hash"),
            "created_at": _epoch_ms_to_datetime(_rec.get("created_at")),
            "updated_at": _epoch_ms_to_datetime(_rec.get("updated_at")),
            "tags": list(_rec.get("tags") or []),
            # The store's metadata map is string -> string, so the write path
            # carries retain_params as one JSON value (see
            # `retain/orchestrator._store_document_bodies`). Documents written
            # before that carry nothing here and still read back with null
            # params — null beats 404-ing the whole document.
            "retain_params": (_rec.get("metadata") or {}).get("retain_params"),
            # The document's attachment names, from the same record. An internal
            # carrier, not a response field: the get-document route and reprocess
            # take it off, so the payload is the same shape on either backend.
            "attachment_filenames": document_attachment_filenames(_rec),
        }
        _page = await self.scan_memories(
            conn=conn, fq_table=fq_table, bank_id=bank_id, document_id=document_id, limit=1_000_000
        )
        doc["unit_count"] = len(_page.memories)
        doc["world_count"] = sum(1 for m in _page.memories if m.fact_type == "world")
        doc["experience_count"] = sum(1 for m in _page.memories if m.fact_type == "experience")
        _sids = [m.unit_id for m in _page.memories if m.fact_type in ("experience", "world")]
        _obs = (
            await self.observations_for_sources(conn=conn, ops=ops, fq_table=fq_table, bank_id=bank_id, unit_ids=_sids)
            if _sids
            else []
        )
        doc["observation_count"] = len(_obs)
        return doc

    async def document_source_units(self, *, conn, fq_table, bank_id: str, document_id: str) -> DocumentSourceUnits:
        """The document's experience/world memory ids (for observation cleanup) and its total
        memory count, read before it is deleted.

        Default: a scan of the document's memories and :meth:`document_memory_counts`. Postgres
        reads `memory_units`."""
        src_page = await self.scan_memories(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            document_id=document_id,
            fact_types=["experience", "world"],
            limit=1_000_000,
        )
        _doc_counts = await self.document_memory_counts(
            conn=conn, fq_table=fq_table, bank_id=bank_id, document_ids=[document_id]
        )
        return DocumentSourceUnits(
            unit_ids=[m.unit_id for m in src_page.memories], units_count=_doc_counts.get(document_id, 0)
        )

    async def delete_document_rows(
        self, *, conn, ops, fq_table, bank_id: str, document_id: str, unit_ids: list[str]
    ) -> DeletedDocument:
        """Delete a document and its memories — the explicit document deletion.

        ``unit_ids`` are the document's memories from :meth:`document_source_units`. Default: the
        store's record decides whether the document existed and where its uploaded file lives,
        then :meth:`delete_document` drops its memories and :meth:`delete_document_record` the
        record and bodies. Postgres drops the units' links and deletes the `documents` row, whose
        FK cascade takes the memories."""
        # Whether the RECORD existed has to be established before deleting it, and it
        # cannot be inferred from `store_owned_for` — that is a capability of the store,
        # true for every bank it serves, so using it here reported a successful deletion
        # for a document that never existed and turned the 404 this endpoint promises
        # into a 200.
        #
        # One read answers both questions: whether the record exists, and where the
        # uploaded original the document was converted from lives (`set_document_file`
        # puts the key in the record's metadata) — there is no SQL row to hold it.
        #
        # When the store owns the document store its RECORD is the authority, and the
        # memory count is deliberately not consulted: "document not found" is a statement
        # about the document, and a document with no memories still exists. It also cannot
        # be trusted here — the per-document count is a per-segment tally that does not
        # subtract a delete still sitting in the un-folded tail, so straight after a delete
        # it reports the pre-delete number and a second delete of the same document would
        # report success.
        _record = await self.get_document_record(bank_id=bank_id, document_id=document_id)
        file_storage_key = ((_record or {}).get("metadata") or {}).get(DOC_META_FILE_STORAGE_KEY)
        await self.delete_document(conn=conn, fq_table=fq_table, bank_id=bank_id, document_id=document_id)
        # The EXPLICIT deletion also drops the document RECORD (its extracted text + chunk
        # bodies; the orphan sweep reclaims the blobs) — distinct from the re-ingest
        # facts-delete above.
        await self.delete_document_record(bank_id=bank_id, document_id=document_id)
        return DeletedDocument(deleted=_record is not None, file_storage_key=file_storage_key)

    async def current_document_tags(self, *, conn, fq_table, bank_id: str, document_id: str) -> DocumentTags:
        """The tags a document carries now, read before a retag overwrites them.

        Default: the store's record. A record that does not carry a "tags" key at all is
        unreadable for this purpose, not empty — collapsing it to [] would read a PATCH of []
        as "unchanged" and skip a real clear-the-tags request. Postgres reads the `documents` row."""
        _record = await self.get_document_record(bank_id=bank_id, document_id=document_id)
        return DocumentTags(
            found=_record is not None,
            tags=list(_record["tags"] or []) if _record is not None and "tags" in _record else None,
        )

    async def documents_tags(self, *, conn, fq_table, bank_id: str, document_ids: list[str]) -> dict[str, list[str]]:
        """document id -> the tags it carries, for the ids that exist. Read to decide whether a
        tag-scoped reader may see a document's source text (#5030), so an id left out is hidden.

        Default: the store's records; a record that does not carry a "tags" key is left out
        rather than read as untagged, which an ``any`` filter would admit. Postgres reads the
        `documents` rows."""
        records = await self.get_document_records(bank_id=bank_id, document_ids=document_ids)
        return {did: list(rec["tags"] or []) for did, rec in records.items() if "tags" in rec}

    async def update_document_tags(
        self, *, conn, fq_table, bank_id: str, document_id: str, tags: list[str] | None, found: bool
    ) -> bool:
        """Mark a document updated and, when ``tags`` is given, replace its OWN tags (its memories
        are retagged by :meth:`retag_document_memories`). ``False`` if the document is absent.

        ``found`` is what :meth:`current_document_tags` saw in the same transaction. Default:
        trusts it and writes the record's tags with :meth:`set_document_tags`. Postgres updates
        the `documents` row and answers from the UPDATE itself."""
        if not found:
            return False
        if tags is not None:
            await self.set_document_tags(bank_id=bank_id, document_id=document_id, tags=list(tags))
        return True

    async def retag_document_memories(
        self,
        *,
        conn,
        ops,
        fq_table,
        bank_id: str,
        document_id: str,
        tags: list[str],
        retagged: Callable[[list[str] | None], list[str]],
        rescoped: Callable[[Any], str | None],
    ) -> int:
        """Retag a document's memories to ``tags``, then invalidate the observations built on them
        and requeue their sources, so the next consolidation rebuilds them under the new tags.
        Returns how many observations were deleted.

        ``retagged(existing_tags)`` is a memory's final tag list (the document's tags plus the
        label projection it keeps); ``rescoped(observation_scopes)`` is its new scopes as JSON,
        or ``None`` when they do not change. Default: patch through :meth:`update_memories`, then
        :meth:`delete_stale_observations` (which requeues surviving co-sources) and
        :meth:`mark_consolidated`. Postgres runs the same cascade as SQL."""
        invalidated_obs = 0
        _doc_page = await self.scan_memories(
            conn=conn, fq_table=fq_table, bank_id=bank_id, document_id=document_id, limit=1_000_000
        )
        _doc_units = _doc_page.memories
        if _doc_units:
            _patches = []
            for m in _doc_units:
                _patch = MemoryPatch(unit_id=m.unit_id, tags=retagged(m.tags))
                _scopes_json = rescoped(m.observation_scopes)
                if _scopes_json is not None:
                    _patch.metadata = {META_OBSERVATION_SCOPES: _scopes_json}
                _patches.append(_patch)
            await self.update_memories(bank_id, _patches)
        _src_ids = [m.unit_id for m in _doc_units if m.fact_type in ("experience", "world")]
        if _src_ids:
            invalidated_obs = await self.delete_stale_observations(
                conn=conn, ops=ops, fq_table=fq_table, bank_id=bank_id, fact_ids=_src_ids
            )
            # Only when an observation actually went — the same condition Postgres requeues
            # under. Requeueing regardless re-consolidates the document's memories on a retag
            # that invalidated nothing, and the marker it clears is the only record that they
            # were ever consolidated: the next refresh has no way to tell the difference.
            if invalidated_obs:
                await self.mark_consolidated(
                    conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=_src_ids, when=None
                )
        return invalidated_obs

    async def list_documents_page(
        self,
        *,
        backend,
        fq_table,
        bank_id: str,
        search_query: str | None,
        tags: list[str] | None,
        tags_match: TagsMatch,
        tag_groups: "list[TagGroup] | None",
        time_field: str | None,
        start_date: datetime | None,
        end_date: datetime | None,
        limit: int,
        offset: int,
    ) -> dict:
        """A page of the bank's documents — ``{items, total, limit, offset}``.

        Default: :meth:`list_documents` on the store's own registry. Postgres pages the
        `documents` table on a connection it acquires from ``backend``."""
        return await self.list_documents(
            bank_id=bank_id,
            search_query=search_query,
            tags=tags,
            tags_match=tags_match,
            tag_groups=tag_groups,
            time_field=time_field,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )

    async def get_chunk_row(self, *, conn, fq_table, bank_id: str | None, chunk_id: str) -> Mapping[str, Any] | None:
        """One chunk by id, or ``None``: ``chunk_id, document_id, bank_id, chunk_index, chunk_text,
        created_at``.

        Called for every id the engine does not serve itself: one that does not parse (``bank_id``
        is ``None`` then), and one that parses to a bank this store does not own (``bank_id`` is
        the parsed bank). An id that parses to a store-owned bank never reaches here — the engine
        answers it from :meth:`get_chunk_text`. Default: ``None`` — a store that owns its banks
        keeps no chunk rows to look any other id up in. Postgres reads the `chunks` row by id."""
        return None

    async def list_document_chunks(
        self, *, backend, fq_table, bank_id: str, document_id: str, limit: int, offset: int
    ) -> dict | None:
        """A page of a document's chunks by index — ``{items, total, limit, offset}`` — or
        ``None`` if the document does not exist.

        Default: :meth:`list_chunk_texts`, with each chunk id rebuilt the way retain builds it.
        Postgres pages the `chunks` rows on a connection it acquires from ``backend``."""
        from ..chunk_ids import build_chunk_id

        texts = await self.list_chunk_texts(bank_id=bank_id, document_id=document_id)
        if texts is None:
            return None
        total = len(texts)
        window = list(enumerate(texts))[offset : offset + limit]
        return {
            "items": [
                {
                    # Rebuilt the same way retain builds it (``engine/chunk_ids.py``) so an
                    # id from this route is accepted by the addressed chunk route.
                    "chunk_id": build_chunk_id(bank_id, document_id, idx),
                    "document_id": document_id,
                    "bank_id": bank_id,
                    "chunk_index": idx,
                    "chunk_text": text,
                    # The store does not carry a per-chunk creation time; the document's is
                    # the closest true value and inventing one per chunk would be worse.
                    "created_at": "",
                }
                for idx, text in window
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------ curation and bank admin
    #
    # The engine's curation and bank-admin surfaces: deleting memories, clearing or deleting a
    # bank, requeueing consolidation, an observation's history, the entity views. Each default
    # below is the answer for a store that owns its memories, built from the methods above;
    # Postgres overrides every one with the SQL the engine used to run inline.

    async def locate_memory(self, *, conn, fq_table, bank_id: str | None, unit_id: str) -> MemoryLocation | None:
        """The bank and fact_type of one memory, or ``None`` if it does not exist.

        Postgres finds the bank from the row (its ids are global), so ``bank_id`` may be None
        there. A store partitioned by bank needs it, and answers ``None`` without one."""
        if not bank_id:
            return None
        found = await self.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[unit_id])
        return MemoryLocation(unit_id=unit_id, bank_id=bank_id, fact_type=found[0].fact_type) if found else None

    async def locate_memories(
        self, *, conn, fq_table, bank_id: str | None, unit_ids: list[str]
    ) -> list[MemoryLocation]:
        """:meth:`locate_memory` for many ids in one read; ids that do not exist are absent.

        Postgres ignores ``bank_id`` and finds each id's bank from its row; a store partitioned by
        bank looks only in ``bank_id`` and finds nothing without one."""
        if not bank_id:
            return []
        found = await self.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids)
        return [MemoryLocation(unit_id=m.unit_id, bank_id=bank_id, fact_type=m.fact_type) for m in found]

    async def delete_memory(self, *, conn, ops, fq_table, bank_id: str | None, unit_id: str) -> str | None:
        """Delete one memory the caller just located; returns its id, or ``None`` if nothing was
        deleted. ``bank_id`` is None when :meth:`locate_memory` found nothing.

        Postgres drops the memory's links in lock order, then the row (which cascades to the
        rest), on the caller's transaction."""
        if not bank_id:
            return None
        await self.delete_facts(bank_id, [unit_id])
        return unit_id

    async def delete_memories(self, *, conn, ops, fq_table, bank_id: str, unit_ids: list[str]) -> int:
        """Delete one bank's memories the caller just located; returns how many were deleted.

        Postgres drops their links in lock order, then the rows in chunks, and counts the rows
        its DELETEs report."""
        await self.delete_facts(bank_id, unit_ids)
        return len(unit_ids)

    async def bank_memories_of_type(self, *, conn, fq_table, bank_id: str, fact_type: str) -> TypedMemoryScope:
        """A bank's memories of one fact_type, for a typed clear: their ids (only for the source
        types, which drive the stale-observation sweep) and their count."""
        unit_ids: list[str] = []
        if fact_type in ("experience", "world"):
            page = await self.scan_memories(
                conn=conn, fq_table=fq_table, bank_id=bank_id, fact_types=[fact_type], limit=1_000_000
            )
            unit_ids = [m.unit_id for m in page.memories]
        counts = await self.count_memories(conn=conn, fq_table=fq_table, bank_id=bank_id)
        return TypedMemoryScope(unit_ids=unit_ids, count=int(counts.get(fact_type, 0)))

    async def delete_bank_memories_of_type(
        self, *, conn, ops, fq_table, bank_id: str, fact_type: str, unit_ids: list[str]
    ) -> None:
        """Delete a bank's live and archived memories of one fact_type, inside the caller's
        transaction.

        A no-op here: a store that owns its memories is told after the commit, through
        :meth:`delete_where`, by the bank delete itself. Postgres deletes the rows (links first,
        in lock order) and the matching archive rows."""

    async def count_bank_contents(self, *, conn, fq_table, bank_id: str) -> BankContentCounts:
        """How many memories, entities and documents a bank holds — what a bank delete reports."""
        counts = await self.count_memories(conn=conn, fq_table=fq_table, bank_id=bank_id)
        documents = await self.count_documents(bank_id=bank_id)
        entities = await self.list_entities(
            conn=conn, fq_table=fq_table, bank_id=bank_id, search=None, limit=1, offset=0
        )
        return BankContentCounts(
            memory_units=sum(counts.values()), entities=int(entities.get("total") or 0), documents=documents
        )

    async def purge_bank_rows(self, *, conn, fq_table, bank_id: str) -> list[str]:
        """Delete a bank's rows inside the caller's transaction, and return the storage keys of
        its pre-tenant-prefix files, read before the rows naming them go (the post-commit sweep
        cannot find those by prefix).

        This default still runs SQL, on the caller's connection: attachments and observation
        history live in Postgres for every store, so it deletes those and reads the legacy keys
        from attachments. The memories themselves are dropped after the commit, through
        :meth:`drop_bank_storage` / :meth:`delete_where`. Postgres also reads the legacy keys
        off its documents and deletes its documents, memories, archive and entities, in one
        statement order."""
        legacy_files = [
            row["storage_key"]
            for row in await conn.fetch(
                f"SELECT storage_key FROM {fq_table('attachments')} "
                f"WHERE bank_id = $1 AND storage_key NOT LIKE 'tenants/%'",
                bank_id,
            )
        ]
        await conn.execute(f"DELETE FROM {fq_table('attachments')} WHERE bank_id = $1", bank_id)
        await conn.execute(f"DELETE FROM {fq_table('observation_history')} WHERE bank_id = $1", bank_id)
        return legacy_files

    async def clear_observations_and_requeue(self, *, conn, fq_table, bank_id: str) -> int:
        """Delete every observation in a bank and requeue every source for re-consolidation;
        returns how many observations there were. Postgres counts, deletes and clears
        ``consolidated_at`` in three statements on the caller's transaction."""
        count = (await self.count_memories(conn=conn, fq_table=fq_table, bank_id=bank_id)).get("observation", 0)
        await self.delete_observations(conn=conn, fq_table=fq_table, bank_id=bank_id)
        page = await self.scan_memories(
            conn=conn, fq_table=fq_table, bank_id=bank_id, fact_types=["experience", "world"], limit=1_000_000
        )
        src_ids = [m.unit_id for m in page.memories]
        if src_ids:
            await self.mark_consolidated(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=src_ids, when=None)
        return count

    async def requeue_failed_consolidation(self, *, conn, fq_table, bank_id: str) -> int:
        """Return a bank's permanently-failed sources to the not-yet-consolidated state; returns
        how many. ``mark_consolidated(when=None)`` clears both the failed and consolidated
        markers. Postgres counts and clears both columns in two statements."""
        failed = await self.find_failed_consolidation(conn=conn, fq_table=fq_table, bank_id=bank_id)
        if failed:
            await self.mark_consolidated(
                conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[m.unit_id for m in failed], when=None
            )
        return len(failed)

    async def requeue_source_memory(self, *, conn, fq_table, bank_id: str, unit_id: uuid.UUID) -> None:
        """Clear one source memory's consolidated marker so it re-consolidates. Scheduler state:
        ``updated_at`` stays put. Postgres clears it only on a world/experience row."""
        await self.mark_consolidated(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(unit_id)], when=None)

    async def entity_names_by_id(self, *, conn, fq_table, bank_id: str, entity_ids: list[str]) -> list[str]:
        """Canonical names of ``entity_ids`` in this bank, ordered by entity id; unknown ids are
        skipped. Feeds a curation re-embed. Postgres reads its ``entities`` registry."""
        names = await self.resolve_entity_names(
            conn=conn, fq_table=fq_table, bank_id=bank_id, entity_ids=[str(e) for e in entity_ids]
        )
        return [names[eid] for eid in sorted(names)]

    async def observation_head(self, *, conn, fq_table, bank_id: str, unit_id: uuid.UUID) -> MemoryLocation | None:
        """One memory's fact_type and current ``source_memory_ids``, or ``None`` if it does not
        exist — the head an observation's history is rebuilt backwards from."""
        found = await self.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(unit_id)])
        if not found:
            return None
        return MemoryLocation(
            unit_id=str(unit_id),
            bank_id=bank_id,
            fact_type=found[0].fact_type,
            source_memory_ids=[str(s) for s in found[0].source_memory_ids],
        )

    async def source_fact_summaries(
        self, *, conn, fq_table, bank_id: str, unit_ids: list[uuid.UUID]
    ) -> list[StoredMemory]:
        """The memories an observation was built from, for its history view. Only ``text``,
        ``fact_type`` and ``context`` are read; missing ids are absent."""
        return await self.get_memories(
            conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(u) for u in unit_ids]
        )

    async def entity_graph(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        limit: int,
        min_count: int,
        tags: "list[str] | None" = None,
        tags_match: TagsMatch = "any",
        tag_groups: "list | None" = None,
    ) -> dict:
        """The entity co-occurrence graph (``{nodes, edges, ...}``). Delegates to
        :meth:`get_entity_graph`, the store's own aggregate; Postgres reads its
        ``entity_cooccurrences`` table on the caller's connection, or recomputes the
        edges from the matching memories when a tag filter is given (see
        :meth:`get_entity_graph` for the filter's contract)."""
        return await self.get_entity_graph(
            bank_id=bank_id,
            limit=limit,
            min_count=min_count,
            tags=tags,
            tags_match=tags_match,
            tag_groups=tag_groups,
        )

    async def count_bank_documents(self, *, conn, fq_table, bank_id: str) -> int:
        """This bank's document count, for the stats page. Delegates to :meth:`count_documents`;
        Postgres counts its ``documents`` table on the caller's connection."""
        return await self.count_documents(bank_id=bank_id)

    async def get_entity_detail(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        entity_id: uuid.UUID,
        tags: list[str] | None = None,
        tags_match: TagsMatch = "any",
        tag_groups: list | None = None,
    ) -> dict[str, Any] | None:
        """One entity rendered for the entity detail view, or ``None`` if the bank has no such entity.

        An addressed lookup against the store's registry, not a page-and-scan, so it stays O(1) in
        the registry size. First/last seen live on the registry record, which this lookup does not
        carry (``list_entities`` surfaces them). Postgres reads its ``entities`` row.

        With a tag filter (``tags``/``tags_match``/``tag_groups``, fuzzy leaves already resolved)
        the count covers the matching memories only, and an entity no matching memory mentions is
        ``None`` — the same answer as an unknown id, so a scoped reader cannot tell it exists in
        another scope (#5031)."""
        eid = str(entity_id)
        names = await self.resolve_entity_names(conn=conn, fq_table=fq_table, bank_id=bank_id, entity_ids=[eid])
        if eid not in names:
            return None
        counts = await self.entity_memory_counts(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            entity_ids=[eid],
            tags=tags,
            tags_match=tags_match,
            tag_groups=tag_groups,
        )
        if tag_filter_active(tags, tags_match, tag_groups) and eid not in counts:
            return None
        return {
            "id": eid,
            "canonical_name": names[eid],
            # Absent from the counts map means no live memories reference it (an orphan), which
            # is 0 rather than missing.
            "mention_count": counts.get(eid, 0),
            "first_seen": None,
            "last_seen": None,
            "metadata": {},
            "observations": [],
        }

    # ------------------------------------------------------------------ retain
    #
    # The retain path's reads and writes of the document row, its chunks and its memories, and
    # the bank-list aggregates over them. Each default is the answer for a store that owns its
    # memories: it keeps the document record itself (``put_document``), so there is no SQL row
    # to read, lock or write. Postgres overrides every one with the SQL retain used to run inline.
    # A default that raises ``NotImplementedError`` is only reached on the Postgres-only path.

    async def read_document_base(self, *, conn, fq_table, bank_id: str, document_id: str) -> DocumentBase | None:
        """The stored body and version an append concatenates onto; ``None`` if the document does
        not exist. Postgres reads the `documents` row; this reads the store's own record."""
        record = await self.get_document_record(bank_id=bank_id, document_id=document_id, include_text=True)
        if not record:
            return None
        return DocumentBase(
            original_text=record.get("original_text"),
            content_hash=record.get("content_hash"),
            attachment_filenames=document_attachment_filenames(record),
            watermark=record.get("watermark"),
        )

    async def document_updated_at(self, *, backend, fq_table, bank_id: str, document_id: str) -> datetime | None:
        """When the document was last written, for retain's best-effort stale-request check.

        ``backend`` is the pool: Postgres acquires a connection for the read. ``None`` here, so the
        check never fires: what serializes writers for a store that owns its documents is its own
        compare-and-set (``put_document(expect_watermark=...)``)."""
        return None

    async def recovery_chunk_hashes(
        self, *, backend, fq_table, bank_id: str, document_id: str, content_hash: str
    ) -> set[str]:
        """Hashes of the chunks a crashed retain of this same content (``content_hash``) already
        committed, so the retry can skip them. ``backend`` is the pool.

        Empty here: a store that owns its memories commits a retain atomically, so there is no
        half-written retain to resume."""
        return set()

    async def lock_document_for_write(self, *, conn, ops, fq_table, bank_id: str, document_id: str) -> str | None:
        """Create the document row if missing and row-lock it; the hash it carried before
        (``'__pending__'`` for a new row) — the streaming write's ownership gate.

        Here the stored hash, ``None`` if absent, and no lock: there is no row, and a store that
        owns its documents serializes writers with ``put_document(expect_watermark=...)``."""
        return await self.document_content_hash(bank_id=bank_id, document_id=document_id)

    async def lock_pending_document(self, *, conn, fq_table, bank_id: str, document_id: str) -> None:
        """Insert a ``'__pending__'`` document row if none exists, then row-lock it — the
        zero-batch retain's document tracking. A no-op here: there is no row."""

    async def lock_document_hash(self, *, conn, fq_table, bank_id: str, document_id: str) -> str | None:
        """Row-lock the document and read its content_hash (``None`` if absent) — the delta
        write's concurrency gate. Here the stored hash, unlocked (see :meth:`lock_document_for_write`)."""
        return await self.document_content_hash(bank_id=bank_id, document_id=document_id)

    async def load_document_chunks(
        self, *, backend, fq_table, bank_id: str, document_id: str, include_text: bool
    ) -> DocumentChunkState:
        """The document's content_hash (and ``original_text`` when ``include_text``), then its
        chunks — the base a delta retain diffs against. ``backend`` is the pool: Postgres acquires a
        connection for the two reads. Here ONE record read, which also carries the watermark."""
        record = await self.get_document_record(bank_id=bank_id, document_id=document_id, include_text=include_text)
        return document_chunk_state(record, bank_id=bank_id, document_id=document_id, include_text=include_text)

    async def load_existing_chunks(self, *, conn, fq_table, bank_id: str, document_id: str) -> list[ExistingChunk]:
        """The document's chunks (id, index, content hash), in chunk order."""
        record = await self.get_document_record(bank_id=bank_id, document_id=document_id)
        return document_chunk_state(record, bank_id=bank_id, document_id=document_id, include_text=False).chunks

    async def current_document_hash(self, *, backend, fq_table, bank_id: str, document_id: str) -> str | None:
        """The document's content_hash now, for the delta's pre-extraction recheck. ``backend`` is
        the pool.

        ``None`` here, so the recheck is skipped: it only saves extraction tokens, and the delta
        write's compare-and-set on the store's watermark is what catches a concurrent writer."""
        return None

    async def document_original_text(self, *, conn, fq_table, bank_id: str, document_id: str) -> str | None:
        """The document's stored text; ``None`` if it does not exist or keeps none."""
        record = await self.get_document_record(bank_id=bank_id, document_id=document_id, include_text=True)
        return (record or {}).get("original_text")

    async def count_document_memories(self, *, conn, fq_table, bank_id: str, document_id: str) -> int:
        """How many memories the document owns — zero means it cannot be recalled."""
        counts = await self.document_memory_counts(
            conn=conn, fq_table=fq_table, bank_id=bank_id, document_ids=[document_id]
        )
        return int(counts.get(document_id, 0))

    async def document_unit_ids(self, *, conn, fq_table, bank_id: str, document_id: str) -> list[str]:
        """Every memory id the document owns, oldest first — what a retain resumed after a crash
        reports as its result."""
        found: list[StoredMemory] = []
        page_token = ""
        while True:
            page = await self.scan_memories(
                conn=conn, fq_table=fq_table, bank_id=bank_id, document_id=document_id, limit=500, page_token=page_token
            )
            found.extend(page.memories)
            page_token = page.next_page_token
            if not page_token:
                break
        # Oldest first, undated last — the order Postgres's `ORDER BY created_at` gives.
        found.sort(key=lambda m: (m.created_at is None, m.created_at or datetime.min.replace(tzinfo=timezone.utc)))
        return [m.unit_id for m in found]

    async def delete_document_for_replace(
        self, *, conn, ops, fq_table, bank_id: str, document_id: str, outgoing_unit_ids: list[str]
    ) -> datetime | None:
        """Remove the document's memories, links and row before it is written again; the row's
        ``created_at``, so the re-ingested document keeps it. Here only the memories go, through
        :meth:`delete_document`: there is no SQL row, so nothing to carry forward."""
        await self.delete_document(conn=conn, fq_table=fq_table, bank_id=bank_id, document_id=document_id)
        return None

    async def upsert_document_row(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        document_id: str,
        original_text: str | None,
        content_hash: str,
        retain_params: dict | None,
        document_tags: list[str] | None,
        preserved_created_at: datetime | None,
    ) -> None:
        """Insert or update the SQL document row. A no-op here: the retain path writes the
        document through :meth:`put_document`, and there is no SQL row."""

    async def relabel_document_memories(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        document_id: str,
        tags: list[str],
        metadata: dict[str, Any],
        observation_scopes: list | str | None,
        final_tags: Callable[[list[str] | None], list[str]],
    ) -> RelabelResult:
        """Give every memory of the document the document's current tags, metadata and
        observation scoping. ``final_tags`` maps a memory's current tags to the ones it ends with
        (the document tags, plus the label projection it already carries)."""
        from ..metadata_utils import drop_null_values
        from ..retain.fact_storage import _normalize_scopes

        page = await self.scan_memories(
            conn=conn, fq_table=fq_table, bank_id=bank_id, document_id=document_id, limit=1_000_000
        )
        # Metadata too, not just tags: a survivor left carrying the PREVIOUS retain's metadata is
        # exactly what this exists to prevent — measured on an append, older units still read
        # {"source": "email"} after a retain carrying {"source": "crm"}.
        #
        # Written under META_METADATA_JSON as one JSON value, which is where the bag contract puts a
        # memory's user metadata and what every read reconstructs it from. A flat {"source": "crm"}
        # would merge a stray top-level key into the record's own bag instead: applied, reported as
        # applied, and invisible to every reader. The bag's other keys are internal (context,
        # chunk_id, consolidation_failed_at, …) and a patch that carried user keys loose among them
        # could not be told apart from one setting an internal field.
        #
        # Set unconditionally, mirroring Postgres's `SET metadata = $4`: a document whose metadata
        # was cleared must clear on its survivors too, which an absent key would not do.
        new_tags_by_unit = {m.unit_id: final_tags(m.tags) for m in page.memories}
        patches = [
            MemoryPatch(
                unit_id=m.unit_id,
                tags=new_tags_by_unit[m.unit_id],
                metadata={
                    META_METADATA_JSON: json.dumps(drop_null_values(metadata or {})),
                    META_OBSERVATION_SCOPES: json.dumps(observation_scopes),
                },
            )
            for m in page.memories
        ]
        if patches:
            await self.update_memories(bank_id, patches)
        # Against the tags the unit will actually END with, not the document's: a survivor
        # keeping its label projection has not been rescoped, and comparing it to the bare
        # document tags reported every such unit as moved on every retain — an observation
        # sweep and a full re-consolidation of the document for no change at all.
        rescoped = [
            m.unit_id
            for m in page.memories
            if m.fact_type in ("experience", "world")
            and (
                set(m.tags or []) != set(new_tags_by_unit[m.unit_id])
                or _normalize_scopes(m.observation_scopes) != _normalize_scopes(observation_scopes)
            )
        ]
        return RelabelResult(updated=len(patches), rescoped_unit_ids=rescoped)

    async def memory_ids_for_chunks(self, *, conn, fq_table, bank_id: str, chunk_ids: list[str]) -> list[str]:
        """Ids of the ``experience``/``world`` memories these chunks own (observations are not
        chunk-scoped). Paged to exhaustion: every id is about to be deleted, and a chunk whose
        facts overflow one page must not keep half of them."""
        unit_ids: list[str] = []
        for chunk_id in chunk_ids:
            page_token = ""
            while True:
                page = await self.scan_memories(
                    conn=conn,
                    fq_table=fq_table,
                    bank_id=bank_id,
                    fact_types=["experience", "world"],
                    metadata_equals={META_CHUNK_ID: chunk_id},
                    limit=500,
                    page_token=page_token,
                )
                unit_ids.extend(m.unit_id for m in page.memories)
                page_token = page.next_page_token
                if not page_token:
                    break
        return unit_ids

    async def delete_chunks(self, *, conn, fq_table, bank_id: str, chunk_ids: list[str]) -> None:
        """Delete the named chunks and the memories (and links) they own. Here: the memories
        carrying each chunk_id, through :meth:`delete_where` — the store keeps no chunk rows, and
        without this a delta re-ingest leaves the old memories as duplicates."""
        for chunk_id in chunk_ids:
            await self.delete_where(bank_id, DeletePredicate(metadata_equals={META_CHUNK_ID: chunk_id}))

    async def upsert_chunks(
        self,
        *,
        conn,
        ops,
        fq_table,
        bank_id: str,
        document_id: str,
        chunk_ids: list[str],
        chunk_texts: list[str],
        chunk_indices: list[int],
        content_hashes: list[str],
    ) -> None:
        """Write the document's chunk rows. A no-op here: the chunk texts and hashes travel with
        the document record (:meth:`put_document`), and there are no SQL chunk rows."""

    async def unit_embeddings(self, *, conn, fq_table, bank_id: str, unit_ids: list[str]) -> list:
        """``(id, embedding, fact_type)`` rows, embedding as pgvector text, for the end-of-retain
        semantic-link pass. Empty here, so the pass links nothing: that pass writes SQL links, and
        a store that owns its memories derives its own (and sets
        :attr:`derives_semantic_links_internally`, which skips the pass altogether)."""
        return []

    # ------------------------------------------------------------------ retain links and entity resolution
    #
    # The retain-time link writes, the ANN neighbour search behind them, and entity resolution.
    # All of them are reached only for a bank whose memories are SQL rows: the store-owned retain
    # and import hand the store entity names and causal edges inline and let it derive its own
    # temporal / semantic links, and the memory edit branches on ``store_owned_for`` before it
    # resolves anything. So each default is the store-owned answer: nothing to write, no
    # neighbours, no entities resolved.

    async def resolve_entities(
        self,
        *,
        entity_resolver: EntityResolverHandle,
        conn,
        bank_id: str,
        unit_ids: list[str],
        sentences: list[str],
        context: str,
        fact_dates: list,
        llm_entities: list[list[dict]],
        log_buffer: list[str] | None = None,
        entity_labels: list | None = None,
    ) -> EntityResolutionResult:
        """Resolve each unit's entity mentions to canonical entity ids, creating the missing ones
        (autocommitted on ``conn``, outside the write transaction).

        Empty here — the same answer retain's phase 1 gives a store-owned bank without asking:
        such a store resolves the names its write carries itself."""
        from ..retain.types import EntityResolutionResult

        return EntityResolutionResult(resolved_entities=[], entity_to_unit=[], unit_to_entity_ids={})

    async def compute_semantic_links_ann(
        self,
        *,
        conn,
        bank_id: str,
        unit_ids: list[str],
        embeddings: Sequence[EmbeddingLike],
        fact_types: list[str] | None = None,
        top_k: int = 50,
        threshold: float,
        log_buffer: list[str] | None = None,
    ) -> list[tuple]:
        """Semantic neighbours among the bank's existing memories, as link tuples, for
        :meth:`create_semantic_links` / :meth:`insert_links`. None here: the store derives its own."""
        return []

    async def create_temporal_links(self, *, conn, ops, bank_id: str, unit_ids: list[str]) -> int:
        """Link each new memory to the memories close to it in time; how many links were written."""
        return 0

    async def create_semantic_links(
        self,
        *,
        conn,
        ops,
        bank_id: str,
        unit_ids: list[str],
        embeddings: Sequence[EmbeddingLike],
        threshold: float,
        pre_computed_ann_links: list[tuple] | None = None,
    ) -> int:
        """Link each new memory to similar ones — within the batch, plus the precomputed ANN
        neighbours; how many links were written."""
        return 0

    async def create_causal_links(
        self, *, conn, ops, bank_id: str, unit_ids: list[str], causal_relations_per_fact: list[list[CausalRelation]]
    ) -> int:
        """Write the new memories' ``caused_by`` edges; how many links were written."""
        return 0

    async def restore_legacy_causal_links(
        self, *, conn, ops, bank_id: str, unit_ids: list[str], causal_relations_per_fact: list[list[CausalRelation]]
    ) -> int:
        """Write an imported archive's historical causal edge types, which retain never creates;
        how many links were written."""
        return 0

    async def insert_links(self, *, conn, ops, bank_id: str, links: list[tuple]) -> None:
        """Insert precomputed link tuples (the end-of-retain semantic pass)."""

    # ------------------------------------------------------------------ consolidation writes and recall expansion
    #
    # Consolidation's writes and checks. Each runs on the caller's connection, inside the
    # consolidation batch's transaction, so every write derived from one LLM response commits or
    # rolls back together (#3876). The defaults are what a store that owns its rows does;
    # Postgres overrides them with the SQL in ``pg/consolidation.py``.

    async def lock_live_memory_ids(self, *, conn, fq_table, bank_id: str, unit_ids: list[uuid.UUID]) -> set[str]:
        """Which of ``unit_ids`` still exist, held so a concurrent delete cannot remove one before
        the caller's transaction commits.

        Postgres takes ``FOR SHARE`` on the rows. A store that keeps memories outside SQL has its
        own concurrency model, so this default is an unlocked existence check.
        """
        present = await self.get_memories(
            conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(mid) for mid in unit_ids]
        )
        return {str(m.unit_id) for m in present}

    async def lock_observation_tags(self, *, conn, fq_table, bank_id: str, observation_id: str) -> list[str] | None:
        """The observation's current tags, held so a concurrent tag edit cannot land before the
        caller's transaction commits. None when the observation is gone.

        Postgres locks the row. A store that keeps memories outside SQL has its own concurrency
        model, so this default is an unlocked read.
        """
        current = await self.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[observation_id])
        return list(current[0].tags or []) if current else None

    async def memories_changed_since(self, *, conn, fq_table, bank_id: str, read_at: dict[str, datetime]) -> list[str]:
        """Ids in ``read_at`` (id -> the ``updated_at`` it was read with) edited since (#4831).

        Postgres compares ``updated_at`` and takes ``FOR SHARE`` so no edit lands before the
        caller's transaction commits. A store whose reads carry no ``updated_at`` cannot tell,
        so this default reports none.
        """
        return []

    async def any_memory_exists(self, *, conn, fq_table, bank_id: str, unit_ids: list[uuid.UUID]) -> bool:
        """Whether any of ``unit_ids`` still exists. A cheap, non-locking preflight."""
        present = await self.get_memories(
            conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(mid) for mid in unit_ids]
        )
        return bool(present)

    async def count_observations_with_tags(self, *, conn, fq_table, bank_id: str, tags: list[str]) -> int:
        """Observations whose tags contain every one of ``tags``. Postgres runs one ``COUNT``."""
        total = 0
        page_token = ""
        for _ in range(100):
            page = await self.scan_memories(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                fact_types=["observation"],
                tags=tags or None,
                tags_match="all",
                limit=500,
                page_token=page_token,
            )
            total += len(page.memories)
            page_token = page.next_page_token
            if not page_token:
                break
        return total

    async def insert_observation(
        self,
        *,
        conn,
        ops,
        fq_table,
        bank_id: str,
        observation_id: uuid.UUID,
        text: str,
        embedding: str | None,
        source_memory_ids: list[uuid.UUID],
        tags: list[str],
        event_date: datetime | None,
        occurred_start: datetime | None,
        occurred_end: datetime | None,
        mentioned_at: datetime | None,
        created_at: datetime,
    ) -> str:
        """Write a new observation; returns its id.

        Postgres inserts a `memory_units` row (plus Oracle's `observation_sources` postings);
        this default hands the store the whole record through :meth:`upsert_observation`.
        """
        await self.upsert_observation(
            conn=conn,
            bank_id=bank_id,
            record=FactRecord(
                unit_id=str(observation_id),
                text=text,
                # `FactRecord.embedding` is declared non-optional, but consolidation has nothing to
                # write when the embedder returned nothing -- so a store reading the seam's own
                # declaration is handed None anyway. Widening the field would make every extension
                # handle it, so the mismatch is named here rather than moved onto implementers.
                embedding=cast("list[float] | str", embedding),
                fact_type="observation",
                tags=list(tags),
                proof_count=len(source_memory_ids),
                source_memory_ids=[str(s) for s in source_memory_ids],
                event_date=event_date,
                occurred_start=occurred_start,
                occurred_end=occurred_end,
                mentioned_at=mentioned_at,
                created_at=created_at,
            ),
        )
        return str(observation_id)

    async def rewrite_observation(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        observation_id: str,
        text: str,
        embedding: str | None,
        source_memory_ids: list[uuid.UUID],
        tags: list[str],
        bounds: _TemporalBounds,
        previous: MemoryFact,
    ) -> bool:
        """Replace an observation's text, embedding, sources and tags, widening its dates by
        ``bounds`` (a consolidator ``_TemporalBounds``). False when the observation is gone.

        Postgres UPDATEs the row in place. This default re-upserts the whole observation, so it
        starts from the store's current copy — or, when that has vanished, from ``previous`` (the
        pre-update recall snapshot) — and keeps the fields the update never touches (created_at).
        """
        from ..consolidation.consolidator import _as_dt, _TemporalBounds

        current = await self.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[observation_id])
        cur = current[0] if current else None
        # Widen the row the store still holds. If it has vanished, fall back to the
        # pre-update recall snapshot — ISO strings, and no event_date on that model.
        current_bounds = (
            _TemporalBounds.of(cur)
            if cur
            else _TemporalBounds(
                occurred_start=_as_dt(previous.occurred_start),
                occurred_end=_as_dt(previous.occurred_end),
                mentioned_at=_as_dt(previous.mentioned_at),
            )
        )
        merged_bounds = current_bounds.merged_with(bounds)
        await self.upsert_observation(
            conn=conn,
            bank_id=bank_id,
            record=FactRecord(
                unit_id=observation_id,
                text=text,
                # Same non-optional-field caveat as in `insert_observation` above.
                embedding=cast("list[float] | str", embedding),
                fact_type="observation",
                tags=tags,
                proof_count=len(source_memory_ids),
                source_memory_ids=[str(s) for s in source_memory_ids],
                event_date=merged_bounds.event_date,
                occurred_start=merged_bounds.occurred_start,
                occurred_end=merged_bounds.occurred_end,
                mentioned_at=merged_bounds.mentioned_at,
                created_at=cur.created_at if cur else None,
            ),
        )
        return True

    async def fold_sources_into_observation(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        observation_id: str,
        expected_text: str,
        merged_text: str,
        source_memory_ids: list[uuid.UUID],
        bounds: _TemporalBounds,
        embeddings: Embeddings,
    ) -> bool:
        """Dedup merge: fold ``source_memory_ids``, ``merged_text`` and ``bounds`` (a consolidator
        ``_TemporalBounds``) into an existing observation. False when the fold did not happen.

        Postgres UPDATEs the row gated on ``expected_text`` (its probe-time text) and keeps the
        stored embedding. This default re-upserts the observation, preserving its other fields,
        and re-embeds the merged text with ``embeddings`` because :meth:`get_memories` does not
        return the stored vector.
        """
        from ..consolidation.consolidator import _TemporalBounds
        from ..retain import embedding_utils

        current = await self.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[observation_id])
        cur = current[0] if current else None
        if cur is None:
            # Nothing to fold into; the caller still skips its CREATE, as it always has here.
            return True
        merged_sources = list(dict.fromkeys([*(cur.source_memory_ids or []), *(str(s) for s in source_memory_ids)]))
        merged_bounds = _TemporalBounds.of(cur).merged_with(bounds)
        vectors = await embedding_utils.generate_embeddings_batch(embeddings, [merged_text])
        await self.upsert_observation(
            conn=conn,
            bank_id=bank_id,
            record=FactRecord(
                unit_id=observation_id,
                text=merged_text,
                # Same non-optional-field caveat as in `insert_observation` above.
                embedding=cast("list[float] | str", str(vectors[0]) if vectors else None),
                fact_type="observation",
                tags=list(cur.tags or []),
                proof_count=len(merged_sources),
                source_memory_ids=merged_sources,
                event_date=merged_bounds.event_date,
                occurred_start=merged_bounds.occurred_start,
                occurred_end=merged_bounds.occurred_end,
                mentioned_at=merged_bounds.mentioned_at,
                created_at=cur.created_at,
            ),
        )
        return True

    async def fold_observation_into_twin(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        observation_id: str,
        observation_text: str,
        twin_id: str,
        twin_text: str,
        merged_text: str,
        embeddings: Embeddings,
    ) -> bool:
        """Dedup merge after an UPDATE: fold an observation's live sources and dates into its twin
        with ``merged_text``. True when folded; the caller then deletes the observation.

        Postgres reads the observation's sources, locks the live ones and UPDATEs the twin, each
        gated on the probe-time texts. This default reads it through the store and folds with
        :meth:`fold_sources_into_observation`.
        """
        from ..consolidation.consolidator import _TemporalBounds

        updated_obs = await self.get_memories(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[observation_id])
        updated_sources = [uuid.UUID(s) for s in updated_obs[0].source_memory_ids or []] if updated_obs else []
        if not updated_sources:
            return False
        live = await self.lock_live_memory_ids(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=updated_sources)
        live_sources = [mid for mid in updated_sources if str(mid) in live]
        if not live_sources:
            return False
        await self.fold_sources_into_observation(
            conn=conn,
            fq_table=fq_table,
            bank_id=bank_id,
            observation_id=twin_id,
            expected_text=twin_text,
            merged_text=merged_text,
            source_memory_ids=live_sources,
            bounds=_TemporalBounds.of(updated_obs[0]),
            embeddings=embeddings,
        )
        return True

    async def delete_observation(self, *, conn, fq_table, bank_id: str, observation_id: str) -> None:
        """Delete one superseded or contradicted observation, on the caller's transaction."""
        await self.delete_facts(bank_id, [observation_id])

    # -- reflect's `expand` tool: memories by id, then their chunks and documents --

    async def expand_memories(self, *, conn, fq_table, bank_id: str, unit_ids: list[uuid.UUID]) -> list:
        """``id`` (a UUID), ``text``, ``chunk_id``, ``document_id``, ``fact_type``, ``context``,
        ``tags`` for each of ``unit_ids`` that exists, as mappings. Postgres returns its rows as
        they are."""
        stored = await self.get_memories(
            conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(u) for u in unit_ids]
        )
        return [
            {
                "id": uuid.UUID(s.unit_id),
                "text": s.text,
                "chunk_id": s.chunk_id,
                "document_id": s.document_id,
                "fact_type": s.fact_type,
                "context": s.context,
                "tags": s.tags,
            }
            for s in stored
        ]

    async def expand_chunks(
        self, *, conn, fq_table, bank_id: str, memories: list, chunk_ids: list[str]
    ) -> dict[str, Any]:
        """chunk_id -> ``chunk_id``, ``chunk_text``, ``chunk_index``, ``document_id`` for the chunks
        behind ``memories`` (rows from :meth:`expand_memories`). Postgres reads ``chunk_ids``."""
        from ..chunk_ids import resolve_chunk_id_in

        # The store addresses a chunk by (document_id, index), and the index is what remains
        # once the known bank/document prefix is removed (see `engine/chunk_ids.py`). Anchored
        # on the ids in hand rather than split on "_", which a bank or document id containing
        # one would break.
        # Deduped by chunk_id: co-located memories share one chunk, and the SQL read collapses
        # them through `= ANY($1)`. Without this the store is asked for the same chunk once per
        # memory sitting in it.
        chunk_map: dict[str, Any] = {}
        refs: list[tuple[str, int]] = []
        ref_owner: list[dict] = []
        seen_chunks: set[str] = set()
        for m in memories:
            cid, did = m["chunk_id"], m["document_id"]
            if not cid or not did:
                continue
            if cid in seen_chunks:
                continue
            # `cid` / `did` are row values, typed as the union of everything the row holds; the
            # `if not cid or not did` guard above is what makes them present, not what types them.
            cid = cast(str, cid)
            did = cast(str, did)
            ref = resolve_chunk_id_in(cid, bank_id)
            if ref is None or ref.document_id != did:
                continue
            index = ref.chunk_index
            seen_chunks.add(cid)
            refs.append((did, index))
            ref_owner.append({"chunk_id": cid, "document_id": did, "chunk_index": index})
        if refs:
            texts = await self.get_chunk_texts(bank_id=bank_id, refs=refs)
            for owner, text in zip(ref_owner, texts):
                if text is None:
                    continue
                chunk_map[owner["chunk_id"]] = {**owner, "chunk_text": text}
        return chunk_map

    async def expand_documents(self, *, conn, fq_table, bank_id: str, document_ids: list[str]) -> dict[str, Any]:
        """document id -> ``id``, ``original_text``, ``retain_params`` for ``document_ids``."""
        # One read per document: the store addresses a document by id and has no batch form here.
        # The set is the documents behind the memories being expanded, which is bounded by the
        # caller's own memory_ids rather than by corpus size.
        doc_map: dict[str, Any] = {}
        for did in document_ids:
            record = await self.get_document_record(bank_id=bank_id, document_id=did, include_text=True)
            if record is None:
                continue
            # The store has no `retain_params` column; it keeps the retain params inside the
            # document record's metadata bag, under that key and serialised as JSON. So they are
            # read back out of the bag rather than reconstructed — the expand tool's
            # `_document_metadata_from_retain_params` already parses the JSON form, which is the
            # same thing Postgres hands it from JSONB.
            doc_map[did] = {
                "id": did,
                "original_text": record.get("original_text"),
                "retain_params": (record.get("metadata") or {}).get("retain_params"),
            }
        return doc_map

    # ------------------------------------------------------------------ transfer, gauges and vector indexes
    #
    # Bank transfer (export / import), the consolidation gauges and the per-bank vector-index
    # counts. The transfer shapes (`_LoadedExport`, `TransferObservation`, `FactLifecycle`, …) live
    # in `engine.transfer`, which imports this module, so they are imported for typing only.
    # The defaults are the store-owned behaviour, expressed through the interface; the transfer
    # assembly they reuse lives next to the archive format, so it is imported at call time.

    async def iter_transfer_documents(
        self,
        *,
        backend,
        fq_table,
        bank_id: str,
        document_ids: list[str] | None,
        include_lifecycle: bool,
        batch_size: int,
    ) -> AsyncIterator[_LoadedExport]:
        """Async-iterate the bank's documents (or ``document_ids``) as ``_LoadedExport`` batches, for export.

        ``backend`` is a pool or one connection. Postgres reads ``batch_size`` documents per batch,
        each on a connection it returns before the batch is yielded. This default takes no
        connection: it yields :meth:`load_transfer_documents` once.
        """
        yield await self.load_transfer_documents(
            conn=None,
            fq_table=fq_table,
            bank_id=bank_id,
            document_ids=document_ids,
            include_lifecycle=include_lifecycle,
        )

    async def load_transfer_documents(
        self, *, conn, fq_table, bank_id: str, document_ids: list[str] | None, include_lifecycle: bool
    ) -> _LoadedExport:
        """The bank's documents (or ``document_ids``) with their chunks and facts, as one ``_LoadedExport``.

        Postgres reads `documents`, `chunks`, `memory_units`, `unit_entities` and `memory_links`
        on ``conn``. This default assembles the same archive through :meth:`list_documents`,
        :meth:`scan_memories` and the edges each memory carries.
        """
        from ..transfer.export import _load_documents_from_store

        return await _load_documents_from_store(self, bank_id, document_ids, include_lifecycle=include_lifecycle)

    async def load_transfer_observations(
        self, *, conn, fq_table, bank_id: str, unit_index: dict[Any, _UnitLocation]
    ) -> list[TransferObservation]:
        """The bank's observations whose every source is in ``unit_index``, as ``TransferObservation``s.

        ``unit_index`` maps an exported fact's unit id to its (document, ordinal) location.
        Postgres reads `memory_units` on ``conn``; this default scans the store's observations.
        """
        from ..transfer.export import _load_observations_from_store

        return await _load_observations_from_store(self, bank_id, unit_index)

    async def dump_entity_maintenance_queue(self, *, conn, fq_table, bank_id: str) -> list[dict]:
        """The bank's entity-maintenance queue for export, each entity named rather than numbered.

        Postgres joins the queue to `entities` on ``conn``. Empty here: a store that keeps its own
        entity registry queues no entity maintenance in Postgres (see
        :meth:`enqueue_entity_prune_candidates`).
        """
        return []

    async def dump_archived_memories(self, *, conn, fq_table, bank_id: str) -> list[dict]:
        """The curation archive (invalidated, still revertable memories) as portable export rows.

        One dict per archived memory, keyed like an `invalidated_memory_units` row, except that
        ``entity_ids`` becomes ``entity_names`` (canonical names), ``chunk_id`` becomes
        ``chunk_index`` (the id embeds the bank), and the surrogate ``id`` is dropped. Postgres
        reads the archive table on ``conn``. No default: the interface has no way to list an
        archive, so a store that owns its memories must answer this itself.
        """
        raise NotImplementedError("this store cannot list its archived memories")

    async def restore_archived_memories(
        self,
        *,
        conn,
        fq_table,
        bank_id: str,
        rows: list[dict],
        unit_id_map: dict[str, str],
        document_id_map: dict[str, str],
        bank_rows_json_encoding: str,
    ) -> int:
        """Write :meth:`dump_archived_memories` rows into this bank's archive; returns how many.

        ``unit_id_map`` / ``document_id_map`` map the source's unit and document ids to the ones the
        import wrote (a causal-link snapshot whose endpoint did not come back is dropped). Postgres
        re-resolves the entity names and inserts on ``conn``. No default, like
        :meth:`dump_archived_memories`.
        """
        raise NotImplementedError("this store cannot restore archived memories")

    async def resolve_entity_ids_by_name(self, *, conn, fq_table, bank_id: str, names: set[str]) -> dict[str, Any]:
        """Canonical entity name -> this bank's entity id, for rows an import restores by name.

        Postgres reads `entities` on ``conn``. Empty here: the only caller restores Postgres
        entity-maintenance queue rows, which a store with its own registry does not have.
        """
        return {}

    async def transfer_document_exists(self, *, backend, fq_table, bank_id: str, document_id: str) -> bool:
        """Whether the bank already holds this document — the import's conflict check.

        Postgres reads `documents` on a connection from ``backend``; this default asks
        :meth:`get_document_record` and takes no connection.
        """
        return await self.get_document_record(bank_id=bank_id, document_id=document_id) is not None

    async def restore_document_created_at(
        self, *, conn, fq_table, bank_id: str, document_id: str, created_at: datetime
    ) -> None:
        """Stamp an imported document with the creation time its archive carries.

        Postgres updates the `documents` row on ``conn``. No default: the document record has no
        field a store could be asked to rewrite, so an import into such a store keeps the import
        time (the importer logs it).
        """
        raise NotImplementedError("this store cannot restore a document's creation time")

    async def restore_fact_lifecycle(self, *, conn, fq_table, bank_id: str, rows: list[FactLifecycle]) -> None:
        """Apply imported facts' source consolidation state (``FactLifecycle`` rows) to their new units.

        Postgres rewrites ``created_at`` / ``consolidated_at`` / ``consolidation_failed_at`` on
        ``conn``, inside the import's transaction. This default stamps the markers through
        :meth:`mark_consolidated`, one call per distinct value. ``created_at`` is not restored here:
        nothing on the interface rewrites it, so such a fact reads as created at import time.
        """
        groups: dict[tuple[datetime, bool], list[str]] = {}
        for row in rows:
            if row.consolidated_at is not None:
                groups.setdefault((row.consolidated_at, False), []).append(row.unit_id)
            elif row.consolidation_failed_at is not None:
                groups.setdefault((row.consolidation_failed_at, True), []).append(row.unit_id)
        for (when, failed), unit_ids in groups.items():
            await self.mark_consolidated(
                conn=None, fq_table=fq_table, bank_id=bank_id, unit_ids=unit_ids, when=when, failed=failed
            )

    async def import_transfer_observations(
        self,
        *,
        backend,
        ops,
        fq_table,
        bank_id: str,
        resolved: list[tuple[TransferObservation, list[str]]],
        processed: list[ProcessedFact],
        outcome: _ObservationOutcome,
    ) -> _ObservationOutcome:
        """Write imported observations whose sources are all live; returns ``outcome``, updated.

        ``resolved`` pairs each ``TransferObservation`` with its sources' new unit ids and
        ``processed`` carries each one's embedded ``ProcessedFact``; an observation with a missing
        source is skipped and counted in ``outcome.skipped``. Postgres re-checks liveness, inserts
        and links in one transaction on a connection from ``backend``. This default checks
        liveness with :meth:`get_memories`, writes each observation whole with
        :meth:`upsert_observation`, and stamps the sources' consolidated marker when they have none.
        """
        all_sources = list(dict.fromkeys(s for _obs, sources in resolved for s in sources))
        # Read once and reuse below: ``upsert_observation`` writes the observation, never its
        # sources, so their ``consolidated_at`` is the same after the loop as it is here.
        source_rows = await self.get_memories(conn=None, fq_table=fq_table, bank_id=bank_id, unit_ids=all_sources)
        live = {m.unit_id for m in source_rows}
        marked: set[str] = set()
        for (obs, sources), fact in zip(resolved, processed):
            missing = [s for s in sources if s not in live]
            if missing:
                logger.warning(
                    "[transfer] Skipping observation for bank %s: %d of %d source units are missing (%s)",
                    bank_id,
                    len(missing),
                    len(sources),
                    ", ".join(sorted(missing)),
                )
                outcome.skipped += 1
                continue
            observation_id = str(uuid.uuid4())
            await self.upsert_observation(
                conn=None,
                bank_id=bank_id,
                record=FactRecord(
                    unit_id=observation_id,
                    text=obs.text,
                    embedding=list(fact.embedding),
                    fact_type="observation",
                    tags=list(obs.tags),
                    proof_count=obs.proof_count,
                    observation_scopes=obs.observation_scopes,
                    source_memory_ids=list(sources),
                    event_date=obs.event_date,
                    occurred_start=obs.occurred_start,
                    occurred_end=obs.occurred_end,
                    mentioned_at=fact.mentioned_at,
                    created_at=obs.created_at,
                ),
            )
            marked.update(sources)
            outcome.imported += 1
            if obs.source_id is not None:
                outcome.remapped_unit_ids[obs.source_id] = observation_id

        # Same rule as the SQL path's COALESCE: a source whose own consolidated marker came from the
        # archive keeps it; only the ones with none are stamped now, so the consolidator skips them.
        unmarked = [m.unit_id for m in source_rows if m.unit_id in marked and m.consolidated_at is None]
        if unmarked:
            await self.mark_consolidated(
                conn=None, fq_table=fq_table, bank_id=bank_id, unit_ids=unmarked, when=datetime.now(timezone.utc)
            )
        return outcome

    async def count_consolidation_backlog(
        self, *, conn, schema: str, per_bank: bool, bank_ids: Callable[[], Awaitable[list[str]]]
    ) -> dict[str | None, int]:
        """Source memories (experience/world) still queued for consolidation, for the backlog gauge.

        Keyed by bank id when ``per_bank``, else a single ``None`` key for the whole ``schema``.
        Permanently failed memories are excluded (they are :meth:`count_consolidation_failed`).
        Postgres counts every bank of ``schema`` in one query on ``conn``. This default asks
        :meth:`count_unconsolidated` per bank, for the banks the async ``bank_ids()`` lists.
        """
        counts: dict[str | None, int] = {} if per_bank else {None: 0}
        for bank_id in await bank_ids():
            # ponytail: capped per bank, so a huge store-owned backlog reads as the cap. Enough
            # for "backlog > 0 for N minutes"; a store with a real count overrides this method.
            n = await self.count_unconsolidated(
                conn=None,
                fq_table=None,
                bank_id=bank_id,
                fact_types=["experience", "world"],
                scopes=[None],
                limit=10_000,
            )
            key = bank_id if per_bank else None
            counts[key] = counts.get(key, 0) + n
        return counts

    async def count_consolidation_failed(
        self, *, conn, schema: str, per_bank: bool, bank_ids: Callable[[], Awaitable[list[str]]]
    ) -> dict[str | None, int]:
        """Source memories whose consolidation permanently failed, keyed like
        :meth:`count_consolidation_backlog`. This default counts :meth:`find_failed_consolidation`
        per bank."""
        counts: dict[str | None, int] = {} if per_bank else {None: 0}
        for bank_id in await bank_ids():
            n = len(await self.find_failed_consolidation(conn=None, fq_table=None, bank_id=bank_id))
            key = bank_id if per_bank else None
            counts[key] = counts.get(key, 0) + n
        return counts


__all__ = [
    "CONSOLIDATED_NO",
    "CONSOLIDATED_YES",
    "META_ATTACHMENT_IDS",
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
    "BankContentCounts",
    "CausalEdgeRecord",
    "DeletePredicate",
    "EntityPrunePassResult",
    "FactRecord",
    "GRAPH_SEED_LIMIT",
    "MemoriesExtension",
    "MemoryLocation",
    "MemoryPatch",
    "RelinkPassResult",
    "ScanPage",
    "SemanticBm25Result",
    "StoredMemory",
    "TypedMemoryScope",
    "build_fact_records",
    "build_text_signals",
    "source_key",
]
