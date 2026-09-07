"""
Fact storage for retain pipeline.

Handles insertion of facts into the database.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from ...config import _get_raw_config
from ..memory_engine import fq_table
from ..metadata_utils import drop_null_values
from .bank_utils import create_bank_row_on_conn
from .fact_extraction import _sanitize_text
from .types import ProcessedFact

logger = logging.getLogger(__name__)

#: Page size for walking a replaced document's outgoing memories. Large enough
#: that one page covers any ordinary document, small enough that a pathological
#: one does not arrive as a single result set.
_OUTGOING_PAGE = 500


async def get_document_content(
    conn,
    bank_id: str,
    document_id: str,
) -> str | None:
    """Fetch the original_text of an existing document.

    Returns None if the document does not exist.
    """
    row = await conn.fetchval(
        f"SELECT original_text FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
        document_id,
        bank_id,
    )
    return row


async def count_document_memory_units(
    conn,
    bank_id: str,
    document_id: str,
) -> int:
    """Count the memory units a document currently owns.

    This is the number reported as ``memory_unit_count`` by the Documents API and
    by the ``retain.completed`` webhook. Zero means the document is stored but
    unreachable through recall/reflect — only memory units carry embeddings, so a
    document without them cannot be retrieved until it is reprocessed (#3040).
    """
    count = await conn.fetchval(
        f"SELECT COUNT(*) FROM {fq_table('memory_units')} WHERE bank_id = $1 AND document_id = $2",
        bank_id,
        document_id,
    )
    return int(count or 0)


async def insert_facts_batch(
    conn,
    bank_id: str,
    facts: list[ProcessedFact],
    document_id: str | None = None,
    ops=None,
    defer_index: bool = False,
) -> list[str]:
    """
    Store facts and return their unit ids, in order.

    Args:
        conn: Database connection
        bank_id: Bank identifier
        facts: List of ProcessedFact objects to insert
        document_id: Optional document ID to associate with facts
        defer_index: Ask for ids without the write. The retain orchestrator needs
            this because it can only supply entity ids and causal edges after
            Phase-1 placeholders have been remapped onto real unit ids; it then
            calls `index_facts` with the complete picture. The Postgres store,
            whose write *is* the insert that mints the ids, ignores it.

    Returns:
        List of unit IDs (UUIDs as strings) for the inserted facts
    """
    if not facts:
        return []

    from ..memories import get_memories

    return await get_memories().insert_facts(
        conn=conn,
        ops=ops,
        bank_id=bank_id,
        facts=facts,
        document_id=document_id,
        defer_index=defer_index,
    )


async def index_facts(
    bank_id: str,
    unit_ids: list[str],
    facts: list[ProcessedFact],
    document_id: str | None = None,
    unit_entity_ids: dict[str, list[str]] | None = None,
) -> None:
    """Complete a deferred `insert_facts_batch`, now that the edges are known.

    ``unit_entity_ids`` is the unit→entity posting and each fact's causal
    relations are its edges; both travel with the memory for a store that owns
    them. A no-op for the Postgres store, which wrote all of it already.

    This is the single, entity-bearing write of the store-owned retain path: the facts
    land ONCE here, complete, rather than write-then-reattach.
    """
    from ..memories import get_memories

    await get_memories().index_facts(bank_id, unit_ids, facts, document_id, unit_entity_ids)


async def ensure_bank_exists(conn, bank_id: str, *, ops) -> None:
    """
    Ensure bank exists in the database.

    Creates bank with default values if it doesn't exist. Retain's entry point
    into the lazy bank-create; the row and its per-bank vector indexes are
    written by ``bank_utils.create_bank_row_on_conn`` so that a bank born here
    is byte-for-byte the same as one born through ``get_or_create_bank_profile``.

    Args:
        conn: Database connection
        bank_id: Bank identifier
        ops: Backend ``DataAccessOps``, needed for the per-bank vector index DDL
            a fresh bank gets while the size threshold is off. Required rather
            than defaulting to None, because it is dereferenced only in that
            branch — a caller that omitted it worked until the threshold was off.
    """
    await create_bank_row_on_conn(conn, bank_id, ops=ops)


async def delete_stale_observations_for_memories(
    conn,
    bank_id: str,
    fact_ids: "list[str | uuid.UUID]",
    ops=None,
) -> int:
    """Delete observations whose source memories are about to be removed.

    Mirrors the cleanup performed by ``MemoryEngine.delete_document`` so that
    every code path that removes memories also removes the observations derived
    from them. Without this, ingesting a fresh version of a document via the
    retain pipeline (which does a full-replace ``DELETE FROM documents``
    cascade) used to leave orphan observations pointing at memory IDs that no
    longer existed.

    For each observation referencing any of ``fact_ids``:
    1. Delete the observation (its text is stale once even one source memory
       disappears).
    2. Reset the consolidated marker on the surviving source memories so they
       get re-consolidated under fresh observations on the next run.

    Must be called within an active transaction, before the source memories are
    deleted.

    Returns:
        Number of observations deleted.
    """
    if not fact_ids:
        return 0

    from ..memories import get_memories

    return await get_memories().delete_stale_observations(
        conn=conn,
        ops=ops,
        fq_table=fq_table,
        bank_id=bank_id,
        fact_ids=fact_ids,
    )


async def handle_document_tracking(
    conn,
    bank_id: str,
    document_id: str,
    combined_content: str,
    is_first_batch: bool,
    retain_params: dict | None = None,
    document_tags: list[str] | None = None,
    ops=None,
    store_document_text: bool | None = None,
    attachment_filenames: dict[str, str] | None = None,
) -> None:
    """
    Handle document tracking in the database (full-replace mode).

    Deletes the existing document (cascading to all units and links) on the
    first batch, then inserts the new document record.

    Args:
        conn: Database connection
        bank_id: Bank identifier
        document_id: Document identifier
        combined_content: Combined content text from all content items
        is_first_batch: Whether this is the first batch (for chunked operations)
        retain_params: Optional parameters passed during retain (context, event_date, etc.)
        document_tags: Optional list of tags to associate with the document
        attachment_filenames: Short id -> the name the caller gave that attachment in
            this document. Recorded on the document edge rather than the blob, since
            a filename describes the reference and the same bytes can carry a
            different name in another document.
        ops: Backend-specific DataAccessOps. Required by the inner
            ``delete_stale_observations_for_memories`` call to choose the PG
            (native array) vs Oracle (junction table) read path. Defaults to
            None so older callers don't break, but the PG branch is only
            taken when ops is non-None — pass ``pool.ops`` from the caller.
    """
    import hashlib

    # Sanitize and calculate content hash
    combined_content = _sanitize_text(combined_content) or ""
    content_hash = hashlib.sha256(combined_content.encode()).hexdigest()

    # Delete old document first (cascades to units and links).
    # Only delete on the first batch to avoid deleting data we just inserted.
    # Before the cascade, fan out to delete observations derived from the
    # outgoing memory_units — otherwise the FK ON DELETE CASCADE removes the
    # source memory_units but leaves observation rows pointing at IDs that
    # no longer exist (consolidated_at on co-source memories also stays
    # frozen). Same cleanup the explicit ``delete_document`` API performs.
    preserved_created_at = None
    if is_first_batch:
        from ..memories import get_memories

        store = get_memories()
        # Which memories the outgoing version left behind. Asked of the store
        # rather than queried here, because it is the store that knows where they
        # are. Paged to exhaustion: every one of them is about to be deleted, and
        # a document whose facts overflow one page must not keep half of them.
        existing_unit_ids: list[str] = []
        page_token = ""
        while True:
            page = await store.scan_memories(
                conn=conn,
                fq_table=fq_table,
                bank_id=bank_id,
                fact_types=["experience", "world"],
                document_id=document_id,
                limit=_OUTGOING_PAGE,
                page_token=page_token,
            )
            existing_unit_ids.extend(m.unit_id for m in page.memories)
            page_token = page.next_page_token
            if not page_token:
                break
        if existing_unit_ids:
            invalidated = await delete_stale_observations_for_memories(conn, bank_id, existing_unit_ids, ops=ops)
            if invalidated:
                logger.info(
                    f"[RETAIN] Document {document_id} re-ingested: invalidated "
                    f"{invalidated} observation(s) derived from {len(existing_unit_ids)} outgoing memory_units"
                )
            else:
                # Logged even at zero: "the sweep matched nothing" and "the sweep never ran"
                # are the two candidates whenever orphan observations are reported, and
                # without this line they look identical from the outside (issue #3294).
                logger.debug(
                    f"[RETAIN] Document {document_id} re-ingested: no observations derived from "
                    f"{len(existing_unit_ids)} outgoing memory_units"
                )
            # Capture link-recompute victims BEFORE the cascade. Same staleness
            # applies on upsert as on explicit delete: surviving units in OTHER
            # documents that linked to these doomed units are about to lose
            # those links. ``ops`` may be None for older callers that haven't
            # been wired up — skip enqueue in that case rather than crash.
            if ops is not None:
                from ..graph_maintenance import enqueue_entity_prune_candidates, enqueue_relink_victims

                doomed_ids = [str(uid) for uid in existing_unit_ids]
                await enqueue_relink_victims(conn, bank_id, doomed_ids)
                # Same timing, different target: the entities these units are
                # about to stop referencing may have no other posting. The
                # re-ingest re-resolves entities from scratch, so the ones the
                # new facts don't name again are orphans the moment this
                # cascade lands.
                await enqueue_entity_prune_candidates(conn, bank_id, doomed_ids)

        # Explicitly delete memory_units by document_id BEFORE deleting the
        # document row. The CASCADE from documents→chunks→memory_units only
        # catches units that have a non-NULL chunk_id FK. Units with chunk_id=NULL
        # (e.g. from partial writes or edge cases) would survive the cascade.
        # This explicit delete ensures complete cleanup.
        await store.delete_document(conn=conn, fq_table=fq_table, bank_id=bank_id, document_id=document_id)
        # Capture created_at before deletion so re-ingestion preserves it.
        preserved_created_at = await conn.fetchval(
            f"DELETE FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2 RETURNING created_at",
            document_id,
            bank_id,
        )

    # Insert document (or update if exists from concurrent operations)
    await _upsert_document_row(
        conn,
        bank_id,
        document_id,
        combined_content,
        content_hash,
        retain_params,
        document_tags,
        preserved_created_at=preserved_created_at,
        store_document_text=store_document_text,
        attachment_filenames=attachment_filenames,
    )


async def upsert_document_metadata(
    conn,
    bank_id: str,
    document_id: str,
    combined_content: str,
    retain_params: dict | None = None,
    document_tags: list[str] | None = None,
    store_document_text: bool | None = None,
    attachment_filenames: dict[str, str] | None = None,
) -> None:
    """
    Update document metadata without deleting existing facts/chunks.

    Used by delta retain: the document row is upserted but chunks and
    memory_units are managed separately at the chunk level.
    """
    import hashlib

    combined_content = _sanitize_text(combined_content) or ""
    content_hash = hashlib.sha256(combined_content.encode()).hexdigest()

    await _upsert_document_row(
        conn,
        bank_id,
        document_id,
        combined_content,
        content_hash,
        retain_params,
        document_tags,
        store_document_text=store_document_text,
        attachment_filenames=attachment_filenames,
    )


async def _upsert_document_row(
    conn,
    bank_id: str,
    document_id: str,
    combined_content: str,
    content_hash: str,
    retain_params: dict | None = None,
    document_tags: list[str] | None = None,
    preserved_created_at: datetime | None = None,
    store_document_text: bool | None = None,
    attachment_filenames: dict[str, str] | None = None,
) -> None:
    """Insert or update a document row.

    When ``preserved_created_at`` is provided, it is used for ``created_at`` on
    INSERT so that re-ingesting a document (which deletes + inserts the row)
    keeps the original creation timestamp. ``updated_at`` is always set to
    ``NOW()`` on both INSERT and the ON CONFLICT UPDATE branch.

    When ``store_document_text`` is disabled, the raw source text
    is dropped and ``original_text`` is stored as NULL. The ``content_hash`` is
    still computed from the real content so delta-retain dedup is unaffected.
    ``store_document_text`` defaults to the server-level config when ``None``;
    the retain path passes the per-bank resolved value.
    """
    # Fallback to the raw global default (not get_config(), which guards
    # bank-configurable fields); the retain path always passes the resolved value.
    store_text = store_document_text if store_document_text is not None else _get_raw_config().store_document_text
    original_text = combined_content if store_text else None
    # A store that owns a dedicated document store keeps the extracted text there, so the
    # SQL documents row holds only its metadata (id, content_hash, tags) with original_text NULL —
    # the bulky body is written to the store up front (orchestrator._store_document_bodies).
    from ..memories import get_memories

    if get_memories().store_owned_for(bank_id):
        original_text = None
    await conn.execute(
        f"""
        INSERT INTO {fq_table("documents")} (id, bank_id, original_text, content_hash, retain_params, tags, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, COALESCE($7, NOW()), NOW())
        ON CONFLICT (id, bank_id) DO UPDATE
        SET original_text = EXCLUDED.original_text,
            content_hash = EXCLUDED.content_hash,
            retain_params = EXCLUDED.retain_params,
            tags = EXCLUDED.tags,
            updated_at = NOW()
        """,
        document_id,
        bank_id,
        original_text,
        content_hash,
        json.dumps(retain_params) if retain_params else None,
        document_tags or [],
        preserved_created_at,
    )
    await sync_document_attachments(conn, bank_id, document_id, combined_content, attachment_filenames)


async def sync_document_attachments(
    conn,
    bank_id: str,
    document_id: str,
    text: str,
    filenames: dict[str, str] | None = None,
) -> None:
    """Record which attachments this document references, derived from its text.

    Called on every document write, from the one place every write funnels
    through. The edge is *derived*, never supplied: the canonical text is the
    source of truth for which attachments a document carries, so a re-ingest, an
    append or a delta re-extraction cannot leave this table disagreeing with it.
    That is why the retain pipeline itself knows nothing about attachments.

    The rows exist for lifecycle and for the filename — a chunk's own text is
    what recall resolves. They die with the document via the composite FK, so
    after a delete a blob that no row in the bank still references can be
    reclaimed.

    ``filenames`` maps short id to the name the caller gave that attachment *in
    this document*. It lives here rather than on the blob because a filename
    describes the reference, not the bytes: the same PDF can be "policy-v1.pdf"
    in one document and "escalation-runbook.pdf" in another, and the blob row is
    written once for the first of them. Absent (an append, a delta re-extraction,
    a reprocess replaying stored text) the existing names are carried over, so a
    write that does not restate them does not erase them.
    """
    from .attachment_content import iter_placeholder_ids

    referenced = sorted(set(iter_placeholder_ids(text or "")))

    # Names already recorded for this document, so a write that does not restate
    # them (append, delta re-extraction, reprocess from stored text) keeps them
    # rather than blanking the column on the delete below.
    existing = {
        row["short_id"]: row["filename"]
        for row in await conn.fetch(
            f"""
            SELECT ba.short_id, da.filename
            FROM {fq_table("document_attachments")} da
            JOIN {fq_table("attachments")} ba
              ON ba.bank_id = da.bank_id AND ba.attachment_hash = da.attachment_hash
            WHERE da.bank_id = $1 AND da.document_id = $2 AND da.filename IS NOT NULL
            """,
            bank_id,
            document_id,
        )
    }
    resolved = {**existing, **(filenames or {})}

    # Delete-then-insert rather than a diff: the set is tiny, and this way a
    # document that lost an attachment on re-ingest cannot keep a stale row.
    await conn.execute(
        f"DELETE FROM {fq_table('document_attachments')} WHERE bank_id = $1 AND document_id = $2",
        bank_id,
        document_id,
    )
    if not referenced:
        return
    # The placeholder carries the short id; the row carries the full digest, so
    # resolve through attachments rather than storing a second key shape. Done as
    # a lookup and then an executemany rather than one INSERT..SELECT joined
    # against `unnest`: pairing two arrays that way is Postgres-only, and the
    # same statement has to run on Oracle. (`ON CONFLICT DO NOTHING` is fine —
    # the Oracle layer rewrites it.)
    pairs = await conn.fetch(
        f"SELECT attachment_hash, short_id FROM {fq_table('attachments')} "
        f"WHERE bank_id = $1 AND short_id = ANY($2::text[])",
        bank_id,
        referenced,
    )
    if not pairs:
        return
    await conn.executemany(
        f"""
        INSERT INTO {fq_table("document_attachments")} (bank_id, document_id, attachment_hash, filename)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (bank_id, document_id, attachment_hash) DO NOTHING
        """,
        [(bank_id, document_id, row["attachment_hash"], resolved.get(row["short_id"])) for row in pairs],
    )


def _normalize_scopes(value: list | str | None) -> list | str | None:
    """Compare-ready form of an ``observation_scopes`` value.

    The column is JSONB, so a read can hand it back as a JSON string
    (``'"per_tag"'`` / ``'[["a"]]'``) while the retain call site supplies the
    parsed value. Normalizing both through ``json.loads`` keeps the comparison
    from reporting a change on the encoding alone. A bare word that is not JSON
    ("per_tag" written unquoted) is already in compare-ready form.
    """
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


async def update_memory_units_metadata_and_tags(
    conn,
    bank_id: str,
    document_id: str,
    tags: list[str],
    metadata: dict[str, Any],
    *,
    observation_scopes: list | str | None = None,
    label_tag_keys: set[str] | None = None,
    ops=None,
) -> int:
    """Update document-level attributes on existing memory units.

    Delta retain preserves unchanged chunks and their facts. Propagate the
    current document tags, metadata and observation scoping so its optimized
    result matches a full replace.

    ``tags`` is the DOCUMENT's tag set and replaces what a survivor carries of it
    outright — the caller owns those, so a re-retain that drops one drops it here too.
    But the column also holds *label tags*: the projection of the ``entity_labels``
    groups flagged ``tag: true``, mirrored out of each unit's own entities at extraction
    (``_inject_label_tags``). Those are derived per fact, not per document, and a unit
    only acquires them by being extracted — so a survivor, which by definition was not
    re-extracted, keeps the ones it has. ``label_tag_keys`` names the group keys that
    projection claims; a ``key:value`` tag under one of them is carried forward rather
    than overwritten (issue #4068).

    Without that carry-forward the same document ended a delta retain holding units
    labelled ``category:durable`` beside units that were not, differing only in whether
    their chunk happened to change — and a tags filter returned an arbitrary slice of it.
    Passing no ``label_tag_keys`` (or a bank with no such group) keeps the plain
    overwrite, which is what a document with no label projection wants.

    ``metadata`` arrives as the raw retain_params bag (the document row keeps
    the caller's input verbatim), so null-valued keys are dropped here — the
    same normalization ``RetainContent`` applies to freshly extracted facts
    (issue #3209). Without it a re-retain would leave surviving units carrying
    nulls while the units around them do not.

    Relabelling is not only a labelling change: consolidation scopes a memory by
    its tag set and routes it by ``observation_scopes``, so an observation built
    over these facts under the OLD scoping is no longer valid the moment either
    one changes. ``MemoryEngine.update_document`` (the tags PATCH) has always run
    that cascade; a re-retain carrying narrower tags took this path instead and
    ran none of it, leaving observations — and the mental models citing them —
    visible to a tag no live fact carries any more. So when the scoping of a
    surviving unit actually changes, delete the observations standing on it and
    requeue it (and their co-sources, which ``delete_stale_observations`` does)
    for re-consolidation under the new scoping. Units whose scoping is unchanged
    keep their observations: an ordinary delta edit must not re-consolidate the
    whole document.

    Returns:
        Number of memory units updated.
    """
    from ..memories import MemoryPatch, get_memories
    from ..memories.base import META_METADATA_JSON, META_OBSERVATION_SCOPES
    from .entity_labels import split_label_tags

    def _tags_for(existing: list[str] | None) -> list[str]:
        """The document tags, plus the label projection this unit already carries."""
        merged = list(tags or [])
        seen = set(merged)
        for t in split_label_tags(existing, label_tag_keys):
            if t not in seen:
                seen.add(t)
                merged.append(t)
        return merged

    store = get_memories()
    if store.store_owned_for(bank_id):
        # A store that keeps memories outside SQL: page the document's memories and patch each
        # one's tags and metadata through the store — the UPDATE below is a no-op on its empty
        # memory_units.
        page = await store.scan_memories(
            conn=conn, fq_table=fq_table, bank_id=bank_id, document_id=document_id, limit=1_000_000
        )
        # Metadata too, not just tags: the SQL branch below sets both, and a survivor left carrying
        # the PREVIOUS retain's metadata is exactly what this function exists to prevent — measured
        # on an append, older units still read {"source": "email"} after a retain carrying
        # {"source": "crm"}.
        #
        # Written under META_METADATA_JSON as one JSON value, which is where the bag contract puts a
        # memory's user metadata and what every read reconstructs it from. A flat {"source": "crm"}
        # would merge a stray top-level key into the record's own bag instead: applied, reported as
        # applied, and invisible to every reader. The bag's other keys are internal (context,
        # chunk_id, consolidation_failed_at, …) and a patch that carried user keys loose among them
        # could not be told apart from one setting an internal field.
        #
        # Set unconditionally, mirroring `SET metadata = $4`: a document whose metadata was cleared
        # must clear on its survivors too, which an absent key would not do.
        new_tags_by_unit = {m.unit_id: _tags_for(m.tags) for m in page.memories}
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
            await store.update_memories(bank_id, patches)
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
        if rescoped:
            await delete_stale_observations_for_memories(conn, bank_id, rescoped, ops=ops)
            # The rescoped units survive, so `delete_stale_observations` (which requeues only
            # an observation's OTHER sources, the ones not being deleted) does not reach them.
            await store.mark_consolidated(conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=rescoped, when=None)
        return len(patches)

    # Read the scoping the survivors carry BEFORE overwriting it — the cascade below has to
    # know which units actually moved, and after the UPDATE that is no longer answerable.
    prior = await conn.fetch(
        f"""
        SELECT id, fact_type, tags, observation_scopes
        FROM {fq_table("memory_units")}
        WHERE bank_id = $1 AND document_id = $2
        """,
        bank_id,
        document_id,
    )
    new_tags_by_id = {row["id"]: _tags_for(row["tags"]) for row in prior}
    # See the store-owned branch: the comparison is against what the unit ends with.
    rescoped_ids = [
        row["id"]
        for row in prior
        if row["fact_type"] in ("experience", "world")
        and (
            set(row["tags"] or []) != set(new_tags_by_id[row["id"]])
            or _normalize_scopes(row["observation_scopes"]) != _normalize_scopes(observation_scopes)
        )
    ]

    result = await conn.execute(
        f"""
        UPDATE {fq_table("memory_units")}
        SET tags = $3, metadata = $4, observation_scopes = $5, updated_at = NOW()
        WHERE bank_id = $1 AND document_id = $2
        """,
        bank_id,
        document_id,
        tags or [],
        json.dumps(drop_null_values(metadata)),
        json.dumps(observation_scopes) if observation_scopes is not None else None,
    )

    # Restore each survivor's label projection over the blanket write above. Done as a
    # follow-up rather than folded into that statement so a row inserted concurrently
    # still gets the document tags and metadata exactly as before — this pass only
    # touches ids that were read, and a document carrying no label tags issues nothing.
    # Grouped by the FINAL array `_tags_for` computed rather than by the projection
    # alone, so the value written here is the one it already deduped — a unit whose
    # label tag is also a document tag must not come back carrying it twice.
    by_final: dict[tuple[str, ...], list] = {}
    for row in prior:
        final = new_tags_by_id[row["id"]]
        if final != list(tags or []):
            by_final.setdefault(tuple(final), []).append(row["id"])
    for final, ids in by_final.items():
        await conn.execute(
            f"""
            UPDATE {fq_table("memory_units")}
            SET tags = $3, updated_at = NOW()
            WHERE bank_id = $1 AND document_id = $2 AND id = ANY($4::uuid[])
            """,
            bank_id,
            document_id,
            list(final),
            ids,
        )

    if rescoped_ids:
        await delete_stale_observations_for_memories(conn, bank_id, rescoped_ids, ops=ops)
        # The rescoped units survive this write, so the requeue inside
        # `delete_stale_observations` — which only covers an observation's OTHER sources,
        # the ones being deleted — skips them. Reset them here or they stay marked
        # consolidated against an observation that no longer exists and are never selected
        # into a batch again. Through the store's `mark_consolidated` rather than a raw
        # UPDATE, the same call the store-owned branch and `update_document` make.
        await store.mark_consolidated(
            conn=conn, fq_table=fq_table, bank_id=bank_id, unit_ids=[str(uid) for uid in rescoped_ids], when=None
        )

    # result is a status string like "UPDATE 5"
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0
