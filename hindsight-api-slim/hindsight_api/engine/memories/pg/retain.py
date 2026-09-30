"""The retain path's statements on `documents`, `chunks` and `memory_units`.

The retain orchestrator used to issue these inline, each behind an
``if store_owned … else <SQL>`` branch or on a Postgres-only path. They moved here
verbatim: same statements, same order, same parameters, on the connection the caller
already holds, so the transaction boundaries and round trips of a retain are unchanged.

Four functions take the *pool* rather than a connection (``document_updated_at``,
``recovery_chunk_hashes``, ``load_document_chunks``, ``current_document_hash``): their
caller used to acquire a connection only for them, and a store that owns its memories must
not acquire one at all.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import Any

from ..base import DocumentBase, DocumentChunkState, ExistingChunk, RelabelResult


async def read_document_base(
    *, conn, fq_table: Callable[[str], str], bank_id: str, document_id: str
) -> DocumentBase | None:
    """The stored body and content_hash an append concatenates onto; ``None`` if no row."""
    base_row = await conn.fetchrow(
        f"SELECT original_text, content_hash FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
        document_id,
        bank_id,
    )
    if base_row is None:
        return None
    return DocumentBase(original_text=base_row["original_text"], content_hash=base_row["content_hash"])


async def document_updated_at(
    *, pool, fq_table: Callable[[str], str], bank_id: str, document_id: str
) -> datetime | None:
    """The document row's ``updated_at``, for the stale-request check; ``None`` if no row."""
    from ...db_utils import acquire_with_retry

    async with acquire_with_retry(pool) as conn:
        doc_row = await conn.fetchrow(
            f"SELECT updated_at FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
            document_id,
            bank_id,
        )
    return doc_row["updated_at"] if doc_row else None


async def load_existing_chunks(
    *, conn, fq_table: Callable[[str], str], bank_id: str, document_id: str
) -> list[ExistingChunk]:
    """The document's chunk ids, indexes and content hashes, in chunk order."""
    rows = await conn.fetch(
        f"""
        SELECT chunk_id, chunk_index, content_hash
        FROM {fq_table("chunks")}
        WHERE document_id = $1 AND bank_id = $2
        ORDER BY chunk_index
        """,
        document_id,
        bank_id,
    )
    return [
        ExistingChunk(
            chunk_id=row["chunk_id"],
            chunk_index=row["chunk_index"],
            content_hash=row["content_hash"],
        )
        for row in rows
    ]


async def recovery_chunk_hashes(
    *, pool, fq_table: Callable[[str], str], bank_id: str, document_id: str, content_hash: str
) -> set[str]:
    """Hashes of the chunks a crashed retain of this exact content already committed.

    Empty unless the document row carries ``content_hash``: a different hash is a new
    version, not a retry.
    """
    from ...db_utils import acquire_with_retry

    async with acquire_with_retry(pool) as conn:
        doc_row = await conn.fetchrow(
            f"SELECT content_hash FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
            document_id,
            bank_id,
        )
        if doc_row and doc_row["content_hash"] == content_hash:
            existing_rows = await load_existing_chunks(
                conn=conn, fq_table=fq_table, bank_id=bank_id, document_id=document_id
            )
            return {c.content_hash for c in existing_rows if c.content_hash}
    return set()


async def lock_document_for_write(
    *, conn, ops, fq_table: Callable[[str], str], bank_id: str, document_id: str
) -> str | None:
    """Create-if-missing and row-lock the document; its pre-existing hash (``'__pending__'`` if new)."""
    return await ops.lock_document_for_write(
        conn,
        fq_table("documents"),
        document_id,
        bank_id,
    )


async def lock_pending_document(*, conn, fq_table: Callable[[str], str], bank_id: str, document_id: str) -> None:
    """Insert a ``'__pending__'`` document row if none exists, then row-lock it."""
    await conn.execute(
        f"INSERT INTO {fq_table('documents')} (id, bank_id, original_text, content_hash) "
        f"VALUES ($1, $2, '', '__pending__') "
        f"ON CONFLICT (id, bank_id) DO NOTHING",
        document_id,
        bank_id,
    )
    await conn.fetchval(
        f"SELECT content_hash FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2 FOR UPDATE",
        document_id,
        bank_id,
    )


async def lock_document_hash(*, conn, fq_table: Callable[[str], str], bank_id: str, document_id: str) -> str | None:
    """Row-lock the document and read its content_hash; ``None`` if there is no row."""
    return await conn.fetchval(
        f"SELECT content_hash FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2 FOR UPDATE",
        document_id,
        bank_id,
    )


async def load_document_chunks(
    *, pool, fq_table: Callable[[str], str], bank_id: str, document_id: str, include_text: bool
) -> DocumentChunkState:
    """The document's hash (and body, if asked), then its chunks — the delta retain's base."""
    from ...db_utils import acquire_with_retry

    async with acquire_with_retry(pool) as conn:
        if include_text:
            doc_row_at_load = await conn.fetchrow(
                f"SELECT content_hash, original_text FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
                document_id,
                bank_id,
            )
            doc_hash_at_load = doc_row_at_load["content_hash"] if doc_row_at_load else None
            original_text_at_load = doc_row_at_load["original_text"] if doc_row_at_load else None
        else:
            doc_hash_at_load = await conn.fetchval(
                f"SELECT content_hash FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
                document_id,
                bank_id,
            )
            original_text_at_load = None

        # Load chunks after the document version. If a concurrent writer commits
        # between these reads, the hash precondition on metadata-only writes (or
        # the delta's extraction freshness recheck) forces a streaming fallback.
        existing_chunks = await load_existing_chunks(
            conn=conn, fq_table=fq_table, bank_id=bank_id, document_id=document_id
        )
    return DocumentChunkState(
        content_hash=doc_hash_at_load, original_text=original_text_at_load, chunks=existing_chunks
    )


async def current_document_hash(*, pool, fq_table: Callable[[str], str], bank_id: str, document_id: str) -> str | None:
    """The document's content_hash right now, unlocked — the delta's pre-extraction recheck."""
    from ...db_utils import acquire_with_retry

    async with acquire_with_retry(pool) as conn:
        return await conn.fetchval(
            f"SELECT content_hash FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
            document_id,
            bank_id,
        )


async def document_original_text(*, conn, fq_table: Callable[[str], str], bank_id: str, document_id: str) -> str | None:
    """The document's ``original_text``; ``None`` if it does not exist."""
    return await conn.fetchval(
        f"SELECT original_text FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2",
        document_id,
        bank_id,
    )


async def count_document_memories(*, conn, fq_table: Callable[[str], str], bank_id: str, document_id: str) -> int:
    """How many memory units the document owns."""
    count = await conn.fetchval(
        f"SELECT COUNT(*) FROM {fq_table('memory_units')} WHERE bank_id = $1 AND document_id = $2",
        bank_id,
        document_id,
    )
    return int(count or 0)


async def document_unit_ids(*, conn, fq_table: Callable[[str], str], bank_id: str, document_id: str) -> list[str]:
    """Every memory id the document owns, oldest first."""
    rows = await conn.fetch(
        f"""
        SELECT id::text FROM {fq_table("memory_units")}
        WHERE bank_id = $1 AND document_id = $2
        ORDER BY created_at
        """,
        bank_id,
        document_id,
    )
    return [row["id"] for row in rows]


async def delete_document_for_replace(
    *,
    conn,
    ops,
    fq_table: Callable[[str], str],
    bank_id: str,
    document_id: str,
    outgoing_unit_ids: list[str],
) -> datetime | None:
    """Drop the document's links in lock order, its memories, then its row; the row's ``created_at``."""
    from . import writes

    # Drop the outgoing facts' links in lock order before the cascade reaches them: a
    # concurrent delete of a document linked to this one would otherwise lock the
    # same bidirectional pairs from the other end (#4251).
    if ops is not None:
        await ops.delete_unit_links(conn, fq_table("memory_links"), bank_id, outgoing_unit_ids)

    # Explicitly delete memory_units by document_id BEFORE deleting the
    # document row. The CASCADE from documents→chunks→memory_units only
    # catches units that have a non-NULL chunk_id FK. Units with chunk_id=NULL
    # (e.g. from partial writes or edge cases) would survive the cascade.
    # This explicit delete ensures complete cleanup.
    await writes.delete_document(conn=conn, fq_table=fq_table, bank_id=bank_id, document_id=document_id)
    # Capture created_at before deletion so re-ingestion preserves it.
    return await conn.fetchval(
        f"DELETE FROM {fq_table('documents')} WHERE id = $1 AND bank_id = $2 RETURNING created_at",
        document_id,
        bank_id,
    )


async def upsert_document_row(
    *,
    conn,
    fq_table: Callable[[str], str],
    bank_id: str,
    document_id: str,
    original_text: str | None,
    content_hash: str,
    retain_params: dict | None,
    document_tags: list[str] | None,
    preserved_created_at: datetime | None,
) -> None:
    """Insert the document row, or update it in place if it exists."""
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


async def relabel_document_memories(
    *,
    conn,
    fq_table: Callable[[str], str],
    bank_id: str,
    document_id: str,
    tags: list[str],
    metadata: dict[str, Any],
    observation_scopes: list | str | None,
    final_tags: Callable[[list[str] | None], list[str]],
) -> RelabelResult:
    """Set the document's tags, metadata and observation scoping on every memory it owns."""
    from ...metadata_utils import drop_null_values
    from ...retain.fact_storage import _normalize_scopes

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
    new_tags_by_id = {row["id"]: final_tags(row["tags"]) for row in prior}
    # The comparison is against what the unit ends with.
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
    # Grouped by the FINAL array `final_tags` computed rather than by the projection
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

    # result is a status string like "UPDATE 5"
    try:
        updated = int(result.split()[-1])
    except (ValueError, IndexError):
        updated = 0
    return RelabelResult(updated=updated, rescoped_unit_ids=[str(uid) for uid in rescoped_ids])


async def memory_ids_for_chunks(
    *, conn, fq_table: Callable[[str], str], bank_id: str, chunk_ids: list[str]
) -> list[str]:
    """Ids of the ``experience``/``world`` memories these chunks own."""
    rows = await conn.fetch(
        f"""
        SELECT id
        FROM {fq_table("memory_units")}
        WHERE bank_id = $1
          AND chunk_id = ANY($2::text[])
          AND fact_type IN ('experience', 'world')
        """,
        bank_id,
        chunk_ids,
    )
    return [str(row["id"]) for row in rows]


async def delete_chunks(*, conn, fq_table: Callable[[str], str], bank_id: str, chunk_ids: list[str]) -> None:
    """Delete the named chunks with their memories and links, taking row locks in a total order."""
    if getattr(conn, "backend_type", None) == "oracle":
        # Oracle has no ctid, DELETE ... USING or FOR UPDATE inside a CTE, so the
        # ordered-lock form below cannot be ported as is: delete plainly, like
        # OracleOps.delete_unit_links (see prune_stale_cooccurrences for that asymmetry).
        units = f"SELECT id FROM {fq_table('memory_units')} WHERE chunk_id = ANY($1::text[]) AND bank_id = $2"
        await conn.execute(
            f"""
            DELETE FROM {fq_table("memory_links")}
            WHERE bank_id = $2 AND (from_unit_id IN ({units}) OR to_unit_id IN ({units}))
            """,
            chunk_ids,
            bank_id,
        )
        # Oracle's memory_units.chunk_id FK is still ON DELETE SET NULL (PG moved it to
        # CASCADE in f6g7h8i9j0k1), so the facts would outlive their chunk as duplicates.
        await conn.execute(
            f"DELETE FROM {fq_table('memory_units')} WHERE chunk_id = ANY($1::text[]) AND bank_id = $2",
            chunk_ids,
            bank_id,
        )
        await conn.execute(
            f"DELETE FROM {fq_table('chunks')} WHERE chunk_id = ANY($1::text[]) AND bank_id = $2",
            chunk_ids,
            bank_id,
        )
        return

    # PostgreSQL's FK cascade deletes child memory_links in executor-chosen
    # order. Concurrent chunk deletes for the same bank can then lock overlapping
    # memory_links in opposite orders and deadlock. Delete links explicitly in a
    # total order before deleting chunks so every writer takes row locks the same
    # way; the FK cascade still handles anything inserted later in this transaction.
    #
    # ``matched_links`` collects the endpoints as a UNION of two single-column joins
    # rather than the one ``tu.id = ml.from_unit_id OR tu.id = ml.to_unit_id`` predicate
    # it replaces. An OR spanning two columns of ``ml`` is not indexable: the planner
    # cannot drive it from either endpoint index, so it made memory_links the outer
    # relation of a nested-loop semi join and sequentially scanned the whole table once
    # per delete — O(rows_in_memory_links x target_units). Past a few million links that
    # exceeded the asyncpg command timeout and delta retain failed with a bare
    # TimeoutError (issue #3387). Split in two, each half is an index scan on
    # idx_memory_links_from_type_weight / idx_memory_links_to_type_weight.
    # The UNION yields the identical row set; the deterministic ORDER BY and
    # FOR UPDATE that #2570 added stay in ``ordered_links``, which locks the rows in
    # that order after the endpoints have been found.
    await conn.execute(
        f"""
        WITH target_units AS MATERIALIZED (
            SELECT id
            FROM {fq_table("memory_units")}
            WHERE chunk_id = ANY($1::text[])
              AND bank_id = $2
        ),
        matched_links AS MATERIALIZED (
            SELECT ml.ctid AS link_ctid
            FROM {fq_table("memory_links")} ml
            JOIN target_units tu ON tu.id = ml.from_unit_id
            UNION
            SELECT ml.ctid AS link_ctid
            FROM {fq_table("memory_links")} ml
            JOIN target_units tu ON tu.id = ml.to_unit_id
        ),
        ordered_links AS MATERIALIZED (
            SELECT ml.ctid
            FROM {fq_table("memory_links")} ml
            JOIN matched_links ON ml.ctid = matched_links.link_ctid
            ORDER BY
                LEAST(ml.from_unit_id, ml.to_unit_id),
                GREATEST(ml.from_unit_id, ml.to_unit_id),
                ml.link_type,
                COALESCE(ml.entity_id, '00000000-0000-0000-0000-000000000000'::uuid)
            FOR UPDATE OF ml
        )
        DELETE FROM {fq_table("memory_links")} ml
        USING ordered_links ol
        WHERE ml.ctid = ol.ctid
        """,
        chunk_ids,
        bank_id,
    )
    await conn.execute(
        f"""
        WITH ordered_chunks AS MATERIALIZED (
            SELECT chunk_id
            FROM {fq_table("chunks")}
            WHERE chunk_id = ANY($1::text[])
              AND bank_id = $2
            ORDER BY chunk_id
            FOR UPDATE
        )
        DELETE FROM {fq_table("chunks")} c
        USING ordered_chunks oc
        WHERE c.chunk_id = oc.chunk_id
        """,
        chunk_ids,
        bank_id,
    )


async def upsert_chunks(
    *,
    conn,
    ops,
    fq_table: Callable[[str], str],
    bank_id: str,
    document_id: str,
    chunk_ids: list[str],
    chunk_texts: list[str],
    chunk_indices: list[int],
    content_hashes: list[str],
) -> None:
    """Write the chunk rows, overwriting any with the same chunk_id."""
    # ON CONFLICT makes this idempotent: re-submitting a retain under the same
    # document_id may produce chunk_ids that already exist. Overwriting is the
    # correct behavior per document_id grouping semantics.
    await ops.bulk_upsert_chunks(
        conn,
        fq_table("chunks"),
        chunk_ids,
        [document_id] * len(chunk_texts),
        [bank_id] * len(chunk_texts),
        chunk_texts,
        chunk_indices,
        content_hashes,
    )


async def unit_embeddings(*, conn, fq_table: Callable[[str], str], bank_id: str, unit_ids: list[str]) -> list:
    """``(id, embedding, fact_type)`` rows for the given memories, embedding as pgvector text."""
    return await conn.fetch(
        f"""
        SELECT id::text, embedding::text, fact_type
        FROM {fq_table("memory_units")}
        WHERE bank_id = $1 AND id = ANY($2::uuid[])
        ORDER BY id
        """,
        bank_id,
        unit_ids,
    )
