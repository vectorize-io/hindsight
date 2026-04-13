"""
Chunk storage for retain pipeline.

Handles storage of document chunks in the database.
"""

import hashlib
import logging
from dataclasses import dataclass

from ..memory_engine import fq_table
from .types import ChunkMetadata

logger = logging.getLogger(__name__)


def compute_chunk_hash(chunk_text: str) -> str:
    """Compute SHA256 hash of chunk text for delta comparison."""
    return hashlib.sha256(chunk_text.encode()).hexdigest()


@dataclass
class ExistingChunk:
    """Represents a chunk already stored in the database."""

    chunk_id: str
    chunk_index: int
    content_hash: str | None


async def load_existing_chunks(conn, bank_id: str, document_id: str) -> list[ExistingChunk]:
    """
    Load existing chunk metadata for a document.

    Returns list of ExistingChunk with chunk_id, chunk_index, and content_hash.
    """
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


async def delete_chunks_by_ids(conn, chunk_ids: list[str]) -> None:
    """
    Delete specific chunks by their IDs.

    This cascades to memory_units (via FK with CASCADE delete)
    and their links.
    """
    if not chunk_ids:
        return
    await conn.execute(
        f"DELETE FROM {fq_table('chunks')} WHERE chunk_id = ANY($1::text[])",
        chunk_ids,
    )


async def store_chunks_batch(conn, bank_id: str, document_id: str, chunks: list[ChunkMetadata]) -> dict[int, str]:
    """
    Store document chunks in the database.

    Args:
        conn: Database connection
        bank_id: Bank identifier
        document_id: Document identifier
        chunks: List of ChunkMetadata objects

    Returns:
        Dictionary mapping global chunk index to chunk_id
    """
    if not chunks:
        return {}

    # Prepare chunk data for batch insert
    chunk_ids = []
    chunk_texts = []
    chunk_indices = []
    content_hashes = []
    chunk_id_map = {}

    for chunk in chunks:
        chunk_id = f"{bank_id}_{document_id}_{chunk.chunk_index}"
        chunk_ids.append(chunk_id)
        chunk_texts.append(chunk.chunk_text)
        chunk_indices.append(chunk.chunk_index)
        content_hashes.append(compute_chunk_hash(chunk.chunk_text))
        chunk_id_map[chunk.chunk_index] = chunk_id

    # Batch upsert all chunks. ON CONFLICT makes this idempotent: re-submitting
    # a retain under the same document_id (the pattern in vectorize-io/hindsight#977)
    # may produce chunk_ids that already exist when upstream cascade-delete or
    # delta-retain paths don't run (or race with a concurrent task). Overwriting
    # is the correct behavior per the document_id grouping semantics — the caller
    # intends this chunk to hold the latest content at that (document_id, index).
    await conn.execute(
        f"""
        INSERT INTO {fq_table("chunks")} (chunk_id, document_id, bank_id, chunk_text, chunk_index, content_hash)
        SELECT * FROM unnest($1::text[], $2::text[], $3::text[], $4::text[], $5::integer[], $6::text[])
        ON CONFLICT (chunk_id) DO UPDATE SET
            chunk_text = EXCLUDED.chunk_text,
            chunk_index = EXCLUDED.chunk_index,
            content_hash = EXCLUDED.content_hash
        """,
        chunk_ids,
        [document_id] * len(chunk_texts),
        [bank_id] * len(chunk_texts),
        chunk_texts,
        chunk_indices,
        content_hashes,
    )

    return chunk_id_map


def map_facts_to_chunks(facts_chunk_indices: list[int], chunk_id_map: dict[int, str]) -> list[str | None]:
    """
    Map fact chunk indices to chunk IDs.

    Args:
        facts_chunk_indices: List of chunk indices for each fact
        chunk_id_map: Dictionary mapping chunk index to chunk_id

    Returns:
        List of chunk_ids (same length as facts_chunk_indices)
    """
    chunk_ids = []
    for chunk_idx in facts_chunk_indices:
        chunk_id = chunk_id_map.get(chunk_idx)
        chunk_ids.append(chunk_id)
    return chunk_ids
