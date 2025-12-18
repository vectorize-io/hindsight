"""
Chunk storage for retain pipeline.

Handles storage of document chunks in the database.
"""

import logging

from ..memory_engine import fq_table
from .types import ChunkMetadata

logger = logging.getLogger(__name__)


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
    chunk_id_map = {}

    for chunk in chunks:
        chunk_id = f"{bank_id}_{document_id}_{chunk.chunk_index}"
        chunk_ids.append(chunk_id)
        chunk_texts.append(chunk.chunk_text)
        chunk_indices.append(chunk.chunk_index)
        chunk_id_map[chunk.chunk_index] = chunk_id

    # Batch insert all chunks
    await conn.execute(
        f"""
        INSERT INTO {fq_table("chunks")} (chunk_id, document_id, bank_id, chunk_text, chunk_index)
        SELECT * FROM unnest($1::text[], $2::text[], $3::text[], $4::text[], $5::integer[])
        """,
        chunk_ids,
        [document_id] * len(chunk_texts),
        [bank_id] * len(chunk_texts),
        chunk_texts,
        chunk_indices,
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
