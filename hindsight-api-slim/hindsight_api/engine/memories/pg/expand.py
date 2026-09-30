"""The reads behind reflect's ``expand`` tool: memories by id, then their chunks and documents.

The SQL is lifted verbatim from ``engine/reflect/tools.py::tool_expand``, which now makes one
store call per read.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any


async def expand_memories(*, conn, fq_table: Callable[[str], str], bank_id: str, unit_ids: list[uuid.UUID]) -> list:
    """``id, text, chunk_id, document_id, fact_type, context`` for each of ``unit_ids`` that exists."""
    return await conn.fetch(
        f"""
            SELECT id, text, chunk_id, document_id, fact_type, context
            FROM {fq_table("memory_units")}
            WHERE id = ANY($1) AND bank_id = $2
            """,
        unit_ids,
        bank_id,
    )


async def expand_chunks(*, conn, fq_table: Callable[[str], str], chunk_ids: list[str]) -> dict[str, Any]:
    """chunk_id -> ``chunk_id, chunk_text, chunk_index, document_id`` for the chunks that exist.

    ``chunk_id`` is bank-encoded (``engine/chunk_ids.py``), so it scopes the read to the bank.
    """
    chunks = await conn.fetch(
        f"""
            SELECT chunk_id, chunk_text, chunk_index, document_id
            FROM {fq_table("chunks")}
            WHERE chunk_id = ANY($1)
            """,
        chunk_ids,
    )
    return {row["chunk_id"]: row for row in chunks}


async def expand_documents(
    *, conn, fq_table: Callable[[str], str], bank_id: str, document_ids: list[str]
) -> dict[str, Any]:
    """document id -> ``id, original_text, retain_params`` for the documents that exist."""
    docs = await conn.fetch(
        f"""
            SELECT id, original_text, retain_params
            FROM {fq_table("documents")}
            WHERE id = ANY($1) AND bank_id = $2
            """,
        document_ids,
        bank_id,
    )
    return {row["id"]: row for row in docs}
