"""
bank profile utilities for disposition and mission management.
"""

import json
import logging
import re
import uuid
from typing import TypedDict

from pydantic import BaseModel, Field

from ..._vector_index import index_using_clause, uses_per_bank_vector_indexes
from ...config import get_config
from ..db_utils import acquire_with_retry
from ..memory_engine import fq_table, get_current_schema
from ..response_models import DispositionTraits

logger = logging.getLogger(__name__)

# Fact types that get per-bank partial vector indexes, mapped to their 4-char index suffix.
_BANK_INDEX_FACT_TYPES: dict[str, str] = {
    "world": "worl",
    "experience": "expr",
    "observation": "obsv",
}


def _bank_index_name(ft: str, internal_id: str) -> str:
    """Deterministic, schema-safe vector index name for a (bank, fact_type) pair.

    Uses the first 16 hex chars of internal_id (8 bytes of entropy) — unique
    enough in practice, fits comfortably within PostgreSQL's 63-char identifier limit.
    """
    uid = str(internal_id).replace("-", "")[:16]
    return f"idx_mu_emb_{_BANK_INDEX_FACT_TYPES[ft]}_{uid}"


def _vector_index_clause() -> str | None:
    """Return the USING clause for per-bank vector indexes, if this backend uses them."""
    ext = get_config().vector_extension
    if not uses_per_bank_vector_indexes(ext):
        return None
    return index_using_clause(ext)


async def create_bank_vector_indexes(conn, bank_id: str, internal_id: str, ops=None) -> None:
    """Create per-(bank, fact_type) partial vector indexes for a newly created bank.

    Respects the HINDSIGHT_API_VECTOR_EXTENSION config to use the appropriate
    index type (HNSW for pgvector, DiskANN for pgvectorscale, vchordrq for vchord).

    AlloyDB ScaNN uses global vector indexes with filtered vector search; it
    cannot safely create per-bank indexes at bank-creation time because new
    banks have no embedding rows.
    bank_id is escaped for SQL literal safety (apostrophes doubled).

    On Oracle 23ai, this is a no-op — Oracle uses a single global vector index
    created during migrations. Partial indexes (WHERE clause) are not supported
    for Oracle vector indexes.
    """
    index_clause = _vector_index_clause()
    if index_clause is None:
        logger.debug("Skipping per-bank vector indexes for configured backend")
        return

    await ops.create_bank_vector_indexes(
        conn,
        fq_table("memory_units"),
        bank_id,
        internal_id,
        index_clause,
        _BANK_INDEX_FACT_TYPES,
    )


async def drop_bank_vector_indexes(conn, internal_id: str, ops=None) -> None:
    """Drop per-(bank, fact_type) partial vector indexes for a bank being deleted.

    Called before the bank row is deleted so internal_id is still known.
    Idempotent via DROP INDEX IF EXISTS.

    On Oracle, this is a no-op (uses single global vector index).
    """
    await ops.drop_bank_vector_indexes(
        conn,
        get_current_schema(),
        internal_id,
        _BANK_INDEX_FACT_TYPES,
    )


DEFAULT_DISPOSITION = {
    "skepticism": 3,
    "literalism": 3,
    "empathy": 3,
}


class BankProfile(TypedDict):
    """Type for bank profile data."""

    name: str
    disposition: DispositionTraits
    mission: str


class MissionMergeResponse(BaseModel):
    """LLM response for mission merge."""

    mission: str = Field(description="Merged mission in first person perspective")


async def get_bank_profile(pool, bank_id: str) -> BankProfile:
    """
    Get bank profile (name, disposition + mission).
    Auto-creates bank with default values if not exists.

    Args:
        pool: Database connection pool
        bank_id: bank IDentifier

    Returns:
        BankProfile with name, typed DispositionTraits, and mission
    """
    profile, _ = await get_or_create_bank_profile(pool, bank_id)
    return profile


async def get_bank_profile_if_exists(pool, bank_id: str) -> BankProfile | None:
    """
    Get bank profile (name, disposition + mission) without auto-creating.

    Returns None if the bank does not exist. This is the read-only variant
    of get_bank_profile, intended for read endpoints where a bank that
    doesn't exist should surface as 404 rather than be silently created.

    Args:
        pool: Database connection pool
        bank_id: bank IDentifier

    Returns:
        BankProfile if the bank exists, otherwise None.
    """
    async with acquire_with_retry(pool) as conn:
        row = await conn.fetchrow(
            f"""
            SELECT name, disposition, mission
            FROM {fq_table("banks")} WHERE bank_id = $1
            """,
            bank_id,
        )
        if not row:
            return None
        disposition_data = row["disposition"]
        if isinstance(disposition_data, str):
            disposition_data = json.loads(disposition_data)
        return BankProfile(
            name=row["name"],
            disposition=DispositionTraits(**disposition_data),
            mission=row["mission"] or "",
        )


async def get_or_create_bank_profile(pool, bank_id: str) -> tuple[BankProfile, bool]:
    """
    Get bank profile, auto-creating with defaults if it doesn't exist.

    Same as get_bank_profile, but also returns a flag indicating whether the
    bank was freshly created on this call. Used by the memory engine to apply
    the HINDSIGHT_API_DEFAULT_BANK_TEMPLATE hook on first bank creation.

    Returns:
        Tuple of (BankProfile, created) where created is True if the bank
        did not exist before this call.
    """
    async with acquire_with_retry(pool) as conn:
        # Try to get existing bank
        row = await conn.fetchrow(
            f"""
            SELECT name, disposition, mission
            FROM {fq_table("banks")} WHERE bank_id = $1
            """,
            bank_id,
        )

        if row:
            # asyncpg returns JSONB as a string, so parse it
            disposition_data = row["disposition"]
            if isinstance(disposition_data, str):
                disposition_data = json.loads(disposition_data)

            return (
                BankProfile(
                    name=row["name"],
                    disposition=DispositionTraits(**disposition_data),
                    mission=row["mission"] or "",
                ),
                False,
            )

        # Bank doesn't exist, create with defaults.
        # Generate internal_id here so we control the value and can use it
        # immediately for vector index creation without a RETURNING round-trip.
        internal_id = uuid.uuid4()
        inserted = await conn.fetchval(
            f"""
            INSERT INTO {fq_table("banks")} (bank_id, name, disposition, mission, internal_id)
            VALUES ($1, $2, $3::jsonb, $4, $5)
            ON CONFLICT (bank_id) DO NOTHING
            RETURNING bank_id
            """,
            bank_id,
            bank_id,  # Default name is the bank_id
            json.dumps(DEFAULT_DISPOSITION),
            "",
            internal_id,
        )

        created = inserted is not None
        if created:
            # Fresh insert — create per-bank vector indexes (instant on empty bank)
            await create_bank_vector_indexes(conn, bank_id, str(internal_id), ops=pool.ops)

        return (
            BankProfile(name=bank_id, disposition=DispositionTraits(**DEFAULT_DISPOSITION), mission=""),
            created,
        )


async def update_bank_disposition(pool, bank_id: str, disposition: dict[str, int]) -> None:
    """
    Update bank disposition traits.

    Args:
        pool: Database connection pool
        bank_id: bank IDentifier
        disposition: Dict with skepticism, literalism, empathy (all 1-5)
    """
    # Ensure bank exists first
    await get_bank_profile(pool, bank_id)

    async with acquire_with_retry(pool) as conn:
        await conn.execute(
            f"""
            UPDATE {fq_table("banks")}
            SET disposition = $2::jsonb,
                updated_at = NOW()
            WHERE bank_id = $1
            """,
            bank_id,
            json.dumps(disposition),
        )


async def set_bank_mission(pool, bank_id: str, mission: str) -> None:
    """
    Set bank mission (replacing any existing mission).

    Args:
        pool: Database connection pool
        bank_id: bank IDentifier
        mission: The mission text
    """
    # Ensure bank exists first
    await get_bank_profile(pool, bank_id)

    async with acquire_with_retry(pool) as conn:
        await conn.execute(
            f"""
            UPDATE {fq_table("banks")}
            SET mission = $2,
                updated_at = NOW()
            WHERE bank_id = $1
            """,
            bank_id,
            mission,
        )


async def merge_bank_mission(pool, llm_config, bank_id: str, new_info: str) -> dict:
    """
    Merge new mission information with existing mission using LLM.
    Normalizes to first person ("I") and resolves conflicts.

    Args:
        pool: Database connection pool
        llm_config: LLM configuration for mission merging
        bank_id: bank IDentifier
        new_info: New mission information to add/merge

    Returns:
        Dict with 'mission' (str) key
    """
    # Get current profile
    profile = await get_bank_profile(pool, bank_id)
    current_mission = profile["mission"]

    # Use LLM to merge missions
    result = await _llm_merge_mission(llm_config, current_mission, new_info)

    merged_mission = result["mission"]

    # Update in database
    async with acquire_with_retry(pool) as conn:
        await conn.execute(
            f"""
            UPDATE {fq_table("banks")}
            SET mission = $2,
                updated_at = NOW()
            WHERE bank_id = $1
            """,
            bank_id,
            merged_mission,
        )

    return {"mission": merged_mission}


async def _llm_merge_mission(llm_config, current: str, new_info: str) -> dict:
    """
    Use LLM to intelligently merge mission information.

    Args:
        llm_config: LLM configuration to use
        current: Current mission text
        new_info: New information to merge

    Returns:
        Dict with 'mission' (str) key
    """
    prompt = f"""You are helping maintain an agent's mission statement.

Current mission: {current if current else "(empty)"}

New information to add: {new_info}

Instructions:
1. Merge the new information with the current mission
2. If there are conflicts, the NEW information overwrites the old
3. Keep additions that don't conflict
4. Output in FIRST PERSON ("I") perspective
5. Be concise - keep it under 500 characters
6. Return ONLY the merged mission text, no explanations

Merged mission:"""

    try:
        messages = [{"role": "user", "content": prompt}]

        content = await llm_config.call(
            messages=messages, scope="bank_mission", temperature=0.3, max_completion_tokens=8192
        )

        logger.info(f"LLM response for mission merge (first 500 chars): {content[:500]}")

        merged = content.strip()
        if not merged or merged.lower() in ["(empty)", "none", "n/a"]:
            merged = new_info if new_info else ""
        return {"mission": merged}

    except Exception as e:
        logger.error(f"Error merging mission with LLM: {e}")
        # Fallback: just append new info
        if current:
            merged = f"{current} {new_info}".strip()
        else:
            merged = new_info

        return {"mission": merged}


async def list_banks(pool) -> list:
    """
    List all banks in the system with summary stats.

    Args:
        pool: Database connection pool

    Returns:
        List of dicts with bank info and stats (document_count, fact_count, last_event_at)
    """
    banks_table = fq_table("banks")
    docs_table = fq_table("documents")
    mu_table = fq_table("memory_units")

    async with acquire_with_retry(pool) as conn:
        rows = await conn.fetch(
            f"""
            SELECT
                b.bank_id, b.name, b.disposition, b.mission,
                b.created_at, b.updated_at,
                COALESCE(m.fact_count, 0) AS fact_count,
                d.last_document_at
            FROM {banks_table} b
            LEFT JOIN (
                SELECT bank_id, MAX(created_at) AS last_document_at
                FROM {docs_table}
                GROUP BY bank_id
            ) d ON d.bank_id = b.bank_id
            LEFT JOIN (
                SELECT bank_id, COUNT(*) AS fact_count
                FROM {mu_table}
                GROUP BY bank_id
            ) m ON m.bank_id = b.bank_id
            ORDER BY d.last_document_at DESC NULLS LAST, b.updated_at DESC
            """
        )

        result = []
        for row in rows:
            disposition_data = row["disposition"]
            if isinstance(disposition_data, str):
                disposition_data = json.loads(disposition_data)

            last_doc = row["last_document_at"]

            result.append(
                {
                    "bank_id": row["bank_id"],
                    "name": row["name"],
                    "disposition": disposition_data,
                    "mission": row["mission"] or "",
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                    "fact_count": row["fact_count"],
                    "last_document_at": last_doc.isoformat() if last_doc else None,
                }
            )

        return result
