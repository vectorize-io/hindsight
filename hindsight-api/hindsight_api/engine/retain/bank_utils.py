"""
bank profile utilities for disposition and mission management.
"""

import json
import logging
import re
import uuid
from typing import TypedDict

from pydantic import BaseModel, Field

from ..db_utils import acquire_with_retry
from ..memory_engine import fq_table, get_current_schema
from ..response_models import DispositionTraits

logger = logging.getLogger(__name__)

# Fact types that get per-bank partial HNSW indexes, mapped to their 4-char index suffix.
_HNSW_FACT_TYPES: dict[str, str] = {
    "world": "worl",
    "experience": "expr",
    "observation": "obsv",
}


def _hnsw_index_name(ft: str, internal_id: str) -> str:
    """Deterministic, schema-safe HNSW index name for a (bank, fact_type) pair.

    Uses the first 16 hex chars of internal_id (8 bytes of entropy) — unique
    enough in practice, fits comfortably within PostgreSQL's 63-char identifier limit.
    """
    uid = str(internal_id).replace("-", "")[:16]
    return f"idx_mu_emb_{_HNSW_FACT_TYPES[ft]}_{uid}"


async def create_bank_hnsw_indexes(conn, bank_id: str, internal_id: str) -> None:
    """Create per-(bank, fact_type) partial HNSW indexes for a newly created bank.

    Called immediately after the bank row is first inserted. Safe on empty banks
    (index build is instant). Idempotent via CREATE INDEX IF NOT EXISTS.
    bank_id is escaped for SQL literal safety (apostrophes doubled).
    """
    table = fq_table("memory_units")
    escaped = bank_id.replace("'", "''")
    for ft in _HNSW_FACT_TYPES:
        idx = _hnsw_index_name(ft, internal_id)
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS {idx} "
            f"ON {table} USING hnsw (embedding vector_cosine_ops) "
            f"WHERE fact_type = '{ft}' AND bank_id = '{escaped}'"
        )


async def drop_bank_hnsw_indexes(conn, internal_id: str) -> None:
    """Drop per-(bank, fact_type) partial HNSW indexes for a bank being deleted.

    Called before the bank row is deleted so internal_id is still known.
    Idempotent via DROP INDEX IF EXISTS.
    """
    schema = get_current_schema()
    for ft in _HNSW_FACT_TYPES:
        idx = _hnsw_index_name(ft, internal_id)
        await conn.execute(f"DROP INDEX IF EXISTS {schema}.{idx}")


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

            return BankProfile(
                name=row["name"],
                disposition=DispositionTraits(**disposition_data),
                mission=row["mission"] or "",
            )

        # Bank doesn't exist, create with defaults.
        # Generate internal_id here so we control the value and can use it
        # immediately for HNSW index creation without a RETURNING round-trip.
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

        if inserted:
            # Fresh insert — create per-bank HNSW indexes (instant on empty bank)
            await create_bank_hnsw_indexes(conn, bank_id, str(internal_id))

        return BankProfile(name=bank_id, disposition=DispositionTraits(**DEFAULT_DISPOSITION), mission="")


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
    List all banks in the system.

    Args:
        pool: Database connection pool

    Returns:
        List of dicts with bank_id, name, disposition, mission, created_at, updated_at
    """
    async with acquire_with_retry(pool) as conn:
        rows = await conn.fetch(
            f"""
            SELECT bank_id, name, disposition, mission, created_at, updated_at
            FROM {fq_table("banks")}
            ORDER BY updated_at DESC
            """
        )

        result = []
        for row in rows:
            # asyncpg returns JSONB as a string, so parse it
            disposition_data = row["disposition"]
            if isinstance(disposition_data, str):
                disposition_data = json.loads(disposition_data)

            result.append(
                {
                    "bank_id": row["bank_id"],
                    "name": row["name"],
                    "disposition": disposition_data,
                    "mission": row["mission"] or "",
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                }
            )

        return result
