"""
bank profile utilities for disposition and mission management.
"""

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, TypedDict

from pydantic import BaseModel, Field

from ..._vector_index import index_using_clause, uses_per_bank_vector_indexes
from ...config import get_config
from ..db.base import DatabaseBackend, DatabaseConnection
from ..db_utils import acquire_with_retry, retry_with_backoff
from ..memory_engine import fq_table, get_current_schema
from ..response_models import DispositionTraits

if TYPE_CHECKING:
    from ..llm_wrapper import LLMProvider

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


@dataclass
class BankProfileResult:
    """Result of a get-or-create bank lookup.

    ``created`` is True when the bank row was freshly inserted on this call,
    which callers use to drive the one-time HINDSIGHT_API_DEFAULT_BANK_TEMPLATE hook.
    """

    profile: BankProfile
    created: bool


@dataclass(frozen=True)
class BankMissionSnapshot:
    """Immutable identity and legacy mission used for a conditional merge."""

    internal_id: uuid.UUID | str
    mission: str | None
    updated_at: datetime


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
    result = await get_or_create_bank_profile(pool, bank_id)
    return result.profile


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
        return await get_bank_profile_on_conn(conn, bank_id)


async def get_bank_profile_on_conn(
    conn: DatabaseConnection,
    bank_id: str,
    *,
    for_update: bool = False,
) -> BankProfile | None:
    """Connection-bound read of an existing bank profile."""
    lock_clause = " FOR UPDATE" if for_update else ""
    row = await conn.fetchrow(
        f"""
        SELECT name, disposition, mission
        FROM {fq_table("banks")} WHERE bank_id = $1{lock_clause}
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


async def get_bank_mission_snapshot_on_conn(
    conn: DatabaseConnection,
    bank_id: str,
    *,
    for_update: bool = False,
) -> BankMissionSnapshot | None:
    """Load the immutable bank identity and legacy mission on one connection."""
    lock_clause = " FOR UPDATE" if for_update else ""
    row = await conn.fetchrow(
        f"""
        SELECT internal_id, mission, updated_at
        FROM {fq_table("banks")} WHERE bank_id = $1{lock_clause}
        """,
        bank_id,
    )
    if not row:
        return None
    return BankMissionSnapshot(
        internal_id=row["internal_id"],
        mission=row["mission"],
        updated_at=row["updated_at"],
    )


async def get_or_create_bank_profile(pool, bank_id: str) -> BankProfileResult:
    """
    Get bank profile, auto-creating with defaults if it doesn't exist.

    Same as get_bank_profile, but also reports whether the bank was freshly
    created on this call (``BankProfileResult.created``). Used by the memory
    engine to apply the HINDSIGHT_API_DEFAULT_BANK_TEMPLATE hook on first bank
    creation.

    Acquires its own connection. When the caller already holds a connection and
    wants the bank row to share its transaction (so the lazy bank-create commits
    or rolls back atomically with the caller's write), use
    ``get_or_create_bank_profile_on_conn`` instead.
    """

    # A fresh bank builds its per-(bank, fact_type) partial vector indexes with
    # a plain CREATE INDEX (it must — this runs inside the bank-create tx, and
    # CONCURRENTLY cannot). That CREATE takes a ShareLock on the shared
    # memory_units table, which can deadlock with concurrent writers. The build
    # is idempotent (INSERT ... ON CONFLICT + CREATE INDEX IF NOT EXISTS), so a
    # transient deadlock (40P01 / ORA-00060) is safe to retry as a whole tx.
    async def _create() -> BankProfileResult:
        async with acquire_with_retry(pool) as conn:
            async with conn.transaction():
                return await get_or_create_bank_profile_on_conn(conn, bank_id, ops=pool.ops)

    return await retry_with_backoff(_create)


async def get_or_create_bank_profile_on_conn(conn, bank_id: str, *, ops) -> BankProfileResult:
    """
    Connection-bound variant of ``get_or_create_bank_profile``.

    Runs the SELECT, the ``INSERT ... ON CONFLICT DO NOTHING`` and the per-bank
    vector index creation on the caller-supplied ``conn``. When ``conn`` is
    inside an open transaction, the lazy bank-create therefore commits (or rolls
    back) atomically with whatever bank-scoped write the caller performs on the
    same connection — closing the window where a freshly-created bank could
    outlive a write that ultimately failed.

    ``ops`` is the backend's dialect ops object (``backend.ops``), needed for
    per-bank vector index DDL.
    """
    # Lock an existing row so a concurrent delete cannot slip between this
    # lifecycle check and the caller's bank-scoped write.
    profile = await get_bank_profile_on_conn(conn, bank_id, for_update=True)
    if profile:
        return BankProfileResult(profile=profile, created=False)

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
        await create_bank_vector_indexes(conn, bank_id, str(internal_id), ops=ops)

    return BankProfileResult(
        profile=BankProfile(name=bank_id, disposition=DispositionTraits(**DEFAULT_DISPOSITION), mission=""),
        created=created,
    )


def _require_bank_row_updated(result: str, bank_id: str) -> None:
    """Fail rather than reporting success when a concurrent delete won."""
    updated = int(result.split()[-1]) if result.startswith("UPDATE") else 0
    if updated == 0:
        raise RuntimeError(f"Bank '{bank_id}' disappeared before the legacy profile write")


async def update_bank_disposition_on_conn(
    conn: DatabaseConnection,
    bank_id: str,
    disposition: dict[str, int],
) -> None:
    """
    Update bank disposition traits on a caller-supplied connection.

    Args:
        conn: Database connection
        bank_id: bank IDentifier
        disposition: Dict with skepticism, literalism, empathy (all 1-5)
    """
    result = await conn.execute(
        f"""
        UPDATE {fq_table("banks")}
        SET disposition = $2::jsonb,
            updated_at = NOW()
        WHERE bank_id = $1
        """,
        bank_id,
        json.dumps(disposition),
    )
    _require_bank_row_updated(result, bank_id)


async def update_bank_disposition(
    pool: DatabaseBackend,
    bank_id: str,
    disposition: dict[str, int],
) -> None:
    """Compatibility wrapper that preserves the legacy lazy-create contract."""
    await get_bank_profile(pool, bank_id)
    async with acquire_with_retry(pool) as conn:
        await update_bank_disposition_on_conn(conn, bank_id, disposition)


async def set_bank_mission_on_conn(conn: DatabaseConnection, bank_id: str, mission: str) -> None:
    """
    Set bank mission on a caller-supplied connection.

    Args:
        conn: Database connection
        bank_id: bank IDentifier
        mission: The mission text
    """
    result = await conn.execute(
        f"""
        UPDATE {fq_table("banks")}
        SET mission = $2,
            updated_at = NOW()
        WHERE bank_id = $1
        """,
        bank_id,
        mission,
    )
    _require_bank_row_updated(result, bank_id)


async def set_bank_mission(pool: DatabaseBackend, bank_id: str, mission: str) -> None:
    """Compatibility wrapper that preserves the legacy lazy-create contract."""
    await get_bank_profile(pool, bank_id)
    async with acquire_with_retry(pool) as conn:
        await set_bank_mission_on_conn(conn, bank_id, mission)


async def merge_bank_mission_from_profile(
    pool: DatabaseBackend,
    llm_config: "LLMProvider",
    bank_id: str,
    current_mission: str,
    new_info: str,
) -> dict[str, str]:
    """
    Merge new mission information with existing mission using LLM.
    Normalizes to first person ("I") and resolves conflicts.

    Args:
        pool: Database connection pool
        llm_config: LLM configuration for mission merging
        bank_id: bank IDentifier
        current_mission: Existing mission loaded by the authorized caller
        new_info: New mission information to add/merge

    Returns:
        Dict with 'mission' (str) key
    """
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


async def merge_bank_mission_from_snapshot(
    pool: DatabaseBackend,
    llm_config: "LLMProvider",
    bank_id: str,
    snapshot: BankMissionSnapshot,
    current_mission: str,
    new_info: str,
) -> dict[str, str]:
    """Merge a mission without overwriting another bank lifecycle or writer."""
    result = await _llm_merge_mission(llm_config, current_mission, new_info)
    merged_mission = result["mission"]

    async with acquire_with_retry(pool) as conn:
        update_result = await conn.execute(
            f"""
            UPDATE {fq_table("banks")}
            SET mission = $2,
                updated_at = NOW()
            WHERE bank_id = $1
              AND internal_id = $3
              AND updated_at = $4
            """,
            bank_id,
            merged_mission,
            snapshot.internal_id,
            snapshot.updated_at,
        )
    updated = int(update_result.split()[-1]) if update_result.startswith("UPDATE") else 0
    if updated == 0:
        raise RuntimeError(f"Bank '{bank_id}' changed while its mission was being merged")

    return {"mission": merged_mission}


async def merge_bank_mission(
    pool: DatabaseBackend,
    llm_config: "LLMProvider",
    bank_id: str,
    new_info: str,
) -> dict[str, str]:
    """Compatibility wrapper that preserves the legacy lazy-create contract."""
    profile = await get_bank_profile(pool, bank_id)
    return await merge_bank_mission_from_profile(
        pool,
        llm_config,
        bank_id,
        profile["mission"],
        new_info,
    )


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
        # A store that keeps memories outside SQL leaves the memory_units join empty, so its
        # per-bank fact_count comes from the store instead (one live count per bank).
        from ..memories import get_memories

        _store = get_memories()

        for row in rows:
            disposition_data = row["disposition"]
            if isinstance(disposition_data, str):
                disposition_data = json.loads(disposition_data)

            last_doc = row["last_document_at"]

            fact_count = row["fact_count"]
            if not _store.writes_memory_rows_in_sql:
                fact_count = sum(
                    (await _store.count_memories(conn=conn, fq_table=fq_table, bank_id=row["bank_id"])).values()
                )

            result.append(
                {
                    "bank_id": row["bank_id"],
                    "name": row["name"],
                    "disposition": disposition_data,
                    "mission": row["mission"] or "",
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                    "fact_count": fact_count,
                    "last_document_at": last_doc.isoformat() if last_doc else None,
                }
            )

        return result
