"""
bank profile utilities for disposition and mission management.
"""

import asyncio
import json
import logging
import unicodedata
import uuid
from contextlib import aclosing
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict

from pydantic import BaseModel, Field

from ..._vector_index import index_using_clause, per_bank_indexes_are_eager, uses_per_bank_vector_indexes
from ...config import get_config
from ..db_utils import acquire_with_retry, retry_with_backoff
from ..memory_engine import fq_table, get_current_schema
from ..response_models import DispositionTraits

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ..db.base import DatabaseConnection
    from ..db.ops import DataAccessOps
    from ..memories.base import BankWritePage, BankWriteTime

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


def bank_indexes_are_store_owned(bank_id: str) -> bool:
    """Whether ``bank_id``'s memories live outside SQL, so it can never have rows to index.

    A store-owned bank writes no ``memory_units`` rows at all, so its three partial
    indexes can only ever be empty — and an empty index is not free. Postgres plans
    against every index on a relation, so 3 x banks empty indexes on the shared
    ``memory_units`` tax every OTHER statement that so much as names the table: a
    tenant with 27,315 store-owned banks carried 82,795 of them and paid ~975 ms of
    planning for a query over zero rows (#4615). #4326 took the same decision for the
    whole-table reconcile — dimension resize, global vector and text indexes — but it
    does not reach the per-bank loops, which are built from the engine, not migrations.

    Asked per bank, not per deployment: a router can keep some banks in SQL and some
    in a store (:meth:`MemoriesExtension.store_owned_for`), and getting this backwards
    is silent — an SQL-owned bank without its index still recalls, just without ANN.

    **Raises rather than guessing when the store cannot answer**, because the two
    callers want opposite fallbacks and only they know which:

    * :func:`create_bank_vector_indexes` catches and builds. One bank's three unused
      indexes cost almost nothing, and a bank that turns out to be SQL-owned without
      them silently loses ANN. It also runs inside the bank-create transaction, where
      an escaping exception would fail an ordinary first retain.
    * :func:`~..vector_index_health.plan_bank_vector_indexes` lets it propagate. Its
      fallback is the destructive one: a transient router blip partway through
      ``repair-bank --all`` would classify every bank it failed on as SQL-backed and
      rebuild all three indexes for each — re-arming #4615 at full scale, from a
      command that then exits 0 and reads as a successful repair. Propagating instead
      makes the submit-time pre-check log a warning and queue nothing, and makes
      ``repair-bank`` report that schema skipped and exit non-zero — loud, and scoped
      to the schema, rather than quietly rebuilding. (The maintenance job itself does
      not catch it, so it would fail the operation into worker retry; the submit gate
      is what keeps the job from being queued in that state at all.)
    """
    # Local import, like the other five get_memories call sites in this module. At
    # module scope it closes a cycle: engine.memories -> memories.base ->
    # hindsight_api.extensions -> extensions.mcp -> `from hindsight_api import
    # MemoryEngine`, which is still partially initialised, so importing
    # hindsight_api.admin.cli dies on ImportError. (An earlier version of this comment
    # blamed engine.memories importing from engine.retain — it does not.)
    from ..memories import get_memories

    return get_memories().store_owned_for(bank_id)


async def create_bank_vector_indexes(
    conn: "DatabaseConnection", bank_id: str, internal_id: str, *, ops: "DataAccessOps"
) -> None:
    """Create per-(bank, fact_type) partial vector indexes for a newly created bank.

    Only does anything when the size threshold is off — the default — where a
    bank is entitled to its indexes from the moment it exists and building them
    here is instant, because the bank is empty. That is the behaviour that
    predates the threshold, and keeping it means the default deployment never
    submits a ``vector_index_maintenance`` operation and never pays a coverage
    check on a write.

    With ``HINDSIGHT_API_VECTOR_INDEX_MIN_ROWS`` set, a fresh bank has not earned
    anything yet, so this is a no-op and the maintenance operation builds an
    index CONCURRENTLY if and when the bank grows into one — which also keeps
    CREATE INDEX's ShareLock on the shared memory_units table off the write path.

    Respects the HINDSIGHT_API_VECTOR_EXTENSION config to use the appropriate
    index type (HNSW for pgvector, DiskANN for pgvectorscale, vchordrq for vchord).

    AlloyDB ScaNN uses global vector indexes with filtered vector search; it
    cannot safely create per-bank indexes at bank-creation time because new
    banks have no embedding rows. On Oracle 23ai this is likewise a no-op —
    Oracle uses a single global vector index created during migrations, and does
    not support partial (WHERE-clause) vector indexes.

    A bank whose memories a custom store owns is the third no-op, and the only one
    decided per bank rather than per deployment: it has no memory_units rows to index
    (see :func:`bank_indexes_are_store_owned`).

    bank_id is escaped for SQL literal safety (apostrophes doubled).

    ``ops`` is required rather than defaulting to None: it is only dereferenced
    in the eager branch, so a caller that forgot it used to work fine until the
    threshold happened to be off, and then failed on an AttributeError from
    inside bank creation.
    """
    if not per_bank_indexes_are_eager():
        return

    index_clause = _vector_index_clause()
    if index_clause is None:
        logger.debug("Skipping per-bank vector indexes for configured backend")
        return

    # Checked in the callee rather than at each caller because import restores a bank
    # around the fresh-INSERT gate and calls this directly (#2645). Last of the three
    # gates: the only one that reaches outside this process, so the two local ones
    # answer first. Catching is the safe direction HERE and only here — see
    # bank_indexes_are_store_owned for why the reconcile chooses the opposite.
    try:
        store_owned = bank_indexes_are_store_owned(bank_id)
    except Exception as e:  # noqa: BLE001 — a store that cannot answer must not fail bank creation
        logger.warning(
            "Store cannot say whether bank %s is store-owned (%s); building its vector indexes as SQL-backed. "
            "If this bank's memories live in the store, those indexes are empty and can be dropped.",
            bank_id,
            e,
        )
        store_owned = False
    if store_owned:
        logger.debug("Skipping per-bank vector indexes for store-owned bank %s", bank_id)
        return

    await ops.create_bank_vector_indexes(
        conn,
        fq_table("memory_units"),
        bank_id,
        internal_id,
        index_clause,
        _BANK_INDEX_FACT_TYPES,
    )


async def drop_bank_vector_indexes(conn: "DatabaseConnection", internal_id: str, *, ops: "DataAccessOps") -> None:
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


async def bank_exists_on_conn(conn: "DatabaseConnection", bank_id: str) -> bool:
    """Existence probe for a bank row on a caller-supplied connection.

    Deliberately narrower than ``get_bank_profile_if_exists_on_conn``: callers on
    the lazy-create hot path only need to know whether the row is there, so this
    selects a constant instead of reading and JSON-decoding the disposition.
    """
    return await conn.fetchval(f"SELECT 1 FROM {fq_table('banks')} WHERE bank_id = $1", bank_id) is not None


async def bank_exists(pool, bank_id: str) -> bool:
    """Dedicated-connection variant of ``bank_exists_on_conn``.

    A bare SELECT with no surrounding transaction — this is the read-only fast
    path taken by every write to an already-existing bank, so it must not pay
    for a BEGIN/COMMIT round trip.
    """
    async with acquire_with_retry(pool) as conn:
        return await bank_exists_on_conn(conn, bank_id)


async def get_bank_profile_if_exists_on_conn(conn: "DatabaseConnection", bank_id: str) -> BankProfile | None:
    """Get bank profile (name, disposition + mission) on conn without auto-creating.

    Returns None if the bank does not exist.
    """
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
    # Cached per process (see engine/bank_info_cache): on a hit this takes no pooled connection at
    # all, which is what lets a store-owned retain -- one that writes nothing to Postgres -- run
    # without touching the pool. A miss reads exactly as before.
    from .. import bank_info_cache

    async def _load() -> dict:
        async with acquire_with_retry(pool) as conn:
            profile = await get_bank_profile_if_exists_on_conn(conn, bank_id)
        # "The bank does not exist" has to be representable as a cached value, or every miss for a
        # missing bank re-reads it. `BankProfile` is a TypedDict, so the only thing that has to be
        # unpacked is `disposition`, which is a pydantic model the cache would otherwise hand back
        # by reference and let a caller mutate for every other holder of the entry.
        if profile is None:
            return {}
        return {**profile, "disposition": profile["disposition"].model_dump()}

    row = await bank_info_cache.get_or_load(bank_id, "profile", _load)
    if not row:
        return None
    return BankProfile(
        name=row["name"],
        disposition=DispositionTraits(**row["disposition"]),
        mission=row["mission"] or "",
    )


# Bytes, not characters: storage keys percent-encode the bank id (up to 3x per UTF-8 byte), so
# 192 bytes is at most 576 encoded bytes, under S3's 1,024-byte key limit with room for the
# tenant, the prefix and the file name, while still fitting 64 CJK characters (#4391).
BANK_ID_MAX_BYTES = 192


def validate_new_bank_id(bank_id: str) -> None:
    """Reject a bank id that must not be created. Existing banks are never re-checked."""
    from hindsight_api.extensions import OperationValidationError

    if not bank_id:
        raise OperationValidationError("Bank id must not be empty", status_code=400)
    if len(bank_id.encode("utf-8")) > BANK_ID_MAX_BYTES:
        raise OperationValidationError(
            f"Bank id is too long: at most {BANK_ID_MAX_BYTES} bytes of UTF-8 are allowed", status_code=400
        )
    if any(unicodedata.category(ch) == "Cc" for ch in bank_id):
        raise OperationValidationError("Bank id must not contain control characters", status_code=400)


async def create_bank_row_on_conn(conn: "DatabaseConnection", bank_id: str, *, ops: "DataAccessOps") -> bool:
    """Idempotently insert a default bank row and its vector indexes on conn.

    Uses ``INSERT ... ON CONFLICT (bank_id) DO NOTHING RETURNING bank_id``.
    Returns True if the bank was freshly inserted on this call, False if it already existed.

    The ``created`` flag drives one-time side effects (per-bank vector indexes
    here, the default-bank-template hook in the caller), so it has to be exact on
    both dialects. It is: the Oracle layer strips RETURNING from ON CONFLICT DO
    NOTHING, but ``fetchval`` compensates — it returns the first bind argument on
    a successful insert and None when it suppresses ORA-00001 (see
    ``db/oracle.py``). Use ``fetchval`` here, not ``fetchrow``, which has no such
    compensation and would report every fresh Oracle insert as an existing row.
    """
    # internal_id is minted here rather than defaulted server-side so its value is
    # known without a RETURNING round-trip: the index names derive from it, both
    # for the eager create below and for the maintenance operation when a
    # threshold is set.
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
        # Fresh insert — create per-bank vector indexes (instant on an empty
        # bank). A no-op unless the size threshold is off; see
        # create_bank_vector_indexes.
        await create_bank_vector_indexes(conn, bank_id, str(internal_id), ops=ops)

    return created


async def create_bank_if_missing(pool, bank_id: str) -> bool:
    """Dedicated-connection variant of ``create_bank_row_on_conn``.

    Returns True if freshly created on this call, False if it already existed.
    """

    # Retried as a whole transaction. With the size threshold off (the default)
    # a fresh SQL-owned bank builds its per-(bank, fact_type) partial vector
    # indexes with a plain CREATE INDEX — it must, since this runs inside the
    # bank-create tx and CONCURRENTLY cannot — and that CREATE takes a ShareLock
    # on the shared memory_units table, which can deadlock with concurrent
    # writers. Even with no DDL to issue, the lazy create can lose a deadlock
    # (40P01 / ORA-00060) to a concurrent writer touching the same bank row.
    # The body is idempotent
    # (INSERT ... ON CONFLICT DO NOTHING + CREATE INDEX IF NOT EXISTS), so
    # retrying the whole tx stays correct and cheap.
    async def _create() -> bool:
        async with acquire_with_retry(pool) as conn:
            async with conn.transaction():
                return await create_bank_row_on_conn(conn, bank_id, ops=pool.ops)

    return await retry_with_backoff(_create)


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

    # Retried as a whole transaction — see the deadlock note on
    # ``create_bank_if_missing``; the body is idempotent, so retrying is safe.
    async def _create() -> BankProfileResult:
        async with acquire_with_retry(pool) as conn:
            async with conn.transaction():
                return await get_or_create_bank_profile_on_conn(conn, bank_id, ops=pool.ops)

    return await retry_with_backoff(_create)


async def get_or_create_bank_profile_on_conn(
    conn: "DatabaseConnection", bank_id: str, *, ops: "DataAccessOps"
) -> BankProfileResult:
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
    profile = await get_bank_profile_if_exists_on_conn(conn, bank_id)
    if profile is not None:
        return BankProfileResult(
            profile=profile,
            created=False,
        )

    # Bank doesn't exist, create with defaults.
    created = await create_bank_row_on_conn(conn, bank_id, ops=ops)

    return BankProfileResult(
        profile=BankProfile(name=bank_id, disposition=DispositionTraits(**DEFAULT_DISPOSITION), mission=""),
        created=created,
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

    # The profile row just changed; the next read must not serve the entry loaded before it.
    from .. import bank_info_cache

    await bank_info_cache.invalidate(bank_id, "profile")


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

    # The profile row just changed; the next read must not serve the entry loaded before it.
    from .. import bank_info_cache

    await bank_info_cache.invalidate(bank_id, "profile")


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

    # The profile row just changed; the next read must not serve the entry loaded before it.
    from .. import bank_info_cache

    await bank_info_cache.invalidate(bank_id, "profile")

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

        call_result = await llm_config.call(
            messages=messages, scope="bank_mission", temperature=0.3, max_completion_tokens=8192
        )
        content = call_result.content

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


# Sort floor for banks that have never been written to and carry no created_at.
_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _as_utc(ts: datetime | None) -> datetime | None:
    """Normalize a DB timestamp to an aware UTC datetime so values stay comparable."""
    if ts is None:
        return None
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


async def list_banks(pool, *, search_query: str | None = None) -> list:
    """
    List banks with summary stats, optionally narrowed by a search string.

    ``last_document_at`` is document *ingestion* time (when a document first
    landed), while ``last_write_at`` is the last time anything was written to
    the bank — a document re-retained/appended to, or a fact stored. Appending
    to a long-lived document does not move ``last_document_at``, which is why
    the two differ and why UIs showing "last write" must use ``last_write_at``.

    ``fact_count`` comes back 0 here and is filled per page by
    :func:`apply_sql_fact_counts` (and, for banks whose memories live outside SQL,
    :func:`apply_store_fact_counts`). Counting every bank's facts in this query meant
    aggregating all of ``memory_units`` to render one page of banks, which is what made a
    365-bank list time out (#4468). The write watermark still has to be read for every bank,
    since it orders the list — but as one index entry per bank off an index that already
    exists, not a GROUP BY over the whole table. Measured on 365 banks / 1.8M facts: 131 ms
    to 6 ms, and no longer growing with the number of facts in the system.

    Args:
        pool: Database connection pool
        search_query: Case-insensitive substring matched against bank ID and name

    Returns:
        List of dicts with bank info and stats (fact_count, last_document_at, last_write_at),
        most recently written bank first.
    """
    banks_table = fq_table("banks")
    docs_table = fq_table("documents")
    mu_table = fq_table("memory_units")

    # Spelled out as UPPER(...) LIKE UPPER(...) rather than ILIKE: the Oracle
    # rewriter only recognizes ILIKE on an unqualified column, and these are
    # alias-qualified.
    where_clause = ""
    params: list[str] = []
    if search_query:
        where_clause = "WHERE (UPPER(b.bank_id) LIKE UPPER($1) OR UPPER(COALESCE(b.name, '')) LIKE UPPER($2))"
        params = [f"%{search_query}%", f"%{search_query}%"]

    async with acquire_with_retry(pool) as conn:
        rows = await conn.fetch(
            f"""
            SELECT
                b.bank_id, b.name, b.disposition, b.mission,
                b.created_at, b.updated_at,
                d.last_document_at,
                d.last_document_write_at,
                -- Per bank, off `idx_memory_units_bank_updated_at`: one index entry read instead of
                -- the GROUP BY over all of memory_units this replaced. `updated_at`, not
                -- `created_at`, because that index is the one that exists — and it is the right
                -- column anyway: this feeds `last_write_at` only, and every write to a fact bumps
                -- `updated_at`, so the watermark is the same or newer, which is what "last written"
                -- means. The documents half stays a GROUP BY: correlating it would be random reads
                -- over the same rows, since this query lists every bank before paging, and
                -- `last_document_at` must stay ingestion time (frozen when a document is rewritten).
                (SELECT MAX(m.updated_at) FROM {mu_table} m WHERE m.bank_id = b.bank_id) AS last_fact_at
            FROM {banks_table} b
            LEFT JOIN (
                SELECT bank_id,
                       MAX(created_at) AS last_document_at,
                       MAX(updated_at) AS last_document_write_at
                FROM {docs_table}
                GROUP BY bank_id
            ) d ON d.bank_id = b.bank_id
            {where_clause}
            ORDER BY b.bank_id
            """,
            *params,
        )

        result = []
        # Banks are ordered by last write in Python rather than SQL: GREATEST() has
        # different NULL semantics on PostgreSQL vs Oracle, and the bank list is small.
        sort_keys: dict[str, datetime] = {}

        for row in rows:
            disposition_data = row["disposition"]
            if isinstance(disposition_data, str):
                disposition_data = json.loads(disposition_data)

            last_doc = _as_utc(row["last_document_at"])
            created_at = _as_utc(row["created_at"])
            updated_at = _as_utc(row["updated_at"])
            # Last write = newest of "a document was (re-)retained" and "a fact was stored".
            # Appending to an existing document only bumps documents.updated_at, and facts
            # written outside a retain (consolidation, curation, import) only bump memory_units.
            write_times = [t for t in (_as_utc(row["last_document_write_at"]), _as_utc(row["last_fact_at"])) if t]
            last_write = max(write_times) if write_times else None

            sort_keys[row["bank_id"]] = last_write or created_at or _UNIX_EPOCH
            result.append(
                {
                    "bank_id": row["bank_id"],
                    "name": row["name"],
                    "disposition": disposition_data,
                    "mission": row["mission"] or "",
                    "created_at": created_at.isoformat() if created_at else None,
                    "updated_at": updated_at.isoformat() if updated_at else None,
                    # Filled per page by apply_sql_fact_counts / apply_store_fact_counts: counting
                    # every bank's facts here is what made this query scan all of memory_units.
                    "fact_count": 0,
                    "last_document_at": last_doc.isoformat() if last_doc else None,
                    "last_write_at": last_write.isoformat() if last_write else None,
                }
            )

    # Banks whose memories live outside SQL have no `documents` / `memory_units` rows for the joins
    # above to take a MAX over, so their `last_write_at` came back NULL and they sorted as if never
    # written. The store can answer it, and the Counters RPC is BATCHED — every such bank costs one
    # call in total — so it is done BEFORE the sort. Doing it in the page-level overlay instead
    # would fix the field but leave the order wrong, and inconsistent across pages, which is worse
    # than uniformly wrong.
    #
    # Outside the connection block on purpose: these are network calls to another service and
    # nothing below needs `conn`. Holding a pooled connection across them is what starves the pool
    # under load, and the retain path enforces the same rule everywhere else.
    await _apply_store_last_write(result, sort_keys)

    result.sort(key=lambda bank: sort_keys[bank["bank_id"]], reverse=True)
    return result


@dataclass
class BankPage:
    """One page of bank rows, plus how many banks matched before paging."""

    banks: list[dict]
    total: int


async def list_banks_page(pool, *, limit: int, offset: int, search_query: str | None = None) -> BankPage:
    """One page of banks, most recently written first, WITHOUT ranking every bank to find it.

    :func:`list_banks` reads the whole tenant because the order depends on a value only the store
    has: it asks for every bank's write time, sorts, and throws away all but a page. That is
    O(total banks) per request — 60-75 s on a tenant with 27,315 banks, and the endpoint's entire
    cost. The work is not the sorting, it is fetching N sort keys to produce 100 rows.

    A store that can hand back its banks ALREADY ordered removes the need to. The page is then a
    merge of two streams that are each already sorted:

      * the store's banks, newest-written first, cursor-paged;
      * this tenant's banks by ``created_at``, newest first, off a SQL index.

    A bank the store has never recorded (never written, or written but not yet folded) comes from
    the second stream and keeps the same fallback key :func:`list_banks` gives it — its creation
    time — so the two agree about where such a bank belongs. Merging two sorted streams costs the
    page, not the tenant.

    That agreement assumes a tenant's memories all live in one place: whichever store answers here
    owns every bank in it. A tenant with BOTH store-owned and Postgres-owned banks would break it,
    because a Postgres-owned bank is absent from the store stream and would then be placed by its
    creation time while :func:`list_banks` places it by its newest document or fact write. No
    deployment mixes the two; a deployment that starts to must key the creation stream off those
    same two columns rather than off `created_at`.

    Falls back to :func:`list_banks` whenever the merge cannot be trusted to page correctly:

      * a store that returns no ordering of its own. Then the write times are SQL columns, the
        ordering is already one indexed query, and there is nothing a store stream could add — a
        merge against an empty stream would silently reorder such a tenant by CREATION time;
      * a search, which matches on ``bank_id`` and ``name``. Those live only in SQL, so the store
        cannot apply it and its stream would be the wrong set. A search narrows to few rows anyway,
        which is where ranking them all is cheap.

    The first of those is decided by ASKING the store for a page, not by reading a capability flag
    off it. ``store_owned`` is the wrong question and answers it wrongly: a router whose banks live
    in different backends declares ``store_owned = False`` at the class level and answers per bank
    through ``store_owned_for``, so a flag check sends an entirely store-owned tenant down the
    ranking path — correct output, and the whole cost this exists to remove. What the merge actually
    needs to know is whether there is an ordering to merge, and the page it is about to consume is
    the answer.

    ``total`` counts the banks matching the search, before paging.
    """
    from ..memories import get_memories

    store = get_memories()

    async def _ranked() -> BankPage:
        banks = await list_banks(pool, search_query=search_query)
        return BankPage(banks=banks[offset : offset + limit], total=len(banks))

    if search_query:
        return await _ranked()

    # How many rows the merge must produce before it can cut the page. Offset paging over an
    # ordered walk necessarily reaches the offset — but it reaches it by walking the ORDER, not by
    # ranking the tenant, so page 1 (the page that is actually requested) costs one page.
    want = offset + limit
    if want <= 0:
        return BankPage(banks=[], total=await _count_banks(pool))

    try:
        # The store's first page decides the path AND seeds the merge, so asking costs nothing
        # extra: an empty page means this store has no ordering to contribute.
        first = await store.list_banks_by_write(limit=_STORE_PAGE)
        if not first.banks:
            return await _ranked()
        ordered_ids = await _merge_ordered_bank_ids(pool, store, want, first)
    except Exception as e:  # noqa: BLE001 - the list must render even if the ordering cannot
        logger.warning(f"Could not page banks by write time ({e}); ranking the tenant instead")
        return await _ranked()

    rows = await _bank_rows(pool, ordered_ids[offset : offset + limit])
    # The store's own write and ingestion times, for the PAGE — the live read is what each row
    # SHOWS, while the ORDER above came from the store's catalog (as of that bank's last fold), so
    # a just-written bank shows the right time while sitting one slot low until it folds.
    #
    # Only on this path. :func:`list_banks` already fills these for every bank it ranks, because it
    # SORTS by them, so overlaying them again for the callers that fall back would be a second
    # round trip to the store for an answer already in hand.
    await _apply_store_last_write(rows, {})
    return BankPage(banks=rows, total=await _count_banks(pool))


#: Banks per store page. The merge consumes one page for a page-1 request however large the tenant.
_STORE_PAGE = 100


async def _merge_ordered_bank_ids(pool, store, want: int, first: "BankWritePage") -> list[str]:
    """The first `want` bank ids, newest-written first, merging the store's order with SQL's.

    Both inputs are already sorted, so this takes whichever head is newer and never sorts anything.
    A bank the store names is emitted at its WRITE time; the same bank reached later through the
    creation-ordered stream is skipped, because it has already been placed at the better key.

    That skip relies on a write never predating its bank's creation, which is what makes the store
    stream win the race for any bank it knows. A bank the store does NOT know is placed by creation
    time — the same fallback :func:`list_banks` uses, so the two orders agree. The one shape that
    breaks the invariant is an import or transfer, which stamps `created_at` now while the store
    carries the source's older write time: such a bank sorts by its arrival here and by its original
    write on the ranking path.

    Closed explicitly: the merge stops as soon as the page is full, leaving the store stream
    suspended at a `yield`, and an async generator finalised by the garbage collector instead
    complains about it at shutdown.
    """
    created = await _banks_by_created(pool, want)

    out: list[str] = []
    seen: set[str] = set()
    async with aclosing(_store_write_stream(store, first)) as store_pages:
        s_head = await anext(store_pages, None)
        c_idx = 0
        while len(out) < want:
            c_head = created[c_idx] if c_idx < len(created) else None
            # Whichever head is newer, with the store winning a tie — a bank it knows is placed at
            # its write time, which is the better key. Each branch tests its own head for None
            # rather than breaking out first, so the types narrow and a head is never read after it
            # has run out.
            if s_head is not None and (c_head is None or s_head.last_write_at >= c_head.last_write_at):
                bank_id = s_head.bank_id
                s_head = await anext(store_pages, None)
            elif c_head is not None:
                bank_id = c_head.bank_id
                c_idx += 1
            else:
                break
            if bank_id in seen:
                continue
            seen.add(bank_id)
            out.append(bank_id)
    return out


async def _store_write_stream(store, first: "BankWritePage") -> "AsyncIterator[BankWriteTime]":
    """The store's banks as one continuous newest-written-first stream, page by page.

    A generator rather than a list so the merge pulls only the pages it needs: a page-1 request
    consumes one store page however many banks the tenant has.

    The first page is handed in because the caller already fetched it to decide whether this store
    orders at all; re-fetching it here would pay for that decision twice.
    """
    from ..memories.base import BankWriteTime

    page = first
    while True:
        for bank in page.banks:
            yield BankWriteTime(bank_id=bank.bank_id, last_write_at=_as_utc(bank.last_write_at))
        if not page.next_page_token:
            return
        page = await store.list_banks_by_write(limit=_STORE_PAGE, page_token=page.next_page_token)


async def _banks_by_created(pool, want: int) -> "list[BankWriteTime]":
    """The `want` most recently CREATED banks, newest first — the other half of the merge.

    Bounded by `want`: this is the stream that places banks the store has no write time for, and
    only the head of it can reach the page.
    """
    from ..memories.base import BankWriteTime

    banks_table = fq_table("banks")
    async with acquire_with_retry(pool) as conn:
        rows = await conn.fetch(
            # `bank_id` breaks ties: banks created in one batch share a `created_at`, and without a
            # tiebreaker Postgres may order them differently for the page-1 LIMIT than for the
            # page-2 one, which shows up as a bank on both pages or on neither.
            f"SELECT bank_id, created_at FROM {banks_table} ORDER BY created_at DESC NULLS LAST, bank_id DESC LIMIT $1",
            want,
        )
    return [BankWriteTime(bank_id=r["bank_id"], last_write_at=_as_utc(r["created_at"]) or _UNIX_EPOCH) for r in rows]


async def _count_banks(pool) -> int:
    banks_table = fq_table("banks")
    async with acquire_with_retry(pool) as conn:
        row = await conn.fetchrow(f"SELECT COUNT(*) AS n FROM {banks_table}")
    return int(row["n"]) if row else 0


async def _bank_rows(pool, bank_ids: "list[str]") -> list:
    """The page's rows, in the order `bank_ids` gives — one query, by id.

    This is the join that FILLS a page rather than deciding one, which is the whole difference: it
    reads the rows that will be shown, not every row in the tenant. `fact_count` is left at 0 and
    filled by the page overlays, exactly as :func:`list_banks` leaves it.

    **`memory_units` is not named at all unless some bank on the page keeps its memories there.**
    That is about PLANNING, not rows. Each bank gets its own partial HNSW indexes on that table, so
    a tenant with 27,315 banks has **82,795 indexes** on it; the planner considers them for any
    query that mentions the table, whatever the query does. Measured on such a tenant, against a
    `memory_units` holding **zero** rows: planning 975 ms, execution 0.05 ms — and `_bank_rows`
    naming it twice cost **3.9 s of a 4.5 s** page.

    So a guard on the COLUMN does not help, and neither does an empty id array: both still mention
    the table. The join has to be absent from the SQL. When the page is entirely store-owned — the
    normal case for such a tenant — it is, and the fact watermark is NULL for every row, which is
    what it would have been anyway.

    The term itself stays for banks that do keep memories in SQL: without it a fact-only write does
    not move `last_write_at`, and the two list paths disagree.
    """
    if not bank_ids:
        return []
    from ..memories import get_memories

    store = get_memories()
    sql_owned = [b for b in bank_ids if not store.store_owned_for(b)]
    banks_table = fq_table("banks")
    docs_table = fq_table("documents")
    if sql_owned:
        mu_table = fq_table("memory_units")
        fact_select = "f.last_fact_at"
        fact_join = f"""
            LEFT JOIN (
                SELECT bank_id, MAX(updated_at) AS last_fact_at
                FROM {mu_table}
                WHERE bank_id = ANY($2::text[])
                GROUP BY bank_id
            ) f ON f.bank_id = b.bank_id"""
        params = (bank_ids, sql_owned)
    else:
        # Not `NULL::timestamptz` off a join that is simply empty — the table must not appear.
        fact_select = "NULL::timestamptz AS last_fact_at"
        fact_join = ""
        params = (bank_ids,)
    async with acquire_with_retry(pool) as conn:
        rows = await conn.fetch(
            f"""
            SELECT b.bank_id, b.name, b.disposition, b.mission, b.created_at, b.updated_at,
                   d.last_document_at, d.last_document_write_at,
                   {fact_select}
            FROM {banks_table} b
            LEFT JOIN (
                SELECT bank_id,
                       MAX(created_at) AS last_document_at,
                       MAX(updated_at) AS last_document_write_at
                FROM {docs_table}
                WHERE bank_id = ANY($1::text[])
                GROUP BY bank_id
            ) d ON d.bank_id = b.bank_id{fact_join}
            WHERE b.bank_id = ANY($1::text[])
            """,
            *params,
        )
    by_id = {}
    for row in rows:
        disposition_data = row["disposition"]
        if isinstance(disposition_data, str):
            disposition_data = json.loads(disposition_data)
        created_at = _as_utc(row["created_at"])
        updated_at = _as_utc(row["updated_at"])
        last_doc = _as_utc(row["last_document_at"])
        write_times = [t for t in (_as_utc(row["last_document_write_at"]), _as_utc(row["last_fact_at"])) if t]
        last_write = max(write_times) if write_times else None
        by_id[row["bank_id"]] = {
            "bank_id": row["bank_id"],
            "name": row["name"],
            "disposition": disposition_data,
            "mission": row["mission"] or "",
            "created_at": created_at.isoformat() if created_at else None,
            "updated_at": updated_at.isoformat() if updated_at else None,
            "fact_count": 0,
            "last_document_at": last_doc.isoformat() if last_doc else None,
            "last_write_at": last_write.isoformat() if last_write else None,
        }
    # In the store's order, and only rows that still exist: a bank deleted between the ordering and
    # this read drops out rather than surfacing as a blank row.
    return [by_id[b] for b in bank_ids if b in by_id]


async def _apply_store_last_write(banks: list[dict], sort_keys: "dict[str, datetime]") -> None:
    """Fill `last_write_at` (and the sort key) from the store, for banks SQL cannot answer for.

    One batched call for all of them. A bank the store has no time for is left exactly as it was —
    absent means "unknown", never "the epoch", so an un-folded bank keeps whatever ordering it had
    rather than being pushed to the bottom on a guess.
    """
    from ..memories import get_memories

    store = get_memories()
    external = [b for b in banks if store.store_owned_for(b["bank_id"])]
    if not external:
        return
    ids = [b["bank_id"] for b in external]
    # Asked for separately, because they are different questions: `last_document_at` is INGESTION
    # time and must not move when an existing document is re-retained, while `last_write_at` must.
    # Reporting one under both names looked harmless and is not — it makes a rewrite read as a new
    # document.
    #
    # Gathered, not awaited in sequence: they are independent round trips to the same service, and
    # `return_exceptions` keeps one failing from discarding the other's answer — a list that sorts
    # correctly but shows no ingestion time is better than one that does neither.
    # The try wraps the gather because building its arguments already calls into the store: a store
    # that does not implement these raises AttributeError right here, before `return_exceptions` can
    # see anything. They are optional — the interface defaults return `{}` — so a store without them
    # must leave the list rendering, not break it.
    try:
        times_r, doc_times_r = await asyncio.gather(
            store.last_write_at_many(bank_ids=ids),
            store.last_document_at_many(bank_ids=ids),
            return_exceptions=True,
        )
    except Exception as e:  # noqa: BLE001 - ordering is a nicety; the list itself must still render
        logger.warning(f"Store cannot report write times for {len(external)} bank(s): {e}")
        return
    if isinstance(times_r, BaseException):
        logger.warning(f"Could not read last_write_at from the store for {len(external)} bank(s): {times_r}")
        times = {}
    else:
        times = times_r
    if isinstance(doc_times_r, BaseException):
        logger.warning(f"Could not read last_document_at from the store for {len(external)} bank(s): {doc_times_r}")
        doc_times = {}
    else:
        doc_times = doc_times_r
    if not times and not doc_times:
        return
    for bank in external:
        doc_when = doc_times.get(bank["bank_id"])
        if doc_when is not None:
            bank["last_document_at"] = doc_when.isoformat()
    for bank in external:
        when = times.get(bank["bank_id"])
        if when is None:
            continue
        bank["last_write_at"] = when.isoformat()
        sort_keys[bank["bank_id"]] = when


async def apply_store_fact_counts(banks: list[dict]) -> None:
    """Replace ``fact_count`` in-place for banks that keep their memories outside SQL.

    Those banks leave the ``memory_units`` join empty, so the count has to come from the store.
    Still page-scoped rather than system-wide, but ONE batched call for that page instead of a
    round trip per bank: asking per bank made the page cost a network hop each, and on dev a
    108-bank page took ~8s end-to-end, almost all of it those hops — most against banks with
    nothing in them. :meth:`count_memories_many` is the same question asked once.

    ``strong=True``, deliberately. The per-bank call this replaces reads the un-folded tail, so
    anything weaker would fold a change in what a just-written bank REPORTS into what was meant to
    be a change in how long it takes. The batched read applies the tail without opening a snapshot,
    so a page of N banks still cannot admit N banks and evict whatever was warm.

    A bank the store has no count for reports zero, per the interface's contract that absent means
    nothing to count — so one bank cannot fail the page.

    No connection is held across this: it is a network call to another service and nothing here
    needs one. Holding a pooled connection across the per-bank loop was the other half of the cost
    under load, and is the rule :func:`_apply_store_last_write` already follows.
    """
    from ..memories import get_memories

    store = get_memories()
    external = [bank for bank in banks if store.store_owned_for(bank["bank_id"])]
    if not external:
        return

    counts = await store.count_memories_many(bank_ids=[bank["bank_id"] for bank in external], strong=True)
    for bank in external:
        bank["fact_count"] = sum(counts.get(bank["bank_id"], {}).values())


async def apply_sql_fact_counts(pool, banks: list[dict]) -> None:
    """Fill ``fact_count`` from ``memory_units``, for the page of banks actually returned.

    :func:`list_banks` used to count every bank in one GROUP BY over the whole table, so
    listing 365 banks aggregated millions of rows whatever the page size (#4468). Scoped to
    the page the count is a bounded index range per bank instead.

    Banks whose memories live outside SQL are NOT asked about at all. They have no rows here, so
    the count would be 0, and :func:`apply_store_fact_counts` overwrites it a moment later with the
    store's live number — the query is pure waste for them, and not a cheap one: counting 100
    store-owned banks against a `memory_units` holding millions of another tenant's rows measured
    **12.8 s** of a 13.5 s bank list, for a hundred zeroes that were then discarded.

    A mixed page is the reason this is per bank rather than an early return: a tenant can have both
    kinds, and the SQL-owned ones still need their count from here.
    """
    if not banks:
        return
    from ..memories import get_memories

    store = get_memories()
    # Zeroed up front so a store-owned bank still carries a number if the store cannot be reached
    # for its real one — which is what this function left behind for them before it stopped asking.
    sql_owned = set()
    for bank in banks:
        bank["fact_count"] = 0
        if not store.store_owned_for(bank["bank_id"]):
            sql_owned.add(bank["bank_id"])
    if not sql_owned:
        return
    async with acquire_with_retry(pool) as conn:
        rows = await conn.fetch(
            f"""
            SELECT bank_id, COUNT(*) AS fact_count
            FROM {fq_table("memory_units")}
            WHERE bank_id = ANY($1)
            GROUP BY bank_id
            """,
            list(sql_owned),
        )
    counts = {row["bank_id"]: row["fact_count"] for row in rows}
    # Only the SQL-owned ones: a store-owned bank keeps the zero above for `apply_store_fact_counts`
    # to replace, rather than being re-zeroed here by a query that never asked about it.
    for bank in banks:
        if bank["bank_id"] in sql_owned:
            bank["fact_count"] = counts.get(bank["bank_id"], 0)
