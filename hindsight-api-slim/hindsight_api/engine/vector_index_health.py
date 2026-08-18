"""Per-bank vector index coverage: what a bank should have, and making it so.

A (bank, fact_type) partition gets its own partial vector index once it holds
``HINDSIGHT_API_VECTOR_INDEX_MIN_ROWS`` rows. At the default of 0 that is every
partition holding any rows — the behaviour before the threshold existed — and a
deployment with thousands of banks raises it, because these indexes live on the
shared ``memory_units`` table: PostgreSQL locks and plans against every index on
a relation, and opens every one for each DML statement, so one bank's index is
charged to every other bank's queries. Three per bank exhausts the lock table at
a few thousand banks (issue #3485). Below the threshold the planner answers the
same query from the ``(bank_id, fact_type)`` B-tree plus a top-N sort, which is
exact rather than approximate and faster.

Nothing here runs on a request path. Index DDL is issued only by the
``vector_index_maintenance`` async operation (submitted after a write that could
have changed a bank's coverage) and by the ``repair-bank`` admin command, both
of which reconcile a bank against the plan this module computes.

All DDL is ``CREATE/DROP INDEX CONCURRENTLY`` on a raw autocommit connection, so
it never takes ``ACCESS EXCLUSIVE`` on the shared table. That is also what keeps
the drop path usable on an instance that has already hit the #3485 wall:
``DROP INDEX`` is a utility statement that locks its own index plus the table,
rather than planning against all of the table's indexes the way any DML must.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .._vector_index import qualifies_for_per_bank_index, should_keep_per_bank_index
from .db_utils import retry_with_backoff
from .retain.bank_utils import _BANK_INDEX_FACT_TYPES, _bank_index_name

logger = logging.getLogger(__name__)

# Postgres renders the partial predicate of an indexdef with parenthesized
# comparison operands and an explicit ::text cast, e.g.
# `... WHERE ((fact_type = 'world'::text) AND (bank_id = 'b1'::text))`.
# fact_type is emitted first (it is written first in the CREATE INDEX). Match
# that exact rendering so a mere name collision never counts as healthy.
_BANK_INDEX_PARTIAL_SUFFIX = " WHERE ((fact_type = "

# Access methods that legitimately back a per-(bank, fact_type) partial index.
# An index whose access method drifted after a backend switch does not match,
# so the health check treats it as unhealthy (rebuild).
_SUPPORTED_INDEX_AM: tuple[str, ...] = (
    "btree",
    "gin",
    "gist",
    "hnsw",
    "ivfflat",
    "diskann",
    "vchordrq",
)


@dataclass
class BankIndexPlan:
    """What one bank's vector-index coverage should become.

    Computed without issuing any DDL so the same plan can answer two questions:
    "is there anything to do?" (the cheap pre-check that keeps every write from
    queueing an empty operation) and "what exactly?" (the operation itself).
    """

    bank_id: str
    # fact_types at or above the build threshold whose index is missing or unhealthy.
    to_build: list[str] = field(default_factory=list)
    # Index names present in the catalog that this bank should no longer carry.
    to_drop: list[str] = field(default_factory=list)
    # Indexes already present and healthy — reported, never touched.
    already_present: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.to_build and not self.to_drop


@dataclass
class BankIndexResult:
    """Outcome of applying a :class:`BankIndexPlan`."""

    bank_id: str
    created: int = 0
    dropped: int = 0
    already_present: int = 0
    # Would-create / would-drop, reported under dry_run.
    skipped: int = 0
    would_drop: int = 0
    failed: int = 0
    failed_indexes: list[str] = field(default_factory=list)


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


async def _index_health(conn: Any, schema: str, index_names: list[str]) -> dict[str, bool]:
    """Return valid-and-usable state for each requested index in one query.

    Health requires the index to be valid AND ready, defined over the expected
    ``memory_units`` table, to use a supported access method, and to carry our
    partial predicate. A name-only match is *not* enough: an INVALID leftover
    (from an interrupted concurrent build) or an index whose access method
    drifted after a backend switch must count as unhealthy so it is rebuilt —
    ``pg_indexes``/``IF NOT EXISTS`` alone would silently treat those as present.
    """
    if not index_names:
        return {}
    rows = await conn.fetch(
        """
        SELECT c.relname AS index_name,
               (i.indisvalid AND i.indisready
                AND t.relname = 'memory_units'
                AND am.amname = ANY($3::text[])
                AND pg_get_indexdef(i.indexrelid) LIKE $4
               ) AS healthy
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_index i ON i.indexrelid = c.oid
        JOIN pg_class t ON t.oid = i.indrelid
        JOIN pg_am am ON am.oid = c.relam
        WHERE n.nspname = $1 AND c.relname = ANY($2::text[])
        """,
        schema,
        index_names,
        list(_SUPPORTED_INDEX_AM),
        "%" + _BANK_INDEX_PARTIAL_SUFFIX + "%",
    )
    return {row["index_name"]: bool(row["healthy"]) for row in rows}


async def plan_bank_vector_indexes(conn: Any, schema: str, bank_id: str) -> BankIndexPlan:
    """Work out what ``bank_id``'s vector-index coverage should become.

    Two cheap, bank-scoped queries plus one catalog lookup, so the write path
    can call this on every write to decide whether an operation is worth
    queueing at all. The row count is an index-only scan of
    ``idx_memory_units_bank_fact_type``; deliberately unfiltered by
    ``embedding IS NOT NULL``, since that predicate is not in the index and
    would turn the scan into a heap read without changing a threshold decision
    by enough to matter.

    A bank whose row is gone yields an empty plan: its indexes are dropped by
    ``delete_bank`` while the internal_id they are named after is still known,
    and a bank-scoped reconcile has no way to name them afterwards.
    """
    plan = BankIndexPlan(bank_id=bank_id)
    qschema = _quote_identifier(schema)

    internal_id = await conn.fetchval(
        f"SELECT internal_id FROM {qschema}.banks WHERE bank_id = $1",  # noqa: S608 — schema is a quoted identifier
        bank_id,
    )
    if internal_id is None:
        return plan

    counts = {
        row["fact_type"]: int(row["row_count"])
        for row in await conn.fetch(
            f"""
            SELECT fact_type, COUNT(*) AS row_count
            FROM {qschema}.memory_units
            WHERE bank_id = $1 AND fact_type = ANY($2::text[])
            GROUP BY fact_type
            """,  # noqa: S608 — schema is a quoted identifier
            bank_id,
            list(_BANK_INDEX_FACT_TYPES),
        )
    }

    names = {ft: _bank_index_name(ft, str(internal_id)) for ft in _BANK_INDEX_FACT_TYPES}
    health = await _index_health(conn, schema, list(names.values()))

    for fact_type, index_name in names.items():
        row_count = counts.get(fact_type, 0)
        healthy = health.get(index_name)
        if qualifies_for_per_bank_index(row_count):
            if healthy is True:
                plan.already_present += 1
            else:
                plan.to_build.append(fact_type)
        elif healthy is not None and not should_keep_per_bank_index(row_count):
            # Present but no longer earned. Keeping has its own, lower bound than
            # building (see should_keep_per_bank_index) so a partition hovering
            # at the threshold does not rebuild and drop the same ANN index on
            # alternating writes.
            plan.to_drop.append(index_name)

    return plan


async def apply_bank_index_plan(
    conn: Any,
    schema: str,
    index_clause: str,
    plan: BankIndexPlan,
    *,
    dry_run: bool = False,
) -> BankIndexResult:
    """Build and drop what ``plan`` calls for, on a raw autocommit connection.

    ``conn`` must not be inside a transaction: ``CREATE INDEX CONCURRENTLY``
    cannot run in one, and both it and ``DROP INDEX CONCURRENTLY`` need a real
    backend session for the whole statement (a transaction-pooled URL will not
    do — that is what ``HINDSIGHT_API_MIGRATION_DATABASE_URL`` is for).

    Concurrency is handled by idempotency, not a lock: the project forbids
    advisory locks, which are unreliable behind connection poolers, and leaning
    on one is why #2803's version of this was rejected. Every build is
    ``CREATE INDEX CONCURRENTLY IF NOT EXISTS`` guarded by a valid/ready health
    check and every drop is ``DROP INDEX CONCURRENTLY IF EXISTS``, so a second
    concurrent run is a no-op on work the first already did.
    """
    result = BankIndexResult(bank_id=plan.bank_id, already_present=plan.already_present)
    qschema = _quote_identifier(schema)

    for index_name in plan.to_drop:
        if dry_run:
            result.would_drop += 1
            continue
        qualified = f"{qschema}.{_quote_identifier(index_name)}"
        try:
            await retry_with_backoff(
                lambda qualified=qualified: conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {qualified}")
            )
            result.dropped += 1
        except Exception as exc:  # noqa: BLE001 — one failed drop must not abort the rest
            result.failed += 1
            result.failed_indexes.append(qualified)
            logger.warning("Failed to drop stale vector index %s: %s", qualified, exc)

    if not plan.to_build:
        return result

    # Render the bank_id literal server-side so escaping does not depend on
    # standard_conforming_strings (the predicate is inlined into the DDL).
    bank_id_literal = await conn.fetchval("SELECT quote_literal($1::text)", plan.bank_id)
    internal_id = await conn.fetchval(
        f"SELECT internal_id FROM {qschema}.banks WHERE bank_id = $1",  # noqa: S608 — quoted identifier
        plan.bank_id,
    )
    if internal_id is None:
        # The bank was deleted between planning and applying; delete_bank has
        # already dropped its indexes and there is nothing left to name.
        return result

    for fact_type in plan.to_build:
        if dry_run:
            result.skipped += 1
            continue
        qindex = _quote_identifier(_bank_index_name(fact_type, str(internal_id)))
        qualified = f"{qschema}.{qindex}"

        async def _rebuild(qindex: str = qindex, qualified: str = qualified, fact_type: str = fact_type) -> None:
            # Always drop first. An unhealthy-but-present index (INVALID
            # leftover, wrong access method) can't be repaired by IF NOT EXISTS,
            # and a prior deadlocked CONCURRENTLY build leaves an INVALID stub
            # that IF NOT EXISTS would likewise skip — so a retry must clear it.
            # DROP ... IF EXISTS is a no-op when the index is simply absent.
            await conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {qualified}")
            await conn.execute(
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {qindex} "
                f"ON {qschema}.memory_units {index_clause} "
                f"WHERE fact_type = '{fact_type}' AND bank_id = {bank_id_literal}"
            )

        try:
            # CREATE INDEX CONCURRENTLY on the live, concurrently-written
            # memory_units table can be chosen as a deadlock victim (40P01).
            # That is transient — Postgres aborts one side to break the cycle —
            # so retry the drop+build before recording a permanent failure.
            await retry_with_backoff(_rebuild)
            result.created += 1
            logger.info("Built vector index %s (bank=%s, fact_type=%s)", qualified, plan.bank_id, fact_type)
        except Exception as exc:  # noqa: BLE001 — one failed index must not abort the rest
            result.failed += 1
            result.failed_indexes.append(qualified)
            logger.warning(
                "Failed to build vector index %s (bank=%s, fact_type=%s): %s — "
                "dropping the invalid leftover so a re-run can retry.",
                qualified,
                plan.bank_id,
                fact_type,
                exc,
            )
            # A failed concurrent build leaves an INVALID index behind that
            # would shadow the good one; drop it so a re-run retries cleanly.
            try:
                await conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {qualified}")
            except Exception as cleanup_exc:  # noqa: BLE001
                logger.warning("Cleanup DROP INDEX for %s also failed: %s", qualified, cleanup_exc)

    return result


async def reconcile_bank_vector_indexes(
    conn: Any,
    schema: str,
    bank_id: str,
    index_clause: str,
    *,
    dry_run: bool = False,
) -> BankIndexResult:
    """Plan and apply one bank's vector-index coverage."""
    plan = await plan_bank_vector_indexes(conn, schema, bank_id)
    return await apply_bank_index_plan(conn, schema, index_clause, plan, dry_run=dry_run)


async def list_bank_ids(conn: Any, schema: str) -> list[str]:
    """Every bank in ``schema``, for the admin command's ``--all`` mode."""
    rows = await conn.fetch(
        f"SELECT bank_id FROM {_quote_identifier(schema)}.banks ORDER BY bank_id"  # noqa: S608 — quoted identifier
    )
    return [row["bank_id"] for row in rows]


async def drop_orphaned_bank_indexes(conn: Any, schema: str, *, dry_run: bool = False) -> list[str]:
    """Drop per-bank vector indexes whose bank no longer exists.

    ``delete_bank`` drops a bank's indexes while the ``internal_id`` they are
    named after is still known, so this should find nothing. It exists for when
    that did not happen: a deployment that hit the #3485 wall could not run
    ``delete_bank`` at all (the delete DML could not plan), so operators dropped
    banks by other means and left the indexes behind — and an orphan is
    unreachable by every bank-scoped path, because there is no bank row to plan
    from.

    Catalog-only, matching each index's name suffix against the live
    ``internal_id`` set, so it answers even on an instance whose lock table is
    exhausted. Only the admin command calls it; the write path has no reason to.
    """
    qschema = _quote_identifier(schema)
    live = {
        str(row["internal_id"]).replace("-", "")[:16]
        for row in await conn.fetch(f"SELECT internal_id FROM {qschema}.banks")  # noqa: S608 — quoted identifier
    }
    rows = await conn.fetch(
        """
        SELECT c.relname AS index_name
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_index i ON i.indexrelid = c.oid
        JOIN pg_class t ON t.oid = i.indrelid
        WHERE n.nspname = $1
          AND t.relname = 'memory_units'
          AND c.relname LIKE 'idx\\_mu\\_emb\\_%'
        """,
        schema,
    )

    orphans = [row["index_name"] for row in rows if row["index_name"].rsplit("_", 1)[-1] not in live]
    if dry_run:
        return orphans

    dropped = []
    for index_name in orphans:
        qualified = f"{qschema}.{_quote_identifier(index_name)}"
        try:
            await retry_with_backoff(
                lambda qualified=qualified: conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {qualified}")
            )
            dropped.append(index_name)
            logger.info("Dropped orphaned vector index %s (no matching bank)", qualified)
        except Exception as exc:  # noqa: BLE001 — one failure must not abort the rest
            logger.warning("Failed to drop orphaned vector index %s: %s", qualified, exc)
    return dropped
