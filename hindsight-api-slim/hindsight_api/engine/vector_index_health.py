"""Per-bank vector index coverage: what a bank should have, and making it so.

``HINDSIGHT_API_VECTOR_INDEX_MIN_ROWS`` decides how coverage is reached, and its
default of ``0`` means nothing in this module runs at all. At ``0`` the three
partial indexes are built in the bank-create transaction and dropped when the
bank is deleted — the behaviour that predates the threshold, and still the right
one for a deployment whose bank count is not the problem.

A deployment holding thousands of banks sets a positive threshold, because these
indexes live on the shared ``memory_units`` table: PostgreSQL locks and plans
against every index on a relation, and opens every one for each DML statement, so
one bank's index is charged to every other bank's queries. Three per bank
exhausts the lock table at a few thousand banks (issue #3485). Above the
threshold a partition earns its own index; below it the planner answers the same
query from the ``(bank_id, fact_type)`` B-tree plus a top-N sort, which is exact
rather than approximate and faster.

With a threshold set, coverage is reconciled by the ``vector_index_maintenance``
async operation (submitted after a write that could have changed it) and by the
``repair-bank`` admin command. Neither runs on a request path. The write path
pays only :func:`plan_bank_vector_indexes`, which is kept cheap two ways: the
:class:`CoverageTrigger` direction settles most partitions from the catalog
without counting anything, and what is left is counted no further than the
threshold rather than exactly.

All DDL is ``CREATE/DROP INDEX CONCURRENTLY`` on a raw autocommit connection, so
it never takes ``ACCESS EXCLUSIVE`` on the shared table. That is also what keeps
the drop path usable on an instance that has already hit the #3485 wall:
``DROP INDEX`` is a utility statement that locks its own index plus the table,
rather than planning against all of the table's indexes the way any DML must.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .._vector_index import (
    per_bank_index_build_bound,
    per_bank_index_keep_bound,
    per_bank_indexes_are_eager,
)
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


class CoverageTrigger(Enum):
    """Which way the write that queued this reconcile moved the bank's row count.

    The direction is what makes the pre-check affordable on every write. Coverage
    can only become wrong in one direction at a time, so most partitions can be
    settled from the catalog alone and never counted at all:

    * ``GREW`` — rows were added. A partition that already has a healthy index
      keeps earning it, because growth cannot take it below the keep bound.
      Only an *unindexed* partition needs counting, to see if it crossed the
      build bound.
    * ``SHRANK`` — rows were removed. A partition with no index cannot have
      earned one, because shrinking cannot take it above the build bound. Only
      an *indexed* partition needs counting, to see if it fell below the keep
      bound.
    * ``FULL`` — count every partition. Used by the maintenance job when it
      re-plans at start and by ``repair-bank``, neither of which knows what
      moved, and which are the paths that recover anything the directional
      short-circuits deferred (a build that failed, an index left over a
      partition that shrank while nothing indexed was written).
    """

    GREW = "grew"
    SHRANK = "shrank"
    FULL = "full"


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


async def _capped_row_counts(
    conn: Any,
    schema: str,
    bank_id: str,
    fact_types: list[str],
    cap: int,
) -> dict[str, int]:
    """Rows each partition holds, counted no further than ``cap``.

    ``min(actual, cap)`` is all the policy needs: with ``cap`` set to the build
    bound, one value answers both "does this reach the build bound?" and "is it
    still at the keep bound?", since the keep bound is the lower of the two.

    Counting to a cap rather than exactly is the point. The honest ``COUNT(*)``
    this replaces was an index-only scan of every row the bank owned, run on
    every retain, import, consolidation and delete, forever — on a large bank
    that is hundreds of thousands of index tuples to rediscover that nothing
    changed. Capped, the scan stops at the threshold, so its cost is set by the
    configured bound instead of by how big the bank got.

    The cap is a query parameter rather than an outer-column reference on
    purpose: PostgreSQL rejects a LIMIT/OFFSET expression containing a variable
    from an outer query level, and the bounds are global anyway, not per-fact-type.
    """
    if not fact_types:
        return {}
    qschema = _quote_identifier(schema)
    rows = await conn.fetch(
        f"""
        SELECT t.fact_type,
               (
                   SELECT count(*) FROM (
                       SELECT 1 FROM {qschema}.memory_units
                       WHERE bank_id = $1 AND fact_type = t.fact_type
                       LIMIT $2
                   ) capped
               ) AS row_count
        FROM unnest($3::text[]) AS t(fact_type)
        """,  # noqa: S608 — schema is a quoted identifier
        bank_id,
        cap,
        fact_types,
    )
    return {row["fact_type"]: int(row["row_count"]) for row in rows}


async def plan_bank_vector_indexes(
    conn: Any,
    schema: str,
    bank_id: str,
    *,
    trigger: CoverageTrigger = CoverageTrigger.FULL,
) -> BankIndexPlan:
    """Work out what ``bank_id``'s vector-index coverage should become.

    One catalog lookup, plus a capped count for only the partitions ``trigger``
    says could have changed — often none, in which case the write path pays a
    single indexed SELECT and one catalog query to decide there is nothing to do.
    See :class:`CoverageTrigger` for which partitions each direction can settle
    without counting.

    With the threshold off, entitlement does not depend on rows at all: every
    partition is owed an index from the moment the bank exists, so this reports
    whatever bank creation did not manage to leave healthy and never drops
    anything. The write path does not reach here in that mode (it short-circuits
    before querying), but ``repair-bank`` does, and it is the path that repairs a
    bank whose creation lost its DDL to a deadlock, or that was restored around
    it.

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

    names = {ft: _bank_index_name(ft, str(internal_id)) for ft in _BANK_INDEX_FACT_TYPES}
    health = await _index_health(conn, schema, list(names.values()))

    if per_bank_indexes_are_eager():
        for fact_type, index_name in names.items():
            if health.get(index_name) is True:
                plan.already_present += 1
            else:
                plan.to_build.append(fact_type)
        return plan

    # Settle what the catalog alone can, and collect the rest for one round trip.
    to_count: list[str] = []
    for fact_type, index_name in names.items():
        healthy = health.get(index_name)
        if trigger is CoverageTrigger.GREW and healthy is True:
            plan.already_present += 1
            continue
        if trigger is CoverageTrigger.SHRANK and healthy is None:
            continue
        to_count.append(fact_type)

    if not to_count:
        return plan

    build_bound = per_bank_index_build_bound()
    keep_bound = per_bank_index_keep_bound()
    counts = await _capped_row_counts(conn, schema, bank_id, to_count, build_bound)

    for fact_type in to_count:
        index_name = names[fact_type]
        healthy = health.get(index_name)
        row_count = counts.get(fact_type, 0)
        if row_count >= build_bound:
            if healthy is True:
                plan.already_present += 1
            else:
                plan.to_build.append(fact_type)
        elif healthy is not None and row_count < keep_bound:
            # Present but no longer earned. Keeping has its own, lower bound than
            # building (see per_bank_index_keep_bound) so a partition hovering at
            # the threshold does not rebuild and drop the same ANN index on
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

    After each successful build the bank row is re-checked: a bank deleted (or
    recreated under a fresh ``internal_id``) mid-build leaves an orphan index
    ``delete_bank`` cannot name, so the build drops its own index instead.
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
    internal_id_str = str(internal_id)

    for fact_type in plan.to_build:
        if dry_run:
            result.skipped += 1
            continue
        qindex = _quote_identifier(_bank_index_name(fact_type, internal_id_str))
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
        else:
            # Re-check the bank after the build: if it was deleted (or
            # recreated under a fresh internal_id) while CREATE INDEX
            # CONCURRENTLY was running, the index just built is an orphan that
            # delete_bank's post-commit drop could not name. Drop it, don't
            # count it, and stop — the remaining fact_types would only build
            # more orphans.
            still = await conn.fetchval(
                f"SELECT internal_id FROM {qschema}.banks WHERE bank_id = $1",  # noqa: S608 — quoted identifier
                plan.bank_id,
            )
            if still is None or str(still) != internal_id_str:
                try:
                    await retry_with_backoff(
                        lambda qualified=qualified: conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {qualified}")
                    )
                except Exception as cleanup_exc:  # noqa: BLE001
                    logger.warning(
                        "Bank %s was deleted during build of %s; orphan drop failed: %s",
                        plan.bank_id,
                        qualified,
                        cleanup_exc,
                    )
                break
            result.created += 1
            logger.info("Built vector index %s (bank=%s, fact_type=%s)", qualified, plan.bank_id, fact_type)

    return result


async def reconcile_bank_vector_indexes(
    conn: Any,
    schema: str,
    bank_id: str,
    index_clause: str,
    *,
    dry_run: bool = False,
) -> BankIndexResult:
    """Plan and apply one bank's vector-index coverage.

    Always plans with :attr:`CoverageTrigger.FULL`. Both callers — the
    maintenance job re-planning at start and ``repair-bank`` — reconcile a bank
    without knowing which way it last moved, and they are what recovers whatever
    a directional pre-check deferred.
    """
    plan = await plan_bank_vector_indexes(conn, schema, bank_id, trigger=CoverageTrigger.FULL)
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
