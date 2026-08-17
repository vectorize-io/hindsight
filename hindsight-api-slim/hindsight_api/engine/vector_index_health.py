"""Per-bank vector index coverage checks and repair.

A (bank, fact_type) partition earns a partial vector index by size, not by
existing: below ``HINDSIGHT_API_VECTOR_INDEX_MIN_ROWS`` the planner answers the
same ANN query from the ``(bank_id, fact_type)`` B-tree plus a top-N sort, which
is exact and cheaper than descending an ANN graph, and the index it would
otherwise carry is paid for by every *other* bank in the deployment — indexes on
the shared ``memory_units`` table are locked and planned against by every query
and opened by every DML statement on that table. Three per bank exhausts the
lock table at a few thousand banks (issue #3485).

This module is the shared engine for reconciling the live index set against that
policy: build what qualifies, drop what no longer does. It is driven by the
background sweep (``MaintenanceLoop._run_vector_index_sweep``) and by the
``repair-bank`` admin command, so both apply identical policy — a threshold that
differed between the builder and the checker would oscillate.

Builds and drops always use ``CREATE/DROP INDEX CONCURRENTLY`` on a raw
autocommit connection, so neither takes ``ACCESS EXCLUSIVE`` on the shared
``memory_units`` table. This is also what makes the drop path usable on an
instance that has already hit the #3485 wall: ``DROP INDEX`` is a utility
statement that locks its own index plus the table, rather than planning against
all of the table's indexes the way any DML must.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .._vector_index import per_bank_index_drop_rows, per_bank_index_min_rows
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
class SchemaVectorIndexResult:
    """Per-schema outcome of a vector-index reconcile pass."""

    schema: str
    banks_scanned: int = 0
    already_present: int = 0
    created: int = 0
    dropped: int = 0
    skipped: int = 0  # would-create, reported under --dry-run
    would_drop: int = 0  # would-drop, reported under --dry-run
    failed: int = 0
    failed_indexes: list[str] = field(default_factory=list)
    # True when a per-pass budget stopped the reconcile with work still to do.
    # The sweep uses this to come back on the short interval instead of waiting
    # out its full one, so a large backlog drains in minutes rather than one
    # budget per hour.
    backlog: bool = False


@dataclass
class VectorIndexBudget:
    """Caps on how much index DDL a single reconcile pass may issue.

    Builds and drops are budgeted separately because they cost wildly different
    amounts: a build is an ANN index construction over the partition's rows,
    while a drop finishes as soon as concurrent readers of that index drain.
    The asymmetry matters most on a deployment recovering from #3485, which has
    tens of thousands of indexes to shed and only a handful worth rebuilding.

    The counters are pass-global, not per-schema: the budget exists to bound the
    load one sweep puts on the *database*, and a deployment with a thousand
    tenant schemas would blow through a per-schema cap a thousand times over.
    One instance is threaded through every schema in the pass, so whichever
    schemas are visited first spend it.
    """

    max_builds: int | None = None
    max_drops: int | None = None
    builds_done: int = 0
    drops_done: int = 0

    @property
    def build_exhausted(self) -> bool:
        return self.max_builds is not None and self.builds_done >= self.max_builds

    @property
    def drop_exhausted(self) -> bool:
        return self.max_drops is not None and self.drops_done >= self.max_drops


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


async def _existing_bank_indexes(conn: Any, schema: str) -> list[str]:
    """Every per-bank vector index currently on ``schema.memory_units``.

    Catalog-only, so it answers even on an instance whose lock table is
    exhausted and cannot plan a query against ``memory_units`` itself. This is
    the set the reconcile drops *from*: a partition that has fallen below the
    threshold, or a bank deleted while its index survived, never appears in the
    row-count discovery and could not be found any other way.
    """
    rows = await conn.fetch(
        """
        SELECT c.relname AS index_name
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_index i ON i.indexrelid = c.oid
        JOIN pg_class t ON t.oid = i.indrelid
        WHERE n.nspname = $1
          AND t.relname = 'memory_units'
          AND c.relname LIKE 'idx_mu_emb\\_%'
        """,
        schema,
    )
    return [row["index_name"] for row in rows]


@dataclass
class _Partition:
    """A (bank, fact_type) pair the reconcile has a row count for."""

    bank_id: str
    internal_id: str
    fact_type: str
    row_count: int

    @property
    def index_name(self) -> str:
        return _bank_index_name(self.fact_type, self.internal_id)


async def _drop_index(conn: Any, qualified: str) -> None:
    await conn.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {qualified}")


async def _scoped_index_names(conn: Any, schema: str, bank_id: str) -> set[str]:
    """The three index names ``bank_id`` could own, from its ``internal_id``.

    Resolved from ``banks`` rather than from the partition list because a bank
    that has fallen below the floor contributes no partitions yet may still own
    indexes that need dropping. An empty set means the bank row is gone, in
    which case a scoped run has nothing it is allowed to touch — the orphan is
    for an unscoped run to collect.
    """
    internal_id = await conn.fetchval(
        f"SELECT internal_id FROM {_quote_identifier(schema)}.banks WHERE bank_id = $1",  # noqa: S608 — quoted identifier
        bank_id,
    )
    if internal_id is None:
        return set()
    return {_bank_index_name(ft, str(internal_id)) for ft in _BANK_INDEX_FACT_TYPES}


async def _reconcile_schema(
    conn: Any,
    schema: str,
    index_clause: str,
    partitions: list[_Partition],
    *,
    dry_run: bool,
    budget: VectorIndexBudget,
    bank_scope: str | None = None,
) -> SchemaVectorIndexResult:
    """Converge ``schema``'s per-bank vector indexes onto the size policy.

    ``partitions`` is every (bank, fact_type) whose row count is at or above the
    *drop* threshold — i.e. everything that may legitimately keep an index. A
    partition at or above the *build* threshold that has no healthy index gets
    one; every existing index not named by that set is dropped, which covers
    both shrunk partitions and indexes orphaned by a deleted bank.

    ``bank_scope`` narrows **both** halves to one bank. It has to narrow the
    drop half too: the drop set is "every index this schema has that the
    partition list does not account for", and with a single bank's partitions
    that description covers every *other* bank's indexes in the schema. A
    scoped run therefore considers only the three names its own bank could own.
    """
    result = SchemaVectorIndexResult(schema=schema)
    qschema = _quote_identifier(schema)
    build_minimum = per_bank_index_min_rows()

    result.banks_scanned = len({p.bank_id for p in partitions})

    keep = {p.index_name for p in partitions}
    wanted = [p for p in partitions if build_minimum > 0 and p.row_count >= build_minimum]
    # Biggest partitions first: they have the most latency to gain, and when the
    # build budget cuts the pass short they are the ones worth spending it on.
    wanted.sort(key=lambda p: p.row_count, reverse=True)

    health = await _index_health(conn, schema, [p.index_name for p in wanted])

    # ── drop what no longer qualifies ──────────────────────────────────────
    existing = set(await _existing_bank_indexes(conn, schema))
    if bank_scope is not None:
        existing &= await _scoped_index_names(conn, schema, bank_scope)
    stale = sorted(existing - keep)
    for dropped_so_far, index_name in enumerate(stale):
        if budget.drop_exhausted:
            result.backlog = True
            logger.info(
                "Vector index sweep %s: drop budget (%s) reached with %s index(es) still stale; "
                "resuming on the next sweep",
                schema,
                budget.max_drops,
                len(stale) - dropped_so_far,
            )
            break
        if dry_run:
            result.would_drop += 1
            continue
        qualified = f"{qschema}.{_quote_identifier(index_name)}"
        try:
            await retry_with_backoff(lambda qualified=qualified: _drop_index(conn, qualified))
            result.dropped += 1
            budget.drops_done += 1
        except Exception as exc:  # noqa: BLE001 — one failed drop must not abort the rest
            result.failed += 1
            result.failed_indexes.append(qualified)
            logger.warning("Failed to drop stale vector index %s: %s", qualified, exc)

    # ── build what has grown into one ──────────────────────────────────────
    for partition in wanted:
        index_name = partition.index_name
        if health.get(index_name) is True:
            result.already_present += 1
            continue
        if budget.build_exhausted:
            result.backlog = True
            logger.info(
                "Vector index sweep %s: build budget (%s) reached; %s partition(s) still awaiting an "
                "index, resuming on the next sweep",
                schema,
                budget.max_builds,
                sum(1 for p in wanted if health.get(p.index_name) is not True) - result.created,
            )
            break
        if dry_run:
            result.skipped += 1
            continue

        qindex = _quote_identifier(index_name)
        qualified = f"{qschema}.{qindex}"
        # Render the bank_id literal server-side so escaping does not depend on
        # standard_conforming_strings (the predicate is inlined into the DDL).
        bank_id_literal = await conn.fetchval("SELECT quote_literal($1::text)", partition.bank_id)

        async def _rebuild(
            qindex: str = qindex,
            qualified: str = qualified,
            fact_type: str = partition.fact_type,
            bank_id_literal: str = bank_id_literal,
        ) -> None:
            # Always drop first. An unhealthy-but-present index (INVALID
            # leftover, wrong access method) can't be repaired by
            # IF NOT EXISTS, and a prior deadlocked CONCURRENTLY build leaves
            # an INVALID stub that IF NOT EXISTS would likewise skip — so a
            # retry must clear it. DROP ... IF EXISTS is a no-op when the
            # index is simply absent (healthy is None).
            await _drop_index(conn, qualified)
            await conn.execute(
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {qindex} "
                f"ON {qschema}.memory_units {index_clause} "
                f"WHERE fact_type = '{fact_type}' AND bank_id = {bank_id_literal}"
            )

        try:
            # CREATE INDEX CONCURRENTLY on the live, concurrently-written
            # memory_units table can be chosen as a deadlock victim
            # (sqlstate 40P01 / ORA-00060). That is transient — Postgres
            # aborts one side to break the cycle — so retry the drop+build a
            # few times before recording a permanent failure.
            await retry_with_backoff(_rebuild)
            result.created += 1
            budget.builds_done += 1
            logger.info(
                "Built vector index %s (bank=%s, fact_type=%s, %s rows)",
                qualified,
                partition.bank_id,
                partition.fact_type,
                partition.row_count,
            )
        except Exception as exc:  # noqa: BLE001 — one failed index must not abort the rest
            result.failed += 1
            result.failed_indexes.append(qualified)
            logger.warning(
                "Failed to build vector index %s (bank=%s, fact_type=%s): %s — "
                "dropping the invalid leftover so a re-run can retry.",
                qualified,
                partition.bank_id,
                partition.fact_type,
                exc,
            )
            # A failed concurrent build leaves an INVALID index behind that
            # would shadow the good one; drop it so a re-run retries cleanly.
            try:
                await _drop_index(conn, qualified)
            except Exception as cleanup_exc:  # noqa: BLE001
                logger.warning("Cleanup DROP INDEX for %s also failed: %s", qualified, cleanup_exc)

    return result


async def _safe_reconcile_schema(
    conn: Any,
    schema: str,
    index_clause: str,
    partitions: list[_Partition],
    *,
    dry_run: bool,
    budget: VectorIndexBudget,
    bank_scope: str | None = None,
) -> SchemaVectorIndexResult:
    try:
        return await _reconcile_schema(
            conn, schema, index_clause, partitions, dry_run=dry_run, budget=budget, bank_scope=bank_scope
        )
    except Exception as exc:  # noqa: BLE001 — one bad schema must not abort the whole sweep
        logger.warning("Vector index reconcile aborted for schema %s: %s", schema, exc)
        return SchemaVectorIndexResult(schema=schema, failed=1, failed_indexes=[f"{schema}.<schema-error>"])


async def discover_partitions(conn: Any, routine: str, bank_id: str | None = None) -> dict[str, list[_Partition]]:
    """Row counts per (bank, fact_type), across every tenant schema, in one round-trip.

    ``routine`` is the fully-qualified ``banks_needing_vector_index`` name (the
    caller resolves it via ``fq_routine``; it is database-global and installed
    only in the configured schema). The floor passed to it is the *drop*
    threshold, not the build threshold, so the caller gets back everything that
    may legitimately keep an index — a partition between the two bounds is one
    the hysteresis says to leave alone, and it must appear here or the reconcile
    would drop its index.

    **A schema is a key here only if the routine actually scanned it.** The
    routine skips schemas that vanished or are held under concurrent DDL, and
    marks each one it completed with a NULL-``bank_id`` sentinel row. The
    distinction is load-bearing: the reconcile drops every index a schema's
    partition list does not account for, so treating a skipped schema as an
    empty one would drop all of its vector indexes and rebuild them on the next
    pass. Absent from this mapping means "unknown, leave alone"; present with an
    empty list means "scanned, owns nothing that qualifies".
    """
    rows = await conn.fetch(f"SELECT * FROM {routine}($1)", per_bank_index_drop_rows())
    by_schema: dict[str, list[_Partition]] = {}
    for row in rows:
        # Sentinel: schema scanned, no partitions of its own to report.
        if row["bank_id"] is None:
            by_schema.setdefault(row["schema_name"], [])
            continue
        if row["fact_type"] not in _BANK_INDEX_FACT_TYPES:
            continue
        partitions = by_schema.setdefault(row["schema_name"], [])
        # The bank filter is applied after the sentinel so a scoped call still
        # confirms the schema was reachable.
        if bank_id is not None and row["bank_id"] != bank_id:
            continue
        partitions.append(
            _Partition(
                bank_id=row["bank_id"],
                internal_id=str(row["internal_id"]),
                fact_type=row["fact_type"],
                row_count=int(row["row_count"]),
            )
        )
    return by_schema


async def reconcile_vector_indexes(
    conn: Any,
    schemas: list[str],
    index_clause: str,
    partitions_by_schema: dict[str, list[_Partition]],
    *,
    dry_run: bool = False,
    budget: VectorIndexBudget | None = None,
    bank_id: str | None = None,
) -> list[SchemaVectorIndexResult]:
    """Converge per-bank vector indexes onto the size policy across ``schemas``.

    ``conn`` must be a raw autocommit PostgreSQL connection: ``CREATE INDEX
    CONCURRENTLY`` cannot run inside a transaction block, and both it and
    ``DROP INDEX CONCURRENTLY`` need a real backend session (a transaction-pooled
    URL will not do — that is what ``HINDSIGHT_API_MIGRATION_DATABASE_URL`` is
    for).

    Only schemas ``partitions_by_schema`` has a key for are visited, including
    those whose list is empty — an empty list means "scanned, owns nothing that
    qualifies", so its indexes are dropped, which is the whole recovery path for
    a deployment that hit #3485. A schema discovery *skipped* has no key and is
    left alone; see :func:`discover_partitions`.

    ``bank_id`` narrows the pass to one bank, in both directions — without it a
    scoped run would read every other bank's indexes as unaccounted-for and drop
    them.

    Concurrency is handled by idempotency, not a lock (project rule: no advisory
    locks — they are unreliable behind connection poolers, and leaning on one is
    why #2803 was rejected). Every build is ``CREATE INDEX CONCURRENTLY IF NOT
    EXISTS`` guarded by a valid/ready health check and every drop is ``DROP INDEX
    CONCURRENTLY IF EXISTS``, so a second concurrent sweep is a no-op on work the
    first already did; if two runs race the *same* missing index, Postgres
    rejects one build and the per-index handler drops the leftover so a re-run
    converges cleanly.
    """
    budget = budget or VectorIndexBudget()
    return [
        await _safe_reconcile_schema(
            conn,
            schema,
            index_clause,
            partitions_by_schema[schema],
            dry_run=dry_run,
            budget=budget,
            bank_scope=bank_id,
        )
        for schema in schemas
        if schema in partitions_by_schema
    ]
