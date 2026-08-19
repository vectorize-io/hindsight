"""Abstract base class for backend-specific data access operations.

SQLDialect handles SQL *fragment* generation (param placeholders, JSON ops, vector
distance, etc.) — stateless, no I/O.

DataAccessOps handles multi-statement *execution* patterns that differ between
backends (unnest batch insert vs executemany, LATERAL fan-out vs per-row query,
DISTINCT ON vs GROUP BY workarounds, etc.).  Methods receive a DatabaseConnection
and execute complete operations.

This eliminates scattered ``if backend_type == "postgresql"`` conditionals from
business logic.  Adding a new backend (e.g. Neon, Databricks) means implementing
this ABC — consumer code never checks the backend directly.

Follows the Strategy pattern (Fowler's "Replace Conditional with Polymorphism")
and mirrors Django's ``DatabaseOperations`` architecture.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .base import DatabaseConnection
from .result import ResultRow


def document_serialization_sql(table: str, alias: str) -> str:
    """SQL predicate keeping one document to a single in-flight retain.

    A retain that targets exactly one document carries it in
    ``serialization_key``. Appending to a document is a read-modify-write over
    its whole text, so two concurrent retains for one document can only produce
    a lost update or a wasted extraction — never more throughput. This
    predicate makes the queue reflect that: a candidate is claimable only when
    no peer for the same document is already ``processing``, and only when it
    is the oldest claimable pending peer for that document.

    Ordering, not just exclusion, is the point. Appends are cumulative, so the
    order they commit in is the order the document ends up in; claiming them by
    ``(created_at, operation_id)`` makes that the submission order. It also
    stops a single claim batch from taking several peers at once, which
    excluding busy documents alone would not prevent.

    Rows with a NULL ``serialization_key`` — multi-document batches, and every
    non-retain operation — are unaffected, and documents are independent of one
    another, so this costs no parallelism across a busy bank: only the retains
    that were racing each other for one document are put in a line.

    A peer wedged in 'processing' holds its document until claim recovery
    releases it, the same caveat ``graph_maintenance_bank_serialization_sql``
    carries and the same general gap.

    The candidate row is always 'pending' and the 'pending' branch is
    strictly-older, so the subquery can never match the candidate itself. The
    fragment carries no SQL comments on purpose — it is rewritten for Oracle by
    regex (``db/oracle.py``).

    Args:
        table: Fully-qualified async_operations table.
        alias: Alias of the outer candidate row in the calling query.
    """
    return f"""
        ({alias}.serialization_key IS NULL OR NOT EXISTS (
            SELECT 1 FROM {table} doc_peer
            WHERE doc_peer.bank_id = {alias}.bank_id
              AND doc_peer.serialization_key = {alias}.serialization_key
              AND (
                  doc_peer.status = 'processing'
                  OR (doc_peer.status = 'pending'
                      AND doc_peer.task_payload IS NOT NULL
                      AND (doc_peer.next_retry_at IS NULL OR doc_peer.next_retry_at <= NOW())
                      AND (doc_peer.created_at < {alias}.created_at
                           OR (doc_peer.created_at = {alias}.created_at
                               AND doc_peer.operation_id < {alias}.operation_id)))
              )
        ))
    """


def graph_maintenance_bank_serialization_sql(table: str, alias: str) -> str:
    """SQL predicate serialising ``graph_maintenance`` claims per bank (#3230).

    Every graph_maintenance run for a bank is interchangeable — the payload
    carries only ``bank_id``, and ``run_graph_maintenance_job`` drains that bank's
    queues — so a second concurrent run for one bank adds no work. It is worse than
    useless: ``claim_graph_maintenance_batch`` locks queue rows ``FOR UPDATE``
    *without* ``SKIP LOCKED`` (it is written assuming a single runner per bank),
    so the runs convoy on each other's row locks while each holds a worker slot.

    Same guarantee ``consolidation`` already gets from its ``bank_id != ALL(busy)``
    exclusion, and the same caveat: a row wedged in 'processing' holds its bank
    until something releases it (``hindsight-admin recover``, or a restart with a
    stable ``HINDSIGHT_API_WORKER_ID`` so ``recover_own_tasks`` matches it). That
    is a general gap in claim recovery, not specific to graph_maintenance.

    Two differences from the consolidation form, both forced by the shape of this
    problem:

    * It is a **predicate**, not a separate claim phase. Pulling graph_maintenance
      into its own phase after the generic shared-pool query would drop it below
      every other operation type: it has no reserved-slot floor
      (``WORKER_SLOT_TYPE_DEFAULTS`` gives consolidation 2 and graph_maintenance
      0), and the poller's fairness pass calls ``claim_tasks`` with
      ``shared_limit=1``, so a single pending retain would starve it indefinitely.
      As a predicate it keeps competing by ``created_at``.
    * It also suppresses every same-bank row but the oldest **within one batch**.
      Excluding busy banks alone does not: with several pending rows and nothing
      yet processing, one batch claims them all — the convoy, unchanged. Several
      pending rows per bank are reachable through the recovery paths
      (``_reclaim_own_processing_tasks`` resets *all* of a worker's processing
      rows in one statement, from ``recover_own_tasks`` at startup and
      ``release_own_tasks`` at shutdown, plus ``_schedule_retry`` /
      ``_defer_operation`` / ``hindsight-admin recover``).

    The candidate row is always 'pending' and the 'pending' branch is
    strictly-older, so the subquery can never match the candidate itself. The
    fragment carries no SQL comments on purpose — it is rewritten for Oracle by
    regex (``db/oracle.py``).

    Args:
        table: Fully-qualified async_operations table.
        alias: Alias of the outer candidate row in the calling query.
    """
    return f"""
        ({alias}.operation_type <> 'graph_maintenance' OR NOT EXISTS (
            SELECT 1 FROM {table} gm_peer
            WHERE gm_peer.bank_id = {alias}.bank_id
              AND gm_peer.operation_type = 'graph_maintenance'
              AND (
                  gm_peer.status = 'processing'
                  OR (gm_peer.status = 'pending'
                      AND gm_peer.task_payload IS NOT NULL
                      AND (gm_peer.next_retry_at IS NULL OR gm_peer.next_retry_at <= NOW())
                      AND (gm_peer.created_at < {alias}.created_at
                           OR (gm_peer.created_at = {alias}.created_at
                               AND gm_peer.operation_id < {alias}.operation_id)))
              )
        ))
    """


@dataclass
class TagListingParts:
    """Backend-specific SQL fragments for the tag listing query."""

    tag_source: str
    non_empty_check: str
    tag_col: str
    bank_prefix: str


@dataclass(frozen=True)
class UpdatedWindow:
    """Recall's ``created_after``/``created_before`` bounds, as SQL for graph expansion.

    Recall applies the window to ``updated_at`` — a consolidation touch makes a
    fact current again — so link expansion has to bound the same column its seed
    query does.  Filtering only the seeds is not enough: a single in-window seed
    would otherwise drag its whole neighbourhood (shared entities, semantic kNN
    links, causal links) into the results no matter how old those neighbours are.

    ``first_param_index`` is where the bounds land in the owning query's param
    list, so each call site keeps the placeholder numbering next to the params it
    binds.  Rendering is per-alias because the same window is applied to several
    correlation names within one query.
    """

    after: datetime | None
    before: datetime | None
    first_param_index: int

    def clause(self, alias: str) -> str:
        """``AND <alias>.updated_at > $n ...`` — empty when the window is unbounded."""
        parts: list[str] = []
        index = self.first_param_index
        if self.after is not None:
            parts.append(f" AND {alias}.updated_at > ${index}")
            index += 1
        if self.before is not None:
            parts.append(f" AND {alias}.updated_at < ${index}")
        return "".join(parts)

    @property
    def params(self) -> list[datetime]:
        """The bound values, in placeholder order. Append to the owning param list."""
        return [bound for bound in (self.after, self.before) if bound is not None]


@dataclass(frozen=True)
class LinkExpansionRows:
    """The three link-expansion signals, kept apart until they are scored.

    They cannot be concatenated at the SQL layer: each carries a different score
    scale (shared-entity count, kNN weight, causal weight) and the caller applies
    a different transformation to each before summing them.
    """

    entity: list[ResultRow]
    semantic: list[ResultRow]
    causal: list[ResultRow]


class DataAccessOps(ABC):
    """Backend-specific multi-statement data access operations.

    Each method encapsulates a complete DB operation that differs
    in execution strategy between backends.
    """

    @property
    def uses_observation_sources_table(self) -> bool:
        """Whether this backend uses the observation_sources junction table.

        PG uses native array ops (source_memory_ids column) for reads and
        skips junction table writes.  Oracle uses the junction table for both.
        """
        return True  # Default: use junction table (Oracle)

    # -- Bulk insert operations ------------------------------------------

    @abstractmethod
    async def bulk_upsert_chunks(
        self,
        conn: DatabaseConnection,
        table: str,
        chunk_ids: list[str],
        document_ids: list[str],
        bank_ids: list[str],
        chunk_texts: list[str],
        chunk_indices: list[int],
        content_hashes: list[str],
    ) -> None:
        """Bulk upsert chunks with ON CONFLICT handling.

        PG uses INSERT ... SELECT FROM unnest() with ON CONFLICT DO UPDATE.
        Non-PG uses bulk_insert_from_arrays (executemany).
        """
        ...

    @abstractmethod
    async def lock_document_for_write(
        self,
        conn: DatabaseConnection,
        table: str,
        doc_id: str,
        bank_id: str,
    ) -> str | None:
        """Ensure the document row exists, take a row lock on it, and return its
        pre-existing ``content_hash``.

        This serializes all concurrent writers for ``doc_id`` at the DB level
        (so interleaved same-document retains can't corrupt each other), while
        creating the row on first write. The returned hash is ``'__pending__'``
        for a freshly inserted row, the stored hash for an existing one, or
        ``None`` if the row could not be read back.

        PG does this in a single statement (``INSERT ... ON CONFLICT DO UPDATE
        ... RETURNING``), which always takes the row lock as part of the upsert.
        Oracle can't (``MERGE`` doesn't support ``RETURNING``), so it splits the
        work into an idempotent insert plus a ``SELECT ... FOR UPDATE``.
        """
        ...

    @abstractmethod
    async def insert_facts_batch(
        self,
        conn: DatabaseConnection,
        bank_id: str,
        fact_texts: list[str],
        embeddings: list[str],
        event_dates: list,
        occurred_starts: list,
        occurred_ends: list,
        mentioned_ats: list,
        contexts: list[str],
        fact_types: list[str],
        metadata_jsons: list[str],
        chunk_ids: list,
        document_ids: list,
        tags_list: list[str],
        observation_scopes_list: list,
        text_signals_list: list,
        text_search_extension: str = "native",
    ) -> list[str]:
        """Batch-insert facts, returning IDs.

        PG uses INSERT ... SELECT FROM unnest() with RETURNING.
        Non-PG inserts row-by-row with individual RETURNING.
        """
        ...

    @abstractmethod
    async def bulk_insert_links(
        self,
        conn: DatabaseConnection,
        table: str,
        sorted_links: list[tuple],
        bank_id: str,
        nil_entity_uuid: str,
        exists_clause: str,
        chunk_size: int = 5000,
    ) -> None:
        """Bulk insert memory_links with conflict handling.

        PG uses INSERT ... SELECT FROM unnest() with chunking.
        Non-PG uses executemany.
        """
        ...

    @abstractmethod
    async def bulk_insert_entities(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        entity_names: list[str],
        entity_dates: list,
        entity_kinds: list[str],
    ) -> dict[str, str]:
        """Bulk insert entities with ON CONFLICT DO NOTHING, returning id-by-lowercase-name.

        ``entity_kinds`` ("regular"/"label", parallel to ``entity_names``) is
        stored on the row so label entities stay out of the partial trigram
        index (#3208).

        PG uses INSERT ... SELECT FROM unnest() with RETURNING.
        Non-PG inserts row-by-row then SELECTs.
        """
        ...

    @abstractmethod
    async def fetch_missing_entity_ids(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        missing_names: list[str],
    ) -> list[ResultRow]:
        """Fetch entity IDs for names that conflicted during insert.

        PG uses unnest + JOIN.
        Non-PG queries each name individually.
        """
        ...

    @abstractmethod
    async def bulk_reassert_entities(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        entity_ids: list[str],
        canonical_names: list[str],
        entity_kinds: list[str],
    ) -> None:
        """Lock resolved parents and re-create any pruned since Phase-1 resolution.

        Closes the retain Phase-1/prune race (#2662): existing rows are locked
        (PG ``FOR KEY SHARE`` / Oracle ``FOR UPDATE``) so a concurrent
        ``prune_orphan_entities`` blocks until the caller's transaction commits,
        while rows already deleted are re-inserted idempotently. ``entity_ids``
        must be sorted by the caller for a stable lock order.
        """
        ...

    @abstractmethod
    async def bulk_insert_unit_entities(
        self,
        conn: DatabaseConnection,
        table: str,
        unit_ids: list,
        entity_ids: list,
    ) -> None:
        """Bulk insert unit_entities links with ON CONFLICT DO NOTHING.

        PG uses INSERT ... SELECT FROM unnest().
        Non-PG uses executemany.
        """
        ...

    # -- LATERAL / fan-out queries ---------------------------------------

    @abstractmethod
    async def fetch_unit_dates(
        self,
        conn: DatabaseConnection,
        mu_table: str,
        unit_ids: list[str],
    ) -> list[ResultRow]:
        """Fetch event_date/fact_type for a list of unit IDs.

        PG uses ANY($1) array binding.
        Non-PG queries each unit individually.
        """
        ...

    @abstractmethod
    async def fetch_temporal_neighbors(
        self,
        conn: DatabaseConnection,
        mu_table: str,
        bank_id: str,
        lateral_unit_ids: list,
        lateral_event_dates: list,
        lateral_fact_types: list,
        half_limit: int,
        batch_size: int = 500,
    ) -> list[ResultRow]:
        """Fetch temporal neighbors using bidirectional index scan.

        PG uses unnest + CROSS JOIN LATERAL for batched bidirectional scan.
        Non-PG queries each unit individually with backward/forward scans.
        """
        ...

    # -- CTE builders for graph retrieval --------------------------------

    @abstractmethod
    def build_entity_expansion_cte(
        self,
        mu_table: str,
        ue_table: str,
        per_entity_limit: int,
        window: UpdatedWindow,
    ) -> str:
        """Build entity expansion CTE for link expansion retrieval.

        PG uses DISTINCT ON with CROSS JOIN LATERAL and GROUP BY.
        Non-PG splits into entity_scores subquery then JOINs for full columns
        (can't GROUP BY CLOB).

        ``window`` narrows candidates *before* the per-entity cap, so out-of-window
        neighbours don't consume an entity's bounded fan-out.
        """
        ...

    @abstractmethod
    def build_semantic_causal_cte(
        self,
        ml_table: str,
        mu_table: str,
        window: UpdatedWindow,
    ) -> str:
        """Build semantic + causal expansion CTEs.

        PG uses DISTINCT ON for deduplication.
        Non-PG computes MAX(weight) in subquery then JOINs for full columns.
        """
        ...

    @abstractmethod
    async def expand_observations(
        self,
        conn: DatabaseConnection,
        mu_table: str,
        ue_table: str,
        ml_table: str,
        seed_ids: list,
        budget: int,
        per_entity_limit: int,
        window: UpdatedWindow,
    ) -> LinkExpansionRows:
        """Observation-specific graph expansion.

        PG uses native array ops (source_memory_ids column) for performance.
        Oracle uses the observation_sources junction table.
        """
        ...

    # -- Tag listing -----------------------------------------------------

    @abstractmethod
    def build_tag_listing_parts(self, mu_table: str) -> TagListingParts:
        """Build SQL fragments for the tag listing query.

        PG uses unnest(tags) to expand the VARCHAR[] column.
        Non-PG uses CROSS APPLY JSON_TABLE on the CLOB column.
        """
        ...

    # -- Bank index management -------------------------------------------

    # No create counterpart: per-(bank, fact_type) partial vector indexes are
    # earned by size and built by the maintenance sweep over its own autocommit
    # connection (see engine/vector_index_health.py), never on a request path.
    # The drop stays here because bank deletion must remove a large bank's
    # indexes while it still knows the internal_id they are named after.
    @abstractmethod
    async def drop_bank_vector_indexes(
        self,
        conn: DatabaseConnection,
        schema: str,
        internal_id: str,
        fact_types: dict[str, str],
    ) -> None:
        """Drop per-bank partial vector indexes.

        PG drops per-(bank, fact_type) indexes.
        Non-PG is a no-op.
        """
        ...

    # -- Entity resolution strategy routing ------------------------------

    @abstractmethod
    def get_entity_resolution_strategy(self) -> str:
        """Return the fuzzy entity matching strategy name.

        PG uses "trigram" (pg_trgm).
        Non-PG uses "oracle_fuzzy" (UTL_MATCH) or falls back to "full".
        """
        ...

    # -- Webhook operations ------------------------------------------------

    @abstractmethod
    async def create_webhook(
        self,
        conn: DatabaseConnection,
        table: str,
        webhook_id: Any,
        bank_id: str,
        url: str,
        secret: str | None,
        event_types: list[str],
        enabled: bool,
        http_config_json: str,
    ) -> ResultRow | None:
        """Insert a webhook row and return the created row."""
        ...

    @abstractmethod
    async def list_webhooks_for_bank(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
    ) -> list[ResultRow]:
        """List all webhooks for a bank, ordered by created_at."""
        ...

    @abstractmethod
    async def get_webhooks_for_dispatch(
        self,
        conn: DatabaseConnection,
        webhook_table: str,
        bank_id: str,
    ) -> list[ResultRow]:
        """Get enabled webhooks matching a bank (bank-specific + global NULL rows)."""
        ...

    @abstractmethod
    async def update_webhook(
        self,
        conn: DatabaseConnection,
        table: str,
        webhook_id: Any,
        bank_id: str,
        set_clauses: list[str],
        params: list[Any],
    ) -> ResultRow | None:
        """Update a webhook and return the updated row, or None if not found."""
        ...

    @abstractmethod
    async def delete_webhook(
        self,
        conn: DatabaseConnection,
        table: str,
        webhook_id: Any,
        bank_id: str,
    ) -> bool:
        """Delete a webhook. Returns True if a row was deleted."""
        ...

    @abstractmethod
    async def list_webhook_deliveries(
        self,
        conn: DatabaseConnection,
        ops_table: str,
        webhook_id: str,
        bank_id: str,
        limit: int,
        cursor: str | None,
    ) -> list[ResultRow]:
        """List webhook delivery operations for a specific webhook, newest first."""
        ...

    @abstractmethod
    async def insert_webhook_delivery_task(
        self,
        conn: DatabaseConnection,
        ops_table: str,
        operation_id: Any,
        bank_id: str,
        payload_json: str,
        timestamp: Any,
    ) -> None:
        """Insert a webhook delivery task into async_operations."""
        ...

    # -- Graph maintenance queue -----------------------------------------

    @abstractmethod
    async def enqueue_graph_maintenance(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        unit_ids: list,
    ) -> None:
        """Insert unit_ids into graph_maintenance_queue, deduplicating on the
        (bank_id, unit_id) primary key.

        Called inside the triggering transaction so enqueue is atomic with
        the mutation that caused it. Order is unspecified.
        """
        ...

    @abstractmethod
    async def claim_graph_maintenance_batch(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        limit: int,
    ) -> list[str]:
        """Atomically claim a batch of rows from graph_maintenance_queue and
        remove them from the table.

        Returns the list of ``unit_id`` strings. Empty list when the queue
        for ``bank_id`` is drained.
        """
        ...

    @abstractmethod
    async def enqueue_entity_maintenance(
        self,
        conn: DatabaseConnection,
        table: str,
        ue_table: str,
        bank_id: str,
        unit_ids: list,
    ) -> int:
        """Enqueue the entities referenced by ``unit_ids`` as prune candidates.

        Reads the entity ids out of ``unit_entities`` and inserts them into
        entity_maintenance_queue, deduplicating on the (bank_id, entity_id)
        primary key. Returns the number of rows the insert added.

        Must run inside the triggering transaction and BEFORE the rows go —
        once the unit_entities rows are deleted (or cascaded away) there is
        nothing left to read the entity ids from.
        """
        ...

    @abstractmethod
    async def claim_entity_maintenance_batch(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        limit: int,
    ) -> list:
        """Atomically claim a batch of rows from entity_maintenance_queue and
        remove them from the table.

        Returns the claimed entity ids. Empty list when the queue for
        ``bank_id`` is drained.
        """
        ...

    @abstractmethod
    async def prune_orphan_entities(
        self,
        conn: DatabaseConnection,
        entities_table: str,
        ue_table: str,
        bank_id: str,
        entity_ids: list,
    ) -> int:
        """Delete those of ``entity_ids`` in ``bank_id`` that no longer have any
        unit_entities rows referencing them. Returns the number of rows deleted.

        FK ON DELETE CASCADE on entity_cooccurrences then removes any
        cooccurrence row pointing at the pruned entities.
        """
        ...

    @abstractmethod
    async def prune_stale_cooccurrences(
        self,
        conn: DatabaseConnection,
        ec_table: str,
        ue_table: str,
        entity_ids: list,
    ) -> int:
        """Delete entity_cooccurrences rows incident to ``entity_ids`` where the
        two entities still exist but no current unit references both of them.

        These are stale-count rows: cooccurrence was real at the time it was
        recorded, but every memory_unit that witnessed both entities has
        since been deleted. Returns the number of rows deleted.
        """
        ...

    # -- Task claiming operations ------------------------------------------

    @abstractmethod
    async def prune_terminal_operations(
        self,
        conn: DatabaseConnection,
        table: str,
        cutoff: datetime,
        *,
        batch_size: int,
    ) -> int:
        """Delete one deterministic batch of terminal operations older than ``cutoff``.

        Implementations must lock candidates without waiting on rows another
        worker is pruning, never select pending/processing rows, and return the
        number deleted. The caller provides a transaction around this method.
        """
        ...

    @abstractmethod
    async def claim_tasks(
        self,
        conn: DatabaseConnection,
        table: str,
        worker_id: str,
        reserved_limits: dict[str, int],
        shared_limit: int,
        *,
        consolidation_bank_priority: dict[str, int] | None = None,
    ) -> list[ResultRow]:
        """Claim pending tasks from the async_operations table.

        PG implementation can use NOT EXISTS + FOR UPDATE SKIP LOCKED in one query.
        Oracle implementation uses two-step claims (query busy banks first, then
        claim excluding them) to avoid ORA-02014.

        Implementations must apply :func:`graph_maintenance_bank_serialization_sql`
        to every query that can return a ``graph_maintenance`` row, so at most one
        such row per bank is ever in flight, and :func:`document_serialization_sql`
        to every query that can return a ``retain`` row, so at most one retain per
        document is ever in flight.

        Args:
            consolidation_bank_priority: Per-bank priority for consolidation scheduling.
                Maps bank name patterns to integer priorities (higher = claimed first).
                Patterns support ``*`` as wildcard (converted to SQL ``%`` for LIKE).
                A bare ``*`` key is the catch-all default for unlisted banks.
                When set, consolidation tasks are claimed in priority tiers.
                None preserves current behavior (pure created_at ordering).

        Returns claimed rows with operation_id, operation_type, task_payload,
        retry_count, bank_id and serialization_key. The caller is responsible for
        building ClaimedTask objects.
        """
        ...

    @abstractmethod
    async def fetch_foldable_retain_peers(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        serialization_key: str,
        limit: int,
    ) -> list[ResultRow]:
        """Lock the pending retains queued behind a just-claimed one, in order.

        Called inside the claim transaction, so the rows come back locked and
        the caller can fold some of them into the claimed execution and leave
        the rest pending simply by not marking them (their locks release with
        the transaction).

        ``SKIP LOCKED`` matters here for liveness, not just speed: a peer some
        other worker is already looking at must never stall this claim.

        Returns rows with operation_id, task_payload and retry_count, ordered by
        ``(created_at, operation_id)`` — the order the fold planner requires.
        """
        ...

    @abstractmethod
    async def mark_operations_processing(
        self,
        conn: DatabaseConnection,
        table: str,
        worker_id: str,
        operation_ids: list,
    ) -> None:
        """Claim the given pending operations for ``worker_id``.

        Used to fold peers into an execution that has already been claimed;
        runs in the same transaction that locked them.
        """
        ...

    # -- Shared helpers (concrete) -----------------------------------------

    def _get_mu_table(self) -> str:
        """Get the fully-qualified memory_units table name."""
        from ..schema import fq_table

        return fq_table("memory_units")
