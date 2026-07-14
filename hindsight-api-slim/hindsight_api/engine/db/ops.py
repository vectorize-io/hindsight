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


@dataclass
class TagListingParts:
    """Backend-specific SQL fragments for the tag listing query."""

    tag_source: str
    non_empty_check: str
    tag_col: str
    bank_prefix: str


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
    ) -> dict[str, str]:
        """Bulk insert entities with ON CONFLICT DO NOTHING, returning id-by-lowercase-name.

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
    ) -> str:
        """Build entity expansion CTE for link expansion retrieval.

        PG uses DISTINCT ON with CROSS JOIN LATERAL and GROUP BY.
        Non-PG splits into entity_scores subquery then JOINs for full columns
        (can't GROUP BY CLOB).
        """
        ...

    @abstractmethod
    def build_semantic_causal_cte(
        self,
        ml_table: str,
        mu_table: str,
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
    ) -> tuple[list[ResultRow], list[ResultRow], list[ResultRow]]:
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

    @abstractmethod
    async def create_bank_vector_indexes(
        self,
        conn: DatabaseConnection,
        table: str,
        bank_id: str,
        internal_id: str,
        index_clause: str,
        fact_types: dict[str, str],
    ) -> None:
        """Create per-bank partial vector indexes.

        PG creates per-(bank, fact_type) partial indexes.
        Non-PG is a no-op (uses global index).
        """
        ...

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
    async def prune_orphan_entities(
        self,
        conn: DatabaseConnection,
        entities_table: str,
        ue_table: str,
        bank_id: str,
    ) -> int:
        """Delete entities in ``bank_id`` that no longer have any unit_entities
        rows referencing them. Returns the number of rows deleted.

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
        entities_table: str,
        bank_id: str,
    ) -> int:
        """Delete entity_cooccurrences rows in ``bank_id`` where the two
        entities still exist but no current unit references both of them.

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

        Args:
            consolidation_bank_priority: Per-bank priority for consolidation scheduling.
                Maps bank name patterns to integer priorities (higher = claimed first).
                Patterns support ``*`` as wildcard (converted to SQL ``%`` for LIKE).
                A bare ``*`` key is the catch-all default for unlisted banks.
                When set, consolidation tasks are claimed in priority tiers.
                None preserves current behavior (pure created_at ordering).

        Returns claimed rows with operation_id, operation_type, task_payload, retry_count.
        The caller is responsible for building ClaimedTask objects.
        """
        ...

    # -- Shared helpers (concrete) -----------------------------------------

    def _get_mu_table(self) -> str:
        """Get the fully-qualified memory_units table name."""
        from ..schema import fq_table

        return fq_table("memory_units")
