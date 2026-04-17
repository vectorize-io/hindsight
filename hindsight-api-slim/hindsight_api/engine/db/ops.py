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
from typing import Any
from uuid import UUID

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
    async def fetch_entity_unit_fanout(
        self,
        conn: DatabaseConnection,
        ue_table: str,
        entity_id_list: list[UUID],
        limit_per_entity: int,
    ) -> list[ResultRow]:
        """Fetch unit_ids for a list of entities with per-entity row cap.

        PG uses unnest + CROSS JOIN LATERAL with LIMIT.
        Non-PG queries each entity individually.
        """
        ...

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
        causal_weight_threshold: float,
    ) -> tuple[list[ResultRow], list[ResultRow], list[ResultRow]]:
        """Observation-specific graph expansion.

        PG uses native array ops (&&, unnest) on source_memory_ids.
        Non-PG uses JSON_TABLE to explode source_memory_ids CLOB.
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
