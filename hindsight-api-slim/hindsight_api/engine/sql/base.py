"""Abstract base class for SQL dialect modules.

Each method encapsulates a SQL pattern that differs between database platforms.
Business logic calls these methods instead of embedding raw SQL fragments.
"""

from abc import ABC, abstractmethod


class SQLDialect(ABC):
    """SQL dialect interface for portable query construction.

    Implementors provide database-specific SQL fragments for operations that
    are not standard across PostgreSQL and Oracle (parameter binding, JSON
    operators, vector distance, full-text search, etc.).
    """

    # -- Parameter binding -----------------------------------------------

    @abstractmethod
    def param(self, n: int) -> str:
        """Return the nth positional parameter placeholder.

        Args:
            n: 1-based parameter index.

        Returns:
            "$1" for PostgreSQL, ":1" for Oracle.
        """
        ...

    # -- Type casting ----------------------------------------------------

    @abstractmethod
    def cast(self, param: str, type_name: str) -> str:
        """Cast a parameter or expression to the given type.

        Args:
            param: The expression to cast (e.g. "$1" or a column name).
            type_name: Target type (e.g. "jsonb", "uuid[]", "vector").

        Returns:
            Cast expression (e.g. "$1::jsonb" for PG, "CAST(:1 AS ...)" for Oracle).
        """
        ...

    # -- Vector operations -----------------------------------------------

    @abstractmethod
    def vector_distance(self, col: str, param: str) -> str:
        """Cosine distance expression between a column and a parameter.

        Args:
            col: Column name containing the vector.
            param: Parameter placeholder for the query vector.

        Returns:
            Distance expression (lower = more similar).
            PG: "col <=> $1::vector"
            Oracle: "VECTOR_DISTANCE(col, :1, COSINE)"
        """
        ...

    @abstractmethod
    def vector_similarity(self, col: str, param: str) -> str:
        """Cosine similarity expression (1 - distance).

        Args:
            col: Column name.
            param: Parameter placeholder.

        Returns:
            Similarity expression (higher = more similar).
        """
        ...

    # -- JSON operations -------------------------------------------------

    @abstractmethod
    def json_extract_text(self, col: str, key: str) -> str:
        """Extract a text value from a JSON/JSONB column.

        Args:
            col: Column name.
            key: JSON key to extract.

        Returns:
            PG: "col ->> 'key'"
            Oracle: "JSON_VALUE(col, '$.key')"
        """
        ...

    @abstractmethod
    def json_contains(self, col: str, param: str) -> str:
        """Test whether a JSON column contains the given JSON object.

        Args:
            col: Column name.
            param: Parameter placeholder for the JSON object to test.

        Returns:
            PG: "col @> $1::jsonb"
            Oracle: "JSON_EXISTS(col, ...)"
        """
        ...

    @abstractmethod
    def json_merge(self, col: str, param: str) -> str:
        """Merge (concatenate) a JSON object into a JSON column.

        Args:
            col: Column name.
            param: Parameter placeholder for the JSON to merge.

        Returns:
            PG: "col || $1::jsonb"
            Oracle: "JSON_MERGEPATCH(col, :1)"
        """
        ...

    # -- Text search -----------------------------------------------------

    @abstractmethod
    def text_search_score(self, col: str, query_param: str, *, index_name: str | None = None) -> str:
        """Relevance score expression for full-text search.

        Args:
            col: Column name (text or tsvector/bm25vector).
            query_param: Parameter placeholder for the search query.
            index_name: Optional index name (needed by some backends).

        Returns:
            Score expression (higher = more relevant).
        """
        ...

    @abstractmethod
    def text_search_order(self, col: str, query_param: str, *, index_name: str | None = None) -> str:
        """ORDER BY expression for full-text search (ascending = best first).

        Args:
            col: Column name.
            query_param: Parameter placeholder for the search query.
            index_name: Optional index name.

        Returns:
            Expression suitable for ORDER BY ... ASC.
        """
        ...

    # -- Fuzzy string matching -------------------------------------------

    @abstractmethod
    def similarity(self, col: str, param: str) -> str:
        """Fuzzy string similarity score between a column and a parameter.

        Args:
            col: Column name.
            param: Parameter placeholder.

        Returns:
            PG: "similarity(col, $1)"
            Oracle: "UTL_MATCH.EDIT_DISTANCE_SIMILARITY(col, :1) / 100.0"
        """
        ...

    # -- Upsert ----------------------------------------------------------

    @abstractmethod
    def upsert(
        self,
        table: str,
        columns: list[str],
        conflict_columns: list[str],
        update_columns: list[str],
    ) -> str:
        """Generate an upsert statement.

        Args:
            table: Fully-qualified table name.
            columns: All columns in the INSERT.
            conflict_columns: Columns that form the unique constraint.
            update_columns: Columns to update on conflict.

        Returns:
            Complete INSERT ... ON CONFLICT DO UPDATE (PG)
            or MERGE INTO ... (Oracle) statement.
        """
        ...

    # -- Bulk operations -------------------------------------------------

    @abstractmethod
    def bulk_unnest(self, param_types: list[tuple[str, str]]) -> str:
        """Generate a bulk unnest/table-value expression.

        Converts parallel arrays into rows.

        Args:
            param_types: List of (param_placeholder, sql_type) pairs
                         e.g. [("$1", "text[]"), ("$2", "uuid[]")]

        Returns:
            PG: "unnest($1::text[], $2::uuid[])"
            Oracle: JSON_TABLE-based equivalent.
        """
        ...

    # -- Pagination ------------------------------------------------------

    @abstractmethod
    def limit_offset(self, limit_param: str, offset_param: str) -> str:
        """Generate LIMIT/OFFSET clause.

        Args:
            limit_param: Parameter placeholder for row limit.
            offset_param: Parameter placeholder for row offset.

        Returns:
            PG: "LIMIT $1 OFFSET $2"
            Oracle: "OFFSET :2 ROWS FETCH FIRST :1 ROWS ONLY"
        """
        ...

    # -- RETURNING clause ------------------------------------------------

    @abstractmethod
    def returning(self, columns: list[str]) -> str:
        """Generate a RETURNING clause.

        Args:
            columns: Column names to return.

        Returns:
            PG: "RETURNING col1, col2"
            Oracle: "RETURNING col1, col2 INTO :out1, :out2" (handled by backend).
        """
        ...

    # -- Pattern matching ------------------------------------------------

    @abstractmethod
    def ilike(self, col: str, param: str) -> str:
        """Case-insensitive LIKE expression.

        Args:
            col: Column name.
            param: Parameter placeholder for the pattern.

        Returns:
            PG: "col ILIKE $1"
            Oracle: "UPPER(col) LIKE UPPER(:1)"
        """
        ...

    # -- Array operations ------------------------------------------------

    @abstractmethod
    def array_any(self, param: str) -> str:
        """IN-array membership expression.

        Args:
            param: Parameter placeholder for the array.

        Returns:
            PG: "= ANY($1)"
            Oracle: "IN (SELECT ... FROM JSON_TABLE(...))"
        """
        ...

    @abstractmethod
    def array_all(self, param: str) -> str:
        """NOT-IN-array expression (not equal to all elements).

        Args:
            param: Parameter placeholder for the array.

        Returns:
            PG: "!= ALL($1)"
        """
        ...

    @abstractmethod
    def array_contains(self, col: str, param: str) -> str:
        """Test whether an array column contains all elements in the parameter.

        Args:
            col: Array column name.
            param: Parameter placeholder for the array to test.

        Returns:
            PG: "col @> $1::varchar[]"
        """
        ...

    # -- Locking ---------------------------------------------------------

    @abstractmethod
    def for_update_skip_locked(self) -> str:
        """FOR UPDATE SKIP LOCKED clause (same on both PG and Oracle)."""
        ...

    @abstractmethod
    def advisory_lock(self, id_param: str) -> str:
        """Advisory lock expression.

        Args:
            id_param: Parameter placeholder for the lock ID.

        Returns:
            PG: "pg_try_advisory_lock($1)"
            Oracle: "SELECT ... FOR UPDATE NOWAIT" equivalent.
        """
        ...

    # -- UUID generation -------------------------------------------------

    @abstractmethod
    def generate_uuid(self) -> str:
        """SQL expression to generate a random UUID.

        Returns:
            PG: "gen_random_uuid()"
            Oracle: "SYS_GUID()"
        """
        ...

    # -- Misc ------------------------------------------------------------

    @abstractmethod
    def greatest(self, *args: str) -> str:
        """GREATEST() function (same on both platforms)."""
        ...

    @abstractmethod
    def current_timestamp(self) -> str:
        """Current timestamp expression.

        Returns:
            PG: "now()"
            Oracle: "SYSTIMESTAMP"
        """
        ...

    @abstractmethod
    def array_agg(self, expr: str) -> str:
        """Aggregate values into an array.

        Args:
            expr: Expression to aggregate.

        Returns:
            PG: "array_agg(expr)"
            Oracle: "CAST(COLLECT(expr) AS ...)" or JSON_ARRAYAGG.
        """
        ...

    # -- Retrieval query arms ----------------------------------------------
    # These build complete subquery arms for the UNION ALL retrieval query.
    # Each database has significantly different syntax for vector search and
    # full-text search, so these belong in the dialect rather than inline
    # conditionals in retrieval.py.

    @abstractmethod
    def build_semantic_arm(
        self,
        *,
        table: str,
        cols: str,
        fact_type: str,
        embedding_param: str,
        bank_id_param: str,
        fetch_limit: int,
        tags_clause: str = "",
        groups_clause: str = "",
        extra_where: str = "",
    ) -> str:
        """Build a semantic (vector similarity) search subquery arm.

        Returns a complete subquery suitable for UNION ALL that selects
        matching rows ordered by cosine similarity.

        Args:
            table: Fully-qualified table name.
            cols: Column list expression.
            fact_type: Fact type literal (inlined, not parameterized).
            embedding_param: Parameter placeholder for query embedding.
            bank_id_param: Parameter placeholder for bank_id.
            fetch_limit: Max rows to fetch (over-fetched for HNSW approximation).
            tags_clause: Optional WHERE clause fragment for tag filtering.
            groups_clause: Optional WHERE clause fragment for tag group filtering.
            extra_where: Optional additional WHERE clause fragment (e.g. time range filter).
        """
        ...

    @abstractmethod
    def build_bm25_arm(
        self,
        *,
        table: str,
        cols: str,
        fact_type: str,
        bank_id_param: str,
        limit_param: str,
        text_param: str,
        tags_clause: str = "",
        groups_clause: str = "",
        arm_index: int = 0,
        text_search_extension: str = "native",
        extra_where: str = "",
    ) -> str:
        """Build a BM25/full-text search subquery arm.

        Returns a complete subquery suitable for UNION ALL that selects
        matching rows ordered by text relevance score.

        Args:
            table: Fully-qualified table name.
            cols: Column list expression.
            fact_type: Fact type literal (inlined, not parameterized).
            bank_id_param: Parameter placeholder for bank_id.
            limit_param: Parameter placeholder for result limit.
            text_param: Parameter placeholder for the search text.
            tags_clause: Optional WHERE clause fragment for tag filtering.
            groups_clause: Optional WHERE clause fragment for tag group filtering.
            arm_index: Index of this arm in the UNION ALL (used by Oracle for
                       unique SCORE labels).
            text_search_extension: Full-text search backend ("native", "vchord",
                                   "pg_textsearch"). Only relevant for PostgreSQL.
            extra_where: Optional additional WHERE clause fragment (e.g. time range filter).
        """
        ...

    @abstractmethod
    def prepare_bm25_text(
        self,
        tokens: list[str],
        query_text: str,
        *,
        text_search_extension: str = "native",
    ) -> str:
        """Prepare the text parameter value for BM25 search.

        Transforms tokens/query text into the format expected by the backend's
        full-text search engine.

        Args:
            tokens: Tokenized query words.
            query_text: Original query text.
            text_search_extension: Full-text search backend variant.

        Returns:
            Prepared text string to bind as the BM25 text parameter.
        """
        ...
