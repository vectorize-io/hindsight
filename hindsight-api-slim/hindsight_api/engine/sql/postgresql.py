"""PostgreSQL SQL dialect implementation.

Provides PostgreSQL-specific SQL fragments for parameter binding, JSON operators,
vector distance (pgvector), full-text search (VectorChord BM25 / tsvector),
and other non-portable patterns.
"""

from .base import SQLDialect


class PostgreSQLDialect(SQLDialect):
    """SQL dialect for PostgreSQL (asyncpg)."""

    # -- Parameter binding -----------------------------------------------

    def param(self, n: int) -> str:
        return f"${n}"

    # -- Type casting ----------------------------------------------------

    def cast(self, param: str, type_name: str) -> str:
        return f"{param}::{type_name}"

    # -- Vector operations -----------------------------------------------

    def vector_distance(self, col: str, param: str) -> str:
        return f"{col} <=> {param}::vector"

    def vector_similarity(self, col: str, param: str) -> str:
        return f"1 - ({col} <=> {param}::vector)"

    # -- JSON operations -------------------------------------------------

    def json_extract_text(self, col: str, key: str) -> str:
        return f"{col} ->> '{key}'"

    def json_contains(self, col: str, param: str) -> str:
        return f"{col} @> {param}::jsonb"

    def json_merge(self, col: str, param: str) -> str:
        return f"{col} || {param}::jsonb"

    # -- Text search -----------------------------------------------------

    def text_search_score(self, col: str, query_param: str, *, index_name: str | None = None) -> str:
        if index_name:
            # VectorChord BM25
            return f"-({col} <@> to_bm25query({query_param}, '{index_name}'))"
        # Fallback to tsvector
        return f"ts_rank_cd({col}, to_tsquery({query_param}))"

    def text_search_order(self, col: str, query_param: str, *, index_name: str | None = None) -> str:
        if index_name:
            # VectorChord BM25 — lower distance = better, so ASC
            return f"{col} <@> to_bm25query({query_param}, '{index_name}') ASC"
        return f"ts_rank_cd({col}, to_tsquery({query_param})) DESC"

    # -- Fuzzy string matching -------------------------------------------

    def similarity(self, col: str, param: str) -> str:
        return f"similarity({col}, {param})"

    # -- Upsert ----------------------------------------------------------

    def upsert(
        self,
        table: str,
        columns: list[str],
        conflict_columns: list[str],
        update_columns: list[str],
    ) -> str:
        col_list = ", ".join(columns)
        placeholders = ", ".join(f"${i + 1}" for i in range(len(columns)))
        conflict = ", ".join(conflict_columns)

        if not update_columns:
            return f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) ON CONFLICT ({conflict}) DO NOTHING"

        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_columns)
        return (
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) ON CONFLICT ({conflict}) DO UPDATE SET {updates}"
        )

    # -- Bulk operations -------------------------------------------------

    def bulk_unnest(self, param_types: list[tuple[str, str]]) -> str:
        args = ", ".join(f"{p}::{t}" for p, t in param_types)
        return f"unnest({args})"

    # -- Pagination ------------------------------------------------------

    def limit_offset(self, limit_param: str, offset_param: str) -> str:
        return f"LIMIT {limit_param} OFFSET {offset_param}"

    # -- RETURNING clause ------------------------------------------------

    def returning(self, columns: list[str]) -> str:
        return f"RETURNING {', '.join(columns)}"

    # -- Pattern matching ------------------------------------------------

    def ilike(self, col: str, param: str) -> str:
        return f"{col} ILIKE {param}"

    # -- Array operations ------------------------------------------------

    def array_any(self, param: str) -> str:
        return f"= ANY({param})"

    def array_all(self, param: str) -> str:
        return f"!= ALL({param})"

    def array_contains(self, col: str, param: str) -> str:
        return f"{col} @> {param}::varchar[]"

    # -- Locking ---------------------------------------------------------

    def for_update_skip_locked(self) -> str:
        return "FOR UPDATE SKIP LOCKED"

    def advisory_lock(self, id_param: str) -> str:
        return f"pg_try_advisory_lock({id_param})"

    # -- UUID generation -------------------------------------------------

    def generate_uuid(self) -> str:
        return "gen_random_uuid()"

    # -- Misc ------------------------------------------------------------

    def greatest(self, *args: str) -> str:
        return f"GREATEST({', '.join(args)})"

    def current_timestamp(self) -> str:
        return "now()"

    def array_agg(self, expr: str) -> str:
        return f"array_agg({expr})"

    # -- Retrieval query arms ----------------------------------------------

    def build_semantic_arm(
        self,
        *,
        table: str,
        cols: str,
        fact_type: str,
        embedding_param: str,
        bank_id_param: str,
        fetch_limit: int,
        min_similarity: float,
        tags_clause: str = "",
        groups_clause: str = "",
        extra_where: str = "",
    ) -> str:
        return (
            f"(SELECT {cols},"
            f"        1 - (embedding <=> {embedding_param}::vector) AS similarity,"
            f"        NULL::float AS bm25_score,"
            f"        'semantic' AS source"
            f" FROM {table}"
            f" WHERE bank_id = {bank_id_param}"
            f"   AND fact_type = '{fact_type}'"
            f"   AND embedding IS NOT NULL"
            f"   AND (1 - (embedding <=> {embedding_param}::vector)) >= {min_similarity}"
            f"   {tags_clause}"
            f"   {groups_clause}"
            f"   {extra_where}"
            f" ORDER BY embedding <=> {embedding_param}::vector"
            f" LIMIT {fetch_limit})"
        )

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
        bm25_language: str = "english",
        bm25_min_score: float = 0.0,
        extra_where: str = "",
    ) -> str:
        if text_search_extension == "vchord":
            # <&> returns the NEGATIVE BM25 score (lower = more relevant), negate
            # for a positive score where higher = more relevant.
            bm25_score_expr = f"-(search_vector <&> to_bm25query('idx_memory_units_text_search', tokenize({text_param}, 'llmlingua2')))"
            bm25_order_by = f"{bm25_score_expr} DESC"
            # Unlike native tsvector (which has a boolean `@@` match gate), the
            # VectorChord operator ranks *every* document, so a bare ORDER BY ...
            # LIMIT pads the result with zero-score, non-matching rows. Gate on the
            # score so only genuine term matches survive into fusion/reranking.
            bm25_where_filter = f"AND -(search_vector <&> to_bm25query('idx_memory_units_text_search', tokenize({text_param}, 'llmlingua2'))) > {bm25_min_score:g}"
        elif text_search_extension == "pg_textsearch":
            bm25_score_expr = f"-({text_param} <@> to_bm25query({text_param}, 'idx_memory_units_text_search'))"
            bm25_order_by = f"text <@> to_bm25query({text_param}, 'idx_memory_units_text_search') ASC"
            bm25_where_filter = ""
        elif text_search_extension == "pgroonga":
            # &@~ accepts pgroonga's query syntax. Escape the bind parameter so
            # literal memory text containing operators like ">" or "(" is not
            # parsed as a malformed query expression.
            bm25_score_expr = "pgroonga_score(tableoid, ctid)"
            bm25_order_by = f"{bm25_score_expr} DESC"
            bm25_where_filter = (
                f"AND (COALESCE(text, '') || ' ' || COALESCE(context, '') || ' ' || COALESCE(text_signals, '')) "
                f"&@~ pgroonga_query_escape({text_param})"
            )
        elif text_search_extension == "pg_search":
            # ParadeDB pg_search: BM25 index over (id, text, context, text_signals)
            # with key_field='id'. The @@@ operator on the key_field requires a
            # field-qualified query (`text:foo`); to keep the bind-parameter form,
            # we fan the query out across all indexed text fields with paradedb.boolean.
            bm25_score_expr = "paradedb.score(id)"
            bm25_order_by = "paradedb.score(id) DESC"
            bm25_where_filter = (
                f"AND id @@@ paradedb.boolean(should => ARRAY["
                f"paradedb.match('text', {text_param}), "
                f"paradedb.match('context', {text_param}), "
                f"paradedb.match('text_signals', {text_param})"
                f"])"
            )
        else:  # native tsvector
            # bm25_language is validated as a PG identifier in HindsightConfig.validate(),
            # so embedding it as a SQL literal here is safe.
            bm25_score_expr = f"ts_rank_cd(search_vector, to_tsquery('{bm25_language}', {text_param}))"
            bm25_order_by = f"{bm25_score_expr} DESC"
            bm25_where_filter = f"AND search_vector @@ to_tsquery('{bm25_language}', {text_param})"

        return (
            f"(SELECT {cols},"
            f"        NULL::float AS similarity,"
            f"        {bm25_score_expr} AS bm25_score,"
            f"        'bm25' AS source"
            f" FROM {table}"
            f" WHERE bank_id = {bank_id_param}"
            f"   AND fact_type = '{fact_type}'"
            f"   {bm25_where_filter}"
            f"   {tags_clause}"
            f"   {groups_clause}"
            f"   {extra_where}"
            f" ORDER BY {bm25_order_by}"
            f" LIMIT {limit_param})"
        )

    def prepare_bm25_text(
        self,
        tokens: list[str],
        query_text: str,
        *,
        text_search_extension: str = "native",
    ) -> str:
        if text_search_extension in ("vchord", "pg_textsearch", "pgroonga", "pg_search"):
            return query_text
        # native tsvector: join tokens with OR operator
        return " | ".join(tokens)
