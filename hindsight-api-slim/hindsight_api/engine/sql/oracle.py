"""Oracle 23ai SQL dialect implementation.

Provides Oracle-specific SQL fragments for parameter binding, JSON operators,
vector distance (VECTOR_DISTANCE), full-text search (Oracle Text), and
other non-portable patterns.
"""

from ...config import get_config
from .base import SQLDialect, bm25_score_gate


class OracleDialect(SQLDialect):
    """SQL dialect for Oracle 23ai (python-oracledb)."""

    # -- Parameter binding -----------------------------------------------

    def param(self, n: int) -> str:
        return f":{n}"

    # -- Type casting ----------------------------------------------------

    def cast(self, param: str, type_name: str) -> str:
        # Oracle uses standard CAST syntax
        oracle_type = self._map_type(type_name)
        return f"CAST({param} AS {oracle_type})"

    @staticmethod
    def _map_type(pg_type: str) -> str:
        """Map PostgreSQL type names to Oracle equivalents."""
        mapping = {
            "jsonb": "CLOB",  # Oracle stores JSON in CLOB
            "json": "CLOB",
            "text": "VARCHAR2(4000)",
            "text[]": "CLOB",  # JSON array
            "uuid": "RAW(16)",
            "uuid[]": "CLOB",  # JSON array
            "varchar[]": "CLOB",  # JSON array
            "float8": "BINARY_DOUBLE",
            "float8[]": "CLOB",
            "timestamptz": "TIMESTAMP WITH TIME ZONE",
            "timestamptz[]": "CLOB",
            "vector": "VECTOR",
            "vector[]": "CLOB",
            "integer": "NUMBER",
            "bigint": "NUMBER",
            "boolean": "NUMBER(1)",
        }
        return mapping.get(pg_type, pg_type.upper())

    # -- Vector operations -----------------------------------------------

    def vector_distance(self, col: str, param: str) -> str:
        return f"VECTOR_DISTANCE({col}, {param}, COSINE)"

    def vector_similarity(self, col: str, param: str) -> str:
        return f"(1 - VECTOR_DISTANCE({col}, {param}, COSINE))"

    # -- JSON operations -------------------------------------------------

    def json_extract_text(self, col: str, key: str) -> str:
        return f"JSON_VALUE({col}, '$.{key}')"

    def json_contains(self, col: str, param: str) -> str:
        return f"JSON_EXISTS({col}, '$?(@  == {param})')"

    def json_merge(self, col: str, param: str) -> str:
        return f"JSON_MERGEPATCH({col}, {param})"

    # -- Text search -----------------------------------------------------

    def text_search_score(self, col: str, query_param: str, *, index_name: str | None = None) -> str:
        # Oracle Text: CONTAINS with SCORE
        return "SCORE(1)"

    def text_search_order(self, col: str, query_param: str, *, index_name: str | None = None) -> str:
        return "SCORE(1) DESC"

    # -- Fuzzy string matching -------------------------------------------

    def similarity(self, col: str, param: str) -> str:
        return f"UTL_MATCH.EDIT_DISTANCE_SIMILARITY({col}, {param}) / 100.0"

    # -- Upsert ----------------------------------------------------------

    def upsert(
        self,
        table: str,
        columns: list[str],
        conflict_columns: list[str],
        update_columns: list[str],
    ) -> str:
        col_list = ", ".join(columns)
        src_cols = ", ".join(f":{i + 1} AS {c}" for i, c in enumerate(columns))
        on_clause = " AND ".join(f"t.{c} = s.{c}" for c in conflict_columns)

        if not update_columns:
            return (
                f"MERGE INTO {table} t "
                f"USING (SELECT {src_cols} FROM DUAL) s "
                f"ON ({on_clause}) "
                f"WHEN NOT MATCHED THEN INSERT ({col_list}) "
                f"VALUES ({', '.join(f's.{c}' for c in columns)})"
            )

        updates = ", ".join(f"t.{c} = s.{c}" for c in update_columns)
        return (
            f"MERGE INTO {table} t "
            f"USING (SELECT {src_cols} FROM DUAL) s "
            f"ON ({on_clause}) "
            f"WHEN MATCHED THEN UPDATE SET {updates} "
            f"WHEN NOT MATCHED THEN INSERT ({col_list}) "
            f"VALUES ({', '.join(f's.{c}' for c in columns)})"
        )

    # -- Bulk operations -------------------------------------------------

    def bulk_unnest(self, param_types: list[tuple[str, str]]) -> str:
        # Oracle: use JSON_TABLE to expand a JSON array into rows
        # Caller passes a JSON array as the parameter
        columns = []
        for i, (param, sql_type) in enumerate(param_types):
            oracle_type = self._map_type(sql_type.rstrip("[]"))
            columns.append(f"c{i} {oracle_type} PATH '$[{i}]'")
        cols_spec = ", ".join(columns)
        # Using first param as the JSON array source
        first_param = param_types[0][0]
        return f"JSON_TABLE({first_param}, '$[*]' COLUMNS ({cols_spec}))"

    # -- Pagination ------------------------------------------------------

    def limit_offset(self, limit_param: str, offset_param: str) -> str:
        return f"OFFSET {offset_param} ROWS FETCH FIRST {limit_param} ROWS ONLY"

    # -- RETURNING clause ------------------------------------------------

    def returning(self, columns: list[str]) -> str:
        # Oracle RETURNING requires INTO clause with output bind variables.
        # The backend layer handles the output variable binding.
        return f"RETURNING {', '.join(columns)} INTO {', '.join(f':out_{c}' for c in columns)}"

    # -- Pattern matching ------------------------------------------------

    def ilike(self, col: str, param: str) -> str:
        return f"UPPER({col}) LIKE UPPER({param})"

    # -- Array operations ------------------------------------------------

    def array_any(self, param: str) -> str:
        # Oracle: expand JSON array to rows for IN clause
        return f"IN (SELECT value FROM JSON_TABLE({param}, '$[*]' COLUMNS (value PATH '$')))"

    def array_all(self, param: str) -> str:
        return f"NOT IN (SELECT value FROM JSON_TABLE({param}, '$[*]' COLUMNS (value PATH '$')))"

    def array_contains(self, col: str, param: str) -> str:
        # Oracle: check all elements of param array exist in col JSON array
        return (
            f"(SELECT COUNT(*) FROM JSON_TABLE({param}, '$[*]' COLUMNS (v PATH '$')) "
            f"WHERE JSON_EXISTS({col}, '$[*]?(@ == v)')) = "
            f"(SELECT COUNT(*) FROM JSON_TABLE({param}, '$[*]' COLUMNS (v PATH '$')))"
        )

    # -- Locking ---------------------------------------------------------

    def for_update_skip_locked(self) -> str:
        return "FOR UPDATE SKIP LOCKED"

    # -- UUID generation -------------------------------------------------

    def generate_uuid(self) -> str:
        return "SYS_GUID()"

    # -- Misc ------------------------------------------------------------

    def greatest(self, *args: str) -> str:
        return f"GREATEST({', '.join(args)})"

    def current_timestamp(self) -> str:
        return "SYSTIMESTAMP"

    def array_agg(self, expr: str) -> str:
        return f"JSON_ARRAYAGG({expr})"

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
        # Oracle 23ai: VECTOR_DISTANCE for cosine, FETCH FIRST for limiting.
        # Wrapped in a derived table to work within UNION ALL.
        # The row-limiting clause picks exact vs approximate search. EXACT must be spelled out:
        # on Autonomous Database a bare FETCH FIRST is answered from a vector index whenever one
        # exists (documented in "Perform Exact Similarity Search"), and with the baseline's global
        # IVF index plus the bank filter it returned 42% of the true top-20 on 26ai. APPROX is
        # the opt-in for large banks. Both values are inlined: the mode is validated against a
        # fixed set and the accuracy is an integer in [1, 100] (see HindsightConfig validation).
        config = get_config()
        if config.oracle_vector_search == "approx":
            fetch = (
                f"FETCH APPROX FIRST {fetch_limit} ROWS ONLY "
                f"WITH TARGET ACCURACY {int(config.oracle_vector_target_accuracy)}"
            )
        else:
            fetch = f"FETCH EXACT FIRST {fetch_limit} ROWS ONLY"
        return (
            f"SELECT * FROM (SELECT {cols},"
            f"        1 - VECTOR_DISTANCE(embedding, {embedding_param}, COSINE) AS similarity,"
            f"        NULL AS bm25_score,"
            f"        'semantic' AS source"
            f" FROM {table}"
            f" WHERE bank_id = {bank_id_param}"
            f"   AND fact_type = '{fact_type}'"
            f"   AND embedding IS NOT NULL"
            f"   AND (1 - VECTOR_DISTANCE(embedding, {embedding_param}, COSINE)) >= {min_similarity}"
            f"   {tags_clause}"
            f"   {groups_clause}"
            f"   {extra_where}"
            f" ORDER BY VECTOR_DISTANCE(embedding, {embedding_param}, COSINE)"
            f" {fetch}) t"
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
        pg_search_function_schema: str = "paradedb",
        pg_search_tokenizer: str = "",
        max_query_terms: int = 0,
        extra_where: str = "",
    ) -> str:
        # Oracle Text: CONTAINS() / SCORE() with the CTXSYS.CONTEXT index.
        # Each arm gets a unique SCORE label (10 + arm_index) to avoid
        # conflicts within the UNION ALL.
        label = 10 + arm_index
        return (
            f"SELECT * FROM (SELECT {cols},"
            f"        NULL AS similarity,"
            f"        SCORE({label}) AS bm25_score,"
            f"        'bm25' AS source"
            f" FROM {table}"
            f" WHERE bank_id = {bank_id_param}"
            f"   AND fact_type = '{fact_type}'"
            # CONTAINS already gates to genuine matches, so at the 0.0 default the
            # gate is the structural `> 0`; a caller's `min_scores.keyword` floor
            # replaces it with an inclusive `>=`, uniform across backends.
            f"   AND CONTAINS(text, {text_param}, {label}) {bm25_score_gate(bm25_min_score)}"
            f"   {tags_clause}"
            f"   {groups_clause}"
            f"   {extra_where}"
            f" ORDER BY SCORE({label}) DESC"
            f" FETCH FIRST {limit_param} ROWS ONLY) t{arm_index}"
        )

    def prepare_bm25_text(
        self,
        tokens: list[str],
        query_text: str,
        *,
        text_search_extension: str = "native",
        max_query_terms: int | None = None,
    ) -> str:
        # Oracle Text: wrap every term in braces, which makes CONTAINS read it literally.
        # Previously only reserved words (NEAR, ABOUT, ...) were braced and terms with
        # operator characters were dropped, but `_` — Oracle Text's one-character wildcard,
        # and a word character to the tokenizer — went through raw: on a live 26ai index
        # 'hindsight_api' matched 0 rows (a wildcard pattern against the lexer's two tokens)
        # where '{hindsight_api}' matched 584, and a lone '_' matched every one-letter token.
        # Braces are stripped from the term so it cannot close the escape early, and the raw
        # query text is never bound: the old all-filtered fallback braced it verbatim, so a
        # '}' in it reopened the expression to operators.
        terms: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            term = token.replace("{", " ").replace("}", " ").strip()
            if not term or term.lower() in seen:
                continue
            seen.add(term.lower())
            terms.append(f"{{{term}}}")
        # Same cap as the PostgreSQL native path; without it a long question became an
        # unbounded OR over every token.
        if max_query_terms:
            terms = terms[:max_query_terms]
        # ACCUM, not OR: OR scores a row by its best single term, so a common word ranks as
        # high as the rare one the question is about; ACCUM ranks rows matching more terms
        # higher and matches the same rows. On BEIR SciFact (100 queries, Gemini 1536, 26ai)
        # keyword nDCG@10 went 0.348 -> 0.666 and the RRF hybrid 0.693 -> 0.806.
        return " ACCUM ".join(terms)
