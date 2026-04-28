"""Oracle 23ai SQL dialect implementation.

Provides Oracle-specific SQL fragments for parameter binding, JSON operators,
vector distance (VECTOR_DISTANCE), full-text search (Oracle Text), and
other non-portable patterns.
"""

from .base import SQLDialect


class OracleDialect(SQLDialect):
    """SQL dialect for Oracle 23ai (python-oracledb)."""

    # Characters that need escaping in Oracle Text CONTAINS queries.
    _ORACLE_TEXT_SPECIAL = frozenset("&|!{}()[]~*?%-$>")

    # Oracle Text reserved words that must be escaped with curly braces
    # when used as plain search terms.  Full list from Oracle Text docs:
    # ABOUT, AND, BT, BTG, BTI, BTP, EQUIV, FUZZY, HASPATH, INPATH,
    # MINUS, NEAR, NOT, NT, NTG, NTI, NTP, OR, PT, RT, SQE, SYN,
    # TR, TRSYN, TT, WITHIN.
    _ORACLE_TEXT_RESERVED = frozenset(
        {
            "about",
            "and",
            "bt",
            "btg",
            "bti",
            "btp",
            "equiv",
            "fuzzy",
            "haspath",
            "inpath",
            "minus",
            "near",
            "not",
            "nt",
            "ntg",
            "nti",
            "ntp",
            "or",
            "pt",
            "rt",
            "sqe",
            "syn",
            "tr",
            "trsyn",
            "tt",
            "within",
        }
    )

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

    def advisory_lock(self, id_param: str) -> str:
        # Oracle doesn't have advisory locks. Use SELECT FOR UPDATE NOWAIT on a lock row.
        return "SELECT 1 FROM dual FOR UPDATE NOWAIT"

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
        tags_clause: str = "",
        groups_clause: str = "",
        extra_where: str = "",
    ) -> str:
        # Oracle 23ai: VECTOR_DISTANCE for cosine, FETCH FIRST for limiting.
        # Wrapped in a derived table to work within UNION ALL.
        return (
            f"SELECT * FROM (SELECT {cols},"
            f"        1 - VECTOR_DISTANCE(embedding, {embedding_param}, COSINE) AS similarity,"
            f"        NULL AS bm25_score,"
            f"        'semantic' AS source"
            f" FROM {table}"
            f" WHERE bank_id = {bank_id_param}"
            f"   AND fact_type = '{fact_type}'"
            f"   AND embedding IS NOT NULL"
            f"   AND (1 - VECTOR_DISTANCE(embedding, {embedding_param}, COSINE)) >= 0.3"
            f"   {tags_clause}"
            f"   {groups_clause}"
            f"   {extra_where}"
            f" ORDER BY VECTOR_DISTANCE(embedding, {embedding_param}, COSINE)"
            f" FETCH FIRST {fetch_limit} ROWS ONLY) t"
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
            f"   AND CONTAINS(text, {text_param}, {label}) > 0"
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
    ) -> str:
        # Oracle Text: filter tokens with special chars, escape reserved words
        # with curly braces (e.g. "about" → "{about}"), and join with OR.
        safe: list[str] = []
        for t in tokens:
            if any(c in self._ORACLE_TEXT_SPECIAL for c in t):
                continue
            if t.lower() in self._ORACLE_TEXT_RESERVED:
                safe.append(f"{{{t}}}")
            else:
                safe.append(t)
        return " OR ".join(safe) if safe else f"{{{tokens[0]}}}"
