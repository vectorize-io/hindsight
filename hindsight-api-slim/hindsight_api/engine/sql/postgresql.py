"""PostgreSQL SQL dialect implementation.

Provides PostgreSQL-specific SQL fragments for parameter binding, JSON operators,
vector distance (pgvector), full-text search (VectorChord BM25 / tsvector),
and other non-portable patterns.
"""

from dataclasses import dataclass

from ..._text_search import mental_models_text_document
from .base import SQLDialect


@dataclass(frozen=True)
class KnowledgeBm25Arm:
    """Backend-specific BM25 clauses for the knowledge-page full-text arm.

    ``score_expr`` is a relevance score where higher = more relevant (used in the
    SELECT list of the BM25-only fallback). ``order_by`` is the arm's ranking
    expression (kept separate so distance-based backends can order by the raw,
    index-friendly ``ASC`` distance). ``match_filter`` is a WHERE predicate that
    keeps only genuine term matches, already prefixed with ``AND `` — empty when
    the backend ranks every row and needs no gate.
    """

    score_expr: str
    order_by: str
    match_filter: str


def knowledge_bm25_arm(
    text_search_extension: str,
    *,
    table_alias: str,
    text_param: str,
) -> KnowledgeBm25Arm:
    """BM25 clauses for ``search_knowledge_pages`` on a given text-search backend.

    Mirrors :meth:`PostgreSQLDialect.build_bm25_arm` (the memory-recall BM25 arm)
    but targets the ``mental_models`` BM25 index (``idx_mental_models_text_search``
    over the page ``name`` + ``content``) that backs knowledge pages. Without this
    dispatch the native tsvector SQL (``ts_rank_cd`` / ``@@``) is sent to every
    backend and 500s wherever ``mental_models.search_vector`` is not a tsvector
    (see issue #3268).

    ``table_alias`` is the alias the ``mental_models`` row carries in the query
    (``mm``); ``text_param`` is the bind placeholder holding the query text.

    ``pgroonga`` queries the multilingual expression index over ``name +
    content``; ``ensure_text_search_extension`` reconciles ``mental_models`` to
    that shape (dummy TEXT ``search_vector``) like every other pgroonga table.
    """
    a = table_alias
    p = text_param

    if text_search_extension == "vchord":
        # VectorChord BM25 over the bm25vector search_vector column, identical to
        # build_bm25_arm's vchord form. This only returns rows because the
        # mental_models write path now tokenizes search_vector for vchord
        # (pg_search_vector_expr, native_inline=False) the same way memory_units
        # does — the column is plain, not generated, so it must be filled on write.
        # <&> is the NEGATIVE score (lower = more relevant); negate it.
        expr = f"-({a}.search_vector <&> to_bm25query('idx_mental_models_text_search', tokenize({p}, 'llmlingua2')))"
        return KnowledgeBm25Arm(
            score_expr=expr,
            order_by=f"{expr} DESC",
            # Gate on a positive score: the operator ranks every row, so a bare
            # LIMIT would pad the arm with zero-score non-matches.
            match_filter=f"AND {expr} > 0",
        )

    if text_search_extension == "pg_search":
        # ParadeDB pg_search: BM25 index over (id, name, content), key_field='id'.
        # Fan the query across both indexed text fields with paradedb.boolean.
        score = f"paradedb.score({a}.id)"
        return KnowledgeBm25Arm(
            score_expr=score,
            order_by=f"{score} DESC",
            match_filter=(
                f"AND {a}.id @@@ paradedb.boolean(should => ARRAY["
                f"paradedb.match('name', {p}), paradedb.match('content', {p})])"
            ),
        )

    if text_search_extension == "pg_textsearch":
        # Timescale pg_textsearch: BM25 index over the `content` column. `<@>` is a
        # distance (lower = closer), so negate it for a higher-is-better score and
        # order by the raw ASC distance so the index drives the ordering.
        distance = f"{a}.content <@> to_bm25query({p}, 'idx_mental_models_text_search')"
        return KnowledgeBm25Arm(
            score_expr=f"-({distance})",
            order_by=f"{distance} ASC",
            match_filter="",
        )

    if text_search_extension == "pgroonga":
        # Same operator/score pair as build_bm25_arm's pgroonga form. The filter
        # repeats idx_mental_models_text_search's indexed expression verbatim (via
        # the shared helper) so the planner can select that expression index —
        # pgroonga_score() only returns a real score off a pgroonga index scan and
        # silently reads 0 for every row otherwise, so the id tiebreak keeps the
        # arm's ordering deterministic if the planner ever picks another plan.
        # pgroonga_query_escape neutralises operator characters in user text.
        score = f"pgroonga_score({a}.tableoid, {a}.ctid)"
        document = mental_models_text_document(a)
        return KnowledgeBm25Arm(
            score_expr=score,
            order_by=f"{score} DESC, {a}.id",
            match_filter=f"AND {document} &@~ pgroonga_query_escape({p})",
        )

    # native: generated tsvector over name + content.
    # The generating expression hard-codes the 'english' config (see the
    # learnings/pinned_reflections migration), so query with 'english' regardless
    # of the configured native language.
    score = f"ts_rank_cd({a}.search_vector, websearch_to_tsquery('english', {p}))"
    return KnowledgeBm25Arm(
        score_expr=score,
        order_by=f"{score} DESC",
        match_filter=f"AND {a}.search_vector @@ websearch_to_tsquery('english', {p})",
    )


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
        max_query_terms: int | None = None,
    ) -> str:
        if text_search_extension in ("vchord", "pg_textsearch", "pgroonga", "pg_search"):
            return query_text
        if max_query_terms is not None and max_query_terms > 0:
            tokens = tokens[:max_query_terms]
        # native tsvector: join tokens with OR operator
        return " | ".join(tokens)
