"""Backend dispatch for the knowledge-page BM25 arm (issue #3268).

``search_knowledge_pages`` used to hard-code the native tsvector SQL
(``ts_rank_cd`` / ``@@``) for every text-search backend, so it 500'd on
``pg_search`` / ``pg_textsearch`` / ``vchord`` where ``mental_models.search_vector``
is not a tsvector. These tests pin the per-backend SQL the read dispatcher emits,
and the write-side ``search_vector`` tokenization the ``mental_models`` insert/
update reuse from the memory-recall path. They need no live extension because they
assert on the generated SQL, the way ``test_multilingual_bm25`` does.
"""

from types import SimpleNamespace
from typing import Any

import pytest

from hindsight_api._text_search import mental_models_text_document
from hindsight_api.engine.db.ops_postgresql import pg_search_vector_expr
from hindsight_api.engine.sql.postgresql import KnowledgeBm25Arm, knowledge_bm25_arm

# Asserts the SQL text this dispatch emits -- the `$3::text` bind and the absence of
# `search_vector`. A store that owns the knowledge index emits no SQL at all, so there is nothing
# here for the assertions to read.
pytestmark = pytest.mark.memory_backend_incompatible


def _cfg(ext: str) -> SimpleNamespace:
    return SimpleNamespace(text_search_extension=ext, text_search_extension_native_language="english")


def _arm(ext: str) -> KnowledgeBm25Arm:
    return knowledge_bm25_arm(ext, table_alias="mm", text_param="$3")


def test_native_uses_tsvector_operators():
    arm = _arm("native")
    # The mental_models tsvector is generated with the 'english' config, so the
    # query must use 'english' regardless of the configured native language.
    # Joining tokens with OR aligns candidate recall with memory-recall BM25;
    # precision is restored downstream via ts_rank_cd ranking and RRF fusion.
    assert "ts_rank_cd(mm.search_vector, to_tsquery('english', $3))" in arm.score_expr
    assert arm.match_filter == "AND mm.search_vector @@ to_tsquery('english', $3)"


def test_pgroonga_uses_multilingual_expression_index():
    arm = _arm("pgroonga")
    assert arm.score_expr == "pgroonga_score(mm.tableoid, mm.ctid)"
    assert "pgroonga_tokenize($3, 'tokenizer', 'TokenBigram', 'normalizer', 'NormalizerNFKC150')" in arm.match_filter
    assert "string_agg(pgroonga_query_escape(elem->>'value'), ' OR ')" in arm.match_filter
    # pgroonga_score() reads 0 off any plan that did not use the pgroonga index,
    # so the ordering carries a tiebreak instead of collapsing to input order.
    assert arm.order_by == "pgroonga_score(mm.tableoid, mm.ctid) DESC, mm.id"


def test_pgroonga_filter_repeats_the_indexed_expression_verbatim():
    """The expression index is only selectable when the query repeats its
    expression exactly — so both sides must come from the shared helper."""
    assert mental_models_text_document("mm") in _arm("pgroonga").match_filter
    assert mental_models_text_document() == "(COALESCE(name, '') || ' ' || content)"


def test_pg_search_uses_paradedb_over_base_columns():
    arm = _arm("pg_search")
    assert arm.score_expr == "paradedb.score(mm.id)"
    assert "mm.id @@@ paradedb.boolean(should => ARRAY[" in arm.match_filter
    assert "paradedb.match('name', $3)" in arm.match_filter
    assert "paradedb.match('content', $3)" in arm.match_filter
    # Must not fall back to the native tsvector function.
    assert "ts_rank_cd" not in arm.score_expr
    assert "ts_rank_cd" not in arm.order_by


def test_pg_search_uses_custom_function_schema():
    arm = knowledge_bm25_arm("pg_search", table_alias="mm", text_param="$3", pg_search_function_schema="pgsearch")
    assert arm.score_expr == "pgsearch.score(mm.id)"
    assert "mm.id @@@ pgsearch.boolean(should => ARRAY[" in arm.match_filter
    assert "pgsearch.match('name', $3)" in arm.match_filter
    assert "pgsearch.match('content', $3)" in arm.match_filter
    assert "paradedb" not in arm.score_expr
    assert "paradedb" not in arm.match_filter


def test_pg_textsearch_ranks_content_by_bm25_distance():
    arm = _arm("pg_textsearch")
    # `<@>` is a distance (lower = closer): order ASC, negate for the score.
    assert arm.order_by == "mm.content <@> to_bm25query($3, 'idx_mental_models_text_search') ASC"
    assert arm.score_expr == "-(mm.content <@> to_bm25query($3, 'idx_mental_models_text_search'))"
    # It ranks every row, so there is no boolean match gate.
    assert arm.match_filter == ""
    assert "ts_rank_cd" not in arm.order_by


def test_vchord_ranks_over_bm25vector_search_vector():
    arm = _arm("vchord")
    # Negated <&> distance over the bm25vector column and the mental_models index,
    # gated on a positive score — the same operator build_bm25_arm uses.
    assert (
        arm.score_expr
        == "-(mm.search_vector <&> to_bm25query('idx_mental_models_text_search', tokenize($3, 'llmlingua2')))"
    )
    assert arm.order_by.endswith(" DESC")
    assert arm.match_filter.endswith(" > 0")
    assert "ts_rank_cd" not in arm.score_expr


def test_text_param_and_alias_are_threaded_through():
    arm = knowledge_bm25_arm("pg_search", table_alias="kbm", text_param="$7")
    assert "paradedb.score(kbm.id)" == arm.score_expr
    assert "kbm.id @@@" in arm.match_filter
    assert "paradedb.match('name', $7)" in arm.match_filter


# --- write side: search_vector tokenization for mental_models ---------------
# mental_models is a two-column (name + content) table whose native search_vector
# is a GENERATED column, so only vchord needs an inline write (native_inline=False).


def test_mm_write_tokenizes_only_for_vchord():
    for ext in ("native", "pgroonga", "pg_search", "pg_textsearch"):
        assert (
            pg_search_vector_expr(_cfg(ext), text_col="$3", context_col="$5", signals_col=None, native_inline=False)
            is None
        ), f"{ext} must leave mental_models.search_vector unwritten"

    vchord = pg_search_vector_expr(
        _cfg("vchord"), text_col="$3", context_col="$5", signals_col=None, native_inline=False
    )
    assert vchord == "tokenize(COALESCE($3, '') || ' ' || COALESCE($5, ''), 'llmlingua2')::bm25_catalog.bm25vector"


def test_memory_units_default_expr_is_unchanged():
    # The recall/insert path keeps its three-column, native-inline behaviour.
    assert pg_search_vector_expr(_cfg("native")) == (
        "to_tsvector('english'::regconfig, "
        "COALESCE(text, '') || ' ' || COALESCE(context, '') || ' ' || COALESCE(text_signals, ''))"
    )
    assert pg_search_vector_expr(_cfg("vchord")) == (
        "tokenize(COALESCE(text, '') || ' ' || COALESCE(context, '') || ' ' || COALESCE(text_signals, ''), "
        "'llmlingua2')::bm25_catalog.bm25vector"
    )
    assert pg_search_vector_expr(_cfg("pg_search")) is None


# --- write side: the name bind must be cast --------------------------------


class _RecordingConn:
    """Captures the SQL of the one fetchrow the pinned-model insert issues."""

    def __init__(self) -> None:
        self.sql = ""

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any]:
        self.sql = sql
        return {}


async def _pinned_insert_sql(ext: str, monkeypatch) -> str:
    """The INSERT that ``_insert_pinned_mental_model`` builds for ``ext``.

    The method reads no attribute off ``self``, so an uninitialised instance is
    enough to reach the SQL without a database, an embedder, or an LLM.
    """
    from hindsight_api.engine import memory_engine as me

    cfg = _cfg(ext)
    cfg.database_schema = "public"  # fq_table() reads it off the same config
    monkeypatch.setattr(me, "get_config", lambda: cfg)
    conn = _RecordingConn()
    await me.MemoryEngine._insert_pinned_mental_model(
        object.__new__(me.MemoryEngine),
        conn,
        mental_model_id="mm-1",
        bank_id="b",
        name="page name",
        source_query="q",
        content="c",
        embedding=None,
        tags=[],
        max_tokens=None,
        trigger=None,
    )
    return conn.sql


async def test_pinned_insert_casts_the_name_bind(monkeypatch):
    """$3 lands in a VARCHAR column and in tokenize(), which takes TEXT.

    PostgreSQL infers one type per parameter, so without the cast the statement
    never reaches execution: prepare fails with "inconsistent types deduced for
    parameter $3: text versus character varying" and every knowledge-page and
    pinned mental-model create 500s on vchord.
    """
    sql = await _pinned_insert_sql("vchord", monkeypatch)
    assert "'pinned', $3::text," in sql
    assert "tokenize(COALESCE($3, '')" in sql


async def test_pinned_insert_keeps_the_cast_without_inline_tokenization(monkeypatch):
    # Backends that leave search_vector alone bind $3 once, so the cast is inert
    # there — it stays for one statement shape across backends.
    sql = await _pinned_insert_sql("native", monkeypatch)
    assert "'pinned', $3::text," in sql
    assert "search_vector" not in sql
