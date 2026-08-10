"""Backend dispatch for the knowledge-page BM25 arm (issue #3268).

``search_knowledge_pages`` used to hard-code the native tsvector SQL
(``ts_rank_cd`` / ``@@``) for every text-search backend, so it 500'd on
``pg_search`` / ``pg_textsearch`` / ``vchord`` where ``mental_models.search_vector``
is not a tsvector. These tests pin the per-backend SQL the dispatcher emits so the
regression can't silently come back — they need no live extension because they
assert on the generated SQL, the way ``test_multilingual_bm25`` does.
"""

from hindsight_api.engine.sql.postgresql import KnowledgeBm25Arm, knowledge_bm25_arm


def _arm(ext: str) -> KnowledgeBm25Arm:
    arm = knowledge_bm25_arm(ext, table_alias="mm", text_param="$3")
    assert arm is not None, f"{ext} unexpectedly degraded to vector-only"
    return arm


def test_native_uses_tsvector_operators():
    arm = _arm("native")
    # The mental_models tsvector is generated with the 'english' config, so the
    # query must use 'english' regardless of the configured native language.
    assert "ts_rank_cd(mm.search_vector, websearch_to_tsquery('english', $3))" in arm.score_expr
    assert arm.match_filter == "AND mm.search_vector @@ websearch_to_tsquery('english', $3)"


def test_pgroonga_is_served_by_the_native_branch():
    # mental_models is never reconciled to pgroonga structures, so it keeps the
    # generated tsvector column and must use the native operators.
    assert knowledge_bm25_arm("pgroonga", table_alias="mm", text_param="$3") == _arm("native")


def test_pg_search_uses_paradedb_over_base_columns():
    arm = _arm("pg_search")
    assert arm.score_expr == "paradedb.score(mm.id)"
    assert "mm.id @@@ paradedb.boolean(should => ARRAY[" in arm.match_filter
    assert "paradedb.match('name', $3)" in arm.match_filter
    assert "paradedb.match('content', $3)" in arm.match_filter
    # Must not fall back to the native tsvector function.
    assert "ts_rank_cd" not in arm.score_expr
    assert "ts_rank_cd" not in arm.order_by


def test_pg_textsearch_ranks_content_by_bm25_distance():
    arm = _arm("pg_textsearch")
    # `<@>` is a distance (lower = closer): order ASC, negate for the score.
    assert arm.order_by == "mm.content <@> to_bm25query($3, 'idx_mental_models_text_search') ASC"
    assert arm.score_expr == "-(mm.content <@> to_bm25query($3, 'idx_mental_models_text_search'))"
    # It ranks every row, so there is no boolean match gate.
    assert arm.match_filter == ""
    assert "ts_rank_cd" not in arm.order_by


def test_vchord_degrades_to_vector_only():
    # vchord's mental_models bm25vector column is never populated on write, so its
    # BM25 index is empty — the caller must fall back to a vector-only search.
    assert knowledge_bm25_arm("vchord", table_alias="mm", text_param="$3") is None


def test_text_param_and_alias_are_threaded_through():
    arm = knowledge_bm25_arm("pg_search", table_alias="kbm", text_param="$7")
    assert "paradedb.score(kbm.id)" == arm.score_expr
    assert "kbm.id @@@" in arm.match_filter
    assert "paradedb.match('name', $7)" in arm.match_filter
