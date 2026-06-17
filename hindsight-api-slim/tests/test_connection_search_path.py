"""Unit tests for build_connection_search_path (issue #2270).

Runtime PG connections must expose the schema that holds extension operators
(pg_trgm's `%`, vchord BM25's `<&>`) on the session search_path. Tenant tables
are reached via fully-qualified names, so search_path only governs operator/type
resolution — but an external Postgres whose role search_path omits the extension
schema fails retain with `operator does not exist: text % text`.
"""

from hindsight_api.engine.memory_engine import build_connection_search_path


def test_pgvector_backend_includes_user_and_public():
    """Default (non-vchord) backend pins "$user", public so public extensions resolve."""
    assert build_connection_search_path("pgvector", "public") == 'SET search_path TO "$user", public'


def test_pg_trgm_in_public_is_not_duplicated():
    """pg_trgm living in public needs no extra entry — public is already present."""
    assert build_connection_search_path("pgvector", "public") == 'SET search_path TO "$user", public'


def test_pg_trgm_in_custom_schema_is_appended_and_quoted():
    """pg_trgm installed into a non-public schema is added so `%` resolves at runtime."""
    assert (
        build_connection_search_path("pgvector", "hindsight")
        == 'SET search_path TO "$user", public, "hindsight"'
    )


def test_pg_trgm_absent_falls_back_to_user_and_public():
    """When pg_trgm is not installed (None), only the base path is set."""
    assert build_connection_search_path("pgvector", None) == 'SET search_path TO "$user", public'


def test_vchord_appends_bm25_catalogs():
    """vchord backend keeps its dedicated catalog schemas on the path."""
    assert (
        build_connection_search_path("vchord", "public")
        == 'SET search_path TO "$user", public, bm25_catalog, tokenizer_catalog'
    )


def test_vchord_with_custom_trgm_schema_includes_both():
    """A custom pg_trgm schema and the vchord catalogs coexist on the path."""
    assert (
        build_connection_search_path("vchord", "hindsight")
        == 'SET search_path TO "$user", public, "hindsight", bm25_catalog, tokenizer_catalog'
    )
