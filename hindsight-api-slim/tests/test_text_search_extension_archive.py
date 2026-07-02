"""Regression tests for text-search reconciliation on cold archive tables."""

import uuid

from sqlalchemy import create_engine, text

from hindsight_api.migrations import ensure_text_search_extension


def test_pgroonga_reconcile_converts_invalidated_archive_search_vector_with_data(pg0_db_url):
    """Archive rows can be type-converted even when hot recall tables are absent.

    pgroonga keeps ``memory_units.search_vector`` as passive TEXT. The
    invalidation archive copies that column during INSERT ... SELECT moves, so
    its inherited native ``tsvector`` column must be reconciled too (#2503).
    """
    schema = f"archive_pgroonga_{uuid.uuid4().hex[:8]}"
    row_id = uuid.uuid4()
    engine = create_engine(pg0_db_url)

    try:
        with engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA {schema}"))
            conn.execute(
                text(
                    f"""
                    CREATE TABLE {schema}.invalidated_memory_units (
                        id UUID PRIMARY KEY,
                        search_vector tsvector
                    )
                    """
                )
            )
            conn.execute(
                text(
                    f"""
                    INSERT INTO {schema}.invalidated_memory_units (id, search_vector)
                    VALUES (:id, to_tsvector('english', 'archived memory'))
                    """
                ),
                {"id": str(row_id)},
            )

        ensure_text_search_extension(pg0_db_url, text_search_extension="pgroonga", schema=schema)

        with engine.connect() as conn:
            column_type = conn.execute(
                text(
                    """
                    SELECT udt_name
                    FROM information_schema.columns
                    WHERE table_schema = :schema
                      AND table_name = 'invalidated_memory_units'
                      AND column_name = 'search_vector'
                    """
                ),
                {"schema": schema},
            ).scalar()
            archived_value = conn.execute(
                text(f"SELECT search_vector FROM {schema}.invalidated_memory_units WHERE id = :id"),
                {"id": str(row_id)},
            ).scalar()
            text_search_indexes = conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM pg_indexes
                    WHERE schemaname = :schema
                      AND tablename = 'invalidated_memory_units'
                      AND indexname LIKE '%text_search%'
                    """
                ),
                {"schema": schema},
            ).scalar()
    finally:
        with engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
        engine.dispose()

    assert column_type == "text"
    assert "archiv" in archived_value
    assert text_search_indexes == 0


def test_hot_table_missing_text_search_index_is_still_recreated(pg0_db_url):
    """Only archive tables may ignore text-search indexes during reconciliation."""
    schema = f"hot_index_{uuid.uuid4().hex[:8]}"
    engine = create_engine(pg0_db_url)

    try:
        with engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA {schema}"))
            conn.execute(text(f"CREATE TABLE {schema}.memory_units (search_vector tsvector)"))

        ensure_text_search_extension(pg0_db_url, text_search_extension="native", schema=schema)

        with engine.connect() as conn:
            index_type = conn.execute(
                text(
                    """
                    SELECT am.amname
                    FROM pg_indexes pi
                    JOIN pg_class c ON c.relname = pi.indexname
                    JOIN pg_am am ON am.oid = c.relam
                    WHERE pi.schemaname = :schema
                      AND pi.tablename = 'memory_units'
                      AND pi.indexname = 'idx_memory_units_text_search'
                    """
                ),
                {"schema": schema},
            ).scalar()
    finally:
        with engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
        engine.dispose()

    assert index_type == "gin"
