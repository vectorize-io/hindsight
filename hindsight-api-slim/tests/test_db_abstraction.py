"""Tests for the database abstraction layer (db + sql modules).

Unit tests that verify the abstraction interfaces work correctly
without requiring a live database connection.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.db import DatabaseBackend, DatabaseConnection, ResultRow, create_database_backend
from hindsight_api.engine.db.postgresql import PostgreSQLBackend
from hindsight_api.engine.sql import SQLDialect, create_sql_dialect
from hindsight_api.engine.sql.postgresql import PostgreSQLDialect

# ---------------------------------------------------------------------------
# ResultRow tests
# ---------------------------------------------------------------------------


class TestResultRow:
    def test_dict_access(self):
        row = ResultRow({"id": 1, "name": "test"})
        assert row["id"] == 1
        assert row["name"] == "test"

    def test_attr_access(self):
        row = ResultRow({"id": 1, "name": "test"})
        assert row.id == 1
        assert row.name == "test"

    def test_get_with_default(self):
        row = ResultRow({"id": 1})
        assert row.get("id") == 1
        assert row.get("missing") is None
        assert row.get("missing", "default") == "default"

    def test_keys(self):
        row = ResultRow({"a": 1, "b": 2})
        assert set(row.keys()) == {"a", "b"}

    def test_values(self):
        row = ResultRow({"a": 1, "b": 2})
        assert set(row.values()) == {1, 2}

    def test_items(self):
        row = ResultRow({"a": 1, "b": 2})
        assert set(row.items()) == {("a", 1), ("b", 2)}

    def test_contains(self):
        row = ResultRow({"id": 1})
        assert "id" in row
        assert "missing" not in row

    def test_len(self):
        row = ResultRow({"a": 1, "b": 2, "c": 3})
        assert len(row) == 3

    def test_bool_always_true(self):
        row = ResultRow({})
        assert bool(row)

    def test_repr(self):
        row = ResultRow({"id": 1})
        assert "ResultRow" in repr(row)

    def test_missing_attr_raises(self):
        row = ResultRow({"id": 1})
        with pytest.raises(AttributeError):
            _ = row.missing


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestFactories:
    def test_create_postgresql_backend(self):
        backend = create_database_backend("postgresql")
        assert isinstance(backend, PostgreSQLBackend)
        assert isinstance(backend, DatabaseBackend)

    def test_create_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown database backend"):
            create_database_backend("mysql")

    def test_create_postgresql_dialect(self):
        dialect = create_sql_dialect("postgresql")
        assert isinstance(dialect, PostgreSQLDialect)
        assert isinstance(dialect, SQLDialect)

    def test_create_unknown_dialect_raises(self):
        with pytest.raises(ValueError, match="Unknown SQL dialect"):
            create_sql_dialect("mysql")


# ---------------------------------------------------------------------------
# PostgreSQLDialect tests
# ---------------------------------------------------------------------------


class TestPostgreSQLDialect:
    @pytest.fixture()
    def d(self):
        return PostgreSQLDialect()

    def test_param(self, d):
        assert d.param(1) == "$1"
        assert d.param(3) == "$3"

    def test_cast(self, d):
        assert d.cast("$1", "jsonb") == "$1::jsonb"
        assert d.cast("$2", "uuid[]") == "$2::uuid[]"

    def test_vector_distance(self, d):
        assert d.vector_distance("embedding", "$1") == "embedding <=> $1::vector"

    def test_vector_similarity(self, d):
        assert d.vector_similarity("embedding", "$1") == "1 - (embedding <=> $1::vector)"

    def test_json_extract_text(self, d):
        assert d.json_extract_text("col", "key") == "col ->> 'key'"

    def test_json_contains(self, d):
        assert d.json_contains("col", "$1") == "col @> $1::jsonb"

    def test_json_merge(self, d):
        assert d.json_merge("col", "$1") == "col || $1::jsonb"

    def test_text_search_score_bm25(self, d):
        result = d.text_search_score("text", "$1", index_name="idx_test")
        assert "to_bm25query" in result

    def test_text_search_score_tsvector(self, d):
        result = d.text_search_score("text", "$1")
        assert "ts_rank_cd" in result

    def test_similarity(self, d):
        assert d.similarity("col", "$1") == "similarity(col, $1)"

    def test_upsert_do_nothing(self, d):
        sql = d.upsert("t", ["a", "b"], ["a"], [])
        assert "ON CONFLICT (a) DO NOTHING" in sql

    def test_upsert_do_update(self, d):
        sql = d.upsert("t", ["a", "b"], ["a"], ["b"])
        assert "ON CONFLICT (a) DO UPDATE SET b = EXCLUDED.b" in sql

    def test_bulk_unnest(self, d):
        result = d.bulk_unnest([("$1", "text[]"), ("$2", "uuid[]")])
        assert result == "unnest($1::text[], $2::uuid[])"

    def test_limit_offset(self, d):
        assert d.limit_offset("$1", "$2") == "LIMIT $1 OFFSET $2"

    def test_returning(self, d):
        assert d.returning(["id", "name"]) == "RETURNING id, name"

    def test_ilike(self, d):
        assert d.ilike("col", "$1") == "col ILIKE $1"

    def test_array_any(self, d):
        assert d.array_any("$1") == "= ANY($1)"

    def test_array_all(self, d):
        assert d.array_all("$1") == "!= ALL($1)"

    def test_array_contains(self, d):
        assert d.array_contains("tags", "$1") == "tags @> $1::varchar[]"

    def test_for_update_skip_locked(self, d):
        assert d.for_update_skip_locked() == "FOR UPDATE SKIP LOCKED"

    def test_advisory_lock(self, d):
        assert d.advisory_lock("$1") == "pg_try_advisory_lock($1)"

    def test_generate_uuid(self, d):
        assert d.generate_uuid() == "gen_random_uuid()"

    def test_greatest(self, d):
        assert d.greatest("a", "b") == "GREATEST(a, b)"

    def test_current_timestamp(self, d):
        assert d.current_timestamp() == "now()"

    def test_array_agg(self, d):
        assert d.array_agg("col") == "array_agg(col)"

    def test_build_semantic_arm(self, d):
        arm = d.build_semantic_arm(
            table="schema.memory_units", cols="id, text", fact_type="world",
            embedding_param="$1", bank_id_param="$2", fetch_limit=100,
        )
        assert "1 - (embedding <=> $1::vector)" in arm
        assert "fact_type = 'world'" in arm
        assert "LIMIT 100" in arm
        assert "'semantic' AS source" in arm

    def test_build_bm25_arm_native(self, d):
        arm = d.build_bm25_arm(
            table="schema.memory_units", cols="id, text", fact_type="world",
            bank_id_param="$2", limit_param="$3", text_param="$4",
        )
        assert "ts_rank_cd" in arm
        assert "to_tsquery" in arm
        assert "'bm25' AS source" in arm
        assert "LIMIT $3" in arm

    def test_build_bm25_arm_vchord(self, d):
        arm = d.build_bm25_arm(
            table="t", cols="id", fact_type="world",
            bank_id_param="$2", limit_param="$3", text_param="$4",
            text_search_extension="vchord",
        )
        assert "to_bm25query" in arm
        assert "tokenize" in arm

    def test_prepare_bm25_text_native(self, d):
        result = d.prepare_bm25_text(["hello", "world"], "hello world")
        assert result == "hello | world"

    def test_prepare_bm25_text_vchord(self, d):
        result = d.prepare_bm25_text(["hello", "world"], "hello world", text_search_extension="vchord")
        assert result == "hello world"


# ---------------------------------------------------------------------------
# OracleDialect tests (no oracledb dependency needed)
# ---------------------------------------------------------------------------


class TestOracleDialect:
    @pytest.fixture()
    def d(self):
        from hindsight_api.engine.sql.oracle import OracleDialect

        return OracleDialect()

    def test_param(self, d):
        assert d.param(1) == ":1"
        assert d.param(3) == ":3"

    def test_vector_distance(self, d):
        assert "VECTOR_DISTANCE" in d.vector_distance("embedding", ":1")
        assert "COSINE" in d.vector_distance("embedding", ":1")

    def test_ilike(self, d):
        assert "UPPER" in d.ilike("col", ":1")

    def test_upsert(self, d):
        sql = d.upsert("t", ["a", "b"], ["a"], ["b"])
        assert "MERGE INTO" in sql

    def test_limit_offset(self, d):
        result = d.limit_offset(":1", ":2")
        assert "FETCH FIRST" in result
        assert "OFFSET" in result

    def test_returning(self, d):
        result = d.returning(["id"])
        assert "RETURNING" in result
        assert "INTO" in result

    def test_generate_uuid(self, d):
        assert d.generate_uuid() == "SYS_GUID()"

    def test_current_timestamp(self, d):
        assert d.current_timestamp() == "SYSTIMESTAMP"

    def test_build_semantic_arm(self, d):
        arm = d.build_semantic_arm(
            table="memory_units", cols="id, text", fact_type="world",
            embedding_param=":1", bank_id_param=":2", fetch_limit=100,
        )
        assert "VECTOR_DISTANCE" in arm
        assert "fact_type = 'world'" in arm
        assert "FETCH FIRST 100 ROWS ONLY" in arm
        assert "'semantic' AS source" in arm

    def test_build_bm25_arm(self, d):
        arm = d.build_bm25_arm(
            table="memory_units", cols="id, text", fact_type="world",
            bank_id_param=":2", limit_param=":3", text_param=":4",
            arm_index=0,
        )
        assert "CONTAINS" in arm
        assert "SCORE(10)" in arm
        assert "'bm25' AS source" in arm
        assert "FETCH FIRST :3 ROWS ONLY" in arm

    def test_build_bm25_arm_unique_labels(self, d):
        """Each arm_index produces a unique SCORE label to avoid conflicts in UNION ALL."""
        arm0 = d.build_bm25_arm(
            table="t", cols="id", fact_type="world",
            bank_id_param=":2", limit_param=":3", text_param=":4", arm_index=0,
        )
        arm1 = d.build_bm25_arm(
            table="t", cols="id", fact_type="experience",
            bank_id_param=":2", limit_param=":3", text_param=":4", arm_index=1,
        )
        assert "SCORE(10)" in arm0
        assert "SCORE(11)" in arm1

    def test_prepare_bm25_text(self, d):
        result = d.prepare_bm25_text(["hello", "world"], "hello world")
        assert result == "hello OR world"

    def test_prepare_bm25_text_special_chars_filtered(self, d):
        result = d.prepare_bm25_text(["hello", "$special", "world"], "hello $special world")
        assert "$special" not in result
        assert "hello" in result


# ---------------------------------------------------------------------------
# Oracle query rewriter tests
# ---------------------------------------------------------------------------


class TestOracleQueryRewriter:
    """Tests for _rewrite_pg_to_oracle which returns (query, has_returning, returning_cols)."""

    def test_param_rewrite(self):
        from hindsight_api.engine.db.oracle import _rewrite_pg_to_oracle

        query, _, _ = _rewrite_pg_to_oracle("SELECT $1 FROM t")
        assert ":1" in query
        query2, _, _ = _rewrite_pg_to_oracle("WHERE a = $1 AND b = $2")
        assert ":2" in query2

    def test_cast_removal(self):
        from hindsight_api.engine.db.oracle import _rewrite_pg_to_oracle

        query, _, _ = _rewrite_pg_to_oracle("$1::jsonb")
        assert "::jsonb" not in query
        assert ":1" in query

    def test_multiple_casts(self):
        from hindsight_api.engine.db.oracle import _rewrite_pg_to_oracle

        query, _, _ = _rewrite_pg_to_oracle("$1::text, $2::uuid, $3::varchar[]")
        assert "::text" not in query
        assert "::uuid" not in query
        assert "::varchar[]" not in query

    def test_now_to_systimestamp(self):
        from hindsight_api.engine.db.oracle import _rewrite_pg_to_oracle

        query, _, _ = _rewrite_pg_to_oracle("updated_at > NOW()")
        assert "SYSTIMESTAMP" in query
        assert "NOW()" not in query

    def test_gen_random_uuid(self):
        from hindsight_api.engine.db.oracle import _rewrite_pg_to_oracle

        query, _, _ = _rewrite_pg_to_oracle("gen_random_uuid()")
        assert "SYS_GUID()" in query

    def test_combined_rewrite(self):
        from hindsight_api.engine.db.oracle import _rewrite_pg_to_oracle

        query, ignore_dup, returning_cols = _rewrite_pg_to_oracle(
            "INSERT INTO t (id, data) VALUES ($1::uuid, $2::jsonb) RETURNING id"
        )
        assert ":1" in query
        assert ":2" in query
        assert "::uuid" not in query
        assert "::jsonb" not in query
        assert not ignore_dup
        assert returning_cols == ["id"]
        assert "RETURNING id INTO :ret_0" in query

    def test_no_rewrite_needed(self):
        from hindsight_api.engine.db.oracle import _rewrite_pg_to_oracle

        query = "SELECT 1 FROM DUAL"
        result_query, ignore_dup, returning_cols = _rewrite_pg_to_oracle(query)
        assert result_query == query
        assert not ignore_dup
        assert returning_cols is None

    def test_jsonb_boolean_rewrite(self):
        """Verify JSONB ->> boolean comparison is rewritten to JSON_VALUE."""
        from hindsight_api.engine.db.oracle import _rewrite_pg_to_oracle

        query, _, _ = _rewrite_pg_to_oracle(
            "WHERE (trigger->>'refresh_after_consolidation')::boolean = true"
        )
        assert "JSON_VALUE" in query
        assert "'true'" in query
        assert "->>" not in query

    def test_jsonb_arrow_text_quoted(self):
        """Verify ->> works with quoted column names."""
        from hindsight_api.engine.db.oracle import _rewrite_pg_to_oracle

        query, _, _ = _rewrite_pg_to_oracle(
            "ORDER BY (result_metadata->>'sub_batch_index')::int"
        )
        assert "JSON_VALUE" in query
        assert "->>" not in query


# ---------------------------------------------------------------------------
# PostgreSQLBackend unit tests (no live DB)
# ---------------------------------------------------------------------------


class TestPostgreSQLBackendUnit:
    def test_uninitialized_acquire_raises(self):
        backend = PostgreSQLBackend()
        with pytest.raises(RuntimeError, match="not initialized"):
            backend.get_pool()

    def test_uninitialized_get_pool_raises(self):
        backend = PostgreSQLBackend()
        with pytest.raises(RuntimeError, match="not initialized"):
            backend.get_pool()


# ---------------------------------------------------------------------------
# Config integration test
# ---------------------------------------------------------------------------


class TestConfig:
    def test_database_backend_field_exists(self):
        # Verify the field exists on the dataclass
        import dataclasses

        from hindsight_api.config import HindsightConfig

        field_names = {f.name for f in dataclasses.fields(HindsightConfig)}
        assert "database_backend" in field_names

    def test_default_database_backend(self):
        from hindsight_api.config import DEFAULT_DATABASE_BACKEND

        assert DEFAULT_DATABASE_BACKEND == "postgresql"


# ---------------------------------------------------------------------------
# OracleOps unit tests (mock DatabaseConnection, no live DB)
# ---------------------------------------------------------------------------


class TestOracleOpsInsertFactsBatch:
    """Verify insert_facts_batch uses executemany with client-side UUIDs
    and correctly maps all input columns to the SQL statement."""

    @pytest.fixture()
    def ops(self):
        from hindsight_api.engine.db.ops_oracle import OracleOps

        return OracleOps()

    @pytest.fixture()
    def mock_conn(self):
        conn = AsyncMock(spec=DatabaseConnection)
        conn.executemany = AsyncMock()
        return conn

    def _make_batch(self, n: int = 2) -> dict:
        """Build a realistic batch of N facts with distinct values per column."""
        from datetime import datetime, timezone

        dates = [datetime(2024, 1, i + 1, tzinfo=timezone.utc) for i in range(n)]
        fact_type_cycle = ["world", "experience"]
        return dict(
            bank_id="bank-1",
            fact_texts=[f"fact-{i}" for i in range(n)],
            embeddings=[f"[0.{i}]" for i in range(n)],
            event_dates=dates,
            occurred_starts=[None] * n,
            occurred_ends=[None] * n,
            mentioned_ats=[None] * n,
            contexts=[f"ctx-{i}" for i in range(n)],
            fact_types=[fact_type_cycle[i % 2] for i in range(n)],
            metadata_jsons=['{"key": "val"}'] * n,
            chunk_ids=[f"chunk-{i}" for i in range(n)],
            document_ids=[f"doc-{i}" for i in range(n)],
            tags_list=[f'["tag-{i}"]' for i in range(n)],
            observation_scopes_list=[None] * n,
            text_signals_list=[None] * n,
        )

    @pytest.mark.asyncio
    async def test_single_executemany_not_row_by_row(self, ops, mock_conn):
        """Must use one executemany call (batch), never fetchval (row-by-row)."""
        batch = self._make_batch(3)
        result = await ops.insert_facts_batch(conn=mock_conn, **batch)

        mock_conn.executemany.assert_called_once()
        mock_conn.fetchval.assert_not_called()
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_returned_ids_are_valid_unique_uuids(self, ops, mock_conn):
        """Each returned ID must be a valid UUID and all must be distinct."""
        import uuid as _uuid

        batch = self._make_batch(5)
        result = await ops.insert_facts_batch(conn=mock_conn, **batch)

        parsed = [_uuid.UUID(r) for r in result]  # Raises ValueError if invalid
        assert len(set(parsed)) == 5, "UUIDs must be unique"

    @pytest.mark.asyncio
    async def test_returned_ids_match_rows_sent_to_db(self, ops, mock_conn):
        """The UUIDs returned to the caller must be the same ones sent to the DB."""
        batch = self._make_batch(2)
        result = await ops.insert_facts_batch(conn=mock_conn, **batch)

        _, rows_data = mock_conn.executemany.call_args.args
        ids_in_rows = [row[0] for row in rows_data]
        assert result == ids_in_rows

    @pytest.mark.asyncio
    async def test_column_values_correctly_mapped(self, ops, mock_conn):
        """Every input column must land in the correct position in the row tuple.

        This is the critical correctness test — a column ordering bug here would
        silently insert data into the wrong columns.
        """
        from datetime import datetime, timezone

        dt = datetime(2024, 6, 15, tzinfo=timezone.utc)
        result = await ops.insert_facts_batch(
            conn=mock_conn,
            bank_id="bank-42",
            fact_texts=["The sky is blue"],
            embeddings=["[0.1, 0.2, 0.3]"],
            event_dates=[dt],
            occurred_starts=[dt],
            occurred_ends=[dt],
            mentioned_ats=[dt],
            contexts=["weather"],
            fact_types=["world"],
            metadata_jsons=['{"source": "obs"}'],
            chunk_ids=["chunk-99"],
            document_ids=["doc-55"],
            tags_list=['["nature", "sky"]'],
            observation_scopes_list=["global"],
            text_signals_list=["positive"],
        )

        query, rows_data = mock_conn.executemany.call_args.args
        assert len(rows_data) == 1
        row = rows_data[0]

        # Verify column order matches: id, bank_id, text, embedding, event_date,
        # occurred_start, occurred_end, mentioned_at, context, fact_type, metadata,
        # chunk_id, document_id, tags, observation_scopes, text_signals
        assert row[0] == result[0], "row[0] should be the generated UUID"
        assert row[1] == "bank-42", "row[1] should be bank_id"
        assert row[2] == "The sky is blue", "row[2] should be text"
        assert row[3] == "[0.1, 0.2, 0.3]", "row[3] should be embedding"
        assert row[4] == dt, "row[4] should be event_date"
        assert row[5] == dt, "row[5] should be occurred_start"
        assert row[6] == dt, "row[6] should be occurred_end"
        assert row[7] == dt, "row[7] should be mentioned_at"
        assert row[8] == "weather", "row[8] should be context"
        assert row[9] == "world", "row[9] should be fact_type"
        assert row[10] == '{"source": "obs"}', "row[10] should be metadata JSON string"
        assert row[11] == "chunk-99", "row[11] should be chunk_id"
        assert row[12] == "doc-55", "row[12] should be document_id"
        assert row[13] == ["nature", "sky"], "row[13] should be decoded tags list"
        assert row[14] == "global", "row[14] should be observation_scopes"
        assert row[15] == "positive", "row[15] should be text_signals"

    @pytest.mark.asyncio
    async def test_sql_column_count_matches_values(self, ops, mock_conn):
        """The INSERT column list and VALUES placeholders must both have 16 entries."""
        batch = self._make_batch(1)
        await ops.insert_facts_batch(conn=mock_conn, **batch)

        query, _ = mock_conn.executemany.call_args.args
        # Extract the column list between "(" and ")" after INSERT INTO ... (
        # and count the $N placeholders in VALUES
        assert query.count("$") == 16, "VALUES clause must have 16 placeholders"

    @pytest.mark.asyncio
    async def test_tags_json_decoded_to_list(self, ops, mock_conn):
        """Tags JSON strings must be decoded to Python lists, not passed as strings."""
        await ops.insert_facts_batch(
            conn=mock_conn, **{**self._make_batch(1), "tags_list": ['["tag1", "tag2"]']}
        )
        _, rows_data = mock_conn.executemany.call_args.args
        assert rows_data[0][13] == ["tag1", "tag2"]
        assert isinstance(rows_data[0][13], list)

    @pytest.mark.asyncio
    async def test_empty_tags_becomes_empty_list(self, ops, mock_conn):
        """Empty/falsy tags string must become [], not crash or pass empty string."""
        await ops.insert_facts_batch(
            conn=mock_conn, **{**self._make_batch(1), "tags_list": [""]}
        )
        _, rows_data = mock_conn.executemany.call_args.args
        assert rows_data[0][13] == []


# ---------------------------------------------------------------------------
# normalize_schema tests
# ---------------------------------------------------------------------------


class TestNormalizeSchema:
    """Verify Backend.normalize_schema() returns correct schema for each backend."""

    def test_postgresql_passes_through(self):
        backend = PostgreSQLBackend()
        assert backend.normalize_schema("public") == "public"
        assert backend.normalize_schema("tenant_abc") == "tenant_abc"
        assert backend.normalize_schema(None) is None

    def test_oracle_maps_public_to_none(self):
        from hindsight_api.engine.db.oracle import OracleBackend

        backend = OracleBackend()
        assert backend.normalize_schema("public") is None
        assert backend.normalize_schema("tenant_abc") == "tenant_abc"
        assert backend.normalize_schema(None) is None
