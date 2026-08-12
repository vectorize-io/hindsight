"""Tests for link_utils datetime handling, temporal link computation, and semantic link splitting."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from hindsight_api.config import DEFAULT_SEMANTIC_LINK_MIN_SIMILARITY, clear_config_cache
from hindsight_api.engine.retain.link_utils import (
    _NIL_ENTITY_UUID,
    MAX_TEMPORAL_LINKS_PER_UNIT,
    _cap_links_per_unit,
    _lock_order_key,
    _normalize_datetime,
    compute_semantic_links_ann,
    compute_semantic_links_within_batch,
)


class TestNormalizeDatetime:
    """Tests for the _normalize_datetime helper function."""

    def test_none_returns_none(self):
        """Test that None input returns None."""
        assert _normalize_datetime(None) is None

    def test_naive_datetime_becomes_utc(self):
        """Test that naive datetimes are converted to UTC."""
        naive_dt = datetime(2024, 6, 15, 10, 30, 0)
        result = _normalize_datetime(naive_dt)

        assert result.tzinfo is not None
        assert result.tzinfo == timezone.utc
        assert result.year == 2024
        assert result.month == 6
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30

    def test_aware_datetime_unchanged(self):
        """Test that timezone-aware datetimes are returned unchanged."""
        aware_dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = _normalize_datetime(aware_dt)

        assert result == aware_dt
        assert result.tzinfo == timezone.utc

    def test_mixed_datetimes_can_be_compared(self):
        """Test that normalized naive and aware datetimes can be compared."""
        naive_dt = datetime(2024, 6, 15, 10, 30, 0)
        aware_dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)

        normalized_naive = _normalize_datetime(naive_dt)
        normalized_aware = _normalize_datetime(aware_dt)

        # Should be able to compare without TypeError
        assert normalized_naive == normalized_aware


class TestCapLinksPerUnit:
    """Tests for the _cap_links_per_unit helper function."""

    def test_empty_links(self):
        assert _cap_links_per_unit([]) == []

    def test_under_cap_unchanged(self):
        links = [
            ("unit_a", "unit_x", "temporal", 0.9, None),
            ("unit_a", "unit_y", "temporal", 0.8, None),
        ]
        result = _cap_links_per_unit(links, max_per_unit=5)
        assert len(result) == 2

    def test_caps_to_max_per_unit(self):
        # Create 30 links from the same unit with descending weights
        links = [("unit_a", f"unit_{i}", "temporal", 1.0 - i * 0.01, None) for i in range(30)]
        result = _cap_links_per_unit(links, max_per_unit=10)
        assert len(result) == 10
        # Should keep the highest-weight links
        weights = [lnk[3] for lnk in result]
        assert weights == sorted(weights, reverse=True)
        assert weights[0] == 1.0  # Highest weight kept

    def test_caps_independently_per_unit(self):
        links_a = [("unit_a", f"target_{i}", "temporal", 0.9 - i * 0.01, None) for i in range(10)]
        links_b = [("unit_b", f"target_{i}", "temporal", 0.8 - i * 0.01, None) for i in range(10)]
        result = _cap_links_per_unit(links_a + links_b, max_per_unit=5)
        # 5 from unit_a + 5 from unit_b
        assert len(result) == 10
        from_a = [lnk for lnk in result if lnk[0] == "unit_a"]
        from_b = [lnk for lnk in result if lnk[0] == "unit_b"]
        assert len(from_a) == 5
        assert len(from_b) == 5

    def test_default_max_is_temporal_constant(self):
        links = [("unit_a", f"target_{i}", "temporal", 1.0 - i * 0.01, None) for i in range(50)]
        result = _cap_links_per_unit(links)
        assert len(result) == MAX_TEMPORAL_LINKS_PER_UNIT

    def test_preserves_tuple_structure(self):
        links = [("from_id", "to_id", "temporal", 0.95, "entity_id")]
        result = _cap_links_per_unit(links, max_per_unit=5)
        assert result[0] == ("from_id", "to_id", "temporal", 0.95, "entity_id")


class TestComputeSemanticLinksWithinBatch:
    """Tests for compute_semantic_links_within_batch.

    This function computes semantic links between units in the same batch
    using numpy dot product (no DB access). It runs in Phase 2 (write
    transaction) while the expensive ANN search against existing units runs
    in Phase 1 on a separate connection to avoid TimeoutErrors from HNSW
    index contention under concurrent load.
    """

    def test_empty_returns_empty(self):
        assert (
            compute_semantic_links_within_batch(
                [],
                [],
                threshold=DEFAULT_SEMANTIC_LINK_MIN_SIMILARITY,
            )
            == []
        )

    def test_single_unit_returns_empty(self):
        emb = [np.random.randn(384).tolist()]
        assert (
            compute_semantic_links_within_batch(
                ["u1"],
                emb,
                threshold=DEFAULT_SEMANTIC_LINK_MIN_SIMILARITY,
            )
            == []
        )

    def test_identical_embeddings_produce_links(self):
        """Two identical embeddings should have similarity=1.0 (above 0.7 threshold)."""
        emb = [0.1] * 384
        links = compute_semantic_links_within_batch(
            ["u1", "u2"],
            [emb, emb],
            threshold=DEFAULT_SEMANTIC_LINK_MIN_SIMILARITY,
        )
        assert len(links) == 2  # bidirectional: u1→u2, u2→u1
        from_ids = {lnk[0] for lnk in links}
        to_ids = {lnk[1] for lnk in links}
        assert from_ids == {"u1", "u2"}
        assert to_ids == {"u1", "u2"}
        for lnk in links:
            assert lnk[2] == "semantic"
            assert lnk[3] >= 0.99  # near-1.0 similarity
            assert lnk[4] is None  # no entity_id

    def test_orthogonal_embeddings_no_links(self):
        """Orthogonal embeddings should have similarity=0 (below 0.7 threshold)."""
        emb1 = [1.0] + [0.0] * 383
        emb2 = [0.0] + [1.0] + [0.0] * 382
        links = compute_semantic_links_within_batch(
            ["u1", "u2"],
            [emb1, emb2],
            threshold=DEFAULT_SEMANTIC_LINK_MIN_SIMILARITY,
        )
        assert len(links) == 0

    def test_non_unit_embeddings_use_cosine_similarity(self):
        """Vector magnitude must not turn a below-threshold cosine pair into a link."""
        emb1 = [10.0, 0.0]
        emb2 = [1.0, 1.0]

        links = compute_semantic_links_within_batch(["u1", "u2"], [emb1, emb2], threshold=0.9)

        assert links == []

    @pytest.mark.parametrize("invalid", [[0.0, 0.0], [float("nan"), 1.0], [float("inf"), 1.0]])
    def test_invalid_cosine_embeddings_do_not_link(self, invalid):
        """Zero-norm and non-finite vectors have no defined cosine similarity."""
        links = compute_semantic_links_within_batch(["invalid", "valid"], [invalid, [1.0, 0.0]], threshold=0.0)

        assert links == []

    def test_respects_threshold(self):
        """Links below threshold should be excluded."""
        emb1 = np.random.randn(384).tolist()
        # Create a slightly similar embedding (add noise)
        emb2 = [x + np.random.randn() * 0.5 for x in emb1]
        # Normalize both
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        emb1 = [x / norm1 for x in emb1]
        emb2 = [x / norm2 for x in emb2]

        links_low = compute_semantic_links_within_batch(["u1", "u2"], [emb1, emb2], threshold=0.0)
        links_high = compute_semantic_links_within_batch(["u1", "u2"], [emb1, emb2], threshold=0.99)
        # Low threshold should have more links than high threshold
        assert len(links_low) >= len(links_high)

    def test_top_k_limits_per_unit(self):
        """Each unit should link to at most top_k other units."""
        n = 10
        # Create similar embeddings (all close to the same vector)
        base = np.random.randn(384)
        base = base / np.linalg.norm(base)
        embs = [(base + np.random.randn(384) * 0.01).tolist() for _ in range(n)]
        unit_ids = [f"u{i}" for i in range(n)]

        links = compute_semantic_links_within_batch(unit_ids, embs, top_k=3, threshold=0.5)
        # Each unit should have at most 3 outgoing links
        from collections import Counter

        from_counts = Counter(lnk[0] for lnk in links)
        for count in from_counts.values():
            assert count <= 3

    def test_link_tuple_structure(self):
        """Verify the tuple format matches what _bulk_insert_links expects."""
        emb = [0.1] * 384
        links = compute_semantic_links_within_batch(
            ["u1", "u2"],
            [emb, emb],
            threshold=DEFAULT_SEMANTIC_LINK_MIN_SIMILARITY,
        )
        for lnk in links:
            assert len(lnk) == 5
            from_id, to_id, link_type, weight, entity_id = lnk
            assert isinstance(from_id, str)
            assert isinstance(to_id, str)
            assert link_type == "semantic"
            assert 0.0 <= weight <= 1.0
            assert entity_id is None


class TestLockOrderKey:
    """The insert-side sort key must reproduce the same total order that
    ``chunk_storage.delete_chunks_by_ids`` locks ``memory_links`` in, so every
    concurrent writer takes index locks in one global order (issue #3396).

    Delete-side order:
        (LEAST(from, to), GREATEST(from, to), link_type, COALESCE(entity_id, nil))
    """

    A = "00000000-0000-0000-0000-00000000000a"
    B = "00000000-0000-0000-0000-00000000000b"

    def test_direction_is_normalised(self):
        """(A, B) and (B, A) collapse to the same first two components so
        opposite-direction edges sort adjacent, matching LEAST/GREATEST."""
        fwd = _lock_order_key((self.A, self.B, "temporal", 1.0, None))
        rev = _lock_order_key((self.B, self.A, "temporal", 1.0, None))
        assert fwd[:2] == rev[:2] == (self.A, self.B)

    def test_link_type_disambiguates_same_pair(self):
        """Two edges sharing a (from, to) pair but differing in link_type must
        get distinct, deterministic keys — the gap that let insert-vs-insert
        deadlock (mechanism 1 in the issue)."""
        semantic = _lock_order_key((self.A, self.B, "semantic", 0.9, None))
        temporal = _lock_order_key((self.A, self.B, "temporal", 1.0, None))
        assert semantic != temporal
        assert semantic < temporal  # "semantic" < "temporal"

    def test_none_entity_id_uses_nil_uuid(self):
        """COALESCE(entity_id, nil) on the delete side ⇒ None maps to the nil
        UUID here, not the string 'None'."""
        key = _lock_order_key((self.A, self.B, "temporal", 1.0, None))
        assert key[3] == _NIL_ENTITY_UUID

    def test_matches_delete_total_order(self):
        """Sorting a mixed batch by the key reproduces the delete's ORDER BY."""
        c = "00000000-0000-0000-0000-00000000000c"
        links = [
            (self.B, self.A, "temporal", 1.0, None),
            (self.A, self.B, "semantic", 0.9, None),
            (self.A, c, "temporal", 1.0, None),
            (self.A, self.B, "temporal", 1.0, None),
        ]
        ordered = sorted(links, key=_lock_order_key)

        def canonical(lnk):
            a, b = str(lnk[0]), str(lnk[1])
            low, high = (a, b) if a <= b else (b, a)
            entity = str(lnk[4]) if lnk[4] is not None else _NIL_ENTITY_UUID
            return (low, high, str(lnk[2]), entity)

        assert [canonical(lnk) for lnk in ordered] == sorted(canonical(lnk) for lnk in links)


class TestComputeSemanticLinksAnnPgBouncerSafety:
    """Regression tests ensuring compute_semantic_links_ann stays in a single
    transaction so that the `_ann_seeds` temp table remains visible when the
    caller's connection goes through pgBouncer in `transaction` pool mode.

    In pgBouncer transaction mode, the backend is only pinned to the client
    for the duration of an actual PostgreSQL transaction. Outside a
    transaction, consecutive statements can land on different backends, and
    session-scoped temp tables (which are bound to the backend that created
    them) become invisible. The observed failure mode was an intermittent
    `relation "_ann_seeds" does not exist` on the statement immediately
    following the CREATE TEMP TABLE.
    """

    @pytest.fixture(autouse=True)
    def _reset_config_cache(self):
        # Tests below monkeypatch HINDSIGHT_API_VECTOR_EXTENSION. The ANN code
        # path reads it through the process-global config cache, and monkeypatch
        # reverts only the env var — not the cache. Left uncleared, a leaked
        # "vchord" makes every later bank-creating test on the same xdist worker
        # emit `USING vchordrq` against the pgvector-only test DB and fail with
        # `access method "vchordrq" does not exist`. Clear before and after so
        # the cache is rebuilt from the current env for each test.
        clear_config_cache()
        yield
        clear_config_cache()

    @pytest.fixture
    def mock_conn(self):
        """An asyncpg-like connection mock with an async `transaction()`
        context manager and awaitable execute/fetch/copy helpers."""
        conn = MagicMock()

        txn_cm = MagicMock()
        txn_cm.__aenter__ = AsyncMock(return_value=None)
        txn_cm.__aexit__ = AsyncMock(return_value=None)
        conn.transaction = MagicMock(return_value=txn_cm)

        conn.execute = AsyncMock()
        conn.copy_records_to_table = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        return conn

    @pytest.mark.asyncio
    async def test_empty_inputs_skip_transaction(self, mock_conn):
        """No seeds -> no work, no transaction, no temp-table churn."""
        result = await compute_semantic_links_ann(
            conn=mock_conn,
            bank_id="bank-1",
            unit_ids=[],
            embeddings=[],
            threshold=DEFAULT_SEMANTIC_LINK_MIN_SIMILARITY,
        )
        assert result == []
        mock_conn.transaction.assert_not_called()
        mock_conn.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_runs_inside_a_transaction(self, mock_conn):
        """The full CREATE TEMP TABLE -> COPY -> SELECT sequence must happen
        inside a single `async with conn.transaction():` block."""
        emb = [0.1] * 384
        await compute_semantic_links_ann(
            conn=mock_conn,
            bank_id="bank-1",
            unit_ids=["u1", "u2"],
            embeddings=[emb, emb],
            fact_types=["world", "world"],
            threshold=DEFAULT_SEMANTIC_LINK_MIN_SIMILARITY,
        )

        # Transaction context manager was entered.
        mock_conn.transaction.assert_called_once()
        txn_cm = mock_conn.transaction.return_value
        txn_cm.__aenter__.assert_awaited_once()
        txn_cm.__aexit__.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_temp_table_uses_on_commit_drop(self, mock_conn):
        """The CREATE TEMP TABLE statement must use ON COMMIT DROP so the
        table is transaction-scoped. Without ON COMMIT DROP the table would
        be session-scoped and would not survive pgBouncer backend rebinding
        between transactions."""
        emb = [0.1] * 384
        await compute_semantic_links_ann(
            conn=mock_conn,
            bank_id="bank-1",
            unit_ids=["u1"],
            embeddings=[emb],
            fact_types=["world"],
            threshold=DEFAULT_SEMANTIC_LINK_MIN_SIMILARITY,
        )

        executed_sql = [call.args[0] for call in mock_conn.execute.call_args_list]
        create_statements = [s for s in executed_sql if "CREATE TEMP TABLE" in s]
        assert len(create_statements) == 1, "Should create _ann_seeds exactly once"
        assert "_ann_seeds" in create_statements[0]
        assert "ON COMMIT DROP" in create_statements[0], (
            "CREATE TEMP TABLE must use ON COMMIT DROP so the table is cleaned "
            "up at transaction end and is transaction-scoped"
        )

        # Must not use IF NOT EXISTS — the table is fresh each transaction.
        assert "IF NOT EXISTS" not in create_statements[0], (
            "With ON COMMIT DROP the table is always fresh at transaction start, "
            "so IF NOT EXISTS is both unnecessary and misleading (suggests the "
            "table might persist across transactions)"
        )

    @pytest.mark.asyncio
    async def test_no_manual_drop_or_truncate(self, mock_conn):
        """With ON COMMIT DROP we must not re-add manual TRUNCATE or DROP
        statements — they were the source of the original pgBouncer bug."""
        emb = [0.1] * 384
        await compute_semantic_links_ann(
            conn=mock_conn,
            bank_id="bank-1",
            unit_ids=["u1"],
            embeddings=[emb],
            fact_types=["world"],
            threshold=DEFAULT_SEMANTIC_LINK_MIN_SIMILARITY,
        )

        executed_sql = [call.args[0] for call in mock_conn.execute.call_args_list]
        assert not any("TRUNCATE _ann_seeds" in s for s in executed_sql), (
            "TRUNCATE is unnecessary with ON COMMIT DROP and was previously "
            "the statement that failed with 'relation does not exist' when "
            "pgBouncer rebound the backend"
        )
        assert not any("DROP TABLE" in s and "_ann_seeds" in s for s in executed_sql), (
            "Explicit DROP is unnecessary with ON COMMIT DROP"
        )

    @pytest.mark.asyncio
    async def test_uses_set_local_for_pgvector_ann_tuning(self, mock_conn, monkeypatch):
        """The per-backend ANN tuning GUC must be set with SET LOCAL so the
        change is scoped to the transaction. Without SET LOCAL, the setting
        would leak onto the pooled backend and affect subsequent recall
        queries that land on the same backend."""
        monkeypatch.setenv("HINDSIGHT_API_VECTOR_EXTENSION", "pgvector")
        guc = "hnsw.ef_search"
        emb = [0.1] * 384
        await compute_semantic_links_ann(
            conn=mock_conn,
            bank_id="bank-1",
            unit_ids=["u1"],
            embeddings=[emb],
            fact_types=["world"],
            threshold=DEFAULT_SEMANTIC_LINK_MIN_SIMILARITY,
        )

        executed_sql = [call.args[0] for call in mock_conn.execute.call_args_list]
        tuning_statements = [s for s in executed_sql if guc in s]
        assert tuning_statements, f"{guc} must be tuned for retain ANN under pgvector"
        for stmt in tuning_statements:
            assert stmt.strip().startswith("SET LOCAL"), f"{guc} must use SET LOCAL, got: {stmt}"
        # And there must not be a RESET — SET LOCAL handles it at commit.
        assert not any(f"RESET {guc}" in s for s in executed_sql)

    @pytest.mark.asyncio
    async def test_vchord_ann_does_not_set_fixed_probe_count(self, mock_conn, monkeypatch):
        """VectorChord probe counts must come from index/default config.

        VectorChord requires vchordrq.probes to match the index's
        build.internal.lists shape. Hindsight must not apply one fixed session
        GUC across listless and partitioned vchordrq indexes.
        """
        monkeypatch.setenv("HINDSIGHT_API_VECTOR_EXTENSION", "vchord")
        emb = [0.1] * 384
        await compute_semantic_links_ann(
            conn=mock_conn,
            bank_id="bank-1",
            unit_ids=["u1"],
            embeddings=[emb],
            fact_types=["world"],
            threshold=DEFAULT_SEMANTIC_LINK_MIN_SIMILARITY,
        )

        executed_sql = [call.args[0] for call in mock_conn.execute.call_args_list]
        assert not any("vchordrq.probes" in s for s in executed_sql)
