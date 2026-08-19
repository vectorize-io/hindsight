from pathlib import Path

from hindsight_api import _vector_index
from hindsight_api._vector_index import (
    SCANN_MIN_ROWS_FOR_AUTO_INDEX,
    ann_search_tuning_settings,
    bootstrap_extension,
    configured_vector_extension,
    index_type_keyword,
    index_using_clause,
    pg_extension_name,
    should_defer_index_creation,
    uses_per_bank_vector_indexes,
    validate_extension,
)
from hindsight_api.engine.retain import bank_utils
from hindsight_api.migrations import _bootstrap_vector_extension_for_migrations


class RecordingConn:
    def __init__(self):
        self.statements = []

    def execute(self, statement, *args, **kwargs):
        self.statements.append(str(statement))


def test_validate_extension_accepts_scann():
    assert validate_extension("scann") == "scann"
    assert validate_extension("ScaNN") == "scann"


def test_pg_extension_name_maps_scann_to_alloydb_extension():
    assert pg_extension_name("scann") == "alloydb_scann"


def test_index_using_clause_scann_uses_cosine_auto_mode():
    clause = index_using_clause("scann")

    assert "USING scann (embedding cosine)" in clause
    assert "mode = 'AUTO'" in clause


def test_index_using_clause_pgvector_matches_existing_clause():
    assert index_using_clause("pgvector") == "USING hnsw (embedding vector_cosine_ops)"


def test_index_using_clause_vchord_uses_cosine_ops():
    # vchordrq opclasses are bound 1:1 to operators in PostgreSQL; the engine
    # uses `<=>` (cosine distance) everywhere, so the index must be declared
    # with vector_cosine_ops or the planner falls back to a sequential scan.
    assert index_using_clause("vchord") == "USING vchordrq (embedding vector_cosine_ops)"


def test_index_type_keyword_scann_round_trips_pg_indexes_indexdef():
    keyword = index_type_keyword("scann")
    indexdef = "CREATE INDEX idx ON memory_units USING scann (embedding cosine) WITH (mode='AUTO')"

    assert keyword == "scann"
    assert keyword in indexdef.lower()


def test_bootstrap_extension_scann_installs_vector_before_alloydb_scann():
    conn = RecordingConn()

    bootstrap_extension(conn, "scann")

    assert conn.statements == [
        "CREATE EXTENSION IF NOT EXISTS vector",
        "CREATE EXTENSION IF NOT EXISTS alloydb_scann CASCADE",
    ]


def test_migration_bootstrap_vchord_skips_pgvector_preflight():
    conn = RecordingConn()

    _bootstrap_vector_extension_for_migrations(conn, "vchord")

    assert conn.statements == ["CREATE EXTENSION IF NOT EXISTS vchord CASCADE"]


def test_migration_bootstrap_scann_uses_dispatcher_without_legacy_pgvector_check():
    conn = RecordingConn()

    _bootstrap_vector_extension_for_migrations(conn, "scann")

    assert conn.statements == [
        "CREATE EXTENSION IF NOT EXISTS vector",
        "CREATE EXTENSION IF NOT EXISTS alloydb_scann CASCADE",
    ]


def test_scann_index_creation_defers_until_table_is_large_enough():
    assert should_defer_index_creation("scann", 0)
    assert should_defer_index_creation("scann", SCANN_MIN_ROWS_FOR_AUTO_INDEX - 1)
    assert not should_defer_index_creation("scann", SCANN_MIN_ROWS_FOR_AUTO_INDEX)
    assert not should_defer_index_creation("pgvector", 0)


def test_ann_search_tuning_settings_pgvector_dispatches_hnsw_ef_search():
    assert ann_search_tuning_settings("pgvector", kind="low_latency") == (("hnsw.ef_search", "60"),)
    assert ann_search_tuning_settings("pgvector", kind="high_recall") == (("hnsw.ef_search", "200"),)


def test_ann_search_tuning_settings_vchord_leaves_probes_to_index_defaults():
    # vchordrq.probes must match the index's build.internal.lists shape.
    # VectorChord 1.1 supports per-index fallback parameters for this; a
    # session GUC would override every index and can be invalid for listless or
    # mixed-layout indexes.
    assert ann_search_tuning_settings("vchord", kind="low_latency") == ()
    assert ann_search_tuning_settings("vchord", kind="high_recall") == ()


def test_ann_search_tuning_settings_returns_empty_for_backends_without_knob():
    for ext in ("vchord", "pgvectorscale", "pg_diskann", "scann"):
        assert ann_search_tuning_settings(ext, kind="low_latency") == ()
        assert ann_search_tuning_settings(ext, kind="high_recall") == ()


def test_configured_vector_extension_defaults_to_pgvector(monkeypatch):
    monkeypatch.delenv("HINDSIGHT_API_VECTOR_EXTENSION", raising=False)
    assert configured_vector_extension() == "pgvector"


def test_configured_vector_extension_reads_env_and_lowercases(monkeypatch):
    monkeypatch.setenv("HINDSIGHT_API_VECTOR_EXTENSION", "VChord")
    assert configured_vector_extension() == "vchord"


def test_configured_vector_extension_rejects_unknown_value(monkeypatch):
    import pytest

    monkeypatch.setenv("HINDSIGHT_API_VECTOR_EXTENSION", "bogus")
    with pytest.raises(ValueError):
        configured_vector_extension()


def test_scann_does_not_use_per_bank_partial_indexes():
    assert not uses_per_bank_vector_indexes("scann")
    assert uses_per_bank_vector_indexes("pgvector")
    assert uses_per_bank_vector_indexes("pgvectorscale")
    assert uses_per_bank_vector_indexes("vchord")


def test_alembic_vector_migrations_freeze_vector_sql_locally():
    # Pre-existing vchord-related migrations keep their vector SQL frozen
    # (inline helpers, no import from hindsight_api._vector_index) so their
    # historical behaviour cannot drift when _vector_index.py is updated.
    # New vchord migrations are reviewed for this trade-off individually and
    # may import the shared helpers when they want to share the canonical
    # source of truth (see b8c9d0e1f2a3_vchord_cosine_opclass.py).
    migration_dir = Path(__file__).resolve().parent.parent / "hindsight_api/alembic/versions"
    frozen_migrations = [
        "5a366d414dce_initial_schema.py",
        "a4b5c6d7e8f9_fix_per_bank_vector_index_type.py",
        "d5e6f7a8b9c0_add_bank_internal_id_and_per_bank_hnsw.py",
        "n9i0j1k2l3m4_learnings_and_pinned_reflections.py",
    ]

    for migration in frozen_migrations:
        text = (migration_dir / migration).read_text()
        assert "hindsight_api._vector_index" not in text


class ScannConfig:
    vector_extension = "scann"


def test_vector_index_clause_is_none_for_scann(monkeypatch):
    """ScaNN has no per-bank index layout, so callers get no USING clause to build with."""
    monkeypatch.setattr(bank_utils, "get_config", lambda: ScannConfig())

    assert bank_utils._vector_index_clause() is None


class _ThresholdConfig:
    def __init__(self, min_rows: int):
        self.vector_index_min_rows = min_rows


def _with_threshold(monkeypatch, min_rows: int) -> None:
    monkeypatch.setattr(_vector_index, "get_config", lambda: _ThresholdConfig(min_rows), raising=False)
    monkeypatch.setattr("hindsight_api.config.get_config", lambda: _ThresholdConfig(min_rows))


def test_qualifies_at_and_around_the_threshold(monkeypatch):
    """The build side is a floor, inclusive: exactly the threshold qualifies."""
    _with_threshold(monkeypatch, 10_000)

    assert not _vector_index.qualifies_for_per_bank_index(9_999)
    assert _vector_index.qualifies_for_per_bank_index(10_000)
    assert _vector_index.qualifies_for_per_bank_index(10_001)


def test_zero_threshold_indexes_every_partition_that_holds_rows(monkeypatch):
    """0 is the shipped default and means "no minimum" — the pre-threshold behaviour."""
    _with_threshold(monkeypatch, 0)

    assert _vector_index.qualifies_for_per_bank_index(1)
    assert _vector_index.qualifies_for_per_bank_index(10_000_000)


def test_an_empty_partition_never_qualifies(monkeypatch):
    """Zero rows is excluded at every threshold, including the default of 0.

    By arithmetic alone `0 >= 0` holds, which would entitle every bank in the
    deployment to three indexes over nothing the moment it was created — the
    index explosion the threshold exists to prevent, reintroduced by its own
    default. An index over no rows serves no query either way.
    """
    _with_threshold(monkeypatch, 0)
    assert not _vector_index.qualifies_for_per_bank_index(0)

    _with_threshold(monkeypatch, 10_000)
    assert not _vector_index.qualifies_for_per_bank_index(0)


def test_emptied_partition_loses_its_index_at_every_threshold(monkeypatch):
    """Zero rows never keeps an index — including at the shipped default of 0.

    Regression for a drop side that was dead in the default configuration. The
    check was `row_count < per_bank_index_drop_rows()`, and at a threshold of 0
    the drop floor is also 0, so it read `0 < 0` — never true. Every bank ever
    written to and then cleared kept three ANN indexes over nothing, forever,
    because nothing writes to an emptied bank. That is the exact accumulation
    the threshold exists to prevent, reintroduced by its own default.
    """
    _with_threshold(monkeypatch, 0)
    assert not _vector_index.should_keep_per_bank_index(0)

    _with_threshold(monkeypatch, 10_000)
    assert not _vector_index.should_keep_per_bank_index(0)


def test_keeping_starts_below_building(monkeypatch):
    """The hysteresis band: a partition between the two bounds is left alone.

    Keeping has to start lower than building, or a bank hovering at the
    threshold rebuilds and drops the same ANN index on alternating writes.
    """
    _with_threshold(monkeypatch, 10_000)
    between = _vector_index.per_bank_index_drop_rows() + 1

    assert not _vector_index.qualifies_for_per_bank_index(between), "not enough to earn a new index"
    assert _vector_index.should_keep_per_bank_index(between), "but enough to keep one it already has"


def test_drop_threshold_sits_strictly_below_the_build_threshold(monkeypatch):
    """The hysteresis gap must be non-empty, or an index at the boundary flaps.

    A partition between the two bounds is neither built nor dropped: if it has
    an index it keeps it, and if it does not it stays without one.
    """
    _with_threshold(monkeypatch, 10_000)

    build = _vector_index.per_bank_index_min_rows()
    drop = _vector_index.per_bank_index_drop_rows()

    assert drop < build
    assert not _vector_index.qualifies_for_per_bank_index(drop)
