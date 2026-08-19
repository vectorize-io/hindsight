"""Shared PostgreSQL vector-extension dispatch helpers."""

from __future__ import annotations

import logging
import os

from sqlalchemy import text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)

# Extensions a user can set via HINDSIGHT_API_VECTOR_EXTENSION.
CONFIGURABLE_EXTENSIONS = ("pgvector", "pgvectorscale", "vchord", "scann")

# Extensions detect_vector_extension() can return. pg_diskann is a runtime-only
# resolution from a configured "pgvectorscale" backend on Azure (uses a different
# WITH clause), never a value the user sets directly.
RESOLVED_EXTENSIONS = (*CONFIGURABLE_EXTENSIONS, "pg_diskann")

# Backwards-compatible alias for older imports.
VALID_EXTENSIONS = CONFIGURABLE_EXTENSIONS

SCANN_MIN_ROWS_FOR_AUTO_INDEX = 10_000


_EXTENSION_NAMES = {
    "pgvector": "vector",
    "pgvectorscale": "vectorscale",
    "vchord": "vchord",
    "scann": "alloydb_scann",
}

_INDEX_USING_CLAUSES = {
    "pgvector": "USING hnsw (embedding vector_cosine_ops)",
    "pgvectorscale": "USING diskann (embedding vector_cosine_ops) WITH (num_neighbors = 50)",
    "pg_diskann": "USING diskann (embedding vector_cosine_ops) WITH (max_neighbors = 50)",
    "vchord": "USING vchordrq (embedding vector_cosine_ops)",
    "scann": "USING scann (embedding cosine) WITH (mode = 'AUTO')",
}

_INDEX_TYPE_KEYWORDS = {
    "pgvector": "hnsw",
    "pgvectorscale": "diskann",
    "pg_diskann": "diskann",
    "vchord": "vchordrq",
    "scann": "scann",
}

# Per-backend ANN search-time tuning GUCs. Each entry is a tuple of
# (guc_name, value) pairs the caller can apply with SET or SET LOCAL.
#
# - pgvector exposes hnsw.ef_search. The 60 / 200 pair is unchanged from the
#   pre-dispatcher code (internal benchmarks tuned around our embedding count
#   and recall floor; see the link_utils / pool init call sites for the
#   latency-vs-recall framing).
# - vchord exposes vchordrq.probes, but its shape must match the index's
#   build.internal.lists hierarchy. VectorChord 1.1 added per-index fallback
#   parameters for this reason: a session GUC overrides every vchordrq index,
#   and a single value can be invalid for listless or mixed-layout indexes.
#   Hindsight's built-in vchord clause does not set lists, so the safe default
#   is no session-level probe override; deployments that partition vchordrq
#   indexes should attach probes to the index storage parameters instead.
# - pgvectorscale / pg_diskann / scann do not expose an equivalent per-statement
#   knob in the engine today, so the dispatcher returns no statements for them.
_ANN_TUNING_LOW_LATENCY: dict[str, tuple[tuple[str, str], ...]] = {
    "pgvector": (("hnsw.ef_search", "60"),),
}
_ANN_TUNING_HIGH_RECALL: dict[str, tuple[tuple[str, str], ...]] = {
    "pgvector": (("hnsw.ef_search", "200"),),
}

_EXTENSION_INSTALL_SQL = {
    "pgvector": ("CREATE EXTENSION IF NOT EXISTS vector",),
    "pgvectorscale": (
        "CREATE EXTENSION IF NOT EXISTS vector",
        "CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE",
    ),
    "vchord": ("CREATE EXTENSION IF NOT EXISTS vchord CASCADE",),
    "scann": (
        "CREATE EXTENSION IF NOT EXISTS vector",
        "CREATE EXTENSION IF NOT EXISTS alloydb_scann CASCADE",
    ),
}

_INSTALL_HINTS = {
    "pgvector": "CREATE EXTENSION vector;",
    "pgvectorscale": "CREATE EXTENSION vector; then CREATE EXTENSION vectorscale CASCADE; (or pg_diskann on Azure)",
    "vchord": "CREATE EXTENSION vchord CASCADE;",
    "scann": "CREATE EXTENSION vector; then CREATE EXTENSION alloydb_scann CASCADE;",
}


def configured_vector_extension() -> str:
    """Return the user-configured vector backend extension.

    Reads ``HINDSIGHT_API_VECTOR_EXTENSION`` (default ``"pgvector"``) and
    validates it via :func:`validate_extension`. This is the single source of
    truth for runtime code that needs to dispatch behaviour by vector backend;
    callers should prefer this over reading the env var directly, so the
    default value and the lookup mechanism live in one place.
    """
    return validate_extension(os.getenv("HINDSIGHT_API_VECTOR_EXTENSION", "pgvector"))


def validate_extension(name: str) -> str:
    """Return a normalized configurable vector extension name or raise.

    Used at the user-facing config boundary; pg_diskann is rejected here because
    it is a detection-time alias, never a value the user sets directly.
    """
    ext = name.lower()
    if ext not in CONFIGURABLE_EXTENSIONS:
        valid = ", ".join(CONFIGURABLE_EXTENSIONS)
        raise ValueError(f"Invalid vector_extension: {name}. Must be one of: {valid}")
    return ext


def _normalize_resolved(name: str) -> str:
    """Normalize either a user-configurable or detect-time extension name."""
    ext = name.lower()
    if ext not in RESOLVED_EXTENSIONS:
        valid = ", ".join(RESOLVED_EXTENSIONS)
        raise ValueError(f"Unknown vector extension: {name}. Must be one of: {valid}")
    return ext


def pg_extension_name(ext: str) -> str:
    """Return the PostgreSQL extension name for a configured vector backend."""
    return _EXTENSION_NAMES[validate_extension(ext)]


def index_using_clause(ext: str) -> str:
    """Return the CREATE INDEX USING clause for the vector backend."""
    return _INDEX_USING_CLAUSES[_normalize_resolved(ext)]


def index_type_keyword(ext: str) -> str:
    """Return the keyword that identifies this index type in pg_indexes.indexdef."""
    return _INDEX_TYPE_KEYWORDS[_normalize_resolved(ext)]


def minimum_rows_for_index(ext: str) -> int:
    """Return the minimum populated embedding rows before creating this index type."""
    return SCANN_MIN_ROWS_FOR_AUTO_INDEX if _normalize_resolved(ext) == "scann" else 0


def should_defer_index_creation(ext: str, row_count: int) -> bool:
    """Return True when index creation should wait for more embeddings."""
    minimum_rows = minimum_rows_for_index(ext)
    return minimum_rows > 0 and row_count < minimum_rows


def ann_search_tuning_settings(ext: str, *, kind: str) -> tuple[tuple[str, str], ...]:
    """Return per-backend (guc_name, value) pairs for ANN search-time tuning.

    ``kind`` is ``"low_latency"`` for retain-side link probing (smaller probe
    count, lower recall, lower latency) and ``"high_recall"`` for connection
    init in the pool (larger probe count, higher recall). Callers wrap each
    pair with ``SET LOCAL`` or ``SET`` themselves so the same dispatcher works
    for both transaction-scoped and session-scoped use. Returns an empty tuple
    for backends without an equivalent knob.
    """
    if kind == "low_latency":
        table = _ANN_TUNING_LOW_LATENCY
    elif kind == "high_recall":
        table = _ANN_TUNING_HIGH_RECALL
    else:
        raise ValueError(f"Unknown ANN tuning kind: {kind!r}")
    return table.get(_normalize_resolved(ext), ())


def uses_per_bank_vector_indexes(ext: str) -> bool:
    """Return whether the backend should create per-bank partial vector indexes."""
    return _normalize_resolved(ext) != "scann"


def per_bank_index_min_rows() -> int:
    """Rows a (bank, fact_type) needs before it earns its own partial vector index.

    Distinct from :func:`minimum_rows_for_index`, which is ScaNN's *build*
    requirement for its single global index (AlloyDB cannot construct one below
    a floor). This is a cost policy for the per-bank backends: the indexes sit on
    the shared ``memory_units`` table, so each one is enumerated and locked at
    plan time by queries belonging to every *other* bank, and opened by every DML
    statement against the table. A small bank's index cannot repay that — the
    ``(bank_id, fact_type)`` B-tree plus a top-N sort answers the same query
    exactly and faster. See issue #3485.

    Read from config rather than passed in because the write path's pre-check,
    the maintenance operation and the admin command must all apply the same
    number; a threshold that differed between the one deciding to queue work and
    the one deciding what to do would either oscillate or never converge.
    """
    from .config import get_config

    return get_config().vector_index_min_rows


def per_bank_index_drop_rows() -> int:
    """Row count below which an existing per-bank vector index is dropped.

    Strictly below :func:`per_bank_index_min_rows` so the build and drop
    decisions cannot both be true at one row count. Without the gap, a bank
    hovering at the threshold — consolidation prunes a few facts, retain adds
    them back — would rebuild and drop the same ANN index on alternating sweeps.
    """
    from .config import VECTOR_INDEX_DROP_RATIO

    return int(per_bank_index_min_rows() * VECTOR_INDEX_DROP_RATIO)


def should_keep_per_bank_index(row_count: int) -> bool:
    """Whether an *existing* index on a partition of ``row_count`` rows is kept.

    The counterpart to :func:`qualifies_for_per_bank_index`, and deliberately a
    separate, lower bound: keeping starts below building, so a partition
    hovering at the threshold does not rebuild and drop the same ANN index on
    alternating writes.

    The ``row_count > 0`` term is not redundant with the ratio. At the default
    threshold of 0 the drop floor is also 0, so a bare ``row_count >= floor``
    keeps an index over an *emptied* partition forever — every bank ever written
    to and then cleared would hold three indexes over nothing, which is the
    accumulation the threshold exists to prevent. An emptied partition loses its
    index at every threshold.
    """
    return row_count > 0 and row_count >= per_bank_index_drop_rows()


def qualifies_for_per_bank_index(row_count: int) -> bool:
    """Whether a (bank, fact_type) holding ``row_count`` rows should have an index.

    At the default threshold of 0 this is true for every partition that holds
    any rows at all, which is the behaviour before the threshold existed.

    An empty partition is excluded explicitly rather than by arithmetic: at a
    threshold of 0, ``row_count >= minimum`` alone is true for zero rows, so
    every bank in the deployment would be entitled to three indexes over nothing
    the moment it was created — the exact index explosion the threshold exists
    to prevent, reintroduced by its own default.

    Only the build side: an existing index is kept until the count falls under
    :func:`per_bank_index_drop_rows`, so callers reconciling live state must
    consult both bounds rather than treating this as the full policy.

    Takes no extension: the backend question is settled before any reconcile
    runs (``uses_per_bank_vector_indexes`` gates the maintenance operation and
    ``_vector_index_clause`` gates the admin command), so re-asking it here
    would be a second, weaker copy of a decision already made.
    """
    return row_count > 0 and row_count >= per_bank_index_min_rows()


def bootstrap_extension(conn: Connection, ext: str) -> None:
    """Install the configured vector extension and any prerequisites if possible."""
    normalized = validate_extension(ext)
    for statement in _EXTENSION_INSTALL_SQL[normalized]:
        conn.execute(text(statement))


def detect_vector_extension(conn: Connection, vector_extension: str = "pgvector") -> str:
    """Validate the configured vector extension exists and return the index backend."""
    configured_ext = validate_extension(vector_extension)

    if configured_ext == "pgvectorscale":
        pgvector_check = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).scalar()
        if not pgvector_check:
            raise RuntimeError(
                "DiskANN (pgvectorscale/pg_diskann) requires pgvector to be installed. "
                f"Install it with: {_INSTALL_HINTS['pgvectorscale']}"
            )

        vectorscale_check = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vectorscale'")).scalar()
        pg_diskann_check = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'pg_diskann'")).scalar()

        if vectorscale_check:
            logger.debug("Using vector extension: pgvectorscale (DiskANN)")
            return "pgvectorscale"
        if pg_diskann_check:
            logger.debug("Using vector extension: pg_diskann (Azure DiskANN)")
            return "pg_diskann"

        raise RuntimeError(
            "Configured vector extension 'pgvectorscale' not found. Install either:\n"
            "  - pgvectorscale (open source): CREATE EXTENSION vectorscale CASCADE;\n"
            "  - pg_diskann (Azure): CREATE EXTENSION pg_diskann CASCADE;"
        )

    extension_name = pg_extension_name(configured_ext)
    extension_check = conn.execute(
        text("SELECT 1 FROM pg_extension WHERE extname = :extension_name"),
        {"extension_name": extension_name},
    ).scalar()
    if not extension_check:
        raise RuntimeError(
            f"Configured vector extension '{configured_ext}' not found. "
            f"Install it with: {_INSTALL_HINTS[configured_ext]}"
        )

    logger.debug("Using configured vector extension: %s", configured_ext)
    return configured_ext
