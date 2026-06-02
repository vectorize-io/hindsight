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
# - vchord exposes vchordrq.probes (no default; see VectorChord issue #392)
#   and vchordrq.epsilon (default 1.9). probes = 10 / 30 are starting
#   defaults pending a workload-specific sweep — vchordrq's recall curve
#   shape differs from HNSW's, so the pgvector numbers don't translate
#   directly. Revisit with a per-cluster benchmark once we have production
#   recall data; until then these are deliberately conservative on the
#   high-recall path. We leave epsilon at its default; tightening it is a
#   separate trade-off.
# - pgvectorscale / pg_diskann / scann do not expose an equivalent per-statement
#   knob in the engine today, so the dispatcher returns no statements for them.
_ANN_TUNING_LOW_LATENCY: dict[str, tuple[tuple[str, str], ...]] = {
    "pgvector": (("hnsw.ef_search", "60"),),
    "vchord": (("vchordrq.probes", "10"),),
}
_ANN_TUNING_HIGH_RECALL: dict[str, tuple[tuple[str, str], ...]] = {
    "pgvector": (("hnsw.ef_search", "200"),),
    "vchord": (("vchordrq.probes", "30"),),
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
