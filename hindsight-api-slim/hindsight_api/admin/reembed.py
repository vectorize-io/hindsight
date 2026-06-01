"""PostgreSQL embedding re-materialization helpers for hindsight-admin."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import asyncpg

from .._vector_index import (
    ann_search_tuning_settings,
    index_type_keyword,
    index_using_clause,
    minimum_rows_for_index,
    resolve_vector_extension_from_installed,
    should_defer_index_creation,
    uses_per_bank_vector_indexes,
    validate_vector_index_dimension,
)
from ..config import (
    DEFAULT_DATABASE_SCHEMA,
    DEFAULT_EMBEDDINGS_OPENAI_MODEL,
    ENV_EMBEDDINGS_OPENAI_MODEL,
    HindsightConfig,
    clear_config_cache,
)
from ..engine.embeddings import create_embeddings_from_env
from ..engine.retain import embedding_processing, embedding_utils
from ..engine.retain.link_utils import SEMANTIC_LINK_THRESHOLD, STREAMING_SEMANTIC_LINK_TOP_K

REEMBED_MIGRATIONS_TABLE = "embedding_reembed_migrations"
REEMBED_SEMANTIC_LINKS_TABLE = "embedding_reembed_semantic_links"
SHADOW_COLUMN = "embedding_reembed"
ACTIVE_REEMBED_STATUSES = ("running", "prepared", "failed")
_MEMORY_UNIT_FACT_TYPES = {"world": "worl", "experience": "expr", "observation": "obsv"}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReembedOptions:
    batch_size: int = 100
    dry_run: bool = False
    max_retries: int = 3
    index_max_parallel_maintenance_workers: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be at least 0")
        if self.index_max_parallel_maintenance_workers is not None and self.index_max_parallel_maintenance_workers < 0:
            raise ValueError("index_max_parallel_maintenance_workers must be at least 0")


@dataclass(frozen=True)
class ReembedReport:
    schema: str
    migration_id: str | None
    status: str
    memory_units_total: int
    memory_units_done: int
    mental_models_total: int
    mental_models_done: int
    semantic_links_staged: int
    shadow_indexes_state: str | None
    message: str


@dataclass(frozen=True)
class ReembedProgress:
    schema: str
    phase: str
    migration_id: str | None = None
    memory_units_done: int | None = None
    memory_units_total: int | None = None
    mental_models_done: int | None = None
    mental_models_total: int | None = None
    semantic_links_staged: int | None = None
    group_index: int | None = None
    group_total: int | None = None
    message: str | None = None


@dataclass
class ReembedProgressCounts:
    memory_units_total: int
    memory_units_done: int
    mental_models_total: int
    mental_models_done: int


@dataclass(frozen=True)
class ShadowEmbeddingCounts:
    memory_units: int
    mental_models: int


@dataclass(frozen=True)
class ReembedStatusMigration:
    migration_id: uuid.UUID
    status: str
    embedding_state: str | None
    shadow_indexes_state: str | None
    semantic_links_state: str | None
    started_at: datetime | None
    updated_at: datetime | None
    error_message: str | None


@dataclass(frozen=True)
class ReembedStatusReport:
    schema: str
    status: str
    migrations: list[ReembedStatusMigration]
    state: ReembedState | None


ProgressCallback = Callable[[ReembedProgress], None]


class _ReembedLockUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class _WorklistIndexSpec:
    name: str
    table: str


_REEMBED_WORKLIST_INDEXES = (
    _WorklistIndexSpec("idx_memory_units_reembed_worklist", "memory_units"),
    _WorklistIndexSpec("idx_mental_models_reembed_worklist", "mental_models"),
)

_EMBEDDING_IDENTITY_FIELDS_BY_PROVIDER = {
    "local": ("embeddings_local_model", "embeddings_local_trust_remote_code"),
    "tei": ("embeddings_tei_url",),
    "openai": ("embeddings_openai_model", "embeddings_openai_base_url", "embeddings_openai_dimensions"),
    "openai-codex": ("embeddings_openai_model", "embeddings_openai_dimensions"),
    "openrouter": ("embeddings_openrouter_model", "embeddings_openai_dimensions"),
    "zeroentropy": (
        "embeddings_zeroentropy_model",
        "embeddings_zeroentropy_base_url",
        "embeddings_zeroentropy_dimensions",
        "embeddings_zeroentropy_encoding_format",
        "embeddings_zeroentropy_latency",
    ),
    "cohere": ("embeddings_cohere_model", "embeddings_cohere_base_url", "embeddings_cohere_output_dimensions"),
    "litellm": ("embeddings_litellm_model", "embeddings_litellm_api_base"),
    "litellm-sdk": (
        "embeddings_litellm_sdk_model",
        "embeddings_litellm_sdk_api_base",
        "embeddings_litellm_sdk_output_dimensions",
        "embeddings_litellm_sdk_encoding_format",
    ),
    "google": (
        "embeddings_gemini_model",
        "embeddings_gemini_output_dimensionality",
        "embeddings_vertexai_project_id",
        "embeddings_vertexai_region",
    ),
}


@dataclass(frozen=True)
class ReembedState:
    active_migration_id: str | None
    active_status: str | None
    embedding_state: str | None
    shadow_indexes_state: str | None
    semantic_links_state: str | None
    shadow_columns: list[str]
    shadow_indexes: list[str]
    staging_rows: int

    @property
    def has_active_migration(self) -> bool:
        return self.active_migration_id is not None

    @property
    def has_shadow_state(self) -> bool:
        return bool(self.shadow_columns or self.shadow_indexes or self.staging_rows)


def _quote_ident(name: str) -> str:
    if not name or "\x00" in name:
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return '"' + name.replace('"', '""') + '"'


def _emit_progress(callback: ProgressCallback | None, progress: ReembedProgress) -> None:
    if callback is not None:
        callback(progress)


def _reembed_lock_key(schema: str) -> str:
    return f"hindsight-reembed:{schema}"


async def _try_acquire_reembed_lock(conn: asyncpg.Connection, schema: str) -> None:
    acquired = bool(await conn.fetchval("SELECT pg_try_advisory_lock(hashtext($1))", _reembed_lock_key(schema)))
    if not acquired:
        raise _ReembedLockUnavailable(f"Another reembed command is already running for schema '{schema}'")


async def _release_reembed_lock(conn: asyncpg.Connection, schema: str) -> None:
    await conn.execute("SELECT pg_advisory_unlock(hashtext($1))", _reembed_lock_key(schema))


def _fq(schema: str, table: str) -> str:
    return f"{_quote_ident(schema)}.{_quote_ident(table)}"


def _format_embedding(values: list[float]) -> str:
    return "[" + ",".join(str(v) for v in values) + "]"


def _embedding_identity_field(config: HindsightConfig, name: str) -> Any:
    if name == "embeddings_openai_model":
        return os.getenv(ENV_EMBEDDINGS_OPENAI_MODEL, DEFAULT_EMBEDDINGS_OPENAI_MODEL)
    return getattr(config, name)


def _model_name(config: HindsightConfig) -> str:
    provider = config.embeddings_provider.lower()
    if provider == "local":
        return config.embeddings_local_model
    if provider == "tei":
        return config.embeddings_tei_url or "tei"
    if provider in {"openai", "openai-codex"}:
        return _embedding_identity_field(config, "embeddings_openai_model")
    if provider == "openrouter":
        return config.embeddings_openrouter_model
    if provider == "zeroentropy":
        return config.embeddings_zeroentropy_model
    if provider == "cohere":
        return config.embeddings_cohere_model
    if provider == "litellm":
        return config.embeddings_litellm_model
    if provider == "litellm-sdk":
        return config.embeddings_litellm_sdk_model
    if provider == "google":
        return config.embeddings_gemini_model
    return provider


def _model_identity(config: HindsightConfig, dimension: int) -> dict[str, Any]:
    provider = config.embeddings_provider.lower()
    identity: dict[str, Any] = {
        "provider": config.embeddings_provider,
        "model": _model_name(config),
        "dimension": dimension,
        "vector_extension": config.vector_extension,
    }
    for name in _EMBEDDING_IDENTITY_FIELDS_BY_PROVIDER.get(provider, ()):
        identity[name] = _embedding_identity_field(config, name)
    return identity


def _config_fingerprint(identity: dict[str, Any]) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _memory_unit_embedding_text(row: asyncpg.Record) -> str:
    source_count = row["source_count"] or 0
    if row["fact_type"] == "observation" and source_count > 0:
        return row["text"]

    return embedding_processing.build_fact_embedding_text(
        fact_text=row["text"],
        occurred_start=row["occurred_start"],
        occurred_end=row["occurred_end"],
        mentioned_at=row["mentioned_at"],
        entities=row["entities"] or [],
        format_date_fn=embedding_processing.format_readable_date,
    )


async def _table_exists(conn: asyncpg.Connection, schema: str, table: str) -> bool:
    return bool(
        await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = $1 AND table_name = $2 AND table_type = 'BASE TABLE'
            )
            """,
            schema,
            table,
        )
    )


async def _existing_tables(conn: asyncpg.Connection, schema: str, tables: tuple[str, ...]) -> set[str]:
    rows = await conn.fetch(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = $1
          AND table_name = ANY($2::text[])
          AND table_type = 'BASE TABLE'
        """,
        schema,
        list(tables),
    )
    return {row["table_name"] for row in rows}


async def _column_exists(conn: asyncpg.Connection, schema: str, table: str, column: str) -> bool:
    return bool(
        await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = $1 AND table_name = $2 AND column_name = $3
            )
            """,
            schema,
            table,
            column,
        )
    )


def _coerce_migration_id(migration_id: str | uuid.UUID) -> uuid.UUID:
    return migration_id if isinstance(migration_id, uuid.UUID) else uuid.UUID(migration_id)


async def _fetch_migration(
    conn: asyncpg.Connection,
    schema: str,
    migration_id: str | uuid.UUID,
) -> asyncpg.Record | None:
    return await conn.fetchrow(
        f"SELECT * FROM {_fq(schema, REEMBED_MIGRATIONS_TABLE)} WHERE migration_id = $1",
        _coerce_migration_id(migration_id),
    )


async def _fetch_active_migration(conn: asyncpg.Connection, schema: str) -> asyncpg.Record | None:
    rows = await conn.fetch(
        f"""
        SELECT *
        FROM {_fq(schema, REEMBED_MIGRATIONS_TABLE)}
        WHERE status = ANY($1::text[])
        ORDER BY updated_at DESC
        """,
        list(ACTIVE_REEMBED_STATUSES),
    )
    if len(rows) > 1:
        migration_ids = ", ".join(str(row["migration_id"]) for row in rows)
        raise RuntimeError(
            f"Schema '{schema}' has multiple active reembed migrations: {migration_ids}. "
            "Resolve or abandon the extra active migrations before continuing."
        )
    return rows[0] if rows else None


def _semantic_seed_filter(alias: str) -> str:
    prefix = f"{alias}." if alias else ""
    return f"NOT ({prefix}fact_type = 'observation' AND COALESCE(cardinality({prefix}source_memory_ids), 0) > 0)"


async def _ensure_reembed_worklist_indexes(conn: asyncpg.Connection, schema: str) -> None:
    for spec in _REEMBED_WORKLIST_INDEXES:
        await conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {_quote_ident(spec.name)}
            ON {_fq(schema, spec.table)} (bank_id, id)
            WHERE embedding IS NOT NULL AND {SHADOW_COLUMN} IS NULL
            """
        )


async def _drop_reembed_worklist_indexes(conn: asyncpg.Connection, schema: str) -> None:
    for spec in _REEMBED_WORKLIST_INDEXES:
        await conn.execute(f"DROP INDEX IF EXISTS {_quote_ident(schema)}.{_quote_ident(spec.name)}")


async def discover_hindsight_schemas(
    conn: asyncpg.Connection,
    base_schema: str = DEFAULT_DATABASE_SCHEMA,
) -> list[str]:
    """Discover initialized Hindsight schemas from the database catalog."""
    rows = await conn.fetch(
        """
        SELECT n.nspname AS schema_name
        FROM pg_namespace n
        WHERE n.nspname NOT LIKE 'pg_%'
          AND n.nspname <> 'information_schema'
          AND EXISTS (
              SELECT 1 FROM information_schema.tables t
              WHERE t.table_schema = n.nspname AND t.table_name = 'alembic_version'
          )
          AND EXISTS (
              SELECT 1 FROM information_schema.tables t
              WHERE t.table_schema = n.nspname AND t.table_name IN ('banks', 'memory_units')
          )
        ORDER BY n.nspname
        """
    )
    discovered_schemas = [row["schema_name"] for row in rows]
    base = base_schema or DEFAULT_DATABASE_SCHEMA
    return list(dict.fromkeys([base, *discovered_schemas] if base in discovered_schemas else discovered_schemas))


async def inspect_reembed_state(conn: asyncpg.Connection, schema: str) -> ReembedState:
    migrations_exists = await _table_exists(conn, schema, REEMBED_MIGRATIONS_TABLE)
    active = None
    if migrations_exists:
        active = await _fetch_active_migration(conn, schema)

    shadow_column_rows = await conn.fetch(
        """
        SELECT table_name
        FROM information_schema.columns
        WHERE table_schema = $1
          AND table_name IN ('memory_units', 'mental_models')
          AND column_name = $2
        ORDER BY table_name
        """,
        schema,
        SHADOW_COLUMN,
    )
    shadow_index_rows = await conn.fetch(
        """
        SELECT indexname
        FROM pg_indexes
        WHERE schemaname = $1
          AND tablename IN ('memory_units', 'mental_models')
          AND indexdef ILIKE '%' || $2 || '%'
        ORDER BY indexname
        """,
        schema,
        SHADOW_COLUMN,
    )
    staging_rows = 0
    if await _table_exists(conn, schema, REEMBED_SEMANTIC_LINKS_TABLE):
        staging_rows = int(await conn.fetchval(f"SELECT COUNT(*) FROM {_fq(schema, REEMBED_SEMANTIC_LINKS_TABLE)}"))
    return ReembedState(
        active_migration_id=str(active["migration_id"]) if active else None,
        active_status=active["status"] if active else None,
        embedding_state=active["embedding_state"] if active else None,
        shadow_indexes_state=active["shadow_indexes_state"] if active else None,
        semantic_links_state=active["semantic_links_state"] if active else None,
        shadow_columns=[row["table_name"] for row in shadow_column_rows],
        shadow_indexes=[row["indexname"] for row in shadow_index_rows],
        staging_rows=staging_rows,
    )


async def ensure_no_active_reembed_state(conn: asyncpg.Connection, schema: str, operation: str) -> None:
    state = await inspect_reembed_state(conn, schema)
    if state.has_active_migration or state.has_shadow_state:
        details = {
            "active_migration_id": state.active_migration_id,
            "active_status": state.active_status,
            "embedding_state": state.embedding_state,
            "shadow_indexes_state": state.shadow_indexes_state,
            "semantic_links_state": state.semantic_links_state,
            "shadow_columns": state.shadow_columns,
            "shadow_indexes": state.shadow_indexes,
            "staging_rows": state.staging_rows,
        }
        raise RuntimeError(
            f"Cannot {operation} while reembed migration state exists in schema '{schema}': "
            f"{json.dumps(details, default=str)}. Complete or abandon the reembed first."
        )


async def _validate_schema_for_reembed(conn: asyncpg.Connection, schema: str) -> None:
    required_tables = (
        "banks",
        "memory_units",
        "mental_models",
        "entities",
        "unit_entities",
        "memory_links",
        REEMBED_MIGRATIONS_TABLE,
        REEMBED_SEMANTIC_LINKS_TABLE,
    )
    existing_tables = await _existing_tables(conn, schema, required_tables)
    missing = [table for table in required_tables if table not in existing_tables]
    if missing:
        raise RuntimeError(
            f"Schema '{schema}' is missing required table(s): {', '.join(missing)}. "
            "Run the reembed admin-table/Alembic migration path before reembed."
        )

    orphan_bank_ids = await conn.fetch(
        f"""
        WITH referenced AS (
            SELECT DISTINCT bank_id FROM {_fq(schema, "memory_units")}
            UNION
            SELECT DISTINCT bank_id FROM {_fq(schema, "mental_models")}
        )
        SELECT r.bank_id
        FROM referenced r
        LEFT JOIN {_fq(schema, "banks")} b ON b.bank_id = r.bank_id
        WHERE b.bank_id IS NULL
        ORDER BY r.bank_id
        LIMIT 20
        """
    )
    if orphan_bank_ids:
        ids = ", ".join(row["bank_id"] for row in orphan_bank_ids)
        raise RuntimeError(
            f"Schema '{schema}' has memory rows whose bank_id is absent from banks: {ids}. "
            "Fix orphan bank rows before reembedding."
        )


async def _progress_counts(conn: asyncpg.Connection, schema: str) -> ReembedProgressCounts:
    memory_total = int(
        await conn.fetchval(f"SELECT COUNT(*) FROM {_fq(schema, 'memory_units')} WHERE embedding IS NOT NULL")
    )
    if await _column_exists(conn, schema, "memory_units", SHADOW_COLUMN):
        memory_remaining = int(
            await conn.fetchval(
                f"""
                SELECT COUNT(*)
                FROM {_fq(schema, "memory_units")}
                WHERE embedding IS NOT NULL AND {SHADOW_COLUMN} IS NULL
                """
            )
        )
    else:
        memory_remaining = 0
    mental_total = int(
        await conn.fetchval(f"SELECT COUNT(*) FROM {_fq(schema, 'mental_models')} WHERE embedding IS NOT NULL")
    )
    if await _column_exists(conn, schema, "mental_models", SHADOW_COLUMN):
        mental_remaining = int(
            await conn.fetchval(
                f"""
                SELECT COUNT(*)
                FROM {_fq(schema, "mental_models")}
                WHERE embedding IS NOT NULL AND {SHADOW_COLUMN} IS NULL
                """
            )
        )
    else:
        mental_remaining = 0
    return ReembedProgressCounts(
        memory_units_total=memory_total,
        memory_units_done=memory_total - memory_remaining,
        mental_models_total=mental_total,
        mental_models_done=mental_total - mental_remaining,
    )


async def _with_retries(coro_factory, *, max_retries: int):
    delay = 1.0
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            if attempt >= max_retries:
                raise
            logger.warning(
                "Reembed embedding batch failed on attempt %s/%s; retrying in %.1fs: %s",
                attempt + 1,
                max_retries + 1,
                delay,
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 10.0)
    raise AssertionError("unreachable")


async def _initialize_or_resume_migration(
    conn: asyncpg.Connection,
    schema: str,
    config: HindsightConfig,
    dimension: int,
) -> asyncpg.Record:
    identity = _model_identity(config, dimension)
    fingerprint = _config_fingerprint(identity)

    async with conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", _reembed_lock_key(schema))

        row = await _fetch_active_migration(conn, schema)

        if row:
            if row["config_fingerprint"] != fingerprint:
                raise RuntimeError(
                    "Refusing to resume reembed with a different embedding configuration. "
                    f"Existing fingerprint={row['config_fingerprint']}, current fingerprint={fingerprint}."
                )
            if row["status"] == "prepared":
                return row
            if row["status"] not in ACTIVE_REEMBED_STATUSES:
                raise RuntimeError(f"Cannot resume migration {row['migration_id']} with status {row['status']}")
            return row

        state = await inspect_reembed_state(conn, schema)
        if state.has_shadow_state:
            raise RuntimeError(
                f"Schema '{schema}' has orphan reembed shadow state. "
                "Run reembed-abandon --orphan-shadow-state before starting a new migration."
            )

        new_id = uuid.uuid4()
        await conn.execute(f"ALTER TABLE {_fq(schema, 'memory_units')} ADD COLUMN {SHADOW_COLUMN} vector({dimension})")
        await conn.execute(f"ALTER TABLE {_fq(schema, 'mental_models')} ADD COLUMN {SHADOW_COLUMN} vector({dimension})")
        await conn.execute(
            f"""
            INSERT INTO {_fq(schema, REEMBED_MIGRATIONS_TABLE)}
            (migration_id, schema_name, provider, model, dimension, vector_extension,
             config_fingerprint, model_identity, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, 'running')
            """,
            new_id,
            schema,
            config.embeddings_provider,
            _model_name(config),
            dimension,
            config.vector_extension,
            fingerprint,
            json.dumps(identity, sort_keys=True, default=str),
        )
        return await conn.fetchrow(
            f"SELECT * FROM {_fq(schema, REEMBED_MIGRATIONS_TABLE)} WHERE migration_id = $1",
            new_id,
        )


async def _reembed_memory_units(conn: asyncpg.Connection, schema: str, embeddings, options: ReembedOptions) -> int:
    rows = await conn.fetch(
        f"""
        SELECT mu.id, mu.text, mu.fact_type, mu.occurred_start, mu.occurred_end, mu.mentioned_at,
               COALESCE(cardinality(mu.source_memory_ids), 0) AS source_count,
               ent.entities
        FROM {_fq(schema, "memory_units")} mu
        LEFT JOIN LATERAL (
            SELECT array_agg(e.canonical_name ORDER BY lower(e.canonical_name), e.id) AS entities
            FROM {_fq(schema, "unit_entities")} ue
            JOIN {_fq(schema, "entities")} e ON e.id = ue.entity_id
            WHERE ue.unit_id = mu.id
        ) ent ON TRUE
        WHERE mu.embedding IS NOT NULL AND mu.{SHADOW_COLUMN} IS NULL
        ORDER BY mu.bank_id, mu.id
        LIMIT $1
        """,
        options.batch_size,
    )
    if not rows:
        return 0

    texts = [_memory_unit_embedding_text(row) for row in rows]
    vectors = await _with_retries(
        lambda: embedding_utils.generate_embeddings_batch(embeddings, texts),
        max_retries=options.max_retries,
    )
    await conn.executemany(
        f"UPDATE {_fq(schema, 'memory_units')} SET {SHADOW_COLUMN} = $2::vector WHERE id = $1",
        [(row["id"], _format_embedding(vector)) for row, vector in zip(rows, vectors, strict=True)],
    )
    return len(rows)


async def _reembed_mental_models(conn: asyncpg.Connection, schema: str, embeddings, options: ReembedOptions) -> int:
    rows = await conn.fetch(
        f"""
        SELECT id, bank_id, name, content
        FROM {_fq(schema, "mental_models")}
        WHERE embedding IS NOT NULL AND {SHADOW_COLUMN} IS NULL
        ORDER BY bank_id, id
        LIMIT $1
        """,
        options.batch_size,
    )
    if not rows:
        return 0

    texts = [embedding_processing.build_mental_model_embedding_text(row["name"], row["content"]) for row in rows]
    vectors = await _with_retries(
        lambda: embedding_utils.generate_embeddings_batch(embeddings, texts),
        max_retries=options.max_retries,
    )
    await conn.executemany(
        f"""
        UPDATE {_fq(schema, "mental_models")}
        SET {SHADOW_COLUMN} = $3::vector
        WHERE id = $1 AND bank_id = $2
        """,
        [(row["id"], row["bank_id"], _format_embedding(vector)) for row, vector in zip(rows, vectors, strict=True)],
    )
    return len(rows)


async def _resolve_vector_extension(conn: asyncpg.Connection, vector_extension: str) -> str:
    rows = await conn.fetch("SELECT extname FROM pg_extension")
    return resolve_vector_extension_from_installed(vector_extension, {row["extname"] for row in rows})


async def _shadow_embedding_counts(conn: asyncpg.Connection, schema: str) -> ShadowEmbeddingCounts:
    row = await conn.fetchrow(
        f"""
        SELECT
            (SELECT COUNT(*) FROM {_fq(schema, "memory_units")} WHERE {SHADOW_COLUMN} IS NOT NULL) AS memory_units,
            (SELECT COUNT(*) FROM {_fq(schema, "mental_models")} WHERE {SHADOW_COLUMN} IS NOT NULL) AS mental_models
        """
    )
    return ShadowEmbeddingCounts(
        memory_units=int(row["memory_units"]),
        mental_models=int(row["mental_models"]),
    )


async def _create_shadow_indexes(
    conn: asyncpg.Connection,
    schema: str,
    resolved_extension: str,
    migration_id: uuid.UUID,
    index_max_parallel_maintenance_workers: int | None = None,
) -> str:
    embedding_counts = await _shadow_embedding_counts(conn, schema)
    deferred_tables = {
        table_name: row_count
        for table_name, row_count in (
            ("memory_units", embedding_counts.memory_units),
            ("mental_models", embedding_counts.mental_models),
        )
        if should_defer_index_creation(resolved_extension, row_count)
    }
    if deferred_tables:
        minimum_rows = minimum_rows_for_index(resolved_extension)
        table_counts = ", ".join(
            f"{table_name}={row_count}" for table_name, row_count in sorted(deferred_tables.items())
        )
        await conn.execute(
            f"""
            UPDATE {_fq(schema, REEMBED_MIGRATIONS_TABLE)}
            SET shadow_indexes_state = 'deferred', updated_at = now()
            WHERE migration_id = $1
            """,
            migration_id,
        )
        return (
            f"deferred: {resolved_extension} index creation requires at least {minimum_rows} rows "
            f"per indexed table; found {table_counts}"
        )

    index_type = index_type_keyword(resolved_extension)
    clause = index_using_clause(resolved_extension, column=SHADOW_COLUMN)
    override_parallel_workers = index_max_parallel_maintenance_workers is not None
    if override_parallel_workers:
        await conn.execute(
            "SELECT set_config('max_parallel_maintenance_workers', $1, false)",
            str(index_max_parallel_maintenance_workers),
        )

    try:
        if uses_per_bank_vector_indexes(resolved_extension):
            banks = await conn.fetch(f"SELECT bank_id, internal_id FROM {_fq(schema, 'banks')} ORDER BY bank_id")
            for bank in banks:
                uid = str(bank["internal_id"]).replace("-", "")[:16]
                escaped_bank_id = str(bank["bank_id"]).replace("'", "''")
                for fact_type, suffix in _MEMORY_UNIT_FACT_TYPES.items():
                    index_name = f"idx_mu_reemb_{suffix}_{uid}"
                    await conn.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {_quote_ident(index_name)}
                        ON {_fq(schema, "memory_units")}
                        {clause}
                        WHERE {SHADOW_COLUMN} IS NOT NULL
                          AND fact_type = '{fact_type}'
                          AND bank_id = '{escaped_bank_id}'
                        """
                    )
        else:
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_memory_units_embedding_reembed_{index_type}
                ON {_fq(schema, "memory_units")}
                {clause}
                """
            )

        await conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_mental_models_embedding_reembed_{index_type}
            ON {_fq(schema, "mental_models")}
            {clause}
            """
        )
    finally:
        if override_parallel_workers:
            await conn.execute("RESET max_parallel_maintenance_workers")

    await conn.execute(
        f"""
        UPDATE {_fq(schema, REEMBED_MIGRATIONS_TABLE)}
        SET shadow_indexes_state = 'ready', updated_at = now()
        WHERE migration_id = $1
        """,
        migration_id,
    )
    return "ready"


async def _stage_semantic_links(
    conn: asyncpg.Connection,
    schema: str,
    migration_id: uuid.UUID,
    resolved_extension: str,
    progress_callback: ProgressCallback | None = None,
) -> int:
    await conn.execute(
        f"DELETE FROM {_fq(schema, REEMBED_SEMANTIC_LINKS_TABLE)} WHERE migration_id = $1",
        migration_id,
    )
    await conn.execute(
        f"""
        UPDATE {_fq(schema, REEMBED_MIGRATIONS_TABLE)}
        SET semantic_links_state = 'pending', updated_at = now()
        WHERE migration_id = $1
        """,
        migration_id,
    )

    groups = await conn.fetch(
        f"""
        SELECT bank_id, fact_type
        FROM {_fq(schema, "memory_units")}
        WHERE {SHADOW_COLUMN} IS NOT NULL
          AND {_semantic_seed_filter("")}
        GROUP BY bank_id, fact_type
        ORDER BY bank_id, fact_type
        """
    )

    staged = 0
    logger.info(
        "Staging reembed semantic links for schema=%s migration_id=%s groups=%s",
        schema,
        migration_id,
        len(groups),
    )
    _emit_progress(
        progress_callback,
        ReembedProgress(
            schema=schema,
            phase="staging-semantic-links",
            migration_id=str(migration_id),
            semantic_links_staged=0,
            group_total=len(groups),
            message="staging semantic links",
        ),
    )
    for group_index, group in enumerate(groups, start=1):
        async with conn.transaction():
            for guc, value in ann_search_tuning_settings(resolved_extension, kind="low_latency"):
                await conn.execute(f"SET LOCAL {guc} = {value}")

            group_staged = await conn.fetchval(
                f"""
                WITH inserted AS (
                    INSERT INTO {_fq(schema, REEMBED_SEMANTIC_LINKS_TABLE)}
                    (migration_id, bank_id, from_unit_id, to_unit_id, weight)
                    SELECT $1, seed.bank_id, seed.id, candidate.id, candidate.similarity
                    FROM {_fq(schema, "memory_units")} seed
                    JOIN LATERAL (
                        SELECT mu.id,
                               GREATEST(0.0, LEAST(1.0, 1.0 - (seed.{SHADOW_COLUMN} <=> mu.{SHADOW_COLUMN}))) AS similarity
                        FROM {_fq(schema, "memory_units")} mu
                        WHERE mu.bank_id = $2
                          AND mu.fact_type = $3
                          AND mu.{SHADOW_COLUMN} IS NOT NULL
                          AND mu.id <> seed.id
                        ORDER BY seed.{SHADOW_COLUMN} <=> mu.{SHADOW_COLUMN}
                        LIMIT {STREAMING_SEMANTIC_LINK_TOP_K}
                    ) candidate ON candidate.similarity >= {SEMANTIC_LINK_THRESHOLD}
                    WHERE seed.{SHADOW_COLUMN} IS NOT NULL
                      AND seed.bank_id = $2
                      AND seed.fact_type = $3
                      AND {_semantic_seed_filter("seed")}
                    ON CONFLICT DO NOTHING
                    RETURNING 1
                )
                SELECT COUNT(*)::int FROM inserted
                """,
                migration_id,
                group["bank_id"],
                group["fact_type"],
            )
            staged += int(group_staged or 0)
        _emit_progress(
            progress_callback,
            ReembedProgress(
                schema=schema,
                phase="staging-semantic-links",
                migration_id=str(migration_id),
                semantic_links_staged=staged,
                group_index=group_index,
                group_total=len(groups),
                message=f"bank_id={group['bank_id']} fact_type={group['fact_type']}",
            ),
        )
        logger.info(
            "Staged %s reembed semantic links for schema=%s migration_id=%s group=%s/%s bank_id=%s fact_type=%s",
            int(group_staged or 0),
            schema,
            migration_id,
            group_index,
            len(groups),
            group["bank_id"],
            group["fact_type"],
        )

    await conn.execute(
        f"""
        UPDATE {_fq(schema, REEMBED_MIGRATIONS_TABLE)}
        SET semantic_links_state = 'complete', updated_at = now()
        WHERE migration_id = $1
        """,
        migration_id,
    )
    return staged


async def _mark_prepared_if_ready(conn: asyncpg.Connection, schema: str, migration_id: uuid.UUID) -> bool:
    row = await conn.fetchrow(
        f"""
        SELECT embedding_state, shadow_indexes_state, semantic_links_state
        FROM {_fq(schema, REEMBED_MIGRATIONS_TABLE)}
        WHERE migration_id = $1
        """,
        migration_id,
    )
    if (
        row
        and row["embedding_state"] == "complete"
        and row["shadow_indexes_state"] in {"ready", "deferred"}
        and row["semantic_links_state"] == "complete"
    ):
        await conn.execute(
            f"""
            UPDATE {_fq(schema, REEMBED_MIGRATIONS_TABLE)}
            SET status = 'prepared', updated_at = now()
            WHERE migration_id = $1
            """,
            migration_id,
        )
        return True
    return False


async def _rename_shadow_indexes(conn: asyncpg.Connection, schema: str, resolved_extension: str) -> None:
    index_type = index_type_keyword(resolved_extension)
    if uses_per_bank_vector_indexes(resolved_extension):
        banks = await conn.fetch(f"SELECT internal_id FROM {_fq(schema, 'banks')} ORDER BY bank_id")
        for bank in banks:
            uid = str(bank["internal_id"]).replace("-", "")[:16]
            for suffix in _MEMORY_UNIT_FACT_TYPES.values():
                await conn.execute(
                    f"DROP INDEX IF EXISTS {_quote_ident(schema)}.{_quote_ident(f'idx_mu_emb_{suffix}_{uid}')}"
                )
                await conn.execute(
                    f"""
                    ALTER INDEX IF EXISTS {_quote_ident(schema)}.{_quote_ident(f"idx_mu_reemb_{suffix}_{uid}")}
                    RENAME TO {_quote_ident(f"idx_mu_emb_{suffix}_{uid}")}
                    """
                )
    else:
        await conn.execute(f"DROP INDEX IF EXISTS {_quote_ident(schema)}.idx_memory_units_embedding")
        await conn.execute(
            f"""
            ALTER INDEX IF EXISTS {_quote_ident(schema)}.{_quote_ident(f"idx_memory_units_embedding_reembed_{index_type}")}
            RENAME TO idx_memory_units_embedding
            """
        )

    await conn.execute(f"DROP INDEX IF EXISTS {_quote_ident(schema)}.idx_mental_models_embedding")
    await conn.execute(
        f"""
        ALTER INDEX IF EXISTS {_quote_ident(schema)}.{_quote_ident(f"idx_mental_models_embedding_reembed_{index_type}")}
        RENAME TO idx_mental_models_embedding
        """
    )


async def _cutover(conn: asyncpg.Connection, schema: str, migration_id: uuid.UUID, resolved_extension: str) -> None:
    row = await _fetch_migration(conn, schema, migration_id)
    if row is None or row["status"] != "prepared":
        raise RuntimeError(f"Migration {migration_id} is not prepared for cutover")

    async with conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", _reembed_lock_key(schema))
        await _drop_reembed_worklist_indexes(conn, schema)
        if row["shadow_indexes_state"] == "ready":
            await _rename_shadow_indexes(conn, schema, resolved_extension)
        else:
            await conn.execute(f"DROP INDEX IF EXISTS {_quote_ident(schema)}.idx_memory_units_embedding")
            await conn.execute(f"DROP INDEX IF EXISTS {_quote_ident(schema)}.idx_mental_models_embedding")

        await conn.execute(
            f"""
            DELETE FROM {_fq(schema, "memory_links")} ml
            USING {_fq(schema, "memory_units")} mu
            WHERE ml.from_unit_id = mu.id
              AND ml.link_type = 'semantic'
              AND mu.{SHADOW_COLUMN} IS NOT NULL
              AND {_semantic_seed_filter("mu")}
            """
        )
        await conn.execute(
            f"""
            INSERT INTO {_fq(schema, "memory_links")}
            (from_unit_id, to_unit_id, link_type, entity_id, weight, bank_id)
            SELECT from_unit_id, to_unit_id, 'semantic', NULL, weight, bank_id
            FROM {_fq(schema, REEMBED_SEMANTIC_LINKS_TABLE)}
            WHERE migration_id = $1
            ON CONFLICT DO NOTHING
            """,
            migration_id,
        )
        await conn.execute(f"ALTER TABLE {_fq(schema, 'memory_units')} DROP COLUMN embedding")
        await conn.execute(f"ALTER TABLE {_fq(schema, 'memory_units')} RENAME COLUMN {SHADOW_COLUMN} TO embedding")
        await conn.execute(f"ALTER TABLE {_fq(schema, 'mental_models')} DROP COLUMN embedding")
        await conn.execute(f"ALTER TABLE {_fq(schema, 'mental_models')} RENAME COLUMN {SHADOW_COLUMN} TO embedding")
        await conn.execute(
            f"DELETE FROM {_fq(schema, REEMBED_SEMANTIC_LINKS_TABLE)} WHERE migration_id = $1",
            migration_id,
        )
        await conn.execute(
            f"""
            UPDATE {_fq(schema, REEMBED_MIGRATIONS_TABLE)}
            SET status = 'completed', updated_at = now()
            WHERE migration_id = $1
            """,
            migration_id,
        )


async def reembed_schema(
    database_url: str,
    schema: str,
    config: HindsightConfig,
    options: ReembedOptions,
    progress_callback: ProgressCallback | None = None,
) -> ReembedReport:
    conn = await asyncpg.connect(database_url)
    lock_acquired = False
    try:
        if not options.dry_run:
            await _try_acquire_reembed_lock(conn, schema)
            lock_acquired = True

        _emit_progress(
            progress_callback,
            ReembedProgress(schema=schema, phase="validating", message="validating schema"),
        )
        if options.dry_run:
            migrations_exists = await _table_exists(conn, schema, REEMBED_MIGRATIONS_TABLE)
            if not migrations_exists:
                return ReembedReport(
                    schema=schema,
                    migration_id=None,
                    status="dry-run",
                    memory_units_total=0,
                    memory_units_done=0,
                    mental_models_total=0,
                    mental_models_done=0,
                    semantic_links_staged=0,
                    shadow_indexes_state=None,
                    message="admin tables are missing; dry-run did not create them",
                )

        await _validate_schema_for_reembed(conn, schema)

        clear_config_cache()
        embeddings = create_embeddings_from_env()
        await embeddings.initialize()
        dimension = embeddings.dimension
        resolved_vector_extension = await _resolve_vector_extension(conn, config.vector_extension)
        validate_vector_index_dimension(resolved_vector_extension, dimension)
        _emit_progress(
            progress_callback,
            ReembedProgress(
                schema=schema,
                phase="initialized",
                message=f"dimension={dimension} vector_extension={resolved_vector_extension}",
            ),
        )

        if options.dry_run:
            state = await inspect_reembed_state(conn, schema)
            if await _column_exists(conn, schema, "memory_units", SHADOW_COLUMN):
                counts = await _progress_counts(conn, schema)
            else:
                counts = ReembedProgressCounts(
                    memory_units_total=int(
                        await conn.fetchval(
                            f"SELECT COUNT(*) FROM {_fq(schema, 'memory_units')} WHERE embedding IS NOT NULL"
                        )
                    ),
                    memory_units_done=0,
                    mental_models_total=int(
                        await conn.fetchval(
                            f"SELECT COUNT(*) FROM {_fq(schema, 'mental_models')} WHERE embedding IS NOT NULL"
                        )
                    ),
                    mental_models_done=0,
                )
            return ReembedReport(
                schema=schema,
                migration_id=state.active_migration_id,
                status="dry-run",
                memory_units_total=counts.memory_units_total,
                memory_units_done=counts.memory_units_done,
                mental_models_total=counts.mental_models_total,
                mental_models_done=counts.mental_models_done,
                semantic_links_staged=state.staging_rows,
                shadow_indexes_state=state.shadow_indexes_state,
                message=f"would use embedding dimension={dimension}",
            )

        migration = await _initialize_or_resume_migration(conn, schema, config, dimension)
        migration_id = migration["migration_id"]
        _emit_progress(
            progress_callback,
            ReembedProgress(
                schema=schema,
                phase="migration-ready",
                migration_id=str(migration_id),
                message=f"status={migration['status']}",
            ),
        )

        if migration["status"] != "prepared":
            await _ensure_reembed_worklist_indexes(conn, schema)
            counts = await _progress_counts(conn, schema)
            _emit_progress(
                progress_callback,
                ReembedProgress(
                    schema=schema,
                    phase="embedding",
                    migration_id=str(migration_id),
                    memory_units_done=counts.memory_units_done,
                    memory_units_total=counts.memory_units_total,
                    mental_models_done=counts.mental_models_done,
                    mental_models_total=counts.mental_models_total,
                    message="embedding rows",
                ),
            )
            while True:
                memory_processed = await _reembed_memory_units(conn, schema, embeddings, options)
                mental_processed = await _reembed_mental_models(conn, schema, embeddings, options)
                processed = memory_processed + mental_processed
                if processed == 0:
                    break
                counts.memory_units_done += memory_processed
                counts.mental_models_done += mental_processed
                _emit_progress(
                    progress_callback,
                    ReembedProgress(
                        schema=schema,
                        phase="embedding",
                        migration_id=str(migration_id),
                        memory_units_done=counts.memory_units_done,
                        memory_units_total=counts.memory_units_total,
                        mental_models_done=counts.mental_models_done,
                        mental_models_total=counts.mental_models_total,
                        message=f"processed_batch={processed}",
                    ),
                )

            counts = await _progress_counts(conn, schema)
            if (
                counts.memory_units_done == counts.memory_units_total
                and counts.mental_models_done == counts.mental_models_total
            ):
                await conn.execute(
                    f"""
                    UPDATE {_fq(schema, REEMBED_MIGRATIONS_TABLE)}
                    SET embedding_state = 'complete', updated_at = now()
                    WHERE migration_id = $1
                    """,
                    migration_id,
                )
                _emit_progress(
                    progress_callback,
                    ReembedProgress(
                        schema=schema,
                        phase="building-shadow-indexes",
                        migration_id=str(migration_id),
                        message="building shadow indexes",
                    ),
                )
                index_message = await _create_shadow_indexes(
                    conn,
                    schema,
                    resolved_vector_extension,
                    migration_id,
                    options.index_max_parallel_maintenance_workers,
                )
                _emit_progress(
                    progress_callback,
                    ReembedProgress(
                        schema=schema,
                        phase="shadow-indexes-ready",
                        migration_id=str(migration_id),
                        message=index_message,
                    ),
                )
                semantic_count = await _stage_semantic_links(
                    conn,
                    schema,
                    migration_id,
                    resolved_vector_extension,
                    progress_callback,
                )
                await _mark_prepared_if_ready(conn, schema, migration_id)
            else:
                index_message = "pending"
                semantic_count = 0
        else:
            counts = await _progress_counts(conn, schema)
            index_message = migration["shadow_indexes_state"]
            semantic_count = int(
                await conn.fetchval(
                    f"SELECT COUNT(*) FROM {_fq(schema, REEMBED_SEMANTIC_LINKS_TABLE)} WHERE migration_id = $1",
                    migration_id,
                )
            )

        prepared = await _fetch_migration(conn, schema, migration_id)
        if prepared and prepared["status"] == "prepared":
            _emit_progress(
                progress_callback,
                ReembedProgress(
                    schema=schema,
                    phase="cutover",
                    migration_id=str(migration_id),
                    message="cutting over",
                ),
            )
            await _cutover(conn, schema, migration_id, resolved_vector_extension)

        final = await _fetch_migration(conn, schema, migration_id)
        final_counts = await _progress_counts(conn, schema)
        _emit_progress(
            progress_callback,
            ReembedProgress(
                schema=schema,
                phase="completed" if final and final["status"] == "completed" else "finished",
                migration_id=str(migration_id),
                memory_units_done=final_counts.memory_units_done,
                memory_units_total=final_counts.memory_units_total,
                mental_models_done=final_counts.mental_models_done,
                mental_models_total=final_counts.mental_models_total,
                semantic_links_staged=semantic_count,
                message=f"status={final['status'] if final else 'unknown'}",
            ),
        )
        return ReembedReport(
            schema=schema,
            migration_id=str(migration_id),
            status=final["status"] if final else "unknown",
            memory_units_total=final_counts.memory_units_total,
            memory_units_done=final_counts.memory_units_done,
            mental_models_total=final_counts.mental_models_total,
            mental_models_done=final_counts.mental_models_done,
            semantic_links_staged=semantic_count,
            shadow_indexes_state=final["shadow_indexes_state"] if final else None,
            message=index_message,
        )
    except _ReembedLockUnavailable:
        raise
    except Exception as exc:
        if await _table_exists(conn, schema, REEMBED_MIGRATIONS_TABLE):
            active = await _fetch_active_migration(conn, schema)
            if active:
                await conn.execute(
                    f"""
                    UPDATE {_fq(schema, REEMBED_MIGRATIONS_TABLE)}
                    SET status = 'failed', error_message = $2, updated_at = now()
                    WHERE migration_id = $1 AND status <> 'completed'
                    """,
                    active["migration_id"],
                    str(exc),
                )
        raise
    finally:
        if lock_acquired:
            try:
                await _release_reembed_lock(conn, schema)
            except Exception:
                logger.warning("Failed to release reembed advisory lock for schema=%s", schema, exc_info=True)
        await conn.close()


async def reembed_status(database_url: str, schema: str) -> ReembedStatusReport:
    conn = await asyncpg.connect(database_url)
    try:
        if not await _table_exists(conn, schema, REEMBED_MIGRATIONS_TABLE):
            return ReembedStatusReport(
                schema=schema,
                status="not-initialized",
                migrations=[],
                state=None,
            )
        rows = await conn.fetch(
            f"""
            SELECT migration_id, status, embedding_state, shadow_indexes_state,
                   semantic_links_state, started_at, updated_at, error_message
            FROM {_fq(schema, REEMBED_MIGRATIONS_TABLE)}
            ORDER BY updated_at DESC
            LIMIT 5
            """
        )
        state = await inspect_reembed_state(conn, schema)
        return ReembedStatusReport(
            schema=schema,
            status="initialized",
            migrations=[
                ReembedStatusMigration(
                    migration_id=row["migration_id"],
                    status=row["status"],
                    embedding_state=row["embedding_state"],
                    shadow_indexes_state=row["shadow_indexes_state"],
                    semantic_links_state=row["semantic_links_state"],
                    started_at=row["started_at"],
                    updated_at=row["updated_at"],
                    error_message=row["error_message"],
                )
                for row in rows
            ],
            state=state,
        )
    finally:
        await conn.close()


async def abandon_reembed(
    database_url: str,
    schema: str,
    *,
    orphan_shadow_state: bool,
) -> ReembedReport:
    conn = await asyncpg.connect(database_url)
    try:
        if not await _table_exists(conn, schema, REEMBED_MIGRATIONS_TABLE):
            if not orphan_shadow_state:
                raise RuntimeError(f"Schema '{schema}' has no reembed admin table")
            target_id = None
        else:
            row = await _fetch_active_migration(conn, schema)
            target_id = row["migration_id"] if row else None
            if target_id is None and not orphan_shadow_state:
                raise RuntimeError(f"Schema '{schema}' has no active reembed migration")

        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", _reembed_lock_key(schema))
            state = await inspect_reembed_state(conn, schema)
            for index_name in state.shadow_indexes:
                await conn.execute(f"DROP INDEX IF EXISTS {_quote_ident(schema)}.{_quote_ident(index_name)}")
            for table in ("memory_units", "mental_models"):
                if await _column_exists(conn, schema, table, SHADOW_COLUMN):
                    await conn.execute(f"ALTER TABLE {_fq(schema, table)} DROP COLUMN {SHADOW_COLUMN}")
            if await _table_exists(conn, schema, REEMBED_SEMANTIC_LINKS_TABLE):
                if target_id is not None:
                    await conn.execute(
                        f"DELETE FROM {_fq(schema, REEMBED_SEMANTIC_LINKS_TABLE)} WHERE migration_id = $1",
                        target_id,
                    )
                elif orphan_shadow_state:
                    await conn.execute(f"DELETE FROM {_fq(schema, REEMBED_SEMANTIC_LINKS_TABLE)}")
            if target_id is not None:
                await conn.execute(
                    f"""
                    UPDATE {_fq(schema, REEMBED_MIGRATIONS_TABLE)}
                    SET status = 'abandoned', updated_at = now()
                    WHERE migration_id = $1 AND status = ANY($2::text[])
                    """,
                    target_id,
                    list(ACTIVE_REEMBED_STATUSES),
                )

        return ReembedReport(
            schema=schema,
            migration_id=str(target_id) if target_id else None,
            status="abandoned",
            memory_units_total=0,
            memory_units_done=0,
            mental_models_total=0,
            mental_models_done=0,
            semantic_links_staged=0,
            shadow_indexes_state=None,
            message="reembed shadow state removed",
        )
    finally:
        await conn.close()
