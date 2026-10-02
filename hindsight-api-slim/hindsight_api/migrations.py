"""
Database migration management using Alembic.

This module provides programmatic access to run database migrations
on application startup. It is designed to be safe for concurrent
execution using PostgreSQL advisory locks to coordinate between
distributed workers.

Supports multi-tenant schema isolation: migrations can target a specific
PostgreSQL schema, allowing each tenant to have isolated tables.

Important: All migrations must be backward-compatible to allow
safe rolling deployments.

No alembic.ini required - all configuration is done programmatically.
"""

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from alembic import command
from alembic.config import Config
from alembic.script.revision import ResolutionError
from alembic.util.exc import CommandError
from sqlalchemy import Connection, create_engine, text
from sqlalchemy.pool import NullPool

from ._pg_extensions import (
    create_extension,
    ensure_extensions_in_public,
    extension_schema,
    relocate_extension_to_public,
)
from ._pg_search import normalize_pg_search_tokenizer, pg_search_bm25_columns
from ._text_search import mental_models_text_document
from ._vector_index import (
    bootstrap_extension,
    configured_vector_extension,
    detect_vector_extension,
    index_type_keyword,
    index_using_clause,
    minimum_rows_for_index,
    should_defer_index_creation,
    uses_per_bank_vector_indexes,
)
from .config import ENV_MIGRATION_DATABASE_URL, ENV_MIGRATION_ISOLATION, get_config
from .db_url import is_oracle_url, to_libpq_url
from .utils import mask_network_location

logger = logging.getLogger(__name__)

#: Set in the migration child's env so it does not spawn a child of its own.
_CHILD_MARKER = "_HINDSIGHT_MIGRATION_CHILD"

# Advisory lock ID for migrations (arbitrary unique number)
MIGRATION_LOCK_ID = 123456789

# Alembic's command.upgrade() is NOT thread-safe: it uses module-level global
# proxies (context._proxy, script) that get overwritten when two threads call
# upgrade() concurrently.  This causes migrations to target the wrong schema
# and crash with "relation already exists" or KeyError: 'script'.
# Serialize all Alembic invocations with a process-level lock.
_alembic_lock = threading.Lock()


def _set_alembic_main_option(config: Config, name: str, value: str) -> None:
    """Set an Alembic option without treating URL percent escapes as interpolation."""
    config.set_main_option(name, value.replace("%", "%%"))


def _detect_vector_extension(conn, vector_extension: str = "pgvector") -> str:
    """Validate configured vector extension and preserve Azure DiskANN detection."""
    return detect_vector_extension(conn, vector_extension)


def _ensure_pgvector_extension_in_public(conn: Connection) -> None:
    """Ensure pgvector is installed in ``public`` before pgvector-backed migrations run."""
    logger.debug("Checking pgvector extension availability...")

    if extension_schema(conn, "vector") is None:
        logger.info("pgvector extension not found, attempting to install...")
        try:
            create_extension(conn, "vector")
            conn.commit()
            logger.info("pgvector extension installed in public schema")
        except Exception as e:
            # Installation failed - this is only fatal if the extension truly
            # doesn't exist; another process may have installed it meanwhile.
            conn.rollback()
            existing = extension_schema(conn, "vector")
            if not existing:
                logger.error(
                    f"pgvector extension is not installed and cannot be installed: {e}. "
                    f"Please ensure pgvector is installed by a database administrator. "
                    f"See: https://github.com/pgvector/pgvector#installation"
                )
                raise RuntimeError(
                    "pgvector extension is required but not installed. Please install it with: CREATE EXTENSION vector;"
                ) from e
            logger.warning(
                f"Could not install pgvector extension (permission denied?), "
                f"but extension exists in '{existing}' schema. Continuing..."
            )

    # Relocate an installation an older version (or an operator) put elsewhere.
    # ALTER EXTENSION ... SET SCHEMA carries its dependent objects along, unlike
    # the DROP ... CASCADE + CREATE this used to do, which took every embedding
    # column with it.
    relocate_extension_to_public(conn, "vector")


def _bootstrap_vector_extension_for_migrations(conn: Connection, vector_extension: str) -> None:
    """Bootstrap the configured vector backend before schema migrations run."""
    if vector_extension == "pgvector":
        _ensure_pgvector_extension_in_public(conn)
    bootstrap_extension(conn, vector_extension)
    # Repair anything an older version installed into a tenant schema, where the
    # runtime (which connects with the default search_path) cannot resolve it — the
    # pg_trgm case that made every retain fail silently in schema mode (#4118).
    ensure_extensions_in_public(conn)
    conn.commit()


def _vector_index_names(
    conn: Connection,
    schema_name: str,
    table_name: str,
    name_like: str | None = None,
    *,
    vector_access_methods_only: bool = True,
) -> list[str]:
    """Names of the vector indexes on ``table_name.embedding``, from the catalog.

    Deliberately NOT ``pg_indexes``: that view renders every row through
    ``pg_get_indexdef()``, which is evaluated for indexes outside the schema we
    asked about. When a concurrent session drops a schema mid-scan — pytest-xdist
    workers do exactly this — the render fails with "cache lookup failed for
    attribute N of relation OID" (an internal_error) and takes the whole
    statement with it. Resolving the relation first and reading ``pg_am`` keeps
    the scan inside one table's own indexes, so an unrelated schema going away
    cannot break it.
    """
    rows = conn.execute(
        text("""
            SELECT i.relname
            FROM pg_class t
            JOIN pg_namespace n ON n.oid = t.relnamespace
            JOIN pg_index x ON x.indrelid = t.oid
            JOIN pg_class i ON i.oid = x.indexrelid
            JOIN pg_am am ON am.oid = i.relam
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(x.indkey)
            WHERE n.nspname = :schema
              AND t.relname = :table
              AND a.attname = 'embedding'
              AND (NOT :vector_ams_only OR am.amname IN ('hnsw', 'vchordrq', 'diskann', 'scann'))
              AND (:name_like IS NULL OR i.relname LIKE :name_like)
        """),
        {
            "schema": schema_name,
            "table": table_name,
            "name_like": name_like,
            "vector_ams_only": vector_access_methods_only,
        },
    ).fetchall()
    return [row[0] for row in rows]


def _drop_index(conn: Connection, schema_name: str, index_name: str) -> None:
    """Drop one index. DDL identifiers cannot be bound parameters, so escape inline."""
    safe_schema = schema_name.replace('"', '""')
    safe_index = index_name.replace('"', '""')
    conn.execute(text(f'DROP INDEX IF EXISTS "{safe_schema}"."{safe_index}"'))


def _drop_per_bank_vector_indexes(conn: Connection, schema_name: str) -> None:
    """Drop per-bank partial memory_units vector indexes after global ScaNN is ready."""
    # Matched by name and column, NOT by access method: this sweep exists to clear
    # per-bank leftovers, and one whose access method drifted after a backend switch
    # (or an INVALID build from an interrupted CREATE INDEX CONCURRENTLY) is exactly
    # the kind that must go. The pg_indexes version this replaced did not filter on
    # the method either.
    for index_name in _vector_index_names(
        conn,
        schema_name,
        "memory_units",
        name_like="idx\\_mu\\_emb\\_%",
        vector_access_methods_only=False,
    ):
        _drop_index(conn, schema_name, index_name)


def _get_schema_lock_id(schema: str) -> int:
    """
    Generate a unique advisory lock ID for a schema.

    Uses hash of schema name to create a deterministic lock ID.
    """
    # Use hash to create a unique lock ID per schema
    # Keep within PostgreSQL's bigint range
    hash_bytes = hashlib.sha256(schema.encode()).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder="big") % (2**31)


#: How often to poll for the migration advisory lock.
_LOCK_POLL_INTERVAL_SECS = 0.5

#: How long a wait has to last before it is worth a log line, and how often to
#: repeat it afterwards.  Queuing briefly behind another worker is routine; only
#: a wait that outlasts this is a symptom.
_LOCK_WAIT_REPORT_AFTER_SECS = 5.0
_LOCK_WAIT_REPORT_INTERVAL_SECS = 30.0


def _advisory_lock_holder(conn: Connection, lock_id: int) -> str | None:
    """Describe who holds the migration advisory lock, or None if not found.

    Best effort: this runs while a migrator is stuck waiting, so any error
    here must not break the wait loop — an empty catalog row just means we
    could not identify the holder.  A failure leaves the transaction aborted,
    so the caller must end it before its next statement — the wait loop's
    ``conn.commit()`` does, which is why this runs before it.  A diagnostic
    must never turn a wait into a failed migration.

    ``pg_advisory_lock(bigint)`` stores the key as classid = high 32 bits,
    objid = low 32 bits, objsubid = 1; the two-argument form uses objsubid = 2.
    Matching objsubid = 1 therefore excludes a two-argument lock with a
    colliding objid, and classid = 0 (``lock_id`` is reduced mod 2**31, so its
    high word is always zero) excludes a 64-bit key that shares the low word.
    """
    try:
        row = conn.execute(
            text(
                "SELECT a.pid, coalesce(a.application_name, ''), coalesce(a.client_addr::text, ''), "
                "coalesce(a.query, '') "
                "FROM pg_locks l "
                "LEFT JOIN pg_stat_activity a ON a.pid = l.pid "
                "WHERE l.locktype = 'advisory' AND l.classid = 0 AND l.objid = :lock_id "
                "AND l.objsubid = 1 AND l.granted "
                "LIMIT 1"
            ),
            {"lock_id": lock_id},
        ).fetchone()
    except Exception as e:
        logger.debug("Could not inspect pg_locks for the migration advisory lock holder: %s", e)
        return None
    if not row:
        return None
    pid, app_name, client_addr, query = row[0], row[1] or "", row[2] or "", (row[3] or "").strip()
    parts = [f"pid={pid}"]
    if app_name:
        parts.append(f"app={app_name}")
    if client_addr:
        parts.append(f"client={client_addr}")
    if query:
        parts.append(f"query={query[:120]}")
    return ", ".join(parts)


def _run_migrations_internal(database_url: str, script_location: str, schema: str | None = None) -> None:
    """
    Internal function to run migrations without locking.

    Args:
        database_url: SQLAlchemy database URL
        script_location: Path to alembic scripts
        schema: Target schema (None for default/public)
    """
    schema_name = schema or "public"
    logger.info(f"Running database migrations to head for schema '{schema_name}'...")
    logger.info(f"Database URL: {mask_network_location(database_url)}")
    logger.info(f"Script location: {script_location}")

    # Create Alembic configuration programmatically (no alembic.ini needed)
    alembic_cfg = Config()

    # Set the script location (where alembic versions are stored)
    _set_alembic_main_option(alembic_cfg, "script_location", script_location)

    # Extension-owned revisions, applied on the same lifecycle as core's: same run,
    # same advisory lock, same `alembic_version` table.
    #
    # `version_locations` must include core's own directory explicitly. Alembic does
    # NOT fall back to `script_location/versions` once this is set, so omitting it
    # would silently reduce the run to the extensions' revisions — a migration that
    # appears to succeed while applying none of core's.
    #
    # Each extension tree is independent: its own base, its own head, its own row in
    # `alembic_version`. That is why `command.upgrade(..., "heads")` below is plural
    # and always was — it already applies every branch. Alembic gives no ordering
    # BETWEEN independent branches, so a revision needing core first declares
    # `depends_on`; see `Extension.alembic_version_locations`.
    # Imported here, not at module import: the loader imports extension modules,
    # and migrations.py is itself imported by paths that must not drag an
    # extension's dependency tree in with them.
    from .extensions.loader import collect_alembic_version_locations

    extension_locations = collect_alembic_version_locations()
    if extension_locations:
        core_versions = str(Path(script_location) / "versions")
        _set_alembic_main_option(
            alembic_cfg,
            "version_locations",
            os.pathsep.join([core_versions, *extension_locations]),
        )
        logger.info("Including %d extension migration location(s) alongside core", len(extension_locations))

    # Set the database URL
    _set_alembic_main_option(alembic_cfg, "sqlalchemy.url", database_url)

    # Configure logging (optional, but helps with debugging)
    # Uses Python's logging system instead of alembic.ini
    _set_alembic_main_option(alembic_cfg, "prepend_sys_path", ".")

    # Set path_separator to avoid deprecation warning
    _set_alembic_main_option(alembic_cfg, "path_separator", "os")

    # If targeting a specific schema, pass it to env.py via config
    # env.py will handle setting search_path and version_table_schema
    if schema:
        _set_alembic_main_option(alembic_cfg, "target_schema", schema)

    # Run migrations under a process-level lock.  Alembic uses module-level
    # global proxies that are not thread-safe, so concurrent command.upgrade()
    # calls from different threads corrupt each other's context.
    try:
        with _alembic_lock:
            command.upgrade(alembic_cfg, "heads")
    except (ResolutionError, CommandError) as e:
        # command.upgrade() wraps ResolutionError in CommandError via
        # ScriptDirectory._catch_revision_errors, so the wrapped form is what
        # actually reaches us; re-raise CommandErrors with any other cause.
        if isinstance(e, CommandError) and not isinstance(e.__cause__, ResolutionError):
            raise
        # This happens during rolling deployments when a newer version of the code
        # has already run migrations, and this older replica doesn't have the new
        # migration files. The database is already at a newer revision than we know.
        # This is safe to ignore - the newer code has already applied its migrations.
        logger.warning(
            f"Database is at a newer migration revision than this code version knows about. "
            f"This is expected during rolling deployments. Skipping migrations. Error: {e}"
        )
        return

    logger.info(f"Database migrations completed successfully for schema '{schema_name}'")


def _should_isolate_migrations() -> bool:
    """Whether to run the migration in a subprocess instead of in this process.

    Controlled by ``HINDSIGHT_API_MIGRATION_ISOLATION``:

        true    isolate — keeps alembic's import graph and its sync engine (psycopg2)
                out of a long-lived server process
        false   (default) never isolate; run in the calling process

    ``_CHILD_MARKER`` stops the child from recursing.
    """
    if os.environ.get(_CHILD_MARKER):
        return False
    return get_config().migration_isolation == "true"


def _run_in_migration_child(target: str, kwargs: dict) -> None:
    """Run the migration in a subprocess so this process never imports psycopg2.

    Alembic drives PostgreSQL through SQLAlchemy's sync engine, i.e. psycopg2, and a
    long-lived server has no other reason to carry that import graph and its thread
    pool for the rest of its life.

    The boundary is the whole migration entrypoint rather than each ``create_engine``
    call: schema migration also reaches ``ensure_embedding_dimension`` and the vector /
    text-search extension helpers, each of which opens its own sync engine. Isolating the
    entrypoint covers all of them in one child instead of one spawn apiece.

    The migration itself is short, rare and not on any hot path, so paying a process
    spawn for it is free.

    The payload goes over stdin, not argv: ``run_migrations_for_schemas`` is called
    with every tenant schema at once, and at the scale that entrypoint is documented
    for (20k schemas) the JSON is hundreds of KB — past ``ARG_MAX`` on macOS and close
    to it on Linux, which would fail as ``E2BIG`` only on the largest deployments.

    The child inherits stdout/stderr instead of having them captured. A full sweep can
    run for the best part of an hour; capturing would hold every line until it finished
    and show an operator nothing while it ran.
    """
    payload = json.dumps({"target": target, "kwargs": kwargs})
    env = {
        **os.environ,
        ENV_MIGRATION_ISOLATION: "false",
        _CHILD_MARKER: "1",
    }
    logger.info("Running migrations in a subprocess (see %s)", ENV_MIGRATION_ISOLATION)
    result = subprocess.run(
        [sys.executable, "-m", "hindsight_api.migrations"],
        input=payload,
        env=env,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Migration subprocess failed (exit {result.returncode}); see the child's output above.")


def run_migrations(
    database_url: str,
    script_location: str | None = None,
    schema: str | None = None,
    migration_database_url: str | None = None,
) -> None:
    """
    Run database migrations to the latest version using programmatic Alembic configuration.

    This function is safe to call from multiple distributed workers simultaneously:
    - Uses PostgreSQL advisory lock to ensure only one worker runs migrations at a time
    - Other workers wait for the lock, then verify migrations are complete
    - If schema is already up-to-date, this is a fast no-op

    Supports multi-tenant schema isolation: when a schema is specified, migrations
    run in that schema instead of public. This allows tenant extensions to provision
    new tenant schemas with their own isolated tables.

    Args:
        database_url: SQLAlchemy database URL (e.g., "postgresql://user:pass@host/db")
        script_location: Path to alembic migrations directory (e.g., "/path/to/alembic").
                        If None, defaults to hindsight-api/alembic directory.
        schema: Target PostgreSQL schema name. If None, uses default (public).
                When specified, creates the schema if needed and runs migrations there.

    Raises:
        RuntimeError: If migrations fail to complete
        FileNotFoundError: If script_location doesn't exist

    Example:
        # Using default location and public schema
        run_migrations("postgresql://user:pass@host/db")

        # Run migrations for a specific tenant schema
        run_migrations("postgresql://user:pass@host/db", schema="tenant_acme")

        # Using custom location (when importing from another project)
        run_migrations(
            "postgresql://user:pass@host/db",
            script_location="/path/to/copied/_alembic"
        )
    """
    # Prefer a dedicated migration URL that bypasses connection poolers (e.g.
    # PgBouncer in transaction mode).  Session-level advisory locks don't
    # survive a PgBouncer transaction-mode cycle, so the distributed lock is
    # ineffective when the app URL goes through a pooler.  Configure
    # HINDSIGHT_API_MIGRATION_DATABASE_URL to the direct PostgreSQL endpoint
    # (e.g. hindsight-pg-rw) to restore correct locking behaviour.
    # When isolation is on, keep psycopg2 out of this process entirely.
    # ``_CHILD_MARKER`` stops the child from recursing.
    if _should_isolate_migrations():
        _run_in_migration_child(
            "run_migrations",
            {
                "database_url": database_url,
                "script_location": script_location,
                "schema": schema,
                "migration_database_url": migration_database_url,
            },
        )
        return

    raw_url = migration_database_url or database_url
    # Oracle URLs are passed through to SQLAlchemy unchanged; only PG URLs
    # need the libpq normalization (asyncpg → psycopg2 driver, ssl → sslmode).
    migration_url = raw_url if is_oracle_url(raw_url) else to_libpq_url(raw_url)

    try:
        # Determine script location
        if script_location is None:
            # Default: use the alembic directory inside the hindsight_api package
            # This file is in: hindsight_api/migrations.py
            # Alembic is in: hindsight_api/alembic/
            package_dir = Path(__file__).parent
            script_location = str(package_dir / "alembic")

        script_path = Path(script_location)
        if not script_path.exists():
            raise FileNotFoundError(
                f"Alembic script location not found at {script_location}. Database migrations cannot be run."
            )

        # Oracle path: no advisory lock, no pgvector. DDL is autocommit on
        # Oracle, ``IF NOT EXISTS`` (Oracle 23ai) and 955 swallowing make
        # repeated runs from concurrent workers safe.
        if is_oracle_url(migration_url):
            _run_migrations_internal(migration_url, script_location, schema=schema)
            return

        # Use schema-specific lock ID for multi-tenant isolation
        lock_id = _get_schema_lock_id(schema) if schema else MIGRATION_LOCK_ID
        schema_name = schema or "public"

        # Use PostgreSQL advisory lock to coordinate between distributed workers.
        #
        # IMPORTANT: We must avoid holding an open transaction on the advisory-lock
        # connection while CREATE INDEX CONCURRENTLY runs inside a migration.
        # CONCURRENTLY waits for ALL active transactions to finish before the index
        # becomes valid.  If the advisory-lock connection (or any waiting worker's
        # connection) holds an open transaction, CONCURRENTLY deadlocks:
        #   - migration worker waits for other workers' transactions to close
        #   - other workers wait for the advisory lock to be released
        #
        # Fix:
        #   1. Use pg_try_advisory_lock (non-blocking) in a poll loop instead of
        #      blocking pg_advisory_lock, so we can COMMIT the transaction between
        #      retries.  Between retries the connection holds no open transaction.
        #   2. After acquiring the lock, COMMIT the transaction on the advisory-lock
        #      connection itself before running migrations.  pg_advisory_lock is
        #      session-level, so the lock survives the COMMIT.
        # NullPool: do not retain the connection in a pool after the migration.
        # Each schema migration opens a few short-lived engines (here plus the
        # ensure_* steps); with the default QueuePool those connections linger
        # until GC, and running many schemas in parallel (migration_concurrency)
        # multiplies that footprint and exhausts max_connections — observed as
        # "FATAL: sorry, too many clients already" sweeping 20k schemas at
        # concurrency 12. NullPool closes the connection on return.
        engine = create_engine(migration_url, poolclass=NullPool)
        with engine.connect() as conn:
            logger.debug(f"Acquiring migration advisory lock for schema '{schema_name}' (id={lock_id})...")
            waited_since = time.monotonic()
            next_report_at = _LOCK_WAIT_REPORT_AFTER_SECS
            while True:
                acquired = conn.execute(text(f"SELECT pg_try_advisory_lock({lock_id})")).scalar()
                if acquired:
                    break
                # A worker stuck here used to look like a silent hang: no log
                # line, no timeout, for as long as the lock stays taken.  Say
                # who is holding it, so an operator can find the blocking
                # backend (and see the pooler-recycled-leak shape of #4611).
                #
                # Report once the wait passes _LOCK_WAIT_REPORT_AFTER_SECS and
                # then every _LOCK_WAIT_REPORT_INTERVAL_SECS: queuing briefly
                # behind another worker is routine during a rolling deploy, and
                # one line per poll buries the signal in exactly the incident
                # this is meant to explain.
                waited = time.monotonic() - waited_since
                if waited >= next_report_at:
                    next_report_at = waited + _LOCK_WAIT_REPORT_INTERVAL_SECS
                    # Look the holder up BEFORE the commit below: this statement
                    # autobegins a transaction, and running it after the commit
                    # would hold its snapshot across the sleep — the very thing
                    # the commit exists to prevent.
                    holder = _advisory_lock_holder(conn, lock_id)
                    logger.warning(
                        "Waiting for migration advisory lock (id=%s, schema=%s) held by %s; "
                        "waited %.0fs so far, polling every %ss. If the holder is a stale pooled "
                        "backend, point %s at the direct PostgreSQL endpoint to bypass the pooler.",
                        lock_id,
                        schema_name,
                        holder or "another session (holder not found in pg_locks)",
                        waited,
                        _LOCK_POLL_INTERVAL_SECS,
                        ENV_MIGRATION_DATABASE_URL,
                    )
                # Commit the transaction so this connection holds no open snapshot
                # while waiting.  This prevents blocking CREATE INDEX CONCURRENTLY
                # that may be running in the migration worker.
                conn.commit()
                time.sleep(_LOCK_POLL_INTERVAL_SECS)

            # Everything from here on is inside the try, so the lock is released
            # however we leave — the commit below included.
            try:
                # Commit AFTER acquiring the lock too.  pg_advisory_lock is
                # session-level and survives the COMMIT, but the open transaction
                # on this connection would otherwise block any CREATE INDEX
                # CONCURRENTLY in the migration.
                conn.commit()
                logger.debug("Migration advisory lock acquired")

                vector_extension = configured_vector_extension()
                _bootstrap_vector_extension_for_migrations(conn, vector_extension)

                # Commit any pending transaction on the advisory-lock connection
                # before running migrations.  Some code paths above (e.g., the
                # pgvector extension check) may have started a transaction via
                # SQLAlchemy's autobegin.  If we leave it open, CREATE INDEX
                # CONCURRENTLY inside a migration will deadlock waiting for it.
                conn.commit()

                # Run migrations while holding the lock
                _run_migrations_internal(migration_url, script_location, schema=schema)
            finally:
                # Release the lock even when the migration failed.  If anything
                # above failed, this connection's transaction is aborted and
                # pg_advisory_unlock would raise InFailedSqlTransaction — the
                # lock would stay on the backend (a pooled one never releases
                # it, wedging every later migrator: #4611).  Roll the aborted
                # transaction back first so the unlock can run, and never let a
                # failing unlock mask the original error.
                try:
                    conn.rollback()
                except Exception as rollback_error:
                    logger.warning(
                        "Could not roll back the migration connection before releasing the advisory lock: %s",
                        rollback_error,
                    )
                try:
                    conn.execute(text(f"SELECT pg_advisory_unlock({lock_id})"))
                    logger.debug("Migration advisory lock released")
                except Exception as unlock_error:
                    logger.error(
                        "Failed to release migration advisory lock (id=%s) — the lock may stay on the "
                        "backend until the connection closes; close pooled connections or recycle the "
                        "pooler backend manually: %s",
                        lock_id,
                        unlock_error,
                    )

    except FileNotFoundError:
        logger.error(f"Alembic script location not found at {script_location}")
        raise
    except SystemExit as e:
        # Catch sys.exit() calls from Alembic
        logger.error(f"Alembic called sys.exit() with code: {e.code}", exc_info=True)
        raise RuntimeError(f"Database migration failed with exit code {e.code}") from e
    except Exception as e:
        logger.error(f"Failed to run database migrations: {e}", exc_info=True)
        raise RuntimeError("Database migration failed") from e


def _drop_embedding_vector_indexes(conn: Connection, schema_name: str, table_name: str) -> None:
    """Drop every vector index on ``table_name.embedding`` (HNSW, DiskANN, vchordrq, ScaNN)."""
    for index_name in _vector_index_names(conn, schema_name, table_name):
        _drop_index(conn, schema_name, index_name)


def _migrate_table_embedding_dimension(
    conn: Connection,
    schema_name: str,
    table_name: str,
    required_dimension: int,
    vector_ext: str,
    *,
    indexed: bool = True,
) -> None:
    """
    Migrate the embedding column of a single table to the required dimension.

    - If dimensions match: no action needed
    - If dimensions differ and table is empty: ALTER COLUMN to new dimension
    - If dimensions differ and table has data: raise error with migration guidance

    ``indexed=False`` keeps the column but with no vector index at all: any existing one is
    dropped and none is created, so the pgvector 2000-dimension index limit does not apply.
    """
    current_dim = conn.execute(
        text("""
            SELECT atttypmod
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = :schema
              AND c.relname = :table
              AND a.attname = 'embedding'
        """),
        {"schema": schema_name, "table": table_name},
    ).scalar()

    if current_dim is None:
        logger.debug(f"No embedding column found on {table_name}, skipping")
        return

    if not indexed:
        # Also on the dimension-match path: the base migrations create this index, so a
        # deployment that switches to a custom store still carries one until it is dropped here.
        _drop_embedding_vector_indexes(conn, schema_name, table_name)
        conn.commit()

    if current_dim == required_dimension:
        logger.debug(f"Embedding dimension OK for {table_name}: {current_dim}")
        return

    logger.info(
        f"Embedding dimension mismatch on {table_name}: database has {current_dim}, model requires {required_dimension}"
    )

    row_count = conn.execute(
        text(f"SELECT COUNT(*) FROM {schema_name}.{table_name} WHERE embedding IS NOT NULL")
    ).scalar()

    if row_count > 0:
        raise RuntimeError(
            f"Cannot change embedding dimension from {current_dim} to {required_dimension}: "
            f"{table_name} table contains {row_count} rows with embeddings. "
            f"To change dimensions, you must either:\n"
            f"  1. Re-embed all data: DELETE FROM {schema_name}.{table_name}; then restart\n"
            f"  2. Use a model with {current_dim}-dimensional embeddings"
        )

    logger.info(f"Altering {table_name}.embedding column dimension from {current_dim} to {required_dimension}")

    _drop_embedding_vector_indexes(conn, schema_name, table_name)
    conn.execute(
        text(f"ALTER TABLE {schema_name}.{table_name} ALTER COLUMN embedding TYPE vector({required_dimension})")
    )
    conn.commit()

    if not indexed:
        logger.info(f"Changed {table_name}.embedding dimension to {required_dimension} (no vector index)")
        return

    _create_embedding_vector_index(conn, schema_name, table_name, required_dimension, vector_ext, row_count)
    logger.info(f"Successfully changed {table_name}.embedding dimension to {required_dimension}")


def _has_embedding_vector_index(conn: Connection, schema_name: str, table_name: str) -> bool:
    return bool(_vector_index_names(conn, schema_name, table_name))


def _create_embedding_vector_index(
    conn: Connection,
    schema_name: str,
    table_name: str,
    dimension: int,
    vector_ext: str,
    row_count: int,
) -> None:
    """Build the vector index on ``table_name.embedding`` for the detected extension."""
    if vector_ext == "pgvector" and dimension > 2000:
        raise RuntimeError(
            f"Embedding dimension {dimension} exceeds pgvector HNSW index limit of 2000. "
            f"Use an embedding model with <= 2000 dimensions, or switch to a vector extension "
            f"that supports higher dimensions (e.g., pgvectorscale/DiskANN or AlloyDB ScaNN)."
        )

    index_type = index_type_keyword(vector_ext)
    if should_defer_index_creation(vector_ext, row_count):
        minimum_rows = minimum_rows_for_index(vector_ext)
        logger.warning(
            "Skipping %s index recreation on %s: AlloyDB ScaNN AUTO indexes need at least %s populated "
            "embedding rows; table currently has %s",
            vector_ext,
            table_name,
            minimum_rows,
            row_count,
        )
        return

    conn.execute(
        text(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding_{index_type}
            ON {schema_name}.{table_name}
            {index_using_clause(vector_ext)}
        """)
    )
    logger.info(f"Created {index_type} index on {table_name} for {dimension}-dimensional embeddings")
    conn.commit()


#: ``USER_TAB_COLUMNS.VECTOR_INFO`` reads ``VECTOR(384,FLOAT32,DENSE)`` for a fixed column and
#: ``VECTOR(*,*,DENSE)`` for a flexible one.
_ORACLE_VECTOR_INFO_RE = re.compile(r"VECTOR\(\s*(\*|\d+)", re.IGNORECASE)

#: Target accuracy of a rebuilt vector index — the value the Oracle baseline index uses.
_ORACLE_VECTOR_INDEX_TARGET_ACCURACY = 95

#: ``USER_INDEXES.INDEX_SUBTYPE`` of a vector index -> its ORGANIZATION clause.
_ORACLE_VECTOR_INDEX_ORGANIZATIONS = {
    "NEIGHBOR_PARTITIONS_IVF": "NEIGHBOR PARTITIONS",
    "INMEMORY_NEIGHBOR_GRAPH_HNSW": "INMEMORY NEIGHBOR GRAPH",
}

#: Distance metric names the vector-index catalogs may report (see _oracle_vector_index_params).
#: L2_SQUARED is the documented alias for EUCLIDEAN_SQUARED that the catalogs may report.
_ORACLE_VECTOR_INDEX_DISTANCES = {
    "EUCLIDEAN",
    "EUCLIDEAN_SQUARED",
    "L2_SQUARED",
    "COSINE",
    "DOT",
    "MANHATTAN",
    "HAMMING",
}

#: Prefix of the column comment that carries the vector-index DDL a resize still has to replay.
#: Oracle DDL is not transactional, so the definitions of the indexes a resize drops are written
#: to the catalog first; a run interrupted anywhere before the rebuild finds them there.
_ORACLE_PENDING_INDEXES_MARKER = "hindsight:pending-vector-indexes:"

#: Attempts at reconciling one table. Workers booting together race on the same DDL; a loser
#: re-reads the catalog, which by then describes the winner's progress, and carries on from there.
_ORACLE_RECONCILE_ATTEMPTS = 3


class _OraclePendingIndexesPayload(TypedDict, total=False):
    ddl: list[str]  # required: the CREATE VECTOR INDEX statements still owed
    comment: str  # the column comment the marker overwrote


#: Oracle caps a column comment at 4000 bytes; a marker that would not fit omits the carried
#: comment rather than fail the COMMENT ON COLUMN (the uninterrupted run still restores it).
_ORACLE_COLUMN_COMMENT_MAX_BYTES = 4000


@dataclass(frozen=True)
class _OracleEmbeddingColumns:
    vector_info: str | None  # VECTOR_INFO of EMBEDDING; None when the column (or table) is missing
    has_legacy: bool  # EMBEDDING_LEGACY left behind by an interrupted resize
    pending_index_ddl: list[str]  # vector indexes an interrupted resize dropped and has not rebuilt
    prior_comment: str  # the column comment the pending marker overwrote ("" when none)


def _oracle_embedding_columns(cursor: Any, table_name: str) -> _OracleEmbeddingColumns:
    cursor.execute(
        "SELECT column_name, vector_info FROM all_tab_columns "
        "WHERE owner = SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') "
        "AND table_name = :table_name AND column_name IN ('EMBEDDING', 'EMBEDDING_LEGACY')",
        {"table_name": table_name},
    )
    rows = cursor.fetchall()
    cursor.execute(
        "SELECT comments FROM all_col_comments "
        "WHERE owner = SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') "
        "AND table_name = :table_name AND column_name IN ('EMBEDDING', 'EMBEDDING_LEGACY')",
        {"table_name": table_name},
    )
    pending_index_ddl: list[str] = []
    prior_comment = ""
    for (c,) in cursor.fetchall():
        if c and c.startswith(_ORACLE_PENDING_INDEXES_MARKER):
            payload = json.loads(c[len(_ORACLE_PENDING_INDEXES_MARKER) :])
            if isinstance(payload, dict):
                # Newer markers also carry the comment they overwrote; ones written before
                # that field existed are a bare DDL list.
                pending_index_ddl = payload["ddl"]
                prior_comment = payload.get("comment", "") or ""
            else:
                pending_index_ddl = payload
            break
    return _OracleEmbeddingColumns(
        vector_info=next((info for name, info in rows if name == "EMBEDDING"), None),
        has_legacy=any(name == "EMBEDDING_LEGACY" for name, _ in rows),
        pending_index_ddl=pending_index_ddl,
        prior_comment=prior_comment,
    )


def _set_oracle_column_comment(cursor: Any, table_name: str, column: str, comment: str) -> None:
    cursor.execute(f"COMMENT ON COLUMN {table_name}.{column} IS '{comment.replace(chr(39), chr(39) * 2)}'")


def _set_oracle_pending_indexes(
    cursor: Any, table_name: str, column: str, index_ddl: list[str], prior_comment: str = ""
) -> None:
    """Record (or, with an empty list, clear) the vector-index DDL still owed on ``table_name``.

    The marker replaces the column's own comment, so the prior one rides inside the payload;
    a run resumed after an interruption puts it back once the indexes are rebuilt.
    """
    comment = ""
    if index_ddl:
        payload: _OraclePendingIndexesPayload = {"ddl": index_ddl}
        comment = _ORACLE_PENDING_INDEXES_MARKER + json.dumps(payload)
        if prior_comment:
            payload["comment"] = prior_comment
            with_comment = _ORACLE_PENDING_INDEXES_MARKER + json.dumps(payload)
            if len(with_comment.encode()) <= _ORACLE_COLUMN_COMMENT_MAX_BYTES:
                comment = with_comment
            else:
                logger.warning(
                    f"Column comment on {table_name}.{column} is too long to ride inside the pending-index "
                    "marker; an interrupted resize will not restore it"
                )
    _set_oracle_column_comment(cursor, table_name, column, comment)


def _oracle_column_comment(cursor: Any, table_name: str, column: str) -> str:
    """The comment currently recorded on ``table_name.column`` ("" when unset)."""
    cursor.execute(
        "SELECT comments FROM all_col_comments "
        "WHERE owner = SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') "
        "AND table_name = :table_name AND column_name = :column_name",
        {"table_name": table_name, "column_name": column.upper()},
    )
    row = cursor.fetchone()
    return str(row[0]) if row and row[0] else ""


def _drop_oracle_embedding_legacy(cursor: Any, table_name: str) -> None:
    """Drop ``embedding_legacy`` once it provably holds no row.

    Oracle DDL commits at every statement, so no lock spans a resize: a write that slips in
    between the emptiness check and the rename lands in ``embedding`` and moves into
    ``embedding_legacy`` with it. Dropping the column then would delete that row silently,
    so the run aborts instead — the pending-index marker still on the table lets the next
    attempt finish once the row is handled.
    """
    cursor.execute(f"SELECT 1 FROM {table_name} WHERE embedding_legacy IS NOT NULL FETCH FIRST 1 ROWS ONLY")
    if cursor.fetchone() is not None:
        raise RuntimeError(
            f"Cannot drop {table_name}.embedding_legacy: it holds a row written while the resize "
            "was in flight. Delete the row (or re-embed it into embedding with the configured "
            "model), then restart to finish the resize."
        )
    # The rename carried embedding's comment here; hand it back to the new column unless the
    # marker (or an already-restored comment) occupies it — never overwrite the pending DDL.
    legacy_comment = _oracle_column_comment(cursor, table_name, "embedding_legacy")
    cursor.execute(f"ALTER TABLE {table_name} DROP COLUMN embedding_legacy")
    if (
        legacy_comment
        and not legacy_comment.startswith(_ORACLE_PENDING_INDEXES_MARKER)
        and not _oracle_column_comment(cursor, table_name, "embedding")
    ):
        _set_oracle_column_comment(cursor, table_name, "embedding", legacy_comment)


def _create_oracle_vector_index(cursor: Any, ddl: str) -> None:
    try:
        cursor.execute(ddl)
    except Exception as e:
        # ORA-00955: a concurrent worker (or the interrupted run) already created it.
        if "ORA-00955" not in str(e):
            raise


def _rebuild_pending_oracle_indexes(
    cursor: Any, table_name: str, index_ddl: list[str], prior_comment: str = ""
) -> None:
    for ddl in index_ddl:
        _create_oracle_vector_index(cursor, ddl)
    _set_oracle_pending_indexes(cursor, table_name, "embedding", [])
    if prior_comment:
        _set_oracle_column_comment(cursor, table_name, "embedding", prior_comment)
    logger.warning(f"Rebuilt {len(index_ddl)} vector index(es) an interrupted resize left on {table_name}")


@dataclass(frozen=True)
class _OracleVectorIndexParams:
    distance: str
    accuracy: int


def _oracle_vector_index_params(cursor: Any, table_name: str, index_name: str) -> _OracleVectorIndexParams:
    """The DISTANCE and TARGET ACCURACY ``index_name`` was created with.

    ``ALL_VECTOR_INDEXES`` reports them where the migration user can read the
    catalog (on Oracle Free none of the three is readable); the fallbacks cover
    where it may be empty — ``V$VECTOR_INDEX`` on Autonomous AI Database,
    ``VECSYS.VECTOR$INDEX`` elsewhere (each is documented as unavailable on the
    other). An index created without the clauses reports COSINE and 95 —
    also the fallback when the migration user cannot read any of the catalogs —
    while other tuning parameters (neighbors, centroids, DOP) reset to defaults.
    A catalog row proves the index exists; when none answers at all the rebuild
    still assumes the defaults, but loudly — a custom-metric index rebuilt as
    COSINE/95 changes retrieval semantics and must not pass unnoticed.
    """
    saw_index = False
    unrecognized_distance = None
    for sql, binds in (
        (
            "SELECT distance_metric, accuracy FROM all_vector_indexes "
            "WHERE owner = SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') "
            "AND index_name = :index_name AND target_table = :table_name",
            {"index_name": index_name, "table_name": table_name},
        ),
        (
            "SELECT distance_type, default_accuracy FROM v$vector_index "
            "WHERE owner = SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') AND index_name = :index_name",
            {"index_name": index_name},
        ),
        (
            "SELECT JSON_VALUE(idx_params, '$.distance'), JSON_VALUE(idx_params, '$.accuracy' RETURNING NUMBER) "
            "FROM vecsys.vector$index WHERE idx_name = :index_name AND idx_base_table_objn = "
            "(SELECT object_id FROM all_objects WHERE owner = SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') "
            "AND object_name = :table_name AND object_type = 'TABLE')",
            {"index_name": index_name, "table_name": table_name},
        ),
    ):
        try:
            cursor.execute(sql, binds)
            row = cursor.fetchone()
        except Exception:
            continue
        if row is None:
            continue
        saw_index = True
        distance = str(row[0]).upper() if row[0] is not None else None
        if distance in _ORACLE_VECTOR_INDEX_DISTANCES:
            try:
                accuracy = int(row[1])
            except (TypeError, ValueError):
                accuracy = 0
            return _OracleVectorIndexParams(
                distance, accuracy if 1 <= accuracy <= 100 else _ORACLE_VECTOR_INDEX_TARGET_ACCURACY
            )
        unrecognized_distance = distance or str(row[0])
    if not saw_index:
        logger.warning(
            f"No vector-index catalog could read {index_name} on {table_name}; "
            "the resize will rebuild it with the baseline defaults (COSINE, TARGET ACCURACY 95). "
            "If it was created with a custom DISTANCE or TARGET ACCURACY, grant the migration "
            "user read access to ALL_VECTOR_INDEXES / V$VECTOR_INDEX and restart."
        )
    else:
        logger.warning(
            f"{index_name} on {table_name} reports distance metric {unrecognized_distance!r}, "
            "which this migration does not recognize; the resize will rebuild it with the "
            "baseline defaults (COSINE, TARGET ACCURACY 95). If it was created with a custom "
            "DISTANCE or TARGET ACCURACY, restore them after the resize."
        )
    return _OracleVectorIndexParams("COSINE", _ORACLE_VECTOR_INDEX_TARGET_ACCURACY)


def _ensure_oracle_table_embedding_dimension(cursor: Any, table_name: str, required_dimension: int) -> None:
    """Reconcile one Oracle table's ``embedding`` column with the model's dimension.

    Mirrors the PostgreSQL rules: a matching column is left alone, an empty table is resized,
    and a table holding embeddings of another dimension fails with guidance instead of
    silently mixing vector spaces. A flexible ``VECTOR(*, *)`` column (not created by the
    baseline, but found on deployments resized by hand) is accepted as long as every stored
    embedding has the model's dimension; it is never altered, since that would rewrite a
    populated column.

    ``cursor`` is a python-oracledb cursor whose session CURRENT_SCHEMA is the target schema.
    """
    columns = _oracle_embedding_columns(cursor, table_name)
    if columns.has_legacy:
        # A previous resize stopped between its DDL steps (Oracle DDL is not transactional).
        # It only ever runs on an empty table, so finishing it cannot lose data — unless a
        # write raced the interrupted run's check; _drop_oracle_embedding_legacy refuses that.
        if columns.vector_info is None:
            cursor.execute(f"ALTER TABLE {table_name} ADD (embedding VECTOR({required_dimension}, FLOAT32))")
        if columns.pending_index_ddl:
            # The marker may still sit on the legacy column; move it before that column goes.
            _set_oracle_pending_indexes(
                cursor, table_name, "embedding", columns.pending_index_ddl, columns.prior_comment
            )
        _drop_oracle_embedding_legacy(cursor, table_name)
        logger.warning(f"Finished an interrupted resize of {table_name}.embedding")
        columns = _oracle_embedding_columns(cursor, table_name)

    if columns.vector_info is None:
        logger.debug(f"No embedding column found on {table_name}, skipping")
        return

    if columns.pending_index_ddl:
        _rebuild_pending_oracle_indexes(cursor, table_name, columns.pending_index_ddl, columns.prior_comment)

    match = _ORACLE_VECTOR_INFO_RE.match(columns.vector_info)
    if match is None:
        raise RuntimeError(f"Unrecognised VECTOR_INFO {columns.vector_info!r} on {table_name}.embedding")
    declared = None if match.group(1) == "*" else int(match.group(1))
    if declared == required_dimension:
        logger.debug(f"Embedding dimension OK for {table_name}: {declared}")
        return

    if declared is None:
        # Stops at the first mismatching row, so a consistent table costs one scan per boot.
        cursor.execute(
            f"SELECT VECTOR_DIMENSION_COUNT(embedding) FROM {table_name} "
            "WHERE embedding IS NOT NULL AND VECTOR_DIMENSION_COUNT(embedding) <> :dim FETCH FIRST 1 ROWS ONLY",
            {"dim": required_dimension},
        )
        other = cursor.fetchone()
        if other is not None:
            raise RuntimeError(
                f"{table_name}.embedding is a flexible VECTOR column holding {other[0]}-dimensional embeddings, "
                f"but the embeddings model produces {required_dimension} dimensions. Re-embed the stored rows "
                f"with the configured model, or configure a model with {other[0]}-dimensional embeddings."
            )
        logger.info(f"{table_name}.embedding is a flexible VECTOR column; stored embeddings match {required_dimension}")
        return

    logger.info(
        f"Embedding dimension mismatch on {table_name}: database has {declared}, model requires {required_dimension}"
    )
    # Hold off writers while the emptiness check decides whether the column can be replaced:
    # an insert landing between the check and the rename below would move into embedding_legacy
    # and be dropped with it. DDL releases the lock at its first commit, so the guarded drop in
    # _drop_oracle_embedding_legacy is the hard stop for anything the lock still misses.
    cursor.execute(f"LOCK TABLE {table_name} IN EXCLUSIVE MODE WAIT 30")
    cursor.execute(
        f"SELECT VECTOR_DIMENSION_COUNT(embedding) FROM {table_name} WHERE embedding IS NOT NULL FETCH FIRST 1 ROWS ONLY"
    )
    if cursor.fetchone() is not None:
        raise RuntimeError(
            f"Cannot change embedding dimension from {declared} to {required_dimension}: "
            f"{table_name} contains rows with embeddings. To change dimensions, you must either:\n"
            f"  1. Re-embed all data: DELETE FROM {table_name}; then restart\n"
            f"  2. Use a model with {declared}-dimensional embeddings"
        )

    # Oracle cannot change a VECTOR column's dimension in place: ALTER TABLE ... MODIFY fails
    # with ORA-51859 even on an empty table (verified on 23.26.3), so the column is replaced.
    # The steps are ordered so that an interruption leaves EMBEDDING_LEGACY behind, which the
    # next run finishes (see above). Vector indexes are dropped first and rebuilt with the same
    # organization, locality, distance metric, and target accuracy; their DDL is recorded in a
    # column comment before the drop and cleared after the rebuild, so a run interrupted in
    # between rebuilds them (see above). Oracle keeps a column's comment across RENAME COLUMN,
    # and the recovery moves it off the legacy column before dropping it. They cannot be
    # replayed from DBMS_METADATA.GET_DDL, which on 23.26.3 returns a plain CREATE INDEX
    # without the VECTOR clauses (ORA-02327 on replay), so the clauses are rebuilt from the
    # catalog instead; tuning parameters beyond distance/accuracy reset to defaults, which is
    # harmless on the empty table this runs on.
    cursor.execute(
        "SELECT index_name, index_subtype, partitioned FROM all_indexes "
        "WHERE owner = SYS_CONTEXT('USERENV', 'CURRENT_SCHEMA') AND table_name = :table_name "
        "AND index_type = 'VECTOR'",
        {"table_name": table_name},
    )
    index_names: list[str] = []
    index_ddl: list[str] = []
    for name, subtype, partitioned in cursor.fetchall():
        organization = _ORACLE_VECTOR_INDEX_ORGANIZATIONS.get(subtype)
        if organization is None:
            raise RuntimeError(f"Cannot rebuild vector index {name} of unknown subtype {subtype!r} on {table_name}")
        params = _oracle_vector_index_params(cursor, table_name, name)
        index_names.append(name)
        index_ddl.append(
            f'CREATE VECTOR INDEX "{name}" ON {table_name} (embedding) ORGANIZATION {organization} '
            f"DISTANCE {params.distance} WITH TARGET ACCURACY {params.accuracy}"
            + (" LOCAL" if partitioned == "YES" else "")
        )
    # The column comment rides the rename onto embedding_legacy and dies with it, and the
    # pending marker would overwrite it first — capture it for the restore at the end either way.
    prior_comment = _oracle_column_comment(cursor, table_name, "embedding")
    if index_ddl:
        _set_oracle_pending_indexes(cursor, table_name, "embedding", index_ddl, prior_comment)
    for name in index_names:
        cursor.execute(f'DROP INDEX "{name}"')
    cursor.execute(f"ALTER TABLE {table_name} RENAME COLUMN embedding TO embedding_legacy")
    cursor.execute(f"ALTER TABLE {table_name} ADD (embedding VECTOR({required_dimension}, FLOAT32))")
    if index_ddl:
        _set_oracle_pending_indexes(cursor, table_name, "embedding", index_ddl, prior_comment)
    _drop_oracle_embedding_legacy(cursor, table_name)
    for ddl in index_ddl:
        _create_oracle_vector_index(cursor, ddl)
    if index_ddl:
        _set_oracle_pending_indexes(cursor, table_name, "embedding", [])
    if prior_comment:
        _set_oracle_column_comment(cursor, table_name, "embedding", prior_comment)
    logger.info(
        f"Changed {table_name}.embedding dimension to {required_dimension} ({len(index_ddl)} vector index(es) rebuilt)"
    )


def _ensure_embedding_dimension_oracle(
    database_url: str, required_dimension: int, schema: str | None, *, store_owned_memories: bool
) -> None:
    """Oracle counterpart of the PostgreSQL dimension reconcile (see ensure_embedding_dimension)."""
    from .engine.db.oracle import _import_oracledb, _oracle_connect_params

    oracledb = _import_oracledb()
    tables = ["MENTAL_MODELS"] if store_owned_memories else ["MEMORY_UNITS", "MENTAL_MODELS"]
    with oracledb.connect(**_oracle_connect_params(database_url)) as conn:
        cursor = conn.cursor()
        # Wait for DDL locks instead of failing immediately (ORA-00054), like the migrations.
        cursor.execute("ALTER SESSION SET DDL_LOCK_TIMEOUT = 30")
        if schema:
            cursor.execute(f'ALTER SESSION SET CURRENT_SCHEMA = "{schema.replace(chr(34), chr(34) * 2)}"')
        for table_name in tables:
            for attempt in range(1, _ORACLE_RECONCILE_ATTEMPTS + 1):
                try:
                    _ensure_oracle_table_embedding_dimension(cursor, table_name, required_dimension)
                    break
                except oracledb.DatabaseError as e:
                    # Every step is re-derived from the catalog, so a retry resumes where the
                    # concurrent worker left the table. RuntimeError (data mismatch) is not retried.
                    if attempt == _ORACLE_RECONCILE_ATTEMPTS:
                        raise
                    logger.warning(f"Reconciling {table_name}.embedding raced another DDL ({e}); re-checking")


def ensure_embedding_dimension(
    database_url: str,
    required_dimension: int,
    schema: str | None = None,
    vector_extension: str = "pgvector",
    store_owned_memories: bool = False,
) -> None:
    """
    Ensure the embedding column dimension matches the model's dimension for all tables.

    Checks and adjusts memory_units.embedding and mental_models.embedding:
    - If dimensions match: no action needed
    - If dimensions differ and table is empty: ALTER COLUMN to new dimension
    - If dimensions differ and table has data: raise error with migration guidance

    Args:
        database_url: SQLAlchemy database URL
        required_dimension: The embedding dimension required by the model
        schema: Target PostgreSQL schema name (None for public)
        vector_extension: Configured vector extension ("pgvector", "vchord", "pgvectorscale", or "scann")
        store_owned_memories: A custom memories store owns the memory rows and the mental-model
            search. memory_units is left untouched (its rows live in the store), and
            mental_models.embedding still follows the model — Postgres keeps writing it — but
            carries no vector index, because the store answers every mental-model vector query

    Raises:
        RuntimeError: If dimension mismatch with existing data
    """
    if is_oracle_url(database_url):
        _ensure_embedding_dimension_oracle(
            database_url, required_dimension, schema, store_owned_memories=store_owned_memories
        )
        return

    schema_name = schema or "public"

    engine = create_engine(to_libpq_url(database_url), poolclass=NullPool)
    with engine.connect() as conn:
        # Check if memory_units table exists (proxy for schema being initialized)
        table_exists = conn.execute(
            text("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = :schema AND table_name = 'memory_units'
                )
            """),
            {"schema": schema_name},
        ).scalar()

        if not table_exists:
            logger.debug(f"memory_units table does not exist in schema '{schema_name}', skipping dimension check")
            return

        # Detect which vector extension is available
        vector_ext = _detect_vector_extension(conn, vector_extension)
        logger.info(f"Using vector extension: {vector_ext}")

        if not store_owned_memories:
            _migrate_table_embedding_dimension(conn, schema_name, "memory_units", required_dimension, vector_ext)
        _migrate_table_embedding_dimension(
            conn, schema_name, "mental_models", required_dimension, vector_ext, indexed=not store_owned_memories
        )
        if not store_owned_memories and not _has_embedding_vector_index(conn, schema_name, "mental_models"):
            # A deployment that ran with a custom store and moved back to Postgres has the column at
            # the right dimension but no index (the store-owned branch above dropped it). Without
            # this the resize path is the only thing that ever builds it, and a matching dimension
            # never resizes — every page search would seq-scan until the model changed.
            row_count = conn.execute(
                text(f"SELECT COUNT(*) FROM {schema_name}.mental_models WHERE embedding IS NOT NULL")
            ).scalar()
            _create_embedding_vector_index(
                conn, schema_name, "mental_models", required_dimension, vector_ext, row_count
            )
        # NOTE: invalidated_memory_units is deliberately omitted. The curation archive has no
        # embedding column at all (dropped in migration d4f6a8c2e1b3) — invalidate stores no
        # embedding and revert recomputes one — so there is no archive vector to re-dimension
        # and a model switch can't trip a dimension mismatch there (#2209).


def ensure_vector_extension(
    database_url: str,
    vector_extension: str = "pgvector",
    schema: str | None = None,
    store_owned_memories: bool = False,
) -> None:
    """
    Ensure the vector indexes match the configured vector extension.

    This function checks the current vector index type in the database
    and adjusts it if necessary:
    - If index type matches configured extension: no action needed
    - If they differ and tables are empty: drop old indexes, recreate with new type
    - If they differ and tables have data: raise error with migration guidance

    Args:
        database_url: SQLAlchemy database URL
        vector_extension: Configured vector extension ("pgvector", "vchord", "pgvectorscale", or "scann")
        schema: Target PostgreSQL schema name (None for public)
        store_owned_memories: Leave memory_units untouched because a custom memories store
            keeps the memory rows (and their vectors) outside Postgres. mental_models is not
            in this reconcile at all; its index is handled by ensure_embedding_dimension

    Raises:
        RuntimeError: If extension mismatch with existing data
    """
    schema_name = schema or "public"

    engine = create_engine(to_libpq_url(database_url), poolclass=NullPool)
    with engine.connect() as conn:
        # Detect which vector extension should be used
        target_ext = _detect_vector_extension(conn, vector_extension)
        logger.info(f"Target vector extension: {target_ext}")

        # Tables with vector indexes to check
        tables_to_check = [
            ("memory_units", "idx_memory_units_embedding"),
            ("learnings", "idx_learnings_embedding"),
            ("pinned_reflections", "idx_pinned_reflections_embedding"),
        ]
        if store_owned_memories:
            tables_to_check = [entry for entry in tables_to_check if entry[0] != "memory_units"]

        target_index_type = index_type_keyword(target_ext)

        mismatched_tables = []
        tables_with_data = []

        for table_name, index_name in tables_to_check:
            # Check if table exists
            table_exists = conn.execute(
                text("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = :schema AND table_name = :table_name
                    )
                """),
                {"schema": schema_name, "table_name": table_name},
            ).scalar()

            if not table_exists:
                logger.debug(f"Table {table_name} does not exist in schema '{schema_name}', skipping")
                continue

            row_count = conn.execute(
                text(f"SELECT COUNT(*) FROM {schema_name}.{table_name} WHERE embedding IS NOT NULL")
            ).scalar()

            # Check current index type by querying pg_indexes
            current_index_rows = conn.execute(
                text("""
                    SELECT indexdef, indexname
                    FROM pg_indexes
                    WHERE schemaname = :schema
                      AND tablename = :table_name
                      AND indexname LIKE :index_pattern
                """),
                {"schema": schema_name, "table_name": table_name, "index_pattern": "%embedding%"},
            ).fetchall()

            if table_name == "memory_units" and uses_per_bank_vector_indexes(target_ext):
                # Per-bank backends never use a GLOBAL memory_units vector index.
                # Every vector search is bank + fact_type scoped, and is served
                # either by the bank's own partial index or — for a bank below
                # the size threshold, which is most of them — by an exact
                # (bank_id, fact_type) B-tree scan plus a top-N sort. The planner
                # never picks a global index when bank_id is in the WHERE clause,
                # which is exactly why migration d5e6f7a8b9c0 drops it for these
                # backends. The partial indexes themselves are owned by the
                # maintenance sweep (engine/vector_index_health.py), not by this
                # reconcile and not by bank creation.
                #
                # The reconcile is strictly hands-off here, in BOTH directions:
                # never create the index (dead weight — an older version of this
                # branch did, which is how legacy schemas ended up carrying it),
                # and never drop or rebuild one that exists. Runtime DROP/CREATE
                # INDEX takes an ACCESS EXCLUSIVE lock on memory_units at
                # unpredictable times (startup, tenant provisioning); index DDL
                # belongs in the versioned migration path. Leftover globals are
                # removed by migration f2a6d8c4b1e9. This continue also keeps
                # memory_units out of the type-mismatch reconcile below, which
                # would otherwise recreate a global index with the new type on a
                # backend switch.
                if current_index_rows:
                    stale_names = ", ".join(row[1] for row in current_index_rows)
                    logger.info(
                        f"Global vector index ({stale_names}) present on {schema_name}.memory_units "
                        f"with per-bank backend ({target_ext}); left untouched — removed by "
                        f"migration f2a6d8c4b1e9"
                    )
                else:
                    logger.debug(
                        f"Per-bank vector backend ({target_ext}); skipping global {index_name} creation on {table_name}"
                    )
                continue

            current_index_info = current_index_rows[0] if current_index_rows else None
            if not current_index_info:
                logger.warning(f"No embedding index found for {table_name}, will create it if safe")
                mismatched_tables.append((table_name, index_name, None, row_count))
                continue

            indexdef = current_index_info[0].lower()
            if "scann" in indexdef:
                current_index_type = "scann"
            elif "diskann" in indexdef:
                current_index_type = "diskann"
            elif "vchordrq" in indexdef:
                current_index_type = "vchordrq"
            elif "hnsw" in indexdef:
                current_index_type = "hnsw"
            else:
                logger.warning(f"Unknown index type for {table_name}: {indexdef}")
                continue

            # Check if index type matches target
            if current_index_type != target_index_type:
                logger.info(
                    f"Index type mismatch on {table_name}: current={current_index_type}, target={target_index_type}"
                )
                mismatched_tables.append((table_name, index_name, current_index_type, row_count))

                if row_count > 0 and target_ext != "scann":
                    tables_with_data.append((table_name, row_count, current_index_type))
            else:
                logger.debug(f"Index type OK for {table_name}: {current_index_type}")
                if target_ext == "scann" and table_name == "memory_units":
                    _drop_per_bank_vector_indexes(conn, schema_name)
                    conn.commit()

        # If no mismatches, we're done
        if not mismatched_tables:
            logger.debug(f"All vector indexes match configured extension: {target_ext}")
            return

        # If there's data in any non-ScaNN mismatched table, raise error
        if tables_with_data:
            table_list = ", ".join([f"{table}({count} rows)" for table, count, _ in tables_with_data])
            current_index_type = tables_with_data[0][2]
            # Map index type back to extension name for error message
            current_ext_name = {
                "diskann": "pgvectorscale",
                "vchordrq": "vchord",
                "hnsw": "pgvector",
                "scann": "scann",
            }.get(current_index_type, current_index_type)

            raise RuntimeError(
                f"Cannot change vector extension from {current_index_type} to {target_index_type}: "
                f"the following tables contain data: {table_list}. "
                f"To change vector extension, you must either:\n"
                f"  1. Re-embed all data: DELETE FROM {schema_name}.memory_units; "
                f"DELETE FROM {schema_name}.learnings; DELETE FROM {schema_name}.pinned_reflections; then restart\n"
                f"  2. Use the current vector extension (set HINDSIGHT_API_VECTOR_EXTENSION='{current_ext_name}')"
            )

        logger.info(f"Reconciling vector indexes for {target_ext}")

        for table_name, index_name, current_type, row_count in mismatched_tables:
            if should_defer_index_creation(target_ext, row_count):
                minimum_rows = minimum_rows_for_index(target_ext)
                logger.warning(
                    "Skipping %s index creation on %s: AlloyDB ScaNN AUTO indexes need at least %s populated "
                    "embedding rows; table currently has %s",
                    target_ext,
                    table_name,
                    minimum_rows,
                    row_count,
                )
                continue

            # Drop existing index if it exists
            if current_type:
                logger.info(f"Dropping {current_type} index on {table_name}")
                conn.execute(text(f"DROP INDEX IF EXISTS {schema_name}.{index_name}"))

            # Create new index with appropriate type
            if target_ext == "pgvector":
                # Check embedding dimension — pgvector HNSW indexes only support up to 2000 dims
                embed_dim = conn.execute(
                    text("""
                        SELECT atttypmod
                        FROM pg_attribute a
                        JOIN pg_class c ON a.attrelid = c.oid
                        JOIN pg_namespace n ON c.relnamespace = n.oid
                        WHERE n.nspname = :schema AND c.relname = :table_name AND a.attname = 'embedding'
                    """),
                    {"schema": schema_name, "table_name": table_name},
                ).scalar()

                if embed_dim and embed_dim > 2000:
                    raise RuntimeError(
                        f"Embedding dimension {embed_dim} on {table_name} exceeds pgvector HNSW index limit of 2000. "
                        f"Use an embedding model with <= 2000 dimensions, or switch to a vector extension "
                        f"that supports higher dimensions (e.g., pgvectorscale/DiskANN or AlloyDB ScaNN)."
                    )

            logger.info(f"Creating {target_index_type} index on {table_name}")
            conn.execute(
                text(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {schema_name}.{table_name}
                    {index_using_clause(target_ext)}
                """)
            )
            if target_ext == "scann" and table_name == "memory_units":
                _drop_per_bank_vector_indexes(conn, schema_name)

        conn.commit()
        logger.info(f"Successfully reconciled vector indexes for {target_ext}")


def _reconcile_needs_no_backfill(
    text_search_extension: str,
    table_name: str,
    current_column_type: str | None,
    current_index_type: str | None,
) -> bool:
    """Is this mismatch safe to reconcile even though the table holds rows?

    Only one transition qualifies: ``mental_models`` sitting on the migration-time
    native tsvector projection while pgroonga is configured. pgroonga indexes
    ``name + content`` directly, so the replacement ``search_vector`` is a dummy
    column with nothing to backfill — dropping the derived tsvector loses no data
    the reconciler would have to recompute.

    This state exists on every pgroonga deployment because the reconciler used to
    check the pre-rename ``reflections`` table name and therefore never converted
    mental models (issue #3307). Every other transition (anything writing
    ``memory_units``, or a target column that stores a per-row tsvector/bm25vector)
    needs values only the write path can produce, so it stays fail-closed.
    """
    return (
        text_search_extension == "pgroonga"
        and table_name == "mental_models"
        and current_column_type == "tsvector"
        and current_index_type in {None, "gin"}
    )


def _ensure_pgroonga_extension(conn: Connection) -> None:
    try:
        create_extension(conn, "pgroonga", cascade=True)
    except Exception:
        # Extension might already exist or user lacks permissions — verify
        has_ext = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'pgroonga'")).fetchone()
        if not has_ext:
            raise


def _create_text_search_index(
    conn: Connection,
    schema_name: str,
    table_name: str,
    text_search_extension: str,
    pg_search_tokenizer: str | None,
) -> None:
    """Build ``idx_<table>_text_search`` for the configured backend over an existing column.

    Re-executable (``IF NOT EXISTS``): replicas boot concurrently and each runs the reconcile.
    """
    index_name = f"idx_{table_name.replace('.', '_')}_text_search"
    if text_search_extension == "vchord":
        logger.info(f"Creating BM25 index on {table_name}")
        conn.execute(
            text(f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {schema_name}.{table_name}
                USING bm25 (search_vector bm25_catalog.bm25_ops)
            """)
        )
    elif text_search_extension == "pg_textsearch":
        logger.info(f"Creating BM25 index on {table_name}")
        # Different expression for each table
        if table_name == "memory_units":
            index_expr = "(COALESCE(text, '') || ' ' || COALESCE(context, ''))"
        else:  # mental_models
            index_expr = mental_models_text_document()

        conn.execute(
            text(f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {schema_name}.{table_name}
                USING bm25({index_expr})
                WITH (text_config='english')
            """)
        )
    elif text_search_extension == "pgroonga":
        # pgroonga index expression mirrors pg_textsearch
        if table_name == "memory_units":
            index_expr = "(COALESCE(text, '') || ' ' || COALESCE(context, '') || ' ' || COALESCE(text_signals, ''))"
        else:  # mental_models — knowledge_bm25_arm repeats this verbatim
            index_expr = mental_models_text_document()

        logger.info(f"Creating pgroonga index on {table_name}")
        # TokenBigram is the polyglot default — falls back to whitespace
        # tokenization for space-separated languages and bigram for CJK.
        # NormalizerNFKC150 handles Unicode normalization (full/half-width,
        # case folding, etc.) which materially improves Japanese recall.
        conn.execute(
            text(f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {schema_name}.{table_name}
                USING pgroonga ({index_expr})
                WITH (tokenizer='TokenBigram', normalizer='NormalizerNFKC150')
            """)
        )
    elif text_search_extension == "pg_search":
        # ParadeDB BM25 index over the table's primary key and text columns.
        # Column list mirrors what the initial / text_signals migrations create.
        if table_name == "memory_units":
            bm25_cols = pg_search_bm25_columns("id", ("text", "context", "text_signals"), pg_search_tokenizer)
        else:  # mental_models
            bm25_cols = pg_search_bm25_columns("id", ("name", "content"), pg_search_tokenizer)

        logger.info(f"Creating ParadeDB BM25 index on {table_name}")
        conn.execute(
            text(f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {schema_name}.{table_name}
                USING bm25 ({bm25_cols})
                WITH (key_field='id')
            """)
        )
    else:  # native
        logger.info(f"Creating GIN index on {table_name}")
        conn.execute(
            text(f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {schema_name}.{table_name}
                USING gin(search_vector)
            """)
        )


def ensure_text_search_extension(
    database_url: str,
    text_search_extension: str = "native",
    schema: str | None = None,
    pg_search_tokenizer: str | None = None,
    store_owned_memories: bool = False,
) -> None:
    """
    Ensure the text search columns and indexes match the configured extension.

    This function checks the current search_vector column type and index type
    in the database and adjusts them if necessary:
    - If they match configured extension: no action needed
    - If they differ and tables are empty: drop old column/index, recreate with new type
    - If they differ and tables have data: raise error with migration guidance,
      except for the mental-model native-to-pgroonga transition, which needs no
      backfill (pgroonga indexes the base columns) and so is safe while populated

    Args:
        database_url: SQLAlchemy database URL
        text_search_extension: Configured text search extension — one of
            "native", "vchord", "pg_textsearch", "pgroonga", or "pg_search"
        schema: Target PostgreSQL schema name (None for public)
        pg_search_tokenizer: Optional ParadeDB tokenizer to apply to pg_search
            BM25 text fields when indexes are created. Empty keeps the
            ParadeDB default.
        store_owned_memories: A custom memories store owns the memory rows and the knowledge-page
            search, so neither table is reconciled and the mental_models BM25 index is dropped:
            nothing reads it, yet on the native backend Postgres maintains it on every page write
            (``search_vector`` is a generated column there)

    Raises:
        RuntimeError: If extension mismatch with existing data
    """
    schema_name = schema or "public"
    pg_search_tokenizer = normalize_pg_search_tokenizer(pg_search_tokenizer)

    engine = create_engine(to_libpq_url(database_url), poolclass=NullPool)
    with engine.connect() as conn:
        if store_owned_memories:
            conn.execute(text(f"DROP INDEX IF EXISTS {schema_name}.idx_mental_models_text_search"))
            conn.commit()
            return

        # Tables with search_vector columns to check
        tables_to_check = ["memory_units", "mental_models"]

        # Determine target column type and index type
        if text_search_extension == "vchord":
            target_column_type = "bm25vector"
            target_index_type = "bm25"
        elif text_search_extension == "pg_textsearch":
            target_column_type = "text"
            target_index_type = "bm25"
        elif text_search_extension == "pgroonga":
            # pgroonga indexes the base text columns directly. We keep a dummy
            # TEXT column named search_vector for symmetry with pg_textsearch
            # and so the column-type mismatch detection above keeps working.
            target_column_type = "text"
            target_index_type = "pgroonga"
        elif text_search_extension == "pg_search":
            # ParadeDB: same column type / access method as pg_textsearch.
            # Disambiguated below by inspecting the index reloptions (key_field).
            target_column_type = "text"
            target_index_type = "bm25"
        else:  # native
            target_column_type = "tsvector"
            target_index_type = "gin"

        mismatched_tables = []
        tables_with_data = []
        # Column already in the target shape, index gone. Rebuilding the index needs no backfill
        # (it is derived from data already in the row), so unlike a real mismatch this is safe on
        # a populated table. The state is what a deployment that ran with a custom memories store
        # (which drops the mental_models index) leaves behind when it moves back to Postgres.
        missing_index_tables = []

        for table_name in tables_to_check:
            # Check if table exists
            table_exists = conn.execute(
                text("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = :schema AND table_name = :table_name
                    )
                """),
                {"schema": schema_name, "table_name": table_name},
            ).scalar()

            if not table_exists:
                logger.debug(f"Table {table_name} does not exist in schema '{schema_name}', skipping")
                continue

            # Get current column type from information_schema
            current_column_info = conn.execute(
                text("""
                    SELECT data_type, udt_name
                    FROM information_schema.columns
                    WHERE table_schema = :schema
                      AND table_name = :table_name
                      AND column_name = 'search_vector'
                """),
                {"schema": schema_name, "table_name": table_name},
            ).fetchone()

            if not current_column_info:
                logger.warning(f"No search_vector column found for {table_name}, will create it")
                mismatched_tables.append((table_name, None, None, False))
                continue

            # Check column type (udt_name contains the actual type: tsvector, bm25vector, etc.)
            current_column_type = current_column_info[1]  # udt_name

            # Get current index type and definition. The definition lets us
            # disambiguate pg_textsearch vs pg_search (both register a `bm25`
            # access method but only pg_search uses the `key_field` reloption).
            current_index_info = conn.execute(
                text("""
                    SELECT am.amname, pi.indexdef
                    FROM pg_indexes pi
                    JOIN pg_class c
                      ON c.relname = pi.indexname
                     AND c.relnamespace = to_regnamespace(pi.schemaname)
                    JOIN pg_am am ON am.oid = c.relam
                    WHERE pi.schemaname = :schema
                      AND pi.tablename = :table_name
                      AND pi.indexname = :index_name
                """),
                {
                    "schema": schema_name,
                    "table_name": table_name,
                    "index_name": f"idx_{table_name.replace('.', '_')}_text_search",
                },
            ).fetchone()

            current_index_type = current_index_info[0] if current_index_info else None
            current_index_def = current_index_info[1] if current_index_info else None

            # Detect pg_search specifically (vs pg_textsearch) via the key_field reloption
            current_is_pg_search = bool(current_index_def and "key_field" in current_index_def)
            want_pg_search = text_search_extension == "pg_search"

            # Check if column and index types match target
            column_matches = current_column_type == target_column_type
            index_matches = current_index_type == target_index_type if current_index_type else False
            # When both target and current sit at column=text/index=bm25, the
            # access-method check alone can't tell pg_textsearch from pg_search —
            # require the key_field reloption to agree with the configured backend.
            if column_matches and index_matches and target_index_type == "bm25" and target_column_type == "text":
                if current_is_pg_search != want_pg_search:
                    index_matches = False

            if column_matches and current_index_type is None:
                logger.info(f"Text search index missing on {table_name}; rebuilding it")
                missing_index_tables.append(table_name)
                continue

            if not (column_matches and index_matches):
                logger.info(
                    f"Text search mismatch on {table_name}: "
                    f"column={current_column_type} (want {target_column_type}), "
                    f"index={current_index_type} (want {target_index_type})"
                )
                mismatched_tables.append((table_name, current_column_type, current_index_type, current_is_pg_search))

                # Check if table has data
                row_count = conn.execute(text(f"SELECT COUNT(*) FROM {schema_name}.{table_name}")).scalar()

                if row_count > 0 and not _reconcile_needs_no_backfill(
                    text_search_extension,
                    table_name,
                    current_column_type,
                    current_index_type,
                ):
                    tables_with_data.append((table_name, row_count))
            else:
                logger.debug(f"Text search OK for {table_name}: {current_column_type}/{current_index_type}")

        if missing_index_tables and text_search_extension == "pgroonga":
            _ensure_pgroonga_extension(conn)
        for table_name in missing_index_tables:
            _create_text_search_index(conn, schema_name, table_name, text_search_extension, pg_search_tokenizer)
        if missing_index_tables:
            conn.commit()

        # If no mismatches, we're done
        if not mismatched_tables:
            logger.debug(f"All text search columns/indexes match configured extension: {text_search_extension}")
            return

        # If there's data in any mismatched table, raise error
        if tables_with_data:
            table_list = ", ".join([f"{table}({count} rows)" for table, count in tables_with_data])
            # Detect current extension from column type, index type, and (for the
            # text/bm25 ambiguity) the key_field reloption. tsvector is
            # unambiguous; text could be pg_textsearch, pgroonga, or pg_search.
            current_col_type = mismatched_tables[0][1]
            current_idx_type = mismatched_tables[0][2]
            first_is_pg_search = mismatched_tables[0][3]
            if current_col_type == "tsvector":
                current_ext = "native"
            elif current_col_type == "bm25vector":
                current_ext = "vchord"
            elif current_col_type == "text" and current_idx_type == "pgroonga":
                current_ext = "pgroonga"
            elif current_col_type == "text":
                current_ext = "pg_search" if first_is_pg_search else "pg_textsearch"
            else:
                current_ext = "unknown"
            raise RuntimeError(
                f"Cannot change text search extension from {current_ext} to {text_search_extension}: "
                f"the following tables contain data: {table_list}. "
                f"To change text search extension, you must either:\n"
                f"  1. Clear all data: DELETE FROM {schema_name}.memory_units; "
                f"DELETE FROM {schema_name}.mental_models; then restart\n"
                f"  2. Use the current text search extension (set HINDSIGHT_API_TEXT_SEARCH_EXTENSION='{current_ext}')"
            )

        # Tables are empty, except for the backfill-free mental-model
        # native-to-pgroonga transition admitted above.
        #
        # Every statement below is written to be safely re-executable: replicas
        # boot concurrently during a rolling restart and each runs this
        # reconciliation, so a plain CREATE/ADD would crash whichever replica
        # loses the race to the first one's committed DDL.
        logger.info(f"Recreating text search columns/indexes for {text_search_extension}")

        for table_name, current_col_type, current_idx_type, _was_pg_search in mismatched_tables:
            # Drop existing index if it exists
            if current_idx_type:
                logger.info(f"Dropping {current_idx_type} index on {table_name}")
                conn.execute(
                    text(f"""
                        DROP INDEX IF EXISTS {schema_name}.idx_{table_name.replace(".", "_")}_text_search
                    """)
                )

            # Drop existing column if it exists
            if current_col_type:
                logger.info(f"Dropping {current_col_type} column on {table_name}")
                conn.execute(text(f"ALTER TABLE {schema_name}.{table_name} DROP COLUMN IF EXISTS search_vector"))

            # Create new column with appropriate type
            if text_search_extension == "vchord":
                logger.info(f"Creating bm25vector column on {table_name}")
                # Note: vchord_bm25 extension creates types in bm25_catalog schema
                conn.execute(
                    text(
                        f"ALTER TABLE {schema_name}.{table_name} "
                        f"ADD COLUMN IF NOT EXISTS search_vector bm25_catalog.bm25vector"
                    )
                )
            elif text_search_extension == "pg_textsearch":
                logger.info(f"Creating TEXT column on {table_name}")
                # Dummy TEXT column for consistency (indexes operate on base columns)
                conn.execute(
                    text(f"ALTER TABLE {schema_name}.{table_name} ADD COLUMN IF NOT EXISTS search_vector TEXT")
                )
            elif text_search_extension == "pgroonga":
                _ensure_pgroonga_extension(conn)
                logger.info(f"Creating dummy TEXT search_vector on {table_name} for pgroonga")
                # pgroonga indexes the base text columns directly, but we keep a
                # dummy search_vector column for symmetry with pg_textsearch and
                # so the column-type mismatch detection above keeps working.
                conn.execute(
                    text(f"ALTER TABLE {schema_name}.{table_name} ADD COLUMN IF NOT EXISTS search_vector TEXT")
                )
            elif text_search_extension == "pg_search":
                logger.info(f"Creating TEXT column on {table_name}")
                # Dummy TEXT column for schema symmetry; pg_search indexes operate on base columns.
                conn.execute(
                    text(f"ALTER TABLE {schema_name}.{table_name} ADD COLUMN IF NOT EXISTS search_vector TEXT")
                )
            else:  # native
                logger.info(f"Creating tsvector column on {table_name}")
                if table_name == "mental_models":
                    # No write path populates mental_models.search_vector for
                    # native (pg_search_vector_expr passes native_inline=False),
                    # so it must be GENERATED exactly like the learnings /
                    # pinned_reflections migration creates it — a plain column
                    # here would stay NULL and silently empty knowledge search.
                    # The 'english' config is hard-coded there and in
                    # knowledge_bm25_arm's native branch; keep all three in step.
                    conn.execute(
                        text(f"""
                            ALTER TABLE {schema_name}.{table_name}
                            ADD COLUMN IF NOT EXISTS search_vector tsvector
                            GENERATED ALWAYS AS (
                                to_tsvector('english', {mental_models_text_document()})
                            ) STORED
                        """)
                    )
                else:
                    # memory_units writes populate this plain column with the
                    # configured native language in ops_postgresql.
                    conn.execute(
                        text(f"ALTER TABLE {schema_name}.{table_name} ADD COLUMN IF NOT EXISTS search_vector tsvector")
                    )

            _create_text_search_index(conn, schema_name, table_name, text_search_extension, pg_search_tokenizer)

        conn.commit()
        logger.info(f"Successfully migrated text search to {text_search_extension}")


def _migrate_one_schema_pg(
    database_url: str,
    schema: str,
    *,
    migration_database_url: str | None,
    embedding_dimension: int | None,
    vector_extension: str,
    text_search_extension: str,
    pg_search_tokenizer: str | None,
    ensure_extensions: bool,
    store_owned_memories: bool = False,
) -> str:
    """Run migrations + post-migration extension setup for a SINGLE PG schema.

    Module-level (not a closure) so it is picklable and can run inside a
    ``ProcessPoolExecutor`` worker. The steps run strictly in order — this is
    the per-tenant sequential unit; parallelism happens only *across* schemas.
    Returns the schema name on success; raises on the first failing step so the
    caller can attribute the failure back to this schema.
    """
    run_migrations(database_url, schema=schema, migration_database_url=migration_database_url)
    if embedding_dimension is not None:
        ensure_embedding_dimension(
            database_url,
            embedding_dimension,
            schema=schema,
            vector_extension=vector_extension,
            store_owned_memories=store_owned_memories,
        )
    if ensure_extensions:
        ensure_vector_extension(
            database_url,
            vector_extension=vector_extension,
            schema=schema,
            store_owned_memories=store_owned_memories,
        )
        ensure_text_search_extension(
            database_url,
            text_search_extension=text_search_extension,
            schema=schema,
            pg_search_tokenizer=pg_search_tokenizer,
            store_owned_memories=store_owned_memories,
        )
    return schema


def _make_migration_executor(max_workers: int):
    """Build the executor that runs per-schema migrations in parallel.

    Each schema must run in its OWN process — Alembic's ``command.upgrade()``
    uses non-thread-safe module globals (serialized in-process by
    ``_alembic_lock``), so a thread pool would not actually run two upgrades at
    once. ``spawn`` gives every worker a clean interpreter on all platforms,
    avoiding the fork-of-a-multithreaded-process deadlock hazard (the API server
    holds threads/pools when migrations run on startup).

    Factored out so tests can substitute an in-process executor.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    return ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context("spawn"))


def run_migrations_for_schemas(
    database_url: str,
    schemas: list[str],
    *,
    concurrency: int = 1,
    migration_database_url: str | None = None,
    embedding_dimension: int | None = None,
    vector_extension: str = "pgvector",
    text_search_extension: str = "native",
    pg_search_tokenizer: str | None = None,
    ensure_extensions: bool = True,
    store_owned_memories: bool = False,
) -> None:
    """Run PostgreSQL migrations for many schemas, up to ``concurrency`` at once.

    Within a schema the work is always sequential (migrate → embedding dim →
    vector ext → text-search ext). Across schemas, when ``concurrency > 1`` each
    schema is migrated in its OWN process: Alembic's ``command.upgrade()`` relies
    on non-thread-safe module-level globals (serialized in-process by
    ``_alembic_lock``), so threads would gain nothing — separate interpreters
    each get a clean Alembic context. Per-schema advisory locks
    (``_get_schema_lock_id``) keep concurrent processes from colliding on the
    same schema across replicas.

    ``database_url`` must already be resolved (e.g. an embedded ``pg0`` instance
    started in the parent) — workers receive it verbatim and only connect.

    Failures are collected per schema and re-raised together so one bad tenant
    does not hide the status of the others.

    ``store_owned_memories`` is set when a custom memories store owns the memory rows and
    answers the mental-model vector search. The post-migration reconcile then stays off
    ``memory_units`` (always empty) and keeps ``mental_models.embedding`` without a vector
    index (never queried by vector). Maintaining either only fails boots for no reason,
    e.g. on pgvector's 2000-dimension HNSW limit with a model the store handles fine.
    """
    # Isolated: keep psycopg2 (and every sync engine this reaches --
    # ensure_embedding_dimension, the vector and text-search extension helpers) out of
    # the caller's process. One child covers the whole sweep.
    if _should_isolate_migrations():
        _run_in_migration_child(
            "run_migrations_for_schemas",
            {
                "database_url": database_url,
                "schemas": schemas,
                "concurrency": concurrency,
                "migration_database_url": migration_database_url,
                "embedding_dimension": embedding_dimension,
                "vector_extension": vector_extension,
                "text_search_extension": text_search_extension,
                "pg_search_tokenizer": pg_search_tokenizer,
                "ensure_extensions": ensure_extensions,
                "store_owned_memories": store_owned_memories,
            },
        )
        return

    if not schemas:
        return

    if is_oracle_url(database_url):
        # Oracle has none of the PG extension/index reconcile steps below, which used to run
        # anyway and failed on Oracle, and Alembic runs sequentially as on API startup.
        # "public" is PG's default schema; on Oracle it means the connecting user's own schema.
        for schema in schemas:
            oracle_schema = None if schema == "public" else schema
            run_migrations(database_url, schema=oracle_schema, migration_database_url=migration_database_url)
            if embedding_dimension is not None:
                ensure_embedding_dimension(
                    migration_database_url or database_url,
                    embedding_dimension,
                    schema=oracle_schema,
                    store_owned_memories=store_owned_memories,
                )
        return

    # A kwargs BAG, assembled conditionally below. Inferred, its value type is the union of
    # everything in it, so the `**` unpack is checked as if every key could be every type --
    # one diagnostic per parameter of the callee, none of them real.
    worker_kwargs: dict[str, Any] = dict(
        migration_database_url=migration_database_url,
        embedding_dimension=embedding_dimension,
        vector_extension=vector_extension,
        text_search_extension=text_search_extension,
        pg_search_tokenizer=pg_search_tokenizer,
        ensure_extensions=ensure_extensions,
        store_owned_memories=store_owned_memories,
    )

    effective = max(1, min(concurrency, len(schemas)))
    if effective == 1:
        # Inline, in-process — no subprocess overhead for the common single
        # tenant / sequential case (and keeps embedded pg0 dev simple).
        for schema in schemas:
            _migrate_one_schema_pg(database_url, schema, **worker_kwargs)
        return

    logger.info("Migrating %d schema(s) with concurrency=%d", len(schemas), effective)
    errors: dict[str, BaseException] = {}
    with _make_migration_executor(effective) as executor:
        futures = {
            executor.submit(_migrate_one_schema_pg, database_url, schema, **worker_kwargs): schema for schema in schemas
        }
        for future in futures:
            schema = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 — aggregate per-schema, re-raise below
                errors[schema] = exc
                logger.error("Migration failed for schema '%s': %s", schema, exc)

    if errors:
        failed = ", ".join(sorted(errors))
        raise RuntimeError(
            f"Database migrations failed for {len(errors)} of {len(schemas)} schema(s): {failed}"
        ) from next(iter(errors.values()))


def _main() -> None:
    """Entry point for the migration subprocess (see ``_run_in_migration_child``).

    Invoked as ``python -m hindsight_api.migrations`` with the JSON payload on stdin.
    Kept deliberately thin: it exists only so the psycopg2 import happens in a
    throwaway process, and it re-enters ``run_migrations`` with ``_CHILD_MARKER``
    set so the subprocess branch is skipped.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    payload = json.loads(sys.stdin.read())
    os.environ[_CHILD_MARKER] = "1"
    targets = {
        "run_migrations": run_migrations,
        "run_migrations_for_schemas": run_migrations_for_schemas,
    }
    target = payload["target"]
    if target not in targets:
        raise SystemExit(f"unknown migration target {target!r}; expected one of {sorted(targets)}")
    targets[target](**payload["kwargs"])


if __name__ == "__main__":
    _main()
