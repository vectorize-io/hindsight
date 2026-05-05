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
import logging
import os
import threading
import time
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script.revision import ResolutionError
from sqlalchemy import Connection, create_engine, text

from ._vector_index import (
    bootstrap_extension,
    detect_vector_extension,
    index_type_keyword,
    index_using_clause,
    minimum_rows_for_index,
    should_defer_index_creation,
    uses_per_bank_vector_indexes,
)
from .db_url import is_oracle_url, to_libpq_url
from .utils import mask_network_location

logger = logging.getLogger(__name__)

# Advisory lock ID for migrations (arbitrary unique number)
MIGRATION_LOCK_ID = 123456789

# Alembic's command.upgrade() is NOT thread-safe: it uses module-level global
# proxies (context._proxy, script) that get overwritten when two threads call
# upgrade() concurrently.  This causes migrations to target the wrong schema
# and crash with "relation already exists" or KeyError: 'script'.
# Serialize all Alembic invocations with a process-level lock.
_alembic_lock = threading.Lock()


def _detect_vector_extension(conn, vector_extension: str = "pgvector") -> str:
    """Validate configured vector extension and preserve Azure DiskANN detection."""
    return detect_vector_extension(conn, vector_extension)


def _drop_per_bank_vector_indexes(conn: Connection, schema_name: str) -> None:
    """Drop per-bank partial memory_units vector indexes after global ScaNN is ready."""
    rows = conn.execute(
        text("""
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = :schema_name
              AND tablename = 'memory_units'
              AND indexname LIKE 'idx_mu_emb_%'
              AND indexdef LIKE '%embedding%'
        """),
        {"schema_name": schema_name},
    ).fetchall()
    safe_schema = schema_name.replace('"', '""')
    for row in rows:
        safe_index = row[0].replace('"', '""')
        conn.execute(text(f'DROP INDEX IF EXISTS "{safe_schema}"."{safe_index}"'))


def _get_schema_lock_id(schema: str) -> int:
    """
    Generate a unique advisory lock ID for a schema.

    Uses hash of schema name to create a deterministic lock ID.
    """
    # Use hash to create a unique lock ID per schema
    # Keep within PostgreSQL's bigint range
    hash_bytes = hashlib.sha256(schema.encode()).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder="big") % (2**31)


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
    alembic_cfg.set_main_option("script_location", script_location)

    # Set the database URL
    alembic_cfg.set_main_option("sqlalchemy.url", database_url)

    # Configure logging (optional, but helps with debugging)
    # Uses Python's logging system instead of alembic.ini
    alembic_cfg.set_main_option("prepend_sys_path", ".")

    # Set path_separator to avoid deprecation warning
    alembic_cfg.set_main_option("path_separator", "os")

    # If targeting a specific schema, pass it to env.py via config
    # env.py will handle setting search_path and version_table_schema
    if schema:
        alembic_cfg.set_main_option("target_schema", schema)

    # Run migrations under a process-level lock.  Alembic uses module-level
    # global proxies that are not thread-safe, so concurrent command.upgrade()
    # calls from different threads corrupt each other's context.
    try:
        with _alembic_lock:
            command.upgrade(alembic_cfg, "heads")
    except ResolutionError as e:
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
        engine = create_engine(migration_url)
        with engine.connect() as conn:
            logger.debug(f"Acquiring migration advisory lock for schema '{schema_name}' (id={lock_id})...")
            while True:
                acquired = conn.execute(text(f"SELECT pg_try_advisory_lock({lock_id})")).scalar()
                if acquired:
                    break
                # Commit the transaction so this connection holds no open snapshot
                # while waiting.  This prevents blocking CREATE INDEX CONCURRENTLY
                # that may be running in the migration worker.
                conn.commit()
                time.sleep(0.5)

            # Commit AFTER acquiring the lock too.  pg_advisory_lock is session-level
            # and survives the COMMIT, but the open transaction on this connection
            # would otherwise block any CREATE INDEX CONCURRENTLY in the migration.
            conn.commit()
            logger.debug("Migration advisory lock acquired")

            try:
                # Ensure pgvector extension is installed globally BEFORE schema migrations
                # This is critical: the extension must exist database-wide before any schema
                # migrations run, otherwise custom schemas won't have access to vector types
                logger.debug("Checking pgvector extension availability...")

                # First, check if extension already exists
                ext_check = conn.execute(
                    text(
                        "SELECT extname, nspname FROM pg_extension e "
                        "JOIN pg_namespace n ON e.extnamespace = n.oid "
                        "WHERE extname = 'vector'"
                    )
                ).fetchone()

                if ext_check:
                    # Extension exists - check if in correct schema
                    ext_schema = ext_check[1]
                    if ext_schema == "public":
                        logger.info("pgvector extension found in public schema - ready to use")
                    else:
                        # Extension in wrong schema - try to fix if we have permissions
                        logger.warning(
                            f"pgvector extension found in schema '{ext_schema}' instead of 'public'. "
                            f"Attempting to relocate..."
                        )
                        try:
                            conn.execute(text("DROP EXTENSION vector CASCADE"))
                            conn.execute(text("SET search_path TO public"))
                            conn.execute(text("CREATE EXTENSION vector"))
                            conn.commit()
                            logger.info("pgvector extension relocated to public schema")
                        except Exception as e:
                            # Failed to relocate - log but don't fail if extension exists somewhere
                            logger.warning(
                                f"Could not relocate pgvector extension to public schema: {e}. "
                                f"Continuing with extension in '{ext_schema}' schema."
                            )
                            conn.rollback()
                else:
                    # Extension doesn't exist - try to install
                    logger.info("pgvector extension not found, attempting to install...")
                    try:
                        conn.execute(text("SET search_path TO public"))
                        conn.execute(text("CREATE EXTENSION vector"))
                        conn.commit()
                        logger.info("pgvector extension installed in public schema")
                    except Exception as e:
                        # Installation failed - this is only fatal if extension truly doesn't exist
                        # Check one more time in case another process installed it
                        conn.rollback()
                        ext_recheck = conn.execute(
                            text(
                                "SELECT nspname FROM pg_extension e "
                                "JOIN pg_namespace n ON e.extnamespace = n.oid "
                                "WHERE extname = 'vector'"
                            )
                        ).fetchone()

                        if ext_recheck:
                            logger.warning(
                                f"Could not install pgvector extension (permission denied?), "
                                f"but extension exists in '{ext_recheck[0]}' schema. Continuing..."
                            )
                        else:
                            # Extension truly doesn't exist and we can't install it
                            logger.error(
                                f"pgvector extension is not installed and cannot be installed: {e}. "
                                f"Please ensure pgvector is installed by a database administrator. "
                                f"See: https://github.com/pgvector/pgvector#installation"
                            )
                            raise RuntimeError(
                                "pgvector extension is required but not installed. "
                                "Please install it with: CREATE EXTENSION vector;"
                            ) from e

                vector_extension = os.getenv("HINDSIGHT_API_VECTOR_EXTENSION", "pgvector").lower()
                bootstrap_extension(conn, vector_extension)

                # Commit any pending transaction on the advisory-lock connection
                # before running migrations.  Some code paths above (e.g., the
                # pgvector extension check) may have started a transaction via
                # SQLAlchemy's autobegin.  If we leave it open, CREATE INDEX
                # CONCURRENTLY inside a migration will deadlock waiting for it.
                conn.commit()

                # Run migrations while holding the lock
                _run_migrations_internal(migration_url, script_location, schema=schema)
            finally:
                # Explicitly release the lock (also released on connection close)
                conn.execute(text(f"SELECT pg_advisory_unlock({lock_id})"))
                logger.debug("Migration advisory lock released")

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


def check_migration_status(
    database_url: str | None = None, script_location: str | None = None
) -> tuple[str | None, str | None]:
    """
    Check current database schema version and latest available version.

    Args:
        database_url: SQLAlchemy database URL. If None, uses HINDSIGHT_API_DATABASE_URL env var.
        script_location: Path to alembic migrations directory. If None, uses default location.

    Returns:
        Tuple of (current_revision, head_revision)
        Returns (None, None) if unable to determine versions
    """
    try:
        from alembic.runtime.migration import MigrationContext
        from alembic.script import ScriptDirectory
        from sqlalchemy import create_engine

        # Get database URL
        if database_url is None:
            database_url = os.getenv("HINDSIGHT_API_DATABASE_URL")
        if not database_url:
            logger.warning(
                "Database URL not provided and HINDSIGHT_API_DATABASE_URL not set, cannot check migration status"
            )
            return None, None

        # Get current revision from database
        engine = create_engine(to_libpq_url(database_url))
        with engine.connect() as connection:
            context = MigrationContext.configure(connection)
            current_rev = context.get_current_revision()

        # Get head revision from migration scripts
        if script_location is None:
            package_dir = Path(__file__).parent
            script_location = str(package_dir / "alembic")

        script_path = Path(script_location)
        if not script_path.exists():
            logger.warning(f"Script location not found at {script_location}")
            return current_rev, None

        # Create config programmatically
        alembic_cfg = Config()
        alembic_cfg.set_main_option("script_location", script_location)
        alembic_cfg.set_main_option("path_separator", "os")

        script = ScriptDirectory.from_config(alembic_cfg)
        head_rev = script.get_current_head()

        return current_rev, head_rev

    except Exception as e:
        logger.warning(f"Unable to check migration status: {e}")
        return None, None


def _migrate_table_embedding_dimension(
    conn: Connection,
    schema_name: str,
    table_name: str,
    required_dimension: int,
    vector_ext: str,
) -> None:
    """
    Migrate the embedding column of a single table to the required dimension.

    - If dimensions match: no action needed
    - If dimensions differ and table is empty: ALTER COLUMN to new dimension
    - If dimensions differ and table has data: raise error with migration guidance
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

    # Drop existing vector index (works for HNSW, DiskANN, vchordrq, and ScaNN)
    conn.execute(
        text(f"""
            DO $$
            DECLARE idx_name TEXT;
            BEGIN
                FOR idx_name IN
                    SELECT indexname FROM pg_indexes
                    WHERE schemaname = '{schema_name}'
                      AND tablename = '{table_name}'
                      AND (indexdef LIKE '%hnsw%' OR indexdef LIKE '%vchordrq%' OR indexdef LIKE '%diskann%' OR indexdef LIKE '%scann%')
                      AND indexdef LIKE '%embedding%'
                LOOP
                    EXECUTE 'DROP INDEX IF EXISTS {schema_name}.' || idx_name;
                END LOOP;
            END $$;
        """)
    )

    conn.execute(
        text(f"ALTER TABLE {schema_name}.{table_name} ALTER COLUMN embedding TYPE vector({required_dimension})")
    )
    conn.commit()

    # Recreate index with appropriate type based on detected extension
    if vector_ext == "pgvector" and required_dimension > 2000:
        raise RuntimeError(
            f"Embedding dimension {required_dimension} exceeds pgvector HNSW index limit of 2000. "
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
    logger.info(f"Created {index_type} index on {table_name} for {required_dimension}-dimensional embeddings")
    conn.commit()

    logger.info(f"Successfully changed {table_name}.embedding dimension to {required_dimension}")


def ensure_embedding_dimension(
    database_url: str,
    required_dimension: int,
    schema: str | None = None,
    vector_extension: str = "pgvector",
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

    Raises:
        RuntimeError: If dimension mismatch with existing data
    """
    schema_name = schema or "public"

    engine = create_engine(to_libpq_url(database_url))
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

        _migrate_table_embedding_dimension(conn, schema_name, "memory_units", required_dimension, vector_ext)
        _migrate_table_embedding_dimension(conn, schema_name, "mental_models", required_dimension, vector_ext)


def ensure_vector_extension(
    database_url: str,
    vector_extension: str = "pgvector",
    schema: str | None = None,
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

    Raises:
        RuntimeError: If extension mismatch with existing data
    """
    schema_name = schema or "public"

    engine = create_engine(to_libpq_url(database_url))
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
            current_index_info = conn.execute(
                text("""
                    SELECT indexdef
                    FROM pg_indexes
                    WHERE schemaname = :schema
                      AND tablename = :table_name
                      AND indexname LIKE :index_pattern
                """),
                {"schema": schema_name, "table_name": table_name, "index_pattern": "%embedding%"},
            ).fetchone()

            if not current_index_info:
                if table_name == "memory_units" and uses_per_bank_vector_indexes(target_ext):
                    # Check whether per-bank partial vector indexes already cover this table
                    # (created by the bank_utils lifecycle — no global index needed in that case)
                    per_bank_index_count = conn.execute(
                        text("""
                            SELECT COUNT(*)
                            FROM pg_indexes
                            WHERE schemaname = :schema
                              AND tablename = :table_name
                              AND indexname LIKE 'idx_mu_emb_%'
                        """),
                        {"schema": schema_name, "table_name": table_name},
                    ).scalar()
                    if per_bank_index_count and per_bank_index_count > 0:
                        logger.debug(
                            f"No global embedding index on {table_name}, but {per_bank_index_count} "
                            f"per-bank partial vector indexes exist — skipping global index creation"
                        )
                        continue
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


def ensure_text_search_extension(
    database_url: str,
    text_search_extension: str = "native",
    schema: str | None = None,
) -> None:
    """
    Ensure the text search columns and indexes match the configured extension.

    This function checks the current search_vector column type and index type
    in the database and adjusts them if necessary:
    - If they match configured extension: no action needed
    - If they differ and tables are empty: drop old column/index, recreate with new type
    - If they differ and tables have data: raise error with migration guidance

    Args:
        database_url: SQLAlchemy database URL
        text_search_extension: Configured text search extension ("native" or "vchord")
        schema: Target PostgreSQL schema name (None for public)

    Raises:
        RuntimeError: If extension mismatch with existing data
    """
    schema_name = schema or "public"

    engine = create_engine(to_libpq_url(database_url))
    with engine.connect() as conn:
        # Tables with search_vector columns to check
        tables_to_check = [
            "memory_units",
            "reflections",  # Renamed from pinned_reflections in p1k2l3m4n5o6 migration
        ]

        # Determine target column type and index type
        if text_search_extension == "vchord":
            target_column_type = "bm25vector"
            target_index_type = "bm25"
        elif text_search_extension == "pg_textsearch":
            target_column_type = "text"
            target_index_type = "bm25"
        else:  # native
            target_column_type = "tsvector"
            target_index_type = "gin"

        mismatched_tables = []
        tables_with_data = []

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
                mismatched_tables.append((table_name, None, None))
                continue

            # Check column type (udt_name contains the actual type: tsvector, bm25vector, etc.)
            current_column_type = current_column_info[1]  # udt_name

            # Get current index type
            current_index_info = conn.execute(
                text("""
                    SELECT am.amname
                    FROM pg_indexes pi
                    JOIN pg_class c ON c.relname = pi.indexname
                    JOIN pg_am am ON am.oid = c.relam
                    WHERE pi.schemaname = :schema
                      AND pi.tablename = :table_name
                      AND pi.indexname LIKE '%text_search%'
                """),
                {"schema": schema_name, "table_name": table_name},
            ).fetchone()

            current_index_type = current_index_info[0] if current_index_info else None

            # Check if column and index types match target
            column_matches = current_column_type == target_column_type
            index_matches = current_index_type == target_index_type if current_index_type else False

            if not (column_matches and index_matches):
                logger.info(
                    f"Text search mismatch on {table_name}: "
                    f"column={current_column_type} (want {target_column_type}), "
                    f"index={current_index_type} (want {target_index_type})"
                )
                mismatched_tables.append((table_name, current_column_type, current_index_type))

                # Check if table has data
                row_count = conn.execute(text(f"SELECT COUNT(*) FROM {schema_name}.{table_name}")).scalar()

                if row_count > 0:
                    tables_with_data.append((table_name, row_count))
            else:
                logger.debug(f"Text search OK for {table_name}: {current_column_type}/{current_index_type}")

        # If no mismatches, we're done
        if not mismatched_tables:
            logger.debug(f"All text search columns/indexes match configured extension: {text_search_extension}")
            return

        # If there's data in any mismatched table, raise error
        if tables_with_data:
            table_list = ", ".join([f"{table}({count} rows)" for table, count in tables_with_data])
            # Detect current extension from column type
            current_col_type = mismatched_tables[0][1]
            if current_col_type == "tsvector":
                current_ext = "native"
            elif current_col_type == "bm25vector":
                current_ext = "vchord"
            elif current_col_type == "text":
                current_ext = "pg_textsearch"
            else:
                current_ext = "unknown"
            raise RuntimeError(
                f"Cannot change text search extension from {current_ext} to {text_search_extension}: "
                f"the following tables contain data: {table_list}. "
                f"To change text search extension, you must either:\n"
                f"  1. Clear all data: DELETE FROM {schema_name}.memory_units; "
                f"DELETE FROM {schema_name}.reflections; then restart\n"
                f"  2. Use the current text search extension (set HINDSIGHT_API_TEXT_SEARCH_EXTENSION='{current_ext}')"
            )

        # Tables are empty, safe to recreate columns/indexes
        logger.info(f"Recreating text search columns/indexes for {text_search_extension}")

        for table_name, current_col_type, current_idx_type in mismatched_tables:
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
                    text(f"ALTER TABLE {schema_name}.{table_name} ADD COLUMN search_vector bm25_catalog.bm25vector")
                )

                # Create BM25 index
                logger.info(f"Creating BM25 index on {table_name}")
                conn.execute(
                    text(f"""
                        CREATE INDEX idx_{table_name.replace(".", "_")}_text_search
                        ON {schema_name}.{table_name}
                        USING bm25 (search_vector bm25_catalog.bm25_ops)
                    """)
                )
            elif text_search_extension == "pg_textsearch":
                logger.info(f"Creating TEXT column on {table_name}")
                # Dummy TEXT column for consistency (indexes operate on base columns)
                conn.execute(text(f"ALTER TABLE {schema_name}.{table_name} ADD COLUMN search_vector TEXT"))

                # Create BM25 index on expression
                logger.info(f"Creating BM25 index on {table_name}")
                # Different expression for each table
                if table_name == "memory_units":
                    index_expr = "(COALESCE(text, '') || ' ' || COALESCE(context, ''))"
                else:  # reflections
                    index_expr = "(COALESCE(name, '') || ' ' || content)"

                conn.execute(
                    text(f"""
                        CREATE INDEX idx_{table_name.replace(".", "_")}_text_search
                        ON {schema_name}.{table_name}
                        USING bm25({index_expr})
                        WITH (text_config='english')
                    """)
                )
            else:  # native
                logger.info(f"Creating tsvector column on {table_name}")
                # Different GENERATED expression for each table
                if table_name == "memory_units":
                    generated_expr = "to_tsvector('english', COALESCE(text, '') || ' ' || COALESCE(context, ''))"
                else:  # reflections
                    generated_expr = "to_tsvector('english', COALESCE(name, '') || ' ' || content)"

                conn.execute(
                    text(f"""
                        ALTER TABLE {schema_name}.{table_name}
                        ADD COLUMN search_vector tsvector
                        GENERATED ALWAYS AS ({generated_expr}) STORED
                    """)
                )

                # Create GIN index
                logger.info(f"Creating GIN index on {table_name}")
                conn.execute(
                    text(f"""
                        CREATE INDEX idx_{table_name.replace(".", "_")}_text_search
                        ON {schema_name}.{table_name}
                        USING gin(search_vector)
                    """)
                )

        conn.commit()
        logger.info(f"Successfully migrated text search to {text_search_extension}")
