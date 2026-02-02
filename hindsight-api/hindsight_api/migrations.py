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
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script.revision import ResolutionError
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

# Advisory lock ID for migrations (arbitrary unique number)
MIGRATION_LOCK_ID = 123456789


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
    logger.info(f"Database URL: {database_url}")
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

    # Run migrations
    try:
        command.upgrade(alembic_cfg, "head")
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

        # Use schema-specific lock ID for multi-tenant isolation
        lock_id = _get_schema_lock_id(schema) if schema else MIGRATION_LOCK_ID
        schema_name = schema or "public"

        # Use PostgreSQL advisory lock to coordinate between distributed workers
        engine = create_engine(database_url)
        with engine.connect() as conn:
            # pg_advisory_lock blocks until the lock is acquired
            # The lock is automatically released when the connection closes
            logger.debug(f"Acquiring migration advisory lock for schema '{schema_name}' (id={lock_id})...")
            conn.execute(text(f"SELECT pg_advisory_lock({lock_id})"))
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

                # Run migrations while holding the lock
                _run_migrations_internal(database_url, script_location, schema=schema)
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
        engine = create_engine(database_url)
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


def ensure_embedding_dimension(
    database_url: str,
    required_dimension: int,
    schema: str | None = None,
) -> None:
    """
    Ensure the embedding column dimension matches the model's dimension.

    This function checks the current vector column dimension in the database
    and adjusts it if necessary:
    - If dimensions match: no action needed
    - If dimensions differ and table is empty: ALTER COLUMN to new dimension
    - If dimensions differ and table has data: raise error with migration guidance

    Args:
        database_url: SQLAlchemy database URL
        required_dimension: The embedding dimension required by the model
        schema: Target PostgreSQL schema name (None for public)

    Raises:
        RuntimeError: If dimension mismatch with existing data
    """
    schema_name = schema or "public"

    engine = create_engine(database_url)
    with engine.connect() as conn:
        # Check if memory_units table exists
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

        # Get current column dimension from pg_attribute
        # pgvector stores dimension in atttypmod
        current_dim = conn.execute(
            text("""
                SELECT atttypmod
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = :schema
                  AND c.relname = 'memory_units'
                  AND a.attname = 'embedding'
            """),
            {"schema": schema_name},
        ).scalar()

        if current_dim is None:
            logger.warning("Could not determine current embedding dimension, skipping check")
            return

        # pgvector stores dimension directly in atttypmod (no offset like other types)
        current_dimension = current_dim

        if current_dimension == required_dimension:
            logger.debug(f"Embedding dimension OK: {current_dimension}")
            return

        logger.info(
            f"Embedding dimension mismatch: database has {current_dimension}, model requires {required_dimension}"
        )

        # Check if table has data
        row_count = conn.execute(
            text(f"SELECT COUNT(*) FROM {schema_name}.memory_units WHERE embedding IS NOT NULL")
        ).scalar()

        if row_count > 0:
            raise RuntimeError(
                f"Cannot change embedding dimension from {current_dimension} to {required_dimension}: "
                f"memory_units table contains {row_count} rows with embeddings. "
                f"To change dimensions, you must either:\n"
                f"  1. Re-embed all data: DELETE FROM {schema_name}.memory_units; then restart\n"
                f"  2. Use a model with {current_dimension}-dimensional embeddings"
            )

        # Table is empty, safe to alter column
        logger.info(f"Altering embedding column dimension from {current_dimension} to {required_dimension}")

        # Drop the HNSW index on embedding column if it exists
        # Only drop indexes that use 'hnsw' and reference the 'embedding' column
        conn.execute(
            text(f"""
                DO $$
                DECLARE idx_name TEXT;
                BEGIN
                    FOR idx_name IN
                        SELECT indexname FROM pg_indexes
                        WHERE schemaname = '{schema_name}'
                          AND tablename = 'memory_units'
                          AND indexdef LIKE '%hnsw%'
                          AND indexdef LIKE '%embedding%'
                    LOOP
                        EXECUTE 'DROP INDEX IF EXISTS {schema_name}.' || idx_name;
                    END LOOP;
                END $$;
            """)
        )

        # Alter the column type
        conn.execute(
            text(f"ALTER TABLE {schema_name}.memory_units ALTER COLUMN embedding TYPE vector({required_dimension})")
        )
        conn.commit()

        # Recreate the HNSW index
        conn.execute(
            text(f"""
                CREATE INDEX IF NOT EXISTS idx_memory_units_embedding_hnsw
                ON {schema_name}.memory_units
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)
        )
        conn.commit()

        logger.info(f"Successfully changed embedding dimension to {required_dimension}")
