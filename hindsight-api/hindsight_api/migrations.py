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

from .utils import mask_network_location

logger = logging.getLogger(__name__)

# Advisory lock ID for migrations (arbitrary unique number)
MIGRATION_LOCK_ID = 123456789


def _detect_vector_extension(conn, vector_extension: str = "pgvector") -> str:
    """
    Validate vector extension: 'vchord' or 'pgvector'.

    Args:
        conn: SQLAlchemy connection object
        vector_extension: Configured extension ("pgvector" or "vchord")

    Returns:
        "vchord" or "pgvector"

    Raises:
        RuntimeError: If configured extension is not installed
    """
    # Verify the configured extension is installed
    if vector_extension == "vchord":
        vchord_check = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vchord'")).scalar()
        if not vchord_check:
            raise RuntimeError(
                "Configured vector extension 'vchord' not found. Install it with: CREATE EXTENSION vchord CASCADE;"
            )
        logger.debug("Using configured vector extension: vchord")
        return "vchord"
    elif vector_extension == "pgvector":
        pgvector_check = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).scalar()
        if not pgvector_check:
            raise RuntimeError(
                "Configured vector extension 'pgvector' not found. Install it with: CREATE EXTENSION vector;"
            )
        logger.debug("Using configured vector extension: pgvector")
        return "pgvector"
    else:
        raise ValueError(f"Invalid vector_extension: {vector_extension}. Must be 'pgvector' or 'vchord'")


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
    vector_extension: str = "pgvector",
    quantization_enabled: bool = False,
    quantization_type: str = "rabitq8",
) -> None:
    """
    Ensure the embedding column dimension matches the model's dimension.

    This function checks the current vector column dimension in the database
    and adjusts it if necessary:
    - If dimensions match: no action needed
    - If dimensions differ and table is empty: ALTER COLUMN to new dimension
    - If dimensions differ and table has data: raise error with migration guidance

    Handles both standard vector types and VectorChord quantized types (rabitq4/rabitq8).

    Args:
        database_url: SQLAlchemy database URL
        required_dimension: The embedding dimension required by the model
        schema: Target PostgreSQL schema name (None for public)
        vector_extension: Configured vector extension ("pgvector" or "vchord")
        quantization_enabled: Whether VectorChord quantization is enabled
        quantization_type: Type of quantization ("rabitq8" or "rabitq4")

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

        # Detect which vector extension is available
        vector_ext = _detect_vector_extension(conn, vector_extension)
        logger.info(f"Using vector extension: {vector_ext}")

        # Get current column type and dimension
        # Query both the type name and dimension (atttypmod)
        type_info = conn.execute(
            text("""
                SELECT t.typname, a.atttypmod
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                JOIN pg_type t ON a.atttypid = t.oid
                WHERE n.nspname = :schema
                  AND c.relname = 'memory_units'
                  AND a.attname = 'embedding'
            """),
            {"schema": schema_name},
        ).fetchone()

        if type_info is None:
            logger.warning("Could not determine current embedding column type, skipping check")
            return

        current_type_name = type_info[0]
        current_dim = type_info[1]

        # Determine if column is using quantized type
        is_quantized = current_type_name in ("rabitq4", "rabitq8")

        if current_dim is None:
            logger.warning("Could not determine current embedding dimension, skipping check")
            return

        # pgvector and VectorChord store dimension directly in atttypmod
        current_dimension = current_dim

        logger.info(
            f"Current embedding column: type={current_type_name}, dimension={current_dimension}, "
            f"quantized={is_quantized}"
        )

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

        # Drop existing vector index (works for both HNSW and vchordrq)
        conn.execute(
            text(f"""
                DO $$
                DECLARE idx_name TEXT;
                BEGIN
                    FOR idx_name IN
                        SELECT indexname FROM pg_indexes
                        WHERE schemaname = '{schema_name}'
                          AND tablename = 'memory_units'
                          AND (indexdef LIKE '%hnsw%' OR indexdef LIKE '%vchordrq%')
                          AND indexdef LIKE '%embedding%'
                    LOOP
                        EXECUTE 'DROP INDEX IF EXISTS {schema_name}.' || idx_name;
                    END LOOP;
                END $$;
            """)
        )

        # Determine target column type
        if quantization_enabled and vector_ext == "vchord":
            target_type = f"{quantization_type}({required_dimension})"
            quantize_func = f"quantize_to_{quantization_type}"
            logger.info(f"Converting to quantized type: {target_type}")
        else:
            target_type = f"vector({required_dimension})"
            logger.info(f"Converting to standard vector type: {target_type}")

        # For dimension changes on quantized types, PostgreSQL doesn't support direct casting
        # from rabitq4/rabitq8 to vector. Instead, we drop and recreate the column.
        # This is safe because we've already verified the table is empty above.
        if is_quantized and current_dimension != required_dimension:
            logger.info(
                f"Dropping and recreating quantized column to change dimension "
                f"(PostgreSQL doesn't support rabitq->vector cast)"
            )

            # Drop the embedding column
            conn.execute(text(f"ALTER TABLE {schema_name}.memory_units DROP COLUMN embedding"))

            # Recreate with new dimension and type
            if quantization_enabled and vector_ext == "vchord":
                # Create as quantized type directly
                conn.execute(
                    text(f"ALTER TABLE {schema_name}.memory_units ADD COLUMN embedding {target_type}")
                )
            else:
                # Create as vector type
                conn.execute(
                    text(f"ALTER TABLE {schema_name}.memory_units ADD COLUMN embedding {target_type}")
                )
        else:
            # Standard ALTER COLUMN for non-quantized types or same-dimension conversions
            if quantization_enabled and vector_ext == "vchord":
                # Converting from vector to quantized type
                if not is_quantized:
                    using_clause = f"{quantize_func}(embedding)"
                else:
                    # Same type, just dimension change (shouldn't happen, but handle it)
                    using_clause = "embedding"
            else:
                # Converting to standard vector type
                if is_quantized:
                    # Converting from quantized to vector - use explicit cast
                    # Note: This cast may fail if VectorChord doesn't support it
                    # In that case, the DROP/CREATE path above will be used
                    using_clause = "embedding::vector"
                else:
                    # Already vector, just change dimension
                    using_clause = f"embedding::vector({required_dimension})"

            # Alter the column type
            conn.execute(
                text(
                    f"ALTER TABLE {schema_name}.memory_units "
                    f"ALTER COLUMN embedding TYPE {target_type} "
                    f"USING {using_clause}"
                )
            )

        conn.commit()

        # Recreate index with appropriate type based on configuration
        if vector_ext == "vchord":
            # VectorChord index
            if quantization_enabled:
                # Quantized index
                operator_class = f"{quantization_type}_l2_ops"
                logger.info(f"Creating vchordrq index with {operator_class} for {required_dimension}D embeddings")
            else:
                # Non-quantized VectorChord index
                operator_class = "vector_l2_ops"
                logger.info(f"Creating vchordrq index with {operator_class} for {required_dimension}D embeddings")

            conn.execute(
                text(f"""
                    CREATE INDEX IF NOT EXISTS idx_memory_units_embedding
                    ON {schema_name}.memory_units
                    USING vchordrq (embedding {operator_class})
                """)
            )
        else:  # pgvector
            conn.execute(
                text(f"""
                    CREATE INDEX IF NOT EXISTS idx_memory_units_embedding
                    ON {schema_name}.memory_units
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                """)
            )
            logger.info(f"Created HNSW index for {required_dimension}-dimensional embeddings")

        conn.commit()

        logger.info(f"Successfully changed embedding dimension to {required_dimension}")


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
        vector_extension: Configured vector extension ("pgvector" or "vchord")
        schema: Target PostgreSQL schema name (None for public)

    Raises:
        RuntimeError: If extension mismatch with existing data
    """
    schema_name = schema or "public"

    engine = create_engine(database_url)
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

        # Determine target index type
        target_index_type = "vchordrq" if target_ext == "vchord" else "hnsw"

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
                logger.warning(f"No embedding index found for {table_name}, will create it")
                mismatched_tables.append((table_name, index_name, None))
                continue

            indexdef = current_index_info[0].lower()
            if "vchordrq" in indexdef:
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
                mismatched_tables.append((table_name, index_name, current_index_type))

                # Check if table has data
                row_count = conn.execute(
                    text(f"SELECT COUNT(*) FROM {schema_name}.{table_name} WHERE embedding IS NOT NULL")
                ).scalar()

                if row_count > 0:
                    tables_with_data.append((table_name, row_count))
            else:
                logger.debug(f"Index type OK for {table_name}: {current_index_type}")

        # If no mismatches, we're done
        if not mismatched_tables:
            logger.debug(f"All vector indexes match configured extension: {target_ext}")
            return

        # If there's data in any mismatched table, raise error
        if tables_with_data:
            table_list = ", ".join([f"{table}({count} rows)" for table, count in tables_with_data])
            raise RuntimeError(
                f"Cannot change vector extension from {current_index_type} to {target_index_type}: "
                f"the following tables contain data: {table_list}. "
                f"To change vector extension, you must either:\n"
                f"  1. Re-embed all data: DELETE FROM {schema_name}.memory_units; "
                f"DELETE FROM {schema_name}.learnings; DELETE FROM {schema_name}.pinned_reflections; then restart\n"
                f"  2. Use the current vector extension (set HINDSIGHT_API_VECTOR_EXTENSION='{current_index_type.replace('vchordrq', 'vchord').replace('hnsw', 'pgvector')}')"
            )

        # Tables are empty, safe to recreate indexes
        logger.info(f"Recreating vector indexes for {target_ext}")

        for table_name, index_name, current_type in mismatched_tables:
            # Drop existing index if it exists
            if current_type:
                logger.info(f"Dropping {current_type} index on {table_name}")
                conn.execute(text(f"DROP INDEX IF EXISTS {schema_name}.{index_name}"))

            # Create new index with appropriate type
            if target_ext == "vchord":
                logger.info(f"Creating vchordrq index on {table_name}")
                # VectorChord requires dimension type modifier for quantized indexes
                # The dimension is specified in the column type modifier, not the index
                conn.execute(
                    text(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON {schema_name}.{table_name}
                        USING vchordrq (embedding vector_l2_ops)
                    """)
                )
            else:  # pgvector
                logger.info(f"Creating HNSW index on {table_name}")
                conn.execute(
                    text(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON {schema_name}.{table_name}
                        USING hnsw (embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = 64)
                    """)
                )

        conn.commit()
        logger.info(f"Successfully migrated vector indexes to {target_ext}")


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

    engine = create_engine(database_url)
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
