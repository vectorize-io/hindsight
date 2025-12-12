"""
Database migration management using Alembic.

This module provides programmatic access to run database migrations
on application startup. It is designed to be safe for concurrent
execution using PostgreSQL advisory locks to coordinate between
distributed workers.

Important: All migrations must be backward-compatible to allow
safe rolling deployments.

No alembic.ini required - all configuration is done programmatically.
"""
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

# Advisory lock ID for migrations (arbitrary unique number)
MIGRATION_LOCK_ID = 123456789


def _run_migrations_internal(database_url: str, script_location: str) -> None:
    """
    Internal function to run migrations without locking.
    """
    logger.info(f"Running database migrations to head...")
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

    # Run migrations to head (latest version)
    command.upgrade(alembic_cfg, "head")

    logger.info("Database migrations completed successfully")


def run_migrations(database_url: str, script_location: Optional[str] = None) -> None:
    """
    Run database migrations to the latest version using programmatic Alembic configuration.

    This function is safe to call from multiple distributed workers simultaneously:
    - Uses PostgreSQL advisory lock to ensure only one worker runs migrations at a time
    - Other workers wait for the lock, then verify migrations are complete
    - If schema is already up-to-date, this is a fast no-op

    Args:
        database_url: SQLAlchemy database URL (e.g., "postgresql://user:pass@host/db")
        script_location: Path to alembic migrations directory (e.g., "/path/to/alembic").
                        If None, defaults to hindsight-api/alembic directory.

    Raises:
        RuntimeError: If migrations fail to complete
        FileNotFoundError: If script_location doesn't exist

    Example:
        # Using default location (hindsight_api package)
        run_migrations("postgresql://user:pass@host/db")

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
                f"Alembic script location not found at {script_location}. "
                "Database migrations cannot be run."
            )

        # Use PostgreSQL advisory lock to coordinate between distributed workers
        engine = create_engine(database_url)
        with engine.connect() as conn:
            # pg_advisory_lock blocks until the lock is acquired
            # The lock is automatically released when the connection closes
            logger.debug(f"Acquiring migration advisory lock (id={MIGRATION_LOCK_ID})...")
            conn.execute(text(f"SELECT pg_advisory_lock({MIGRATION_LOCK_ID})"))
            logger.debug("Migration advisory lock acquired")

            try:
                # Run migrations while holding the lock
                _run_migrations_internal(database_url, script_location)
            finally:
                # Explicitly release the lock (also released on connection close)
                conn.execute(text(f"SELECT pg_advisory_unlock({MIGRATION_LOCK_ID})"))
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


def check_migration_status(database_url: Optional[str] = None, script_location: Optional[str] = None) -> tuple[str | None, str | None]:
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
            logger.warning("Database URL not provided and HINDSIGHT_API_DATABASE_URL not set, cannot check migration status")
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
