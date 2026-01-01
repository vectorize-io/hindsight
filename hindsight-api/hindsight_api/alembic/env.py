"""
Alembic environment configuration for SQLAlchemy with pgvector.
Uses synchronous psycopg2 driver for migrations to avoid pgbouncer issues.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

from alembic import context

# Import your models here
from hindsight_api.models import Base


# Load environment variables based on HINDSIGHT_API_DATABASE_URL env var or default to local
def load_env():
    """Load environment variables from .env"""
    # Check if HINDSIGHT_API_DATABASE_URL is already set (e.g., by CI/CD)
    if os.getenv("HINDSIGHT_API_DATABASE_URL"):
        return

    # Look for .env file in the parent directory (root of the workspace)
    root_dir = Path(__file__).parent.parent.parent
    env_file = root_dir / ".env"

    if env_file.exists():
        load_dotenv(env_file)


load_env()

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Note: We don't call fileConfig() here to avoid overriding the application's logging configuration.
# Alembic will use the existing logging configuration from the application.

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_database_url() -> str:
    """
    Get and process the database URL from config or environment.

    Returns the URL with the correct driver (psycopg2) for migrations.
    """
    # Get database URL from config (set programmatically) or environment
    database_url = config.get_main_option("sqlalchemy.url")
    if not database_url:
        database_url = os.getenv("HINDSIGHT_API_DATABASE_URL")
        if not database_url:
            raise ValueError(
                "Database URL not found. "
                "Set HINDSIGHT_API_DATABASE_URL environment variable or pass database_url to run_migrations()."
            )

    # For migrations, use psycopg2 (sync driver) to avoid pgbouncer prepared statement issues
    if database_url.startswith("postgresql+asyncpg://"):
        database_url = database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    elif database_url.startswith("postgres+asyncpg://"):
        database_url = database_url.replace("postgres+asyncpg://", "postgresql://", 1)

    # Update config with processed URL for engine_from_config to use
    config.set_main_option("sqlalchemy.url", database_url)

    return database_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    logging.info("running offline")
    database_url = get_database_url()

    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode with synchronous engine."""
    from sqlalchemy import event, text

    get_database_url()  # Process and set the database URL in config

    # Check if we're targeting a specific schema (for multi-tenant isolation)
    target_schema = config.get_main_option("target_schema")

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    # Add event listener to ensure connection is in read-write mode
    # This is needed for Supabase which may start connections in read-only mode
    @event.listens_for(connectable, "connect")
    def set_read_write_mode(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ WRITE")
        # If targeting a specific schema, set search_path
        # Include public in search_path for access to shared extensions (pgvector)
        if target_schema:
            cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{target_schema}"')
            cursor.execute(f'SET search_path TO "{target_schema}", public')
        cursor.close()

    with connectable.connect() as connection:
        # Also explicitly set read-write mode on this connection
        connection.execute(text("SET SESSION CHARACTERISTICS AS TRANSACTION READ WRITE"))

        # If targeting a specific schema, set search_path
        # Include public in search_path for access to shared extensions (pgvector)
        if target_schema:
            connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{target_schema}"'))
            connection.execute(text(f'SET search_path TO "{target_schema}", public'))

        connection.commit()  # Commit the SET command

        # Configure context with version_table_schema if using a specific schema
        context_opts = {
            "connection": connection,
            "target_metadata": target_metadata,
        }
        if target_schema:
            context_opts["version_table_schema"] = target_schema

        context.configure(**context_opts)

        with context.begin_transaction():
            context.run_migrations()

        # Explicit commit to ensure changes are persisted (especially for Supabase)
        connection.commit()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
