"""Add file_storage table for BYTEA-based file storage

Revision ID: a1b2c3d4e5f6
Revises: y0t1u2v3w4x5
Create Date: 2026-02-16

Creates a dedicated table for storing uploaded files using BYTEA.
This provides zero-config file storage that "just works" for development
and small deployments. For production/scale, use S3-compatible storage.

Files are stored in a separate table to avoid bloating the documents table.
"""

from collections.abc import Sequence

from alembic import context, op

revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "y0t1u2v3w4x5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (required for multi-tenant support)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    """Create file_storage table for BYTEA storage."""
    schema = _get_schema_prefix()

    # Create file_storage table (minimal: just key + data)
    op.execute(
        f"""
        CREATE TABLE {schema}file_storage (
            storage_key TEXT PRIMARY KEY,
            data BYTEA NOT NULL
        )
    """
    )

    # Add file tracking columns to documents table
    op.execute(
        f"""
        ALTER TABLE {schema}documents
        ADD COLUMN IF NOT EXISTS file_storage_key TEXT,
        ADD COLUMN IF NOT EXISTS file_original_name TEXT,
        ADD COLUMN IF NOT EXISTS file_content_type TEXT
    """
    )


def downgrade() -> None:
    """Remove file_storage table and related columns."""
    schema = _get_schema_prefix()

    # Drop columns from documents table
    op.execute(
        f"""
        ALTER TABLE {schema}documents
        DROP COLUMN IF EXISTS file_storage_key,
        DROP COLUMN IF EXISTS file_original_name,
        DROP COLUMN IF EXISTS file_content_type
    """
    )

    # Drop file_storage table
    op.execute(f"DROP TABLE IF EXISTS {schema}file_storage")
