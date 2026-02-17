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

    # Create file_storage table
    op.execute(
        f"""
        CREATE TABLE {schema}file_storage (
            storage_key TEXT PRIMARY KEY,
            data BYTEA NOT NULL,
            content_type TEXT,
            size_bytes BIGINT NOT NULL,
            metadata JSONB DEFAULT '{{}}'::jsonb NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
            access_count INTEGER DEFAULT 0 NOT NULL
        )
    """
    )

    # Index for metadata queries
    op.execute(
        f"""
        CREATE INDEX idx_file_storage_metadata
        ON {schema}file_storage
        USING gin(metadata)
    """
    )

    # Index for cleanup queries (find old/unused files)
    op.execute(
        f"""
        CREATE INDEX idx_file_storage_accessed_at
        ON {schema}file_storage (accessed_at DESC)
    """
    )

    # Add file storage tracking columns to documents table
    op.execute(
        f"""
        ALTER TABLE {schema}documents
        ADD COLUMN IF NOT EXISTS file_storage_key TEXT,
        ADD COLUMN IF NOT EXISTS file_original_name TEXT,
        ADD COLUMN IF NOT EXISTS file_content_type TEXT,
        ADD COLUMN IF NOT EXISTS file_size_bytes BIGINT
    """
    )

    # Foreign key to file_storage (with SET NULL for cleanup)
    op.execute(
        f"""
        ALTER TABLE {schema}documents
        ADD CONSTRAINT fk_documents_file_storage
        FOREIGN KEY (file_storage_key)
        REFERENCES {schema}file_storage(storage_key)
        ON DELETE SET NULL
    """
    )

    # Index for queries by storage key
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_documents_storage_key
        ON {schema}documents (file_storage_key)
        WHERE file_storage_key IS NOT NULL
    """
    )


def downgrade() -> None:
    """Remove file_storage table and related columns."""
    schema = _get_schema_prefix()

    # Drop foreign key constraint
    op.execute(
        f"""
        ALTER TABLE {schema}documents
        DROP CONSTRAINT IF EXISTS fk_documents_file_storage
    """
    )

    # Drop columns from documents table
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_documents_storage_key")
    op.execute(
        f"""
        ALTER TABLE {schema}documents
        DROP COLUMN IF EXISTS file_storage_key,
        DROP COLUMN IF EXISTS file_original_name,
        DROP COLUMN IF EXISTS file_content_type,
        DROP COLUMN IF EXISTS file_size_bytes,
        DROP COLUMN IF EXISTS converted_from_file
    """
    )

    # Drop file_storage table
    op.execute(f"DROP TABLE IF EXISTS {schema}file_storage")
