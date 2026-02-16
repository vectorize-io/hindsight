"""Add RaBitQ quantization support for VectorChord

Revision ID: z1a2b3c4d5e6
Revises: x9s0t1u2v3w4
Create Date: 2025-02-16
"""

import os
from collections.abc import Sequence

from alembic import context, op
from sqlalchemy import text, create_engine

revision: str = "z1a2b3c4d5e6"
down_revision: str | Sequence[str] | None = "x9s0t1u2v3w4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _get_schema_prefix() -> str:
    """Get schema prefix for table names."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def upgrade() -> None:
    schema = _get_schema_prefix()
    vector_ext = os.getenv("HINDSIGHT_API_VECTOR_EXTENSION", "pgvector").lower()
    quantization_enabled = os.getenv(
        "HINDSIGHT_API_VECTOR_QUANTIZATION_ENABLED", "false"
    ).lower() == "true"
    quantization_type = os.getenv(
        "HINDSIGHT_API_VECTOR_QUANTIZATION_TYPE", "rabitq8"
    ).lower()
    embedding_dimension = os.getenv("DEFAULT_EMBEDDING_DIMENSION", "384")

    if not quantization_enabled or vector_ext != "vchord":
        return  # Skip if quantization disabled or not using vchord

    # Validate quantization type
    if quantization_type not in ("rabitq8", "rabitq4"):
        raise ValueError(f"Invalid quantization type: {quantization_type}")

    # Get current embedding dimension from the database if available
    try:
        conn = op.get_bind()
        current_dim = conn.execute(
            text(f"""
                SELECT atttypmod
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = :schema
                  AND c.relname = 'memory_units'
                  AND a.attname = 'embedding'
            """),
            {"schema": schema.strip('".') if schema else "public"},
        ).scalar()
        if current_dim and current_dim > 0:
            embedding_dimension = str(current_dim)
    except Exception:
        # If we can't get dimension from DB, use environment variable or default
        pass

    # Drop existing vector index
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_embedding")

    # Alter embedding column type with dimension specification
    # Use VectorChord's quantization function for proper conversion
    quantize_func = f"quantize_to_{quantization_type}"
    op.execute(f"""
        ALTER TABLE {schema}memory_units
        ALTER COLUMN embedding TYPE {quantization_type}({embedding_dimension})
        USING {quantize_func}(embedding)
    """)

    # Recreate index - dimension is now part of column type definition
    operator_class = f"{quantization_type}_l2_ops"
    op.execute(f"""
        CREATE INDEX idx_memory_units_embedding ON {schema}memory_units
        USING vchordrq (embedding {operator_class})
    """)


def downgrade() -> None:
    schema = _get_schema_prefix()

    # Drop quantized index
    op.execute(f"DROP INDEX IF EXISTS {schema}idx_memory_units_embedding")

    # Restore to float32 vector with default dimension
    embedding_dimension = os.getenv("DEFAULT_EMBEDDING_DIMENSION", "384")

    op.execute(f"""
        ALTER TABLE {schema}memory_units
        ALTER COLUMN embedding TYPE vector({embedding_dimension})
        USING embedding::vector
    """)

    # Recreate original index
    op.execute(f"""
        CREATE INDEX idx_memory_units_embedding ON {schema}memory_units
        USING vchordrq (embedding vector_l2_ops)
    """)

