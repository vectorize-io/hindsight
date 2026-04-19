"""Fix per-bank vector indexes to match configured extension

Revision ID: a4b5c6d7e8f9
Revises: 2eee35aa3cfc
Create Date: 2026-04-01

Migration d5e6f7a8b9c0 hardcoded HNSW when creating per-bank partial vector
indexes, ignoring HINDSIGHT_API_VECTOR_EXTENSION. Banks that existed when that
migration ran got HNSW indexes even when pgvectorscale (DiskANN) or vchord
was configured.

This migration detects the mismatch and recreates the affected indexes with
the correct type. Skipped entirely when the configured extension is pgvector
(the default), since those indexes are already correct.
"""

import os
from collections.abc import Sequence

from alembic import context, op
from sqlalchemy import text

revision: str = "a4b5c6d7e8f9"
down_revision: str | Sequence[str] | None = "2eee35aa3cfc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FACT_TYPES: dict[str, str] = {
    "world": "worl",
    "experience": "expr",
    "observation": "obsv",
}


def _get_schema_prefix() -> str:
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _target_index_type() -> str | None:
    """Return the target index type, or None if pgvector (no fix needed)."""
    ext = os.getenv("HINDSIGHT_API_VECTOR_EXTENSION", "pgvector").lower()
    if ext == "pgvectorscale":
        return "diskann"
    elif ext == "vchord":
        return "vchordrq"
    return None


def _vector_index_using_clause() -> str:
    """Return the USING clause based on the configured vector extension."""
    ext = os.getenv("HINDSIGHT_API_VECTOR_EXTENSION", "pgvector").lower()
    if ext == "pgvectorscale":
        return "USING diskann (embedding vector_cosine_ops) WITH (num_neighbors = 50)"
    elif ext == "vchord":
        return "USING vchordrq (embedding vector_l2_ops)"
    else:
        return "USING hnsw (embedding vector_cosine_ops)"


def upgrade() -> None:
    target = _target_index_type()
    if target is None:
        # pgvector — indexes are already HNSW, nothing to fix
        return

    bind = op.get_bind()
    schema_name = context.config.get_main_option("target_schema")
    schema = _get_schema_prefix()
    table_ref = f'"{schema_name}".memory_units' if schema_name else "memory_units"
    banks_ref = f'"{schema_name}".banks' if schema_name else "banks"
    using_clause = _vector_index_using_clause()
    pg_schema = schema_name or "public"

    rows = bind.execute(text(f"SELECT bank_id, internal_id FROM {banks_ref}")).fetchall()  # noqa: S608
    for row in rows:
        bank_id = row[0]
        internal_id = str(row[1]).replace("-", "")[:16]
        escaped_bank_id = bank_id.replace("'", "''")
        for ft, ft_short in _FACT_TYPES.items():
            idx_name = f"idx_mu_emb_{ft_short}_{internal_id}"

            # Check if this index exists and what type it is
            idx_info = bind.execute(
                text("SELECT indexdef FROM pg_indexes WHERE schemaname = :schema AND indexname = :idx"),
                {"schema": pg_schema, "idx": idx_name},
            ).fetchone()

            if idx_info is None:
                # Index doesn't exist — create it with the correct type
                bind.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS {idx_name} "
                        f"ON {table_ref} {using_clause} "
                        f"WHERE fact_type = '{ft}' AND bank_id = '{escaped_bank_id}'"
                    )
                )
                continue

            indexdef = idx_info[0].lower()
            if target in indexdef:
                # Already the correct type
                continue

            # Wrong type — drop and recreate
            bind.execute(text(f"DROP INDEX IF EXISTS {schema}{idx_name}"))
            bind.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} "
                    f"ON {table_ref} {using_clause} "
                    f"WHERE fact_type = '{ft}' AND bank_id = '{escaped_bank_id}'"
                )
            )


def downgrade() -> None:
    # Downgrade recreates indexes as HNSW (the original hardcoded behavior)
    target = _target_index_type()
    if target is None:
        return

    bind = op.get_bind()
    schema_name = context.config.get_main_option("target_schema")
    schema = _get_schema_prefix()
    table_ref = f'"{schema_name}".memory_units' if schema_name else "memory_units"
    banks_ref = f'"{schema_name}".banks' if schema_name else "banks"

    rows = bind.execute(text(f"SELECT bank_id, internal_id FROM {banks_ref}")).fetchall()  # noqa: S608
    for row in rows:
        bank_id = row[0]
        internal_id = str(row[1]).replace("-", "")[:16]
        escaped_bank_id = bank_id.replace("'", "''")
        for ft, ft_short in _FACT_TYPES.items():
            idx_name = f"idx_mu_emb_{ft_short}_{internal_id}"
            bind.execute(text(f"DROP INDEX IF EXISTS {schema}{idx_name}"))
            bind.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} "
                    f"ON {table_ref} USING hnsw (embedding vector_cosine_ops) "
                    f"WHERE fact_type = '{ft}' AND bank_id = '{escaped_bank_id}'"
                )
            )
