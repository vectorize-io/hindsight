"""Backfill banks.config from legacy per-column bank config fields

Revision ID: a0b1c2d3e4f5
Revises: 8c6fa6f7230b
Create Date: 2026-04-19

Closes: #1157

`x9s0t1u2v3w4_add_bank_config_column.py` (2026-02-09) introduced
`banks.config JSONB DEFAULT '{}'::jsonb` as the new per-bank override
store, but did not copy existing per-column values into it. Any user
who had bank config set before Feb 2026 and upgraded past
`x9s0t1u2v3w4` silently lost their `retain_mission`,
`observations_mission`, `disposition_*`, and other override values.

This migration is idempotent and defensive:

  * For each legacy override field (the full list from
    `hindsight_api/config.py::BankConfigOverride`), it checks whether
    the corresponding legacy column still exists on `banks`.
  * If the column exists AND the row has a non-null value AND the
    new `banks.config` JSONB does not already carry that key, the
    legacy value is copied into `banks.config`.
  * Downgrade is intentionally a no-op: removing backfilled keys
    from `config` would destroy the only remaining copy of the data.

This migration cannot recover users whose legacy columns were
subsequently dropped after `x9s0t1u2v3w4` shipped. Those users must
restore from a pre-upgrade dump; see issue #1157 for recovery notes.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import context, op

# revision identifiers, used by Alembic.
revision: str = "a0b1c2d3e4f5"
down_revision: str | Sequence[str] | None = "8c6fa6f7230b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Fields previously stored as individual columns on `banks` that now
# live as keys under `banks.config` (JSONB). Sourced from
# `hindsight_api/config.py::BankConfigOverride`.
LEGACY_FIELDS: tuple[str, ...] = (
    "llm_gemini_safety_settings",
    "consolidation_llm_provider",
    "consolidation_llm_api_key",
    "consolidation_llm_model",
    "consolidation_llm_base_url",
    "consolidation_llm_max_concurrent",
    "consolidation_llm_max_retries",
    "consolidation_llm_initial_backoff",
    "consolidation_llm_max_backoff",
    "consolidation_llm_timeout",
    "mcp_enabled_tools",
    "recall_max_concurrent",
    "recall_connection_budget",
    "recall_max_query_tokens",
    "retain_chunk_size",
    "retain_extraction_mode",
    "retain_mission",
    "retain_custom_instructions",
    "retain_default_strategy",
    "retain_strategies",
    "retain_chunk_batch_size",
    "enable_observations",
    "consolidation_batch_size",
    "consolidation_max_memories_per_round",
    "consolidation_llm_batch_size",
    "consolidation_max_tokens",
    "consolidation_source_facts_max_tokens",
    "consolidation_source_facts_max_tokens_per_observation",
    "consolidation_max_attempts",
    "observations_mission",
    "max_observations_per_scope",
    "entity_labels",
    "entities_allow_free_form",
    "reflect_mission",
    "reflect_source_facts_max_tokens",
    "recall_include_chunks",
    "recall_max_tokens",
    "recall_chunks_max_tokens",
    "recall_budget_function",
    "recall_budget_fixed_low",
    "recall_budget_fixed_mid",
    "recall_budget_fixed_high",
    "recall_budget_adaptive_low",
    "recall_budget_adaptive_mid",
    "recall_budget_adaptive_high",
    "recall_budget_min",
    "recall_budget_max",
    "disposition_skepticism",
    "disposition_literalism",
    "disposition_empathy",
)


def _get_schema_prefix() -> str:
    """Get schema prefix for table names (e.g., 'tenant_x.' or '' for public)."""
    schema = context.config.get_main_option("target_schema")
    return f'"{schema}".' if schema else ""


def _get_target_schema() -> str:
    """Get the target schema name (tenant schema or 'public')."""
    schema = context.config.get_main_option("target_schema")
    return schema if schema else "public"


def upgrade() -> None:
    """Backfill banks.config from any surviving legacy override columns."""
    conn = op.get_bind()
    schema_prefix = _get_schema_prefix()
    schema_name = _get_target_schema()

    # Confirm the `banks.config` column exists (added by x9s0t1u2v3w4).
    # If it doesn't, we're running on a pre-x9s0t1u2v3w4 schema — nothing
    # to backfill into.
    config_col_exists = conn.execute(
        sa.text(
            """
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = :schema
              AND table_name = 'banks'
              AND column_name = 'config'
            """
        ),
        {"schema": schema_name},
    ).fetchone()
    if not config_col_exists:
        return

    for field in LEGACY_FIELDS:
        legacy_col_exists = conn.execute(
            sa.text(
                """
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = :schema
                  AND table_name = 'banks'
                  AND column_name = :col
                """
            ),
            {"schema": schema_name, "col": field},
        ).fetchone()
        if not legacy_col_exists:
            # Column was dropped or never existed on this install.
            continue

        # Idempotent copy: only set config[field] when legacy column has
        # a value AND the JSONB key is not already present. `to_jsonb`
        # handles any column type (text, int, float, bool, jsonb, array).
        conn.execute(
            sa.text(
                f"""
                UPDATE {schema_prefix}banks
                SET config = config || jsonb_build_object(:key, to_jsonb({field}))
                WHERE {field} IS NOT NULL
                  AND NOT (config ? :key)
                """
            ),
            {"key": field},
        )


def downgrade() -> None:
    """No-op: removing backfilled keys would destroy the only remaining copy."""
    # Intentionally blank. If you need to undo this migration, do so by
    # restoring from a pre-upgrade database dump.
    pass
