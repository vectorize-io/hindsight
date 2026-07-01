"""Control-plane tables (SQLAlchemy Core).

11 tables that make the control plane the source of truth. Conventions (matching
MemLord): Core only — no ORM relationships, no ``Mapped``. Portable types only
(``String``/``Text``/``Integer``/``Boolean``/``JSON``/``DateTime(timezone=True)``)
so the schema runs unchanged on Postgres and SQLite. Primary keys are app-issued
UUID strings (see ``app.db.ids.new_id``) — no DB-specific UUID type.
"""

from __future__ import annotations

import sqlalchemy as sa

metadata = sa.MetaData()


def _id_col() -> sa.Column:
    return sa.Column("id", sa.String(36), primary_key=True)


def _ts(name: str) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=False)


users = sa.Table(
    "users",
    metadata,
    _id_col(),
    sa.Column("email", sa.String(320), nullable=False, unique=True),
    sa.Column("display_name", sa.String(200)),
    sa.Column("is_operator", sa.Boolean, nullable=False, default=False),
    _ts("created_at"),
)

workspaces = sa.Table(
    "workspaces",
    metadata,
    _id_col(),
    sa.Column("name", sa.String(200), nullable=False),
    sa.Column("owner_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
    _ts("created_at"),
)

workspace_members = sa.Table(
    "workspace_members",
    metadata,
    _id_col(),
    sa.Column("workspace_id", sa.String(36), sa.ForeignKey("workspaces.id"), nullable=False),
    sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
    sa.Column("role", sa.String(32), nullable=False, default="member"),  # owner|admin|member
    _ts("created_at"),
    sa.UniqueConstraint("workspace_id", "user_id", name="uq_ws_member"),
)

source_connectors = sa.Table(
    "source_connectors",
    metadata,
    _id_col(),
    sa.Column("workspace_id", sa.String(36), sa.ForeignKey("workspaces.id"), nullable=False),
    sa.Column("provider", sa.String(64), nullable=False),  # "google_drive"
    sa.Column("status", sa.String(32), nullable=False, default="disconnected"),
    sa.Column("connected_by", sa.String(36), sa.ForeignKey("users.id")),
    # Token material is referenced, never inlined as a secret in app payloads.
    sa.Column("account_email", sa.String(320)),
    sa.Column("config", sa.JSON, nullable=False, default=dict),  # e.g. {"folder_ids": [...]}
    _ts("created_at"),
    _ts("updated_at"),
)

source_documents = sa.Table(
    "source_documents",
    metadata,
    _id_col(),
    sa.Column("workspace_id", sa.String(36), sa.ForeignKey("workspaces.id"), nullable=False),
    sa.Column("connector_id", sa.String(36), sa.ForeignKey("source_connectors.id"), nullable=False),
    sa.Column("provider", sa.String(64), nullable=False),
    sa.Column("external_id", sa.String(256), nullable=False),  # drive_file_id
    sa.Column("name", sa.String(1024)),
    sa.Column("mime_type", sa.String(256)),
    sa.Column("size", sa.Integer),
    sa.Column("web_view_link", sa.String(2048)),
    sa.Column("checksum", sa.String(128)),  # version/hash if available
    sa.Column("trashed", sa.Boolean, nullable=False, default=False),
    sa.Column("enabled", sa.Boolean, nullable=False, default=True),  # operator can disable
    sa.Column("sync_status", sa.String(32), nullable=False, default="discovered"),
    sa.Column("metadata", sa.JSON, nullable=False, default=dict),  # owners, parents, times
    _ts("created_at"),
    _ts("updated_at"),
    sa.UniqueConstraint("connector_id", "external_id", name="uq_doc_external"),
)

source_document_permissions = sa.Table(
    "source_document_permissions",
    metadata,
    _id_col(),
    sa.Column("document_id", sa.String(36), sa.ForeignKey("source_documents.id"), nullable=False),
    sa.Column("external_permission_id", sa.String(256)),
    sa.Column("ptype", sa.String(32), nullable=False),  # user|group|domain|anyone
    sa.Column("role", sa.String(32), nullable=False),  # owner|organizer|writer|commenter|reader
    sa.Column("email_address", sa.String(320)),
    sa.Column("domain", sa.String(256)),
    sa.Column("allow_file_discovery", sa.Boolean),
    sa.Column("expiration_time", sa.DateTime(timezone=True)),
    _ts("snapshot_at"),
)

source_ingestion_jobs = sa.Table(
    "source_ingestion_jobs",
    metadata,
    _id_col(),
    sa.Column("workspace_id", sa.String(36), sa.ForeignKey("workspaces.id"), nullable=False),
    sa.Column("connector_id", sa.String(36), sa.ForeignKey("source_connectors.id"), nullable=False),
    sa.Column("document_id", sa.String(36), sa.ForeignKey("source_documents.id")),
    sa.Column("source", sa.String(64), nullable=False, default="google_drive"),
    sa.Column("external_id", sa.String(256), nullable=False),
    sa.Column("mime_type", sa.String(256)),
    sa.Column("operation", sa.String(32), nullable=False, default="index"),
    # pending|downloading|parsing|chunking|embedding|indexed|skipped|failed
    sa.Column("status", sa.String(32), nullable=False, default="pending"),
    sa.Column("error", sa.Text),
    sa.Column("metadata", sa.JSON, nullable=False, default=dict),
    _ts("created_at"),
    _ts("updated_at"),
)

source_audit_events = sa.Table(
    "source_audit_events",
    metadata,
    _id_col(),
    sa.Column("workspace_id", sa.String(36)),
    sa.Column("actor_id", sa.String(36)),
    sa.Column("source", sa.String(64)),
    sa.Column("source_file_id", sa.String(256)),
    sa.Column("action", sa.String(64), nullable=False),
    sa.Column("status", sa.String(32), nullable=False, default="success"),
    sa.Column("metadata", sa.JSON, nullable=False, default=dict),
    _ts("created_at"),
)

source_sync_state = sa.Table(
    "source_sync_state",
    metadata,
    _id_col(),
    sa.Column("connector_id", sa.String(36), sa.ForeignKey("source_connectors.id"), nullable=False),
    sa.Column("last_sync_at", sa.DateTime(timezone=True)),
    sa.Column("last_status", sa.String(32)),
    sa.Column("cursor", sa.String(512)),  # page token / change cursor for future delta sync
    sa.Column("stats", sa.JSON, nullable=False, default=dict),
    _ts("updated_at"),
    sa.UniqueConstraint("connector_id", name="uq_sync_connector"),
)

agent_activity = sa.Table(
    "agent_activity",
    metadata,
    _id_col(),
    sa.Column("agent_id", sa.String(128), nullable=False),
    sa.Column("workspace_id", sa.String(36)),
    sa.Column("requested_action", sa.String(128), nullable=False),
    sa.Column("target_resource", sa.String(512)),
    sa.Column("decision", sa.String(32), nullable=False),  # allowed|denied|requires_approval
    sa.Column("reason", sa.Text),
    sa.Column("metadata", sa.JSON, nullable=False, default=dict),
    _ts("created_at"),
)

operator_approvals = sa.Table(
    "operator_approvals",
    metadata,
    _id_col(),
    sa.Column("workspace_id", sa.String(36)),
    sa.Column("requested_by", sa.String(128), nullable=False),  # agent_id or user_id
    sa.Column("action", sa.String(128), nullable=False),
    sa.Column("target_resource", sa.String(512)),
    sa.Column("status", sa.String(32), nullable=False, default="pending"),  # pending|approved|denied  # noqa: E501
    sa.Column("decided_by", sa.String(36)),
    sa.Column("reason", sa.Text),
    sa.Column("metadata", sa.JSON, nullable=False, default=dict),
    _ts("created_at"),
    sa.Column("decided_at", sa.DateTime(timezone=True)),
)

memory_source_mappings = sa.Table(
    "memory_source_mappings",
    metadata,
    _id_col(),
    sa.Column("workspace_id", sa.String(36), sa.ForeignKey("workspaces.id"), nullable=False),
    sa.Column("memory_id", sa.String(256), nullable=False),  # ID in memory backend
    sa.Column("memory_backend", sa.String(64), nullable=False),  # memlord|internal|openmemory
    sa.Column("document_id", sa.String(36), sa.ForeignKey("source_documents.id"), nullable=False),
    _ts("created_at"),
    sa.UniqueConstraint("memory_backend", "memory_id", "document_id", name="uq_memory_source"),
)

provider_endpoints = sa.Table(
    "provider_endpoints",
    metadata,
    _id_col(),
    sa.Column("provider_id", sa.String(64), nullable=False, unique=True),  # "localai"
    sa.Column("base_url", sa.String(2048), nullable=False),
    sa.Column("api_key_configured", sa.Boolean, nullable=False, default=False),
    sa.Column("api_style", sa.String(32), nullable=False, default="openai_compatible"),
    sa.Column("enabled", sa.Boolean, nullable=False, default=True),
    sa.Column("health_status", sa.String(16), nullable=False, default="unknown"),  # unknown|healthy|degraded|down
    sa.Column("last_health_check", sa.DateTime(timezone=True)),
    sa.Column("config", sa.JSON, nullable=False, default=dict),
    _ts("created_at"),
    _ts("updated_at"),
)

model_inventory = sa.Table(
    "model_inventory",
    metadata,
    _id_col(),
    sa.Column("provider_id", sa.String(64), nullable=False),          # "localai"
    sa.Column("model_id", sa.String(256), nullable=False),            # "llama-3-8b"
    sa.Column("display_name", sa.String(512)),
    sa.Column("family", sa.String(64)),                                # llama|mistral|gpt|etc
    sa.Column("capabilities", sa.JSON, nullable=False, default=dict), # {chat, embedding, ...}
    sa.Column("context_window", sa.Integer),
    sa.Column("cost_input_per_1m", sa.Float),
    sa.Column("cost_output_per_1m", sa.Float),
    sa.Column("cost_currency", sa.String(8), nullable=False, default="USD"),
    sa.Column("latency_ms", sa.Integer),
    sa.Column("health", sa.String(16), nullable=False, default="unknown"),
    sa.Column("is_active", sa.Boolean, nullable=False, default=True), # still returned by provider
    _ts("first_seen"),
    _ts("last_seen"),
    _ts("updated_at"),
    sa.Column("extra_metadata", sa.JSON, nullable=False, default=dict),
    sa.UniqueConstraint("provider_id", "model_id", name="uq_model_inventory"),
)

router_decisions = sa.Table(
    "router_decisions",
    metadata,
    _id_col(),
    sa.Column("tenant_id", sa.String(36), nullable=False),
    sa.Column("actor_id", sa.String(36), nullable=False),
    sa.Column("request_type", sa.String(32), nullable=False),  # chat|reason|tool|retrieval|voice|other
    sa.Column("selected_model", sa.String(128), nullable=True),  # NULL for no_selection; otherwise gpt-5|claude|...
    sa.Column("candidate_models", sa.JSON, nullable=False, default=list),  # ["gpt-5", "claude"]
    sa.Column("selection_reason", sa.Text),
    sa.Column("latency_ms", sa.Integer),
    sa.Column("estimated_cost", sa.Float),
    sa.Column("fallback_chain", sa.JSON, nullable=False, default=list),  # ["gpt-5", "claude", "ollama"]
    sa.Column("status", sa.String(32), nullable=False),  # selected|fallback|failed|no_selection
    sa.Column("trace_id", sa.String(64)),
    _ts("created_at"),
)

quarantine_items = sa.Table(
    "quarantine_items",
    metadata,
    _id_col(),
    sa.Column("tenant_id", sa.String(36), nullable=False),
    sa.Column("content_hash", sa.String(64), nullable=False),
    sa.Column("content", sa.Text, nullable=False),
    sa.Column("classification", sa.String(50), nullable=False),  # restricted|private
    sa.Column("status", sa.String(50), nullable=False, default="pending"),  # pending|approved|rejected
    sa.Column("created_by", sa.String(36)),
    sa.Column("reason", sa.Text),
    sa.Column("approved_by", sa.String(36)),
    _ts("created_at"),
    sa.Column("approved_at", sa.DateTime(timezone=True)),
)

approval_decisions = sa.Table(
    "approval_decisions",
    metadata,
    _id_col(),
    sa.Column("quarantine_item_id", sa.String(36), sa.ForeignKey("quarantine_items.id"), nullable=False),
    sa.Column("approver_id", sa.String(36), nullable=False),
    sa.Column("decision", sa.String(20), nullable=False),  # approved|rejected
    sa.Column("reason", sa.Text),
    _ts("decided_at"),
)

execution_ledger = sa.Table(
    "execution_ledger",
    metadata,
    _id_col(),
    sa.Column("tenant_id", sa.String(36), nullable=False),
    sa.Column("action_type", sa.String(64), nullable=False),  # docker_deploy|update_env|run_script|etc
    sa.Column("target", sa.String(256), nullable=False),  # resource identifier
    sa.Column("agent_id", sa.String(128), nullable=False),  # who proposed
    sa.Column("agent_role", sa.String(64), nullable=False),  # devops-agent|dev-agent|etc
    sa.Column("status", sa.String(32), nullable=False, default="staged"),  # staged|approved|executing|completed|failed
    sa.Column("risk_level", sa.String(32), nullable=False),  # low|medium|high|critical
    sa.Column("params", sa.JSON, nullable=False, default=dict),  # action parameters (secrets redacted)
    sa.Column("approver_id", sa.String(36)),  # who approved
    sa.Column("approval_note", sa.Text),
    sa.Column("result", sa.JSON),  # execution result {exit_code, output, container_id, etc}
    sa.Column("error_message", sa.Text),  # if failed
    sa.Column("duration_seconds", sa.Float),  # execution time
    _ts("created_at"),
    sa.Column("approved_at", sa.DateTime(timezone=True)),
    sa.Column("started_at", sa.DateTime(timezone=True)),
    sa.Column("completed_at", sa.DateTime(timezone=True)),
)

execution_lineage = sa.Table(
    "execution_lineage",
    metadata,
    _id_col(),
    sa.Column("parent_execution_id", sa.String(36), sa.ForeignKey("execution_ledger.id"), nullable=False),
    sa.Column("child_execution_id", sa.String(36), sa.ForeignKey("execution_ledger.id"), nullable=False),
    sa.Column("relationship", sa.String(32), nullable=False),  # triggered_by|depends_on|rollback_of
    sa.Column("context", sa.JSON),  # additional relationship context
    _ts("created_at"),
)

__all__ = [
    "metadata",
    "users",
    "workspaces",
    "workspace_members",
    "source_connectors",
    "source_documents",
    "source_document_permissions",
    "source_ingestion_jobs",
    "source_audit_events",
    "source_sync_state",
    "agent_activity",
    "operator_approvals",
    "memory_source_mappings",
    "model_inventory",
    "router_decisions",
    "provider_endpoints",
    "quarantine_items",
    "approval_decisions",
    "execution_ledger",
    "execution_lineage",
]
