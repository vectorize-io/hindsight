"""Agent action control (Phase 12).

Agents *request* actions; the Central API decides. High-impact actions require
operator approval in v0.1. Every request is recorded as agent_activity (visible
in the GUI) and an audit event.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.connectors.google_drive import audit_actions as A
from app.db import repositories as repo

# Actions an agent may never auto-execute in v0.1 — operator must approve.
HIGH_IMPACT_ACTIONS = frozenset({
    "bulk_sync",
    "source_deletion",
    "permission_override",
    "connector_disconnect",
    "retrieval_across_restricted_sources",
})

# Safe read-only actions agents may perform directly.
ALLOWED_ACTIONS = frozenset({
    "list_workspaces",
    "list_connected_sources",
    "list_source_documents",
    "list_ingestion_jobs",
    "get_source_audit",
    "search_governed_documents",
    "sync_source",
})


@dataclass(frozen=True)
class AgentDecision:
    decision: str  # allowed|denied|requires_approval
    reason: str


def decide(action: str) -> AgentDecision:
    if action in HIGH_IMPACT_ACTIONS:
        return AgentDecision("requires_approval", "high_impact_action")
    if action in ALLOWED_ACTIONS:
        return AgentDecision("allowed", "read_only_governed")
    return AgentDecision("denied", "unknown_action")  # fail closed


async def request_action(session: AsyncSession, *, agent_id: str, action: str,
                         workspace_id: str | None = None, target_resource: str | None = None,
                         metadata: dict | None = None) -> dict:
    """Record an agent action request, apply the decision, and audit it."""
    d = decide(action)
    await repo.record_agent_activity(
        session, agent_id=agent_id, requested_action=action, decision=d.decision,
        workspace_id=workspace_id, target_resource=target_resource, reason=d.reason,
        metadata=metadata or {},
    )
    await repo.write_audit(session, action=A.AGENT_ACTION_REQUESTED, actor_id=agent_id,
                           workspace_id=workspace_id, source="agent",
                           metadata={"action": action, "decision": d.decision})

    if d.decision == "requires_approval":
        approval = await repo.create_approval(
            session, requested_by=agent_id, action=action, workspace_id=workspace_id,
            target_resource=target_resource, metadata=metadata or {},
        )
        return {"decision": d.decision, "reason": d.reason, "approval_id": approval["id"]}
    if d.decision == "denied":
        await repo.write_audit(session, action=A.AGENT_ACTION_DENIED, actor_id=agent_id,
                               workspace_id=workspace_id, source="agent",
                               metadata={"action": action})
    return {"decision": d.decision, "reason": d.reason}
