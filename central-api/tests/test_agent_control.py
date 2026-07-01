"""Agent action control decisions (pure decide()) + DB-backed request flow."""

import asyncio

from app.db import repositories as repo
from app.db.engine import init_models, session_scope
from app.governance import agent_control


def test_decide_high_impact_requires_approval():
    for action in ("bulk_sync", "source_deletion", "connector_disconnect",
                   "permission_override", "retrieval_across_restricted_sources"):
        assert agent_control.decide(action).decision == "requires_approval"


def test_decide_allowed_read_only():
    for action in ("list_workspaces", "search_governed_documents", "list_ingestion_jobs"):
        assert agent_control.decide(action).decision == "allowed"


def test_decide_unknown_denied():
    assert agent_control.decide("rm_rf_everything").decision == "denied"


def test_request_action_records_activity_and_approval():
    async def scenario():
        await init_models()
        async with session_scope() as s:
            high = await agent_control.request_action(
                s, agent_id="agent-1", action="bulk_sync", workspace_id="ws1")
            denied = await agent_control.request_action(
                s, agent_id="agent-1", action="unknown_x", workspace_id="ws1")
            allowed = await agent_control.request_action(
                s, agent_id="agent-1", action="list_workspaces", workspace_id="ws1")
        async with session_scope() as s:
            activity = await repo.list_agent_activity(s, workspace_id="ws1")
        return high, denied, allowed, activity

    high, denied, allowed, activity = asyncio.run(scenario())
    assert high["decision"] == "requires_approval" and "approval_id" in high
    assert denied["decision"] == "denied"
    assert allowed["decision"] == "allowed"
    assert len(activity) == 3  # all three recorded for operator visibility
