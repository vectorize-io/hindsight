"""Tests for hindsight-tealtiger governance memory."""

from unittest.mock import MagicMock

from hindsight_tealtiger import HindsightGovernanceMemory


def _make_mock_client():
    client = MagicMock()
    client.retain.return_value = {"status": "ok", "id": "mem-123"}
    client.recall.return_value = [
        {"content": "Governance decision: DENY | Agent: agent-1", "importance": 0.9}
    ]
    client.reflect.return_value = "Agent-1 has been denied 5 times for PII."
    return client


def _make_memory(**kwargs):
    client = _make_mock_client()
    memory = HindsightGovernanceMemory(client=client, **kwargs)
    return memory, client


class TestStore:
    def test_deny_high_importance(self):
        memory, client = _make_memory()
        memory.store({"action": "DENY", "correlation_id": "d1", "agent_id": "a1",
                      "tool_name": "send_email", "reason_codes": ["PII"], "risk_score": 90})
        kw = client.retain.call_args[1]
        assert kw["importance"] == 0.90
        assert "DENY" in kw["content"]

    def test_allow_low_importance(self):
        memory, client = _make_memory()
        memory.store({"action": "ALLOW", "correlation_id": "a1", "tool_name": "search"})
        assert client.retain.call_args[1]["importance"] == 0.55

    def test_monitor_medium_importance(self):
        memory, client = _make_memory()
        memory.store({"action": "MONITOR", "correlation_id": "m1"})
        assert client.retain.call_args[1]["importance"] == 0.70

    def test_custom_importance_fn(self):
        fn = lambda d: 0.99 if d.get("risk_score", 0) > 80 else 0.40
        memory, client = _make_memory(importance_fn=fn)
        memory.store({"action": "DENY", "risk_score": 95, "correlation_id": "c1"})
        assert client.retain.call_args[1]["importance"] == 0.99

    def test_tags_include_action_agent_tool(self):
        memory, client = _make_memory()
        memory.store({"action": "DENY", "correlation_id": "t1",
                      "agent_id": "coder", "tool_name": "shell"})
        tags = client.retain.call_args[1]["tags"]
        assert "action:deny" in tags
        assert "agent:coder" in tags
        assert "tool:shell" in tags

    def test_store_count(self):
        memory, _ = _make_memory()
        memory.store({"action": "ALLOW", "correlation_id": "1"})
        memory.store({"action": "DENY", "correlation_id": "2", "reason_codes": ["X"]})
        assert memory.store_count == 2

    def test_custom_bank_id(self):
        memory, client = _make_memory(bank_id="my-bank")
        memory.store({"action": "ALLOW", "correlation_id": "b1"})
        assert client.retain.call_args[1]["bank_id"] == "my-bank"


class TestRecall:
    def test_recall_by_agent(self):
        memory, client = _make_memory()
        memory.recall(agent_id="agent-1", limit=3)
        kw = client.recall.call_args[1]
        assert "agent-1" in kw["query"]
        assert kw["max_results"] == 3

    def test_recall_with_context(self):
        memory, client = _make_memory()
        memory.recall(agent_id="agent-1", context="tool:send_email")
        assert "tool:send_email" in client.recall.call_args[1]["query"]

    def test_recall_custom_query(self):
        memory, client = _make_memory()
        memory.recall(query="PII denials last week")
        assert client.recall.call_args[1]["query"] == "PII denials last week"

    def test_recall_count(self):
        memory, _ = _make_memory()
        memory.recall(agent_id="a")
        memory.recall(agent_id="b")
        assert memory.recall_count == 2


class TestReflect:
    def test_reflect_on_agent(self):
        memory, client = _make_memory()
        memory.reflect(agent_id="agent-1")
        assert "agent-1" in client.reflect.call_args[1]["query"]

    def test_reflect_custom_query(self):
        memory, client = _make_memory()
        memory.reflect(query="What patterns emerge?")
        assert client.reflect.call_args[1]["query"] == "What patterns emerge?"
