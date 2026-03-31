from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hindsight_agentcore import (
    HindsightRuntimeAdapter,
    RecallPolicy,
    RetentionPolicy,
    TurnContext,
    configure,
    reset_config,
)


def _ctx(**kwargs):
    defaults = dict(
        runtime_session_id="sess-123",
        user_id="user-456",
        agent_name="support-agent",
        tenant_id="acme",
        request_id="req-789",
    )
    return TurnContext(**{**defaults, **kwargs})


def _make_recall_result(text="Prior context", type_="observation"):
    r = MagicMock()
    r.text = text
    r.type = type_
    r.mentioned_at = None
    return r


def _make_recall_response(*texts):
    resp = MagicMock()
    resp.results = [_make_recall_result(t) for t in texts]
    return resp


class TestBeforeTurn:
    def setup_method(self):
        reset_config()
        configure(hindsight_api_url="http://fake:9077")

    def teardown_method(self):
        reset_config()

    def _make_adapter(self, **kwargs):
        adapter = HindsightRuntimeAdapter(**kwargs)
        mock_client = MagicMock()
        adapter._local.client = mock_client
        return adapter, mock_client

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty_string(self):
        adapter, mock_client = self._make_adapter()
        result = await adapter.before_turn(_ctx(), query="   ")
        assert result == ""
        mock_client.recall.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_formatted_memories(self):
        adapter, mock_client = self._make_adapter()
        mock_client.recall.return_value = _make_recall_response(
            "User prefers Postgres", "Rate limiting blocked by Priya"
        )
        result = await adapter.before_turn(_ctx(), query="project status")
        assert "User prefers Postgres" in result
        assert "Rate limiting blocked by Priya" in result

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_results(self):
        adapter, mock_client = self._make_adapter()
        resp = MagicMock()
        resp.results = []
        mock_client.recall.return_value = resp
        result = await adapter.before_turn(_ctx(), query="anything")
        assert result == ""

    @pytest.mark.asyncio
    async def test_gracefully_degrades_on_exception(self):
        adapter, mock_client = self._make_adapter()
        mock_client.recall.side_effect = ConnectionError("unreachable")
        result = await adapter.before_turn(_ctx(), query="anything")
        assert result == ""

    @pytest.mark.asyncio
    async def test_uses_correct_bank_id(self):
        adapter, mock_client = self._make_adapter()
        mock_client.recall.return_value = _make_recall_response()
        await adapter.before_turn(_ctx(tenant_id="acme"), query="test query")
        call_kwargs = mock_client.recall.call_args[1]
        assert call_kwargs["bank_id"] == "tenant:acme:user:user-456:agent:support-agent"

    @pytest.mark.asyncio
    async def test_session_id_not_in_bank_id(self):
        adapter, mock_client = self._make_adapter()
        mock_client.recall.return_value = _make_recall_response()
        await adapter.before_turn(_ctx(), query="test")
        call_kwargs = mock_client.recall.call_args[1]
        assert "sess-123" not in call_kwargs["bank_id"]

    @pytest.mark.asyncio
    async def test_uses_recall_policy_budget(self):
        adapter, mock_client = self._make_adapter(
            recall_policy=RecallPolicy(budget="high", max_tokens=2000)
        )
        mock_client.recall.return_value = _make_recall_response()
        await adapter.before_turn(_ctx(), query="test")
        call_kwargs = mock_client.recall.call_args[1]
        assert call_kwargs["budget"] == "high"
        assert call_kwargs["max_tokens"] == 2000

    @pytest.mark.asyncio
    async def test_reflect_mode_calls_reflect(self):
        adapter, mock_client = self._make_adapter(
            recall_policy=RecallPolicy(mode="reflect")
        )
        reflect_resp = MagicMock()
        reflect_resp.answer = "Synthesized context about the user."
        mock_client.reflect.return_value = reflect_resp
        result = await adapter.before_turn(_ctx(), query="what should I prioritize?")
        mock_client.recall.assert_not_called()
        mock_client.reflect.assert_called_once()
        assert "Synthesized context" in result


class TestAfterTurn:
    def setup_method(self):
        reset_config()
        configure(hindsight_api_url="http://fake:9077", retain_async=False)

    def teardown_method(self):
        reset_config()

    def _make_adapter(self, **kwargs):
        adapter = HindsightRuntimeAdapter(**kwargs)
        mock_client = MagicMock()
        adapter._local.client = mock_client
        return adapter, mock_client

    @pytest.mark.asyncio
    async def test_empty_result_does_not_retain(self):
        adapter, mock_client = self._make_adapter()
        await adapter.after_turn(_ctx(), result="   ", query="test")
        mock_client.retain.assert_not_called()

    @pytest.mark.asyncio
    async def test_retains_result(self):
        adapter, mock_client = self._make_adapter()
        await adapter.after_turn(_ctx(), result="Fixed the auth bug.", query="help with auth")
        mock_client.retain.assert_called_once()

    @pytest.mark.asyncio
    async def test_retained_content_includes_user_message(self):
        adapter, mock_client = self._make_adapter()
        await adapter.after_turn(_ctx(), result="Found the bug.", query="why is login broken?")
        call_kwargs = mock_client.retain.call_args[1]
        assert "why is login broken?" in call_kwargs["content"]
        assert "Found the bug." in call_kwargs["content"]

    @pytest.mark.asyncio
    async def test_tags_include_user_agent_session(self):
        adapter, mock_client = self._make_adapter()
        await adapter.after_turn(_ctx(), result="Done.", query="task")
        call_kwargs = mock_client.retain.call_args[1]
        assert "user:user-456" in call_kwargs["tags"]
        assert "agent:support-agent" in call_kwargs["tags"]
        assert "session:sess-123" in call_kwargs["tags"]

    @pytest.mark.asyncio
    async def test_metadata_includes_runtime_session_id(self):
        adapter, mock_client = self._make_adapter()
        await adapter.after_turn(_ctx(), result="Done.", query="task")
        call_kwargs = mock_client.retain.call_args[1]
        assert call_kwargs["metadata"]["runtime_session_id"] == "sess-123"
        assert call_kwargs["metadata"]["channel"] == "agentcore-runtime"

    @pytest.mark.asyncio
    async def test_gracefully_degrades_on_exception(self):
        adapter, mock_client = self._make_adapter()
        mock_client.retain.side_effect = Exception("network error")
        # Should not raise
        await adapter.after_turn(_ctx(), result="output", query="task")

    @pytest.mark.asyncio
    async def test_extra_tags_from_retention_policy(self):
        adapter, mock_client = self._make_adapter(
            retention_policy=RetentionPolicy(extra_tags=["project:payments"])
        )
        await adapter.after_turn(_ctx(), result="Done.", query="task")
        call_kwargs = mock_client.retain.call_args[1]
        assert "project:payments" in call_kwargs["tags"]


class TestRunTurn:
    def setup_method(self):
        reset_config()
        configure(hindsight_api_url="http://fake:9077", retain_async=False)

    def teardown_method(self):
        reset_config()

    def _make_adapter(self):
        adapter = HindsightRuntimeAdapter()
        mock_client = MagicMock()
        mock_client.recall.return_value = _make_recall_response("Prior invoice issue resolved")
        adapter._local.client = mock_client
        return adapter, mock_client

    @pytest.mark.asyncio
    async def test_returns_agent_result(self):
        adapter, _ = self._make_adapter()

        async def my_agent(payload, memory_context):
            return {"output": "Handled the request."}

        result = await adapter.run_turn(_ctx(), {"prompt": "help"}, agent_callable=my_agent)
        assert result["output"] == "Handled the request."

    @pytest.mark.asyncio
    async def test_injects_memories_into_agent(self):
        adapter, _ = self._make_adapter()
        received_context: list[str] = []

        async def my_agent(payload, memory_context):
            received_context.append(memory_context)
            return {"output": "done"}

        await adapter.run_turn(_ctx(), {"prompt": "project status"}, agent_callable=my_agent)
        assert "Prior invoice issue resolved" in received_context[0]

    @pytest.mark.asyncio
    async def test_retains_agent_output(self):
        adapter, mock_client = self._make_adapter()

        async def my_agent(payload, memory_context):
            return {"output": "Fixed the billing discrepancy."}

        await adapter.run_turn(_ctx(), {"prompt": "invoice issue"}, agent_callable=my_agent)
        mock_client.retain.assert_called_once()
        call_kwargs = mock_client.retain.call_args[1]
        assert "Fixed the billing discrepancy." in call_kwargs["content"]

    @pytest.mark.asyncio
    async def test_default_agent_name_from_adapter(self):
        adapter = HindsightRuntimeAdapter(agent_name="default-agent")
        mock_client = MagicMock()
        mock_client.recall.return_value = _make_recall_response()
        adapter._local.client = mock_client

        ctx = TurnContext(
            runtime_session_id="sess-1",
            user_id="user-1",
            agent_name="",  # not set in context
        )

        async def my_agent(payload, memory_context):
            return {"output": "done"}

        await adapter.run_turn(ctx, {"prompt": "hi"}, agent_callable=my_agent)
        call_kwargs = mock_client.recall.call_args[1]
        assert "default-agent" in call_kwargs["bank_id"]
