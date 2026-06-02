"""Tests for the create_hindsight_tools factory."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hindsight_google_adk import (
    HindsightError,
    configure,
    create_hindsight_tools,
    reset_config,
)


@pytest.fixture(autouse=True)
def _reset_global_config():
    reset_config()
    yield
    reset_config()


def _mock_client() -> MagicMock:
    client = MagicMock()
    client.aretain = AsyncMock()
    client.arecall = AsyncMock(return_value=SimpleNamespace(results=[]))
    client.areflect = AsyncMock(return_value=SimpleNamespace(text="reflection"))
    return client


class TestFactory:
    def test_requires_client_or_config(self):
        with pytest.raises(HindsightError):
            create_hindsight_tools(bank_id="bank-1")

    def test_returns_three_tools_by_default(self):
        tools = create_hindsight_tools(bank_id="bank-1", client=_mock_client())
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"hindsight_retain", "hindsight_recall", "hindsight_reflect"}

    def test_include_flags_select_subsets(self):
        tools = create_hindsight_tools(
            bank_id="bank-1",
            client=_mock_client(),
            include_retain=True,
            include_recall=False,
            include_reflect=False,
        )
        assert {t.name for t in tools} == {"hindsight_retain"}

    def test_resolves_client_from_url(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_API_KEY", "k")
        configure(hindsight_api_url="https://api.example.com", api_key="k")
        # Resolved client is constructed against the URL; tools should be returned
        tools = create_hindsight_tools(bank_id="bank-1")
        assert len(tools) == 3


def _tool_by_name(tools, name):
    for t in tools:
        if t.name == name:
            return t
    raise AssertionError(f"Tool {name} not found in {[t.name for t in tools]}")


class TestRetainTool:
    async def test_executes_retain(self):
        client = _mock_client()
        tools = create_hindsight_tools(
            bank_id="bank-1", client=client, tags=["env:prod"]
        )
        retain = _tool_by_name(tools, "hindsight_retain")
        result = await retain.func(content="Remember this")
        assert result == "Memory stored successfully."
        client.aretain.assert_awaited_once()
        kwargs = client.aretain.await_args.kwargs
        assert kwargs["bank_id"] == "bank-1"
        assert kwargs["content"] == "Remember this"
        assert kwargs["tags"] == ["env:prod"]

    async def test_wraps_error_as_hindsight_error(self):
        client = _mock_client()
        client.aretain.side_effect = RuntimeError("api down")
        tools = create_hindsight_tools(bank_id="bank-1", client=client)
        retain = _tool_by_name(tools, "hindsight_retain")
        with pytest.raises(HindsightError):
            await retain.func(content="x")


class TestRecallTool:
    async def test_executes_recall(self):
        client = _mock_client()
        client.arecall.return_value = SimpleNamespace(
            results=[
                SimpleNamespace(id="m1", text="first"),
                SimpleNamespace(id="m2", text="second"),
            ]
        )
        tools = create_hindsight_tools(bank_id="bank-1", client=client)
        recall = _tool_by_name(tools, "hindsight_recall")
        result = await recall.func(query="what did I say?")
        assert "1. first" in result
        assert "2. second" in result

    async def test_empty_results_friendly_message(self):
        tools = create_hindsight_tools(bank_id="bank-1", client=_mock_client())
        recall = _tool_by_name(tools, "hindsight_recall")
        assert "No relevant memories" in await recall.func(query="anything")

    async def test_recall_tags_forwarded(self):
        client = _mock_client()
        tools = create_hindsight_tools(
            bank_id="bank-1",
            client=client,
            recall_tags=["env:prod"],
            recall_tags_match="all",
        )
        recall = _tool_by_name(tools, "hindsight_recall")
        await recall.func(query="q")
        kwargs = client.arecall.await_args.kwargs
        assert kwargs["tags"] == ["env:prod"]
        assert kwargs["tags_match"] == "all"


class TestReflectTool:
    async def test_executes_reflect(self):
        client = _mock_client()
        tools = create_hindsight_tools(bank_id="bank-1", client=client)
        reflect = _tool_by_name(tools, "hindsight_reflect")
        result = await reflect.func(query="what do I prefer?")
        assert result == "reflection"

    async def test_reflect_context_and_schema_forwarded(self):
        client = _mock_client()
        tools = create_hindsight_tools(
            bank_id="bank-1",
            client=client,
            reflect_context="user is a developer",
            reflect_response_schema={"type": "object"},
        )
        reflect = _tool_by_name(tools, "hindsight_reflect")
        await reflect.func(query="q")
        kwargs = client.areflect.await_args.kwargs
        assert kwargs["context"] == "user is a developer"
        assert kwargs["response_schema"] == {"type": "object"}

    async def test_empty_reflect_response_friendly(self):
        client = _mock_client()
        client.areflect.return_value = SimpleNamespace(text="")
        tools = create_hindsight_tools(bank_id="bank-1", client=client)
        reflect = _tool_by_name(tools, "hindsight_reflect")
        assert "No relevant memories" in await reflect.func(query="q")
