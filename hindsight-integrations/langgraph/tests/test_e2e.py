"""End-to-end tests for the Hindsight-LangGraph integration.

Exercises the three integration patterns — tools, graph nodes, and the
``memory_instructions`` helper — against a live Hindsight instance. By
default these tests are skipped; point ``HINDSIGHT_API_URL`` at a reachable
Hindsight server to enable them.

The LangGraph patterns talk to Hindsight directly (retain/recall/reflect);
they don't call an external LLM themselves, so only a running Hindsight
instance is required. The graph-node tests additionally require ``langgraph``.

Run with::

    uv run pytest tests/test_e2e.py -v -s

Environment variables:
    HINDSIGHT_API_URL     URL of a reachable Hindsight server
                          (default: http://localhost:8888)
    HINDSIGHT_API_KEY     API key for the Hindsight server (optional)
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
import pytest_asyncio
import requests
from hindsight_client import Hindsight

from hindsight_langgraph import (
    create_hindsight_tools,
    memory_instructions,
)

HINDSIGHT_API_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")

_NO_MEMORIES = "No relevant memories found."


def _hindsight_available() -> bool:
    try:
        r = requests.get(f"{HINDSIGHT_API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _langgraph_available() -> bool:
    try:
        import langgraph  # noqa: F401

        return True
    except ImportError:
        return False


requires_hindsight = pytest.mark.skipif(
    not _hindsight_available(),
    reason=f"Hindsight not reachable at {HINDSIGHT_API_URL}",
)
requires_langgraph = pytest.mark.skipif(
    not _langgraph_available(),
    reason="langgraph not installed",
)

# Every test in this file drives a live Hindsight server (server-side fact
# extraction powers the retain->recall roundtrip), so the whole module is the
# "real LLM" bucket — excluded from the deterministic PR-CI bucket via
# `-m "not requires_real_llm"` and run on its own via `-m requires_real_llm`.
# The skipif guards above still apply at runtime.
pytestmark = pytest.mark.requires_real_llm


@pytest_asyncio.fixture
async def live_client():
    """Yield ``(client, bank_id)`` backed by a freshly created test bank.

    The bank is deleted and the client closed on teardown so aiohttp doesn't
    warn about unclosed sessions and banks don't leak between runs.
    """
    client = Hindsight(base_url=HINDSIGHT_API_URL)
    bank_id = f"langgraph-e2e-{uuid.uuid4().hex[:8]}"
    await client.acreate_bank(bank_id, name=f"LangGraph E2E {bank_id}")
    try:
        yield client, bank_id
    finally:
        try:
            await client.adelete_bank(bank_id)
        except Exception:
            pass
        await client.aclose()


async def _recall_tool_until_nonempty(recall_tool, query, attempts=10, delay=1.0):
    """Poll the recall tool until it surfaces a memory.

    Retain takes a moment to flow through fact extraction + indexing; an
    empty result under that delay would otherwise let an assertion-less
    roundtrip silently pass.
    """
    for _ in range(attempts):
        result = await recall_tool.ainvoke(query)
        if result and result != _NO_MEMORIES:
            return result
        await asyncio.sleep(delay)
    pytest.fail(
        f"recall({query!r}) returned no memories after {attempts * delay:.0f}s — "
        "either retain failed to surface or the query no longer matches."
    )


async def _recall_node_until_nonempty(recall_node, query, attempts=10, delay=1.0):
    """Poll a recall node until it injects at least one message."""
    from langchain_core.messages import HumanMessage

    for _ in range(attempts):
        out = await recall_node({"messages": [HumanMessage(content=query)]})
        if out.get("messages"):
            return out
        await asyncio.sleep(delay)
    pytest.fail(
        f"recall_node({query!r}) injected nothing after {attempts * delay:.0f}s — "
        "either retain failed to surface or the query no longer matches."
    )


@requires_hindsight
class TestE2ETools:
    """create_hindsight_tools() retain/recall/reflect against live Hindsight."""

    @pytest.mark.asyncio
    async def test_retain_and_recall_roundtrip(self, live_client):
        client, bank_id = live_client
        retain, recall, _reflect = create_hindsight_tools(client=client, bank_id=bank_id)

        result = await retain.ainvoke("The team uses PostgreSQL 16 and deploys to us-east-1.")
        assert result == "Memory stored successfully."

        recalled = await _recall_tool_until_nonempty(recall, "What technologies does the team use?")
        lowered = recalled.lower()
        assert "postgresql" in lowered or "us-east-1" in lowered, (
            f"Recall surfaced results but none referenced the stored content: {recalled}"
        )

    @pytest.mark.asyncio
    async def test_reflect_synthesizes_from_memory(self, live_client):
        client, bank_id = live_client
        retain, recall, reflect = create_hindsight_tools(client=client, bank_id=bank_id)

        await retain.ainvoke("The team uses PostgreSQL 16 and deploys to us-east-1.")
        # Make sure the memory is queryable before reflecting.
        await _recall_tool_until_nonempty(recall, "What technologies does the team use?")

        result = await reflect.ainvoke("What do I know about the team's tech stack?")
        assert result and result != _NO_MEMORIES, "Reflect should synthesise non-empty text"
        lowered = result.lower()
        assert "postgresql" in lowered or "us-east" in lowered, (
            f"Reflect text didn't reference the stored memory: {result[:300]}"
        )

    @pytest.mark.asyncio
    async def test_recall_empty_bank(self, live_client):
        client, bank_id = live_client
        tools = create_hindsight_tools(
            client=client,
            bank_id=bank_id,
            include_retain=False,
            include_reflect=False,
        )
        result = await tools[0].ainvoke("anything at all")
        assert result == _NO_MEMORIES


@requires_hindsight
@requires_langgraph
class TestE2ENodes:
    """create_recall_node() / create_retain_node() against live Hindsight."""

    @pytest.mark.asyncio
    async def test_retain_node_then_recall_node(self, live_client):
        from langchain_core.messages import HumanMessage

        from hindsight_langgraph import create_recall_node, create_retain_node

        client, bank_id = live_client
        retain_node = create_retain_node(client=client, bank_id=bank_id, retain_human=True)
        recall_node = create_recall_node(client=client, bank_id=bank_id)

        await retain_node({"messages": [HumanMessage(content="The team uses PostgreSQL 16 and deploys to us-east-1.")]})

        out = await _recall_node_until_nonempty(recall_node, "What database does the team use?")
        text = " ".join(m.content for m in out["messages"]).lower()
        assert "postgresql" in text or "us-east-1" in text, (
            f"recall node injected messages but none referenced the stored memory: {text}"
        )

    @pytest.mark.asyncio
    async def test_full_graph_flow_recall_agent_retain(self, live_client):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        from langgraph.graph import END, START, MessagesState, StateGraph

        from hindsight_langgraph import create_recall_node, create_retain_node

        client, bank_id = live_client
        recall = create_recall_node(client=client, bank_id=bank_id)
        retain = create_retain_node(client=client, bank_id=bank_id, retain_human=True)

        async def agent_node(state: MessagesState):
            last = state["messages"][-1]
            return {"messages": [AIMessage(content=f"I heard: {last.content}")]}

        builder = StateGraph(MessagesState)
        builder.add_node("recall", recall)
        builder.add_node("agent", agent_node)
        builder.add_node("retain", retain)
        builder.add_edge(START, "recall")
        builder.add_edge("recall", "agent")
        builder.add_edge("agent", "retain")
        builder.add_edge("retain", END)
        graph = builder.compile()

        # First turn stores the hiking fact (no memories injected yet).
        await graph.ainvoke({"messages": [HumanMessage(content="I love hiking in the mountains")]})

        # Wait until the stored memory is queryable before the second turn.
        await _recall_node_until_nonempty(recall, "What outdoor activities do I enjoy?")

        result = await graph.ainvoke({"messages": [HumanMessage(content="What outdoor activities do I enjoy?")]})
        injected = " ".join(m.content for m in result["messages"] if isinstance(m, SystemMessage)).lower()
        assert "hiking" in injected or "mountain" in injected, (
            f"Recall node didn't inject the stored hiking memory into the graph: {injected}"
        )


@requires_hindsight
class TestE2EMemoryInstructions:
    """memory_instructions() standalone helper against live Hindsight."""

    @pytest.mark.asyncio
    async def test_injects_recalled_memory(self, live_client):
        client, bank_id = live_client
        get_instructions = memory_instructions(
            client=client,
            bank_id=bank_id,
            base_instructions="You are a helpful assistant.",
            query="the user's programming languages and tools",
        )

        await client.aretain(
            bank_id=bank_id,
            content="The user codes primarily in Python and SQL and prefers VS Code.",
        )

        result = None
        for _ in range(10):
            result = await get_instructions()
            if result != "You are a helpful assistant.":
                break
            await asyncio.sleep(1.0)

        assert result.startswith("You are a helpful assistant."), (
            f"Base instructions should be preserved, got: {result[:200]}"
        )
        lowered = result.lower()
        assert "python" in lowered or "sql" in lowered or "vs code" in lowered, (
            f"memory_instructions didn't append the recalled memory: {result[:300]}"
        )
