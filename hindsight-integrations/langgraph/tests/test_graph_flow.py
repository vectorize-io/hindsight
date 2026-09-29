"""Deterministic compiled-graph flow test (mocked Hindsight client).

This is the in-CI / no-keys analog of the live graph test in test_e2e.py: it
wires a real compiled ``StateGraph`` (recall -> agent -> retain) but backs it
with a mocked Hindsight client, so it exercises the full node orchestration
deterministically without a live server or LLM. It belongs to the deterministic
bucket (no ``requires_real_llm`` marker).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

pytest.importorskip("langgraph")

from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402

from hindsight_langgraph import create_recall_node, create_retain_node  # noqa: E402


def _mock_client(recall_texts: list[str]) -> MagicMock:
    client = MagicMock()
    client.aretain = AsyncMock()
    response = MagicMock()
    response.results = [MagicMock(text=t) for t in recall_texts]
    client.arecall = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_compiled_graph_recall_agent_retain_flow():
    """recall node injects the mocked memory; retain node stores the human turn."""
    client = _mock_client(["The user loves hiking in the mountains"])
    recall = create_recall_node(client=client, bank_id="u1")
    retain = create_retain_node(client=client, bank_id="u1", retain_human=True)

    async def agent(state: MessagesState):
        return {"messages": [AIMessage(content="ack")]}

    builder = StateGraph(MessagesState)
    builder.add_node("recall", recall)
    builder.add_node("agent", agent)
    builder.add_node("retain", retain)
    builder.add_edge(START, "recall")
    builder.add_edge("recall", "agent")
    builder.add_edge("agent", "retain")
    builder.add_edge("retain", END)
    graph = builder.compile()

    result = await graph.ainvoke({"messages": [HumanMessage(content="What do I enjoy outdoors?")]})

    # recall node injected the mocked memory as a SystemMessage
    system_text = " ".join(m.content for m in result["messages"] if isinstance(m, SystemMessage)).lower()
    assert "hiking" in system_text, f"memory not injected into graph state: {system_text}"

    # retain node stored the human message
    client.aretain.assert_awaited()
    stored = client.aretain.call_args.kwargs.get("content", "")
    assert "outdoors" in stored.lower(), f"human turn not retained: {stored!r}"


@pytest.mark.asyncio
async def test_compiled_graph_no_memories_injects_nothing():
    """With an empty bank, the recall node injects no SystemMessage."""
    client = _mock_client([])
    recall = create_recall_node(client=client, bank_id="u1")

    async def agent(state: MessagesState):
        return {"messages": [AIMessage(content="ack")]}

    builder = StateGraph(MessagesState)
    builder.add_node("recall", recall)
    builder.add_node("agent", agent)
    builder.add_edge(START, "recall")
    builder.add_edge("recall", "agent")
    builder.add_edge("agent", END)
    graph = builder.compile()

    result = await graph.ainvoke({"messages": [HumanMessage(content="anything")]})
    assert not any(isinstance(m, SystemMessage) for m in result["messages"])
