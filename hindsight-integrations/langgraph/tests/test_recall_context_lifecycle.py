"""Compiled graph state must not keep old memory context after an empty recall."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from hindsight_client import Hindsight
from hindsight_client_api.models.recall_response import RecallResponse
from hindsight_client_api.models.recall_result import RecallResult
from hindsight_langgraph import create_recall_node
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph


@pytest.mark.parametrize("exit_case", ["empty_results", "empty_query", "missing_bank"])
async def test_empty_recall_removes_previous_context_without_removing_other_messages(exit_case: str) -> None:
    calls: list[str] = []

    async def recall(request: web.Request) -> web.Response:
        calls.append((await request.json())["query"])
        results = [RecallResult(id="m1", text="Favorite color: blue")] if len(calls) == 1 else []
        return web.json_response(RecallResponse(results=results).to_dict())

    app = web.Application()
    app.router.add_post("/v1/default/banks/test/memories/recall", recall)
    async with TestServer(app) as server:
        client = Hindsight(base_url=str(server.make_url("")))
        try:
            builder = StateGraph(MessagesState)
            builder.add_node(
                "recall", create_recall_node(client=client, bank_id=None if exit_case == "missing_bank" else "test")
            )
            builder.add_edge(START, "recall")
            builder.add_edge("recall", END)
            graph = builder.compile()
            first = await graph.ainvoke(
                {"messages": [SystemMessage(content="Be helpful", id="app-system"), HumanMessage(content="Color?")]},
                config={"configurable": {"user_id": "test"}},
            )
            assert any(message.id == "hindsight_memory_context" for message in first["messages"])
            second = await graph.ainvoke(
                {
                    "messages": [
                        *first["messages"],
                        HumanMessage(content="" if exit_case == "empty_query" else "Unrelated topic?"),
                    ]
                }
            )
        finally:
            await client.aclose()

    assert calls == (
        ["Color?", "" if exit_case == "empty_query" else "Unrelated topic?"]
        if exit_case == "empty_results"
        else ["Color?"]
    )
    assert all(message.id != "hindsight_memory_context" for message in second["messages"])
    assert any(message.id == "app-system" and message.content == "Be helpful" for message in second["messages"])
    assert [message.content for message in second["messages"] if isinstance(message, HumanMessage)] == [
        "Color?",
        "" if exit_case == "empty_query" else "Unrelated topic?",
    ]
