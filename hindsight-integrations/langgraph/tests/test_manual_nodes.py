"""Manual test of recall/retain nodes with a real graph."""

import asyncio

from hindsight_client import Hindsight
from hindsight_langgraph import create_recall_node, create_retain_node
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph


async def main():
    client = Hindsight(base_url="http://localhost:8888")
    try:
        await client.acreate_bank("langgraph-nodes-test", name="Nodes Test")

        recall = create_recall_node(client=client, bank_id="langgraph-nodes-test")
        retain = create_retain_node(client=client, bank_id="langgraph-nodes-test")

        # Fake agent node that just echoes
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

        # First invocation — no memories yet
        print("--- First call (no memories) ---")
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="I love hiking in the mountains")]}
        )
        for msg in result["messages"]:
            print(f"  [{msg.type}] {msg.content[:100]}")

        await asyncio.sleep(2)

        # Second invocation — should recall the hiking memory
        print("\n--- Second call (should recall hiking) ---")
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="What outdoor activities do I enjoy?")]}
        )
        for msg in result["messages"]:
            print(f"  [{msg.type}] {msg.content[:100]}")

        await client.adelete_bank("langgraph-nodes-test")
        print("\n--- Done, bank cleaned up ---")
    finally:
        # Close the client so aiohttp doesn't warn about unclosed sessions.
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
