"""
Hindsight LangGraph integration examples.
These snippets are embedded in the documentation via CodeSnippet.
"""

# [docs:setup]
from hindsight_client import Hindsight
from hindsight_langgraph import create_hindsight_tools

client = Hindsight(base_url="http://localhost:8888")

tools = create_hindsight_tools(client=client, bank_id="user-123")
# [/docs:setup]

# [docs:react-agent]
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    ChatOpenAI(model="gpt-4o"),
    tools=tools,
    prompt=(
        "You are a helpful assistant with long-term memory. "
        "Use hindsight_retain to store important facts about the user. "
        "Use hindsight_recall to search your memory before answering. "
        "Use hindsight_reflect for thoughtful summaries of what you know."
    ),
)

result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Remember that I prefer dark mode"}]}
)
# [/docs:react-agent]

# [docs:memory-nodes]
from hindsight_langgraph import create_recall_node, create_retain_node
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END

recall = create_recall_node(
    client=client,
    bank_id_from_config="user_id",
    budget="mid",
    max_results=5,
)
retain = create_retain_node(
    client=client,
    bank_id_from_config="user_id",
    tags=["source:auto"],
)


async def llm_node(state: MessagesState):
    model = ChatOpenAI(model="gpt-4o")
    response = await model.ainvoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("recall", recall)
builder.add_node("llm", llm_node)
builder.add_node("retain", retain)

builder.add_edge(START, "recall")
builder.add_edge("recall", "llm")
builder.add_edge("llm", "retain")
builder.add_edge("retain", END)

graph = builder.compile()

result = await graph.ainvoke(
    {"messages": [HumanMessage(content="What exercise should I do today?")]},
    config={"configurable": {"user_id": "user-456"}},
)
# [/docs:memory-nodes]

# [docs:base-store]
from hindsight_langgraph import HindsightStore
from langgraph.checkpoint.memory import MemorySaver

store = HindsightStore(client=client)
checkpointer = MemorySaver()

graph = builder.compile(checkpointer=checkpointer, store=store)

# Store memories via the BaseStore API
await store.aput(("user", "123", "prefs"), "theme", {"value": "dark mode"})
await store.aput(("user", "123", "prefs"), "language", {"value": "Python"})

# Semantic search across stored memories
results = await store.asearch(("user", "123", "prefs"), query="display preferences")
for item in results:
    print(f"{item.key}: {item.value}")
# [/docs:base-store]

# [docs:constructor-options]
tools = create_hindsight_tools(
    client=client,
    bank_id="user-123",
    budget="high",
    max_tokens=2048,
    tags=["env:prod", "app:support"],
    recall_tags=["env:prod"],
    recall_tags_match="any",
    retain_metadata={"version": "2.0"},
    retain_document_id="session-abc",
    recall_types=["experience", "world"],
    recall_include_entities=True,
    reflect_context="The user is a senior engineer.",
    include_retain=True,
    include_recall=True,
    include_reflect=True,
)
# [/docs:constructor-options]
