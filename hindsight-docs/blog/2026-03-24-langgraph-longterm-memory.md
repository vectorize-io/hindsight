---
title: "Adding Long-Term Memory to LangGraph and LangChain Agents"
description: Learn how to add long-term memory to LangGraph and LangChain agents using three integration patterns — tools, nodes, and BaseStore — with per-user memory banks and semantic recall.
authors: [DK09876]
date: 2026-03-24T12:00
tags: [langgraph, langchain, integrations, agents, memory, tutorial]
image: /img/blog/langgraph-longterm-memory.png
hide_table_of_contents: true
---

![Adding Long-Term Memory to LangGraph and LangChain Agents](/img/blog/langgraph-longterm-memory.png)

LangGraph agents are stateful by design — checkpointers save graph state between steps, and the Store API persists data across threads. But neither gives agents true long-term memory: the ability to extract meaning from conversations, build up knowledge over time, and recall it semantically when relevant.

That's what Hindsight adds. Hindsight is a memory layer for LLM applications that automatically extracts facts from conversations, builds entity graphs, and retrieves relevant context using four parallel recall strategies. The `hindsight-langgraph` package brings that to LangGraph — and since the memory tools are standard LangChain `@tool` functions, they work with plain LangChain too.

<!-- truncate -->

## The problem

LangGraph's built-in persistence is designed for graph state — checkpoints, intermediate values, cross-thread key-value storage. It's good at "what did this graph do last time?" but not at "what does this agent know about this user?"

Consider a support agent that talks to the same customer across dozens of sessions. With checkpointers alone, each new thread starts cold. With `InMemoryStore` or `PostgresStore`, you can manually store and retrieve facts, but you're responsible for:

- Deciding what to store (fact extraction)
- Deciding what's relevant (semantic retrieval)
- Handling contradictions and updates
- Building knowledge graphs from raw conversations

Hindsight does all of this automatically. You retain conversations, and it extracts facts, builds entity graphs, and retrieves relevant memories using four parallel strategies: **semantic** (embedding similarity), **BM25** (keyword overlap), **graph traversal** (entity relationships), and **temporal** (recency weighting). Each strategy catches different things — semantic recall finds conceptually similar memories, graph traversal finds memories linked through shared entities, and temporal weighting surfaces recent context before older facts. Together they substantially outperform single-strategy retrieval.

## Three integration patterns

We built three ways to add Hindsight memory to LangGraph, at different abstraction levels.

### 1. Tools — the agent decides (LangChain & LangGraph)

Give the agent retain/recall/reflect tools and let it decide when to use memory. These are standard LangChain `@tool` functions, so they work with both LangGraph (via `create_react_agent`) and plain LangChain (via `bind_tools()`).

```python
from hindsight_client import Hindsight
from hindsight_langgraph import create_hindsight_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

client = Hindsight(base_url="http://localhost:8888")
tools = create_hindsight_tools(client=client, bank_id="user-123")

# With LangGraph
agent = create_react_agent(ChatOpenAI(model="gpt-4o"), tools=tools)

# Or with plain LangChain
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)
```

The agent gets three tools:

- **`hindsight_retain`** — stores the conversation and extracts facts from it
- **`hindsight_recall`** — searches the memory bank for relevant context
- **`hindsight_reflect`** — synthesizes across multiple memories to produce a summary or answer a question about what the agent knows (useful for questions like "what has this user told me about their stack?")

The agent calls these based on conversation context — storing facts when the user shares something important, recalling when asked about past context, and reflecting when it needs to synthesize accumulated knowledge.

**Best for**: ReAct agents that need to reason about when memory is relevant. Works with LangGraph for automatic tool execution loops or with plain LangChain if you manage the loop yourself.

### 2. Nodes — memory as graph steps

Add recall and retain as automatic nodes in your graph. No tool-calling required — memory runs on every turn.

```python
from hindsight_langgraph import create_recall_node, create_retain_node
from langgraph.graph import StateGraph, MessagesState, START, END

recall = create_recall_node(client=client, bank_id_from_config="user_id")
retain = create_retain_node(client=client, bank_id_from_config="user_id")

builder = StateGraph(MessagesState)
builder.add_node("recall", recall)
builder.add_node("agent", agent_node)
builder.add_node("retain", retain)
builder.add_edge(START, "recall")
builder.add_edge("recall", "agent")
builder.add_edge("agent", "retain")
builder.add_edge("retain", END)
```

The recall node runs before the LLM, searches Hindsight for memories relevant to the user's message, and injects them as a `SystemMessage`. The retain node runs after, storing the conversation. Both resolve per-user bank IDs from `RunnableConfig` at runtime.

**Best for**: Agents where you always want memory context injected automatically, without relying on the LLM to decide when to use memory tools.

### 3. BaseStore — drop-in backend

Replace LangGraph's `InMemoryStore` with Hindsight as the storage backend. If your team already uses LangGraph's store patterns, this is the lowest-friction path.

```python
from hindsight_langgraph import HindsightStore

store = HindsightStore(client=client)
graph = builder.compile(checkpointer=checkpointer, store=store)
```

Namespace tuples map to Hindsight bank IDs (`("user", "123")` → bank `user.123`), banks are auto-created, and `search()` uses Hindsight's full semantic recall instead of basic vector similarity.

**Best for**: Teams already using LangGraph's `store` patterns who want better retrieval without restructuring their graph.

---

### Which pattern fits your use case?

| | Tools | Nodes | BaseStore |
|---|---|---|---|
| Works with plain LangChain | Yes | No | No |
| Memory runs automatically | No (LLM decides) | Yes | Yes |
| Uses existing store interface | No | No | Yes |
| LLM controls when to remember | Yes | No | No |
| Lowest migration cost | — | Low | Lowest |

---

## Complete working example

Here's a full support agent that remembers each user across sessions using the nodes pattern. This is copy-pasteable and runnable against either self-hosted Hindsight or Hindsight Cloud.

```python
import asyncio
from hindsight_client import Hindsight
from hindsight_langgraph import create_recall_node, create_retain_node
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

# --- Setup ---

client = Hindsight(base_url="http://localhost:8888")
# For Hindsight Cloud:
# client = Hindsight(base_url="https://api.hindsight.vectorize.io", api_key="...")

llm = ChatOpenAI(model="gpt-4o")
checkpointer = MemorySaver()

# --- Memory nodes ---
# bank_id_from_config pulls the user ID from RunnableConfig at runtime,
# so one graph definition serves all users with isolated memory banks.

recall = create_recall_node(client=client, bank_id_from_config="user_id")
retain = create_retain_node(client=client, bank_id_from_config="user_id")

# --- Agent node ---

async def agent_node(state: MessagesState):
    system = SystemMessage(content=(
        "You are a helpful support agent. "
        "Relevant memories about this user have been injected above. "
        "Use them to personalize your response."
    ))
    response = await llm.ainvoke([system] + state["messages"])
    return {"messages": [response]}

# --- Graph ---

builder = StateGraph(MessagesState)
builder.add_node("recall", recall)
builder.add_node("agent", agent_node)
builder.add_node("retain", retain)
builder.add_edge(START, "recall")
builder.add_edge("recall", "agent")
builder.add_edge("agent", "retain")
builder.add_edge("retain", END)

graph = builder.compile(checkpointer=checkpointer)

# --- Run ---

async def chat(user_id: str, thread_id: str, message: str):
    config = {
        "configurable": {
            "user_id": user_id,
            "thread_id": thread_id,
        }
    }
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )
    return result["messages"][-1].content


async def main():
    # Session 1: user shares context
    print("Session 1")
    print(await chat("user-42", "thread-1", "Hi! I'm running into issues with our Postgres connection pool. We're on SQLAlchemy 2.0."))
    print(await chat("user-42", "thread-1", "We're using async sessions with asyncpg. The pool keeps exhausting under load."))

    # Session 2: new thread, same user — agent remembers
    print("\nSession 2 (new thread)")
    print(await chat("user-42", "thread-2", "Hey, back again. Still fighting the connection pool issue."))
    # Agent recalls SQLAlchemy 2.0, asyncpg, and the pool exhaustion context
    # without the user having to repeat themselves.


asyncio.run(main())
```

What Hindsight extracts from Session 1 and stores in `user-42`'s memory bank:

```
- Uses SQLAlchemy 2.0 with async sessions
- Uses asyncpg driver
- Experiencing connection pool exhaustion under load
- Running Postgres
```

When Session 2 starts on a fresh thread, the recall node searches the memory bank for context relevant to "Still fighting the connection pool issue" and injects those facts as a `SystemMessage` before the LLM responds. The agent picks up exactly where the last session ended.

---

## Per-user memory in one line

All three patterns support dynamic bank IDs. Instead of hardcoding a bank, resolve it from the graph's config at runtime:

```python
recall = create_recall_node(client=client, bank_id_from_config="user_id")

# Each invocation gets its own isolated memory bank
await graph.ainvoke(
    {"messages": [...]},
    config={"configurable": {"user_id": "user-456"}},
)
```

One graph definition serves all users. Memory banks are created automatically and kept fully isolated.

## Getting started

```bash
pip install hindsight-langgraph
```

Works with both self-hosted Hindsight and [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup). For cloud, pass your API key when creating the client:

```python
client = Hindsight(base_url="https://api.hindsight.vectorize.io", api_key="your-key")
# or
from hindsight_client import configure
configure(api_key="your-key")  # defaults to the cloud URL
```

## What to build with this

Long-term memory unlocks a different class of agent behavior. A few patterns we've seen work well:

- **Support agents** that remember each customer's history, preferences, and past issues across sessions
- **Sales assistants** that accumulate context about prospects over multiple touchpoints
- **Personal productivity agents** that build up a model of a user's work style, priorities, and decisions

In all three cases, the agent gets meaningfully better the longer it runs — not just because of a longer context window, but because Hindsight distills conversations into structured knowledge it can retrieve precisely when relevant.

Full docs: [LangGraph integration](/sdks/integrations/langgraph) | [GitHub](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/langgraph)
