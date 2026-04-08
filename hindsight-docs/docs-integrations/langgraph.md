---
sidebar_position: 7
title: "LangGraph & LangChain Persistent Memory with Hindsight"
description: "Add long-term memory to LangGraph and LangChain agents with Hindsight. Three integration patterns — tools, nodes, and BaseStore adapter — for persistent memory across agent runs."
---

# LangGraph / LangChain

Persistent long-term memory for [LangGraph](https://langchain-ai.github.io/langgraph/) and [LangChain](https://python.langchain.com/) agents via Hindsight. Three integration patterns at different abstraction levels — the tools pattern works with both LangChain and LangGraph, while nodes and the BaseStore adapter are LangGraph-specific.

[View Changelog →](/changelog/integrations/langgraph)

## Features

- **Memory Tools** — retain, recall, and reflect as LangChain `@tool` functions compatible with `bind_tools()` and `ToolNode`. Works with **both LangChain and LangGraph** — no LangGraph dependency required for this pattern.
- **Graph Nodes** *(LangGraph)* — Pre-built nodes that auto-inject memories before LLM calls and auto-store after responses
- **BaseStore Adapter** *(LangGraph)* — Drop-in `BaseStore` implementation backed by Hindsight, for LangGraph's native memory patterns
- **Dynamic Banks** — Resolve bank IDs per-request from `RunnableConfig` for per-user memory
- **Async-Native** — Uses `aretain`, `arecall`, `areflect` directly — no thread-pool workarounds

## Installation

```bash
pip install hindsight-langgraph
```

## Quick Start: Tools (LangChain & LangGraph)

The tools pattern creates standard LangChain `@tool` functions that work with any LangChain-compatible model via `bind_tools()`. You can use them with a LangGraph agent or with plain LangChain — no LangGraph required.

**With LangGraph (recommended):**

```python
from hindsight_client import Hindsight
from hindsight_langgraph import create_hindsight_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

client = Hindsight(base_url="http://localhost:8888")
tools = create_hindsight_tools(client=client, bank_id="user-123")

agent = create_react_agent(ChatOpenAI(model="gpt-4o"), tools=tools)

result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Remember that I prefer dark mode"}]}
)
```

**With plain LangChain:**

```python
from hindsight_client import Hindsight
from hindsight_langgraph import create_hindsight_tools
from langchain_openai import ChatOpenAI

client = Hindsight(base_url="http://localhost:8888")
tools = create_hindsight_tools(client=client, bank_id="user-123")

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)
response = await model.ainvoke("Remember that I prefer dark mode")
```

When using plain LangChain, you handle the tool execution loop yourself — call the model, check for `tool_calls`, execute them, and feed results back. LangGraph automates this loop for you.

The agent gets three tools it can call:

- **`hindsight_retain`** — Store information to long-term memory
- **`hindsight_recall`** — Search long-term memory for relevant facts
- **`hindsight_reflect`** — Synthesize a reasoned answer from memories

## Quick Start: Memory Nodes (LangGraph)

Add recall and retain nodes to your graph for automatic memory injection and storage.

```python
from hindsight_client import Hindsight
from hindsight_langgraph import create_recall_node, create_retain_node
from langgraph.graph import StateGraph, MessagesState, START, END

client = Hindsight(base_url="http://localhost:8888")

recall = create_recall_node(client=client, bank_id="user-123")
retain = create_retain_node(client=client, bank_id="user-123")

builder = StateGraph(MessagesState)
builder.add_node("recall", recall)
builder.add_node("agent", agent_node)  # your LLM node
builder.add_node("retain", retain)

builder.add_edge(START, "recall")
builder.add_edge("recall", "agent")
builder.add_edge("agent", "retain")
builder.add_edge("retain", END)

graph = builder.compile()
```

The recall node extracts the latest user message, searches Hindsight, and injects matching memories as a `SystemMessage`. The retain node stores human messages (optionally AI messages too) after the response.

## Quick Start: BaseStore (LangGraph)

Use Hindsight as a LangGraph `BaseStore` for cross-thread persistent memory with semantic search.

```python
from hindsight_client import Hindsight
from hindsight_langgraph import HindsightStore

client = Hindsight(base_url="http://localhost:8888")
store = HindsightStore(client=client)

graph = builder.compile(checkpointer=checkpointer, store=store)

# Store and search via the store API
await store.aput(("user", "123", "prefs"), "theme", {"value": "dark mode"})
results = await store.asearch(("user", "123", "prefs"), query="theme preference")
```

Namespace tuples are mapped to Hindsight bank IDs with `.` as separator (e.g., `("user", "123")` becomes bank `user.123`). Banks are auto-created on first access.

## Dynamic Bank IDs

Both nodes and the store support per-user bank resolution from `RunnableConfig`:

```python
recall = create_recall_node(client=client, bank_id_from_config="user_id")
retain = create_retain_node(client=client, bank_id_from_config="user_id")

# Bank ID resolved at runtime from config
result = await graph.ainvoke(
    {"messages": [{"role": "user", "content": "hello"}]},
    config={"configurable": {"user_id": "user-456"}},
)
```

## Selecting Tools

Include only the tools you need:

```python
tools = create_hindsight_tools(
    client=client,
    bank_id="user-123",
    include_retain=True,
    include_recall=True,
    include_reflect=False,  # Omit reflect
)
```

## Global Configuration

Instead of passing a client to every call, configure once:

```python
from hindsight_langgraph import configure, create_hindsight_tools

configure(
    hindsight_api_url="http://localhost:8888",
    api_key="your-api-key",       # Or set HINDSIGHT_API_KEY env var
    budget="mid",                  # Recall budget: low/mid/high
    max_tokens=4096,               # Max tokens for recall results
    tags=["env:prod"],             # Tags for stored memories
    recall_tags=["scope:global"],  # Tags to filter recall
    recall_tags_match="any",       # Tag match mode: any/all/any_strict/all_strict
)

# Now create tools without passing client — uses global config
tools = create_hindsight_tools(bank_id="user-123")
```

## Retain Node Options

```python
retain = create_retain_node(
    client=client,
    bank_id="user-123",
    retain_human=True,    # Store human messages (default: True)
    retain_ai=False,      # Store AI responses (default: False)
    tags=["source:chat"], # Tags applied to stored memories
)
```

## Recall Node Options

```python
recall = create_recall_node(
    client=client,
    bank_id="user-123",
    budget="low",          # Recall budget: low/mid/high
    max_results=10,        # Max memories injected
    max_tokens=4096,       # Max tokens for recall
    tags=["scope:user"],   # Filter by tags
    tags_match="all",      # Tag match mode
)
```

### Using `output_key` for Prompt Control

By default, the recall node appends a `SystemMessage` to `messages`. Use `output_key` to write memory text to a custom state field instead, giving you full control over prompt ordering:

```python
from typing import Optional
from langgraph.graph import MessagesState

class AgentState(MessagesState):
    memory_context: Optional[str] = None

recall = create_recall_node(
    client=client,
    bank_id="user-123",
    output_key="memory_context",
)

# In your agent node, read state["memory_context"] and prepend it
# to the system prompt before calling the model.
```

## Limitations and Notes

### HindsightStore

- **Async-only.** All sync methods (`batch`, `get`, `put`, `delete`, `search`, `list_namespaces`) raise `NotImplementedError`. Use the async variants (`abatch`, `aget`, `aput`, `adelete`, `asearch`, `alist_namespaces`) instead.
- **`get()` relies on recall.** There is no direct key lookup — the key is used as a recall query and only exact `document_id` matches are returned. Items that do not rank in the top recall results may appear missing.
- **`list_namespaces` is session-scoped.** It only tracks namespaces that have been written to via `aput()` during the current process. After a restart, `list_namespaces` returns empty even though data still exists in Hindsight.
- **`delete` is a no-op.** Calling `adelete()` logs a debug message but does not remove data from Hindsight. Hindsight's memory model is append-oriented; fact superseding is handled automatically during retain.

### Memory Nodes

- **SystemMessage ordering.** The recall node adds a `SystemMessage` with recalled memories. Because `MessagesState` uses `add_messages` (which appends), this message appears after existing messages rather than at position 0. The message has a stable ID (`hindsight_memory_context`) so it is updated rather than duplicated across invocations. If your LLM provider requires system messages first, sort or filter messages in your agent node before passing them to the model.

### Error Handling

- **Tools** raise `HindsightError` on failure, which surfaces to the agent as a tool error.
- **Nodes** silently log errors and return empty messages, so a Hindsight outage does not crash your graph.

## API Reference

### `create_hindsight_tools()`

| Parameter | Default | Description |
|---|---|---|
| `bank_id` | *required* | Hindsight memory bank ID |
| `client` | `None` | Pre-configured Hindsight client |
| `hindsight_api_url` | `None` | API URL (used if no client provided) |
| `api_key` | `None` | API key (used if no client provided) |
| `budget` | `"mid"` | Recall/reflect budget level (low/mid/high) |
| `max_tokens` | `4096` | Maximum tokens for recall results |
| `tags` | `None` | Tags applied when storing memories |
| `recall_tags` | `None` | Tags to filter when searching |
| `recall_tags_match` | `"any"` | Tag matching mode (any/all/any\_strict/all\_strict) |
| `retain_metadata` | `None` | Default metadata dict for retain operations |
| `retain_document_id` | `None` | Default document\_id for retain (groups/upserts memories) |
| `recall_types` | `None` | Fact types to filter (world, experience, opinion, observation) |
| `recall_include_entities` | `False` | Include entity information in recall results |
| `reflect_context` | `None` | Additional context for reflect operations |
| `reflect_max_tokens` | `None` | Max tokens for reflect results (defaults to `max_tokens`) |
| `reflect_response_schema` | `None` | JSON schema to constrain reflect output format |
| `reflect_tags` | `None` | Tags to filter memories used in reflect (defaults to `recall_tags`) |
| `reflect_tags_match` | `None` | Tag matching for reflect (defaults to `recall_tags_match`) |
| `include_retain` | `True` | Include the retain (store) tool |
| `include_recall` | `True` | Include the recall (search) tool |
| `include_reflect` | `True` | Include the reflect (synthesize) tool |

### `create_recall_node()`

| Parameter | Default | Description |
|---|---|---|
| `bank_id` | `None` | Static bank ID (or use `bank_id_from_config`) |
| `client` | `None` | Pre-configured Hindsight client |
| `hindsight_api_url` | `None` | API URL (used if no client provided) |
| `api_key` | `None` | API key (used if no client provided) |
| `budget` | `"mid"` | Recall budget level |
| `max_tokens` | `4096` | Max tokens for recall results |
| `max_results` | `10` | Max memories to inject |
| `tags` | `None` | Tags to filter recall results |
| `tags_match` | `"any"` | Tag matching mode |
| `bank_id_from_config` | `"user_id"` | Config key to resolve bank ID at runtime |
| `output_key` | `None` | If set, write memory text to this state key instead of appending a SystemMessage to `messages` |

### `create_retain_node()`

| Parameter | Default | Description |
|---|---|---|
| `bank_id` | `None` | Static bank ID (or use `bank_id_from_config`) |
| `client` | `None` | Pre-configured Hindsight client |
| `hindsight_api_url` | `None` | API URL (used if no client provided) |
| `api_key` | `None` | API key (used if no client provided) |
| `tags` | `None` | Tags applied to stored memories |
| `bank_id_from_config` | `"user_id"` | Config key to resolve bank ID at runtime |
| `retain_human` | `True` | Store human messages |
| `retain_ai` | `False` | Store AI responses |

### `HindsightStore()`

| Parameter | Default | Description |
|---|---|---|
| `client` | `None` | Pre-configured Hindsight client |
| `hindsight_api_url` | `None` | API URL (used if no client provided) |
| `api_key` | `None` | API key (used if no client provided) |
| `tags` | `None` | Tags applied to all retain operations |

### `configure()`

| Parameter | Default | Description |
|---|---|---|
| `hindsight_api_url` | Production API | Hindsight API URL |
| `api_key` | `HINDSIGHT_API_KEY` env | API key for authentication |
| `budget` | `"mid"` | Default recall budget level |
| `max_tokens` | `4096` | Default max tokens for recall |
| `tags` | `None` | Default tags for retain operations |
| `recall_tags` | `None` | Default tags to filter recall |
| `recall_tags_match` | `"any"` | Default tag matching mode |
| `verbose` | `False` | Enable verbose logging |

## Requirements

- Python >= 3.10
- langchain-core >= 0.3.0
- hindsight-client >= 0.4.0
- langgraph >= 0.3.0 *(only for nodes and store patterns — install with `pip install hindsight-langgraph[langgraph]`)*
