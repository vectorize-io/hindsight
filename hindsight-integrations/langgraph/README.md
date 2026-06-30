# hindsight-langgraph

LangGraph and LangChain integration for [Hindsight](https://github.com/vectorize-io/hindsight) — persistent long-term memory for AI agents.

Provides three integration patterns:
- **Tools** — retain/recall/reflect as LangChain `@tool` functions for agent-driven memory. Works with **both LangChain and LangGraph**.
- **Nodes** *(LangGraph)* — pre-built graph nodes for automatic memory injection and storage
- **Memory Instructions** — pre-fetch memories into a system prompt string. Works with **any LangChain model**, no graph needed.

## Prerequisites

- A [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) account or a [self-hosted](https://github.com/vectorize-io/hindsight#quick-start) Hindsight instance
- Python 3.10+

## Installation

```bash
pip install hindsight-langgraph
```

## Quick Start: Tools

Bind Hindsight memory tools to your LangGraph agent so it can store and retrieve memories on demand.

```python
from hindsight_langgraph import create_hindsight_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Set HINDSIGHT_API_KEY env var to authenticate
tools = create_hindsight_tools(bank_id="user-123")

agent = create_react_agent(
    ChatOpenAI(model="gpt-4o"),
    tools=tools,
)

result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Remember that I prefer dark mode"}]}
)
```

### Dynamic Bank IDs with Tools

If you build the agent once and serve multiple users, omit the static `bank_id`
and resolve it per request from `config["configurable"]`:

```python
tools = create_hindsight_tools(bank_id_from_config="user_id")
agent = create_react_agent(ChatOpenAI(model="gpt-4o"), tools=tools)

result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Remember that I prefer dark mode"}]},
    config={"configurable": {"user_id": "user-456"}},
)
```

Passing `bank_id="user-123"` still pins all tool calls to that bank and takes
precedence over `bank_id_from_config`.

## Quick Start: Memory Nodes

Add recall and retain nodes to your graph for automatic memory injection before LLM calls and storage after responses.

```python
from hindsight_langgraph import create_recall_node, create_retain_node
from langgraph.graph import StateGraph, MessagesState, START, END

recall = create_recall_node(bank_id="user-123")
retain = create_retain_node(bank_id="user-123")

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

### Dynamic Bank IDs

Use `bank_id_from_config` to resolve the bank per-request from the graph's config:

```python
recall = create_recall_node(bank_id_from_config="user_id")
retain = create_retain_node(bank_id_from_config="user_id")

# Bank ID resolved at runtime
result = await graph.ainvoke(
    {"messages": [{"role": "user", "content": "hello"}]},
    config={"configurable": {"user_id": "user-456"}},
)
```

## Quick Start: Memory Instructions

Pre-fetch memories and inject them into a system prompt. Works with any LangChain model — no graph needed.

```python
from hindsight_langgraph import memory_instructions
from langchain_openai import ChatOpenAI

get_instructions = memory_instructions(
    bank_id="user-123",
    base_instructions="You are a helpful assistant.",
)

# Each call re-fetches memories, so it stays up to date
instructions = await get_instructions()
response = await ChatOpenAI(model="gpt-4o").ainvoke([
    {"role": "system", "content": instructions},
    {"role": "user", "content": "What do you know about me?"},
])
```

## Configuration

### Global config

```python
from hindsight_langgraph import configure

configure(
    api_key="your-api-key",  # or set HINDSIGHT_API_KEY env var
    budget="mid",
    tags=["source:langgraph"],
)
```

### Self-hosted instance

To connect to a self-hosted Hindsight instance instead of Hindsight Cloud:

```python
configure(
    hindsight_api_url="http://localhost:8888",
)
```

Or pass `hindsight_api_url` directly to any factory function:

```python
tools = create_hindsight_tools(bank_id="user-123", hindsight_api_url="http://localhost:8888")
```

### Per-call overrides

All factory functions accept `client`, `hindsight_api_url`, and `api_key` to override the global config.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `hindsight_api_url` | Hindsight API URL | `https://api.hindsight.vectorize.io` |
| `api_key` | API key (or `HINDSIGHT_API_KEY` env var) | `None` |
| `budget` | Recall budget: `low`, `mid`, `high` | `mid` |
| `max_tokens` | Max tokens for recall results | `4096` |
| `tags` | Tags applied to retain operations | `None` |
| `recall_tags` | Tags to filter recall results | `None` |
| `recall_tags_match` | Tag matching: `any`, `all`, `any_strict`, `all_strict` | `any` |

## Requirements

- Python 3.10+
- `langchain-core >= 0.3.0`
- `hindsight-client >= 0.4.0`
- `langgraph >= 0.3.0` *(only for nodes pattern — install with `pip install hindsight-langgraph[langgraph]`)*

## Documentation

- [Integration docs](https://hindsight.vectorize.io/sdks/integrations/langgraph)
- [Cookbook: ReAct agent with memory](https://docs.hindsight.vectorize.io/cookbook/recipes/langgraph-react-agent)
- [Hindsight API docs](https://docs.hindsight.vectorize.io)
