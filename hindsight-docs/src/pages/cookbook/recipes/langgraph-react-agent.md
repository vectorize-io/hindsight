---
sidebar_position: 6
---

# LangGraph ReAct Agent with Long-Term Memory


:::tip Run this notebook
This recipe is available as an interactive Jupyter notebook.
[**Open in GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/blob/main/notebooks/06-langgraph-react-agent.ipynb)
:::


Build a ReAct agent with LangGraph that remembers user preferences and past interactions across conversations using Hindsight memory.

This notebook demonstrates two patterns:
- **Tools pattern**: The agent decides when to store/retrieve memories
- **Nodes pattern**: Memory injection and storage happen automatically as graph steps

## Prerequisites

- Python 3.10+
- A [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) account **or** a self-hosted Hindsight instance
- An OpenAI API key (or any LangChain-supported model)

### Option A: Hindsight Cloud

Sign up at [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io/signup) and grab your API key from the dashboard.

### Option B: Self-Hosted (Docker)

```bash
export OPENAI_API_KEY=your-key

docker run --rm -it --pull always -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  -e HINDSIGHT_API_LLM_MODEL=gpt-4o-mini \
  -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

## Installation

```python
!pip install hindsight-langgraph langchain-openai nest_asyncio python-dotenv -U
```

## Setup

Configure the Hindsight client. Set `HINDSIGHT_API_URL` and `HINDSIGHT_API_KEY` in your `.env` file or environment:

| | Hindsight Cloud | Self-Hosted |
|---|---|---|
| `HINDSIGHT_API_URL` | `https://api.hindsight.vectorize.io` | `http://localhost:8888` |
| `HINDSIGHT_API_KEY` | Your cloud API key | *(not required)* |

> **Note:** You also need `OPENAI_API_KEY` set in your environment or `.env` file for the LangChain model.

```python
import nest_asyncio
nest_asyncio.apply()

import os
from dotenv import load_dotenv

load_dotenv()

HINDSIGHT_API_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")
HINDSIGHT_API_KEY = os.getenv("HINDSIGHT_API_KEY")  # Required for Hindsight Cloud

from hindsight_client import Hindsight

client_kwargs = {"base_url": HINDSIGHT_API_URL}
if HINDSIGHT_API_KEY:
    client_kwargs["api_key"] = HINDSIGHT_API_KEY

client = Hindsight(**client_kwargs)
```

## Pattern 1: Tools — The Agent Decides

Give the agent retain/recall/reflect tools and let it decide when to use memory.
This is best for ReAct agents that need to reason about when memory is relevant.

### Create the Agent

```python
from hindsight_langgraph import create_hindsight_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


async def chat(user_id: str, message: str) -> str:
    """Send a message to the agent with per-user memory."""
    tools = create_hindsight_tools(
        client=client,
        bank_id=f"user-{user_id}",
        tags=["source:chat"],
        budget="mid",
    )

    agent = create_react_agent(
        ChatOpenAI(model="gpt-4o-mini"),
        tools=tools,
        prompt=(
            "You are a helpful assistant with long-term memory. "
            "Use hindsight_retain to store important facts about the user. "
            "Use hindsight_recall to search your memory before answering. "
            "Use hindsight_reflect for thoughtful summaries of what you know."
        ),
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": message}]}
    )
    return result["messages"][-1].content
```

### Create the Memory Bank

```python
await client.acreate_bank("user-alice", name="Alice's Memory")
print("Bank created: user-alice")
```

### Conversation 1: Store Preferences

The agent should recognize important facts and store them using `hindsight_retain`.

```python
response = await chat(
    "alice",
    "Hi! I'm Alice. I'm a data scientist who works with Python and SQL. "
    "I prefer dark mode and use VS Code. Please remember all of this.",
)
print(f"Agent: {response}")
```

```python
import asyncio

# Hindsight processes retained content asynchronously (extracting facts, entities, embeddings).
# The sleep gives the server time to finish before we recall.
await asyncio.sleep(3)
```

### Conversation 2: Recall Preferences

A new graph invocation — but memory persists. The agent should search memory to answer.

```python
response = await chat("alice", "What IDE do I use? And what's my job?")
print(f"Agent: {response}")
```

### Conversation 3: Reflect on Knowledge

The agent uses `hindsight_reflect` to synthesize a thoughtful answer from accumulated memories.

```python
response = await chat(
    "alice",
    "Based on everything you know about me, what tools and setup "
    "would you recommend for a new machine learning project?",
)
print(f"Agent: {response}")
```

## Pattern 2: Nodes — Automatic Memory

Instead of relying on the agent to call tools, add memory as automatic graph steps:
- **Recall node** runs before the LLM, injecting relevant memories as a `SystemMessage`
- **Retain node** runs after, storing the conversation

Best when you always want memory context without relying on the LLM to use tools.

### Build the Graph

```python
from hindsight_langgraph import create_recall_node, create_retain_node
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END


async def llm_node(state: MessagesState):
    """LLM call — memories are already injected as a SystemMessage."""
    model = ChatOpenAI(model="gpt-4o-mini")
    response = await model.ainvoke(state["messages"])
    return {"messages": [response]}


recall = create_recall_node(
    client=client,
    bank_id_from_config="user_id",
    budget="low",
    max_results=5,
)
retain = create_retain_node(
    client=client,
    bank_id_from_config="user_id",
    tags=["source:auto"],
)

builder = StateGraph(MessagesState)
builder.add_node("recall", recall)
builder.add_node("llm", llm_node)
builder.add_node("retain", retain)

builder.add_edge(START, "recall")
builder.add_edge("recall", "llm")
builder.add_edge("llm", "retain")
builder.add_edge("retain", END)

graph = builder.compile()
print("Graph compiled.")
```

### Create a Bank for the Second User

```python
await client.acreate_bank("user-bob", name="Bob's Memory")
print("Bank created: user-bob")
```

### Message 1: Retained Automatically

The retain node stores the human message after the LLM responds.

```python
result = await graph.ainvoke(
    {"messages": [HumanMessage(content="I'm training for a marathon next month")]},
    config={"configurable": {"user_id": "user-bob"}},
)
print(result["messages"][-1].content)
```

```python
# Hindsight processes retained content asynchronously (extracting facts, entities, embeddings).
# The sleep gives the server time to finish before we recall.
await asyncio.sleep(3)
```

### Message 2: Memories Recalled Automatically

The recall node searches Hindsight and injects relevant context before the LLM sees the message.

```python
result = await graph.ainvoke(
    {"messages": [HumanMessage(content="What exercise should I do today?")]},
    config={"configurable": {"user_id": "user-bob"}},
)
print(result["messages"][-1].content)
```

## Cleanup

```python
await client.adelete_bank("user-alice")
await client.adelete_bank("user-bob")
print("Banks deleted.")
```

## Key Takeaways

- **Tools pattern**: The agent decides when to store/retrieve. Best for complex reasoning flows.
- **Nodes pattern**: Memory happens automatically. Best when you always want context injection.
- **Dynamic banks**: Use `bank_id_from_config` or parameterized `bank_id` for per-user isolation.
- **Tags**: Scope memories by source, conversation, or topic for precise recall.

For a simpler LangChain-only approach without LangGraph, see the [LangChain Memory](./07-langchain-memory.ipynb) notebook.