---
sidebar_position: 8
---

# LlamaIndex Agents with Long-Term Memory


:::tip Run this notebook
This recipe is available as an interactive Jupyter notebook.
[**Open in GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/blob/main/notebooks/08-llamaindex-react-agent.ipynb)
:::


Build LlamaIndex agents that remember user preferences and past interactions across conversations using Hindsight memory.

This notebook demonstrates three patterns:
- **Automatic Memory**: Use `HindsightMemory` (BaseMemory) for transparent recall/retain — the simplest approach
- **Tool Spec**: Use `HindsightToolSpec` for agent-driven memory tools (retain/recall/reflect)
- **Factory**: Use `create_hindsight_tools()` for quick setup with include/exclude flags

## Prerequisites

- Python 3.10+
- A [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) account **or** a self-hosted Hindsight instance
- An OpenAI API key (or any LlamaIndex-supported model)

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
!pip install hindsight-llamaindex llama-index-llms-openai llama-index-core python-dotenv -U
```

## Setup

Configure the Hindsight client. Set `HINDSIGHT_API_URL` and `HINDSIGHT_API_KEY` in your `.env` file or environment:

| | Hindsight Cloud | Self-Hosted |
|---|---|---|
| `HINDSIGHT_API_URL` | `https://api.hindsight.vectorize.io` | `http://localhost:8888` |
| `HINDSIGHT_API_KEY` | Your cloud API key | *(not required for local)* |

> **Note:** Self-hosted Hindsight does not require an API key by default. If you've configured authentication on your instance, provide the key accordingly.

> **Note:** You also need `OPENAI_API_KEY` set in your environment or `.env` file for the LlamaIndex model.

```python
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

## Pattern 1: Automatic Memory (BaseMemory)

The simplest way to add Hindsight memory. Messages are automatically retained on each turn,
and relevant memories are recalled and injected as context — the agent doesn't need to do anything.

This is the recommended pattern for most use cases.

### Create the Agent with Automatic Memory

```python
from hindsight_llamaindex import HindsightMemory
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent


def create_memory_agent(user_id: str) -> ReActAgent:
    """Create a ReAct agent with automatic per-user memory."""
    memory = HindsightMemory.from_client(
        client=client,
        bank_id=f"user-{user_id}",
        mission="Track user preferences, background, and project context",
        tags=["source:chat"],
        context="llamaindex-cookbook",
    )

    return ReActAgent(
        tools=[],  # No memory tools needed — memory is automatic
        llm=OpenAI(model="gpt-4o-mini"),
        memory=memory,
        system_prompt="You are a helpful assistant. Answer questions using your memory of past conversations.",
        verbose=True,
    )
```

### Conversation 1: Store Preferences

With automatic memory, the agent doesn't need to call any tools — `put()` retains
each message to Hindsight automatically. The `mission` parameter ensures the bank
is created on first use.

```python
# Clean up any leftover bank from a previous run
try:
    await client.adelete_bank("user-alice")
except Exception:
    pass
```

```python
agent = create_memory_agent("alice")
response = await agent.run(
    "Hi! I'm Alice. I'm a data scientist who works with Python and SQL. "
    "I prefer dark mode and use VS Code.",
    max_iterations=10,
)
print(f"\nAgent: {response}")
```

```python
import time

# Hindsight processes retained content asynchronously (extracting facts, entities, embeddings).
# The sleep gives the server time to finish before we recall.
time.sleep(3)
```

### Conversation 2: Automatic Recall

A new agent instance — but memory persists. `get(input)` automatically recalls
relevant memories and prepends them as a system message.

```python
agent = create_memory_agent("alice")
response = await agent.run("What IDE do I use? And what's my job?", max_iterations=10)
print(f"\nAgent: {response}")
```

## Pattern 2: Tool Spec — Agent-Driven Memory

Use `HindsightToolSpec` when you want the agent to **decide** when to use memory.
The agent gets retain/recall/reflect as tools and chooses when to call them.

This gives the agent explicit control — useful when you want it to reflect or
selectively store information.

```python
from hindsight_llamaindex import HindsightToolSpec


def create_tool_agent(user_id: str) -> ReActAgent:
    """Create a ReAct agent with explicit memory tools."""
    spec = HindsightToolSpec(
        client=client,
        bank_id=f"user-{user_id}",
        mission="Track user preferences, background, and project context",
        tags=["source:chat"],
        retain_context="llamaindex-cookbook",
    )
    tools = spec.to_tool_list()

    return ReActAgent(
        tools=tools,
        llm=OpenAI(model="gpt-4o-mini"),
        system_prompt=(
            "You are a helpful assistant with long-term memory. "
            "Use retain_memory to store important facts about the user. "
            "Use recall_memory to search your memory before answering. "
            "Use reflect_on_memory for thoughtful summaries of what you know."
        ),
        verbose=True,
    )
```

### Reflect on Knowledge

The agent uses `reflect_on_memory` to synthesize a thoughtful answer from Alice's accumulated memories.

```python
agent = create_tool_agent("alice")
response = await agent.run(
    "Based on everything you know about me, what tools and setup "
    "would you recommend for a new machine learning project?",
    max_iterations=10,
)
print(f"\nAgent: {response}")
```

## Pattern 3: Factory — Selective Tools

Use `create_hindsight_tools()` for a simpler API. You can include/exclude specific tools
and pass all configuration in one call.

```python
from hindsight_llamaindex import create_hindsight_tools

# Create only retain + recall tools (no reflect)
tools = create_hindsight_tools(
    client=client,
    bank_id="user-alice",
    tags=["source:chat"],
    budget="mid",
    include_reflect=False,
)

print(f"Tools created: {[t.metadata.name for t in tools]}")

agent = ReActAgent(
    tools=tools,
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant with long-term memory.",
    verbose=True,
)
```

```python
response = await agent.run(
    "I also enjoy hiking on weekends and reading sci-fi novels.",
    max_iterations=10,
)
print(f"\nAgent: {response}")
```

```python
time.sleep(3)

response = await agent.run("What do you know about my hobbies?", max_iterations=10)
print(f"\nAgent: {response}")
```

## Selective Tools via `to_tool_list()`

You can also use the tool spec's `to_tool_list(spec_functions=...)` for fine-grained control:

```python
spec = HindsightToolSpec(client=client, bank_id="user-alice")

# Only expose recall and reflect — read-only memory access
read_only_tools = spec.to_tool_list(
    spec_functions=["recall_memory", "reflect_on_memory"]
)

print(f"Read-only tools: {[t.metadata.name for t in read_only_tools]}")
```

## Cleanup

```python
await client.adelete_bank("user-alice")
print("Bank deleted.")
```

## Key Takeaways

- **Automatic Memory** (`HindsightMemory`): The simplest approach — messages are auto-retained and recalled transparently. Best for most use cases.
- **Tool Spec** (`HindsightToolSpec`): Agent-driven memory — the agent decides when to retain, recall, or reflect. Best when you want explicit control.
- **Factory** (`create_hindsight_tools`): Quick setup with include/exclude flags for selective tool exposure.
- **Bank missions**: Use `mission=` to auto-create banks with context for fact extraction — no manual `create_bank` step needed.
- **Per-user banks**: Use `bank_id=f"user-{user_id}"` for per-user memory isolation.
- **Tags & context**: Scope memories by source, conversation, or topic for precise recall. Use `context=` to label the source of retained data.