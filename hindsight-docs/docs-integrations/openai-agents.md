---
sidebar_position: 13
title: "OpenAI Agents SDK Persistent Memory with Hindsight | Integration"
description: "Add long-term memory to OpenAI Agents SDK agents with Hindsight. Provides FunctionTool instances for retain, recall, and reflect that plug directly into the Agents SDK async runtime."
---

# OpenAI Agents SDK

Persistent long-term memory for [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) agents via Hindsight. Provides `FunctionTool` instances that plug directly into the OpenAI Agents SDK `Agent`.

## Features

- **Memory Tools** — retain, recall, and reflect as OpenAI Agents SDK `FunctionTool` instances compatible with `Agent(tools=[...])`
- **Async-Native** — Uses `aretain`, `arecall`, `areflect` directly — works seamlessly in the Agents SDK async runtime
- **Auto-Inject Memories** — Use `memory_instructions()` to automatically inject relevant memories into the agent's system prompt every turn
- **Selective Tools** — Include only the tools you need with `include_retain/recall/reflect` flags
- **Tag-Based Scoping** — Partition memories by topic, session, or user with tags
- **Global Configuration** — Configure once with `configure()`, create tools anywhere

## Installation

```bash
pip install hindsight-openai-agents openai-agents
```

`hindsight-openai-agents` pulls in `openai-agents` and `hindsight-client`.

## Quick Start

```python
import asyncio
from agents import Agent, Runner
from hindsight_client import Hindsight
from hindsight_openai_agents import create_hindsight_tools

async def main():
    client = Hindsight(base_url="http://localhost:8888")
    await client.acreate_bank(bank_id="user-123")

    tools = create_hindsight_tools(client=client, bank_id="user-123")

    agent = Agent(
        name="assistant",
        instructions="You are a helpful assistant with long-term memory. Use hindsight_retain to store important facts. Use hindsight_recall to search memory before answering.",
        tools=tools,
    )

    # Store a memory
    result = await Runner.run(agent, "Remember that I prefer dark mode")
    print(result.final_output)

    # Hindsight processes retained content asynchronously (fact extraction,
    # entity resolution, embeddings). A brief pause ensures memories are
    # searchable before the next recall. In production, this delay is only
    # needed when retain and recall happen back-to-back in the same script.
    await asyncio.sleep(3)

    # Recall it later
    result = await Runner.run(agent, "What are my UI preferences?")
    print(result.final_output)

    # Clean up
    await client.aclose()

asyncio.run(main())
```

:::tip Jupyter Notebooks
If you're running in a Jupyter notebook, you don't need `asyncio.run()` — just use `await` directly in cells since the notebook already has an active event loop.
:::

The agent gets three tools it can call:

- **`hindsight_retain`** — Store information to long-term memory
- **`hindsight_recall`** — Search long-term memory for relevant facts
- **`hindsight_reflect`** — Synthesize a reasoned answer from memories

## Auto-Inject Memories with `memory_instructions()`

Instead of relying on the agent to call `hindsight_recall` explicitly, you can auto-inject relevant memories into the system prompt on every turn:

```python
from hindsight_openai_agents import create_hindsight_tools, memory_instructions

agent = Agent(
    name="assistant",
    instructions=memory_instructions(
        client=client,
        bank_id="user-123",
        base_instructions="You are a helpful assistant with long-term memory.",
    ),
    tools=create_hindsight_tools(
        client=client,
        bank_id="user-123",
        include_recall=False,  # recall handled by memory_instructions
    ),
)
```

`memory_instructions()` returns an async callable compatible with `Agent(instructions=...)`. On each turn it recalls relevant memories and appends them to your base instructions. If recall fails or returns nothing, it gracefully falls back to `base_instructions` alone.

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
from hindsight_openai_agents import configure, create_hindsight_tools

configure(
    hindsight_api_url="http://localhost:8888",
    api_key="your-api-key",       # Or set HINDSIGHT_API_KEY env var
    budget="mid",                  # Recall budget: low/mid/high
    max_tokens=4096,               # Max tokens for recall results
    tags=["env:prod"],             # Tags for stored memories
    recall_tags=["scope:global"],  # Tags to filter recall
    recall_tags_match="any",       # Tag match mode
)

# Now create tools without passing client — uses global config
tools = create_hindsight_tools(bank_id="user-123")
```

## Memory Scoping with Tags

Use tags to partition memories by topic, session, or user:

```python
# Store memories tagged by source
tools = create_hindsight_tools(
    client=client,
    bank_id="user-123",
    tags=["source:chat", "session:abc"],
    recall_tags=["source:chat"],
    recall_tags_match="any",
)
```

## Production Patterns

### Error Handling

Tools surface errors to the agent as tool error results. The OpenAI Agents SDK catches exceptions from tools automatically and returns them as error strings, allowing the agent to handle failures gracefully:

```python
from hindsight_openai_agents.errors import HindsightError

# The agent will see error messages and can decide how to proceed
result = await Runner.run(agent, "What do you remember about me?")
print(result.final_output)
```

### Bank Lifecycle

Create banks before first use and clean up when done:

```python
async def main():
    client = Hindsight(base_url="http://localhost:8888")

    # Create bank (idempotent)
    await client.acreate_bank(bank_id="user-123")

    tools = create_hindsight_tools(client=client, bank_id="user-123")
    # ... use tools ...

    # Optional: delete bank when no longer needed
    await client.adelete_bank(bank_id="user-123")
```

### Multi-Agent Workflows

Give each agent its own memory bank, or share a bank across agents:

```python
# Per-agent memory
researcher_tools = create_hindsight_tools(client=client, bank_id="researcher-memory")
writer_tools = create_hindsight_tools(client=client, bank_id="writer-memory")

# Shared memory
shared_tools = create_hindsight_tools(
    client=client,
    bank_id="team-shared",
    tags=["team:content"],
)
```

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

### `memory_instructions()`

| Parameter | Default | Description |
|---|---|---|
| `bank_id` | *required* | Hindsight memory bank to recall from |
| `base_instructions` | `""` | Static instructions prepended before memories |
| `client` | `None` | Pre-configured Hindsight client |
| `hindsight_api_url` | `None` | API URL (used if no client provided) |
| `api_key` | `None` | API key (used if no client provided) |
| `query` | `"relevant context about the user"` | The recall query to find relevant memories |
| `budget` | `"mid"` | Recall budget level (low/mid/high) |
| `max_results` | `5` | Maximum number of memories to include |
| `max_tokens` | `4096` | Maximum tokens for recall results |
| `prefix` | `"\n\nRelevant memories:\n"` | Text prepended before the memory list |
| `tags` | `None` | Tags to filter when searching |
| `tags_match` | `"any"` | Tag matching mode |

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

## Requirements

- Python >= 3.10
- openai-agents >= 0.7.0
- hindsight-client >= 0.4.0
