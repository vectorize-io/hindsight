---
sidebar_position: 12
title: "AutoGen Persistent Memory with Hindsight | Integration Guide"
description: "Add long-term memory to AutoGen agents with Hindsight. Provides FunctionTool instances for retain, recall, and reflect that plug directly into AutoGen's AssistantAgent."
---

# AutoGen

Persistent long-term memory for [AutoGen](https://microsoft.github.io/autogen/) agents via Hindsight. Provides `FunctionTool` instances that plug directly into AutoGen's `AssistantAgent`.

## Features

- **Memory Tools** — retain, recall, and reflect as AutoGen `FunctionTool` instances compatible with `AssistantAgent(tools=[...])`
- **Async-Native** — Uses `aretain`, `arecall`, `areflect` directly — works seamlessly in AutoGen's async runtime
- **Selective Tools** — Include only the tools you need with `include_retain/recall/reflect` flags
- **Tag-Based Scoping** — Partition memories by topic, session, or user with tags
- **Global Configuration** — Configure once with `configure()`, create tools anywhere

## Installation

```bash
pip install hindsight-autogen autogen-agentchat "autogen-ext[openai]"
```

`hindsight-autogen` pulls in `autogen-core` and `hindsight-client`. You also need `autogen-agentchat` for `AssistantAgent` and `autogen-ext[openai]` for the OpenAI model client.

## Quick Start

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from hindsight_client import Hindsight
from hindsight_autogen import create_hindsight_tools

async def main():
    client = Hindsight(base_url="http://localhost:8888")
    await client.acreate_bank(bank_id="user-123")

    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    tools = create_hindsight_tools(client=client, bank_id="user-123")

    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=tools,
    )

    # Store a memory
    result = await agent.run(task="Remember that I prefer dark mode")
    print(result.messages[-1].content)

    # Hindsight processes retained content asynchronously (fact extraction,
    # entity resolution, embeddings). A brief pause ensures memories are
    # searchable before the next recall. In production, this delay is only
    # needed when retain and recall happen back-to-back in the same script.
    await asyncio.sleep(3)

    # Recall it later
    result = await agent.run(task="What are my UI preferences?")
    print(result.messages[-1].content)

    # Clean up
    await client.aclose()
    await model_client.close()

asyncio.run(main())
```

:::tip Jupyter Notebooks
If you're running in a Jupyter notebook, you don't need `asyncio.run()` — just use `await` directly in cells since the notebook already has an active event loop.
:::

The agent gets three tools it can call:

- **`hindsight_retain`** — Store information to long-term memory
- **`hindsight_recall`** — Search long-term memory for relevant facts
- **`hindsight_reflect`** — Synthesize a reasoned answer from memories

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
from hindsight_autogen import configure, create_hindsight_tools

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

Tools raise `HindsightError` on failure, which AutoGen surfaces to the agent as a tool error. Wrap agent calls for graceful degradation:

```python
from hindsight_autogen.errors import HindsightError

try:
    result = await agent.run(task="What do you remember about me?")
except HindsightError as e:
    print(f"Memory operation failed: {e}")
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

### Multi-Agent Teams

Give each agent its own memory bank, or share a bank across a team:

```python
# Per-agent memory
researcher_tools = create_hindsight_tools(client=client, bank_id="researcher-memory")
writer_tools = create_hindsight_tools(client=client, bank_id="writer-memory")

# Shared team memory
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
- autogen-core >= 0.4.0
- hindsight-client >= 0.4.0
