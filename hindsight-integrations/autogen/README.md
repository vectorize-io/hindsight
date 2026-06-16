# hindsight-autogen

AutoGen integration for [Hindsight](https://github.com/vectorize-io/hindsight) — persistent long-term memory for AI agents.

Provides `FunctionTool` instances that give [AutoGen](https://microsoft.github.io/autogen/) agents the ability to store, search, and synthesize memories across conversations.

## Prerequisites

- A running Hindsight instance ([self-hosted via Docker](https://github.com/vectorize-io/hindsight#quick-start) or [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup))
- Python 3.10+

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

The agent gets three tools:

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

# Now create tools without passing client
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

## Configuration Reference

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

## Requirements

- Python >= 3.10
- autogen-core >= 0.4.0
- hindsight-client >= 0.4.0

## Documentation

- [Integration docs](https://hindsight.vectorize.io/sdks/integrations/autogen)
- [Cookbook: AutoGen assistant with memory](https://docs.hindsight.vectorize.io/cookbook/recipes/autogen-assistant-agent)
- [Hindsight API docs](https://docs.hindsight.vectorize.io)
