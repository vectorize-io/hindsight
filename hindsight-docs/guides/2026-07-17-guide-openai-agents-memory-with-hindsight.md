---
title: "Guide: Add OpenAI Agents SDK Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, openai-agents, agents, memory]
description: "Add OpenAI Agents SDK memory with Hindsight using the hindsight-openai-agents package, so your agents can retain, recall, and reflect on long-term memory as native FunctionTool instances."
image: /img/guides/guide-openai-agents-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add OpenAI Agents SDK Memory with Hindsight](/img/guides/guide-openai-agents-memory-with-hindsight.svg)

If you want **OpenAI Agents SDK memory with Hindsight**, the cleanest setup is the `hindsight-openai-agents` package. It provides `FunctionTool` instances for retain, recall, and reflect that plug straight into `Agent(tools=[...])`. That gives your agents long-term memory across conversations instead of forcing every new run to start from a blank slate.

This is a good fit for the OpenAI Agents SDK because the SDK is async-native and tool-driven. The package uses the async Hindsight client directly (`aretain`, `arecall`, `areflect`), so it works seamlessly inside the Agents SDK runtime. You can let the agent call the memory tools itself, or auto-inject relevant memories into the system prompt on every turn with `memory_instructions()`.

This guide walks through installing the package, pointing it at your Hindsight backend, wiring the tools into an agent, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-openai-agents openai-agents`.
> 2. Create a `Hindsight` client pointed at your backend and create a bank.
> 3. Build tools with `create_hindsight_tools(client=client, bank_id="...")`.
> 4. Pass them to `Agent(tools=tools)` — the agent gets retain, recall, and reflect.
> 5. Verify that a later run recalls what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- Python 3.10 or newer
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server
- The OpenAI Agents SDK installed (`openai-agents`)

## Step 1: Install the package

Install the integration alongside the Agents SDK itself.

```bash
pip install hindsight-openai-agents openai-agents
```

`hindsight-openai-agents` pulls in `openai-agents` and `hindsight-client`, so you get the memory tools and the Hindsight client in one install.

## Step 2: Create a client and a bank

Point a `Hindsight` client at your backend, then create the memory bank the agent will use. `acreate_bank` is idempotent, so it is safe to call on every startup.

```python
from hindsight_client import Hindsight

client = Hindsight(base_url="http://localhost:8888")
await client.acreate_bank(bank_id="user-123")
```

For Hindsight Cloud, configure your API key with the `HINDSIGHT_API_KEY` environment variable, or pass it explicitly through `configure()` (see Step 4).

## Step 3: Build the tools and attach them to an agent

Create the memory tools with `create_hindsight_tools()` and pass them to your `Agent`:

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
        instructions=(
            "You are a helpful assistant with long-term memory. "
            "Use hindsight_retain to store important facts. "
            "Use hindsight_recall to search memory before answering."
        ),
        tools=tools,
    )

    result = await Runner.run(agent, "Remember that I prefer dark mode")
    print(result.final_output)

    # Hindsight processes retained content asynchronously (fact extraction,
    # entity resolution, embeddings). A brief pause ensures memories are
    # searchable before the next recall. In production, this delay is only
    # needed when retain and recall happen back-to-back in the same script.
    await asyncio.sleep(3)

    result = await Runner.run(agent, "What are my UI preferences?")
    print(result.final_output)

    await client.aclose()

asyncio.run(main())
```

The agent gets three tools it can call:

- **`hindsight_retain`** — store information to long-term memory
- **`hindsight_recall`** — search long-term memory for relevant facts
- **`hindsight_reflect`** — synthesize a reasoned answer from memories

## How the tools use memory

The tools map directly onto Hindsight's core operations, and the OpenAI Agents SDK decides when to call them based on your instructions:

- **Retain:** when the agent calls `hindsight_retain`, the content is stored to the bank. Hindsight extracts facts, resolves entities, and generates embeddings asynchronously.
- **Recall:** when the agent calls `hindsight_recall`, it searches the bank for relevant facts before answering.
- **Reflect:** when the agent calls `hindsight_reflect`, Hindsight synthesizes a reasoned answer from the stored memories.

If you would rather not rely on the agent to call recall explicitly, use `memory_instructions()` to auto-inject relevant memories into the system prompt on every turn. It returns an async callable compatible with `Agent(instructions=...)`; on each turn it recalls relevant memories and appends them to your base instructions, and falls back to `base_instructions` alone if recall fails or returns nothing:

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

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Selecting tools and scoping memory

You can include only the tools you need with the `include_retain`, `include_recall`, and `include_reflect` flags:

```python
tools = create_hindsight_tools(
    client=client,
    bank_id="user-123",
    include_retain=True,
    include_recall=True,
    include_reflect=False,  # omit reflect
)
```

Use tags to partition memories by topic, session, or user:

```python
tools = create_hindsight_tools(
    client=client,
    bank_id="user-123",
    tags=["source:chat", "session:abc"],
    recall_tags=["source:chat"],
    recall_tags_match="any",
)
```

You can also configure once with `configure()` and then create tools anywhere without passing a client — see the [OpenAI Agents SDK integration docs](https://hindsight.vectorize.io/docs/integrations/openai-agents) for the full parameter reference.

## Verify that memory is working

A good test sequence is:

1. build the tools and attach them to an agent
2. run the agent with a fact worth remembering
3. wait a few seconds so retained content becomes searchable
4. run the agent again with a question about that fact
5. confirm the answer reflects what you stored

For example:

- run one: "Remember that I prefer dark mode"
- run two: "What are my UI preferences?"

If the second run answers with the earlier preference, the setup is working.

## Common mistakes

### Forgetting to create the bank

Create the bank with `acreate_bank` before first use. It is idempotent, so calling it on every startup is safe.

### Testing recall too early

Retained content is processed asynchronously. If retain and recall happen back-to-back in the same script, add a brief pause so the memory is searchable before you recall it.

### Expecting recall without instructing the agent

If you rely on the tools, the agent only recalls when it decides to. Tell it to recall before answering, or use `memory_instructions()` to inject memories automatically every turn.

### Mixing unrelated memory in one bank

Each bank is an isolated memory store. Give each agent or user its own `bank_id`, or use tags to partition memories within a shared bank.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — point the `Hindsight` client at your server's `base_url`.

### Do I have to pass a client to every call?

No. Call `configure()` once with your API URL and key, then create tools without passing a client — they use the global configuration.

### Does this work with the async Agents SDK runtime?

Yes. The tools use the async Hindsight client (`aretain`, `arecall`, `areflect`) directly, so they run seamlessly inside the Agents SDK async runtime.

### How is memory scoped across multiple agents?

Give each agent its own `bank_id` for private memory, or share a `bank_id` across agents for shared memory. Tags partition memories further within a bank.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [OpenAI Agents SDK integration docs](https://hindsight.vectorize.io/docs/integrations/openai-agents)
