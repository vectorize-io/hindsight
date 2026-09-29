---
title: "Guide: Add AutoGen Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, autogen, agents, memory]
description: "Add AutoGen memory with Hindsight using the hindsight-autogen tools, so your AssistantAgent can retain, recall, and reflect on long-term memory across conversations."
image: /img/guides/guide-autogen-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add AutoGen Memory with Hindsight](/img/guides/guide-autogen-memory-with-hindsight.svg)

If you want **AutoGen memory with Hindsight**, the cleanest setup is the `hindsight-autogen` package. It gives you `FunctionTool` instances for retain, recall, and reflect that plug directly into AutoGen's `AssistantAgent(tools=[...])`. That means your agent can store what it learns, search it back later, and reason over it — across conversations instead of losing everything when the run ends.

This is a good fit for AutoGen because the framework is built around tools. Rather than a plugin or an external hook, memory becomes three tools the agent decides to call: `hindsight_retain`, `hindsight_recall`, and `hindsight_reflect`. The package is async-native, so it uses Hindsight's async client methods directly and works cleanly in AutoGen's async runtime.

This guide walks through installing the package, creating a memory bank, wiring the tools into an `AssistantAgent`, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-autogen autogen-agentchat "autogen-ext[openai]"`.
> 2. Point a `Hindsight` client at your backend and create a bank with `acreate_bank`.
> 3. Build tools with `create_hindsight_tools(client=client, bank_id="user-123")`.
> 4. Pass those tools to `AssistantAgent(tools=tools)`.
> 5. Verify that a later run recalls what an earlier run stored.

## Prerequisites

Before you start, make sure you have:

- Python 3.10 or newer
- AutoGen installed (`autogen-agentchat` and `autogen-ext[openai]` for the OpenAI model client)
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server

## Step 1: Install the package

Install the integration alongside AutoGen itself.

```bash
pip install hindsight-autogen autogen-agentchat "autogen-ext[openai]"
```

`hindsight-autogen` pulls in `autogen-core` and `hindsight-client`. You also need `autogen-agentchat` for `AssistantAgent` and `autogen-ext[openai]` for the OpenAI model client.

## Step 2: Create the tools and wire them into your agent

Create a Hindsight client, make a bank, build the tools, and pass them to your agent:

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
    # searchable before the next recall.
    await asyncio.sleep(3)

    # Recall it later
    result = await agent.run(task="What are my UI preferences?")
    print(result.messages[-1].content)

    await client.aclose()
    await model_client.close()

asyncio.run(main())
```

If you're running in a Jupyter notebook, you don't need `asyncio.run()` — just use `await` directly in cells since the notebook already has an active event loop.

## What the agent gets

Wiring in the tools gives the agent three memory operations it can call on its own:

- **`hindsight_retain`** — Store information to long-term memory
- **`hindsight_recall`** — Search long-term memory for relevant facts
- **`hindsight_reflect`** — Synthesize a reasoned answer from memories

Because these are ordinary AutoGen `FunctionTool` instances, the model decides when to call them, the same way it decides to call any other tool. For the lower-level behavior behind them, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

If you only want some of them, include just the ones you need:

```python
tools = create_hindsight_tools(
    client=client,
    bank_id="user-123",
    include_retain=True,
    include_recall=True,
    include_reflect=False,  # Omit reflect
)
```

## Step 3: Configure once, or scope with tags

Instead of passing a client to every call, you can configure global defaults once and then create tools anywhere:

```python
from hindsight_autogen import configure, create_hindsight_tools

configure(
    hindsight_api_url="http://localhost:8888",
    api_key="your-api-key",       # Or set HINDSIGHT_API_KEY env var
    budget="mid",                  # Recall budget: low/mid/high
    max_tokens=4096,
    tags=["env:prod"],             # Tags for stored memories
    recall_tags=["scope:global"],  # Tags to filter recall
    recall_tags_match="any",
)

tools = create_hindsight_tools(bank_id="user-123")
```

Tags let you partition memories by topic, session, or user. Store with `tags`, filter recall with `recall_tags`, and control matching with `recall_tags_match`:

```python
tools = create_hindsight_tools(
    client=client,
    bank_id="user-123",
    tags=["source:chat", "session:abc"],
    recall_tags=["source:chat"],
    recall_tags_match="any",
)
```

For multi-agent teams, give each agent its own bank or share one bank across a team — see the [integration docs](https://hindsight.vectorize.io/docs/integrations/autogen) for the full pattern list.

## Verify that memory is working

A good test sequence is:

1. run an agent turn that stores something, e.g. `agent.run(task="Remember that I prefer dark mode")`
2. wait a moment so Hindsight finishes processing the retained content
3. run a later turn that asks about it, e.g. `agent.run(task="What are my UI preferences?")`
4. confirm the answer reflects the earlier fact

If the second run surfaces the preference from the first, memory is wired up correctly.

## Common mistakes

### Forgetting to create the bank

Create the bank with `acreate_bank(bank_id=...)` before first use. It's idempotent, so it's safe to call every run.

### Recalling immediately after retain

Hindsight processes retained content asynchronously. When retain and recall happen back-to-back in the same script, add a brief pause so memories are searchable before you query.

### Not handling memory errors

Tools raise `HindsightError` on failure, which AutoGen surfaces to the agent as a tool error. Wrap agent calls if you want graceful degradation instead of a hard failure.

### Mismatched tags

If you store with one set of `tags` but filter recall with `recall_tags` that don't overlap, recall can come back empty. Keep your store and recall tag conventions aligned.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — point the `Hindsight` client at its `base_url` (or set `hindsight_api_url` via `configure()`).

### Does this work in AutoGen's async runtime?

Yes. The tools are async-native and use Hindsight's async client methods (`aretain`, `arecall`, `areflect`) directly.

### Can I use only recall, without reflect?

Yes. Use the `include_retain`, `include_recall`, and `include_reflect` flags on `create_hindsight_tools` to include only the tools you need.

### How do I share memory across a team of agents?

Point multiple agents at the same `bank_id`, or give each agent its own bank. Tags let you further scope what each agent stores and recalls.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [AutoGen integration docs](https://hindsight.vectorize.io/docs/integrations/autogen)
