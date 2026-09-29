---
title: "Guide: Add Haystack Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, haystack, agents, memory]
description: "Add Haystack memory with Hindsight using the hindsight-haystack package, so your agent can recall relevant memories and retain new facts across turns."
image: /img/guides/guide-haystack-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add Haystack Memory with Hindsight](/img/guides/guide-haystack-memory-with-hindsight.svg)

If you want **Haystack memory with Hindsight**, the cleanest setup is the `hindsight-haystack` package. It gives any Haystack `Agent` persistent long-term memory backed by Hindsight's retain, recall, and reflect APIs. That means your agent can remember facts across turns and sessions instead of starting cold every time.

The package offers two complementary patterns. `create_hindsight_tools(...)` returns a list of Haystack `Tool`s — `retain_memory`, `recall_memory`, and `reflect_on_memory` — that the model can call directly inside a turn. `HindsightMemoryWrapper` is a Haystack `Toolset` that bundles the same tools and adds optional auto-recall (inject relevant memories into the system prompt before each turn) and auto-retain (store user and assistant messages after each turn).

This guide walks through installing the package, pointing it at your Hindsight backend, wiring the tools into an `Agent`, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-haystack`.
> 2. Create a `Hindsight` client pointed at your backend (`base_url` or `HINDSIGHT_API_KEY` for Cloud).
> 3. Build tools with `create_hindsight_tools(client=client, bank_id="user-123")`.
> 4. Pass the tools to a Haystack `Agent`, or use `HindsightMemoryWrapper` for automatic recall/retain.
> 5. Verify that a later turn recalls what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- Python 3.10+
- A Haystack project using `haystack-ai >= 2.12.0`
- `hindsight-client >= 0.4.0` (installed as a dependency of `hindsight-haystack`)
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server

## Step 1: Install the package

Install the integration package:

```bash
pip install hindsight-haystack
```

This pulls in the Hindsight client so you can build memory tools for a Haystack `Agent`.

## Step 2: Point it at Hindsight

Create a `Hindsight` client for your backend. For a self-hosted server, pass the `base_url`:

```python
from hindsight_client import Hindsight

client = Hindsight(base_url="http://localhost:8888")
```

For Hindsight Cloud, the API URL defaults to `https://api.hindsight.vectorize.io` and the API key falls back to the `HINDSIGHT_API_KEY` environment variable. You can also set connection defaults once with `configure()` so you can omit `client=` / `hindsight_api_url=` on every call:

```python
from hindsight_haystack import configure

configure(
    hindsight_api_url="http://localhost:8888",
    api_key="your-api-key",
    budget="mid",
    tags=["source:haystack"],
    context="my-app",
    mission="Track user preferences",
)
```

After `configure()`, you can build tools with just a `bank_id`.

## Step 3: Add memory tools to your Agent

Create the Hindsight tools and pass them to a Haystack `Agent`:

```python
from hindsight_haystack import create_hindsight_tools
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

tools = create_hindsight_tools(
    client=client,
    bank_id="user-123",
    mission="Track user preferences",
)

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
    tools=tools,
    system_prompt=(
        "You are a helpful assistant with long-term memory. "
        "Use retain_memory to store important facts. "
        "Use recall_memory to search memory before answering."
    ),
)

result = agent.run(messages=[ChatMessage.from_user("Remember that I prefer dark mode")])
print(result["messages"][-1].text)
```

If you only want retain and recall, drop the reflect tool:

```python
# Only retain + recall (no reflect)
tools = create_hindsight_tools(
    client=client,
    bank_id="user-123",
    include_reflect=False,
)
```

## How memory works

With `create_hindsight_tools(...)`, the model decides when to call memory. It calls `retain_memory` to store important facts, `recall_memory` to search memory before answering, and `reflect_on_memory` to reason over what it has stored. The `bank_id` scopes those memories, so a per-user or per-agent bank keeps stores isolated.

If you would rather not rely on the model to call the tools, use `HindsightMemoryWrapper`, a `Toolset` that bundles the same tools plus automatic behavior:

```python
from hindsight_haystack import HindsightMemoryWrapper

toolset = HindsightMemoryWrapper(
    client=client,
    bank_id="user-123",
    mission="Track user preferences",
    auto_recall=True,   # Inject memories into the system prompt before each turn
    auto_retain=True,   # Store user + assistant messages after each turn
)

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
    tools=toolset,
    system_prompt="You are a helpful assistant with long-term memory.",
)

# Use toolset.run() for automatic memory behavior
result = toolset.run(agent, messages=[ChatMessage.from_user("I prefer dark mode")])
```

With `auto_recall=True`, relevant memories are injected into the system prompt before each turn. With `auto_retain=True`, the user and assistant messages are stored after each turn. To get that automatic behavior, call `toolset.run(agent, ...)` rather than `agent.run(...)`.

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Verify that memory is working

A good test sequence is:

1. run the agent and tell it something to remember (for example, "Remember that I prefer dark mode")
2. let the agent retain that fact — either through a `retain_memory` tool call or `auto_retain`
3. start a fresh turn (or a new run) and ask about the earlier fact
4. confirm the agent recalls it

For example:

- turn one stores that the user prefers dark mode
- turn two asks what the user's UI preferences are

If the agent surfaces the earlier preference, the setup is working.

## Common mistakes

### Expecting auto behavior from `agent.run()`

Auto-recall and auto-retain run through `HindsightMemoryWrapper.run()`. If you call `agent.run(...)` directly with the wrapper, you get the tools but not the automatic recall/retain — use `toolset.run(agent, ...)`.

### Reusing one bank for every user

Memories are scoped by `bank_id`. If every user shares a bank, their memories mix. Use a distinct `bank_id` (for example per user or per agent) to keep stores isolated.

### Assuming the model always calls the tools

With `create_hindsight_tools(...)` alone, memory is model-driven. If the system prompt does not instruct the model to store and search memory, it may skip the calls. Guide it in the prompt, or switch to `HindsightMemoryWrapper` for automatic behavior.

### Forgetting to point at the right backend

The API URL defaults to Hindsight Cloud (`https://api.hindsight.vectorize.io`). For a self-hosted server, pass `base_url` on the client (or `hindsight_api_url` to `configure()`), or the calls go to Cloud.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — pass its `base_url` to the `Hindsight` client (or `hindsight_api_url` to `configure()`).

### What is the difference between the tools and the wrapper?

`create_hindsight_tools(...)` gives the model memory tools it can call itself. `HindsightMemoryWrapper` bundles the same tools and adds optional auto-recall and auto-retain so memory happens without the model needing to call the tools.

### How is memory scoped?

By `bank_id`. Each bank is an isolated memory store, so use a distinct bank per user or agent.

### Can I use only retain and recall?

Yes. Pass `include_reflect=False` to `create_hindsight_tools(...)` to get just `retain_memory` and `recall_memory`.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Haystack integration docs](https://hindsight.vectorize.io/docs/integrations/haystack)
