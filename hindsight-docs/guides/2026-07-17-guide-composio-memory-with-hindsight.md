---
title: "Guide: Add Composio Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, composio, agents, memory]
description: "Add Composio memory with Hindsight by registering retain, recall, and reflect as Composio custom tools, with memory isolated per session user automatically."
image: /img/guides/guide-composio-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add Composio Memory with Hindsight](/img/guides/guide-composio-memory-with-hindsight.svg)

If you want **Composio memory with Hindsight**, the cleanest setup is the `hindsight-composio` package. It exposes Hindsight's retain, recall, and reflect operations as Composio in-process custom tools, so your agent can store and search long-term memory directly through the tools it already calls. That gives Composio agents memory that persists across sessions instead of starting cold every time.

This is a good fit for Composio because the integration registers custom tools via `composio.experimental.tool()` and binds them to a session. The memory bank for each call is the session's `user_id`, so a single registered tool set isolates memory per user automatically, with a configurable fallback bank when a call has no `user_id`.

This guide walks through installing the package, pointing it at your Hindsight backend, registering the tools on a session, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-composio`.
> 2. Set `HINDSIGHT_API_KEY` (Cloud) or pass `hindsight_api_url` (self-hosted).
> 3. Call `register_hindsight_tools(composio, ...)` to build the tool set.
> 4. Create a session with `user_id` — it becomes the Hindsight `bank_id`.
> 5. Verify that a later session recalls what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- Python >= 3.10
- Composio installed and working (composio >= 0.13.1, < 1) with a `COMPOSIO_API_KEY`
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server

## Step 1: Install the package

Install the integration package.

```bash
pip install hindsight-composio
```

This pulls in `hindsight-client` (>= 0.4.0) and gives you `register_hindsight_tools` and `configure`, the two functions you use to wire Hindsight into Composio.

## Step 2: Point the tools at Hindsight

For Hindsight Cloud, set your API key as an environment variable or pass it directly:

```bash
export HINDSIGHT_API_KEY="hsk_..."
```

Then register the tools against your `Composio` instance:

```python
from composio import Composio
from hindsight_composio import register_hindsight_tools

composio = Composio()  # uses COMPOSIO_API_KEY

tools = register_hindsight_tools(
    composio,
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",  # or set HINDSIGHT_API_KEY env var
)
```

For a self-hosted Hindsight server, swap the URL:

```python
tools = register_hindsight_tools(
    composio,
    hindsight_api_url="http://localhost:8888",
)
```

See the [installation guide](https://hindsight.vectorize.io/developer/installation) for self-hosting setup.

## Step 3: Register the tools on a session

Create a Composio session and pass the tools in as custom tools. The `user_id` becomes the Hindsight bank:

```python
session = composio.create(
    user_id="user-123",  # becomes the Hindsight bank_id
    experimental={"custom_tools": tools},
)

# Pass session.tools() to your agent/LLM as usual.
```

The session now has three tools the agent can call:

- **`HINDSIGHT_RETAIN`** — Store information to long-term memory
- **`HINDSIGHT_RECALL`** — Search long-term memory for relevant facts
- **`HINDSIGHT_REFLECT`** — Synthesize a reasoned answer from memories

If you only want some of them, pass `enable_retain`, `enable_recall`, or `enable_reflect` to `register_hindsight_tools`:

```python
tools = register_hindsight_tools(
    composio,
    hindsight_api_url="https://api.hindsight.vectorize.io",
    enable_retain=True,
    enable_recall=True,
    enable_reflect=False,  # omit reflect
)
```

## How the tools use memory

The integration exposes Hindsight's three core operations as Composio custom tools:

- **Retain:** the agent calls `HINDSIGHT_RETAIN` to store information to long-term memory.
- **Recall:** the agent calls `HINDSIGHT_RECALL` to search long-term memory for relevant facts.
- **Reflect:** the agent calls `HINDSIGHT_REFLECT` to synthesize a reasoned answer from memories.

Because the tools are registered directly on the session, the agent decides when to store and when to search — memory is part of the same tool-calling loop it already runs.

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Per-user memory banks

The bank is resolved per call:

1. The session's `user_id` (recommended — one tool set, isolated per user).
2. `default_bank` (passed to `register_hindsight_tools` or `configure`) when a call has no `user_id`.

If neither is available, the tool raises an error. To set a fallback bank:

```python
tools = register_hindsight_tools(
    composio,
    hindsight_api_url="https://api.hindsight.vectorize.io",
    default_bank="shared",  # used only when a session has no user_id
)
```

You can also configure connection details, budgets, and tags once globally instead of passing them every time:

```python
from hindsight_composio import configure, register_hindsight_tools

configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="your-api-key",       # or set HINDSIGHT_API_KEY env var
    default_bank="shared",         # fallback bank when no user_id
    budget="mid",                  # recall/reflect budget: low/mid/high
    max_tokens=4096,               # max tokens for recall results
    tags=["env:prod"],             # tags for stored memories
    recall_tags=["scope:global"],  # tags to filter recall
    recall_tags_match="any",       # any/all/any_strict/all_strict
)

tools = register_hindsight_tools(composio)
```

## Verify that memory is working

A good test sequence is:

1. register the tools and create a session with a `user_id`
2. have the agent call `HINDSIGHT_RETAIN` to store a fact
3. start a new session with the same `user_id`
4. have the agent call `HINDSIGHT_RECALL` for that fact
5. confirm the earlier fact comes back

For example:

- session one stores a user preference during a conversation
- session two, for the same user, recalls that preference on a later request

If the recall surfaces what the earlier session stored, the setup is working.

## Common mistakes

### Forgetting to set a `user_id`

The bank is resolved from the session's `user_id`. Without it and without a `default_bank`, the tool raises an error.

### Expecting cross-user recall

Each `user_id` maps to its own bank. Memory stored under one user is not visible to another user by default — that isolation is the point.

### Targeting the wrong Composio SDK

This integration targets the Composio 0.13.x SDK (`from composio import Composio`) and intentionally excludes the in-progress `1.0.0` rewrite, whose API differs.

### Registering tools you don't want the agent to call

All three tools are enabled by default. If the agent should not store or synthesize, disable them with `enable_retain`, `enable_recall`, or `enable_reflect`.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — pass `hindsight_api_url="http://localhost:8888"` (or your server URL) to `register_hindsight_tools`.

### How is memory scoped?

Per session `user_id`, which maps to the Hindsight `bank_id`. A single registered tool set isolates memory per user automatically.

### Which tools does the agent get?

`HINDSIGHT_RETAIN`, `HINDSIGHT_RECALL`, and `HINDSIGHT_REFLECT` by default. Include any combination via the `enable_*` flags.

### Does Composio's experimental custom-tools API affect this?

Composio's custom-tools API is currently experimental. This integration targets the Composio 0.13.x SDK and is built against that experimental surface.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Composio integration docs](https://hindsight.vectorize.io/docs/integrations/composio)
