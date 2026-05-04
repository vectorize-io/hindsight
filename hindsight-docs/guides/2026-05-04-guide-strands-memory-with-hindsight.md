---
title: "Guide: Add Strands Persistent Memory with Hindsight"
authors: [benfrank241]
date: 2026-05-04T15:00:00Z
tags: [how-to, strands, agent-frameworks, memory]
description: "Add Strands persistent memory with Hindsight using native tools, optional memory instructions, and stable per user banks across agent sessions."
image: /img/guides/guide-strands-memory-with-hindsight.png
hide_table_of_contents: true
---

![Guide: Add Strands Persistent Memory with Hindsight](/img/guides/guide-strands-memory-with-hindsight.png)

If you want **Strands persistent memory with Hindsight**, the simplest pattern is to create Hindsight tools for retain, recall, and reflect, then optionally add recalled memory to the system prompt with `memory_instructions()`. That gives a Strands agent durable continuity across sessions while keeping the rest of the SDK usage familiar.

This is a natural fit because Strands agents already treat tools as plain functions. Hindsight can plug into that model without adding a separate memory daemon inside the agent runtime.

If you want the underlying reference open while you work, keep [the Strands integration docs](https://hindsight.vectorize.io/docs/integrations/strands), [the docs home](https://hindsight.vectorize.io/docs), [the quickstart guide](https://hindsight.vectorize.io/docs/quickstart), [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall), and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain) nearby.

<!-- truncate -->

> **Quick answer**
>
> 1. Install the Strands integration or plugin.
> 2. Point it at Hindsight Cloud or a local Hindsight API.
> 3. Wire memory into your Strands runtime with a stable bank ID.
> 4. Store one preference or project fact, then start a fresh run.
> 5. Confirm that recall brings the earlier context back automatically.

## Why this setup works

The Strands SDK is already opinionated about tools and prompts, so Hindsight only needs two insertion points: tool functions for explicit memory actions, and optional injected instructions for automatic recall. That gives you a small, predictable integration surface.

## Prerequisites

- A working Strands agent
- Python and `hindsight-strands` installed
- A bank ID scheme that remains stable for the same user or project

## Step 1: Install the integration

```bash
pip install hindsight-strands
```

## Step 2: Connect Strands to Hindsight

```python
from hindsight_strands import configure

configure(
    hindsight_api_url="http://localhost:8888",
    budget="mid",
    max_tokens=4096,
)
```

## Step 3: Wire memory into your runtime

```python
from strands import Agent
from hindsight_strands import create_hindsight_tools, memory_instructions

tools = create_hindsight_tools(
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
)

memories = memory_instructions(
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
)

agent = Agent(
    tools=tools,
    system_prompt=f"You are a helpful assistant.

{memories}",
)
```

If you do not want automatic injection, remove `memory_instructions()` and let the agent call recall explicitly when needed.

## Step 4: Choose the right bank strategy

Per user banks are usually right for assistants. Per project banks are better when the same user moves between unrelated workstreams. Whatever you choose, keep the same bank value in both the memory instructions and the memory tools.

## Step 5: Verify that memory is working

1. Store one preference or working fact in the first run.
2. Start a second run with the same bank ID.
3. Ask for the earlier fact and confirm that the agent answers consistently.
4. Test a different bank ID to make sure memory isolation behaves the way you expect.

If the second run can answer with details from the first run, your setup is working. If it cannot, turn on debug logging, check the configured bank ID, and confirm that the retain call actually completed.

## Common mistakes

- Using memory instructions built from one bank while the tools point somewhere else
- Forgetting that automatic injection is optional and must be added explicitly
- Choosing a shared bank when your app needs hard user separation

## FAQ

### Can I use only tool based memory?

Yes. The tools are enough if you want the agent to decide when memory should be queried.

### What does reflect add beyond recall?

Reflect produces a synthesized answer from memory, which is useful when several memories need to be combined.

### Should I configure globally or per call?

Global configuration is convenient for one service. Per call settings are safer when different agents need different memory behavior.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted memory backend
- Read [the full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow [the quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Compare a related workflow in [Agno persistent memory](https://hindsight.vectorize.io/blog/agno-persistent-memory)
