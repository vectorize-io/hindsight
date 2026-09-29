---
title: "Guide: Type-Safe, Async Agent Memory with Pydantic AI"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, pydantic-ai, memory]
description: "Add async-native, type-safe long-term memory to Pydantic AI agents with Hindsight — retain, recall, and reflect tools that fit the event loop with no thread-pool hacks."
image: /img/guides/guide-pydantic-ai-type-safe-agent-memory.svg
hide_table_of_contents: true
---

![Guide: Type-Safe, Async Agent Memory with Pydantic AI](/img/guides/guide-pydantic-ai-type-safe-agent-memory.svg)

Pydantic AI is built around **async-native**, strongly typed agents, and memory should follow the same contract. Hindsight's Pydantic AI integration exposes retain, recall, and reflect as async tools that run on the event loop directly — no synchronous client wrapped in a thread pool, no bridging between an async agent and a blocking memory call. The agent stays typed and the flow stays predictable.

That matters because many memory add-ons assume a synchronous client. To use those from an async agent you end up offloading each memory call to a thread pool, which adds latency, hides errors behind executor boundaries, and makes the flow harder to reason about. Hindsight avoids that: its tools are awaitable, so `await agent.run(...)` drives memory on the same loop as everything else the agent does.

This guide is about *design*, not installation. The [setup guide](https://hindsight.vectorize.io/guides/2026/05/04/guide-pydantic-ai-memory-with-hindsight) already covers install and wiring. Here the focus is why async-native memory fits Pydantic AI, where the memory tools attach, and how to keep the whole flow typed and predictable in production.

<!-- truncate -->

> **Quick answer**
>
> 1. Install and connect using the [setup guide](https://hindsight.vectorize.io/guides/2026/05/04/guide-pydantic-ai-memory-with-hindsight).
> 2. Attach memory with `create_hindsight_tools(client=..., bank_id=...)` so retain, recall, and reflect are awaitable tools on the agent.
> 3. Add `memory_instructions(...)` if you want relevant memory injected before every run.
> 4. Keep one stable `bank_id` per user or project so recall stays consistent.
> 5. All memory runs on the event loop — no thread-pool wrappers, no blocking calls.

## Why async-native memory matters

Pydantic AI agents run on `asyncio`. `agent.run(...)` is awaitable, tool calls are awaited, and model calls are non-blocking. The whole point is that a single agent can fan out work concurrently without blocking the event loop.

Memory should participate in that model, not fight it. Hindsight's integration uses Pydantic AI's async tool interface directly, so `hindsight_retain`, `hindsight_recall`, and `hindsight_reflect` are awaited like any other tool. A recall during a run does not stall the loop; other awaited work can proceed while the memory call is in flight.

The alternative — a synchronous memory client called from inside an async agent — blocks the loop for the duration of every memory operation. In a service handling concurrent requests, one recall can hold up unrelated work. Async-native memory sidesteps that entirely.

## No thread-pool hacks

When a memory library only ships a synchronous client, the usual workaround inside an async agent is to push each call onto a thread pool executor. That works, but it costs you:

- **Latency and overhead** from hopping onto a worker thread for every retain or recall.
- **Opaque errors**, since exceptions cross the executor boundary and lose their natural async stack context.
- **Harder reasoning**, because the flow is no longer a straight line of awaited calls.

Hindsight's Pydantic AI tools are awaitable end to end, so none of that is needed. Retain, recall, and reflect are `await`ed directly on the same loop as the agent. The mental model stays simple: the agent awaits its tools, and memory is just another awaited tool.

## Where memory tools attach to a Pydantic AI agent

There are two clean attachment points, and they map onto two different intents.

**Tools** — pass `create_hindsight_tools(...)` to the agent's `tools=[...]`. This gives the agent explicit, awaitable actions it can choose to call: store a fact (`hindsight_retain`), search memory (`hindsight_recall`), or synthesize an answer from memory (`hindsight_reflect`). The model decides when memory is relevant.

**Instructions** — pass `memory_instructions(...)` to the agent's `instructions=[...]`. This recalls relevant memory and injects it into the system prompt *before* the run starts, so context is present without the agent having to decide to fetch it.

```python
from hindsight_client import Hindsight
from hindsight_pydantic_ai import create_hindsight_tools, memory_instructions
from pydantic_ai import Agent

client = Hindsight(base_url="https://api.hindsight.vectorize.io", api_key="hsk_...")

agent = Agent(
    "openai:gpt-4o",
    tools=create_hindsight_tools(client=client, bank_id="user-123"),
    instructions=[memory_instructions(client=client, bank_id="user-123")],
)

result = await agent.run("What do you remember about my preferences?")
print(result.output)
```

Use both when you want automatic context injection *and* explicit memory actions. Use tools only when the agent should decide when to reach for memory. Use instructions only when you want recalled context injected without exposing memory tools to the model. If you only need some of the tools, `include_retain`, `include_recall`, and `include_reflect` let you select which ones attach.

## Connect Pydantic AI to Hindsight

Installation and connection are covered in the standard setup guide — `pip install hindsight-pydantic-ai`, point the client at Hindsight Cloud or a local server, and wire the tools into your agent. Follow it once, then come back here for the design details:

- [Setup guide: Add Pydantic AI Persistent Memory with Hindsight](https://hindsight.vectorize.io/guides/2026/05/04/guide-pydantic-ai-memory-with-hindsight)

## Keeping the memory flow typed and predictable

Predictability comes from a few deliberate choices around configuration and scope.

**Pick one bank strategy and keep it stable.** Recall and retain both key off `bank_id`. Use a stable ID per user for personal assistants, or per project when one user drives several unrelated systems. Rotating the bank ID per request makes the agent look stateless even when the integration is correct.

**Configure once, override deliberately.** You can pass a client to every call, or call `configure(...)` once and create tools without a client each time. Per-call constructor arguments (like `budget` or `tags`) override global config — so if a value surprises you, check whether a per-call argument is winning over your global setup.

**Scope recall with tags.** `recall_tags` and `recall_tags_match` narrow what memory a run can see, which keeps the injected context relevant and the flow reproducible across runs.

Because retain, recall, and reflect are awaited on the loop, the flow reads as a straight sequence of typed, awaited calls — the same shape as the rest of a Pydantic AI agent.

## Verify that memory is working

1. Run the agent once with a stable `bank_id` and have it store a preference or operating rule.
2. Start a fresh run with the same `bank_id` and ask for that detail.
3. Check whether the answer reflects the earlier memory before any explicit tool call is made — that confirms `memory_instructions(...)` injected recalled context.
4. If it does not, inspect the instruction output and confirm recall used the expected bank.

If the second run answers from the first run's memory, the setup is working. For lower-level behavior, see [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Common mistakes

### Wrapping the client in a thread pool anyway

The tools are already awaitable. Awaiting them directly is correct — pushing them onto an executor adds overhead and hides errors for no benefit.

### Using a different bank ID in tools and instructions

Recall then reads from a different bank than retain wrote to, so injected context looks empty. Keep the `bank_id` identical across both.

### Expecting auto-injection from tools alone

Tools give the agent the *option* to call memory. If you want context present before the run without a tool call, add `memory_instructions(...)`.

### Forgetting per-call overrides win

A per-call `budget`, `tags`, or `max_tokens` overrides the global `configure(...)` value. If a setting looks ignored, check for a constructor argument overriding it.

## FAQ

### Why does async-native matter for Pydantic AI specifically?

Because the agent runs on the event loop. Awaitable memory tools keep memory on that loop, so a recall does not block unrelated concurrent work and the flow stays a clean sequence of awaited calls.

### Do I need both tools and memory instructions?

No. Use both for automatic injection plus explicit memory actions, or use whichever one fits your agent design.

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — point the client at your local server (for example `Hindsight(base_url="http://localhost:8888")`).

### Can I limit which memory tools the agent gets?

Yes. `include_retain`, `include_recall`, and `include_reflect` on `create_hindsight_tools(...)` let you attach only the tools you need.

## Next Steps

- Complete install and wiring in the [Pydantic AI setup guide](https://hindsight.vectorize.io/guides/2026/05/04/guide-pydantic-ai-memory-with-hindsight)
- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Pydantic AI integration docs](https://hindsight.vectorize.io/docs/integrations/pydantic-ai)
