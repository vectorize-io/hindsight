---
title: "Stop Building Microsoft Agent Framework Agents That Forget"
authors: [benfrank241]
slug: "2026/06/18/microsoft-agent-framework-memory"
date: 2026-06-18T12:00
tags: [microsoft-agent-framework, semantic-kernel, memory, persistent-memory, hindsight, agents, tutorial]
description: "Add persistent long-term memory to Microsoft Agent Framework agents with Hindsight. Plug in one context provider and every run recalls relevant memories and retains the conversation automatically. No MCP, no tools the model has to remember to call."
image: /img/blog/microsoft-agent-framework-memory.png
hide_table_of_contents: true
---

![Persistent Memory for Microsoft Agent Framework with Hindsight](/img/blog/microsoft-agent-framework-memory.png)

[Microsoft Agent Framework](https://github.com/microsoft/agent-framework) is Microsoft's open-source successor to Semantic Kernel: a framework for building agents that plan, call tools, and hold a conversation. What it doesn't do out of the box is remember anything once a session ends. Start a new run and the agent is back to square one, with no recollection of who the user is or what was decided last time.

The new Hindsight integration fixes that. It plugs in as a **context provider**, so every agent run automatically recalls the memories relevant to the user's message and retains the conversation afterward. There's no MCP server in the loop and no memory tool the model has to decide to call. Memory just happens.

## TL;DR

<!-- truncate -->

- Microsoft Agent Framework agents are stateless across runs. Each session starts cold.
- The Hindsight integration adds one **context provider**, `HindsightProvider`. `pip install hindsight-agent-framework`, pass it to your agent, done.
- **Recall is automatic and deterministic.** It runs on the framework's `before_run` hook, so memory is injected before the model sees the prompt. There's no tool the model can forget to call.
- After each run, `after_run` retains the exchange, so the next run builds on it.
- Memory lives in a Hindsight **bank** you choose per user, agent, or session, which makes per-user isolation a one-argument change.
- Hindsight Cloud means no infrastructure. [Sign up free.](https://ui.hindsight.vectorize.io/signup)

## Why Microsoft Agent Framework Needs Persistent Memory

Microsoft Agent Framework gives you a clean way to stand up a capable agent: a chat client, instructions, tools, sessions. Within a single session the agent has working context. But that context evaporates when the session ends.

For anything you run more than once against the same user, that's the limitation you hit first. A support agent re-asks for the account details every conversation. A coding assistant re-learns the project's conventions on every reload. A personal assistant forgets the preferences the user stated yesterday. You either build a memory layer yourself (a datastore, embeddings, retrieval, deduplication) or you ship an agent with amnesia.

Hindsight is that memory layer, and the context provider wires it in without changing how you write your agent.

## How It Works

Microsoft Agent Framework lets you attach **context providers** to an agent. A context provider gets two lifecycle hooks, and the Hindsight integration uses both:

| Hook | What Hindsight does |
| --- | --- |
| `before_run` | Recall memories relevant to the user's message and inject them as a `## Memories` block into the agent's instructions. |
| `after_run` | Retain the user input plus the agent's response, so future runs build on them. |

Because recall runs on `before_run`, it is **deterministic**: the memories are in the agent's context before the model generates a token. Compare that to a tool-based approach, where the model has to decide to call a `recall` tool and might not. There's no decision to get wrong and no extra tool-use round trip.

The recalled memories arrive as a labeled block prepended to the agent's instructions:

```
## Memories
Consider the following memories when responding. Ignore any that are not relevant:
- User prefers vegetarian food
- User is allergic to peanuts
- User is cooking for four this week
```

The model reads it as part of its instructions; your application code never has to thread memory through by hand.

## Setup

Install the package:

```bash
pip install hindsight-agent-framework
```

The recommended backend is **Hindsight Cloud**: [sign up free](https://ui.hindsight.vectorize.io/signup), create an API key, and set it once:

```bash
export HINDSIGHT_API_KEY=hsk_your_token
```

Self-hosting works the same way; run the API locally and point the provider at it:

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-openai-key
hindsight-api  # http://localhost:8888
```

```python
HindsightProvider(bank_id="user-123", hindsight_api_url="http://localhost:8888")
```

## Adding Memory to an Agent

This is the entire integration. Attach `HindsightProvider` to your agent's `context_providers` and it does the rest:

```python
from agent_framework.openai import OpenAIChatClient
from hindsight_agent_framework import HindsightProvider

agent = OpenAIChatClient().as_agent(
    name="assistant",
    instructions="You are a helpful assistant.",
    context_providers=[HindsightProvider(bank_id="user-123")],
)

session = agent.create_session()
await agent.run("Remember that I prefer vegetarian food.", session=session)

# ...later, even in a new process:
await agent.run("Suggest a recipe.", session=session)  # recalls the preference
```

The second run, even from a cold process, recalls the vegetarian preference and the recipe suggestion respects it. Nothing in your agent loop changed; the provider handled recall before the run and retain after it.

## Per-User Memory

The `bank_id` is the routing key for memory. A Hindsight bank is just a namespace, and you decide its granularity: one per user, one per agent, one per session, whatever fits your application.

The most common pattern is per-user. Set `bank_id` to the user's ID and each person gets an isolated memory that follows them across sessions and machines:

```python
HindsightProvider(bank_id=f"user-{current_user.id}")
```

A support agent built this way recognizes a returning customer's history. A personal assistant remembers one user's preferences without leaking them into another's. Because the bank is server-side (on Cloud or your own instance), the same memory is available wherever the agent runs.

## Configuration

`HindsightProvider(bank_id, ...)` takes a handful of optional arguments:

| Argument | Default | What it does |
| --- | --- | --- |
| `budget` | `mid` | Recall depth: `low` (fast) / `mid` / `high` (thorough). |
| `max_tokens` | `4096` | Cap on the recalled-memory block injected into instructions. |
| `auto_recall` | `true` | Recall and inject memories before each run. |
| `auto_retain` | `true` | Retain the exchange after each run. |
| `recall_tags` / `recall_tags_match` | none / `any` | Filter recall to tagged memories. |
| `tags` | none | Tags applied to retained memories. |
| `mission` | none | Creates the bank with a fact-extraction persona that steers what gets remembered. |
| `api_key` / `hindsight_api_url` | env | Connection settings, if not using environment variables. |

You can also call `configure(...)` once to set process-wide defaults instead of repeating them on every provider.

## Three Things It Gets Right

**No tool-calling required.** Memory is wired to lifecycle hooks, not exposed as a tool. The model can't forget to use it, and there's no tool-use latency. This is the same reason the [Cline integration](https://hindsight.vectorize.io/blog/2026/06/09/cline-persistent-memory) uses hooks instead of MCP: deterministic beats hopeful.

**It never blocks your agent.** Recall and retain are best-effort. If Hindsight is briefly unreachable, the failure is swallowed and logged, and the agent runs normally without memory for that turn. A memory hiccup never takes down the agent.

**No feedback loop.** The memories injected at the start of a run are not retained back at the end of it. Only the genuine user input and agent response are stored, so recalled context doesn't get re-ingested and amplified over time. (A regression test exists specifically to keep it that way.)

## Recap

| | Microsoft Agent Framework default | With Hindsight |
| --- | --- | --- |
| Memory across runs | None | Automatic, per bank |
| Setup | n/a | One context provider |
| Recall mechanism | n/a | `before_run` hook, injected as instructions |
| Retain mechanism | n/a | `after_run` hook |
| Model tool-calling needed | n/a | No |
| Per-user isolation | n/a | Set `bank_id` |
| Failure behavior | n/a | Best-effort, never blocks the agent |

## Next Steps

- **Hindsight Cloud:** [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io/signup)
- **Integration docs:** [Microsoft Agent Framework + Hindsight](/sdks/integrations/agent-framework)
- **Source:** [vectorize-io/hindsight/hindsight-integrations/agent-framework](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/agent-framework)
- **Why hooks beat tools:** [Cline Persistent Memory](https://hindsight.vectorize.io/blog/2026/06/09/cline-persistent-memory)
