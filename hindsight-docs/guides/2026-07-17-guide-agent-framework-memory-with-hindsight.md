---
title: "Guide: Add Microsoft Agent Framework Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, agent-framework, agents, memory]
description: "Add Microsoft Agent Framework memory with Hindsight using the hindsight-agent-framework context provider, so every agent run recalls relevant context before it runs and retains the conversation after."
image: /img/guides/guide-agent-framework-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add Microsoft Agent Framework Memory with Hindsight](/img/guides/guide-agent-framework-memory-with-hindsight.svg)

If you want **Microsoft Agent Framework memory with Hindsight**, the cleanest setup is the `hindsight-agent-framework` context provider. You pass a `HindsightProvider` to your agent's `context_providers`, and from then on every agent run recalls relevant memories into the agent's context before it runs and retains the conversation afterward. That gives your agent long-term memory across runs and processes instead of forgetting everything between sessions.

This is a good fit for Agent Framework because the integration plugs in as a context provider — there is no MCP server to run and no tool the model has to remember to call. Recall and retain happen automatically on the framework's `before_run` and `after_run` hooks, and both are best-effort, so a memory hiccup never blocks the agent.

This guide walks through installing the provider, pointing it at your Hindsight backend, understanding how banks scope memory, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-agent-framework`.
> 2. Set `HINDSIGHT_API_KEY` (Cloud) or pass `hindsight_api_url=` (self-hosted).
> 3. Add `HindsightProvider(bank_id="user-123")` to your agent's `context_providers`.
> 4. Run the agent normally — recall and retain happen automatically per run.
> 5. Verify that a later run remembers what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- A Microsoft Agent Framework agent you can add a context provider to
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server
- A Hindsight API key (Cloud) or a self-hosted server URL

## Step 1: Install the provider

Install the integration package.

```bash
pip install hindsight-agent-framework
```

`hindsight-agent-framework` adds a `HindsightProvider` you attach to an agent. It does not change how you build or run agents — it wraps each run so memory is recalled before and retained after.

## Step 2: Point the provider at Hindsight

For Hindsight Cloud, set your API key once:

```bash
export HINDSIGHT_API_KEY=your-hindsight-key
```

Then attach the provider to your agent:

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
await agent.run("Suggest a recipe.", session=session)  # recalls the preference
```

For a self-hosted Hindsight server, run one locally and point the provider at it:

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-openai-key
hindsight-api  # http://localhost:8888
```

```python
HindsightProvider(bank_id="user-123", hindsight_api_url="http://localhost:8888")
```

## How the provider uses memory

Because the provider plugs into Agent Framework's run lifecycle, it works at the two points the framework exposes:

- **Recall (`before_run`):** it queries Hindsight for memories relevant to the user's message and injects them as a `## Memories` block in the agent's instructions, so the model sees them for that run.
- **Retain (`after_run`):** when the run finishes, it retains the user input plus the agent response so future runs build on them.

There is no MCP, and no tool the model has to remember to call — recall and retain are automatic. Both are best-effort, so a memory hiccup never blocks the agent.

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Memory banks

Memories live in a Hindsight **bank**, and you choose the scope with `bank_id` — one bank per user, agent, or session. In the example above, `bank_id="user-123"` gives that user their own isolated memory store.

The provider accepts more than just `bank_id`. `HindsightProvider(bank_id, ...)` also takes `client`, `hindsight_api_url`, `api_key`, `budget` (low/mid/high), `max_tokens`, `context`, `tags`, `recall_tags`, `recall_tags_match`, `mission` (creates the bank with a fact-extraction persona), `auto_recall`, `auto_retain`, and `source_id`. You can set process-wide defaults with `configure(...)`.

For the full list of options, see the [integration source](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/agent-framework).

## Verify that memory is working

A good test sequence is:

1. attach `HindsightProvider(bank_id="user-123")` to an agent
2. run the agent and have it record a preference or fact
3. run the agent again against the same `bank_id`
4. ask about the earlier fact

For example:

- run one tells the agent "Remember that I prefer vegetarian food."
- run two asks it to "Suggest a recipe."

If the second run reflects the earlier preference, the setup is working. Because retain persists to the bank, this holds even across separate processes.

## Common mistakes

### Using a different `bank_id` between runs

Memory is scoped by `bank_id`. If the second run uses a different bank, it will not recall what the first run stored.

### Expecting the model to call a tool

There is no memory tool for the model to invoke. Recall and retain happen automatically on the run hooks — you only attach the provider.

### Forgetting to set credentials

Set `HINDSIGHT_API_KEY` for Cloud, or pass `hindsight_api_url=` (and `api_key=` if needed) to the provider for self-hosting.

### Not attaching the provider to the agent

The provider must be in the agent's `context_providers` list. Building it without attaching it means no recall or retain runs.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — run `hindsight-api` and pass `hindsight_api_url=` to the provider.

### Does this change how I build agents?

No. You attach one context provider and run the agent normally. There is no MCP server and no tool the model has to call.

### How is memory scoped?

By the `bank_id` you pass to `HindsightProvider` — one bank per user, agent, or session.

### Can I turn recall or retain off?

Yes. The provider accepts `auto_recall` and `auto_retain` options, so you can control each independently.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Microsoft Agent Framework integration docs](https://hindsight.vectorize.io/docs/integrations/agent-framework)
