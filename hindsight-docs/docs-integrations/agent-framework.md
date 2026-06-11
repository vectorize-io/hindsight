---
sidebar_position: 8
title: "Microsoft Agent Framework Persistent Memory with Hindsight | Integration Guide"
description: "Add persistent memory to Microsoft Agent Framework agents with Hindsight via a context provider. Recalls relevant context before each run and retains the conversation after — no MCP, no tools to call."
---

# Microsoft Agent Framework

[View Changelog →](/changelog/integrations/agent-framework)

Persistent memory for [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) — the successor to Semantic Kernel — using [Hindsight](https://vectorize.io/hindsight). The integration plugs in as a **context provider**, so every agent run automatically recalls relevant memories into the agent's context and retains the conversation afterward. No MCP, and no tools the model has to remember to call.

## Quick Start

:::tip Recommended: Hindsight Cloud
[Sign up free](https://ui.hindsight.vectorize.io/signup) for a Hindsight Cloud API key — no self-hosting required.
:::

```bash
pip install hindsight-agent-framework
export HINDSIGHT_API_KEY=your-hindsight-key
```

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

## How It Works

| Hook | Behavior |
| --- | --- |
| `before_run` | Recall memories relevant to the user's message and inject them as a `## Memories` block in the agent's instructions. |
| `after_run` | Retain the user input + agent response so future runs build on them. |

Memories live in a Hindsight **bank** — one per user, agent, or session (you choose via `bank_id`). Recall and retain are best-effort: a memory hiccup never blocks the agent.

## Configuration

`HindsightProvider(bank_id, ...)` accepts `hindsight_api_url`, `api_key`, `budget` (low/mid/high), `max_tokens`, `tags`, `recall_tags`, `mission`, `auto_recall`, `auto_retain`, and more. Process-wide defaults can be set with `configure(...)`.

## Self-Hosting

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-openai-key
hindsight-api  # http://localhost:8888
```

```python
HindsightProvider(bank_id="user-123", hindsight_api_url="http://localhost:8888")
```

See the [integration source](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/agent-framework) for full details.
