---
title: "Guide: Add Claude Agent SDK Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, claude-agent-sdk, agents, memory]
description: "Add Claude Agent SDK memory with Hindsight using the hindsight-claude-agent-sdk package, so agents recall relevant memories every turn and retain what they learn."
image: /img/guides/guide-claude-agent-sdk-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add Claude Agent SDK Memory with Hindsight](/img/guides/guide-claude-agent-sdk-memory-with-hindsight.svg)

If you want **Claude Agent SDK memory with Hindsight**, the `hindsight-claude-agent-sdk` package gives you two ways to add persistent long-term memory to Anthropic's Claude Agent SDK. You can expose retain/recall/reflect as MCP tools so the agent decides when to use memory, or you can wire up hooks that inject relevant memories before every turn and retain conversation content afterward — with no explicit tool calls.

This is a good fit for the Claude Agent SDK because the SDK already supports in-process MCP servers and lifecycle hooks. The package plugs into both: `create_hindsight_server` builds an MCP server the agent can call, and `create_memory_hooks` builds hooks that recall and retain automatically. Memory is isolated per agent or user through a `bank_id`, so one agent's memory stays separate from another's.

This guide walks through installing the package, pointing it at your Hindsight backend, choosing between the tools and hooks approaches, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-claude-agent-sdk`.
> 2. Set `HINDSIGHT_API_KEY` (Cloud) or pass `hindsight_api_url` for self-hosted.
> 3. Build a server with `create_hindsight_server(bank_id="my-agent")`.
> 4. Pass it to `ClaudeAgentOptions` as an MCP server, or add `create_memory_hooks` for automatic memory.
> 5. Verify that a later turn recalls what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- The Claude Agent SDK installed and working (`claude-agent-sdk`)
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server
- A `bank_id` to scope memory to a given agent or user

## Step 1: Install the package

Install the integration package.

```bash
pip install hindsight-claude-agent-sdk
```

This adds `create_hindsight_server`, `create_memory_hooks`, and the `configure` helper. It does not change how the Claude Agent SDK works — it provides an in-process MCP server and optional hooks you attach through `ClaudeAgentOptions`.

## Step 2: Point the package at Hindsight

By default the package targets Hindsight Cloud at `https://api.hindsight.vectorize.io`. For Cloud, set your API key:

```bash
export HINDSIGHT_API_KEY="your-api-key"
```

For a self-hosted Hindsight server, pass the URL when you build the server:

```python
server = create_hindsight_server(
    bank_id="my-agent",
    hindsight_api_url="http://localhost:8888",
)
```

You can also set connection and default settings globally with `configure`:

```python
from hindsight_claude_agent_sdk import configure

configure(
    hindsight_api_url="http://localhost:8888",
    api_key="your-api-key",
    budget="mid",
)
```

## Step 3: Give the agent memory tools

Expose retain/recall/reflect as MCP tools so the agent can decide when to use memory:

```python
from claude_agent_sdk import query, ClaudeAgentOptions
from hindsight_claude_agent_sdk import create_hindsight_server

server = create_hindsight_server(
    bank_id="my-agent",
    hindsight_api_url="http://localhost:8888",
)

async for msg in query(
    prompt="Remember that I prefer dark mode. Then check what you know about me.",
    options=ClaudeAgentOptions(
        mcp_servers={"hindsight": server},
        allowed_tools=["mcp__hindsight__*"],
    ),
):
    print(msg)
```

The agent now has retain, recall, and reflect available as tools and calls them when it judges memory is relevant.

## Step 4: Add automatic memory hooks

If you would rather not depend on the agent choosing to call tools, add hooks. They recall relevant memories before each prompt and retain results after each session, with no explicit tool calls:

```python
from claude_agent_sdk import query, ClaudeAgentOptions
from hindsight_claude_agent_sdk import create_hindsight_server, create_memory_hooks

server = create_hindsight_server(bank_id="my-agent", hindsight_api_url="http://localhost:8888")
hooks = create_memory_hooks(bank_id="my-agent", hindsight_api_url="http://localhost:8888")

async for msg in query(
    prompt="Help me refactor the auth module.",
    options=ClaudeAgentOptions(
        mcp_servers={"hindsight": server},
        allowed_tools=["mcp__hindsight__*"],
        hooks=hooks,
    ),
):
    print(msg)
```

You can fine-tune the automatic behavior with `MemoryHookConfig`:

```python
from hindsight_claude_agent_sdk import MemoryHookConfig, create_memory_hooks

hooks = create_memory_hooks(
    bank_id="my-agent",
    hook_config=MemoryHookConfig(
        auto_recall=True,           # inject memories before each prompt
        auto_retain=True,           # save results after each session
        retain_on_tools=["Bash"],   # also retain notable Bash outputs
        recall_max_results=5,       # max memories to inject
        retain_tags=["source:my-app"],
    ),
)
```

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## What you get: tools vs hooks

The package gives you two complementary paths to the same memory backend:

- **Tools (explicit memory):** retain, recall, and reflect are exposed as MCP tools the agent can call on its own when it decides memory is relevant.
- **Hooks (automatic memory):** relevant memories are injected into context before each turn and conversation content is retained after, without the agent needing to call a tool.

You can use either on its own or both together — the tools and hooks share the same `bank_id`, so what one path stores the other can recall.

## Per-agent memory banks

Memory is scoped by the `bank_id` you pass. Give each agent or user its own bank to keep their memory isolated, and reuse the same bank across sessions so memory persists over time.

If you want to share a bank across several agents, point them at the same `bank_id`. For the full set of configuration options, see the [package README](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/claude-agent-sdk).

## Verify that memory is working

A good test sequence is:

1. run a query that stores a fact, for example `"Remember that I prefer dark mode."`
2. let the agent retain it (via the retain tool or auto-retain hook)
3. run a second query in the same bank, for example `"What do you know about me?"`
4. confirm the agent recalls the earlier fact

If the second turn surfaces the fact stored in the first, the setup is working.

## Common mistakes

### Forgetting to allow the tools

The tools are namespaced under `mcp__hindsight__*`. If you register the server but do not include that pattern in `allowed_tools`, the agent cannot call retain/recall/reflect.

### Changing the bank_id between sessions

Memory is scoped per `bank_id`. If you use a different bank on the second run, it will not recall what the first run stored.

### Expecting hooks without registering them

Adding `create_hindsight_server` alone gives the agent tools, not automatic memory. Pass `create_memory_hooks(...)` to `hooks=` if you want recall and retain to happen automatically.

### Pointing at the wrong backend

By default the package targets Hindsight Cloud. For a self-hosted server, set `hindsight_api_url` (or `configure(...)`) so retain and recall reach your deployment.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — pass `hindsight_api_url` to `create_hindsight_server` (or `configure`) pointing at your deployment.

### Should I use tools or hooks?

Use tools if you want the agent to decide when to use memory, and hooks if you want recall and retain to happen automatically every turn. You can use both together with the same `bank_id`.

### How is memory scoped?

Per `bank_id`. Give each agent or user its own bank to keep memory isolated, or share a bank across agents that should share memory.

### Can I fine-tune the automatic behavior?

Yes. Pass a `MemoryHookConfig` to `create_memory_hooks` to control auto-recall, auto-retain, which tool outputs to retain, how many memories to inject, and retain tags.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Claude Agent SDK integration docs](https://hindsight.vectorize.io/docs/integrations/claude-agent-sdk)
