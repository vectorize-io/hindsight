---
sidebar_position: 42
title: "OpenBot Shared Agent Memory with Hindsight | Integration"
description: "Give your OpenBot teammates persistent long-term memory that outlives a single thread and is shared across the whole team — connect Hindsight's MCP server to any OpenBot agent."
---

# OpenBot

Give your [OpenBot](https://openbot.run) teammates a memory that lasts, powered by [Hindsight](https://hindsight.vectorize.io). OpenBot runs a team of persistent AI agents on your own computer — Claude, Codex, Grok, Gemini, OpenCode and more. Connect Hindsight as an MCP server and every agent can **recall** what it learned in earlier threads, **retain** new facts as it works, and **reflect** over everything it knows about you and your projects.

## Why this matters

OpenBot agents are persistent, but their context is not: each thread compacts as it fills, and what one agent learns is invisible to the next. OpenBot's built-in memory helps, but it's small (a few kilobytes per agent) and private to one agent. Hindsight adds a durable layer that sits beside it:

- **Recall** is semantic — it finds the relevant fact even when the words don't match.
- **Retain** keeps facts, experiences and decisions across threads and restarts.
- **Reflect** reasons over everything an agent knows to answer a question.
- **One shared bank** means the whole team draws on the same memory, instead of 17 agents each keeping their own stale copy.

## What it does

Connecting the Hindsight MCP server gives every OpenBot agent the Hindsight tools (`recall`, `retain`, `sync_retain`, `reflect`, mental-model and document tools). Point each agent — or your whole team — at a Hindsight **bank**, and:

- **Memory outlives the thread** — facts retained in one session are recalled in the next, after compaction, and after a restart.
- **Shared or isolated, your choice** — give the team one bank for shared memory, or give each agent its own bank for strict isolation. The bank is chosen by the MCP URL.
- **Same data everywhere** — the memory an OpenBot agent writes is the same bank your other Hindsight integrations and automations read, so they see one consistent view.

## Hindsight and OpenBot's built-in memory

OpenBot ships its own built-in memory (the `remember` / `search_memories` tools) — a small, per-agent store. When Hindsight is connected, an agent has **both** sets of memory tools available, and the two stores don't share. Left to its own judgement, an agent may still save some facts to OpenBot's built-in store instead of to Hindsight.

Two things follow:

- **The MCP connection is a capability, not a habit.** Connecting Hindsight makes its tools available to every agent on the server — but an agent only *uses* them when it decides to. To make agents consistently recall from and retain to Hindsight (rather than the built-in store), pair the connection with the **Hindsight Memory skill** (see [Make it automatic](#make-it-automatic-optional)).
- **Hindsight is the shared, durable layer.** Built-in memory is per-agent and small; a Hindsight bank is semantic, long-lived, and shared across every agent pointed at it.

## Installation

> ✨ **Recommended:** [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) — sign up free, create a bank, grab an API key, and skip self-hosting.

### Via the OpenBot marketplace (recommended)

Open the marketplace from the storefront button at the bottom of OpenBot's **left rail** (it is not in the main navigation), then open **Apps**, find **Hindsight**, and click **Connect** ([listing on the web](https://openbot.run/plugins/hindsight)). You are not asked for a URL: the listing is pinned to Hindsight Cloud's multi-bank address, `https://api.hindsight.vectorize.io/mcp`. Choose how to authenticate:

| Option          | What happens                                                                                                                 |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Sign in**     | Sign in to Hindsight Cloud in the browser and pick the organization OpenBot may use. An Owner or Admin has to approve it.      |
| **Use an API key instead** | Paste a key (`hsk_…`) from the [Connect page](https://ui.hindsight.vectorize.io/connect). OpenBot stores it encrypted. |

Then enable the app for the agents that should share the memory.

:::note Banks work differently on this path
Because the listing connects to the **multi-bank** address, agents also get `list_banks`, `create_bank` and `get_bank_stats`, and every memory tool takes an optional `bank_id`. Tell each agent which bank to use, in its instructions or in the task. An agent that names no bank gets one called `default`, created on first use, and memories there are not visible from other banks.

If you want a connection locked to a single bank, with no `bank_id` to pass and no way to reach another bank, add it manually instead.
:::

### Add it manually as a custom MCP server

Use this when you want one connection scoped to exactly one bank, or when you are self-hosting. In OpenBot's **server settings**, add a URL MCP server:

- **URL**: `https://api.hindsight.vectorize.io/mcp/<bank>/`
- **Header**: `Authorization: Bearer hsk_…`

## Self-hosting

Prefer to run Hindsight on your own machine? Start a local server and point the MCP URL at it:

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-openai-key
hindsight-api
```

Then use `http://localhost:8888/mcp/<bank>/` as the MCP URL. A local server with no auth needs no API key — leave that field blank.

## Choosing a bank

How the bank is chosen depends on which path you used above.

**Manual setup.** The bank is part of the MCP URL (`/mcp/<bank>/`), so each connection is scoped to exactly one bank — agents never have to pass a `bank_id`.

- **Team memory** — give every agent the same URL (e.g. `/mcp/team/`).
- **Per-agent memory** — give each agent its own URL (e.g. `/mcp/research-bot/`, `/mcp/ops-bot/`).

**Marketplace listing.** The connection reaches every bank in the account, and the agent names the bank on each call. Put the bank in the agent's instructions, otherwise it falls back to `default`.

See the [MCP server reference](/developer/mcp-server) for the full tool list and bank-selection rules.

## Make it automatic (optional)

Connecting the app gives agents the memory _tools_; the **Hindsight Memory skill** gives them the _habit_ — recall before acting, retain durable facts, and use Hindsight's tools instead of OpenBot's built-in memory. [Download `SKILL.md`](https://hindsight.vectorize.io/skills/openbot/hindsight-memory/SKILL.md), add it to OpenBot's skill library, and enable it for the agents that use Hindsight.

**Enabling it across agents.** A skill is enabled **per agent** — there's no "all agents" switch, and new agents don't inherit it automatically:

- Enable it on each agent that should use Hindsight (from the agent's **Skills** panel, or ask an agent to run `install_local_skill`).
- The MCP server connection, by contrast, is added once in the Local server's settings and is available to **every** agent — so the tools reach everyone; only the skill (the habit) is per-agent.
- For **Claude** agents specifically, a skill placed in `~/.agents/skills/<slug>/` is loaded by every Claude agent automatically. Other providers (OpenCode, Codex, Gemini, …) need the per-agent install.

> The cleanest long-term answer to "every agent should use Hindsight automatically" is Hindsight _replacing_ OpenBot's built-in memory rather than sitting beside it — tracked upstream in [nightly-labs/openbot#3682](https://github.com/nightly-labs/openbot/issues/3682).

## Try it

Ask an agent to remember something, then start a fresh thread and ask it back:

> **Remember that we decided to standardize on Postgres for all new services.**

> _(new thread)_ **What database did we standardize on, and why?**

With the **Hindsight Memory skill** enabled, the agent saves with Hindsight's `sync_retain` and the second thread recalls the decision from Hindsight — even though its own context was empty.

> **Tip:** if the first thread reports it saved to its own "essential memories" and the second thread can't recall it, the agent used OpenBot's built-in memory instead of Hindsight. Enable the memory skill (below) so it uses Hindsight's `retain`/`recall`.
