---
title: "Guide: Add Zed Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, zed, coding-agents, memory]
description: "Add Zed memory with Hindsight by wiring the Agent Panel to the Hindsight MCP server, so the assistant recalls relevant memory before a task and retains durable facts as it goes."
image: /img/guides/guide-zed-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add Zed Memory with Hindsight](/img/guides/guide-zed-memory-with-hindsight.svg)

If you want **Zed memory with Hindsight**, the cleanest setup is the `hindsight-zed` package. One command wires Zed's Agent Panel to the Hindsight MCP server and adds a rule telling the agent to use it, so the assistant recalls relevant memory at the start of a task and retains durable facts as it goes. That gives the Zed AI assistant long-term memory instead of starting every conversation from scratch.

This is a good fit for Zed because Zed has no pre-prompt hook, but it does support MCP context servers and a global instructions file. The integration uses both: it registers the Hindsight MCP server under `context_servers` in `settings.json`, giving the agent `recall` / `retain` / `reflect` tools, and it adds a small rule to `~/.config/zed/AGENTS.md` telling the agent to recall first and retain what it learns. Recall happens at query time against your actual message, so there's no lag, and from your seat it's automatic.

This guide walks through installing the package, running `init` to wire everything up, understanding how the MCP tools and the always-on rule work together, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-zed`.
> 2. Run `hindsight-zed init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-memory`.
> 3. Restart Zed and open the Agent Panel — the `hindsight` server should show a green dot.
> 4. The always-on rule tells the agent to recall first and retain what it learns.
> 5. Verify that a later conversation remembers what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- Zed installed with its AI assistant / Agent Panel
- Node.js installed, since the MCP server is connected through the `mcp-remote` stdio bridge (run via `npx`)
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server

## Step 1: Install the package

Install the `hindsight-zed` package.

```bash
pip install hindsight-zed
```

`hindsight-zed` is a small CLI that configures Zed to use Hindsight. It does not change how Zed works — it wires the Agent Panel to the Hindsight MCP server and adds a recall/retain rule.

## Step 2: Wire Zed to Hindsight

Run `init` with your Hindsight API key and the bank you want to use:

```bash
hindsight-zed init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-memory
```

`init` adds the `hindsight` MCP server to `~/.config/zed/settings.json` and the recall/retain rule to `~/.config/zed/AGENTS.md`. Restart Zed, open the Agent Panel, and the `hindsight` server should show a green dot.

For a self-hosted Hindsight server, point at it with `--api-url` instead:

```bash
hindsight-zed init --api-url http://localhost:8888 --bank-id my-memory
```

An open local server needs no token.

If your `settings.json` contains comments (JSONC), `init` won't rewrite it — it prints the exact `context_servers` entry for you to paste instead. You can see that snippet any time without writing the file:

```bash
hindsight-zed init --print-only
```

## Step 3: Confirm the wiring

Use the `status` command to check whether the server and rule are configured:

```bash
hindsight-zed status
```

The available commands are:

| Command | Description |
| --- | --- |
| `hindsight-zed init` | Add the MCP server + recall/retain rule |
| `hindsight-zed status` | Show whether the server + rule are configured |
| `hindsight-zed uninstall` | Remove the server + rule |
| `hindsight-zed init --print-only` | Print the config to add manually |

## How the integration uses memory

Zed has no pre-prompt hook, so the integration works at the two points Zed does expose:

- **MCP context servers:** Zed runs MCP servers configured under `context_servers` in `settings.json` and surfaces their tools in the Agent Panel. `hindsight-zed` registers the Hindsight MCP server there, giving the agent `recall` / `retain` / `reflect` tools. Because Zed doesn't yet have native HTTP-MCP transport, the server is connected through the [`mcp-remote`](https://www.npmjs.com/package/mcp-remote) stdio bridge, run via `npx` — which is why Node.js is required.
- **A global instructions file** (`~/.config/zed/AGENTS.md`) that Zed includes in every conversation. The integration adds a small rule there, inside a fenced `<!-- HINDSIGHT -->` block that leaves the rest of the file untouched, telling the agent to recall first and retain durable facts.

Recall and retain run through the MCP tools the agent calls, guided by the always-on rule. That makes recall query-time precise — it runs against your actual message with no lag — with the tradeoff that it relies on the agent following the "recall first" instruction rather than the editor enforcing it.

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Verify that memory is working

A good test sequence is:

1. run `hindsight-zed status` to confirm the server and rule are configured
2. open the Agent Panel in Zed and check the `hindsight` server shows a green dot
3. in one conversation, tell the agent a durable fact or convention and let it retain
4. start a new conversation
5. ask about the earlier fact

For example:

- conversation one establishes an auth convention and asks the agent to remember it
- conversation two asks what the current auth conventions are

If the agent recalls the earlier fact, the setup is working.

## Common mistakes

### Node.js not installed

The MCP server runs through the `mcp-remote` stdio bridge via `npx`, so Node.js must be installed or the `hindsight` server won't connect.

### Editing a JSONC settings file

If your `settings.json` has comments, `init` won't rewrite it — it prints the entry to paste. Paste that snippet, or run `hindsight-zed init --print-only` to see it again.

### Not restarting Zed

The `hindsight` server and rule are picked up on restart. If the green dot isn't showing, restart Zed and reopen the Agent Panel.

### Expecting the editor to force recall

Recall relies on the agent following the always-on "recall first" rule, not on the editor enforcing it. If a task didn't recall, remind the agent to recall.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — point at it with `--api-url http://localhost:8888` (no token needed for an open local server).

### Does this change how I use Zed?

No. The integration wires the Agent Panel to the Hindsight MCP server and adds a rule. You keep using Zed's AI assistant the same way.

### How is memory scoped?

By bank. Pass `--bank-id` at `init` (the default bank id is `zed`).

### Is this similar to other coding-agent integrations?

Yes in spirit. Zed uses MCP context servers and a global instructions file instead of a wrapper or hook, but the recall-before / retain-after pattern is the same.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Zed integration docs](https://hindsight.vectorize.io/docs/integrations/zed)
