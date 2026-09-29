---
title: "Guide: Add Windsurf Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, windsurf, coding-agents, memory]
description: "Add Windsurf memory with Hindsight using the hindsight-windsurf CLI, which wires the Hindsight MCP server into Windsurf plus an always-on recall/retain rule so Cascade remembers across tasks."
image: /img/guides/guide-windsurf-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add Windsurf Memory with Hindsight](/img/guides/guide-windsurf-memory-with-hindsight.svg)

If you want **Windsurf memory with Hindsight**, the cleanest setup is the `hindsight-windsurf` CLI. One `init` command connects Windsurf's Cascade agent to the Hindsight MCP server and adds an always-on rule telling the agent to recall relevant memory before a task and retain durable facts as it works. That gives Windsurf long-term memory across tasks instead of forcing every new conversation to rediscover the same project context.

This is a good fit for Windsurf because Windsurf runs MCP servers and applies workspace rules to every Cascade request. The integration uses both: it registers the Hindsight MCP endpoint so Cascade gets `recall` / `retain` / `reflect` tools, and it writes a small `trigger: always_on` rule that tells the agent to recall first and retain what it learns. Recall runs at query time against your actual message, so from your seat it is automatic.

This guide walks through installing the CLI, wiring it into Windsurf, understanding how the MCP-plus-rule approach works, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-windsurf`.
> 2. `cd` into your project.
> 3. Run `hindsight-windsurf init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-memory`.
> 4. Reload Windsurf (or refresh MCP servers in Cascade) so the `hindsight` tools load.
> 5. Verify that a later task remembers what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- Windsurf (Codeium) installed and working, with the Cascade agent
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server
- Python with `pip`, so you can install the `hindsight-windsurf` CLI

## Step 1: Install the CLI

Install the integration CLI with pip.

```bash
pip install hindsight-windsurf
```

`hindsight-windsurf` does not run in place of Windsurf. It is a one-time setup tool: `init` wires the Hindsight MCP server and rule into Windsurf's config, and then Cascade uses the tools on its own.

## Step 2: Wire Hindsight into Windsurf

From your project directory, run `init` with your Hindsight key and a bank id.

```bash
cd your-project
hindsight-windsurf init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-memory
```

`init` adds the `hindsight` MCP server to `~/.codeium/windsurf/mcp_config.json` (Windsurf's single global MCP config) and writes the recall/retain rule to `./.windsurf/rules/hindsight.md`. Reload Windsurf (or refresh MCP servers in Cascade), and the `hindsight` server's tools become available.

For a self-hosted Hindsight server, point at it with `--api-url` instead of a Cloud token (no token needed for an open local server):

```bash
hindsight-windsurf init --api-url http://localhost:8888 --bank-id my-memory
```

If your `mcp_config.json` isn't plain JSON, `init` won't rewrite it — it prints the entry to paste yourself. You can also run `hindsight-windsurf init --print-only` anytime to get the MCP snippet and rule text without writing anything.

## How the integration uses memory

Windsurf supports two things this integration uses, and it wires up both:

- **MCP server:** `init` registers the Hindsight MCP endpoint under `mcpServers` in `~/.codeium/windsurf/mcp_config.json`. Windsurf connects to remote servers via a `serverUrl` field with optional headers, so the Hindsight endpoint connects directly — no bridge needed — giving Cascade `recall` / `retain` / `reflect` tools.
- **Always-on rule:** `init` writes a rule with `trigger: always_on` frontmatter to `./.windsurf/rules/hindsight.md`. Windsurf includes that rule in every Cascade request in the workspace, so the agent is told to recall relevant memory first and retain durable facts as it goes.

Recall runs at query time against your actual message, so it is precise with no lag. The tradeoff is that it relies on the agent following the "recall first" instruction rather than the editor enforcing it.

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Checking and removing the setup

The CLI has three commands so you can inspect or reverse the wiring:

| Command | Description |
| --- | --- |
| `hindsight-windsurf init` | Add the MCP server + recall/retain rule |
| `hindsight-windsurf status` | Show whether the server + rule are configured |
| `hindsight-windsurf uninstall` | Remove the server + rule |

Run `hindsight-windsurf status` after `init` to confirm both the MCP server and the rule are installed.

## Verify that memory is working

A good test sequence is:

1. run `hindsight-windsurf init` in a project and reload Windsurf
2. in Cascade, do a task and let the agent record a decision or convention
3. start a new task or conversation
4. ask about the earlier decision

For example:

- one task refactors the auth module and notes why the retry logic changed
- a later task asks what the current auth conventions are

If Cascade recalls the earlier decision, the setup is working.

## Common mistakes

### Forgetting to reload Windsurf

The `hindsight` tools only appear after you reload Windsurf or refresh MCP servers in Cascade. If the tools are missing, reload first.

### Editing a non-JSON mcp_config.json by hand

If your `mcp_config.json` isn't plain JSON, `init` won't rewrite it — it prints the entry to paste. Add that entry yourself rather than expecting the file to change.

### Expecting the editor to force recall

Recall and retain run through MCP tools the agent calls, guided by the always-on rule. The editor doesn't enforce it, so if the agent skips recall, prompt it to recall relevant memory first.

### Forgetting the API token on Cloud

Hindsight Cloud requires a key. Pass `--api-token` (or set `HINDSIGHT_API_TOKEN`). A self-hosted open local server needs no token.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — point `init` at it with `--api-url` (no token needed for an open local server).

### Does this change how I use Windsurf?

No. `init` is a one-time setup command. After that you use Cascade normally, and it calls the Hindsight tools on its own, guided by the always-on rule.

### How is memory scoped?

By the bank id you pass to `init` with `--bank-id` (the bank defaults to `windsurf`). Use the same bank across tools to share a project's memory.

### Is this similar to other coding-agent integrations?

Yes in spirit. Windsurf uses an MCP server plus an always-on rule instead of a wrapper command, but the recall-before / retain-after pattern is the same. See the [full configuration options](https://hindsight.vectorize.io/docs/integrations/windsurf) in the integration docs.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Windsurf integration docs](https://hindsight.vectorize.io/docs/integrations/windsurf)
