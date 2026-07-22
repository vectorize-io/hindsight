---
title: "Guide: Add Roo Code Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, roo-code, coding-agents, memory]
description: "Add Roo Code memory with Hindsight using the hindsight-roo-code installer, so every Roo Code task recalls relevant project context before it starts and retains learnings after."
image: /img/guides/guide-roo-code-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add Roo Code Memory with Hindsight](/img/guides/guide-roo-code-memory-with-hindsight.svg)

If you want **Roo Code memory with Hindsight**, the cleanest setup is the `hindsight-roo-code` installer. You run it once, and it registers Hindsight's MCP server with Roo Code and injects custom rules that teach Roo to recall relevant context before each task and retain learnings after. That gives Roo Code long-term memory across coding sessions instead of forcing every new task to rediscover the same project context.

This is a good fit for Roo Code because Roo Code exposes two extensibility mechanisms that pair perfectly with memory: **MCP servers** for tools and **custom rules** for system-prompt injection. The installer uses both. It wires Hindsight's `/mcp` endpoint in as an MCP server, then drops a rules file that tells Roo when to call `recall` and `retain` so memory becomes part of the normal task loop.

This guide walks through installing the CLI, running the installer against your Hindsight backend, understanding how the rules and MCP config work together, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-roo-code`.
> 2. Have a Hindsight backend ready — Hindsight Cloud (default) or a self-hosted server.
> 3. Run `hindsight-roo-code install` from your project directory.
> 4. Restart Roo Code so it picks up the new MCP server and rules.
> 5. Verify that a later task remembers what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- Roo Code installed and working
- A reachable Hindsight backend, [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) or a self-hosted server
- Python and `pip` available to install the CLI

For self-hosting, run Hindsight locally first:

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-openai-key
hindsight-api  # starts on http://localhost:8888
```

## Step 1: Install the CLI

Install the installer package:

```bash
pip install hindsight-roo-code
```

This gives you the `hindsight-roo-code` command, which wires Hindsight into Roo Code's config directory. It does not change how Roo Code works — it registers an MCP server and adds a rules file.

## Step 2: Run the installer

From your project directory, run:

```bash
hindsight-roo-code install
```

By default this targets Hindsight Cloud. For a self-hosted server, pass the local URL:

```bash
hindsight-roo-code install --api-url http://localhost:8888
```

The installer writes two files:

- **`.roo/mcp.json`** — registers Hindsight's `/mcp` endpoint as an MCP server, with `recall` and `retain` auto-approved
- **`.roo/rules/hindsight-memory.md`** — instructions injected into every Roo system prompt

Then restart Roo Code so it loads the new MCP server and rules.

## Step 3: Choose project-local or global

By default the installer writes to `.roo/` in the current directory, so memory is scoped to that project:

```bash
hindsight-roo-code install --project-dir /path/to/project
```

To apply the integration across all your projects, install globally to `~/.roo/`:

```bash
hindsight-roo-code install --global
```

To update the API URL after installation, re-run the installer or edit `.roo/mcp.json` directly.

## How the integration uses memory

Roo Code has two primary extensibility mechanisms — MCP servers for tools and custom rules for system-prompt injection — and this integration uses both:

- **Recall (before):** the rules file instructs Roo to call `recall` when a new task starts, so relevant memories are injected into context automatically.
- **Retain (during and after):** agents can call `retain` mid-task to store significant decisions or discoveries, and the rules file instructs Roo to call `retain` with a summary when the task ends.

Hindsight exposes exactly two tools through its `/mcp` endpoint:

| Tool | Description |
|------|-------------|
| `recall` | Search memory for context relevant to a query |
| `retain` | Store content in memory immediately |

Because both tools are listed under `alwaysAllow` in `.roo/mcp.json`, Roo can invoke them without a per-call approval prompt. For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Verify that memory is working

A good test sequence is:

1. start Hindsight and run the installer
2. open Roo Code in your project
3. check **Settings → MCP Servers** — `hindsight` should show as connected
4. start a task and watch for `recall` in the tool call log
5. let a task record a decision or convention, then start a later task and ask about it

For example:

- one task refactors the auth module and notes why the retry logic changed
- a later task asks what the current auth conventions are

If the later task surfaces the earlier decision, the setup is working.

## Common mistakes

### Not restarting Roo Code after install

The installer writes `.roo/mcp.json` and `.roo/rules/hindsight-memory.md`, but Roo Code loads those at startup. Restart it so the MCP server and rules take effect.

### Pointing at the wrong backend

If you self-host, pass `--api-url http://localhost:8888`. Without it the installer targets Hindsight Cloud, so a running local server would go unused.

### Confusing project-local and global installs

`hindsight-roo-code install` writes to `.roo/` in the current directory; `--global` writes to `~/.roo/`. If memory does not appear in a project, confirm which scope you installed into.

### Expecting memory before the MCP server connects

If **Settings → MCP Servers** does not show `hindsight` as connected, recall will not run. Confirm the connection before judging whether memory works.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — run `hindsight-api` and install with `--api-url http://localhost:8888`.

### Does this change how I use Roo Code?

No. The installer only registers an MCP server and adds a rules file. You use Roo Code exactly as before; recall and retain happen through the normal task loop.

### Can the agent call recall and retain mid-task?

Yes. The rules file drives automatic calls at task start and end, but agents can also call `recall` and `retain` explicitly while working.

### How do I change the API URL later?

Re-run `hindsight-roo-code install` with a new `--api-url`, or edit `.roo/mcp.json` directly.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Roo Code integration docs](https://hindsight.vectorize.io/docs/integrations/roo-code)
