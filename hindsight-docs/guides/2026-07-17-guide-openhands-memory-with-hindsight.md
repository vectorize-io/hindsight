---
title: "Guide: Add OpenHands Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, openhands, coding-agents, memory]
description: "Add OpenHands memory with Hindsight using the hindsight-openhands package, which wires the Hindsight MCP server into OpenHands and adds a recall/retain rule so the agent recalls memory at the start of a task and retains durable facts as it works."
image: /img/guides/guide-openhands-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add OpenHands Memory with Hindsight](/img/guides/guide-openhands-memory-with-hindsight.svg)

If you want **OpenHands memory with Hindsight**, the cleanest setup is the `hindsight-openhands` package. One `init` command wires the Hindsight MCP server into OpenHands' `config.toml` and adds a recall/retain rule to your project's `AGENTS.md`. That gives OpenHands long-term memory across tasks instead of forcing every new task to rediscover the same decisions and conventions.

This is a good fit for OpenHands because OpenHands has native Streamable-HTTP MCP support, so the Hindsight MCP endpoint connects directly with no bridge. OpenHands also loads `AGENTS.md` into the agent's context on every task, which is exactly where the recall/retain rule belongs. With both in place the agent has `recall`, `retain`, and `reflect` tools and, guided by the rule, recalls relevant memory at the start of a task and retains durable facts as it works.

This guide walks through installing the package, pointing it at your Hindsight backend, understanding how the MCP server and rule fit together, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-openhands`.
> 2. `cd your-project`.
> 3. Run `hindsight-openhands init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-project`.
> 4. Start OpenHands in that project — the `recall`/`retain`/`reflect` tools are available and used automatically.
> 5. Verify that a later task remembers what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- OpenHands installed and working
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server
- A project directory you want memory scoped to

## Step 1: Install the package

Install the package with pip.

```bash
pip install hindsight-openhands
```

`hindsight-openhands` is a small setup CLI. It does not run as a background process — it just writes configuration that OpenHands reads.

## Step 2: Wire up the MCP server and rule

From inside the project you want memory for, run `init`:

```bash
cd your-project
hindsight-openhands init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-project
```

`init` merges the `[mcp]` entry into `./config.toml` and writes the recall/retain rule into `./AGENTS.md`. Use a [Hindsight Cloud](https://hindsight.vectorize.io) key, or point at a self-hosted server with `--api-url http://localhost:8888` (no token needed for an open local server).

If `config.toml` can't be parsed safely, `init` prints the exact `[mcp]` snippet to paste instead of touching the file. You can also run `hindsight-openhands init --print-only` anytime to see the snippet and rule without writing anything.

Once that's done, start OpenHands in the project. The Hindsight MCP tools are available and used automatically.

## How the MCP server and rule work together

OpenHands has native Streamable-HTTP MCP support, so the Hindsight MCP endpoint connects directly (no bridge):

```toml
[mcp]
shttp_servers = [
    {url = "https://api.hindsight.vectorize.io/mcp/my-project/", api_key = "hsk_..."}
]
```

That gives the agent the `recall`, `retain`, and `reflect` tools. OpenHands also loads `AGENTS.md` (and repo microagents) into the agent's context on every task — that's where the recall/retain rule lives, telling the agent to recall first and retain durable facts as it works.

- **Recall (start of task):** the rule tells the agent to call `recall` with the user's request to load relevant decisions, preferences, and project context before it acts.
- **Retain (as it works):** when the agent learns a durable fact — an architectural decision, a user preference, a convention — the rule tells it to call `retain` to store it across sessions.

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Memory banks

The MCP server is wired to a bank id — the `--bank-id` you pass to `init` (default: `openhands`). A bank is an isolated memory store, so pointing a project at its own bank keeps its memory separate from unrelated work, while pointing multiple tools at the same bank lets them share project memory.

That means if you also use Hindsight with another editor or coding agent against the same bank, they draw from the same project memory. A convention learned in one tool is available in the others.

The bank can also come from the `HINDSIGHT_OPENHANDS_BANK_ID` environment variable. For the full set of configuration options, see the [package README](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/openhands).

## Verify that memory is working

You can check the wiring first, then the round trip.

Check that the server and rule are configured:

```bash
hindsight-openhands status
```

Then a good test sequence is:

1. start OpenHands in the project and let it record a decision or convention during a task
2. start a new task in the same project
3. ask about the earlier decision

For example:

- one task refactors the auth module and notes why the retry logic changed
- a later task asks what the current auth conventions are

If the agent recalls the earlier decision, the setup is working.

## Common mistakes

### Running init outside the project

`init` writes `./config.toml` and `./AGENTS.md` in the current directory, so `cd` into the project you want memory for before running it.

### Forgetting the Cloud token

Hindsight Cloud requires a token. Pass `--api-token` (or set `HINDSIGHT_API_TOKEN`). A self-hosted open local server needs no token.

### Editing config.toml by hand when init couldn't parse it

If `init` reports it couldn't safely edit `config.toml`, it prints the exact `[mcp]` snippet — paste that in rather than guessing.

### Assuming memory is global across banks

Memory is scoped to the bank the MCP server points at. Do not expect a project on one bank to recall another bank's context.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — point `init` at it with `--api-url` (for example `--api-url http://localhost:8888`), and no token is needed for an open local server.

### Does this run a background process?

No. `hindsight-openhands` only writes configuration. The `recall`/`retain`/`reflect` tools run through OpenHands' native MCP support.

### How is memory scoped?

By bank id — the `--bank-id` passed to `init` (default `openhands`), which can also come from `HINDSIGHT_OPENHANDS_BANK_ID`.

### How do I remove it?

Run `hindsight-openhands uninstall` to remove the MCP server entry and the recall/retain rule.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [OpenHands integration docs](https://hindsight.vectorize.io/docs/integrations/openhands)
