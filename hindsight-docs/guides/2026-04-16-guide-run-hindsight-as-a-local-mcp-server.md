---
title: "Guide: Run Hindsight as a Local MCP Server"
authors: [benfrank241]
date: 2026-04-16
tags: [how-to, mcp, local, memory]
description: "Run Hindsight as a local MCP server with embedded PostgreSQL so Claude, Cursor, and other MCP clients get persistent memory without extra infra."
image: /img/blog/guide-run-hindsight-as-a-local-mcp-server.png
hide_table_of_contents: true
---

![Guide: Run Hindsight as a Local MCP Server](/img/blog/guide-run-hindsight-as-a-local-mcp-server.png)

If you want to **run Hindsight as a local MCP server**, the good news is that the setup is smaller than it sounds. Hindsight can run locally with an embedded PostgreSQL database, expose an MCP endpoint on your machine, and give tools like Claude Code, Claude Desktop, Cursor, and Windsurf persistent memory without a separate external database.

This is a strong pattern for personal use, development, and privacy-focused setups. Instead of exposing memory over a public endpoint or managing a full hosted stack first, you start one local process and connect your MCP client to `localhost`.

This guide walks through the local server startup flow, explains single-bank and multi-bank modes, and shows how to verify that the MCP tools are working before you trust them in real work. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) open while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. Start `hindsight-local-mcp` with your preferred LLM provider.
> 2. Connect your MCP client to `http://localhost:8888/mcp/`.
> 3. Choose multi-bank mode or pin a single-bank URL.
> 4. Restart the client so the memory tools appear.
> 5. Test retain, recall, and reflect with a short memory check.

## Prerequisites

Before you start, make sure you have:

- a machine where you can run the local server
- an LLM provider key, unless you are using a local model like Ollama
- an MCP-compatible client, such as Claude Code, Claude Desktop, Cursor, or Windsurf

If you would rather skip local infrastructure entirely, [Hindsight Cloud](https://hindsight.vectorize.io) is the easier managed alternative.

## Step 1: Start the local MCP server

The simplest local startup command is:

```bash
HINDSIGHT_API_LLM_API_KEY=sk-... uvx --from hindsight-api hindsight-local-mcp
```

If you want to use Ollama instead of a cloud provider:

```bash
HINDSIGHT_API_LLM_PROVIDER=ollama \
HINDSIGHT_API_LLM_MODEL=llama3.2 \
uvx --from hindsight-api hindsight-local-mcp
```

That starts the full Hindsight API locally and exposes the MCP endpoint at:

```text
http://localhost:8888/mcp/
```

Behind the scenes, Hindsight also starts an embedded PostgreSQL database so you do not need to provision a separate database service just to test or use memory locally.

## What the local server actually includes

The local MCP server is not a thin demo wrapper. It runs the full memory API locally, including:

- retain
- recall
- reflect
- mental model tools
- document tools
- bank management tools in multi-bank mode

This is why it works well for both development and day-to-day single-user setups.

For retrieval details, see [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall). For storage details, see [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Step 2: Choose multi-bank or single-bank mode

Hindsight supports both patterns locally.

### Multi-bank mode

Use the root MCP path:

```text
http://localhost:8888/mcp/
```

This exposes all tools, including bank management. It is the right choice when you want to work with multiple banks or let your agent specify the bank per request.

Example for Claude Code:

```bash
claude mcp add --transport http hindsight http://localhost:8888/mcp/
```

### Single-bank mode

Use a bank-pinned URL:

```text
http://localhost:8888/mcp/my-bank/
```

This removes the need to pass a `bank_id` in each tool call and keeps the whole client pinned to one memory bank.

Example:

```bash
claude mcp add --transport http hindsight http://localhost:8888/mcp/my-bank/
```

If you know one client should always use one bank, this mode is simpler.

## Step 3: Connect your MCP client

Any MCP-compatible client can point at the local endpoint.

For clients that use JSON config, add an HTTP MCP server pointing to the local URL.

Typical examples:

- Claude Desktop config file
- Cursor MCP settings
- Windsurf MCP settings
- Claude Code CLI registration

If you are comparing client patterns, the [Claude Code integration](https://hindsight.vectorize.io/docs/integrations/claude-code) and [Adding Memory to Codex with Hindsight](https://hindsight.vectorize.io/blog/adding-memory-to-codex-with-hindsight) are useful related reads.

## Step 4: Verify that the tools are live

A simple check is to ask the client:

> What memory tools do you have available?

You should see retain, recall, and reflect at minimum.

Then run a short memory test:

1. tell the agent to remember a preference or fact
2. open a fresh session or new thread
3. ask for that fact back

If memory is working, recall should surface the stored context without you manually repeating it.

A second test is to ask for synthesis rather than literal recall. That checks reflect, not just search.

## Common issues

### Slow first startup

The first run may need time to initialize the database and local models. Later starts are faster.

### Port 8888 is already in use

Change the port:

```bash
HINDSIGHT_API_LLM_API_KEY=sk-... HINDSIGHT_API_PORT=9000 uvx --from hindsight-api hindsight-local-mcp
```

Then point your MCP client at `http://localhost:9000/mcp/`.

### I stored something, but cannot recall it immediately

Retain is asynchronous. Give the pipeline a few seconds and try again.

### The tools never appear in the client

Restart the client fully. Many MCP clients only refresh the tool list on startup.

### I want more logs

Turn on debug logging:

```bash
HINDSIGHT_API_LLM_API_KEY=sk-... HINDSIGHT_API_LOG_LEVEL=debug uvx --from hindsight-api hindsight-local-mcp
```

## When local MCP is the right fit

Local MCP is a great choice when:

- you want memory on your own machine
- you are testing memory behavior before broader rollout
- you want minimal infrastructure
- you want to avoid exposing a public endpoint

If you eventually want multi-device access or easier OAuth-based client setup, move to [Hindsight Cloud](https://hindsight.vectorize.io).

## FAQ

### Do I need Docker for local MCP?

No. The local MCP server can run directly through the Hindsight tooling.

### Is local MCP only for Claude?

No. Any MCP-compatible client can use it.

### Should I use single-bank or multi-bank mode?

Use single-bank when one client should always use one bank. Use multi-bank when you need flexibility.

### Does the data survive restarts?

Yes, the embedded PostgreSQL data persists locally across restarts.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted alternative
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Compare coding workflows in [Team Shared Memory for AI Coding Agents](https://hindsight.vectorize.io/blog/team-shared-memory-ai-coding-agents)
