---
title: "Guide: Add GitHub Copilot Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, github-copilot, coding-agents, memory]
description: "Add GitHub Copilot memory with Hindsight using the hindsight-copilot CLI, which wires the Hindsight MCP server into VS Code and adds a recall/retain rule so Copilot's agent mode remembers across tasks."
image: /img/guides/guide-github-copilot-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add GitHub Copilot Memory with Hindsight](/img/guides/guide-github-copilot-memory-with-hindsight.svg)

If you want **GitHub Copilot memory with Hindsight**, the cleanest setup is the `hindsight-copilot` CLI. One command wires the Hindsight MCP server into VS Code's `.vscode/mcp.json` and adds a recall/retain rule to `.github/copilot-instructions.md`. That gives Copilot's agent mode long-term memory across tasks instead of forcing every new chat to rediscover the same project context.

This is a good fit for VS Code Copilot because it supports two things the integration uses: MCP servers in `.vscode/mcp.json` (including HTTP servers with headers, so the Hindsight MCP endpoint connects directly), and `.github/copilot-instructions.md`, which Copilot applies to every chat in the workspace. The MCP server gives agent mode `recall`, `retain`, and `reflect` tools; the instructions file holds the rule that tells Copilot to recall relevant memory at the start of a task and retain durable facts as it works.

This guide walks through installing the CLI, running `init` to wire everything up, understanding how the memory bank is scoped, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-copilot`.
> 2. From your project, run `hindsight-copilot init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-project`.
> 3. `init` writes the MCP server into `.vscode/mcp.json` and the rule into `.github/copilot-instructions.md`.
> 4. Reload VS Code, open Copilot Chat in agent mode, and start the `hindsight` MCP server from the chat's tools menu.
> 5. Verify that a later task recalls what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- VS Code with GitHub Copilot, and Copilot Chat in **agent mode**
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server
- A project folder open as your VS Code workspace, since `init` writes into `.vscode/` and `.github/`

## Step 1: Install the CLI

Install the CLI with pip.

```bash
pip install hindsight-copilot
```

`hindsight-copilot` is configuration-only. It does not run at chat time — it wires the Hindsight MCP server and the recall/retain rule into your workspace, and the memory operations run through the MCP server at runtime.

## Step 2: Wire up your project

From inside the project you want memory for, run `init`:

```bash
pip install hindsight-copilot
cd your-project
hindsight-copilot init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-project
```

`init` merges the `servers` entry into `./.vscode/mcp.json` and writes the rule into `./.github/copilot-instructions.md`. The MCP server entry looks like this:

```json
{
  "servers": {
    "hindsight": {
      "type": "http",
      "url": "https://api.hindsight.vectorize.io/mcp/my-project/",
      "headers": { "Authorization": "Bearer hsk_..." }
    }
  }
}
```

Reload VS Code, open Copilot Chat in **agent mode**, and start the `hindsight` MCP server from the chat's tools menu.

For a self-hosted Hindsight server, pass `--api-url http://localhost:8888` (no token needed for an open local server):

```bash
hindsight-copilot init --api-url http://localhost:8888 --bank-id my-project
```

If your `mcp.json` has comments, `init` won't rewrite it — it prints the snippet to paste instead. You can also print the config anytime without touching any files:

```bash
hindsight-copilot init --print-only
```

## How the integration uses memory

VS Code Copilot has no per-prompt wrapper here — instead the integration leans on the two surfaces Copilot exposes:

- **MCP server (tools):** the `hindsight` HTTP MCP server gives Copilot's agent mode the `recall`, `retain`, and `reflect` tools directly. The server is scoped to a single memory bank — the last path segment of the MCP endpoint URL.
- **Instructions rule (behavior):** the recall/retain rule written into `.github/copilot-instructions.md` is applied to every chat in the workspace. Guided by the rule, Copilot recalls relevant memory at the start of a task and retains durable facts as it works.

That combination means you don't have to remember to call the tools yourself — the rule tells Copilot when to reach for them.

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Memory banks

The MCP server is scoped to a single **bank**, which is the last path segment of the MCP endpoint URL. You set it with `--bank-id` (default: `copilot`), so a project's memory stays isolated from unrelated work.

If you also use Hindsight with another editor or coding agent pointed at the same bank, they draw from the same project memory — a convention learned in one tool is available in the others.

You can also set the bank via the `HINDSIGHT_COPILOT_BANK_ID` environment variable, or the API URL and token via `HINDSIGHT_API_URL` and `HINDSIGHT_API_TOKEN`. See the [package README](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/github-copilot) for the full configuration options.

## Verify that memory is working

A good test sequence is:

1. run `hindsight-copilot status` to confirm the MCP server and rule are configured
2. in Copilot agent mode, work on a task and let Copilot record a decision or convention
3. start a fresh chat in the same workspace
4. ask about the earlier decision
5. confirm the recalled memory surfaces it

For example:

- the first task refactors the auth module and notes why the retry logic changed
- a later chat asks what the current auth conventions are

If Copilot recalls the earlier decision, the setup is working.

## Common mistakes

### Not starting the MCP server in the chat

After `init`, you still have to reload VS Code, open Copilot Chat in **agent mode**, and start the `hindsight` MCP server from the chat's tools menu. If the server isn't running, the tools aren't available.

### Editing an mcp.json that has comments

If `.vscode/mcp.json` has comments, `init` won't rewrite it. Paste the printed `servers` snippet yourself, or run `hindsight-copilot init --print-only` to see it.

### Expecting Cloud to work without a token

Hindsight Cloud requires an API token. Pass `--api-token` (or set `HINDSIGHT_API_TOKEN`). A self-hosted open local server needs no token.

### Assuming banks are shared by default

Each bank is isolated. If you want two projects or tools to share memory, point them at the same `--bank-id`; otherwise they won't see each other's context.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — pass `--api-url http://localhost:8888` (no token needed for an open local server).

### Does this change how I use Copilot?

No. It adds the `hindsight` MCP server and a recall/retain rule to your workspace. You keep using Copilot Chat in agent mode as usual — the rule tells Copilot when to recall and retain.

### How is memory scoped?

Per bank, set with `--bank-id` (default `copilot`). The bank is the last path segment of the MCP endpoint URL.

### How do I remove it?

Run `hindsight-copilot uninstall`, which removes the `hindsight` MCP server entry and the recall/retain rule. Use `hindsight-copilot status` to check the current state.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [GitHub Copilot integration docs](https://hindsight.vectorize.io/docs/integrations/github-copilot)
