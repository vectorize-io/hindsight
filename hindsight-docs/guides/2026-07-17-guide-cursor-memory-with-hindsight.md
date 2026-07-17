---
title: "Guide: Add Cursor Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, cursor, coding-agents, memory]
description: "Add Cursor memory with Hindsight using the hindsight-cursor plugin, so each session recalls relevant project context at start and retains the conversation after each task."
image: /img/guides/guide-cursor-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add Cursor Memory with Hindsight](/img/guides/guide-cursor-memory-with-hindsight.svg)

If you want **Cursor memory with Hindsight**, the cleanest setup is the `hindsight-cursor` plugin. A single `hindsight-cursor init` command installs plugin hooks that recall relevant project memory at session start and retain the conversation after each task, plus an MCP integration that gives the agent explicit `recall`, `retain`, and `reflect` tools mid-session. That gives Cursor long-term memory across coding sessions instead of forcing every new session to rediscover the same project context.

This is a good fit for Cursor because Cursor exposes both a hook system and native MCP support. The plugin uses both: the `sessionStart` hook injects recalled memory as context automatically, the `stop` hook retains the transcript when a task completes, and the MCP tools handle targeted, on-demand lookups. Ambient memory needs no user intervention, while the tools stay available when the agent wants to reach for memory explicitly.

This guide walks through installing the plugin, pointing it at your Hindsight backend, understanding how session memory reaches the agent, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-cursor` from inside your project directory.
> 2. Run `hindsight-cursor init --api-url ... --api-token ...` (Cloud) or `--api-url http://localhost:8888` (self-hosted).
> 3. Fully quit and reopen Cursor — plugins load at startup.
> 4. Session recall injects project memory automatically; auto-retain saves the conversation after each task.
> 5. Verify that a later session remembers what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- Cursor installed and working
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server
- A project directory, since `init` installs plugin files into the project

## Step 1: Install the plugin

From inside the project you want memory for, install the plugin.

```bash
cd /path/to/your-project
pip install hindsight-cursor
```

If you would rather not install the package permanently, you can run it with `uvx hindsight-cursor init` instead of `pip install` plus `hindsight-cursor init`.

## Step 2: Point the plugin at Hindsight

For Hindsight Cloud, pass your API URL and token to `init`:

```bash
hindsight-cursor init --api-url https://api.hindsight.vectorize.io --api-token YOUR_HINDSIGHT_API_TOKEN
```

For a self-hosted Hindsight server:

```bash
hindsight-cursor init --api-url http://localhost:8888
```

If you do not have a Cloud token yet, sign up at [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) and create an API key, or start Hindsight locally with Docker:

```bash
export OPENAI_API_KEY=your-key
docker run --rm -it --pull always -p 8888:8888 \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  -e HINDSIGHT_API_LLM_MODEL=gpt-4o-mini \
  -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

`init` sets up both mechanisms at once. Use `--no-mcp` to skip the MCP configuration if you only want the hooks, and `--force` to overwrite an existing installation.

## Step 3: Fully quit and reopen Cursor

Plugins are loaded at startup, so **fully quit Cursor and reopen it** after installing. A simple window reload is not enough. If you add the plugin to an already-open workspace and skip this step, the plugin will not activate.

## What `init` does

The `init` command:

- Copies plugin files into `.cursor-plugin/hindsight-memory/`
- Creates `~/.hindsight/cursor.json` with your connection settings, if that file does not already exist
- Writes `.cursor/mcp.json` with the Hindsight MCP endpoint for on-demand `recall`, `retain`, and `reflect` tools

The plugin also ships an on-demand `hindsight-recall` skill for manual lookups and an always-on rule (`hindsight-memory.mdc`) that instructs the agent to use recalled memories and the MCP tools.

## How the plugin uses memory

The plugin works through two complementary mechanisms, both configured by `init`:

- **Plugin hooks (automatic):** the `sessionStart` hook fires when a new session begins. It performs a broad project-level recall and surfaces those memories to the agent as context. The `stop` hook fires when a task completes, extracts the conversation transcript, and retains it to Hindsight for future recall.
- **MCP tools (on-demand):** `.cursor/mcp.json` connects Cursor's native MCP support to Hindsight's MCP endpoint, giving the agent explicit `recall`, `retain`, and `reflect` tools for targeted mid-session use.

Session recall is silent — it injects memory into the agent's context rather than showing a visible tool call. If you see an explicit "Ran Recall in hindsight" message in the agent window, that is the MCP path, not the plugin hook. Both can work together.

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Memory banks

By default the bank is `cursor`, set via the `bankId` setting in `~/.hindsight/cursor.json`. A bank is an isolated memory store — like a separate "brain" — so memories from one bank never leak into another.

If you want per-agent, per-project, or per-session isolation, set `dynamicBankId` to `true`. The bank ID is then derived from the fields listed in `dynamicBankGranularity`, which defaults to `["agent", "project"]`. Every setting in `~/.hindsight/cursor.json` can also be overridden with an environment variable, with later entries in the loading order winning. See the [Cursor integration docs](https://hindsight.vectorize.io/docs/integrations/cursor) for the full configuration reference.

## Verify that memory is working

The plugin writes a status file on every hook invocation — even when no memories are found or retain is skipped. Check them to confirm hooks are firing:

```bash
cat ~/.hindsight/cursor-state/state/last_recall.json
cat ~/.hindsight/cursor-state/state/last_retain.json
```

Each file records a `saved_at` timestamp, a `status` of `success`, `empty`, `skipped`, or `error`, the `bank_id` used, and a `result_count` (recall) or `message_count` (retain). If `saved_at` updates when you use Cursor, the hooks are firing; check `status` to understand what happened.

A good end-to-end test sequence is:

1. use Cursor in a project and let the agent record a decision or convention
2. finish the task so the transcript is retained
3. start a new session in the same project
4. ask about the earlier decision

If the new session surfaces the earlier decision, the setup is working.

## Common mistakes

### Not fully restarting Cursor

Plugins load at startup. A window reload is not enough — fully quit and reopen Cursor after installing.

### Confusing MCP recall with plugin recall

Plugin-based recall is silent. A visible "Ran Recall in hindsight" message means MCP is doing the work, not the hook. Both can run together.

### Testing retain too early

Auto-retain runs when a task completes. If you check before the task stops, the conversation may not have been stored yet.

### Expecting memories with an empty bank

Recall can only surface what has been retained. A brand-new bank needs at least one retain cycle before recall returns anything.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — pass its URL with `--api-url http://localhost:8888`, or run Hindsight locally with Docker.

### Do I have to use MCP?

No. MCP tools are optional. Pass `--no-mcp` to `init` if you only want the automatic plugin hooks.

### How is memory scoped?

By default all sessions share the `cursor` bank. Set `dynamicBankId` to `true` for per-agent, per-project, or per-session isolation.

### Is this similar to other coding-agent integrations?

Yes in spirit. Cursor uses plugin hooks plus MCP instead of a wrapper command, but the recall-before / retain-after pattern is the same as the other Hindsight editor integrations.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Cursor integration docs](https://hindsight.vectorize.io/docs/integrations/cursor)
