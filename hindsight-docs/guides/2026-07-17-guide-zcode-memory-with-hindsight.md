---
title: "Guide: Add ZCode Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, zcode, coding-agents, memory]
description: "Add ZCode memory with Hindsight using the hindsight-zcode hooks, so ZCode recalls relevant context before each prompt and retains each turn automatically."
image: /img/guides/guide-zcode-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add ZCode Memory with Hindsight](/img/guides/guide-zcode-memory-with-hindsight.svg)

If you want **ZCode memory with Hindsight**, the cleanest setup is the `hindsight-zcode` hooks. ZCode — Z.ai's GLM desktop coding agent — embeds the Claude Code agent runtime, so Python hook scripts can recall relevant memory before each prompt and retain each turn after the agent responds. That gives ZCode long-term memory across sessions instead of forcing every new conversation to rediscover the same project context.

This works well for ZCode because it ships a native process-hook system and reads the standard Claude Code hook schema from its own config namespace. The hooks use two points ZCode exposes: `UserPromptSubmit` to inject recalled memory before the model sees the turn, and `Stop` to store the turn once the agent finishes. There is no MCP server to run alongside ZCode — the hook scripts call Hindsight's REST API directly.

This guide walks through installing the hooks, pointing them at your Hindsight backend, understanding how bank scoping works, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-zcode`.
> 2. Run `hindsight-zcode install --api-url https://api.hindsight.vectorize.io --api-token your-api-key` (Cloud), or `hindsight-zcode install` for a local daemon.
> 3. Restart ZCode so it loads the new hooks.
> 4. The bank defaults to `zcode`, shared across your Hindsight integrations; enable `dynamicBankId` for per-project isolation.
> 5. Verify that a later prompt recalls what an earlier turn stored.

## Prerequisites

Before you start, make sure you have:

- ZCode installed with config-hooks support (`~/.zcode/cli/config.json`)
- Python 3.9+ for the hook scripts (stdlib only — no runtime dependencies)
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server

## Step 1: Install the CLI

Install the `hindsight-zcode` package. The `pip install` only ships the one-time installer; the hook scripts themselves are pure Python stdlib.

```bash
pip install hindsight-zcode
```

## Step 2: Run the installer

For Hindsight Cloud, pass your API URL and token:

```bash
hindsight-zcode install --api-url https://api.hindsight.vectorize.io --api-token your-api-key
```

For a local `hindsight-embed` daemon, omit the flags:

```bash
hindsight-zcode install
```

The installer copies the hook scripts to `~/.zcode/hooks/hindsight/`, merges Hindsight's hooks into `~/.zcode/cli/config.json` under `hooks.events` (preserving any existing keys and foreign hooks), sets `hooks.enabled` to `true`, and seeds `~/.hindsight/zcode.json` for your personal config. It never touches your Claude Code config at `~/.claude/settings.json`.

Restart ZCode to load the hooks — memory is then live.

To uninstall, which removes the scripts and strips only Hindsight's entries from the config:

```bash
hindsight-zcode uninstall
```

## Step 3: Configure the connection (optional)

The installer seeds `~/.hindsight/zcode.json` for personal overrides that survive updates. A typical Cloud config:

```json
{
  "hindsightApiUrl": "https://api.hindsight.vectorize.io",
  "hindsightApiToken": "your-api-key",
  "bankId": "my-zcode-memory"
}
```

To connect to a local daemon instead, run `hindsight-embed` separately (for example `uvx hindsight-embed`), leave `hindsightApiUrl` empty, and the plugin connects to the daemon on `apiPort` (default `9077`). Every setting also has an environment-variable override — for example `HINDSIGHT_API_URL`, `HINDSIGHT_API_TOKEN`, and `HINDSIGHT_BANK_ID`.

## How the hooks use memory

ZCode reads the standard Claude Code hook schema, so the plugin wires three hook events:

- **Session start (`SessionStart`):** confirms Hindsight is reachable and pre-warms the local daemon if needed.
- **Recall (`UserPromptSubmit`):** queries Hindsight for the most relevant memories and injects them via `hookSpecificOutput.additionalContext`, so the model sees them but they stay out of the transcript. It also stashes the prompt for the next retain.
- **Retain (`Stop`):** pairs the stashed prompt with the agent's reply and POSTs the turn to Hindsight.

ZCode does not provide a `SessionEnd` event, so retention rides `Stop`. With the default `retainEveryNTurns`, turns are stored as they complete, each as its own memory.

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Bank scoping and cross-tool memory

The bank defaults to `zcode`, and all sessions share that bank unless you enable dynamic bank IDs. Because the same Hindsight bank is shared across Claude Code, Cursor, and other Hindsight integrations, memory follows you between tools — a convention learned in one tool is available in the others.

For per-project isolation, set `dynamicBankId` to `true`. That derives a unique bank ID from the configured granularity fields, producing banks like `zcode::my-project` automatically. See the [package README](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/zcode) for the full configuration options.

ZCode also ships its own local, per-project memory. Hindsight is complementary: it stores memory in a cloud or self-hosted bank that is shared across tools and machines, rather than staying local to one ZCode project.

## Verify that memory is working

A good test sequence is:

1. start ZCode after installing the hooks
2. state a decision or convention in a prompt and let the turn complete so it is retained
3. start a new session or ask a follow-up
4. ask about the earlier decision

For example:

- one turn establishes that the project uses FastAPI with asyncpg, not SQLAlchemy
- a later prompt asks what the project's database stack is

If the recalled context surfaces the earlier decision, the setup is working. Enable debug mode (`"debug": true`, or `HINDSIGHT_DEBUG=true`) if you want to see what the hooks are doing.

## Common mistakes

### Not restarting ZCode

ZCode picks up new hooks on session restart. If memory is not recalled or retained right after install, restart ZCode first.

### `hooks.enabled` left off

The installer sets `hooks.enabled` to `true`, but if hooks are not firing, confirm `~/.zcode/cli/config.json` is valid JSON with `"hooks": {"enabled": true, ...}` and the Hindsight entries present under `hooks.events`.

### `python3` not on PATH

The hook scripts run via `python3`. If they do not execute, confirm `python3` is on `$PATH` from your shell.

### Expecting per-project memory by default

By default every ZCode session shares the `zcode` bank. If you want project isolation, enable `dynamicBankId`.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — run `hindsight-embed` locally, install without the flags, and leave `hindsightApiUrl` empty so the plugin connects to the local daemon.

### Does this change how I use ZCode?

No. The hooks run in the background; you use ZCode exactly as before. Memory is recalled before each prompt and retained after each turn.

### Is an MCP server required?

No. The integration is plain Python hook scripts that call Hindsight's REST API. There is nothing to run alongside ZCode.

### Does this touch my Claude Code config?

No. The installer writes to ZCode's config namespace (`~/.zcode/cli/config.json`) and never modifies `~/.claude/settings.json`.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [ZCode integration docs](https://hindsight.vectorize.io/docs/integrations/zcode)
