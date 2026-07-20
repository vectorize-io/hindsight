---
title: "Guide: Add Cursor CLI Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, cursor-cli, coding-agents, memory]
description: "Add Cursor CLI memory with Hindsight using the hindsight-cursor-cli hook scripts, so each prompt recalls relevant context and each turn retains the conversation automatically."
image: /img/guides/guide-cursor-cli-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add Cursor CLI Memory with Hindsight](/img/guides/guide-cursor-cli-memory-with-hindsight.svg)

If you want **Cursor CLI memory with Hindsight**, the cleanest setup is the `hindsight-cursor-cli` package. It installs Python hook scripts that recall relevant memories before each prompt and retain the conversation after each turn. That gives Cursor CLI long-term memory across sessions instead of forcing every new conversation to rediscover your stack, preferences, and past decisions.

This is a good fit for Cursor CLI because Cursor CLI exposes hook events instead of an MCP client. The package wires into four of them — `sessionStart`, `beforeSubmitPrompt`, `stop`, and `sessionEnd` — so recall and retain happen automatically with no changes to your Cursor workflow. The hook scripts are pure Python stdlib, so the `pip install` only ships a one-time installer.

This guide walks through installing the hooks, pointing them at your Hindsight backend, understanding the per-project bank strategy, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-cursor-cli`.
> 2. Run `hindsight-cursor-cli install --api-url https://api.hindsight.vectorize.io --api-token your-api-key` (or omit the flags for a local daemon).
> 3. Restart Cursor CLI — memory is live.
> 4. The bank defaults to `cursor-cli`; enable `dynamicBankId` for per-project isolation.
> 5. Verify that a later session remembers what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- Cursor CLI v0.45+ with hooks support installed and working
- Python 3.9+ available (the hook scripts are stdlib only)
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted `hindsight-embed` daemon

## Step 1: Install the package

Install the one-time installer with pip.

```bash
pip install hindsight-cursor-cli
```

The `pip install` only ships the installer — the hook scripts themselves are pure Python stdlib with zero runtime dependencies.

## Step 2: Install the hooks

Run the installer once to wire the hooks into Cursor CLI.

For Hindsight Cloud:

```bash
hindsight-cursor-cli install --api-url https://api.hindsight.vectorize.io --api-token your-api-key
```

For a self-hosted local daemon, omit the flags — the installer defaults to a local `hindsight-embed` connection:

```bash
hindsight-cursor-cli install
```

The installer copies the hook scripts to `~/.cursor/hooks/cursor-cli/`, writes `~/.cursor/hooks.json` (merged with any existing entries), and creates `~/.hindsight/cursor-cli.json` for your personal config. Restart Cursor CLI to load the hooks.

To uninstall, which removes the hook scripts and strips Hindsight's entries from `~/.cursor/hooks.json` while preserving your personal config:

```bash
hindsight-cursor-cli uninstall
```

## Step 3: Point the hooks at Hindsight

If you did not pass `--api-url` and `--api-token` during install, set your connection in `~/.hindsight/cursor-cli.json`. This is the user config that survives updates:

```json
{
  "hindsightApiUrl": "https://api.hindsight.vectorize.io",
  "hindsightApiToken": "hsk_your_token"
}
```

To connect to a local daemon instead, leave `hindsightApiUrl` empty and start `hindsight-embed` separately — the `session_start.py` hook detects it on `apiPort` (default `9077`):

```bash
uvx hindsight-embed
```

Config loads in order, with later entries winning: built-in defaults, the plugin `settings.json` at `~/.cursor/hooks/cursor-cli/settings.json`, your user config at `~/.hindsight/cursor-cli.json`, then environment variables. Most settings can also be overridden via environment variable — for example `HINDSIGHT_API_URL`, `HINDSIGHT_API_TOKEN`, and `HINDSIGHT_BANK_ID`.

## How the hooks use memory

Cursor CLI has no MCP client, so the package works through four hook events:

- **Recall (before):** on `beforeSubmitPrompt`, `recall.py` reads the prompt, queries Hindsight for the most relevant memories, and injects them as additional context. Cursor prepends this block to the conversation before sending it to the model — it is visible to the model, not the transcript.
- **Retain (after):** on `stop`, `retain.py` reads the session transcript, strips previously injected memory tags to prevent feedback loops, and POSTs the conversation to Hindsight asynchronously. On `sessionEnd`, `session_end.py` forces a final retain so the last turns aren't lost.

Retain uses the session ID as the document ID, so re-running the same session updates the stored content rather than duplicating it. By default `retainMode` is `"full-session"`, sending the full transcript per session; `"chunked"` sends sliding windows every N turns instead.

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Per-project memory banks

By default all sessions share one bank named `cursor-cli`. To give each project its own isolated memory, enable dynamic bank IDs in your config:

```json
{
  "dynamicBankId": true,
  "dynamicBankGranularity": ["agent", "project"]
}
```

With this config, running Cursor in `~/projects/api` and `~/projects/frontend` stores and recalls memories separately. Bank IDs are derived from the working directory. To share memory across all worktrees of the same repo, use `gitProject` instead of `project` in `dynamicBankGranularity`.

For the full set of configuration options, see the [package README](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/cursor-cli).

## Verify that memory is working

A good test sequence is:

1. start a Cursor CLI session
2. tell it a decision or convention worth remembering
3. end the session so the transcript is retained
4. start a new Cursor CLI session
5. ask about the earlier decision

Because `retainEveryNTurns` defaults to `10`, the `stop` hook only fires a retain every 10 turns. While testing, add `"retainEveryNTurns": 1` to `~/.hindsight/cursor-cli.json` so retain fires every turn. The `sessionEnd` hook also forces a final retain when you close the session.

If the recalled context surfaces the earlier decision in the new session, the setup is working.

## Common mistakes

### Testing retain too early

Recall returns results only after something has been retained. Complete one Cursor session first, then start a new one.

### Leaving retainEveryNTurns at the default while testing

`retainEveryNTurns` defaults to `10`, so short test sessions may not fire a mid-session retain. Set it to `1` while verifying, and rely on the `sessionEnd` flush for short sessions.

### Hooks not firing after install

Confirm `~/.cursor/hooks.json` exists and that `python3` is on your shell's `$PATH`. Cursor CLI requires a session restart to pick up new hooks — re-run `hindsight-cursor-cli install` to rewrite the entries if needed.

### Assuming memory is per-project by default

By default every session shares the single `cursor-cli` bank. Enable `dynamicBankId` if you want each project to have its own isolated memory.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted local `hindsight-embed` daemon works too — omit the install flags and run `uvx hindsight-embed` separately.

### Does this change how I use Cursor CLI?

No. The hooks run automatically on Cursor's events, so there are no workflow changes required.

### How is memory scoped?

All sessions share one bank (`cursor-cli`) by default. Enable `dynamicBankId` to derive a separate bank per project from the working directory.

### Why aren't my memories being stored during a short session?

The `stop` hook only retains every `retainEveryNTurns` turns (default `10`). Set `"retainEveryNTurns": 1` while testing, or rely on the `sessionEnd` hook, which forces a final retain when you close the session.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Cursor CLI integration docs](https://hindsight.vectorize.io/docs/integrations/cursor-cli)
