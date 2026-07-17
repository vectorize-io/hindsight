---
title: "Guide: Add OMO (oh-my-openagent) Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, omo, agents, memory]
description: "Add OMO (oh-my-openagent) memory with Hindsight using five lifecycle hooks that recall context before each prompt and retain conversations after each turn."
image: /img/guides/guide-omo-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add OMO (oh-my-openagent) Memory with Hindsight](/img/guides/guide-omo-memory-with-hindsight.svg)

If you want **OMO (oh-my-openagent) memory with Hindsight**, the cleanest setup is the five-hook plugin. It installs into OMO's hook system, recalls relevant memories before each prompt, and retains the conversation transcript after each turn. That gives OMO long-term memory across sessions instead of starting every new session from scratch.

This is a good fit for OMO because OMO exposes lifecycle hook events — `SessionStart`, `UserPromptSubmit`, `Stop`, `SubagentStop`, and `SessionEnd`. The plugin uses all five: it queries Hindsight on every user prompt and injects results as `additionalContext`, then reads the session transcript on stop and POSTs it to Hindsight. Because OMO delegates to sub-agents, the `SubagentStop` hook also captures what those delegated sub-agents learned.

This guide walks through installing the hooks, pointing OMO at your Hindsight backend, understanding per-project bank isolation, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. From `hindsight-integrations/omo/`, copy `hooks/hooks.json`, the `scripts/` directory, and `settings.json` into `~/.omo/`, and copy `rules/hindsight-memory.md` into your project's `.omo/rules/`.
> 2. Set `HINDSIGHT_API_TOKEN` (Cloud) or `HINDSIGHT_API_URL` (self-hosted).
> 3. Add `HINDSIGHT_API_URL`, `HINDSIGHT_API_TOKEN`, and `HINDSIGHT_BANK_ID` to `mcp_env_allowlist` in `~/.config/opencode/oh-my-openagent.jsonc`.
> 4. The bank defaults to `omo`; enable `dynamicBankId` for per-project isolation.
> 5. Verify that a later session remembers what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- OMO ([oh-my-openagent](https://github.com/code-yeongyu/oh-my-openagent)) installed and working
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server
- Python 3 available (the hooks are pure Python stdlib, no `pip install` required)

## Step 1: Install the hooks

From the `hindsight-integrations/omo/` directory, install the hooks, scripts, and settings globally:

```bash
# Hooks (global)
mkdir -p ~/.omo/hooks
cp hooks/hooks.json ~/.omo/hooks/hindsight-hooks.json

# Scripts + settings (global)
mkdir -p ~/.omo/plugins/hindsight/scripts
cp -r scripts/ ~/.omo/plugins/hindsight/scripts/
cp settings.json ~/.omo/plugins/hindsight/settings.json
```

Then install the memory rules per project, from your project root:

```bash
# Rules (per-project — run from your project root)
mkdir -p .omo/rules
cp rules/hindsight-memory.md .omo/rules/hindsight-memory.md
```

The plugin has zero dependencies — it is pure Python stdlib, so there is no package to install.

## Step 2: Point the plugin at Hindsight

For Hindsight Cloud, set your API key:

```bash
export HINDSIGHT_API_TOKEN=hsk_your_key_here
```

For a self-hosted Hindsight server, override the API URL:

```bash
export HINDSIGHT_API_URL=http://localhost:8888
```

No API token is required for local instances. You can also set these persistently in `~/.hindsight/omo.json`:

```json
{
  "hindsightApiUrl": "http://localhost:8888",
  "hindsightApiToken": null
}
```

## Step 3: Allow the env vars in OMO's config

OMO gates which environment variables reach its hooks. Add the Hindsight vars to the allowlist in `~/.config/opencode/oh-my-openagent.jsonc`:

```jsonc
{
  "mcp_env_allowlist": [
    "HINDSIGHT_API_URL",
    "HINDSIGHT_API_TOKEN",
    "HINDSIGHT_BANK_ID"
  ]
}
```

Start a new OMO session — memory is live.

## How the plugin uses memory

The plugin wires into five OMO hook events:

| Hook | Event | Purpose |
|------|-------|---------|
| `session_start.py` | `SessionStart` | Warm up — verify Hindsight is reachable |
| `recall.py` | `UserPromptSubmit` | Auto-recall — query memories, inject as `additionalContext` |
| `retain.py` | `Stop` | Auto-retain — extract transcript, POST to Hindsight (async) |
| `retain.py` | `SubagentStop` | Sub-agent retain — capture delegated sub-agent learnings |
| `session_end.py` | `SessionEnd` | Force final retain for short sessions |

On `UserPromptSubmit`, the hook reads the prompt, queries Hindsight for the most relevant memories, and outputs a `hookSpecificOutput.additionalContext` block that OMO prepends to the conversation before sending it to the model. The injected block is invisible to the transcript but visible to OMO.

On `Stop` and `SubagentStop`, the hook reads the session transcript, strips previously injected memory tags (to prevent feedback loops), and POSTs the conversation to Hindsight asynchronously. It uses the session ID as the document ID, so re-running the same session updates the stored content rather than duplicating it.

All hooks degrade gracefully: if Hindsight is unreachable, OMO keeps working normally without memory.

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Per-project memory banks

By default all sessions share the `omo` bank. To give each project its own isolated memory, enable dynamic bank IDs in `~/.hindsight/omo.json`:

```json
{
  "dynamicBankId": true,
  "dynamicBankGranularity": ["agent", "project"]
}
```

With this config, running OMO in `~/projects/api` and `~/projects/frontend` stores and recalls memories separately. Bank IDs are derived from the working directory path (for example `omo::api`, `omo::frontend`).

You can also query additional banks alongside the primary one with `recallAdditionalBanks`, for example a shared team knowledge bank.

## Verify that memory is working

A good test sequence is:

1. start an OMO session
2. tell OMO a decision or convention worth remembering
3. let the session end so the transcript is retained
4. start a new OMO session in the same context
5. ask about the earlier decision

If the recalled context surfaces the earlier decision, the setup is working.

Because `retainEveryNTurns` defaults to `10`, retain only fires every 10 turns. While testing, add `"retainEveryNTurns": 1` to `~/.hindsight/omo.json` so retain fires every turn. Add `"debug": true` to see what Hindsight is doing on each turn — all log lines are prefixed with `[Hindsight]`.

## Common mistakes

### Forgetting the env allowlist

If `HINDSIGHT_API_URL`, `HINDSIGHT_API_TOKEN`, and `HINDSIGHT_BANK_ID` are not in `mcp_env_allowlist`, OMO never passes them to the hooks and memory silently does nothing.

### Cloud mode with no token

If `hindsightApiUrl` points to Hindsight Cloud but no `hindsightApiToken` is set, the hooks silently skip. Set `HINDSIGHT_API_TOKEN` or add it to `~/.hindsight/omo.json`.

### Testing before retain fires

`retainEveryNTurns` defaults to `10`, so a short session may not store anything. Set `"retainEveryNTurns": 1` while testing.

### Expecting memories on the first session

Recall returns results only after something has been retained. Complete one OMO session first, then start a new one to see memories.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — set `HINDSIGHT_API_URL` and leave the token unset for local instances.

### Does this change how I use OMO?

No. The hooks run automatically around your normal OMO workflow. If Hindsight is unreachable, OMO continues working without memory.

### How is memory scoped?

By default all sessions share the `omo` bank. Enable `dynamicBankId` with `dynamicBankGranularity: ["agent", "project"]` for per-project isolation, producing banks like `omo::myproject`.

### Does it capture sub-agent work?

Yes. OMO delegates to sub-agents (Claude Code, Codex, OpenCode), and the `SubagentStop` hook captures learnings from those delegated sub-agents.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [OMO integration docs](https://hindsight.vectorize.io/docs/integrations/omo)
