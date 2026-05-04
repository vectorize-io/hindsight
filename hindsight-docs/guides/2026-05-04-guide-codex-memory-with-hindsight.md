---
title: "Guide: Add Codex CLI Persistent Memory with Hindsight"
authors: [benfrank241]
date: 2026-05-04T15:00:00Z
tags: [how-to, codex, coding-agents, memory]
description: "Add Codex CLI persistent memory with Hindsight using hook based recall, automatic retain, and project scoped bank IDs that survive across sessions."
image: /img/guides/guide-codex-memory-with-hindsight.png
hide_table_of_contents: true
---

![Guide: Add Codex CLI Persistent Memory with Hindsight](/img/guides/guide-codex-memory-with-hindsight.png)

If you want **Codex CLI persistent memory with Hindsight**, the fastest path is to install the Hindsight hook bundle, point Codex at Hindsight Cloud or a local daemon, and let the session hooks handle recall and retain automatically. That gives Codex continuity across coding sessions without making you re explain the same repo habits and project context every time.

This pattern works especially well for project based development. Codex already exposes session hooks, so Hindsight can recall relevant context before each prompt and retain the finished turn after each response.

If you want the underlying reference open while you work, keep [the docs home](https://hindsight.vectorize.io/docs), [the quickstart guide](https://hindsight.vectorize.io/docs/quickstart), [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall), and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain) nearby.

<!-- truncate -->

> **Quick answer**
>
> 1. Install the Codex CLI integration or plugin.
> 2. Point it at Hindsight Cloud or a local Hindsight API.
> 3. Wire memory into your Codex CLI runtime with a stable bank ID.
> 4. Store one preference or project fact, then start a fresh run.
> 5. Confirm that recall brings the earlier context back automatically.

## Why this setup works

Codex CLI exposes exactly the lifecycle points a memory system needs. `SessionStart` can warm the server, `UserPromptSubmit` can inject recalled context, and `Stop` can retain the conversation. That means you get automatic memory without adding manual retain calls to every workflow.

## Prerequisites

- OpenAI Codex CLI version 0.116.0 or later
- Python 3.9 or later for the hook scripts
- A reachable Hindsight backend and one decision about whether memory should be shared or project scoped

## Step 1: Install the integration

```bash
curl -fsSL https://hindsight.vectorize.io/get-codex | bash
```

The installer places the hook scripts under `~/.hindsight/codex/scripts/`, writes `~/.codex/hooks.json`, and enables hook support in `~/.codex/config.toml`.

## Step 2: Connect Codex CLI to Hindsight

```json
{
  "hindsightApiUrl": "https://api.hindsight.vectorize.io",
  "hindsightApiToken": "your-api-key",
  "bankId": "my-codex-memory"
}
```

Save that as `~/.hindsight/codex.json`. For local daemon mode, you can leave `hindsightApiUrl` empty and export a provider key before starting Codex:

```bash
export OPENAI_API_KEY=sk-your-key
codex
```

## Step 3: Wire memory into your runtime

```json
{
  "dynamicBankId": true,
  "dynamicBankGranularity": ["agent", "project"],
  "autoRecall": true,
  "autoRetain": true,
  "retainEveryNTurns": 10,
  "recallBudget": "mid"
}
```

That configuration keeps memory separate per project. If you want one shared bank for everything, set `dynamicBankId` to `false` and keep a single `bankId`.

## Step 4: Choose the right bank strategy

For Codex, project scoped banks are usually the right default. They stop unrelated repositories from leaking into each other while still giving you continuity for the same codebase. A shared bank only makes sense when the same conventions should follow you everywhere, such as global shell preferences or personal coding habits.

## Step 5: Verify that memory is working

1. Use Codex in one repo and store a fact such as the preferred lint command or release process.
2. End the session and start a new Codex session in the same repo.
3. Ask Codex to recall the repo preference you stored earlier.
4. If recall misses, confirm that hooks are enabled and that both runs used the same derived bank ID.

If the second run can answer with details from the first run, your setup is working. If it cannot, turn on debug logging, check the configured bank ID, and confirm that the retain call actually completed.

## Common mistakes

- Installing the hook bundle on an older Codex version that does not support hooks
- Forgetting to turn on `codex_hooks = true` in `~/.codex/config.toml`
- Keeping `dynamicBankId` off when you expected memory isolation per project

## FAQ

### Does this work with Hindsight Cloud?

Yes. Set `hindsightApiUrl` to `https://api.hindsight.vectorize.io` and provide your token in `~/.hindsight/codex.json`.

### Can I use a local server instead?

Yes. Leave `hindsightApiUrl` empty and let the hooks connect to a local Hindsight daemon.

### How do I debug missing memories?

Turn on `debug`, check stderr output, and confirm that the same bank ID is used when Codex retains and recalls.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted memory backend
- Read [the full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow [the quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Compare a related workflow in [Adding Memory to Codex with Hindsight](https://hindsight.vectorize.io/blog/adding-memory-to-codex-with-hindsight)
