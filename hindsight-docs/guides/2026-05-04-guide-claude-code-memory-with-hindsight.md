---
title: "Guide: Add Claude Code Persistent Memory with Hindsight"
authors: [benfrank241]
date: 2026-05-04T15:00:00Z
tags: [how-to, claude-code, coding-agents, memory]
description: "Add Claude Code persistent memory with Hindsight using the memory plugin, automatic recall hooks, and project aware bank IDs across sessions."
image: /img/guides/guide-claude-code-memory-with-hindsight.png
hide_table_of_contents: true
---

![Guide: Add Claude Code Persistent Memory with Hindsight](/img/guides/guide-claude-code-memory-with-hindsight.png)

If you want **Claude Code persistent memory with Hindsight**, the cleanest setup is to install the Hindsight memory plugin, connect it to Hindsight Cloud or a local daemon, and let the built in hooks handle recall and retain automatically. That gives Claude Code continuity across sessions instead of forcing every new run to rediscover your project conventions, preferences, or earlier decisions.

The plugin is a strong fit for coding work because it wires memory directly into Claude Code's session lifecycle. Recall runs before prompts, retain runs after responses, and dynamic bank IDs let you decide whether memory should be shared across projects or isolated per repo.

If you want the underlying reference open while you work, keep [the Claude Code integration docs](https://hindsight.vectorize.io/docs/integrations/claude-code), [the docs home](https://hindsight.vectorize.io/docs), [the quickstart guide](https://hindsight.vectorize.io/docs/quickstart), [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall), and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain) nearby.

<!-- truncate -->

> **Quick answer**
>
> 1. Install the Claude Code integration or plugin.
> 2. Point it at Hindsight Cloud or a local Hindsight API.
> 3. Wire memory into your Claude Code runtime with a stable bank ID.
> 4. Store one preference or project fact, then start a fresh run.
> 5. Confirm that recall brings the earlier context back automatically.

## Why this setup works

The Claude Code plugin uses hook events that already exist in the runtime. `UserPromptSubmit` injects recalled memories as additional context, `Stop` retains the new conversation content, and `SessionStart` can warm up the Hindsight side early so recall is ready when you need it. You do not need to teach the model to remember manually, because the integration handles the memory loop for you.

## Prerequisites

- Claude Code installed and the plugin marketplace available
- A reachable Hindsight backend, either [Hindsight Cloud](https://hindsight.vectorize.io) or a local daemon
- One decision about memory scope, shared across all work or isolated per project

## Step 1: Install the integration

```bash
claude plugin marketplace add vectorize-io/hindsight
claude plugin install hindsight-memory
```

## Step 2: Connect Claude Code to Hindsight

```json
{
  "hindsightApiUrl": "https://api.hindsight.vectorize.io",
  "hindsightApiToken": "your-api-key"
}
```

Save that as `~/.hindsight/claude-code.json`. If you prefer a local daemon, leave `hindsightApiUrl` empty and export an LLM provider key before launching Claude Code. For example:

```bash
export OPENAI_API_KEY="sk-your-key"
claude
```

## Step 3: Wire memory into your runtime

```json
{
  "dynamicBankId": true,
  "dynamicBankGranularity": ["agent", "project"],
  "agentName": "claude-code",
  "autoRecall": true,
  "autoRetain": true,
  "recallBudget": "mid"
}
```

With that config, the plugin derives a bank per project while still using the same Claude Code agent identity. If you want one shared bank for everything, set `dynamicBankId` to `false` and provide a fixed `bankId` instead.

## Step 4: Choose the right bank strategy

Use a static bank when you want one long running memory for all Claude Code work, for example a personal coding assistant that should remember preferences everywhere. Use dynamic bank IDs when each repo should stay isolated. The default project based pattern is usually the safer choice because it prevents unrelated codebases from polluting each other.

## Step 5: Verify that memory is working

1. Start a Claude Code session and tell it one fact that should be remembered, such as your preferred test command or branch naming rule.
2. End the session, then start a fresh session in the same project.
3. Ask Claude Code what it remembers about the project setup or your preferred workflow.
4. If needed, enable debug logging and confirm the same bank ID was used for both runs.

If the second run can answer with details from the first run, your setup is working. If it cannot, turn on debug logging, check the configured bank ID, and confirm that the retain call actually completed.

## Common mistakes

- Installing the plugin but never creating `~/.hindsight/claude-code.json`, so the plugin has nowhere to connect
- Using one static bank across unrelated repos, which makes memory feel noisy instead of useful
- Running local daemon mode without an LLM provider configured, which prevents retain from completing

## FAQ

### Do users see the recalled memory block in the transcript?

No. The plugin injects recalled memories as additional context for Claude Code, not as visible chat text.

### Should I use Hindsight Cloud or a local daemon?

Cloud is the fastest way to get started. A local daemon is useful when you want a local first setup or tighter control over the backend.

### Can I keep memory separate per repo?

Yes. Turn on `dynamicBankId` and include `project` in `dynamicBankGranularity` so each repo gets its own bank.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted memory backend
- Read [the full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow [the quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Compare a related workflow in [Adding Memory to Codex with Hindsight](https://hindsight.vectorize.io/blog/adding-memory-to-codex-with-hindsight)
