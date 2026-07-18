---
title: "Guide: Add Grok Build Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, grok-build, coding-agents, memory]
description: "Add Grok Build memory with Hindsight using the hindsight-memory plugin, so Grok Build recalls relevant context on every prompt and retains each conversation for later sessions."
image: /img/guides/guide-grok-build-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add Grok Build Memory with Hindsight](/img/guides/guide-grok-build-memory-with-hindsight.svg)

If you want **Grok Build memory with Hindsight**, the setup is the `hindsight-memory` plugin. Grok Build natively reads the Claude Code plugin format, so the same plugin that powers Claude Code installs and runs in Grok Build without modification. On every user prompt it recalls relevant memories and injects them as context, and after each response it extracts and retains the conversation for long-term storage.

This works because Grok Build reads Claude Code plugins natively — hooks, MCP servers, skills, and marketplace metadata all work as-is. The plugin hooks into Grok Build's lifecycle events: a `UserPromptSubmit` hook drives auto-recall, and a `Stop` hook drives auto-retain. That gives Grok Build long-term memory across sessions instead of forcing every new session to rediscover the same context.

This guide walks through adding the marketplace, installing the plugin, configuring an LLM provider for memory extraction, understanding how memory is scoped, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `claude plugin marketplace add vectorize-io/hindsight`.
> 2. `claude plugin install hindsight-memory`.
> 3. Set an LLM provider key for extraction (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`).
> 4. Run `grok` — the plugin activates automatically.
> 5. Verify that a later session remembers what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- Grok Build installed and working (the `grok` command)
- The `claude` CLI available to add the marketplace and install the plugin
- An LLM provider key for memory extraction, or a reachable external Hindsight server
- Optionally, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server if you don't want the plugin to run Hindsight locally

## Step 1: Add the marketplace and install the plugin

Grok Build reads Claude Code plugins natively, so installation uses the standard Claude Code commands. Grok Build will discover and activate the plugin automatically.

```bash
claude plugin marketplace add vectorize-io/hindsight
claude plugin install hindsight-memory
```

The `hindsight-memory` plugin is the same one that powers Claude Code — all features, configuration options, and knowledge tools are fully available in Grok Build.

## Step 2: Configure a provider for memory extraction

The plugin needs an LLM provider to extract facts from your conversations. Set one of the auto-detected provider keys:

```bash
# Option A: OpenAI (auto-detected)
export OPENAI_API_KEY="sk-your-key"

# Option B: Anthropic (auto-detected)
export ANTHROPIC_API_KEY="your-key"
```

Or connect to an external Hindsight server instead of running Hindsight locally:

```bash
mkdir -p ~/.hindsight
echo '{"hindsightApiUrl": "https://your-hindsight-server.com"}' > ~/.hindsight/claude-code.json
```

## Step 3: Start Grok Build

Start Grok Build normally — the plugin activates automatically:

```bash
grok
```

From here on the plugin captures and recalls memories in the background with no changes to your workflow.

## How the plugin uses memory

The plugin hooks into Grok Build's lifecycle events:

- **Auto-recall (before):** on every user prompt, the `UserPromptSubmit` hook queries Hindsight for relevant memories and injects them as `additionalContext` — invisible to the chat transcript, visible to Grok.
- **Auto-retain (after):** after every response (or every N turns), the `Stop` hook extracts the conversation content and retains it to Hindsight for long-term storage.
- **Knowledge tools:** Grok can read, write, and search its own memory directly through MCP tools such as `agent_knowledge_recall` and `agent_knowledge_ingest`.
- **Daemon management:** the plugin can auto-start and stop a local `hindsight-embed` daemon, or connect to an external Hindsight server.

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Per-project and per-tool memory

By default Grok Build and Claude Code share `~/.hindsight/claude-code.json`, so they share memory. To give each project its own isolated memory bank, set this in `~/.hindsight/claude-code.json`:

```json
{
  "dynamicBankId": true,
  "dynamicBankGranularity": ["agent", "project"]
}
```

With this config, running Grok Build in `~/projects/api` and `~/projects/frontend` stores and recalls memories separately. Git worktrees of the same repo share a bank by default.

If you want Grok Build and Claude Code to keep separate memory instead of sharing, override the agent name in your Grok Build shell:

```bash
export HINDSIGHT_AGENT_NAME=grok-build
export HINDSIGHT_BANK_ID=grok_build
grok
```

For the full configuration reference, see the [Claude Code configuration docs](https://hindsight.vectorize.io/sdks/integrations/claude-code#configuration).

## Verify that memory is working

A good test sequence is:

1. run `grok` in a project
2. tell it a decision or convention worth remembering
3. exit so the transcript is retained
4. run `grok` again in the same project
5. ask about the earlier decision

Memories need at least one retain cycle before they're available, so complete a full session first — say something, exit, then start a new session. If the new session surfaces the earlier decision, the setup is working. You can also enable `"debug": true` in your config to see `[Hindsight]` messages in stderr confirming recall and retain.

## Common mistakes

### Checking recall before the first retain cycle

Memories need at least one retain cycle before they can be recalled. If you check on the very first session, there is nothing stored yet — complete a full session and start a new one.

### Forgetting the extraction provider

The plugin needs an LLM provider to extract facts. If you skip setting `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (and aren't pointing at an external server), memory extraction won't run.

### Assuming Grok Build and Claude Code memory are separate

By default both tools share `~/.hindsight/claude-code.json` and the same memory. If you want isolation, override the agent name and bank ID as shown above.

### Expecting projects to be isolated automatically

Without `dynamicBankId`, all projects share one bank. Enable it with `dynamicBankGranularity` if you want per-project memory.

## FAQ

### Do I need Hindsight Cloud?

No. The plugin can run Hindsight locally via the `hindsight-embed` daemon, or you can point it at a self-hosted server with `hindsightApiUrl`. Hindsight Cloud is one option, not a requirement.

### Does this change how I use Grok Build?

No. The plugin activates automatically and captures and recalls memories in the background — no changes to your workflow are required.

### How is memory scoped?

Shared across tools by default. Enable `dynamicBankId` with `dynamicBankGranularity` for per-project isolation, or set `HINDSIGHT_AGENT_NAME` and `HINDSIGHT_BANK_ID` to separate Grok Build from Claude Code.

### The plugin isn't listed or hooks aren't firing — what do I check?

Run `grok plugin list` to confirm `hindsight-memory` is installed, and `grok inspect` to check the Hooks section. Enable `"debug": true` in your config to see `[Hindsight]` messages in stderr.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Grok Build integration docs](https://hindsight.vectorize.io/docs/integrations/grok-build)
