---
title: "What's new in hindsight-openclaw 0.6"
authors: [benfrank241]
date: 2026-04-13T12:00
tags: [release, openclaw, memory, setup-wizard, plugin]
image: /img/blog/openclaw-plugin-06.png
hide_table_of_contents: true
description: "hindsight-openclaw 0.6 adds an interactive setup wizard, per-channel OpenClaw memory banks, an external API backend for shared persistent memory, and several reliability improvements."
---

`@vectorize-io/hindsight-openclaw` 0.6 is a significant update to the Hindsight OpenClaw memory plugin. The headline change is an interactive setup wizard that replaces manual configuration, alongside a breaking config overhaul that moves all plugin settings into `openclaw.json`. This release also adds per-channel memory banks, external Hindsight API support, recall injection controls, a JSONL-backed retain queue, and session filtering.

<!-- truncate -->

![Hindsight x OpenClaw](/img/blog/openclaw-plugin-06.png)

- [**Setup Wizard**](#setup-wizard): One command gets you from zero to working memory in under two minutes.
- [**Config Overhaul**](#config-overhaul-breaking-change): All settings now live in `openclaw.json`. Environment variables are gone.
- [**Per-Channel Memory Banks**](#per-channel-memory-banks): Memory is now isolated per agent, channel, and user by default.
- [**External API Backend**](#external-api-backend): Point multiple OpenClaw instances at a shared Hindsight server.
- [**Recall Injection Controls**](#recall-injection-controls): Choose where recalled memories land in context, before or after your system prompt.
- [**Reliability: JSONL Retain Queue**](#reliability-jsonl-retain-queue): Conversations are queued locally when the API is unreachable and replayed when connectivity returns.
- [**Session Filtering**](#session-filtering): Mark specific session patterns as stateless to skip memory operations entirely.
- [**Configurable Tags**](#configurable-tags): Tag retained memories for organization and filtered recall.
- [**Backfill CLI**](#backfill-cli): Ingest historical conversations into Hindsight retroactively.
- [**Conversation Format (0.6.2)**](#conversation-format-062): Conversations are now stored in Anthropic-style JSON, preserving tool use and tool result blocks.

---

## Setup Wizard

Getting started with `hindsight-openclaw` used to require setting the right environment variables in the right places. 0.6.0 replaces that with an interactive wizard:

```bash
npx --package @vectorize-io/hindsight-openclaw hindsight-openclaw-setup
```

The wizard walks through three modes:

- **Cloud (recommended)**: connects to managed Hindsight at `https://api.hindsight.vectorize.io`. Prompts for your [Cloud API token](https://ui.hindsight.vectorize.io/signup). No local setup required.
- **External API**: connects to a self-hosted Hindsight server. Prompts for URL and optional token.
- **Embedded daemon**: spawns a local `hindsight-embed` daemon on the machine. Prompts for LLM provider and API key.

The wizard writes the result to `~/.openclaw/openclaw.json`. From that point, `openclaw gateway` picks it up automatically.

For scripted or CI setups, the wizard also runs non-interactively:

```bash
# Cloud
npx --package @vectorize-io/hindsight-openclaw hindsight-openclaw-setup \
    --mode cloud --token hsk_your_cloud_token

# Embedded with OpenAI
npx --package @vectorize-io/hindsight-openclaw hindsight-openclaw-setup \
    --mode embedded --provider openai --api-key sk-...

# Embedded with Claude Code (authenticates via the Claude Code CLI, no separate API key required)
npx --package @vectorize-io/hindsight-openclaw hindsight-openclaw-setup \
    --mode embedded --provider claude-code
```

The LLM configured here is used only for memory extraction. Your OpenClaw agent uses whatever model you configure separately. For a full walkthrough of installing and configuring the plugin from scratch, see [How to Add Persistent Memory to OpenClaw with Hindsight](/blog/2026/03/06/adding-memory-to-openclaw-with-hindsight).

---

## Config Overhaul (Breaking Change)

Prior to 0.6.0, plugin configuration was read from environment variables. That approach made scripting awkward and buried settings in shell profiles rather than alongside the rest of OpenClaw config.

In 0.6.0, all configuration lives in `~/.openclaw/openclaw.json` under the plugin entry:

```json
{
  "plugins": {
    "entries": {
      "hindsight-openclaw": {
        "enabled": true,
        "config": {
          "hindsightApiUrl": "http://localhost:9077",
          "provider": "openai",
          "model": "gpt-4o-mini",
          "apiKey": "sk-..."
        }
      }
    }
  }
}
```

**Migration**: If you were setting `HINDSIGHT_EMBED_API_URL`, `HINDSIGHT_PROVIDER`, or similar environment variables, move those values into the `config` block above. The setup wizard handles this automatically for new installs. See the [full configuration reference](/sdks/integrations/openclaw) for all available options.

---

## Per-Channel Memory Banks

By default, the plugin now creates isolated memory banks based on agent, channel, and user context. A Slack DM and a Telegram group chat involving the same user get separate memory stores.

This is controlled by `dynamicBankGranularity`:

```json
{
  "config": {
    "dynamicBankGranularity": ["agent", "channel", "user"]
  }
}
```

Available isolation dimensions:

| Field | Description |
|-------|-------------|
| `agent` | The bot identity |
| `channel` | The conversation or group ID |
| `user` | The person interacting with the bot |
| `provider` | The messaging platform (Slack, Telegram, etc.) |

The default is `["agent", "channel", "user"]`, giving full isolation per user per channel per agent. To share memory across all channels for a user, use `["user"]`. To use a single shared bank for everything, set `dynamicBankId: false`.

For static bank configurations, `bankId` is now supported:

```json
{
  "config": {
    "dynamicBankId": false,
    "bankId": "my-shared-bank"
  }
}
```

Use `bankIdPrefix` to namespace banks across environments (e.g. `"prod"` vs `"staging"`). See the [per-user memory cookbook](/cookbook/recipes/per-user-memory) for a worked example of isolating memory per user across platforms.

---

## External API Backend

You can now point the plugin at a self-hosted Hindsight API server instead of running a local embedded daemon. This is the right setup when multiple OpenClaw instances should share the same memory store, or when you want to separate the memory service from the gateway machine. See the [self-hosting quickstart](/developer/api/quickstart) for infrastructure requirements.

```json
{
  "config": {
    "hindsightApiUrl": "https://your-hindsight-server.example.com",
    "hindsightApiToken": "YOUR_API_TOKEN"
  }
}
```

The plugin performs a health check against the remote API on startup. If the check fails, the gateway will log a warning but still start. Retain operations that occur while the API is unreachable are queued locally (see [JSONL Retain Queue](#reliability-jsonl-retain-queue) below).

[Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) works as an external API endpoint. Use your Cloud URL and token, or run the wizard with `--mode cloud`.

---

## Recall Injection Controls

0.5.0 added `recallInjectionPosition`, which controls where recalled memories are inserted into context:

| Value | Behavior |
|-------|----------|
| `"prepend"` | Injected before the system prompt (default) |
| `"append"` | Injected after the system prompt |
| `"user"` | Injected as a user message |

`"append"` is useful when you have a large static system prompt that should benefit from prompt caching: injecting memories after it means the static portion stays stable between turns.

```json
{
  "config": {
    "recallInjectionPosition": "append"
  }
}
```

The full set of recall controls:

| Option | Default | Description |
|--------|---------|-------------|
| `autoRecall` | `true` | Auto-inject memories before each turn |
| `recallBudget` | `"mid"` | Recall effort: `low`, `mid`, or `high` |
| `recallMaxTokens` | `1024` | Max tokens for injected memories |
| `recallTypes` | `["world", "experience"]` | Memory types to include |
| `recallTopK` | unlimited | Hard cap on memories injected per turn |
| `recallContextTurns` | `1` | Prior user turns used to compose the recall query |
| `recallInjectionPosition` | `"prepend"` | Where to inject recalled memories |

And retention controls:

| Option | Default | Description |
|--------|---------|-------------|
| `autoRetain` | `true` | Auto-retain conversations after each turn |
| `retainEveryNTurns` | `1` | Retain every Nth turn |
| `retainOverlapTurns` | `0` | Extra prior turns included when chunked retention fires |

---

## Reliability: JSONL Retain Queue

When the plugin is configured in external API mode and the API is temporarily unreachable, retain operations are now queued in a local JSONL file rather than dropped. When connectivity is restored, the queue is replayed and conversations are retained as normal.

No configuration is required. The queue file lives alongside the plugin's working directory and is cleaned up after successful replay.

---

## Session Filtering

Some sessions should not be retained or recalled: operational messages from bots, internal system events, or anything inherently stateless. 0.6.0 adds session pattern filtering to handle these cases:

```json
{
  "config": {
    "skipSessionPatterns": ["^bot-", "^system-event-"]
  }
}
```

Sessions whose IDs match any pattern in `skipSessionPatterns` are treated as stateless: no retain, no recall, no memory operations at all.

---

## Configurable Tags

Memories retained by the plugin can now be tagged for organization and filtered recall:

```json
{
  "config": {
    "retainTags": ["openclaw", "env:prod"]
  }
}
```

Tags appear on all memories retained by this plugin instance. Use them to scope recall queries, separate environments, or distinguish memories from different deployments sharing the same bank.

---

## Backfill CLI

If you're enabling Hindsight memory on an existing OpenClaw deployment, the backfill CLI lets you ingest historical conversations retroactively:

```bash
npx --package @vectorize-io/hindsight-openclaw hindsight-openclaw-backfill
```

The CLI reads your `openclaw.json` plugin config and processes historical conversation data into your configured Hindsight backend. Run it once after enabling the plugin to prime the memory store before users interact with the agent. The plugin is published on [npm](https://www.npmjs.com/package/@vectorize-io/hindsight-openclaw).

---

## Conversation Format (0.6.2)

Starting in 0.6.2, conversations are stored in Anthropic-style JSON format, preserving `tool_use` and `tool_result` content blocks. Previously, tool interactions were either dropped or flattened to plain text during retention.

This change improves the fidelity of stored conversations for analysis and replay, and ensures that complex agentic interactions are captured in full rather than losing the structure of tool calls.

0.6.2 also includes session stability improvements: session identity is now more consistent across turns, and non-user operational turns are skipped during retention to reduce noise in the memory store.

---

## Upgrading

Install the latest version:

```bash
openclaw plugins install @vectorize-io/hindsight-openclaw
```

If you were using environment variables for configuration, run the setup wizard to migrate:

```bash
npx --package @vectorize-io/hindsight-openclaw hindsight-openclaw-setup
```

The wizard will detect your existing setup and write the equivalent configuration to `openclaw.json`.

---

## Get Started

- [Sign up for Hindsight Cloud](https://ui.hindsight.vectorize.io/signup), the fastest path to working memory, no local infrastructure required.
- [OpenClaw integration docs](/sdks/integrations/openclaw), full configuration reference.
- [OpenClaw plugin changelog](/changelog/integrations/openclaw), complete list of changes since 0.5.0.
