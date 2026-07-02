---
sidebar_position: 20
title: "OpenCode Persistent Memory with Hindsight | Integration"
description: "Add long-term memory to OpenCode with Hindsight. Automatically captures conversations and recalls relevant context across coding sessions."
---

# OpenCode

Persistent long-term memory plugin for [OpenCode](https://opencode.ai) using [Hindsight](https://vectorize.io/hindsight). Automatically captures conversations, recalls relevant context on every turn, and provides retain/recall/reflect tools the agent can call directly.

## Quick Start

Add to your `opencode.json` (project) or `~/.config/opencode/opencode.json` (global):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["@vectorize-io/opencode-hindsight"]
}
```

OpenCode auto-installs plugins in the `"plugin"` array on startup — no `npm install` required.

Point the plugin at your Hindsight server and start OpenCode:

```bash
export HINDSIGHT_API_URL="http://localhost:8888"
opencode
```

### Using Hindsight Cloud

Get an API key at [ui.hindsight.vectorize.io/connect](https://ui.hindsight.vectorize.io/connect):

```bash
export HINDSIGHT_API_URL="https://api.hindsight.vectorize.io"
export HINDSIGHT_API_TOKEN="your-api-key"
opencode
```

Or configure inline via plugin options in `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": [
    ["@vectorize-io/opencode-hindsight", {
      "hindsightApiUrl": "https://api.hindsight.vectorize.io",
      "hindsightApiToken": "your-api-key"
    }]
  ]
}
```

## Features

### Custom Tools

The plugin registers three tools the agent can call explicitly:

| Tool | Description |
|---|---|
| `hindsight_retain` | Store information in long-term memory |
| `hindsight_recall` | Search long-term memory for relevant information |
| `hindsight_reflect` | Generate a synthesized answer from long-term memory |

### Auto-Retain

After each agent response (when the `session.idle` event fires), the plugin automatically retains the full conversation transcript to Hindsight as an upsert. This ensures even one-shot prompts are captured reliably. A pre-compaction retain serves as a backup before context is compressed.

### Per-Turn Recall

On every turn, the plugin recalls relevant memories keyed on the latest user message and injects them into the system prompt. This ensures injected memories are always contextually relevant to the current question, not stale from a previous turn.

### Compaction Hook

When OpenCode compacts the context window, the plugin:
1. Retains the current conversation before compaction
2. Recalls relevant memories and injects them into the compaction context

This ensures memories survive context window trimming.

## Configuration

### Plugin Options

```json
{
  "plugin": [
    ["@vectorize-io/opencode-hindsight", {
      "hindsightApiUrl": "http://localhost:8888",
      "hindsightApiToken": "your-api-key",
      "bankId": "my-project",
      "autoRecall": true,
      "autoRetain": true,
      "recallBudget": "mid",
      "recallMaxTokens": 1024,
      "recallTypes": ["observation", "world", "experience"],
      "recallContextTurns": 1,
      "recallTags": [],
      "recallTagsMatch": "any",
      "retainContext": "conversation between OpenCode Agent and the User",
      "retainTags": [],
      "debug": false
    }]
  ]
}
```

> **Note:** The plugin performs one recall API call per turn and one retain upsert per agent response.
> If you want to reduce API load, you can disable `autoRecall` or `autoRetain`, or lower `recallMaxTokens`.

### Config File

Create `~/.hindsight/opencode.json` for persistent configuration that applies across all projects:

```json
{
  "hindsightApiUrl": "http://localhost:8888",
  "hindsightApiToken": "your-api-key",
  "recallBudget": "mid"
}
```

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `HINDSIGHT_API_URL` | Hindsight API base URL | `https://api.hindsight.vectorize.io` |
| `HINDSIGHT_API_TOKEN` | API key for authentication | |
| `HINDSIGHT_BANK_ID` | Static memory bank ID | `opencode` |
| `HINDSIGHT_AGENT_NAME` | Agent name for dynamic bank IDs | `opencode` |
| `HINDSIGHT_AUTO_RECALL` | Auto-recall on every turn | `true` |
| `HINDSIGHT_AUTO_RETAIN` | Auto-retain on session idle | `true` |
| `HINDSIGHT_RECALL_BUDGET` | Recall budget: `low`, `mid`, `high` | `mid` |
| `HINDSIGHT_RECALL_MAX_TOKENS` | Max tokens for recall results | `1024` |
| `HINDSIGHT_RECALL_MAX_QUERY_CHARS` | Max chars for recall query | `800` |
| `HINDSIGHT_RECALL_CONTEXT_TURNS` | Context turns for recall query | `1` |
| `HINDSIGHT_RECALL_TAGS` | Comma-separated tags to filter recall results | |
| `HINDSIGHT_RECALL_TAGS_MATCH` | Tag match mode: `any`, `all`, `any_strict`, `all_strict` | `any` |
| `HINDSIGHT_RETAIN_TAGS` | Comma-separated tags for retained documents | |
| `HINDSIGHT_DYNAMIC_BANK_ID` | Enable dynamic bank ID derivation | `false` |
| `HINDSIGHT_BANK_MISSION` | Bank mission/context for reflect | |

Configuration priority (later wins): defaults < `~/.hindsight/opencode.json` < plugin options < env vars.

### Logging & debugging

The plugin logs through OpenCode's own log stream (`service=hindsight`), visible with `opencode --print-logs` or in the OpenCode log files. Errors (failed retain/recall, unreachable API, auth problems) and the resolved API URL + bank are logged **by default** — so if memories aren't saving, the reason is visible without any opt-in.

Verbose tracing is controlled by the `debug` option, which is **config-only** (set `"debug": true` in `opencode.json` plugin options or `~/.hindsight/opencode.json`). There is intentionally no `HINDSIGHT_DEBUG` environment variable: env vars are unreliable to set for OpenCode's plugin runtime (notably on Windows, where a persistent OpenCode server may never see them).

```json
{
  "plugin": [["@vectorize-io/opencode-hindsight", { "debug": true }]]
}
```

## Dynamic Bank IDs

For multi-project isolation, enable dynamic bank ID derivation:

```bash
export HINDSIGHT_DYNAMIC_BANK_ID=true
```

The bank ID is composed from granularity fields (default: `agent::project`). Supported fields: `agent`, `project`, `channel`, `user`.

For multi-user scenarios (e.g., shared agent serving multiple users):

```bash
export HINDSIGHT_CHANNEL_ID="slack-general"
export HINDSIGHT_USER_ID="user123"
```

## How It Works

1. **Plugin loads** when OpenCode starts — creates a `HindsightClient`, derives the bank ID, and registers tools + hooks
2. **Every turn** — `system.transform` hook recalls relevant memories keyed on the latest user message and injects them into the system prompt
3. **Agent works** — can call `hindsight_recall` and `hindsight_retain` explicitly during the session
4. **Agent responds** — `session.idle` event fires after each agent response, triggering auto-retain (upsert) of the conversation
5. **Compaction** — if the context window fills up, memories are preserved through the compaction