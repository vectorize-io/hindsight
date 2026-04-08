---
sidebar_position: 20
title: "OpenCode Persistent Memory with Hindsight | Integration"
description: "Add long-term memory to OpenCode with Hindsight. Automatically captures conversations and recalls relevant context across coding sessions."
---

# OpenCode

Persistent long-term memory plugin for [OpenCode](https://opencode.ai) using [Hindsight](https://vectorize.io/hindsight). Automatically captures conversations, recalls relevant context on session start, and provides retain/recall/reflect tools the agent can call directly.

## Quick Start

```bash
# 1. Install the plugin
npm install @vectorize-io/opencode-hindsight
```

Add to your `opencode.json`:

```json
{
  "plugin": ["@vectorize-io/opencode-hindsight"]
}
```

```bash
# 2. Configure your Hindsight server
export HINDSIGHT_API_URL="http://localhost:8888"

# Optional: API key for Hindsight Cloud
export HINDSIGHT_API_TOKEN="your-api-key"

# 3. Start OpenCode — the plugin activates automatically
opencode
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

When the session goes idle (`session.idle` event), the plugin automatically retains the conversation transcript to Hindsight. Configurable via `retainEveryNTurns` to control frequency.

### Session Recall

When a new session starts, the plugin recalls relevant project context and injects it into the system prompt, giving the agent access to memories from prior sessions.

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
      "retainEveryNTurns": 10,
      "debug": false
    }]
  ]
}
```

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
| `HINDSIGHT_API_URL` | Hindsight API base URL | *(required)* |
| `HINDSIGHT_API_TOKEN` | API key for authentication | |
| `HINDSIGHT_BANK_ID` | Static memory bank ID | `opencode` |
| `HINDSIGHT_AGENT_NAME` | Agent name for dynamic bank IDs | `opencode` |
| `HINDSIGHT_AUTO_RECALL` | Auto-recall on session start | `true` |
| `HINDSIGHT_AUTO_RETAIN` | Auto-retain on session idle | `true` |
| `HINDSIGHT_RETAIN_MODE` | `full-session` or `last-turn` | `full-session` |
| `HINDSIGHT_RECALL_BUDGET` | Recall budget: `low`, `mid`, `high` | `mid` |
| `HINDSIGHT_RECALL_MAX_TOKENS` | Max tokens for recall results | `1024` |
| `HINDSIGHT_DYNAMIC_BANK_ID` | Enable dynamic bank ID derivation | `false` |
| `HINDSIGHT_BANK_MISSION` | Bank mission/context for reflect | |
| `HINDSIGHT_DEBUG` | Enable debug logging to stderr | `false` |

Configuration priority (later wins): defaults < `~/.hindsight/opencode.json` < plugin options < env vars.

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
2. **Session starts** — `session.created` event triggers, plugin marks session for recall injection
3. **System transform** — on the first LLM call, recalled memories are injected into the system prompt
4. **Agent works** — can call `hindsight_recall` and `hindsight_retain` explicitly during the session
5. **Session idles** — `session.idle` event triggers auto-retain of the conversation
6. **Compaction** — if the context window fills up, memories are preserved through the compaction
