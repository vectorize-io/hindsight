---
sidebar_position: 6
title: "OpenCode Persistent Memory with Hindsight | Integration"
description: "Add long-term memory to OpenCode with Hindsight. Automatically captures coding sessions, recalls relevant context via MCP, and supports backfilling historical sessions."
---

# OpenCode

Persistent long-term memory for [OpenCode](https://opencode.ai) using [Hindsight](https://vectorize.io/hindsight). Automatically captures coding sessions and recalls relevant context across sessions via MCP.

## Quick Start

```bash
# 1. Install the plugin (in your opencode.json)
```

```json title="opencode.json"
{
  "plugin": ["opencode-hindsight"]
}
```

```bash
# 2. Configure your LLM provider for memory extraction
# Option A: OpenAI (auto-detected)
export OPENAI_API_KEY="sk-your-key"

# Option B: Anthropic (auto-detected)
export ANTHROPIC_API_KEY="your-key"

# Option C: Connect to an external Hindsight server
mkdir -p ~/.hindsight
echo '{"hindsightApiUrl": "https://your-hindsight-server.com"}' > ~/.hindsight/opencode.json

# 3. Start OpenCode -- the plugin activates automatically
opencode
```

### Add On-Demand Recall (MCP)

For the agent to actively recall and retain memories during sessions, add Hindsight's built-in MCP server to your OpenCode config:

```json title="opencode.json"
{
  "plugin": ["opencode-hindsight"],
  "mcp": {
    "hindsight": {
      "type": "remote",
      "url": "http://localhost:9077/mcp/opencode/"
    }
  }
}
```

The default port is `9077` (the `hindsight-embed` daemon default). If you're running the full Hindsight API server, the default is `8888` instead.

This gives the agent `retain`, `recall`, and `reflect` tools natively via MCP.

## Features

- **Auto-retain** -- on every `session.idle` event, extracts the conversation from OpenCode's database and retains it to Hindsight for long-term storage
- **MCP recall** -- Hindsight's built-in MCP server provides retain/recall/reflect tools directly to the agent
- **Compaction memory** -- recalled memories are injected into compaction context, so they persist across session compaction boundaries
- **Dynamic bank IDs** -- supports per-project memory isolation based on working directory
- **Backfill CLI** -- import historical OpenCode sessions and JSONL transcripts
- **Daemon management** -- auto-starts `hindsight-embed` locally or connects to an external Hindsight server

## Architecture

The plugin uses two OpenCode hook points:

| Hook | Event | Purpose |
|------|-------|---------|
| `event` | `session.idle` | **Auto-retain**: read conversation from SQLite, POST to Hindsight |
| `experimental.session.compacting` | compaction | **Context injection**: recall memories and inject into compaction prompt |

For on-demand recall during active sessions, the MCP server approach is used (no plugin code needed; Hindsight exposes `/mcp/{bank_id}/` natively).

### How Auto-Retain Works

```
session.idle event
      |
      v
  Read OpenCode SQLite DB (~/.local/share/opencode/opencode.db)
      |
      v
  Reconstruct transcript from message + part tables
      |
      v
  POST /v1/default/banks/{bank_id}/retain
  (document_id = session ID for idempotent upserts)
```

The plugin reads conversations directly from OpenCode's SQLite database using `bun:sqlite`. Each session is retained with a `document_id` derived from the session ID, making re-runs idempotent (no duplicate memories).

## Connection Modes

### 1. External API (recommended for production)

Connect to a running Hindsight server (cloud or self-hosted):

```json title="~/.hindsight/opencode.json"
{
  "hindsightApiUrl": "https://your-hindsight-server.com",
  "hindsightApiToken": "your-token"
}
```

### 2. Local Daemon (auto-managed)

The plugin auto-starts `hindsight-embed` via `uvx`. Requires an LLM provider API key:

```bash
export OPENAI_API_KEY="sk-your-key"
```

### 3. Existing Local Server

If `hindsight-embed` is already running, the plugin detects it on the configured port (default: 9077).

## Configuration

All settings live in `~/.hindsight/opencode.json`. Every setting can be overridden via environment variables.

### Connection

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `hindsightApiUrl` | `HINDSIGHT_API_URL` | `""` | External Hindsight API URL. Empty uses local daemon. |
| `hindsightApiToken` | `HINDSIGHT_API_TOKEN` | `null` | Auth token for external API. |
| `apiPort` | `HINDSIGHT_API_PORT` | `9077` | Port for local daemon. |
| `embedVersion` | `HINDSIGHT_EMBED_VERSION` | `"latest"` | Version of `hindsight-embed` to install. |

### Memory Bank

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `bankId` | `HINDSIGHT_BANK_ID` | `"opencode"` | Static bank ID (when `dynamicBankId` is false). |
| `dynamicBankId` | `HINDSIGHT_DYNAMIC_BANK_ID` | `false` | Derive bank ID from context (e.g., project directory). |
| `dynamicBankGranularity` | -- | `["project"]` | Fields for dynamic bank ID: `agent`, `project`. |
| `bankMission` | `HINDSIGHT_BANK_MISSION` | coding assistant prompt | Bank identity for reasoning context. |

### Auto-Retain

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `autoRetain` | `HINDSIGHT_AUTO_RETAIN` | `true` | Enable auto-retain on session completion. |
| `retainContext` | -- | `"opencode"` | Context label attached to retained memories. |
| `retainToolCalls` | -- | `false` | Include tool call markers in transcripts. |
| `retainMinChars` | -- | `200` | Skip sessions shorter than this. |
| `retainSkipSubagent` | -- | `true` | Skip subagent (Task tool) sessions. |

### Auto-Recall

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `autoRecall` | `HINDSIGHT_AUTO_RECALL` | `true` | Enable memory injection during compaction. |
| `recallBudget` | `HINDSIGHT_RECALL_BUDGET` | `"mid"` | Search depth: `low`, `mid`, `high`. |
| `recallMaxTokens` | `HINDSIGHT_RECALL_MAX_TOKENS` | `1024` | Max tokens for recalled memories. |

### Debug

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `debug` | `HINDSIGHT_DEBUG` | `false` | Verbose logging to stderr. |

## Backfilling Historical Sessions

The backfill CLI imports existing session history into Hindsight. It supports two sources:

### OpenCode Sessions

```bash
pip install hindsight-client

# Backfill all sessions
python backfill.py opencode \
  --hindsight-url http://localhost:8888 \
  --bank-id opencode

# Backfill specific project, since a date
python backfill.py opencode \
  --hindsight-url http://localhost:8888 \
  --bank-id opencode \
  --since 2026-03-01 \
  --project my-project \
  --verbose

# Preview without ingesting
python backfill.py opencode \
  --hindsight-url http://localhost:8888 \
  --bank-id opencode \
  --dry-run
```

### JSONL Transcripts

For Claude Code transcripts or any conversation logs in JSONL format:

```bash
python backfill.py jsonl \
  --hindsight-url http://localhost:8888 \
  --bank-id my-agent \
  --input "./transcripts/*.jsonl"
```

Supported JSONL formats:
- `{"role": "user", "content": "..."}` (flat)
- `{"type": "user", "message": {"role": "user", "content": "..."}}` (Claude Code nested)

### Backfill Options

| Flag | Default | Description |
|------|---------|-------------|
| `--since DATE` | all | Only sessions after this ISO date |
| `--project NAME` | all | Only sessions in this project directory |
| `--skip-subagent` | `true` | Skip subagent sessions |
| `--min-chars N` | `200` | Minimum transcript length |
| `--include-tools` | `false` | Include tool call markers |
| `--async` | `false` | Use async retain (faster, 50% cost savings) |
| `--dry-run` | `false` | Preview without ingesting |

## Troubleshooting

**Plugin not activating**: Enable `"debug": true` in `~/.hindsight/opencode.json` and check stderr for `[Hindsight]` messages.

**No memories after retain**: Memories need fact extraction (async). Check `GET /v1/default/banks/{bank_id}/operations` on the Hindsight API for pending operations.

**Daemon not starting**: Ensure an LLM API key is set (e.g., `OPENAI_API_KEY`) and `uvx` is available on PATH.

**MCP tools not appearing**: Verify the Hindsight server is running and the MCP URL in `opencode.json` matches. Test with `curl http://localhost:8888/health`.
