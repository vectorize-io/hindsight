---
sidebar_position: 6
title: "Cursor Persistent Memory with Hindsight | Integration"
description: "Add long-term memory to Cursor with Hindsight. Automatically recalls relevant context at session start and retains conversations after each task using Cursor's plugin architecture."
---

# Cursor

Biomimetic long-term memory for [Cursor](https://cursor.com) using [Hindsight](https://vectorize.io/hindsight). Automatically recalls relevant project context at session start and retains conversation transcripts after each task — adapted to Cursor's hook-based plugin architecture with MCP for on-demand tools.

[View Changelog →](/changelog/integrations/cursor)

## How It Works

The Hindsight plugin uses two complementary mechanisms:

| | Plugin Hooks (automatic) | MCP Tools (on-demand) |
|--|--------------------------|----------------------|
| **Install** | `pip install hindsight-cursor && hindsight-cursor init` | Configured automatically by `init` |
| **Recall** | Session start — memories injected via `additionalContext` | Agent calls `recall` tool mid-session |
| **Retain** | Automatic on task stop | Agent calls `retain` tool explicitly |
| **Reflect** | Not available via hooks | Available as a tool |
| **Best for** | Ambient project memory with no user intervention | Targeted lookups and explicit memory operations |

Both are set up by a single `hindsight-cursor init` command. Use `--no-mcp` to skip the MCP integration if you only want hooks.

## Quick Start

```bash
# 1. Install the plugin
pip install hindsight-cursor
cd /path/to/your-project

# 2a. Connect to Hindsight Cloud (fastest — no local server needed)
hindsight-cursor init --api-url https://api.hindsight.vectorize.io --api-token YOUR_HINDSIGHT_API_TOKEN

# 2b. Or connect to a local Hindsight server
hindsight-cursor init --api-url http://localhost:8888

# 3. Fully quit and reopen Cursor — plugins load at startup
```

Sign up at [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) to get a token, or start Hindsight locally with Docker:

```bash
export OPENAI_API_KEY=your-key
docker run --rm -it --pull always -p 8888:8888 \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  -e HINDSIGHT_API_LLM_MODEL=gpt-4o-mini \
  -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

### What `init` does

- Copies plugin files into `.cursor-plugin/hindsight-memory/`
- Creates `~/.hindsight/cursor.json` with your connection settings (if the file does not already exist)
- Writes `.cursor/mcp.json` with the Hindsight MCP endpoint for on-demand tools
- Use `--force` to overwrite an existing installation
- Use `--no-mcp` to skip the MCP configuration

:::caution
If you add the plugin to an already-open workspace, **fully quit Cursor and reopen it**. Plugins are loaded at startup — a simple window reload is not enough.
:::

## Features

- **Session recall** — at the start of each session, queries Hindsight for relevant project memories and injects them as context via `additionalContext` (invisible to the chat, visible to the agent)
- **Auto-retain** — after every task completion, extracts and retains conversation content to Hindsight for long-term storage
- **On-demand MCP tools** — `recall`, `retain`, and `reflect` tools for explicit mid-session memory operations
- **On-demand recall skill** — use the `hindsight-recall` skill for manual memory lookups
- **Daemon management** — can auto-start/stop `hindsight-embed` locally or connect to an external Hindsight server
- **Dynamic bank IDs** — supports per-agent, per-project, or per-session memory isolation
- **Zero runtime dependencies** — plugin scripts use pure Python stdlib only

## Architecture

The plugin uses Cursor's hook system:

| Hook | Event | Purpose |
|------|-------|---------|
| `session_start.py` | `sessionStart` | **Session recall** — query memories, inject as `additionalContext` |
| `retain.py` | `stop` | **Auto-retain** — extract transcript, POST to Hindsight |

The `sessionStart` hook fires once when a new Cursor session begins. It performs a broad project-level recall and injects relevant memories as hidden context.

The `init` command also configures Cursor's MCP support (`.cursor/mcp.json`) to connect to Hindsight's MCP endpoint, giving the agent explicit `recall`, `retain`, and `reflect` tools for mid-session use.

Additionally, the plugin provides:
- **Skill** (`hindsight-recall`) — on-demand memory querying
- **Rule** (`hindsight-memory.mdc`) — always-on rule instructing the agent to leverage recalled memories and MCP tools

## Connection Modes

### 1. External API (recommended for production)

Connect to a running Hindsight server (cloud or self-hosted). No local LLM needed — the server handles fact extraction.

```json
{
  "hindsightApiUrl": "https://your-hindsight-server.com",
  "hindsightApiToken": "your-token"
}
```

### 2. Local Daemon (auto-managed)

The plugin automatically starts and stops `hindsight-embed` via `uvx`. Requires an LLM provider API key for local fact extraction.

Set an LLM provider:
```bash
export OPENAI_API_KEY="sk-your-key"
# or
export ANTHROPIC_API_KEY="your-key"
```

The model is selected automatically by the Hindsight API. To override, set `HINDSIGHT_LLM_MODEL`.

### 3. Existing Local Server

If you already have `hindsight-embed` running, leave `hindsightApiUrl` empty and set `apiPort` to match your server's port. The plugin will detect it automatically.

## Configuration

All settings live in `~/.hindsight/cursor.json`. Every setting can also be overridden via environment variables. The plugin ships with sensible defaults — you only need to configure what you want to change.

**Loading order** (later entries win):
1. Built-in defaults (hardcoded in the plugin)
2. Plugin `settings.json` (ships with the plugin, at `CURSOR_PLUGIN_ROOT/settings.json`)
3. User config (`~/.hindsight/cursor.json` — recommended for your overrides)
4. Environment variables

---

### Connection & Daemon

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `hindsightApiUrl` | `HINDSIGHT_API_URL` | `""` (empty) | URL of an external Hindsight API server. When empty, the plugin uses a local daemon instead. |
| `hindsightApiToken` | `HINDSIGHT_API_TOKEN` | `null` | Authentication token for the external API. Only needed when `hindsightApiUrl` is set. |
| `apiPort` | `HINDSIGHT_API_PORT` | `9077` | Port used by the local `hindsight-embed` daemon. |
| `daemonIdleTimeout` | `HINDSIGHT_DAEMON_IDLE_TIMEOUT` | `300` | Seconds of inactivity before the local daemon shuts itself down. `0` means never. |
| `embedVersion` | `HINDSIGHT_EMBED_VERSION` | `"latest"` | Which version of `hindsight-embed` to install via `uvx`. |
| `embedPackagePath` | `HINDSIGHT_EMBED_PACKAGE_PATH` | `null` | Local path to a `hindsight-embed` checkout for development. |

---

### LLM Provider (local daemon only)

These settings configure which LLM the local daemon uses for fact extraction. They are **ignored** when connecting to an external API.

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `llmProvider` | `HINDSIGHT_LLM_PROVIDER` | auto-detect | LLM provider: `openai`, `anthropic`, `gemini`, `groq`, `ollama`. Auto-detects by checking for API key env vars. |
| `llmModel` | `HINDSIGHT_LLM_MODEL` | provider default | Override the default model for the chosen provider. |
| `llmApiKeyEnv` | — | provider standard | Name of the env var holding the API key, if non-standard. |

---

### Memory Bank

A **bank** is an isolated memory store — like a separate "brain."

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `bankId` | `HINDSIGHT_BANK_ID` | `"cursor"` | The bank ID when `dynamicBankId` is `false`. |
| `bankMission` | `HINDSIGHT_BANK_MISSION` | generic assistant prompt | Description of the agent's identity and purpose. |
| `dynamicBankId` | `HINDSIGHT_DYNAMIC_BANK_ID` | `false` | When `true`, derives a unique bank ID from context fields (see `dynamicBankGranularity`). |
| `dynamicBankGranularity` | — | `["agent", "project"]` | Fields to combine for dynamic bank IDs: `agent`, `project`, `session`, `channel`, `user`. |
| `bankIdPrefix` | — | `""` | String prepended to all bank IDs for namespacing. |
| `agentName` | `HINDSIGHT_AGENT_NAME` | `"cursor"` | Name used for the `agent` field in dynamic bank ID derivation. |

---

### Session Recall

Session recall runs once at the start of each session. It queries Hindsight for relevant project memories and injects them into the agent's context as invisible `additionalContext`.

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `autoRecall` | `HINDSIGHT_AUTO_RECALL` | `true` | Master switch for session recall. |
| `recallBudget` | `HINDSIGHT_RECALL_BUDGET` | `"mid"` | Search thoroughness: `"low"`, `"mid"`, `"high"`. |
| `recallMaxTokens` | `HINDSIGHT_RECALL_MAX_TOKENS` | `1024` | Max tokens in the recalled memory block. |
| `recallTypes` | — | `["world", "experience"]` | Memory types to retrieve. |
| `recallMaxQueryChars` | `HINDSIGHT_RECALL_MAX_QUERY_CHARS` | `800` | Max character length of the query. |

---

### Auto-Retain

Auto-retain runs after the agent completes a task. It extracts the conversation transcript and sends it to Hindsight.

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `autoRetain` | `HINDSIGHT_AUTO_RETAIN` | `true` | Master switch for auto-retain. |
| `retainMode` | `HINDSIGHT_RETAIN_MODE` | `"full-session"` | Retention strategy. `"full-session"` or `"chunked"`. |
| `retainEveryNTurns` | `HINDSIGHT_RETAIN_EVERY_N_TURNS` | `10` | How often to retain. `1` = every turn. |
| `retainOverlapTurns` | — | `2` | Extra turns included from the previous chunk for continuity. |
| `retainContext` | `HINDSIGHT_RETAIN_CONTEXT` | `"cursor"` | Source label for retained memories. |
| `retainToolCalls` | — | `false` | Whether to include tool calls in the retained transcript. |

---

### Debug

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `debug` | `HINDSIGHT_DEBUG` | `false` | Enable verbose logging to stderr. Prefixed with `[Hindsight]`. |

## Verifying Plugin Hooks

The plugin writes a status file on every hook invocation — even when no memories are found or retain is skipped. Check them to confirm hooks are firing:

```bash
# Default location when CURSOR_PLUGIN_DATA is not set:
cat ~/.hindsight/cursor-state/state/last_recall.json
cat ~/.hindsight/cursor-state/state/last_retain.json
```

Each file contains:
- `saved_at` — timestamp of the last invocation
- `status` — one of `success`, `empty`, `skipped`, or `error`
- `bank_id` — which bank was used (present on `success` and `empty`)
- `mode` — always `plugin`
- `hook` — `sessionStart` for recall
- `result_count` (recall) or `message_count` (retain) — present on `success`

If `saved_at` updates when you use Cursor, the hooks are firing. Check `status` to understand what happened.

## Troubleshooting

**Plugin not activating**: Check that `.cursor-plugin/plugin.json` exists in the plugin directory. Enable `"debug": true` in `~/.hindsight/cursor.json` and check stderr output.

**Seeing "Ran Recall in hindsight" in the Agent Window?** That is MCP, not the plugin. Plugin-based recall is silent — it injects context via `additionalContext` without a visible tool call. If you see explicit Hindsight tool calls, you have MCP configured in `.cursor/mcp.json`. Both can work together.

**Recall returning no memories**: Verify the Hindsight server is reachable (`curl http://localhost:9077/health`). Memories need at least one retain cycle.

**Daemon not starting**: Ensure an LLM API key is set. Review daemon logs at `~/.hindsight/profiles/cursor.log`.

**High latency on session start**: The session recall hook has a 15-second timeout. Use `recallBudget: "low"` or reduce `recallMaxTokens`.
