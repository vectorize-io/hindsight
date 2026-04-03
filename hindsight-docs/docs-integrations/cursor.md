---
sidebar_position: 6
title: "Cursor Persistent Memory with Hindsight | Integration"
description: "Add long-term memory to Cursor with Hindsight. Automatically captures conversations and recalls relevant context across sessions using Cursor's plugin architecture."
---

# Cursor

Biomimetic long-term memory for [Cursor](https://cursor.com) using [Hindsight](https://vectorize.io/hindsight). Automatically captures conversations and intelligently recalls relevant context — adapted to Cursor's hook-based plugin architecture.

## Quick Start

```bash
# 1. Copy the plugin into your project
cp -r hindsight-integrations/cursor /path/to/your-project/.cursor-plugin/hindsight-memory

# 2. Configure your LLM provider for memory extraction
# Option A: OpenAI (auto-detected)
export OPENAI_API_KEY="sk-your-key"

# Option B: Anthropic (auto-detected)
export ANTHROPIC_API_KEY="your-key"

# Option C: Connect to an external Hindsight server
mkdir -p ~/.hindsight
echo '{"hindsightApiUrl": "https://your-hindsight-server.com"}' > ~/.hindsight/cursor.json

# 3. Open Cursor — the plugin activates automatically
```

That's it! The plugin will automatically start capturing and recalling memories.

:::tip Alternative: MCP Integration
Cursor also supports MCP servers natively. If you prefer MCP over the plugin system, see [Local MCP Server](./local-mcp) and add Hindsight to `.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "hindsight": { "url": "http://localhost:8888/mcp/" }
  }
}
```
:::

## Features

- **Auto-recall** — on every user prompt, queries Hindsight for relevant memories and injects them as context via `additionalContext` (invisible to the chat, visible to the agent)
- **Auto-retain** — after every task completion, extracts and retains conversation content to Hindsight for long-term storage
- **On-demand recall** — use the `hindsight-recall` skill to manually query memories mid-conversation
- **Daemon management** — can auto-start/stop `hindsight-embed` locally or connect to an external Hindsight server
- **Dynamic bank IDs** — supports per-agent, per-project, or per-session memory isolation
- **Zero dependencies** — pure Python stdlib, no pip install required

## Architecture

The plugin uses Cursor's hook system:

| Hook | Event | Purpose |
|------|-------|---------|
| `recall.py` | `beforeSubmitPrompt` | **Auto-recall** — query memories, inject as `additionalContext` |
| `retain.py` | `stop` | **Auto-retain** — extract transcript, POST to Hindsight |

Additionally, the plugin provides:
- **Skill** (`hindsight-recall`) — on-demand memory querying via `/hindsight-recall`
- **Rule** (`hindsight-memory.mdc`) — always-on rule instructing the agent to leverage recalled memories proactively

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
| `daemonIdleTimeout` | `HINDSIGHT_DAEMON_IDLE_TIMEOUT` | `0` | Seconds of inactivity before the local daemon shuts itself down. `0` means the daemon stays running until the session ends. |
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

### Auto-Recall

Auto-recall runs on every user prompt. It queries Hindsight for relevant memories and injects them into the agent's context as invisible `additionalContext`.

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `autoRecall` | `HINDSIGHT_AUTO_RECALL` | `true` | Master switch for auto-recall. |
| `recallBudget` | `HINDSIGHT_RECALL_BUDGET` | `"mid"` | Search thoroughness: `"low"`, `"mid"`, `"high"`. |
| `recallMaxTokens` | `HINDSIGHT_RECALL_MAX_TOKENS` | `1024` | Max tokens in the recalled memory block. |
| `recallTypes` | — | `["world", "experience"]` | Memory types to retrieve. |
| `recallContextTurns` | `HINDSIGHT_RECALL_CONTEXT_TURNS` | `1` | Prior turns to include in the recall query. |
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

## Troubleshooting

**Plugin not activating**: Check that `.cursor-plugin/plugin.json` exists in the plugin directory. Enable `"debug": true` in `~/.hindsight/cursor.json` and check stderr output.

**Recall returning no memories**: Verify the Hindsight server is reachable (`curl http://localhost:9077/health`). Memories need at least one retain cycle.

**Daemon not starting**: Ensure an LLM API key is set. Review daemon logs at `~/.hindsight/profiles/cursor.log`.

**High latency on recall**: The recall hook has a 12-second timeout. Use `recallBudget: "low"` or reduce `recallMaxTokens`.
