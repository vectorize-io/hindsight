---
sidebar_position: 5
title: "Claude Code Persistent Memory with Hindsight | Integration"
description: "Add long-term memory to Claude Code with Hindsight. Automatically captures conversations and recalls relevant context across sessions using Claude Code's hook-based architecture."
---

# Claude Code

Biomimetic long-term memory for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) using [Hindsight](https://vectorize.io/hindsight). Automatically captures conversations and intelligently recalls relevant context — a complete port of [`hindsight-openclaw`](./openclaw) adapted to Claude Code's hook-based plugin architecture.

[View Changelog →](/changelog/integrations/claude-code)

## Quick Start

```bash
# 1. Add the Hindsight marketplace and install the plugin
claude plugin marketplace add vectorize-io/hindsight
claude plugin install hindsight-memory

# 2. Configure your LLM provider for memory extraction
# Option A: OpenAI (auto-detected)
export OPENAI_API_KEY="sk-your-key"

# Option B: Anthropic (auto-detected)
export ANTHROPIC_API_KEY="your-key"

# Option C: No API key needed (uses Claude Code's own model — personal/local use only)
export HINDSIGHT_LLM_PROVIDER=claude-code

# Option D: Connect to an external Hindsight server instead of running locally
mkdir -p ~/.hindsight
echo '{"hindsightApiUrl": "https://your-hindsight-server.com"}' > ~/.hindsight/claude-code.json

# 3. Start Claude Code — the plugin activates automatically
claude
```

That's it! The plugin will automatically start capturing and recalling memories.

## Features

- **Auto-recall** — on every user prompt, queries Hindsight for relevant memories and injects them as context (invisible to the chat transcript, visible to Claude)
- **Auto-retain** — after every response (or every N turns), extracts and retains conversation content to Hindsight for long-term storage
- **Daemon management** — can auto-start/stop `hindsight-embed` locally or connect to an external Hindsight server
- **Dynamic bank IDs** — supports per-agent, per-project, or per-session memory isolation
- **Channel-agnostic** — works with Claude Code Channels (Telegram, Discord, Slack) or interactive sessions
- **Zero dependencies** — pure Python stdlib, no pip install required

## Architecture

The plugin uses all four Claude Code hook events:

| Hook | Event | Purpose |
|------|-------|---------|
| `session_start.py` | `SessionStart` | Health check — verify Hindsight is reachable |
| `recall.py` | `UserPromptSubmit` | **Auto-recall** — query memories, inject as `additionalContext` |
| `retain.py` | `Stop` | **Auto-retain** — extract transcript, POST to Hindsight (async) |
| `session_end.py` | `SessionEnd` | Cleanup — stop auto-managed daemon if started |

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
# or
export HINDSIGHT_LLM_PROVIDER=claude-code # No API key needed
```

The model is selected automatically by the Hindsight API. To override, set `HINDSIGHT_LLM_MODEL`.

### 3. Existing Local Server

If you already have `hindsight-embed` running, leave `hindsightApiUrl` empty and set `apiPort` to match your server's port. The plugin will detect it automatically.

## Configuration

All settings live in `~/.hindsight/claude-code.json`. Every setting can also be overridden via environment variables. The plugin ships with sensible defaults — you only need to configure what you want to change.

**Loading order** (later entries win):
1. Built-in defaults (hardcoded in the plugin)
2. Plugin `settings.json` (ships with the plugin, at `CLAUDE_PLUGIN_ROOT/settings.json`)
3. User config (`~/.hindsight/claude-code.json` — recommended for your overrides)
4. Environment variables

---

### Connection & Daemon

These settings control how the plugin connects to the Hindsight API.

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `hindsightApiUrl` | `HINDSIGHT_API_URL` | `""` (empty) | URL of an external Hindsight API server. When empty, the plugin uses a local daemon instead. |
| `hindsightApiToken` | `HINDSIGHT_API_TOKEN` | `null` | Authentication token for the external API. Only needed when `hindsightApiUrl` is set. |
| `apiPort` | `HINDSIGHT_API_PORT` | `9077` | Port used by the local `hindsight-embed` daemon. Change this if you run multiple instances or have a port conflict. |
| `daemonIdleTimeout` | `HINDSIGHT_DAEMON_IDLE_TIMEOUT` | `0` | Seconds of inactivity before the local daemon shuts itself down. `0` means the daemon stays running until the session ends. |
| `embedVersion` | `HINDSIGHT_EMBED_VERSION` | `"latest"` | Which version of `hindsight-embed` to install via `uvx`. Pin to a specific version (e.g. `"0.5.2"`) for reproducibility. |
| `embedPackagePath` | `HINDSIGHT_EMBED_PACKAGE_PATH` | `null` | Local filesystem path to a `hindsight-embed` checkout. When set, the plugin runs from this path instead of installing via `uvx`. Useful for development. |

---

### LLM Provider (local daemon only)

These settings configure which LLM the local daemon uses for fact extraction. They are **ignored** when connecting to an external API (the server uses its own LLM configuration).

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `llmProvider` | `HINDSIGHT_LLM_PROVIDER` | auto-detect | Which LLM provider to use. Supported values: `openai`, `anthropic`, `gemini`, `groq`, `ollama`, `openai-codex`, `claude-code`. When omitted, the plugin auto-detects by checking for API key env vars in order: `OPENAI_API_KEY` → `ANTHROPIC_API_KEY` → `GEMINI_API_KEY` → `GROQ_API_KEY`. |
| `llmModel` | `HINDSIGHT_LLM_MODEL` | provider default | Override the default model for the chosen provider (e.g. `"gpt-4o"`, `"claude-sonnet-4-20250514"`). When omitted, the Hindsight API picks a sensible default for each provider. |
| `llmApiKeyEnv` | — | provider standard | Name of the environment variable that holds the API key. Normally auto-detected (e.g. `OPENAI_API_KEY` for the `openai` provider). Set this only if your key is in a non-standard env var. |

---

### Memory Bank

A **bank** is an isolated memory store — like a separate "brain." These settings control which bank the plugin reads from and writes to.

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `bankId` | `HINDSIGHT_BANK_ID` | `"claude_code"` | The bank ID to use when `dynamicBankId` is `false`. All sessions share this single bank. |
| `bankMission` | `HINDSIGHT_BANK_MISSION` | generic assistant prompt | A short description of the agent's identity and purpose. Sent to Hindsight when creating or updating the bank, and used during recall to contextualize results. |
| `retainMission` | — | extraction prompt | Instructions for the fact extraction LLM — tells it *what* to extract from conversations (e.g. "Extract technical decisions and user preferences"). |
| `dynamicBankId` | `HINDSIGHT_DYNAMIC_BANK_ID` | `false` | When `true`, the plugin derives a unique bank ID from context fields (see `dynamicBankGranularity`), giving each combination its own isolated memory. |
| `dynamicBankGranularity` | — | `["agent", "project"]` | Which context fields to combine when building a dynamic bank ID. Available fields: `agent` (agent name), `project` (working directory), `session` (session ID), `channel` (channel ID), `user` (user ID). |
| `bankIdPrefix` | — | `""` | A string prepended to all bank IDs — both static and dynamic. Useful for namespacing (e.g. `"prod"` or `"staging"`). |
| `agentName` | `HINDSIGHT_AGENT_NAME` | `"claude-code"` | Name used for the `agent` field in dynamic bank ID derivation. |

---

### Auto-Recall

Auto-recall runs on every user prompt. It queries Hindsight for relevant memories and injects them into Claude's context as invisible `additionalContext` (the user doesn't see them in the chat transcript).

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `autoRecall` | `HINDSIGHT_AUTO_RECALL` | `true` | Master switch for auto-recall. Set to `false` to disable memory retrieval entirely. |
| `recallBudget` | `HINDSIGHT_RECALL_BUDGET` | `"mid"` | Controls how hard Hindsight searches for memories. `"low"` = fast, fewer strategies; `"mid"` = balanced; `"high"` = thorough, slower. Affects latency directly. |
| `recallMaxTokens` | `HINDSIGHT_RECALL_MAX_TOKENS` | `1024` | Maximum number of tokens in the recalled memory block. Lower values reduce context usage but may truncate relevant memories. |
| `recallTypes` | — | `["world", "experience"]` | Which memory types to retrieve. `"world"` = general facts; `"experience"` = personal experiences; `"observation"` = raw observations. |
| `recallContextTurns` | `HINDSIGHT_RECALL_CONTEXT_TURNS` | `1` | How many prior conversation turns to include when composing the recall query. `1` = only the latest user message; higher values give more context but may dilute the query. |
| `recallMaxQueryChars` | `HINDSIGHT_RECALL_MAX_QUERY_CHARS` | `800` | Maximum character length of the query sent to Hindsight. Longer queries are truncated. |
| `recallRoles` | — | `["user", "assistant"]` | Which message roles to include when building the recall query from prior turns. |
| `recallPromptPreamble` | — | built-in string | Text placed above the recalled memories in the injected context block. Customize this to change how Claude interprets the memories. |

---

### Auto-Retain

Auto-retain runs after Claude responds. It extracts the conversation transcript and sends it to Hindsight for long-term storage and fact extraction.

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `autoRetain` | `HINDSIGHT_AUTO_RETAIN` | `true` | Master switch for auto-retain. Set to `false` to disable memory storage entirely. |
| `retainMode` | `HINDSIGHT_RETAIN_MODE` | `"full-session"` | Retention strategy. `"full-session"` sends the full conversation transcript (with chunking). |
| `retainEveryNTurns` | — | `10` | How often to retain. `1` = every turn; `10` = every 10th turn. Higher values reduce API calls but delay memory capture. Values > 1 enable **chunked retention** with a sliding window. |
| `retainOverlapTurns` | — | `2` | When chunked retention fires, this many extra turns from the previous chunk are included for continuity. Total window size = `retainEveryNTurns + retainOverlapTurns`. |
| `retainRoles` | — | `["user", "assistant"]` | Which message roles to include in the retained transcript. |
| `retainToolCalls` | — | `true` | Whether to include tool calls (function invocations and results) in the retained transcript. Captures structured actions like file reads, searches, and code edits. |
| `retainTags` | — | `["{session_id}"]` | Tags attached to the retained document. Supports `{session_id}` placeholder which is replaced with the current session ID at runtime. |
| `retainMetadata` | — | `{}` | Arbitrary key-value metadata attached to the retained document. |
| `retainContext` | — | `"claude-code"` | A label attached to retained memories identifying their source. Useful when multiple integrations write to the same bank. |

---

### Debug

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `debug` | `HINDSIGHT_DEBUG` | `false` | Enable verbose logging to stderr. All log lines are prefixed with `[Hindsight]`. Useful for diagnosing connection issues, recall/retain behavior, and bank ID derivation. |

## Claude Code Channels

With [Claude Code Channels](https://docs.anthropic.com/en/docs/claude-code), Claude Code can operate as a persistent background agent connected to Telegram, Discord, Slack, and other messaging platforms. This plugin gives Channel-based agents the same long-term memory that `hindsight-openclaw` provides for Openclaw agents.

For Channel agents, enable dynamic bank IDs for per-channel/per-user memory isolation:

```json
{
  "dynamicBankId": true,
  "dynamicBankGranularity": ["agent", "channel", "user"]
}
```

And set channel context via environment variables:

```bash
export HINDSIGHT_CHANNEL_ID="telegram-group-12345"
export HINDSIGHT_USER_ID="user-67890"
```

## Troubleshooting

**Plugin not activating**: Check Claude Code logs for `[Hindsight]` messages. Enable `"debug": true` in `~/.hindsight/claude-code.json`.

**Recall returning no memories**: Verify the Hindsight server is reachable (`curl http://localhost:9077/health`). Memories need at least one retain cycle before they're available.

**Daemon not starting**: Ensure an LLM API key is set (or use `HINDSIGHT_LLM_PROVIDER=claude-code`). Review daemon logs at `~/.hindsight/profiles/claude-code.log`.

**High latency on recall**: The recall hook has a 12-second timeout. Use `recallBudget: "low"` or reduce `recallMaxTokens` for faster responses.
