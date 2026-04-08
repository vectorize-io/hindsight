---
sidebar_position: 10
title: "Hermes Agent Persistent Memory with Hindsight | Integration"
description: "Add long-term memory to Hermes Agent with Hindsight. Automatically recalls context before every LLM call and retains conversations for future sessions."
---

# Hermes Agent

Persistent long-term memory for [Hermes Agent](https://github.com/NousResearch/hermes-agent) using [Hindsight](https://vectorize.io/hindsight). Automatically recalls relevant context before every LLM call and retains conversations for future sessions — plus explicit retain/recall/reflect tools.

## Quick Start

**1. Get an API key** at [ui.hindsight.vectorize.io/connect](https://ui.hindsight.vectorize.io/connect). The API endpoint is `https://api.hindsight.vectorize.io`.

**2. Run the setup wizard:**

```bash
hermes memory setup    # select "hindsight"
```

The wizard will prompt for your API key and API URL, and configure everything automatically.

Or configure manually:

```bash
hermes config set memory.provider hindsight
# Add your key and the API endpoint
echo "HINDSIGHT_API_KEY=your-key" >> ~/.hermes/.env
echo "HINDSIGHT_API_URL=https://api.hindsight.vectorize.io" >> ~/.hermes/.env
```

**3. Confirm memory is active:**

```bash
hermes memory status
```

## Features

- **Auto-recall** — on every turn, queries Hindsight for relevant memories and injects them into the system prompt (via `pre_llm_call` hook)
- **Auto-retain** — after every response, retains the user/assistant exchange to Hindsight (via `post_llm_call` hook)
- **Explicit tools** — `hindsight_retain`, `hindsight_recall`, `hindsight_reflect` for direct model control
- **Memory modes** — choose between automatic injection, tools-only, or hybrid
- **Zero config overhead** — env vars work as overrides for CI/automation

:::note
The lifecycle hooks (`pre_llm_call`/`post_llm_call`) require hermes-agent with [PR #2823](https://github.com/NousResearch/hermes-agent/pull/2823) or later. On older versions, only the three tools are registered — hooks are silently skipped.
:::

## Architecture

The plugin registers via Hermes's `hermes_agent.plugins` entry point system:

| Component | Purpose |
|-----------|---------|
| `pre_llm_call` hook | **Auto-recall** — query memories, inject as ephemeral system prompt context |
| `post_llm_call` hook | **Auto-retain** — store user/assistant exchange to Hindsight |
| `hindsight_retain` tool | Explicit memory storage (model-initiated) |
| `hindsight_recall` tool | Explicit memory search (model-initiated) |
| `hindsight_reflect` tool | LLM-synthesized answer from stored memories |

## Connection Modes

### 1. Cloud (recommended for production)

Connect to Hindsight Cloud at `https://api.hindsight.vectorize.io`. Get an API key at [ui.hindsight.vectorize.io/connect](https://ui.hindsight.vectorize.io/connect).

```json
{
  "mode": "cloud",
  "api_url": "https://api.hindsight.vectorize.io",
  "api_key": "hsk_your_token",
  "bank_id": "hermes"
}
```

### 2. Local (embedded)

Runs an embedded Hindsight server with built-in PostgreSQL. Requires an LLM API key for memory extraction and synthesis. The daemon starts automatically in the background on first use.

```json
{
  "mode": "local",
  "llm_provider": "groq",
  "llm_api_key": "your-groq-key"
}
```

:::note
The embedded server starts on the first message when Hermes says "starting agent". On a fresh system this can take over a minute while the embedded PostgreSQL initializes. Subsequent startups are fast.
:::

Daemon startup logs: `~/.hermes/logs/hindsight-embed.log`  
Daemon runtime logs: `~/.hindsight/profiles/<profile>.log`

## Configuration

All settings are in `~/.hermes/hindsight/config.json`. Every setting can also be overridden via environment variables (env vars take priority).

### Connection & Daemon

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `mode` | `cloud` | `HINDSIGHT_MODE` | `cloud` or `local` |
| `api_url` | `https://api.hindsight.vectorize.io` | `HINDSIGHT_API_URL` | Hindsight API URL |
| `api_key` | `null` | `HINDSIGHT_API_KEY` | Auth token for Hindsight Cloud |
| `apiPort` | `9077` | `HINDSIGHT_API_PORT` | Port for local Hindsight daemon |
| `daemonIdleTimeout` | `0` | `HINDSIGHT_DAEMON_IDLE_TIMEOUT` | Seconds before idle daemon shuts down (0 = never) |
| `embedVersion` | `"latest"` | `HINDSIGHT_EMBED_VERSION` | `hindsight-embed` version for `uvx` |

### LLM Provider (local mode only)

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `llm_provider` | `openai` | `HINDSIGHT_LLM_PROVIDER` | LLM provider: `openai`, `anthropic`, `gemini`, `groq`, `minimax`, `ollama`, `lmstudio` |
| `llm_api_key` | — | `HINDSIGHT_LLM_API_KEY` | API key for the chosen LLM provider |
| `llm_model` | provider default | `HINDSIGHT_LLM_MODEL` | Model override (auto-defaults per provider) |

Default models per provider: `openai` → `gpt-4o-mini`, `anthropic` → `claude-haiku-4-5`, `gemini` → `gemini-2.5-flash`, `groq` → `openai/gpt-oss-120b`, `minimax` → `MiniMax-M2.7`, `ollama` → `gemma3:12b`.

### Memory Bank

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `bank_id` | `hermes` | `HINDSIGHT_BANK_ID` | Memory bank ID |
| `bankMission` | `""` | `HINDSIGHT_BANK_MISSION` | Agent identity/purpose for the memory bank |
| `retainMission` | `null` | — | Custom retain mission (what to extract from conversations) |

### Auto-Recall

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `autoRecall` | `true` | `HINDSIGHT_AUTO_RECALL` | Enable automatic memory recall via `pre_llm_call` hook |
| `recallBudget` | `"mid"` | `HINDSIGHT_RECALL_BUDGET` | Recall effort: `low`, `mid`, `high` |
| `recallMaxTokens` | `4096` | `HINDSIGHT_RECALL_MAX_TOKENS` | Max tokens in recall response |
| `recallMaxQueryChars` | `800` | `HINDSIGHT_RECALL_MAX_QUERY_CHARS` | Max chars of user message used as query |
| `recallPromptPreamble` | see below | — | Header text injected before recalled memories |

Default preamble:
> Relevant memories from past conversations (prioritize recent when conflicting). Only use memories that are directly useful to continue this conversation; ignore the rest:

### Auto-Retain

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `autoRetain` | `true` | `HINDSIGHT_AUTO_RETAIN` | Enable automatic retention via `post_llm_call` hook |
| `retainEveryNTurns` | `1` | — | Retain every Nth turn |
| `retainOverlapTurns` | `2` | — | Extra overlap turns for continuity |
| `retainRoles` | `["user", "assistant"]` | — | Which message roles to retain |

### Integration Mode

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `memory_mode` | `hybrid` | — | How memories are integrated into the agent (see below) |
| `prefetch_method` | `recall` | — | Method used for automatic context injection (see below) |

**memory_mode:**
- `hybrid` — automatic context injection before each turn, plus tools available to the LLM
- `context` — automatic injection only; no tools exposed to the model
- `tools` — tools only (`hindsight_retain`, `hindsight_recall`, `hindsight_reflect`); no automatic injection

**prefetch_method:**
- `recall` — injects raw memory facts into the system prompt (fast)
- `reflect` — injects an LLM-synthesized summary of relevant memories (slower, more coherent)

### Miscellaneous

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `debug` | `false` | `HINDSIGHT_DEBUG` | Enable debug logging to stderr |

## Hermes Gateway (Telegram, Discord, Slack)

When using Hermes in gateway mode (multi-platform messaging), the plugin works across all platforms. Hermes creates a fresh `AIAgent` per message, and the plugin's `pre_llm_call` hook ensures relevant memories are recalled for each turn regardless of platform.

## Disabling Hermes's Built-in Memory

Hermes has a built-in `memory` tool that saves to local markdown files. If both are active, the LLM may prefer the built-in one. Disable it:

```bash
hermes tools disable memory
```

Re-enable later with `hermes tools enable memory`.

## Troubleshooting

**Plugin not loading**: Verify the entry point is registered:
```bash
python -c "
import importlib.metadata
eps = importlib.metadata.entry_points(group='hermes_agent.plugins')
print(list(eps))
"
```
You should see `EntryPoint(name='hindsight', value='hindsight_hermes', ...)`.

**Tools don't appear in `/tools`**: Check that `api_url` (or `HINDSIGHT_API_URL`) is set, or that `HINDSIGHT_API_KEY` is set for cloud mode. The plugin silently skips tool registration when unconfigured.

**Connection refused**: Verify the Hindsight API is running:
```bash
curl http://localhost:9077/health
```

**Local daemon not starting**: Check the daemon log for errors:
```bash
cat ~/.hermes/logs/hindsight-embed.log
```

**Recall returning no memories**: Memories need at least one retain cycle. Try storing a fact first, then asking about it in a new session.
