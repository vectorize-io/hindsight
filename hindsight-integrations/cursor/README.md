# Hindsight Memory Plugin for Cursor

Biomimetic long-term memory for [Cursor](https://cursor.com) using [Hindsight](https://vectorize.io/hindsight). Automatically captures conversations and intelligently recalls relevant context using Cursor's plugin architecture.

## Quick Start

### Option A — Hindsight Cloud (fastest)

No local server needed. [Sign up for Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) and create an API key under **Settings > API Keys**.

```bash
cd /path/to/your-project
pip install hindsight-cursor
hindsight-cursor init --api-url https://api.hindsight.vectorize.io --api-token YOUR_HINDSIGHT_API_TOKEN
```

> If Cursor is already open, **fully quit and reopen it** after installing. Plugins load at startup.

### Option B — Local Hindsight server

Start Hindsight locally with Docker:

```bash
export OPENAI_API_KEY=your-key
docker run --rm -it --pull always -p 8888:8888 \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  -e HINDSIGHT_API_LLM_MODEL=gpt-4o-mini \
  -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

Then install the plugin:

```bash
cd /path/to/your-project
pip install hindsight-cursor
hindsight-cursor init --api-url http://localhost:8888
```

> You can also use `uvx hindsight-cursor init` instead of `pip install` + `hindsight-cursor init` if you prefer not to install the package permanently.

### What `init` does

- Copies plugin files into `.cursor-plugin/hindsight-memory/`
- Creates `~/.hindsight/cursor.json` with your connection settings (if the file does not already exist)
- Use `--force` to overwrite an existing installation

After installing, **fully quit and reopen Cursor**. The plugin activates automatically.

## Features

- **Auto-recall** — on every user prompt, queries Hindsight for relevant memories and injects them as context via `additionalContext`
- **Auto-retain** — after every response, extracts and retains conversation content to Hindsight for long-term storage
- **On-demand recall** — use the `hindsight-recall` skill to manually query memories
- **Daemon management** — can auto-start/stop `hindsight-embed` locally or connect to an external Hindsight server
- **Dynamic bank IDs** — supports per-agent, per-project, or per-session memory isolation
- **Zero runtime dependencies** — plugin scripts use pure Python stdlib only

## Architecture

The plugin uses Cursor's hook system:

| Hook | Event | Purpose |
|------|-------|---------|
| `recall.py` | `beforeSubmitPrompt` | **Auto-recall** — query memories, inject as `additionalContext` |
| `retain.py` | `stop` | **Auto-retain** — extract transcript, POST to Hindsight |

### Library Modules

| Module | Purpose |
|--------|---------|
| `lib/client.py` | Hindsight REST API client (stdlib `urllib`) |
| `lib/config.py` | Configuration loader (settings.json + env overrides) |
| `lib/daemon.py` | `hindsight-embed` daemon lifecycle (start/stop/health) |
| `lib/bank.py` | Bank ID derivation + mission management |
| `lib/content.py` | Content processing (transcript parsing, memory formatting, tag stripping) |
| `lib/state.py` | File-based state persistence with `fcntl` locking |
| `lib/llm.py` | LLM provider auto-detection for daemon mode |

### How Recall Works

1. User sends a prompt -> `beforeSubmitPrompt` hook fires
2. Plugin resolves Hindsight API URL (external, local, or auto-start daemon)
3. Derives bank ID (static or dynamic from project context)
4. Composes query from current prompt + optional prior turns
5. Calls Hindsight recall API
6. Formats memories into `<hindsight_memories>` block
7. Outputs via `hookSpecificOutput.additionalContext` — the agent sees it, user doesn't

### How Retain Works

1. Agent completes a task -> `stop` hook fires
2. Reads conversation transcript from Cursor's JSONL file
3. Applies chunked retention logic (every N turns with sliding window)
4. Strips `<hindsight_memories>` tags to prevent feedback loops
5. POSTs formatted transcript to Hindsight retain API

## Connection Modes

### 1. External API (recommended for production)

Connect to a running Hindsight server (cloud or self-hosted).

```json
{
  "hindsightApiUrl": "https://your-hindsight-server.com",
  "hindsightApiToken": "your-token"
}
```

### 2. Local Daemon (auto-managed)

The plugin automatically starts and stops `hindsight-embed` via `uvx`. Requires an LLM provider API key.

```json
{
  "hindsightApiUrl": "",
  "apiPort": 9077
}
```

### 3. Existing Local Server

If you already have `hindsight-embed` running, leave `hindsightApiUrl` empty and set `apiPort` to match your server's port.

## Configuration

All settings live in `~/.hindsight/cursor.json`. Every setting can also be overridden via environment variables. The plugin ships with sensible defaults.

**Loading order** (later entries win):
1. Built-in defaults (hardcoded in the plugin)
2. Plugin `settings.json` (ships with the plugin)
3. User config (`~/.hindsight/cursor.json`)
4. Environment variables

### Connection & Daemon

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `hindsightApiUrl` | `HINDSIGHT_API_URL` | `""` | URL of an external Hindsight API server |
| `hindsightApiToken` | `HINDSIGHT_API_TOKEN` | `null` | Authentication token for the external API |
| `apiPort` | `HINDSIGHT_API_PORT` | `9077` | Port for the local `hindsight-embed` daemon |
| `daemonIdleTimeout` | `HINDSIGHT_DAEMON_IDLE_TIMEOUT` | `300` | Seconds before idle daemon shuts down (0 = never) |
| `embedVersion` | `HINDSIGHT_EMBED_VERSION` | `"latest"` | Version of `hindsight-embed` to install |
| `embedPackagePath` | `HINDSIGHT_EMBED_PACKAGE_PATH` | `null` | Local path to `hindsight-embed` package (dev override) |

### Memory Bank

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `bankId` | `HINDSIGHT_BANK_ID` | `"cursor"` | Bank ID when `dynamicBankId` is false |
| `dynamicBankId` | `HINDSIGHT_DYNAMIC_BANK_ID` | `false` | Derive bank ID from context fields |
| `dynamicBankGranularity` | — | `["agent", "project"]` | Fields used to derive dynamic bank ID (agent, project, session) |
| `bankIdPrefix` | — | `""` | Prefix prepended to all bank IDs |
| `agentName` | `HINDSIGHT_AGENT_NAME` | `"cursor"` | Agent name for dynamic bank ID |
| `bankMission` | `HINDSIGHT_BANK_MISSION` | `""` | Mission statement set on the bank (first use only) |
| `retainMission` | — | `null` | Custom retain mission for the bank |

### Auto-Recall

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `autoRecall` | `HINDSIGHT_AUTO_RECALL` | `true` | Enable/disable auto-recall |
| `recallBudget` | `HINDSIGHT_RECALL_BUDGET` | `"mid"` | Search thoroughness: low, mid, high |
| `recallMaxTokens` | `HINDSIGHT_RECALL_MAX_TOKENS` | `1024` | Max tokens in recalled memory block |
| `recallTypes` | — | `["world", "experience"]` | Memory types to recall |
| `recallContextTurns` | `HINDSIGHT_RECALL_CONTEXT_TURNS` | `1` | Number of prior turns to include in recall query |
| `recallMaxQueryChars` | `HINDSIGHT_RECALL_MAX_QUERY_CHARS` | `800` | Max characters in the recall query |
| `recallPromptPreamble` | — | *(see settings.json)* | Text prepended to recalled memories |

### Auto-Retain

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `autoRetain` | `HINDSIGHT_AUTO_RETAIN` | `true` | Enable/disable auto-retain |
| `retainMode` | `HINDSIGHT_RETAIN_MODE` | `"full-session"` | Retention strategy: `full-session` or `chunked` |
| `retainEveryNTurns` | `HINDSIGHT_RETAIN_EVERY_N_TURNS` | `10` | Retain every N turns (1 = every turn) |
| `retainOverlapTurns` | — | `2` | Overlap turns between chunks (chunked mode only) |
| `retainToolCalls` | — | `false` | Include tool call messages in retained transcript |
| `retainContext` | `HINDSIGHT_RETAIN_CONTEXT` | `"cursor"` | Source label for retained memories |
| `retainTags` | — | `[]` | Tags applied to retained documents (supports `{session_id}` template) |
| `retainMetadata` | — | `{}` | Extra metadata on retained documents |

### LLM (Daemon Mode)

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `llmProvider` | `HINDSIGHT_LLM_PROVIDER` | `null` | LLM provider override for daemon mode |
| `llmModel` | `HINDSIGHT_LLM_MODEL` | `null` | LLM model override for daemon mode |
| `llmApiKeyEnv` | — | `null` | Environment variable name containing the LLM API key |

### Debug

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `debug` | `HINDSIGHT_DEBUG` | `false` | Enable verbose logging to stderr |

## Alternative: MCP Integration

Cursor also supports MCP servers natively. If you prefer MCP over the plugin system, you can connect directly to Hindsight's MCP endpoint:

```json
// .cursor/mcp.json
{
  "mcpServers": {
    "hindsight": {
      "url": "http://localhost:8888/mcp/"
    }
  }
}
```

This gives you access to all Hindsight tools (retain, recall, reflect) without the plugin.

## Development

```bash
# Run tests
pip install pytest
python -m pytest tests/ -v
```

## Links

- [Hindsight Documentation](https://vectorize.io/hindsight)
- [Cursor Documentation](https://docs.cursor.com)
- [GitHub Repository](https://github.com/vectorize-io/hindsight)

## License

MIT
