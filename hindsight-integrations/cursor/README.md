# Hindsight Memory Plugin for Cursor

Biomimetic long-term memory for [Cursor](https://cursor.com) using [Hindsight](https://vectorize.io/hindsight). Automatically captures conversations and intelligently recalls relevant context using Cursor's plugin architecture.

## Quick Start

### 1. Install the plugin

```bash
mkdir -p /path/to/your-project/.cursor-plugin
cp -r hindsight-integrations/cursor /path/to/your-project/.cursor-plugin/hindsight-memory
```

> If Cursor is already open, **fully quit and reopen it** after adding the plugin. Plugins load at startup.

### 2. Configure Hindsight

Create `~/.hindsight/cursor.json`:

**Option A — Hindsight Cloud** (no local server needed):

```json
{
  "hindsightApiUrl": "https://api.hindsight.vectorize.io",
  "hindsightApiToken": "YOUR_HINDSIGHT_API_TOKEN",
  "bankId": "cursor"
}
```

Sign up at [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) to get a token.

**Option B — Local server:**

```json
{
  "hindsightApiUrl": "http://localhost:8888",
  "bankId": "cursor"
}
```

**Option C — Auto-managed daemon** (requires an LLM API key):

```bash
export OPENAI_API_KEY="sk-your-key"
# or: export ANTHROPIC_API_KEY="your-key"
```

Leave `hindsightApiUrl` empty and the plugin will auto-start `hindsight-embed` locally.

### 3. Open Cursor

Open the target project in Cursor. The plugin activates automatically.

## Features

- **Auto-recall** — on every user prompt, queries Hindsight for relevant memories and injects them as context via `additionalContext`
- **Auto-retain** — after every response, extracts and retains conversation content to Hindsight for long-term storage
- **On-demand recall** — use the `hindsight-recall` skill to manually query memories
- **Daemon management** — can auto-start/stop `hindsight-embed` locally or connect to an external Hindsight server
- **Dynamic bank IDs** — supports per-agent, per-project, or per-session memory isolation
- **Zero dependencies** — pure Python stdlib, no pip install required

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
| `embedVersion` | `HINDSIGHT_EMBED_VERSION` | `"latest"` | Version of `hindsight-embed` to install |

### Memory Bank

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `bankId` | `HINDSIGHT_BANK_ID` | `"cursor"` | Bank ID when `dynamicBankId` is false |
| `dynamicBankId` | `HINDSIGHT_DYNAMIC_BANK_ID` | `false` | Derive bank ID from context fields |
| `agentName` | `HINDSIGHT_AGENT_NAME` | `"cursor"` | Agent name for dynamic bank ID |

### Auto-Recall

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `autoRecall` | `HINDSIGHT_AUTO_RECALL` | `true` | Enable/disable auto-recall |
| `recallBudget` | `HINDSIGHT_RECALL_BUDGET` | `"mid"` | Search thoroughness: low, mid, high |
| `recallMaxTokens` | `HINDSIGHT_RECALL_MAX_TOKENS` | `1024` | Max tokens in recalled memory block |

### Auto-Retain

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `autoRetain` | `HINDSIGHT_AUTO_RETAIN` | `true` | Enable/disable auto-retain |
| `retainEveryNTurns` | `HINDSIGHT_RETAIN_EVERY_N_TURNS` | `10` | Retain frequency |
| `retainContext` | `HINDSIGHT_RETAIN_CONTEXT` | `"cursor"` | Source label for retained memories |

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
