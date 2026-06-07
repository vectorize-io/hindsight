# Hindsight Memory Plugin for Cursor

Biomimetic long-term memory for [Cursor](https://cursor.com) using [Hindsight](https://vectorize.io/hindsight). Automatically recalls relevant context at session start and retains conversation transcripts for future use.

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
- Writes `.cursor/mcp.json` with the Hindsight MCP endpoint for on-demand recall/retain/reflect tools
- Use `--force` to overwrite an existing installation
- Use `--no-mcp` to skip the MCP configuration

After installing, **fully quit and reopen Cursor**. The plugin activates automatically.

## Features

- **Session recall** — at the start of each session, queries Hindsight for relevant project memories and injects them as context via `additionalContext`
- **Auto-retain** — after every task, extracts and retains conversation content to Hindsight for long-term storage
- **On-demand MCP tools** — use `recall`, `retain`, and `reflect` tools from the Hindsight MCP server for explicit memory operations during a session
- **On-demand recall skill** — use the `hindsight-recall` skill for explicit memory lookups
- **Daemon management** — can auto-start/stop `hindsight-embed` locally or connect to an external Hindsight server
- **Dynamic bank IDs** — supports per-agent, per-project, or per-session memory isolation
- **Zero runtime dependencies** — plugin scripts use pure Python stdlib only

## Architecture

The plugin uses two complementary mechanisms:

### 1. Plugin Hooks (automatic)

| Hook | Event | Purpose |
|------|-------|---------|
| `session_start.py` | `sessionStart` | **Session recall** — query memories, write `<workspace>/.cursor/rules/hindsight-session.mdc` so the agent gets memory in its system context, *and* emit `additionalContext` JSON (forward-compat). |
| `retain.py` | `stop` | **Auto-retain** — extract transcript, POST to Hindsight |

The `sessionStart` hook fires when the agent processes the first prompt of each new chat. It performs a broad project-level recall and surfaces the result two ways — see ["How session memory reaches the agent"](#how-session-memory-reaches-the-agent) below for why both.

The `stop` hook fires when the agent completes a task. It reads the conversation transcript and retains it to Hindsight for future recall.

### How session memory reaches the agent

Cursor's native injection channel for session-start hooks is the `additionalContext` JSON field — the hook returns memory text on stdout and Cursor places it in the agent's system prompt. **That channel has been broken in Cursor 3.x** (acknowledged by Cursor staff in [forum thread 158452](https://forum.cursor.com/t/sessionstart-hook-additional-context-is-never-injected-into-agents-initial-system-context/158452); reconfirmed still-open against Cursor 3.6.31). When `additionalContext` is the only delivery path, recalled memories never reach the model — the agent answers as if Hindsight isn't installed.

This plugin works around the bug by **also** writing the recalled memories to `<workspace>/.cursor/rules/hindsight-session.mdc` with `alwaysApply: true` in the frontmatter. Workspace rules files *are* reliably injected by Cursor's rules engine, so the agent sees the memories on the very first prompt of each new chat.

What this means in practice:

- **Every new agent's first prompt has memories.** Cursor blocks prompt submission until the `sessionStart` hook returns — verified empirically. The only delay is the recall latency itself (typically <1s).
- **The rules file is regenerated at the top of every `sessionStart`.** Stale memories from a previous session never linger.
- **The rules file is auto-`.gitignore`'d** in git workspaces. It's safe to delete by hand; it'll be regenerated.
- **`additionalContext` is still emitted to stdout** for forward-compat. If Cursor restores the native channel, the same plugin keeps working without code changes.

You can disable the rules-file write entirely (`useRulesFileFallback: false`) — then the plugin relies on `additionalContext`, which means no memory delivery until Cursor fixes the upstream bug. Useful only if you'd rather see the bug bite than have the plugin touch your workspace.

### 2. MCP Server (on-demand)

The `init` command also configures Cursor's native MCP support (`.cursor/mcp.json`) to connect directly to Hindsight's MCP endpoint. This gives the agent access to explicit tools:

- **recall** — search for specific memories by query
- **retain** — store specific content to memory
- **reflect** — reason over accumulated memories with a question

The agent can use these tools mid-session when it needs memory beyond what was injected at session start.

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

### Session Recall

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `autoRecall` | `HINDSIGHT_AUTO_RECALL` | `true` | Enable/disable session-start recall |
| `recallBudget` | `HINDSIGHT_RECALL_BUDGET` | `"mid"` | Search thoroughness: low, mid, high |
| `recallMaxTokens` | `HINDSIGHT_RECALL_MAX_TOKENS` | `1024` | Max tokens in recalled memory block |
| `recallTypes` | — | `["world", "experience"]` | Memory types to recall |
| `recallMaxQueryChars` | `HINDSIGHT_RECALL_MAX_QUERY_CHARS` | `800` | Max characters in the recall query |
| `recallPromptPreamble` | — | *(see settings.json)* | Text prepended to recalled memories |
| `useRulesFileFallback` | `HINDSIGHT_USE_RULES_FILE_FALLBACK` | `true` | Write recalled memories to `<workspace>/.cursor/rules/hindsight-session.mdc` so Cursor's rules engine injects them. Workaround for [the broken native `additionalContext` channel](#how-session-memory-reaches-the-agent). |
| `appendToGitignore` | `HINDSIGHT_APPEND_TO_GITIGNORE` | `true` | When writing the rules-file fallback, idempotently append its path to the workspace `.gitignore` (no-op for non-git workspaces). |

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

