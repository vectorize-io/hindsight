# Hindsight Memory Plugin for Claude Code

Biomimetic long-term memory for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) using [Hindsight](https://vectorize.io/hindsight). Automatically captures conversations and intelligently recalls relevant context — a complete port of [`hindsight-openclaw`](../openclaw/) adapted to Claude Code's hook-based plugin architecture.

## Quick Start

```bash
# 1. Configure your LLM provider for memory extraction
# Option A: OpenAI (auto-detected)
export OPENAI_API_KEY="sk-your-key"

# Option B: Anthropic (auto-detected)
export ANTHROPIC_API_KEY="your-key"

# Option C: Connect to an external Hindsight server (no local LLM needed)
# Edit settings.json: set "hindsightApiUrl": "https://your-hindsight-server.com"

# 2. Install the plugin
claude /plugin install /path/to/hindsight-integrations/claude-code

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

1. User sends a prompt → `UserPromptSubmit` hook fires
2. Plugin resolves Hindsight API URL (external, local, or auto-start daemon)
3. Derives bank ID (static or dynamic from project context)
4. Composes query from current prompt + optional prior turns
5. Calls Hindsight recall API
6. Formats memories into `<hindsight_memories>` block
7. Outputs via `hookSpecificOutput.additionalContext` — Claude sees it, user doesn't

### How Retain Works

1. Claude responds → `Stop` hook fires (async, non-blocking)
2. Reads conversation transcript from Claude Code's JSONL file
3. Applies chunked retention logic (every N turns with sliding window)
4. Strips `<hindsight_memories>` tags to prevent feedback loops
5. Extracts text from channel messages (Telegram reply tool calls, etc.)
6. POSTs formatted transcript to Hindsight retain API

## Connection Modes

The plugin supports three connection modes, matching the Openclaw plugin:

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

```json
{
  "hindsightApiUrl": "",
  "apiPort": 9077
}
```

Set an LLM provider:
```bash
export OPENAI_API_KEY="sk-your-key"      # Auto-detected, uses gpt-4o-mini
# or
export ANTHROPIC_API_KEY="your-key"       # Auto-detected, uses claude-3-5-haiku
```

### 3. Existing Local Server

If you already have `hindsight-embed` running, leave `hindsightApiUrl` empty and set `apiPort` to match your server's port. The plugin will detect it automatically.

## Configuration

All settings are in `settings.json` at the plugin root. Every setting can also be overridden via environment variables.

### Connection & Daemon

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `hindsightApiUrl` | `""` | `HINDSIGHT_API_URL` | External Hindsight API URL. Empty = use local daemon. |
| `hindsightApiToken` | `null` | `HINDSIGHT_API_TOKEN` | Auth token for external API |
| `apiPort` | `9077` | `HINDSIGHT_API_PORT` | Port for local Hindsight daemon |
| `daemonIdleTimeout` | `0` | `HINDSIGHT_DAEMON_IDLE_TIMEOUT` | Seconds before idle daemon shuts down (0 = never) |
| `embedVersion` | `"latest"` | `HINDSIGHT_EMBED_VERSION` | `hindsight-embed` version for `uvx` |
| `embedPackagePath` | `null` | `HINDSIGHT_EMBED_PACKAGE_PATH` | Local path to `hindsight-embed` for development |

### LLM Provider (daemon mode only)

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `llmProvider` | auto-detect | `HINDSIGHT_LLM_PROVIDER` | LLM provider: `openai`, `anthropic`, `gemini`, `groq`, `ollama`, `openai-codex`, `claude-code` |
| `llmModel` | provider default | `HINDSIGHT_LLM_MODEL` | Model override |
| `llmApiKeyEnv` | provider standard | — | Custom env var name for API key |

Auto-detection checks these env vars in order: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`.

### Memory Bank

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `bankId` | `"claude_code"` | `HINDSIGHT_BANK_ID` | Static bank ID (when `dynamicBankId` is false) |
| `bankMission` | generic assistant | `HINDSIGHT_BANK_MISSION` | Agent identity/purpose for the memory bank |
| `retainMission` | extraction prompt | — | Custom retain mission (what to extract from conversations) |
| `dynamicBankId` | `false` | `HINDSIGHT_DYNAMIC_BANK_ID` | Enable per-context memory banks |
| `dynamicBankGranularity` | `["agent", "project"]` | — | Fields for dynamic bank ID: `agent`, `project`, `session`, `channel`, `user` |
| `bankIdPrefix` | `""` | — | Prefix for all bank IDs (e.g. `"prod"`) |
| `agentName` | `""` | `HINDSIGHT_AGENT_NAME` | Agent name for dynamic bank ID derivation |

### Auto-Recall

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `autoRecall` | `true` | `HINDSIGHT_AUTO_RECALL` | Enable automatic memory recall |
| `recallBudget` | `"mid"` | `HINDSIGHT_RECALL_BUDGET` | Recall effort: `low`, `mid`, `high` |
| `recallMaxTokens` | `1024` | `HINDSIGHT_RECALL_MAX_TOKENS` | Max tokens in recall response |
| `recallTypes` | `["world", "experience"]` | — | Memory types: `world`, `experience`, `observation` |
| `recallContextTurns` | `1` | `HINDSIGHT_RECALL_CONTEXT_TURNS` | Prior turns for query composition (1 = latest only) |
| `recallMaxQueryChars` | `800` | `HINDSIGHT_RECALL_MAX_QUERY_CHARS` | Max query length |
| `recallRoles` | `["user", "assistant"]` | — | Roles included in query context |
| `recallTopK` | `null` | — | Hard cap on memories per turn |
| `recallPromptPreamble` | built-in string | — | Text placed above recalled memories |

### Auto-Retain

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `autoRetain` | `true` | `HINDSIGHT_AUTO_RETAIN` | Enable automatic retention |
| `retainRoles` | `["user", "assistant"]` | — | Which roles to retain |
| `retainEveryNTurns` | `10` | — | Retain every Nth turn. Values >1 enable chunked retention with a sliding window. |
| `retainOverlapTurns` | `2` | — | Extra overlap turns included when chunked retention fires. Window = `retainEveryNTurns + retainOverlapTurns` (default: 12 turns). |
| `retainContext` | `"claude-code"` | — | Context label for retained memories |

### Miscellaneous

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `debug` | `false` | `HINDSIGHT_DEBUG` | Enable debug logging to stderr |

## Claude Code Channels

With [Claude Code Channels](https://docs.anthropic.com/en/docs/claude-code), Claude Code can operate as a persistent background agent connected to Telegram, Discord, Slack, and other messaging platforms. This plugin gives Channel-based agents the same long-term memory that `hindsight-openclaw` provides for Openclaw agents.

For Channel agents, set these environment variables in your Channel configuration:

```bash
# Per-channel/per-user memory isolation
export HINDSIGHT_CHANNEL_ID="telegram-group-12345"
export HINDSIGHT_USER_ID="user-67890"
```

And enable dynamic bank IDs:

```json
{
  "dynamicBankId": true,
  "dynamicBankGranularity": ["agent", "channel", "user"]
}
```

## Troubleshooting

### Plugin not activating

- Verify installation: check that `.claude-plugin/plugin.json` exists in the installed plugin directory
- Check Claude Code logs for `[Hindsight]` messages (enable `"debug": true` in settings.json)

### Recall returning no memories

- Verify the Hindsight server is reachable: `curl http://localhost:9077/health`
- Check that the bank has retained content: memories need at least one retain cycle
- Try increasing `recallBudget` to `"mid"` or `"high"`

### Daemon not starting

- Ensure `uvx` is installed: `pip install uv` or `brew install uv`
- Check that an LLM API key is set (required for local daemon)
- Review daemon logs: `tail -f ~/.hindsight/profiles/claude-code.log`
- Try starting manually: `uvx hindsight-embed@latest daemon --profile claude-code start`

### High latency on recall

- The recall hook has a 12-second timeout. If Hindsight is slow:
  - Use `recallBudget: "low"` (fewer retrieval strategies)
  - Reduce `recallMaxTokens`
  - Consider using an external API with a faster server

### State file issues

- State is stored in `$CLAUDE_PLUGIN_DATA/state/`
- To reset: delete the `state/` directory
- Turn counts, bank missions, and daemon state are tracked here

## Development

To test local changes to `hindsight-embed`:

```json
{
  "embedPackagePath": "/path/to/hindsight-embed"
}
```

The plugin will use `uv run --directory <path> hindsight-embed` instead of `uvx hindsight-embed@latest`.

To view daemon logs:

```bash
# Check daemon status
uvx hindsight-embed@latest daemon --profile claude-code status

# View logs
tail -f ~/.hindsight/profiles/claude-code.log

# List profiles
uvx hindsight-embed@latest profile list
```

## Links

- [Hindsight Documentation](https://vectorize.io/hindsight)
- [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
- [GitHub Repository](https://github.com/vectorize-io/hindsight)

## License

MIT
