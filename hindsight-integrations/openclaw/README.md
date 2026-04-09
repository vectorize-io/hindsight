# Hindsight Memory Plugin for OpenClaw

Biomimetic long-term memory for [OpenClaw](https://openclaw.ai) using [Hindsight](https://vectorize.io/hindsight). Automatically captures conversations and intelligently recalls relevant context.

## Quick Start

```bash
# 1. Configure your LLM provider for memory extraction
# Option A: OpenAI
export OPENAI_API_KEY="sk-your-key"

# Option B: Claude Code (no API key needed)
export HINDSIGHT_API_LLM_PROVIDER=claude-code

# Option C: OpenAI Codex (no API key needed)
export HINDSIGHT_API_LLM_PROVIDER=openai-codex

# 2. Install and enable the plugin
openclaw plugins install @vectorize-io/hindsight-openclaw

# 3. Start OpenClaw
openclaw gateway
```

That's it! The plugin will automatically start capturing and recalling memories.

## Features

- **Auto-capture** and **auto-recall** of memories each turn, injected into system prompt space so recalled memories stay out of the visible chat transcript
- **Memory isolation** — configurable per agent, channel, user, or provider via `dynamicBankGranularity`
- **Historical backfill CLI** — import prior OpenClaw session history into Hindsight using the active plugin bank-routing config by default
- **Retention controls** — choose which message roles to retain, toggle auto-retain on/off, and stamp retained documents with consistent tags/source metadata

## Configuration

Optional settings in `~/.openclaw/openclaw.json` under `plugins.entries.hindsight-openclaw.config`:

| Option | Default | Description |
|--------|---------|-------------|
| `apiPort` | `9077` | Port for the local Hindsight daemon |
| `daemonIdleTimeout` | `0` | Seconds before daemon shuts down from inactivity (0 = never) |
| `embedPort` | `0` | Port for `hindsight-embed` server (`0` = auto-assign) |
| `embedVersion` | `"latest"` | hindsight-embed version |
| `embedPackagePath` | — | Local path to `hindsight-embed` package for development |
| `bankMission` | — | Agent identity/purpose stored on the memory bank. Helps the engine understand context for better fact extraction. Set once per bank — not a recall prompt. |
| `llmProvider` | auto-detect | LLM provider override for memory extraction (`openai`, `anthropic`, `gemini`, `groq`, `ollama`, `openai-codex`, `claude-code`) |
| `llmModel` | provider default | LLM model override used with `llmProvider` |
| `llmApiKeyEnv` | provider standard env var | Custom env var name for the provider API key |
| `dynamicBankId` | `true` | Enable per-context memory banks |
| `bankId` | — | Static bank ID used when `dynamicBankId` is `false`. Can also be set with `HINDSIGHT_BANK_ID`. |
| `bankIdPrefix` | — | Prefix for bank IDs (e.g. `"prod"`) |
| `retainTags` | `[]` | Tags applied to every retained document, useful for cross-agent/source labeling (e.g. `source_system:openclaw`, `agent:agentname`) |
| `retainSource` | `"openclaw"` | `source` value written into retained document metadata |
| `dynamicBankGranularity` | `["agent", "channel", "user"]` | Fields used to derive bank ID. Options: `agent`, `channel`, `user`, `provider` |
| `excludeProviders` | `["heartbeat"]` | Message providers to skip for recall/retain (e.g. `heartbeat`, `slack`, `telegram`, `discord`) |
| `autoRecall` | `true` | Auto-inject memories before each turn. Set to `false` when the agent has its own recall tool. |
| `autoRetain` | `true` | Auto-retain conversations after each turn |
| `retainRoles` | `["user", "assistant"]` | Which message roles to retain. Options: `user`, `assistant`, `system`, `tool` |
| `retainEveryNTurns` | `1` | Retain every Nth turn. `1` = every turn (default). Values > 1 enable chunked retention with a sliding window. |
| `retainOverlapTurns` | `0` | Extra prior turns included when chunked retention fires. Window = `retainEveryNTurns + retainOverlapTurns`. Only applies when `retainEveryNTurns > 1`. |
| `recallBudget` | `"mid"` | Recall effort: `low`, `mid`, or `high`. Higher budgets use more retrieval strategies. |
| `recallMaxTokens` | `1024` | Max tokens for recall response. Controls how much memory context is injected per turn. |
| `recallTypes` | `["world", "experience"]` | Memory types to recall. Options: `world`, `experience`, `observation`. Excludes verbose `observation` entries by default. |
| `recallRoles` | `["user", "assistant"]` | Roles included when building prior context for recall query composition. Options: `user`, `assistant`, `system`, `tool`. |
| `recallTopK` | — | Max number of memories to inject per turn. Applied after API response as a hard cap. |
| `recallContextTurns` | `1` | Number of user turns to include when composing recall query context. `1` keeps latest-message-only behavior. |
| `recallMaxQueryChars` | `800` | Maximum character length for the composed recall query before calling recall. |
| `recallPromptPreamble` | built-in string | Prompt text placed above recalled memories in the injected `<hindsight_memories>` system-context block. |
| `hindsightApiUrl` | — | External Hindsight API URL (skips local daemon) |
| `hindsightApiToken` | — | Auth token for external API |
| `ignoreSessionPatterns` | `[]` | Session key glob patterns to skip entirely — no recall, no retain (e.g. `["agent:*:cron:**"]`) |
| `statelessSessionPatterns` | `[]` | Session key glob patterns for read-only sessions — retain is always skipped; recall is skipped when `skipStatelessSessions` is `true` (e.g. `["agent:*:subagent:**", "agent:*:heartbeat:**"]`) |
| `skipStatelessSessions` | `true` | When `true`, sessions matching `statelessSessionPatterns` also skip recall. Set to `false` to allow recall but still skip retain. |

### Session pattern filtering

`ignoreSessionPatterns` and `statelessSessionPatterns` accept glob patterns matched against the session key (format: `agent:<agentId>:<type>:<uuid>`).

Glob syntax:
- `*` — matches any characters except `:` (single segment)
- `**` — matches anything including `:` (multiple segments)

| Pattern | Matches |
|---|---|
| `agent:*:cron:**` | All cron sessions for any agent |
| `agent:*:subagent:**` | All subagent sessions for any agent |
| `agent:main:**` | All sessions under the `main` agent |

**Difference between the two options:**

| | `ignoreSessionPatterns` | `statelessSessionPatterns` |
|---|---|---|
| Retain | Skipped | Always skipped |
| Recall | Skipped | Skipped only when `skipStatelessSessions: true` |

**Example config** — exclude cron jobs from memory entirely, allow subagents to read but not write memories:

```json
{
  "ignoreSessionPatterns": ["agent:*:cron:**"],
  "statelessSessionPatterns": ["agent:*:subagent:**"],
  "skipStatelessSessions": false
}
```

## Retention details

Retained documents use stable session-scoped IDs like `openclaw:agent:agentname:discord:channel:123:turn:000001` (or `...:window:000002` for chunked retention), and include richer metadata such as `session_key`, `agent_id`, `provider`, `channel_id`, `thread_id`, `sender_id`, `turn_index`, and `retention_scope`.

## Documentation

For full documentation, configuration options, troubleshooting, and development guide, see:

**[OpenClaw Integration Documentation](https://vectorize.io/hindsight/sdks/integrations/openclaw)**

## Development

To test local changes to the Hindsight package before publishing:

1. Add `embedPackagePath` to your plugin config in `~/.openclaw/openclaw.json`:
```json
{
  "plugins": {
    "entries": {
      "hindsight-openclaw": {
        "enabled": true,
        "config": {
          "embedPackagePath": "/path/to/hindsight-wt3/hindsight-embed"
        }
      }
    }
  }
}
```

2. The plugin will use `uv run --directory <path> hindsight-embed` instead of `uvx hindsight-embed@latest`

3. To use a specific profile for testing:
```bash
# Check daemon status
uvx hindsight-embed@latest -p openclaw daemon status

# View logs
tail -f ~/.hindsight/profiles/openclaw.log

# List profiles
uvx hindsight-embed@latest profile list
```

## Backfilling Existing OpenClaw History

The package includes a config-aware backfill CLI for importing historical OpenClaw sessions into Hindsight.

By default it mirrors the active plugin settings for:

- `dynamicBankId`
- `dynamicBankGranularity`
- `bankIdPrefix`
- local daemon vs external `hindsightApiUrl`

Dry-run example:

```bash
npx --package @vectorize-io/hindsight-openclaw hindsight-openclaw-backfill \
  --openclaw-root ~/.openclaw \
  --dry-run
```

Direct invocation from a built checkout:

```bash
node dist/backfill.js --openclaw-root ~/.openclaw --dry-run
```

Migration-oriented overrides are explicit:

```bash
node dist/backfill.js \
  --openclaw-root ~/.openclaw \
  --bank-strategy agent \
  --agent proj-run \
  --resume \
  --max-pending-operations 10
```

Useful options:

- `--agent <id>` limit import to selected agents
- `--exclude-archive` ignore `sessions-archive-from-migration_backup`
- `--bank-strategy mirror-config|agent|fixed`
- `--resume` skip only entries already finalized as completed
- `--checkpoint <path>` store progress outside the default location
- `--wait-until-drained` block until the touched bank queues have finished and checkpoint state can be finalized

## Links

- [Hindsight Documentation](https://vectorize.io/hindsight)
- [OpenClaw Documentation](https://openclaw.ai)
- [GitHub Repository](https://github.com/vectorize-io/hindsight)

## License

MIT
