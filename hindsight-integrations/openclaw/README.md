# Hindsight Memory Plugin for OpenClaw

Biomimetic long-term memory for [OpenClaw](https://openclaw.ai) using [Hindsight](https://vectorize.io/hindsight). Automatically captures conversations and intelligently recalls relevant context.

## Quick Start

```bash
# 1. Install the plugin
openclaw plugins install @vectorize-io/hindsight-openclaw

# 2. Run the interactive setup wizard
npx --package @vectorize-io/hindsight-openclaw hindsight-openclaw-setup

# 3. Start OpenClaw
openclaw gateway
```

`hindsight-openclaw-setup` walks you through picking one of three modes:

- **Cloud** — managed Hindsight. Paste your cloud API token, done.
- **External API** — your own running Hindsight deployment. Prompts for the URL and optional token.
- **Embedded daemon** — spawns a local `hindsight-embed` daemon on this machine. Prompts for the LLM provider (OpenAI / Anthropic / Gemini / Groq / Claude Code / Codex / Ollama) and its API key.

The interactive wizard stores credentials **inline** in `openclaw.json` for simplicity — the value is masked as you paste it. For CI / production you can store credentials as a `SecretRef` (resolved from an env var, file, or exec source at startup) by using the non-interactive flags with `--token-env` / `--api-key-env`, or by switching an existing field afterwards with `openclaw config set ... --ref-source env --ref-id …`.

### Manual configuration (without the wizard)

The wizard is a convenience wrapper — all of the same fields can be set directly with `openclaw config set`:

```bash
# Embedded daemon with OpenAI
openclaw config set plugins.entries.hindsight-openclaw.config.llmProvider openai
openclaw config set plugins.entries.hindsight-openclaw.config.llmApiKey \
    --ref-source env --ref-provider default --ref-id OPENAI_API_KEY

# Or: Claude Code (no API key needed)
openclaw config set plugins.entries.hindsight-openclaw.config.llmProvider claude-code

# Or: use OpenClaw's existing credentials (no separate API key needed)
openclaw config set plugins.entries.hindsight-openclaw.config.llmProvider anthropic
openclaw config set plugins.entries.hindsight-openclaw.config.llmAuthSource openclaw

# Or: point at an external Hindsight API
openclaw config set plugins.entries.hindsight-openclaw.config.hindsightApiUrl https://mcp.hindsight.example.com
openclaw config set plugins.entries.hindsight-openclaw.config.hindsightApiToken \
    --ref-source env --ref-id HINDSIGHT_API_TOKEN
```

### Using OpenClaw credentials

If you already have LLM providers authenticated in OpenClaw (via `openclaw models auth login` or setup), Hindsight can reuse those credentials directly — no separate API keys or CLI installations required.

```bash
openclaw config set plugins.entries.hindsight-openclaw.config.llmProvider anthropic
openclaw config set plugins.entries.hindsight-openclaw.config.llmAuthSource openclaw
openclaw gateway restart
```

The plugin reads credentials from OpenClaw's `auth-profiles.json` and passes them to the Hindsight daemon. Supported providers and their OpenClaw profile types:

| Hindsight provider | OpenClaw profile type | Notes                                          |
| ------------------ | --------------------- | ---------------------------------------------- |
| `openai`           | `api_key`             | Standard API key                               |
| `anthropic`        | `token` or `api_key`  | OAuth tokens (OAT) and standard keys both work |
| `openai-codex`     | `oauth`               | Uses ChatGPT Plus/Pro subscription             |
| `groq`             | `api_key`             | Standard API key                               |
| `deepseek`         | `api_key`             | Standard API key                               |
| `gemini`           | `api_key`             | Mapped from OpenClaw's `google` provider       |

On 401/403 errors, the daemon automatically re-reads the auth-profiles file to pick up tokens refreshed by OpenClaw's gateway.

## Migrating from 0.5.x

0.6.0 removes all process-environment reads from the plugin. Configuration that
previously came from shell env vars must now go through OpenClaw's plugin config
(with SecretRef for credentials). Concrete mappings:

| Old (0.5.x)                              | New (0.6.0)                                                                                                                                                                                                |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OPENAI_API_KEY=…` (auto-detected)       | `openclaw config set plugins.entries.hindsight-openclaw.config.llmProvider openai` <br> `openclaw config set plugins.entries.hindsight-openclaw.config.llmApiKey --ref-source env --ref-id OPENAI_API_KEY` |
| `HINDSIGHT_API_LLM_PROVIDER=…`           | `openclaw config set plugins.entries.hindsight-openclaw.config.llmProvider …`                                                                                                                              |
| `HINDSIGHT_API_LLM_MODEL=…`              | `openclaw config set plugins.entries.hindsight-openclaw.config.llmModel …`                                                                                                                                 |
| `HINDSIGHT_API_LLM_API_KEY=…`            | `openclaw config set plugins.entries.hindsight-openclaw.config.llmApiKey --ref-source env --ref-id …`                                                                                                      |
| `HINDSIGHT_API_LLM_BASE_URL=…`           | `openclaw config set plugins.entries.hindsight-openclaw.config.llmBaseUrl …`                                                                                                                               |
| `HINDSIGHT_EMBED_API_URL=…`              | `openclaw config set plugins.entries.hindsight-openclaw.config.hindsightApiUrl …`                                                                                                                          |
| `HINDSIGHT_EMBED_API_TOKEN=…`            | `openclaw config set plugins.entries.hindsight-openclaw.config.hindsightApiToken --ref-source env --ref-id …`                                                                                              |
| `HINDSIGHT_BANK_ID=…`                    | `openclaw config set plugins.entries.hindsight-openclaw.config.bankId …`                                                                                                                                   |
| `llmApiKeyEnv: "MY_KEY"` (plugin config) | `llmApiKey` configured as a SecretRef with `--ref-id MY_KEY`                                                                                                                                               |

If your shell already exports `OPENAI_API_KEY`, the SecretRef config above resolves
to the same value at startup — no need to change your shell setup, just point the
plugin at the variable explicitly. Run `openclaw config validate` after migrating
to confirm the new shape parses cleanly.

## Features

- **Auto-capture** and **auto-recall** of memories each turn, injected into system prompt space so recalled memories stay out of the visible chat transcript
- **Memory isolation** — configurable per agent, channel, user, or provider via `dynamicBankGranularity`
- **Historical backfill CLI** — import prior OpenClaw session history into Hindsight using the active plugin bank-routing config by default
- **Retention controls** — choose which message roles to retain, toggle auto-retain on/off, and stamp retained documents with consistent tags/source metadata

## Configuration

Optional settings in `~/.openclaw/openclaw.json` under `plugins.entries.hindsight-openclaw.config`:

| Option                     | Default                        | Description                                                                                                                                                                                                                                                                                                 |
| -------------------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `apiPort`                  | `9077`                         | Port for the local Hindsight daemon                                                                                                                                                                                                                                                                         |
| `daemonIdleTimeout`        | `0`                            | Seconds before daemon shuts down from inactivity (0 = never)                                                                                                                                                                                                                                                |
| `embedPort`                | `0`                            | Port for `hindsight-embed` server (`0` = auto-assign)                                                                                                                                                                                                                                                       |
| `embedVersion`             | `"latest"`                     | hindsight-embed version                                                                                                                                                                                                                                                                                     |
| `embedPackagePath`         | —                              | Local path to `hindsight-embed` package for development                                                                                                                                                                                                                                                     |
| `bankMission`              | —                              | Agent identity/purpose stored on the memory bank. Helps the engine understand context for better fact extraction. Set once per bank — not a recall prompt.                                                                                                                                                  |
| `llmAuthSource`            | `"env"`                        | Credential source. `"env"` reads API keys from plugin config / env vars (default). `"openclaw"` reads credentials from OpenClaw's auth-profiles, reusing existing provider authentication — no separate API key needed.                                                                                     |
| `llmProvider`              | —                              | LLM provider for memory extraction (`openai`, `anthropic`, `gemini`, `groq`, `ollama`, `openai-codex`, `claude-code`). Required unless `hindsightApiUrl` is set.                                                                                                                                            |
| `llmModel`                 | provider default               | LLM model used with `llmProvider`                                                                                                                                                                                                                                                                           |
| `llmApiKey`                | —                              | API key for the LLM provider. **Sensitive** — set via `openclaw config set ... --ref-source env --ref-id OPENAI_API_KEY` to reference an env var (or `--ref-source file`/`exec` for mounted-secret/Vault sources).                                                                                          |
| `llmBaseUrl`               | —                              | Optional base URL override for OpenAI-compatible providers (e.g. `https://openrouter.ai/api/v1`)                                                                                                                                                                                                            |
| `dynamicBankId`            | `true`                         | Enable per-context memory banks                                                                                                                                                                                                                                                                             |
| `bankId`                   | —                              | Static bank ID used when `dynamicBankId` is `false`.                                                                                                                                                                                                                                                        |
| `bankIdPrefix`             | —                              | Prefix for bank IDs (e.g. `"prod"`)                                                                                                                                                                                                                                                                         |
| `retainTags`               | `[]`                           | Tags applied to every retained document, useful for cross-agent/source labeling (e.g. `source_system:openclaw`, `agent:agentname`). Auto-retain also merges inline per-message tags from `<retain_tags>...</retain_tags>` or `<hindsight_retain_tags>...</hindsight_retain_tags>` blocks in user messages.  |
| `retainSource`             | `"openclaw"`                   | `source` value written into retained document metadata                                                                                                                                                                                                                                                      |
| `dynamicBankGranularity`   | `["agent", "channel", "user"]` | Fields used to derive bank ID. Options: `agent`, `channel`, `user`, `provider`                                                                                                                                                                                                                              |
| `excludeProviders`         | `["heartbeat"]`                | Message providers to skip for recall/retain (e.g. `heartbeat`, `slack`, `telegram`, `discord`)                                                                                                                                                                                                              |
| `autoRecall`               | `true`                         | Auto-inject memories before each turn. Set to `false` when the agent has its own recall tool.                                                                                                                                                                                                               |
| `autoRetain`               | `true`                         | Auto-retain conversations after each turn                                                                                                                                                                                                                                                                   |
| `retainRoles`              | `["user", "assistant"]`        | Which message roles to retain. Options: `user`, `assistant`, `system`, `tool`                                                                                                                                                                                                                               |
| `retainFormat`             | `"json"`                       | Serialization format for retained conversation content. `"json"` emits a structured array of `{role, content}` messages (matches Claude Code). `"text"` emits legacy `[role: x] … [x:end]` markers.                                                                                                         |
| `retainToolCalls`          | `true`                         | With `retainFormat: "json"`, each message's content is an Anthropic-shaped block array (`text` / `tool_use` / `tool_result`). Tool results are truncated at 2000 chars. Hindsight's own MCP tools (recall/retain/search/…) are filtered to prevent feedback loops. Set `false` to retain text-only content. |
| `retainEveryNTurns`        | `1`                            | Retain every Nth turn. `1` = every turn (default). Values > 1 enable chunked retention with a sliding window.                                                                                                                                                                                               |
| `retainOverlapTurns`       | `0`                            | Extra prior turns included when chunked retention fires. Window = `retainEveryNTurns + retainOverlapTurns`. Only applies when `retainEveryNTurns > 1`.                                                                                                                                                      |
| `recallBudget`             | `"mid"`                        | Recall effort: `low`, `mid`, or `high`. Higher budgets use more retrieval strategies.                                                                                                                                                                                                                       |
| `recallMaxTokens`          | `1024`                         | Max tokens for recall response. Controls how much memory context is injected per turn.                                                                                                                                                                                                                      |
| `recallTypes`              | `["world", "experience"]`      | Memory types to recall. Options: `world`, `experience`, `observation`. Excludes verbose `observation` entries by default.                                                                                                                                                                                   |
| `recallRoles`              | `["user", "assistant"]`        | Roles included when building prior context for recall query composition. Options: `user`, `assistant`, `system`, `tool`.                                                                                                                                                                                    |
| `recallTopK`               | —                              | Max number of memories to inject per turn. Applied after API response as a hard cap.                                                                                                                                                                                                                        |
| `recallContextTurns`       | `1`                            | Number of user turns to include when composing recall query context. `1` keeps latest-message-only behavior.                                                                                                                                                                                                |
| `recallMaxQueryChars`      | `800`                          | Maximum character length for the composed recall query before calling recall.                                                                                                                                                                                                                               |
| `recallPromptPreamble`     | built-in string                | Prompt text placed above recalled memories in the injected `<hindsight_memories>` system-context block.                                                                                                                                                                                                     |
| `hindsightApiUrl`          | —                              | External Hindsight API URL (skips local daemon)                                                                                                                                                                                                                                                             |
| `hindsightApiToken`        | —                              | Auth token for external API. **Sensitive** — set via `openclaw config set ... --ref-source env --ref-id HINDSIGHT_API_TOKEN`.                                                                                                                                                                               |
| `ignoreSessionPatterns`    | `[]`                           | Session key glob patterns to skip entirely — no recall, no retain (e.g. `["agent:*:cron:**"]`)                                                                                                                                                                                                              |
| `statelessSessionPatterns` | `[]`                           | Session key glob patterns for read-only sessions — retain is always skipped; recall is skipped when `skipStatelessSessions` is `true` (e.g. `["agent:*:subagent:**", "agent:*:heartbeat:**"]`)                                                                                                              |
| `skipStatelessSessions`    | `true`                         | When `true`, sessions matching `statelessSessionPatterns` also skip recall. Set to `false` to allow recall but still skip retain.                                                                                                                                                                           |

### Session pattern filtering

`ignoreSessionPatterns` and `statelessSessionPatterns` accept glob patterns matched against the session key (format: `agent:<agentId>:<type>:<uuid>`).

Glob syntax:

- `*` — matches any characters except `:` (single segment)
- `**` — matches anything including `:` (multiple segments)

| Pattern               | Matches                             |
| --------------------- | ----------------------------------- |
| `agent:*:cron:**`     | All cron sessions for any agent     |
| `agent:*:subagent:**` | All subagent sessions for any agent |
| `agent:main:**`       | All sessions under the `main` agent |

**Difference between the two options:**

|        | `ignoreSessionPatterns` | `statelessSessionPatterns`                      |
| ------ | ----------------------- | ----------------------------------------------- |
| Retain | Skipped                 | Always skipped                                  |
| Recall | Skipped                 | Skipped only when `skipStatelessSessions: true` |

**Example config** — exclude cron jobs from memory entirely, allow subagents to read but not write memories:

```json
{
  "ignoreSessionPatterns": ["agent:*:cron:**"],
  "statelessSessionPatterns": ["agent:*:subagent:**"],
  "skipStatelessSessions": false
}
```

## Retention details

Retained documents use stable session-scoped IDs derived from the OpenClaw `sessionKey`. By default (`retainDocumentScope: 'session'`) every retain in a session shares one document id like `openclaw:agent:agentname:discord:channel:123`, so all turns of the conversation accumulate under a single Hindsight document. Set `retainDocumentScope: 'turn'` to fall back to the per-retain ids (`...:turn:000001`, `...:window:000002` for chunked retention). Either way, retained documents include richer metadata such as `session_key`, `agent_id`, `provider`, `channel_id`, `thread_id`, `sender_id`, `turn_index`, and `retention_scope`. Each message in the retained JSON also carries a structured `timestamp` field (ISO 8601) lifted from OpenClaw's per-message time, so facts are not polluted by inline weekday/date prefixes.

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
