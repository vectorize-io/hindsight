---
sidebar_position: 4
title: "OpenClaw Persistent Memory with Hindsight | Plugin Integration"
description: "Add persistent, automated memory to your OpenClaw agent with Hindsight. Local-first, open source — one plugin install replaces built-in memory with structured knowledge extraction and auto-recall."
---

# OpenClaw

Local, long term memory for [OpenClaw](https://openclaw.ai) agents using [Hindsight](https://vectorize.io/hindsight).

This plugin integrates [hindsight-embed](https://vectorize.io/hindsight/cli), a standalone daemon that bundles Hindsight's memory engine (API + PostgreSQL) into a single command. Everything runs locally on your machine, reuses the LLM you're already paying for, and costs nothing extra.

[View Changelog →](/changelog/integrations/openclaw)

## Quick Start

**Step 1: Install the plugin**

```bash
openclaw plugins install @vectorize-io/hindsight-openclaw
```

**Step 2: Run the setup wizard**

`openclaw plugins install` unpacks the plugin into `~/.openclaw/extensions/`
but does not put its bins on `PATH`. Run the wizard through `npx` instead —
it resolves the bin out of the published package:

```bash
npx --package @vectorize-io/hindsight-openclaw hindsight-openclaw-setup
```

The wizard walks you through picking one of three install modes:

- **Cloud** — managed Hindsight at `https://api.hindsight.vectorize.io`. Paste your cloud API token when prompted (masked input). No local setup needed.
- **External API** — your own running Hindsight deployment. Prompts for the URL and, optionally, the token value (masked).
- **Embedded daemon** — spawns a local `hindsight-embed` daemon on this machine. Prompts for the LLM provider (OpenAI / Anthropic / Gemini / Groq / Claude Code / OpenAI Codex / Ollama) and the API key (masked).

The interactive wizard stores credentials **inline** in `openclaw.json` for simplicity. For CI / production you can store credentials as a [`SecretRef`](#llm-configuration) (resolved from an env var, file, or exec source at startup, never saved on disk) by either passing `--token-env` / `--api-key-env` to the non-interactive wizard or switching an existing field afterwards via `openclaw config set ... --ref-source env|file|exec`.

For CI and scripted setups the wizard also runs non-interactively — either with an inline value or with an env var reference:

```bash
# Cloud — inline token (simplest)
npx --package @vectorize-io/hindsight-openclaw hindsight-openclaw-setup \
    --mode cloud --token hsk_your_cloud_token

# Cloud — SecretRef (read from env at gateway startup)
npx --package @vectorize-io/hindsight-openclaw hindsight-openclaw-setup \
    --mode cloud --token-env HINDSIGHT_CLOUD_TOKEN

# External API (no auth)
npx --package @vectorize-io/hindsight-openclaw hindsight-openclaw-setup \
    --mode api --api-url https://mcp.hindsight.example.com --no-token

# Embedded daemon with OpenAI — inline API key
npx --package @vectorize-io/hindsight-openclaw hindsight-openclaw-setup \
    --mode embedded --provider openai --api-key sk-...

# Embedded daemon with OpenAI — SecretRef
npx --package @vectorize-io/hindsight-openclaw hindsight-openclaw-setup \
    --mode embedded --provider openai --api-key-env OPENAI_API_KEY

# Embedded daemon with Claude Code (no API key needed)
npx --package @vectorize-io/hindsight-openclaw hindsight-openclaw-setup \
    --mode embedded --provider claude-code
```

Run `npx --package @vectorize-io/hindsight-openclaw hindsight-openclaw-setup --help` for the full flag list.

**Step 3: Start OpenClaw**

```bash
openclaw gateway
```

The plugin will automatically capture conversations after each turn and inject relevant memories before agent responses.

**Important:** The LLM you configure above is **only for memory extraction** (background processing). Your main OpenClaw agent can use any model you configure separately.

**Migrating from 0.5.x?** See the [Migration from 0.5.x](#migration-from-05x) section below for the env-var → SecretRef mapping.

## How It Works

**Auto-Capture:** Every conversation is automatically stored after each turn. Facts, entities, and relationships are extracted in the background.

**Auto-Recall:** Before each agent response, relevant memories are automatically injected into the context (up to 1024 tokens). The agent uses past context without needing to call tools.

**Feedback Loop Prevention:** The plugin automatically strips injected memory tags (`<hindsight_memories>`) before storing conversations. This prevents recalled memories from being re-extracted as new facts, which would cause exponential memory growth and duplicate entries.

Traditional memory systems give agents a `search_memory` tool - but models don't use it consistently. Auto-recall solves this by injecting memories automatically before every turn.

## Configuration

### Plugin Settings

Optional settings in `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "hindsight-openclaw": {
        "enabled": true,
        "config": {
          "apiPort": 9077,
          "daemonIdleTimeout": 0,
          "embedVersion": "latest"
        }
      }
    }
  }
}
```

**Options:**
- `apiPort` - Port for the openclaw profile daemon (default: `9077`)
- `daemonIdleTimeout` - Seconds before daemon shuts down from inactivity (default: `0` = never)
- `embedVersion` - hindsight-embed version (default: `"latest"`)
- `llmProvider` - LLM provider for memory extraction (`openai`, `anthropic`, `gemini`, `groq`, `ollama`, `openai-codex`, `claude-code`). Required unless `hindsightApiUrl` is set.
- `llmModel` - LLM model used with `llmProvider` (provider default if omitted)
- `llmApiKey` - API key for the LLM provider. **Sensitive** — set via `openclaw config set ... --ref-source env --ref-id OPENAI_API_KEY` to reference an env var.
- `llmBaseUrl` - Optional base URL override for OpenAI-compatible providers (e.g. `https://openrouter.ai/api/v1`)
- `bankMission` - Agent identity/purpose stored on the memory bank. Helps the memory engine understand context for better fact extraction during retain. Set once per bank on first use — not a recall prompt.
- `dynamicBankId` - Enable per-context memory banks (default: `true`)
- `bankId` - Static bank ID used when `dynamicBankId` is `false`.
- `bankIdPrefix` - Optional prefix for bank IDs (e.g. `"prod"` → `"prod-slack-C123"` or `"prod-shared-bank"`)
- `dynamicBankGranularity` - Fields used to derive bank ID: `agent`, `channel`, `user`, `provider` (default: `["agent", "channel", "user"]`)
- `excludeProviders` - Message providers to skip for recall/retain (e.g. `["slack"]`, `["telegram"]`, `["discord"]`)
- `autoRecall` - Auto-inject memories before each turn (default: `true`). Set to `false` when the agent has its own recall tool.
- `autoRetain` - Auto-retain conversations after each turn (default: `true`)
- `retainRoles` - Which message roles to retain (default: `["user", "assistant"]`). Options: `user`, `assistant`, `system`, `tool`
- `recallBudget` - Recall effort: `"low"`, `"mid"`, or `"high"` (default: `"mid"`). Higher budgets use more retrieval strategies for better results.
- `recallMaxTokens` - Max tokens for recall response (default: `1024`). Controls how much memory context is injected per turn.
- `recallTopK` - Max number of memories to inject per turn (default: unlimited).
- `recallTypes` - Memory types to recall (default: `["world", "experience"]`). Options: `world`, `experience`, `observation`.
- `recallContextTurns` - Number of prior user turns to include in the recall query (default: `1`).
- `recallMaxQueryChars` - Max characters for the composed recall query (default: `800`).
- `recallPromptPreamble` - Custom preamble text placed above recalled memories. Overrides the built-in guidance text.
- `recallInjectionPosition` - Where to inject recalled memories: `"prepend"` (default), `"append"`, or `"user"`. Use `"append"` to preserve prompt caching with large static system prompts. Use `"user"` to inject before the user message instead of in the system prompt.
- `recallRoles` - Which message roles to include when composing the contextual recall query (default: `["user", "assistant"]`).
- `retainEveryNTurns` - Retain every Nth turn (default: `1` = every turn). Values > 1 enable chunked retention.
- `retainOverlapTurns` - Extra prior turns included when chunked retention fires (default: `0`).
- `debug` - Enable debug logging (default: `false`).

### Memory Isolation

The plugin creates separate memory banks based on conversation context. By default, banks are derived from the `agent`, `channel`, and `user` fields — so each unique combination gets its own isolated memory store.

You can customize which fields are used for bank segmentation with `dynamicBankGranularity`:

```json
{
  "plugins": {
    "entries": {
      "hindsight-openclaw": {
        "enabled": true,
        "config": {
          "dynamicBankGranularity": ["provider", "user"]
        }
      }
    }
  }
}
```

In this example, memories are isolated per provider + user, meaning the same user shares memories across all channels within a provider.

Available isolation fields:
- `agent` - The agent/bot identity
- `channel` - The channel or conversation ID
- `user` - The user interacting with the agent
- `provider` - The message provider (e.g. Slack, Discord)

Use `bankIdPrefix` to namespace bank IDs across environments (e.g. `"prod"`, `"staging"`). Set `dynamicBankId` to `false` to use a single shared bank for all conversations. In static mode, the plugin uses `bankId` if set, otherwise the default `openclaw` bank name.

### Retention Controls

By default, the plugin retains `user` and `assistant` messages after each turn. You can customize this behavior:

```json
{
  "plugins": {
    "entries": {
      "hindsight-openclaw": {
        "enabled": true,
        "config": {
          "autoRetain": true,
          "retainRoles": ["user", "assistant", "system"]
        }
      }
    }
  }
}
```

- `autoRetain` - Set to `false` to disable automatic retention entirely (useful if you handle retention yourself)
- `retainRoles` - Controls which message roles are included in the retained transcript. Only messages from the last user message onward are retained each turn, preventing duplicate storage.

### LLM Configuration

> If you used `hindsight-openclaw-setup` in Quick Start, this section is
> already handled for you — read on if you want to edit `openclaw.json`
> directly or switch to a file/exec secret source.

Configure the memory-extraction LLM via OpenClaw's plugin config. API keys
should be stored as `SecretRef` values so they're resolved from env vars,
mounted files, or `exec`-style secret managers (Vault, etc.) at runtime
instead of sitting in plaintext on disk.

| Provider | `llmProvider` | API key |
|---|---|---|
| OpenAI | `openai` | required |
| Anthropic | `anthropic` | required |
| Gemini | `gemini` | required |
| Groq | `groq` | required |
| Ollama | `ollama` | not required (local) |
| Claude Code | `claude-code` | not required (uses Claude Code CLI) |
| OpenAI Codex | `openai-codex` | not required (uses Codex CLI auth) |

**Set provider + API key:**

```bash
openclaw config set plugins.entries.hindsight-openclaw.config.llmProvider openai
openclaw config set plugins.entries.hindsight-openclaw.config.llmApiKey \
    --ref-source env --ref-provider default --ref-id OPENAI_API_KEY
```

**Override the model (optional — Hindsight picks a sensible default per provider):**

```bash
openclaw config set plugins.entries.hindsight-openclaw.config.llmModel gpt-4o-mini
```

**OpenAI-compatible providers (OpenRouter, Azure OpenAI, vLLM, ...):**

```bash
openclaw config set plugins.entries.hindsight-openclaw.config.llmProvider openai
openclaw config set plugins.entries.hindsight-openclaw.config.llmBaseUrl https://openrouter.ai/api/v1
openclaw config set plugins.entries.hindsight-openclaw.config.llmApiKey \
    --ref-source env --ref-provider default --ref-id OPENROUTER_API_KEY
openclaw config set plugins.entries.hindsight-openclaw.config.llmModel xiaomi/mimo-v2-flash
```

**Use a file or exec source instead of env (for K8s secrets, Vault, etc.):**

```bash
# File source (e.g. mounted Docker/K8s secret)
openclaw config set plugins.entries.hindsight-openclaw.config.llmApiKey \
    --ref-source file --ref-provider mounted-json --ref-id /providers/openai/apiKey

# Exec source (e.g. HashiCorp Vault)
openclaw config set plugins.entries.hindsight-openclaw.config.llmApiKey \
    --ref-source exec --ref-provider vault --ref-id openai/api-key
```

The corresponding secret provider needs to be configured under `secrets.providers`
in your OpenClaw config — see `openclaw config set --help` for the
`--provider-source`/`--provider-path`/`--provider-command` builder flags.

### External API (Advanced)

> `hindsight-openclaw-setup --mode api --api-url <url>` covers this path
> interactively — this section documents the underlying config fields.

Connect to a remote Hindsight API server instead of running a local daemon. This is useful for:

- **Shared memory** across multiple OpenClaw instances
- **Production deployments** with centralized memory storage
- **Team environments** where agents share knowledge

#### Plugin Configuration

Configure in `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "hindsight-openclaw": {
        "enabled": true,
        "config": {
          "hindsightApiUrl": "https://your-hindsight-server.com",
          "hindsightApiToken": "your-api-token"
        }
      }
    }
  }
}
```

**Options:**
- `hindsightApiUrl` - Full URL to external Hindsight API (e.g., `https://mcp.hindsight.example.com`)
- `hindsightApiToken` - API token for authentication (optional). **Sensitive** — set as a SecretRef:

  ```bash
  openclaw config set plugins.entries.hindsight-openclaw.config.hindsightApiToken \
      --ref-source env --ref-provider default --ref-id HINDSIGHT_API_TOKEN
  ```

#### Behavior

When external API mode is enabled:
- **No local daemon** is started (no hindsight-embed process)
- **Health check** runs on startup to verify API connectivity
- **All memory operations** (retain, recall, reflect) go to the external API
- **Faster startup** since no local PostgreSQL or embedding models are needed

#### Verification

Check OpenClaw logs for external API mode:

```bash
tail -f /tmp/openclaw/openclaw-*.log | grep Hindsight

# Should see on startup:
# [Hindsight] External API mode enabled: https://your-hindsight-server.com
# [Hindsight] External API health check passed
```

If you see daemon startup messages instead, verify your configuration is correct.

## Inspecting Memories

### Check Configuration

View the daemon config that was written by the plugin:

```bash
cat ~/.hindsight/profiles/openclaw.env
```

This shows the LLM provider, model, port, and other settings the daemon is using.

### Check Daemon Status

```bash
# Check if daemon is running
uvx hindsight-embed@latest -p openclaw daemon status

# View daemon logs
tail -f ~/.hindsight/profiles/openclaw.log
```

### Query Memories

```bash
# Search memories
uvx hindsight-embed@latest -p openclaw memory recall openclaw "user preferences"

# View recent memories
uvx hindsight-embed@latest -p openclaw memory list openclaw --limit 10

# Open web UI (uses openclaw profile's daemon)
uvx hindsight-embed@latest -p openclaw ui
```

## Troubleshooting

### Plugin not loading

```bash
openclaw plugins list | grep hindsight
# Should show: ✓ enabled │ Hindsight Memory │ ...

# Reinstall if needed
openclaw plugins install @vectorize-io/hindsight-openclaw
```

### Daemon not starting

```bash
# Check daemon status (note: -p openclaw uses the openclaw profile)
uvx hindsight-embed@latest -p openclaw daemon status

# View logs for errors
tail -f ~/.hindsight/profiles/openclaw.log

# Check configuration
cat ~/.hindsight/profiles/openclaw.env

# List all profiles
uvx hindsight-embed@latest profile list
```

### No API key error

Make sure you've configured the LLM provider through `openclaw config set`
(or use a provider that doesn't require a key):

```bash
# Option 1 — OpenAI (requires OPENAI_API_KEY in your env)
openclaw config set plugins.entries.hindsight-openclaw.config.llmProvider openai
openclaw config set plugins.entries.hindsight-openclaw.config.llmApiKey \
    --ref-source env --ref-provider default --ref-id OPENAI_API_KEY

# Option 2 — Anthropic
openclaw config set plugins.entries.hindsight-openclaw.config.llmProvider anthropic
openclaw config set plugins.entries.hindsight-openclaw.config.llmApiKey \
    --ref-source env --ref-provider default --ref-id ANTHROPIC_API_KEY

# Option 3 — Claude Code (no API key needed)
openclaw config set plugins.entries.hindsight-openclaw.config.llmProvider claude-code

# Option 4 — OpenAI Codex (no API key needed)
openclaw config set plugins.entries.hindsight-openclaw.config.llmProvider openai-codex

# Verify the config is valid
openclaw config validate

# Inspect the current value
openclaw config get plugins.entries.hindsight-openclaw.config.llmProvider
```

If you used `--ref-source env`, double-check that the referenced env var
(e.g. `OPENAI_API_KEY`) is exported in the shell that runs `openclaw gateway`.

### Verify it's working

Check gateway logs for memory operations:

```bash
tail -f /tmp/openclaw/openclaw-*.log | grep Hindsight

# Should see on startup:
# [Hindsight] ✓ Using provider: openai, model: gpt-4o-mini
# or
# [Hindsight] ✓ Using provider: claude-code, model: claude-sonnet-4-20250514

# Should see after conversations:
# [Hindsight] Retained X messages for session ...
# [Hindsight] Auto-recall: Injecting X memories
```

## Migration from 0.5.x

0.6.0 removes all process-environment reads from the plugin. Configuration that
previously came from shell env vars must now go through OpenClaw's plugin config
(with `SecretRef` for credentials). The plugin no longer auto-detects providers
from `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / etc. — you must set
`llmProvider` explicitly.

| Old (0.5.x) | New (0.6.0) |
|---|---|
| `OPENAI_API_KEY=…` (auto-detected) | `openclaw config set plugins.entries.hindsight-openclaw.config.llmProvider openai` <br/> `openclaw config set plugins.entries.hindsight-openclaw.config.llmApiKey --ref-source env --ref-id OPENAI_API_KEY` |
| `HINDSIGHT_API_LLM_PROVIDER=…` | `openclaw config set plugins.entries.hindsight-openclaw.config.llmProvider …` |
| `HINDSIGHT_API_LLM_MODEL=…` | `openclaw config set plugins.entries.hindsight-openclaw.config.llmModel …` |
| `HINDSIGHT_API_LLM_API_KEY=…` | `openclaw config set plugins.entries.hindsight-openclaw.config.llmApiKey --ref-source env --ref-id …` |
| `HINDSIGHT_API_LLM_BASE_URL=…` | `openclaw config set plugins.entries.hindsight-openclaw.config.llmBaseUrl …` |
| `HINDSIGHT_EMBED_API_URL=…` | `openclaw config set plugins.entries.hindsight-openclaw.config.hindsightApiUrl …` |
| `HINDSIGHT_EMBED_API_TOKEN=…` | `openclaw config set plugins.entries.hindsight-openclaw.config.hindsightApiToken --ref-source env --ref-id …` |
| `HINDSIGHT_BANK_ID=…` | `openclaw config set plugins.entries.hindsight-openclaw.config.bankId …` |
| `llmApiKeyEnv: "MY_KEY"` (plugin config) | `llmApiKey` configured as a SecretRef with `--ref-id MY_KEY` |

If your shell already exports `OPENAI_API_KEY`, the SecretRef config above
resolves to the same value at startup — you don't need to change your shell
setup, just point the plugin at the variable explicitly. Run
`openclaw config validate` after migrating to confirm the new shape parses
cleanly.
