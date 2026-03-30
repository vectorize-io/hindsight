# Hindsight for OpenAI Codex CLI

Long-term memory for [OpenAI Codex CLI](https://github.com/openai/codex) — remembers your projects, preferences, and past sessions across every conversation.

## How it works

Three Codex hooks keep memory in sync automatically:

| Hook | Action |
|------|--------|
| `SessionStart` | Warms up the Hindsight server in the background |
| `UserPromptSubmit` | Recalls relevant memories and injects them into context |
| `Stop` | Retains the conversation to long-term memory |

## Requirements

- **OpenAI Codex CLI** v0.116.0 or later (hooks support)
- **Python 3.9+** (for hook scripts)
- **Hindsight**: [Hindsight Cloud](https://hindsight.vectorize.io) or local `hindsight-embed`

## Installation

```bash
curl -fsSL https://hindsight.vectorize.io/get-codex | bash
```

The installer:
1. Downloads scripts to `~/.hindsight/codex/scripts/`
2. Writes `~/.codex/hooks.json` with absolute paths to the scripts
3. Adds `codex_hooks = true` to `~/.codex/config.toml`

### Uninstall

```bash
curl -fsSL https://hindsight.vectorize.io/get-codex | bash -s -- --uninstall
```

## Configuration

The default config is written to `~/.hindsight/codex/settings.json` on first install.

For personal overrides (stable across updates), create `~/.hindsight/codex.json`:

```json
{
  "hindsightApiUrl": "https://api.hindsight.vectorize.io",
  "hindsightApiToken": "your-api-key",
  "bankId": "my-codex-memory"
}
```

### Hindsight Cloud

```json
{
  "hindsightApiUrl": "https://api.hindsight.vectorize.io",
  "hindsightApiToken": "your-api-key"
}
```

### Local daemon (hindsight-embed)

Set an LLM API key and Hindsight will start the local server automatically:

```bash
export OPENAI_API_KEY=sk-your-key
# or
export ANTHROPIC_API_KEY=your-key
```

### Configuration options

| Key | Default | Description |
|-----|---------|-------------|
| `hindsightApiUrl` | `""` | External API URL (empty = local daemon) |
| `hindsightApiToken` | `null` | API token for Hindsight Cloud |
| `bankId` | `"codex"` | Memory bank identifier |
| `bankMission` | (set) | Guides what facts Hindsight retains |
| `autoRecall` | `true` | Inject memories before each prompt |
| `autoRetain` | `true` | Store conversations after each turn |
| `retainMode` | `"full-session"` | `"full-session"` or `"chunked"` |
| `retainEveryNTurns` | `10` | Retain every N turns (1 = every turn) |
| `recallBudget` | `"mid"` | Recall depth: `"low"`, `"mid"`, `"high"` |
| `recallMaxTokens` | `1024` | Max tokens for injected memories |
| `dynamicBankId` | `false` | Separate bank per project/session |
| `dynamicBankGranularity` | `["agent", "project"]` | Fields for dynamic bank ID |
| `debug` | `false` | Log debug info to stderr |

### Environment variable overrides

All settings can also be set via environment variables:

```bash
export HINDSIGHT_API_URL=https://api.hindsight.vectorize.io
export HINDSIGHT_API_TOKEN=your-api-key
export HINDSIGHT_BANK_ID=my-project
export HINDSIGHT_DEBUG=true
```

## How memory works

**Recall** — before each prompt, Hindsight searches your memory bank for facts relevant to what you're about to ask. Found memories are injected as context so Codex has continuity across sessions.

**Retain** — after each turn, Codex's conversation is stored to Hindsight. The memory engine extracts facts, relationships, and experiences — so you don't need to re-explain your stack, preferences, or past decisions.

## Dynamic bank IDs

To keep separate memory per project:

```json
{
  "dynamicBankId": true,
  "dynamicBankGranularity": ["agent", "project"]
}
```

This creates banks like `codex::my-project` automatically, using the working directory name.

## Troubleshooting

**Memory not appearing**: Enable debug mode (`"debug": true`) and check stderr output.

**Server not starting**: Set `hindsightApiUrl` to use an external server, or ensure `uvx` is on PATH for local daemon mode.

**Hooks not firing**: Check that `~/.codex/config.toml` contains `codex_hooks = true` under `[features]`, and that your Codex CLI version supports hooks (v0.116.0+).
