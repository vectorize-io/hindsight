# Hindsight for Cursor CLI

Long-term memory for [Cursor CLI](https://cursor.com/docs/cli) — remembers your projects, preferences, and past sessions across every conversation.

## How it works

Four Cursor CLI hooks keep memory in sync automatically:

| Hook | Action |
|------|--------|
| `sessionStart` | Confirms Hindsight is reachable and pre-warms the local daemon if needed |
| `beforeSubmitPrompt` | Recalls relevant memories and injects them as `additional_context` |
| `stop` | Retains the conversation to long-term memory every configured N turns |
| `sessionEnd` | Forces a final retain so short sessions are still stored |

## Requirements

- **Cursor CLI** v0.45+ with hooks support
- **Python 3.9+** (for hook scripts; stdlib only — no pip install required)
- **Hindsight**: [Hindsight Cloud](https://hindsight.vectorize.io) or local `hindsight-embed`

## Installation

Sign up free at [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io/signup) for a Hindsight Cloud API key — or run a local server.

```bash
pip install hindsight-cursor-cli
```

Then run the installer once:

```bash
# Hindsight Cloud
hindsight-cursor-cli install --api-url https://api.hindsight.vectorize.io --api-token your-api-key

# Local daemon (hindsight-embed) — omit the flags
hindsight-cursor-cli install
```

The installer:

1. Copies the hook scripts to `~/.cursor/hooks/cursor-cli/`
2. Writes `~/.cursor/hooks.json` (merged with any existing entries) with absolute paths to the scripts
3. Seeds `~/.hindsight/cursor-cli.json` if it doesn't exist (drop your `hindsightApiToken` here later)

Restart Cursor CLI to load the hooks. If memories are not recalled or retained, check that
`~/.cursor/hooks.json` exists and that `python3` is on `$PATH` from your shell.

### Uninstall

```bash
hindsight-cursor-cli uninstall
```

This removes the hook scripts and strips Hindsight's entries from `~/.cursor/hooks.json`. Your
personal config at `~/.hindsight/cursor-cli.json` is preserved.

## Configuration

Default config lives in `~/.cursor/hooks/cursor-cli/settings.json`. For personal overrides stable across updates, create `~/.hindsight/cursor-cli.json`:

```json
{
  "hindsightApiUrl": "https://api.hindsight.vectorize.io",
  "hindsightApiToken": "your-api-key",
  "bankId": "my-cursor-memory"
}
```

### Configuration options

| Key | Default | Description |
|-----|---------|-------------|
| `hindsightApiUrl` | `""` | External API URL (empty = local daemon) |
| `hindsightApiToken` | `null` | API token for Hindsight Cloud |
| `bankId` | `"cursor-cli"` | Memory bank identifier |
| `bankMission` | (set) | Guides what facts Hindsight retains |
| `autoRecall` | `true` | Inject memories before each prompt |
| `autoRetain` | `true` | Store conversations after each turn |
| `retainMode` | `"full-session"` | `"full-session"` or `"chunked"` |
| `retainEveryNTurns` | `10` | Retain every N turns (1 = every turn) |
| `recallBudget` | `"mid"` | Recall depth: `"low"`, `"mid"`, `"high"` |
| `recallMaxTokens` | `1024` | Max tokens for injected memories |
| `recallTimeout` | `10` | Timeout in seconds for recall API calls |
| `dynamicBankId` | `false` | Separate bank per project |
| `dynamicBankGranularity` | `["agent", "project"]` | Fields for dynamic bank ID |
| `debug` | `false` | Log debug info to stderr |

### Environment variable overrides

All settings can also be set via environment variables:

```bash
export HINDSIGHT_API_URL=https://api.hindsight.vectorize.io
export HINDSIGHT_API_TOKEN=your-api-key
export HINDSIGHT_BANK_ID=my-project
export HINDSIGHT_RECALL_TIMEOUT=30
export HINDSIGHT_DEBUG=true
```

## How memory works

**Recall** — before each prompt, Hindsight searches your memory bank for facts relevant to what you're about to ask. Found memories are injected as `additional_context` so the agent has continuity across sessions.

**Retain** — after configured turns and again when the session ends, Cursor's conversation transcript is stored to Hindsight. The memory engine extracts facts, relationships, and experiences — so you don't need to re-explain your stack, preferences, or past decisions.

## Dynamic bank IDs

To keep separate memory per project:

```json
{
  "dynamicBankId": true,
  "dynamicBankGranularity": ["agent", "project"]
}
```

This creates banks like `cursor-cli::my-project` automatically, using either `CURSOR_PROJECT_DIR` (Cursor's env var) or the first entry of `workspace_roots` from the hook's common input fields.

To share memory across all worktrees of the same repo, use `gitProject` instead of `project`:

```json
{
  "dynamicBankId": true,
  "dynamicBankGranularity": ["agent", "gitProject"]
}
```

## Troubleshooting

**No "Hindsight is active" note on session start**: run with `"debug": true` (or `HINDSIGHT_DEBUG=true`) and check stderr.

**Memory not appearing**: enable debug mode (`"debug": true`) and check that `HINDSIGHT_API_URL` points to a reachable server.

**Hooks not firing**: check that `~/.cursor/hooks.json` is valid JSON and contains the four hook entries. Cursor CLI requires a session restart to pick up new hooks.

## Development

```bash
cd hindsight-integrations/cursor-cli
uv sync
uv run pytest tests/ -v
```

The tests mock the HTTP client, the stdin/stdout pipe, and the file-based state. No live Hindsight server is required.

## License

MIT
