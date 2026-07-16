# Hindsight for GitHub Copilot CLI

Long-term memory for [GitHub Copilot CLI](https://docs.github.com/en/copilot/concepts/agents/copilot-cli/about-copilot-cli) — remembers your projects, preferences, and past sessions across every conversation, and gives subagents (`explore`, `task`, `research`, `code-review`, `rubber-duck`, `security-review`, custom agents) baseline project memory too.

## How it works

Four Copilot CLI hooks keep memory in sync automatically:

| Hook | Action |
|------|--------|
| `sessionStart` | Recalls relevant memories (using the queued prompt, or a project-derived fallback query) and injects them as `additionalContext` |
| `subagentStart` | Recalls memories for a spawned subagent (using the same fallback query — the payload never carries the subagent's specific task) and injects them as `additionalContext` |
| `agentStop` | Retains the conversation to long-term memory every configured N turns |
| `sessionEnd` | Forces a final retain (using the transcript path cached from the last `agentStop`) so short sessions are still stored |

## Limitations

Copilot CLI's `userPromptSubmitted` and `preToolUse` hooks do **not** support injecting `additionalContext` — only `sessionStart`, `subagentStart`, `postToolUse`, and `notification` can. That means recall happens **once per session start** (and once per subagent spawn), not before every prompt. If your session runs long, memories recalled at the start may go stale; a `postToolUse`-based per-turn refresh is a documented possible follow-up, not implemented here.

The built-in `general-purpose` subagent does not emit `subagentStart`/`subagentStop` — it never receives injected memory from this integration.

`subagentStop` (retaining each subagent's own transcript) is intentionally not implemented in v1 — it risks duplicating content already captured by the parent session's `agentStop`, and adds noisy memories for short-lived exploratory subagents.

## Requirements

- **GitHub Copilot CLI** with [hooks support](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/use-hooks)
- **Python 3.9+** (for hook scripts; stdlib only — no pip install required)
- **Hindsight**: [Hindsight Cloud](https://hindsight.vectorize.io) or local `hindsight-embed`

## Installation

Sign up free at [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io/signup) for a Hindsight Cloud API key — or run a local server.

```bash
pip install hindsight-copilot-cli
```

Then run the installer once:

```bash
# Hindsight Cloud
hindsight-copilot-cli install --api-url https://api.hindsight.vectorize.io --api-token your-api-key

# Local daemon (hindsight-embed) — omit the flags
hindsight-copilot-cli install
```

The installer:

1. Copies the hook scripts to `~/.copilot/hindsight-copilot-cli/scripts/` (or `$COPILOT_HOME/hindsight-copilot-cli/scripts/` if `COPILOT_HOME` is set)
2. Writes a standalone `~/.copilot/hooks/hindsight-copilot-cli.json` with absolute paths to the scripts — Copilot CLI loads every `*.json` file in its hooks directory, so this never needs to merge with other tools' hook files
3. Seeds `~/.hindsight/copilot-cli.json` if it doesn't exist (drop your `hindsightApiToken` here later)

Restart Copilot CLI to load the hooks. If memories are not recalled or retained, check that
`~/.copilot/hooks/hindsight-copilot-cli.json` exists and that `python3` is on `$PATH` from your shell.

### Repo-scope install (shared team hooks)

To register the hooks at the repository level instead (`.github/hooks/hindsight-copilot-cli.json`, so a team can check in shared hook config):

```bash
hindsight-copilot-cli install --scope repo
```

The hook *scripts* are still installed once per machine under `~/.copilot`; only the registration
pointer is written to `.github/hooks/hindsight-copilot-cli.json`. Since that path is absolute and per-machine,
every teammate needs to run the installer locally too. Never commit an API token — set
`HINDSIGHT_API_TOKEN` in each environment instead.

### Uninstall

```bash
hindsight-copilot-cli uninstall
# or, for a repo-scope install:
hindsight-copilot-cli uninstall --scope repo
```

This removes the hook scripts and the `hindsight-copilot-cli.json` registration file for the given scope. Your
personal config at `~/.hindsight/copilot-cli.json` is preserved.

## Configuration

Default config lives in `~/.copilot/hindsight-copilot-cli/settings.json`. For personal overrides stable across updates, create `~/.hindsight/copilot-cli.json`:

```json
{
  "hindsightApiUrl": "https://api.hindsight.vectorize.io",
  "hindsightApiToken": "your-api-key",
  "bankId": "my-copilot-memory"
}
```

### Configuration options

| Key | Default | Description |
|-----|---------|-------------|
| `hindsightApiUrl` | `""` | External API URL (empty = local daemon) |
| `hindsightApiToken` | `null` | API token for Hindsight Cloud |
| `bankId` | `"copilot-cli"` | Memory bank identifier |
| `bankMission` | (set) | Guides what facts Hindsight retains |
| `autoRecall` | `true` | Recall memories on `sessionStart` and `subagentStart` |
| `autoRetain` | `true` | Store conversations after each turn |
| `retainMode` | `"full-session"` | `"full-session"` or `"chunked"` |
| `retainEveryNTurns` | `10` | Retain every N turns (1 = every turn) |
| `retainToolCalls` | `true` | Preserve tool calls/results as structured blocks when retaining |
| `recallFallbackQueryTemplate` | (set) | Query used when there's no specific prompt to recall against (interactive `sessionStart` with no queued prompt, or any `subagentStart`). `{project}` is replaced with the `cwd` basename |
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

**Recall** — on `sessionStart`, Hindsight searches your memory bank for facts relevant to the queued prompt (or a generic project-context query when there isn't one) and injects them as `additionalContext`. The same happens on `subagentStart` for every subagent spawned during the session, since subagents otherwise run in a fully isolated context with no access to the parent session's memory.

**Retain** — after configured turns (`agentStop`) and again when the session ends (`sessionEnd`), the session transcript is read from Copilot CLI's `transcriptPath` and stored to Hindsight. The memory engine extracts facts, relationships, and experiences — so you don't need to re-explain your stack, preferences, or past decisions.

## Dynamic bank IDs

To keep separate memory per project:

```json
{
  "dynamicBankId": true,
  "dynamicBankGranularity": ["agent", "project"]
}
```

This creates banks like `copilot-cli::my-project` automatically, deriving the project name from the `cwd` field every Copilot CLI hook payload carries.

## Troubleshooting

**Memory not appearing**: enable debug mode (`"debug": true`) and check that `HINDSIGHT_API_URL` points to a reachable server. Logs go to `~/.hindsight/copilot-cli/state/*.log` when debug is on and stderr is redirected — otherwise check the hook's stderr directly via Copilot CLI's own debug output.

**Hooks not firing**: check that `~/.copilot/hooks/hindsight-copilot-cli.json` (or `.github/hooks/hindsight-copilot-cli.json` for repo scope) is valid JSON. Copilot CLI loads hook configuration when it starts, so restart the CLI after installing or updating hooks.

**Subagents not getting memory**: the built-in `general-purpose` agent never emits `subagentStart` — this is a Copilot CLI limitation, not a bug in this integration. All other built-in agents (`explore`, `task`, `code-review`, `rubber-duck`, `research`, `security-review`) and custom agents do emit it.

## Development

```bash
cd hindsight-integrations/copilot-cli
uv sync
uv run pytest tests/ -v
```

The tests mock the HTTP client, the stdin/stdout pipe, and the file-based state. No live Hindsight server or Copilot CLI installation is required.

## License

MIT
