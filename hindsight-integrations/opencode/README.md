# @vectorize-io/opencode-hindsight

Hindsight memory plugin for [OpenCode](https://opencode.ai) — give your AI coding agent persistent long-term memory across sessions.

## Features

- **Custom tools**: `hindsight_retain`, `hindsight_recall`, `hindsight_reflect` — the agent calls these explicitly
- **Auto-retain**: Captures conversation on `session.idle` and stores to Hindsight
- **Memory injection**: Recalls relevant memories when a new session starts
- **Compaction hook**: Injects memories during context compaction so they survive window trimming

## Quick Start

The plugin defaults to **Hindsight Cloud** (`https://api.hindsight.vectorize.io`). Just enable it and provide your API key.

### 1. Enable the plugin

Add to your `opencode.json` (project) or `~/.config/opencode/opencode.json` (global):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["@vectorize-io/opencode-hindsight"]
}
```

OpenCode auto-installs plugins listed here on startup — no `npm install` required.

### 2. Provide your Hindsight Cloud API key

Get an API key at [ui.hindsight.vectorize.io/connect](https://ui.hindsight.vectorize.io/connect), then:

```bash
export HINDSIGHT_API_TOKEN="your-api-key"

# Optional: override the memory bank ID (defaults to "opencode")
export HINDSIGHT_BANK_ID="my-project"
```

That's it — the plugin now reads/writes against your Cloud bank.

### Using a self-hosted Hindsight instance

Point `HINDSIGHT_API_URL` at your server (the API key is then optional):

```bash
export HINDSIGHT_API_URL="http://localhost:8888"
```

Or configure inline in `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": [
    [
      "@vectorize-io/opencode-hindsight",
      {
        "hindsightApiUrl": "http://localhost:8888"
      }
    ]
  ]
}
```

## Configuration

### Plugin Options

Pass options directly in `opencode.json`:

```json
{
  "plugin": [
    [
      "@vectorize-io/opencode-hindsight",
      {
        "hindsightApiUrl": "http://localhost:8888",
        "bankId": "my-project",
        "autoRecall": true,
        "autoRetain": true,
        "recallBudget": "mid"
      }
    ]
  ]
}
```

### Config File

Create `~/.hindsight/opencode.json` for persistent configuration:

```json
{
  "hindsightApiUrl": "http://localhost:8888",
  "hindsightApiToken": "your-api-key",
  "recallBudget": "mid",
  "retainEveryNTurns": 3,
  "debug": false
}
```

### Per-Agent API Keys

OpenCode drives sessions with named agents (e.g. `build`, `code-reviewer`, or
your own custom agents). You can give each agent its own Hindsight API key —
the plugin resolves the key at request time from the **name of the agent**
running the current turn:

```json
{
  "plugin": [
    [
      "@vectorize-io/opencode-hindsight",
      {
        "hindsightApiToken": "fallback-key",
        "hindsightApiTokens": {
          "build": "build-agent-key",
          "code-reviewer": "reviewer-key",
          "angular-dev-expert": "angular-key"
        }
      }
    ]
  ]
}
```

Resolution order (first non-empty value wins):

1. `hindsightApiTokens[<current agent name>]` — the agent driving the turn.
   Tools read the agent from OpenCode's tool context; hooks read it from the
   most recent user message in the session.
2. `hindsightApiTokens[agentName]` — the entry for the configured default
   `agentName`.
3. `hindsightApiToken` — the single static fallback key (legacy behavior).

Agents with no entry in the map fall back to `hindsightApiToken`, so existing
single-key setups keep working unchanged. Set `"dynamicApiKey": false` to
disable per-agent resolution and force use of `hindsightApiToken` everywhere.

The per-agent map can also be supplied via the `HINDSIGHT_API_TOKENS`
environment variable as a JSON object (see below).

### Per-Agent Bank IDs

By default every agent in a project shares one memory bank (derived from
`bankId`, or the dynamic `agent::project` composition). You can give each
agent its **own memory bank** — the plugin resolves the bank at request time
from the **name of the agent** driving the turn. There are two ways to map
agents to bank IDs.

#### Agent file frontmatter (highest precedence)

Add a `bankid` field to the YAML frontmatter of any agent definition file
(`.opencode/agent/<name>.md`, `~/.config/opencode/agents/<name>.md`, etc.):

```markdown
---
description: Reviews recently written code
mode: all
bankid: opencode-code-reviewer
---

You are an elite code reviewer…
```

When present, this value **takes precedence over every other source**
(including `hindsightBankIds` and the static `bankId`). The `bankIdPrefix`
is applied. File results are cached per agent name, keyed on file mtime, so
repeated lookups during a session are cheap. Project-scoped agent files
(`.opencode/agent(s)/`) take precedence over global ones
(`~/.config/opencode/agent(s)/`), mirroring how OpenCode merges configs.

#### Config-file map (`hindsightBankIds`)

Map agent names to bank IDs in `opencode.json` or `~/.hindsight/opencode.json`:

```json
{
  "plugin": [
    [
      "@vectorize-io/opencode-hindsight",
      {
        "bankId": "fallback-bank",
        "hindsightBankIds": {
          "build": "build-bank",
          "code-reviewer": "reviewer-bank",
          "angular-dev-expert": "angular-bank"
        }
      }
    ]
  ]
}
```

#### Resolution order (first defined value wins)

1. **Agent `.md` frontmatter `bankid`** — read from the agent's definition
   file (highest precedence). `bankIdPrefix` is applied.
2. `hindsightBankIds[<current agent name>]` — the agent driving the turn
   (read from the tool context, or the most recent user message in a session).
   `bankIdPrefix` is applied.
3. `hindsightBankIds[agentName]` — the entry for the configured default
   `agentName`. `bankIdPrefix` is applied.
4. The normal `deriveBankId` result — static `bankId`, or the dynamic
   `dynamicBankGranularity` composition (`agent::project`, `gitProject`, …).
   `bankIdPrefix` is already applied by the derivation.

Agents with no `bankid` in their `.md` file and no entry in the map fall back
to the shared bank, so existing single-bank setups keep working unchanged. The
per-agent map can also be supplied via the `HINDSIGHT_BANK_IDS` environment
variable as a JSON object.

### Environment Variables

| Variable                      | Description                                              | Default                               |
| ----------------------------- | -------------------------------------------------------- | ------------------------------------- |
| `HINDSIGHT_API_URL`           | Hindsight API base URL                                   | `https://api.hindsight.vectorize.io`  |
| `HINDSIGHT_API_TOKEN`         | API key for authentication (fallback for unmapped agents) | (none — required for Hindsight Cloud) |
| `HINDSIGHT_API_TOKENS`        | JSON object mapping agent name → API key                 | `{}`                                  |
| `HINDSIGHT_DYNAMIC_API_KEY`   | Enable per-agent key resolution from `hindsightApiTokens` | `true`                              |
| `HINDSIGHT_BANK_ID`           | Static memory bank ID (fallback for unmapped agents)     | `opencode`                            |
| `HINDSIGHT_BANK_IDS`          | JSON object mapping agent name → bank ID                 | `{}`                                  |
| `HINDSIGHT_AGENT_NAME`        | Agent name for dynamic bank IDs                          | `opencode`                            |
| `HINDSIGHT_AUTO_RECALL`       | Auto-recall on session start                             | `true`                                |
| `HINDSIGHT_AUTO_RETAIN`       | Auto-retain on session idle                              | `true`                                |
| `HINDSIGHT_RETAIN_MODE`       | `full-session` or `last-turn`                            | `full-session`                        |
| `HINDSIGHT_RECALL_BUDGET`     | Recall budget: `low`, `mid`, `high`                      | `mid`                                 |
| `HINDSIGHT_RECALL_MAX_TOKENS` | Max tokens for recall results                            | `1024`                                |
| `HINDSIGHT_RECALL_TAGS`       | Comma-separated, filter recalls                          | (none)                                |
| `HINDSIGHT_RECALL_TAGS_MATCH` | Tag match mode: `any`, `all`, `any_strict`, `all_strict` | `any`                                 |
| `HINDSIGHT_RETAIN_TAGS`       | Comma-separated, added to every retain                   | (none)                                |
| `HINDSIGHT_DYNAMIC_BANK_ID`   | Enable dynamic bank ID derivation                        | `false`                               |
| `HINDSIGHT_BANK_MISSION`      | Bank mission/context                                     | (none)                                |

> **Debug logging** is a config-only option (`"debug": true` in `opencode.json`
> plugin options or `~/.hindsight/opencode.json`) — there is intentionally no
> `HINDSIGHT_DEBUG` env var, because environment variables are unreliable to set
> for OpenCode's plugin runtime (notably on Windows). Errors and the resolved
> API URL/bank are logged regardless of this setting; `debug` only adds verbose
> tracing. All plugin logs go to OpenCode's log stream (`service=hindsight`),
> visible with `--print-logs` or in the OpenCode log files.

### Configuration Priority

Settings are loaded in this order (later wins):

1. Built-in defaults
2. `~/.hindsight/opencode.json`
3. Plugin options from `opencode.json`
4. Environment variables

## Tools

### `hindsight_retain`

Store information in long-term memory. The agent uses this to save important facts, user preferences, project context, and decisions.

### `hindsight_recall`

Search long-term memory. The agent uses this proactively before answering questions where prior context would help.

### `hindsight_reflect`

Generate a synthesized answer from long-term memory. Unlike recall (raw memories), reflect produces a coherent summary.

## Dynamic Bank IDs

For multi-project setups, enable dynamic bank ID derivation:

```bash
export HINDSIGHT_DYNAMIC_BANK_ID=true
```

The bank ID is composed from granularity fields (default: `agent::project`). Supported fields: `agent`, `project`, `gitProject`, `channel`, `user`.

- `project` uses the working directory basename. With this field, separate git worktrees of the same repository end up with different bank IDs because their paths differ.
- `gitProject` resolves to the main worktree's basename via `git rev-parse --git-common-dir`, so all linked worktrees of the same repository share a single bank. Falls back to the working directory basename when git is unavailable or the directory is not a repo. Use this in place of `project` if you want worktrees to share memory:

```json
{
  "dynamicBankId": true,
  "dynamicBankGranularity": ["agent", "gitProject"]
}
```

**Note:** The bank ID is derived once when the plugin loads, from environment variables set before OpenCode starts. These dimensions are process-scoped — they don't change per session within a running OpenCode process. For per-user isolation, set the env vars before launching each user's OpenCode instance:

```bash
export HINDSIGHT_CHANNEL_ID="slack-general"
export HINDSIGHT_USER_ID="user123"
```

## Development

```bash
npm install
npm test        # Run tests
npm run build   # Build to dist/
```

## License

MIT
