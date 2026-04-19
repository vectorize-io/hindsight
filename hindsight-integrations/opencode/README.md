# @vectorize-io/opencode-hindsight

Hindsight memory plugin for [OpenCode](https://opencode.ai) — give your AI coding agent persistent long-term memory across sessions.

## Features

- **Custom tools**: `hindsight_retain`, `hindsight_recall`, `hindsight_reflect` — the agent calls these explicitly
- **Auto-retain**: Captures conversation on `session.idle` and stores to Hindsight
- **Memory injection**: Recalls relevant memories when a new session starts
- **Compaction hook**: Injects memories during context compaction so they survive window trimming

## Quick Start

### 1. Enable the plugin

Add to your `opencode.json` (project) or `~/.config/opencode/opencode.json` (global):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": ["@vectorize-io/opencode-hindsight"]
}
```

OpenCode auto-installs plugins listed here on startup — no `npm install` required.

### 2. Point to your Hindsight server

```bash
# Self-hosted
export HINDSIGHT_API_URL="http://localhost:8888"

# Optional: override the memory bank ID
export HINDSIGHT_BANK_ID="my-project"
```

### Using Hindsight Cloud

Get an API key at [ui.hindsight.vectorize.io/connect](https://ui.hindsight.vectorize.io/connect), then either export env vars:

```bash
export HINDSIGHT_API_URL="https://api.hindsight.vectorize.io"
export HINDSIGHT_API_TOKEN="your-api-key"
```

Or configure inline in `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": [
    [
      "@vectorize-io/opencode-hindsight",
      {
        "hindsightApiUrl": "https://api.hindsight.vectorize.io",
        "hindsightApiToken": "your-api-key"
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
  "retainEveryNTurns": 10,
  "debug": false
}
```

### Environment Variables

| Variable                      | Description                         | Default        |
| ----------------------------- | ----------------------------------- | -------------- |
| `HINDSIGHT_API_URL`           | Hindsight API base URL              | (required)     |
| `HINDSIGHT_API_TOKEN`         | API key for authentication          | (none)         |
| `HINDSIGHT_BANK_ID`           | Static memory bank ID               | `opencode`     |
| `HINDSIGHT_AGENT_NAME`        | Agent name for dynamic bank IDs     | `opencode`     |
| `HINDSIGHT_AUTO_RECALL`       | Auto-recall on session start        | `true`         |
| `HINDSIGHT_AUTO_RETAIN`       | Auto-retain on session idle         | `true`         |
| `HINDSIGHT_RETAIN_MODE`       | `full-session` or `last-turn`       | `full-session` |
| `HINDSIGHT_RECALL_BUDGET`     | Recall budget: `low`, `mid`, `high` | `mid`          |
| `HINDSIGHT_RECALL_MAX_TOKENS` | Max tokens for recall results       | `1024`         |
| `HINDSIGHT_DYNAMIC_BANK_ID`   | Enable dynamic bank ID derivation   | `false`        |
| `HINDSIGHT_BANK_MISSION`      | Bank mission/context                | (none)         |
| `HINDSIGHT_DEBUG`             | Enable debug logging                | `false`        |

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

The bank ID is composed from granularity fields (default: `agent::project`). Supported fields: `agent`, `project`, `channel`, `user`.

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
