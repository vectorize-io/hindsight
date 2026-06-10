---
sidebar_position: 7
title: "Cursor CLI Persistent Memory with Hindsight | Integration Guide"
description: "Add persistent memory to Cursor CLI with Hindsight. Python hook scripts automatically recall context before each prompt and retain conversations — no workflow changes required."
---

# Cursor CLI

[View Changelog →](/changelog/integrations/cursor-cli)

Persistent memory for [Cursor CLI](https://docs.cursor.com/en/cli/overview) using [Hindsight](https://vectorize.io/hindsight). Python hook scripts automatically recall relevant context before each prompt and retain conversations after each turn — no changes to your Cursor workflow required.

## Quick Start

:::tip Recommended: Hindsight Cloud
[Sign up free](https://ui.hindsight.vectorize.io/signup) for a Hindsight Cloud API key — no self-hosting, no local daemon to manage.
:::

```bash
# Install the CLI
pip install hindsight-cursor-cli

# Install the hooks (defaults to Hindsight Cloud)
hindsight-cursor-cli install --api-url https://api.hindsight.vectorize.io --api-token your-api-key

# Restart Cursor CLI — memory is live
```

The installer copies the hook scripts to `~/.cursor/hooks/cursor-cli/`, writes `~/.cursor/hooks.json` (merged with any existing entries), and creates `~/.hindsight/cursor-cli.json` for your personal config.

**Self-hosting alternative** — connect to a local `hindsight-embed` daemon by omitting the flags:

```bash
hindsight-cursor-cli install
```

To uninstall:

```bash
hindsight-cursor-cli uninstall
```

## Features

- **Auto-recall** — before each prompt, queries Hindsight for relevant memories and injects them as additional context (visible to the model, not the transcript)
- **Auto-retain** — after each response, and again on session end, stores the conversation to Hindsight for future recall
- **Dynamic bank IDs** — supports per-project memory isolation based on the working directory
- **Session-level upsert** — uses the session ID as the document ID so re-running the same session updates rather than duplicates stored content
- **Zero runtime dependencies** — the hook scripts are pure Python stdlib; the `pip install` only ships the one-time installer

## Architecture

The plugin uses four Cursor CLI hook events:

| Hook | Event | Purpose |
|------|-------|---------|
| `session_start.py` | `sessionStart` | Warm up — verify Hindsight is reachable |
| `recall.py` | `beforeSubmitPrompt` | **Auto-recall** — query memories, inject as additional context |
| `retain.py` | `stop` | **Auto-retain** — extract transcript, POST to Hindsight (async) |
| `session_end.py` | `sessionEnd` | **Final flush** — force a retain so the last turns aren't lost |

On `beforeSubmitPrompt`, the hook reads the prompt, queries Hindsight for the most relevant memories, and injects a context block. Cursor prepends this to the conversation before sending it to the model:

```
<hindsight_memories>
Relevant memories from past conversations...
Current time - 2026-03-27 09:14

- Project uses FastAPI with asyncpg — not SQLAlchemy [world] (2026-03-26)
- Preferred testing framework: pytest with pytest-asyncio [experience] (2026-03-26)
</hindsight_memories>
```

On `stop` (and again on `sessionEnd`), the hook reads the session transcript, strips previously injected memory tags (to prevent feedback loops), and POSTs the conversation to Hindsight asynchronously.

## Connection Modes

### 1. External API (recommended)

Connect to a running Hindsight server (cloud or self-hosted) via `~/.hindsight/cursor-cli.json`:

```json
{
  "hindsightApiUrl": "https://api.hindsight.vectorize.io",
  "hindsightApiToken": "hsk_your_token"
}
```

### 2. Local Daemon

Run `hindsight-embed` locally. The `session_start.py` hook detects it on `apiPort` (default `9077`). The daemon is not auto-started by the plugin — start it separately:

```bash
uvx hindsight-embed
```

Then leave `hindsightApiUrl` empty in your config and the plugin connects to `http://localhost:9077`.

## Configuration

Default config ships in `~/.cursor/hooks/cursor-cli/settings.json`. For personal overrides that survive updates, create `~/.hindsight/cursor-cli.json`. Most settings can also be overridden via environment variable.

**Loading order** (later entries win):

1. Built-in defaults
2. Plugin `settings.json` (at `~/.cursor/hooks/cursor-cli/settings.json`)
3. User config (`~/.hindsight/cursor-cli.json`)
4. Environment variables

---

### Connection

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `hindsightApiUrl` | `HINDSIGHT_API_URL` | `""` | URL of the Hindsight API server. Empty = local daemon. |
| `hindsightApiToken` | `HINDSIGHT_API_TOKEN` | `null` | API token for authentication. Required for Hindsight Cloud. |
| `apiPort` | `HINDSIGHT_API_PORT` | `9077` | Port for the local `hindsight-embed` daemon. |

---

### Memory Bank

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `bankId` | `HINDSIGHT_BANK_ID` | `"cursor-cli"` | The bank to read from and write to. All sessions share this bank unless `dynamicBankId` is enabled. |
| `bankMission` | `HINDSIGHT_BANK_MISSION` | coding assistant prompt | Describes the agent's purpose. Sent when creating or updating the bank. |
| `retainMission` | — | extraction prompt | Instructions for Hindsight's fact extraction — what to extract from coding conversations. |
| `dynamicBankId` | `HINDSIGHT_DYNAMIC_BANK_ID` | `false` | When `true`, derives a unique bank ID from `dynamicBankGranularity` fields — useful for per-project isolation. |
| `dynamicBankGranularity` | — | `["agent", "project"]` | Which fields to combine for dynamic bank IDs. `"project"` = working directory, `"agent"` = agent name. |
| `agentName` | `HINDSIGHT_AGENT_NAME` | `"cursor-cli"` | Agent name used in dynamic bank ID derivation. |

---

### Auto-Recall

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `autoRecall` | `HINDSIGHT_AUTO_RECALL` | `true` | Master switch for auto-recall. |
| `recallBudget` | `HINDSIGHT_RECALL_BUDGET` | `"mid"` | Search depth: `"low"` (fast), `"mid"` (balanced), `"high"` (thorough). |
| `recallMaxTokens` | `HINDSIGHT_RECALL_MAX_TOKENS` | `1024` | Max tokens in the recalled memory block. |
| `recallTypes` | — | `["world", "experience"]` | Memory types to retrieve. |
| `recallContextTurns` | `HINDSIGHT_RECALL_CONTEXT_TURNS` | `1` | Prior turns to include when building the recall query. `1` = latest prompt only. |

---

### Auto-Retain

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `autoRetain` | `HINDSIGHT_AUTO_RETAIN` | `true` | Master switch for auto-retain. |
| `retainMode` | `HINDSIGHT_RETAIN_MODE` | `"full-session"` | `"full-session"` sends the full transcript per session (upserted by session ID). `"chunked"` sends sliding windows every N turns. |
| `retainEveryNTurns` | — | `10` | Retain fires every N turns. `1` = every turn. Higher values reduce API calls. |
| `retainContext` | — | `"cursor-cli"` | Label identifying the source integration. Useful when multiple integrations write to the same bank. |

---

### Debug

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `debug` | `HINDSIGHT_DEBUG` | `false` | Enable verbose logging to stderr. All log lines are prefixed with `[Hindsight]`. |

## Per-Project Memory

To give each project its own isolated memory bank, enable dynamic bank IDs:

```json
{
  "dynamicBankId": true,
  "dynamicBankGranularity": ["agent", "project"]
}
```

With this config, running Cursor in `~/projects/api` and `~/projects/frontend` stores and recalls memories separately. Bank IDs are derived from the working directory path.

## Troubleshooting

**Hooks not firing**: Confirm `~/.cursor/hooks.json` exists and that `python3` is on your shell's `$PATH`. Re-run `hindsight-cursor-cli install` to rewrite the hook entries.

**No memories recalled**: Recall returns results only after something has been retained. Complete one Cursor session first, then start a new one.

**Memory not being stored**: `retainEveryNTurns` defaults to `10` — the `stop` hook only fires a retain every 10 turns. While testing, add `"retainEveryNTurns": 1` to `~/.hindsight/cursor-cli.json`. The `sessionEnd` hook also forces a final retain when you close the session.

**Debug mode**: Add `"debug": true` to `~/.hindsight/cursor-cli.json` to see what Hindsight is doing on each turn.
