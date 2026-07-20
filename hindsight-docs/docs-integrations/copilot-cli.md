---
sidebar_position: 7
title: "GitHub Copilot CLI Persistent Memory with Hindsight | Integration Guide"
description: "Add persistent memory to GitHub Copilot CLI with Hindsight. Python hook scripts automatically recall context at session start and retain conversations — no workflow changes required."
---

# GitHub Copilot CLI

[View Changelog →](/changelog/integrations/copilot-cli)

Persistent memory for [GitHub Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/use-hooks) using [Hindsight](https://vectorize.io/hindsight). Python hook scripts automatically recall relevant context at session start (and for every subagent Copilot CLI spawns) and retain conversations as they happen — no changes to your Copilot CLI workflow required.

## Quick Start

:::tip Recommended: Hindsight Cloud
[Sign up free](https://ui.hindsight.vectorize.io/signup) for a Hindsight Cloud API key — no self-hosting, no local daemon to manage.
:::

```bash
# Install the CLI
pip install hindsight-copilot-cli

# Install the hooks (defaults to Hindsight Cloud)
hindsight-copilot-cli install --api-url https://api.hindsight.vectorize.io --api-token your-api-key

# Restart Copilot CLI — memory is live
```

The installer copies the hook scripts to `~/.copilot/hindsight-copilot-cli/`, writes a standalone `~/.copilot/hooks/hindsight-copilot-cli.json` (Copilot CLI loads every `*.json` file in its hooks directory, so this never needs to merge with other tools' hook files), and creates `~/.hindsight/copilot-cli.json` for your personal config.

**Self-hosting alternative** — connect to a local `hindsight-embed` daemon by omitting the flags:

```bash
hindsight-copilot-cli install
```

To uninstall:

```bash
hindsight-copilot-cli uninstall
```

### Repo-level install

For a team-shared configuration checked into version control, register the hooks at the repository level instead:

```bash
hindsight-copilot-cli install --scope repo
```

This writes `.github/hooks/hindsight-copilot-cli.json` in the current repo. Hook scripts are still installed to `~/.copilot/hindsight-copilot-cli/` (per-machine), so the token must come from the `HINDSIGHT_API_TOKEN` environment variable rather than a committed file.

## Features

- **Auto-recall at session start** — when a Copilot CLI session starts, queries Hindsight for relevant memories and injects them as additional context. Uses the queued `initialPrompt` when one exists, or falls back to a generic project-context query derived from the working directory.
- **Auto-recall for subagents** — Copilot CLI's built-in subagents (`explore`, `task`, `research`, `code-review`, `rubber-duck`, `security-review`, and custom agents) run in their own isolated context and otherwise get none of the memory injected at session start. The `subagentStart` hook gives them baseline project memory too.
- **Auto-retain** — after each turn (subject to `retainEveryNTurns`), and again on session end, stores the conversation to Hindsight for future recall.
- **Dynamic bank IDs** — supports per-project memory isolation based on the working directory.
- **Session-level upsert** — uses the session ID as the document ID so re-running the same session updates rather than duplicates stored content.
- **Zero runtime dependencies** — the hook scripts are pure Python stdlib; the `pip install` only ships the one-time installer.

## Architecture

The plugin uses four Copilot CLI hook events:

| Hook | Event | Purpose |
|------|-------|---------|
| `session_start.py` | `sessionStart` | **Auto-recall** — query memories using the initial prompt (or a project-context fallback), inject as additional context |
| `subagent_start.py` | `subagentStart` | **Auto-recall for subagents** — query memories using a project-context fallback query (subagent payloads carry no task text), inject as additional context |
| `agent_stop.py` | `agentStop` | **Auto-retain** — read the transcript, apply the `retainEveryNTurns` cadence, POST to Hindsight (async); caches the transcript path for `sessionEnd` |
| `session_end.py` | `sessionEnd` | **Final flush** — force a retain using the transcript path cached by the last `agentStop`, since `sessionEnd`'s payload has no transcript path of its own |

On `sessionStart`, the hook queries Hindsight for the most relevant memories and injects a context block that Copilot CLI adds to the session:

```
<hindsight_memories>
Relevant memories from past conversations...
Current time - 2026-03-27 09:14

- Project uses FastAPI with asyncpg — not SQLAlchemy [world] (2026-03-26)
- Preferred testing framework: pytest with pytest-asyncio [experience] (2026-03-26)
</hindsight_memories>
```

`subagentStart` fires for every subagent Copilot CLI spawns except the built-in `general-purpose` agent (which never emits the event) and injects the same style of context block, prepended to that subagent's prompt.

On `agentStop` (and again on `sessionEnd`), the hook reads the session transcript, strips previously injected memory tags (to prevent feedback loops), and POSTs the conversation to Hindsight asynchronously.

:::note Recall is not refreshed per turn
Unlike some other integrations, Copilot CLI's `userPromptSubmitted` and `preToolUse` hooks don't support injecting additional context, so recall only runs once per session (at `sessionStart`) and once per subagent (at `subagentStart`) rather than being refreshed on every turn. This keeps the integration simple and reliable, at the cost of not picking up new memories mid-session.
:::

## Connection Modes

### 1. External API (recommended)

Connect to a running Hindsight server (cloud or self-hosted) via `~/.hindsight/copilot-cli.json`:

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

Default config ships in `~/.copilot/hindsight-copilot-cli/settings.json`. For personal overrides that survive updates, create `~/.hindsight/copilot-cli.json`. Most settings can also be overridden via environment variable.

**Loading order** (later entries win):

1. Built-in defaults
2. Plugin `settings.json` (at `~/.copilot/hindsight-copilot-cli/settings.json`)
3. User config (`~/.hindsight/copilot-cli.json`)
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
| `bankId` | `HINDSIGHT_BANK_ID` | `"copilot-cli"` | The bank to read from and write to. All sessions share this bank unless `dynamicBankId` is enabled. |
| `bankMission` | `HINDSIGHT_BANK_MISSION` | coding assistant prompt | Describes the agent's purpose. Sent when creating or updating the bank. |
| `retainMission` | — | extraction prompt | Instructions for Hindsight's fact extraction — what to extract from coding conversations. |
| `dynamicBankId` | `HINDSIGHT_DYNAMIC_BANK_ID` | `false` | When `true`, derives a unique bank ID from `dynamicBankGranularity` fields — useful for per-project isolation. |
| `dynamicBankGranularity` | — | `["agent", "project"]` | Which fields to combine for dynamic bank IDs. `"project"` = working directory, `"agent"` = agent name. |
| `agentName` | `HINDSIGHT_AGENT_NAME` | `"copilot-cli"` | Agent name used in dynamic bank ID derivation. |

---

### Auto-Recall

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `autoRecall` | `HINDSIGHT_AUTO_RECALL` | `true` | Master switch for auto-recall (covers both `sessionStart` and `subagentStart`). |
| `recallBudget` | `HINDSIGHT_RECALL_BUDGET` | `"mid"` | Search depth: `"low"` (fast), `"mid"` (balanced), `"high"` (thorough). |
| `recallMaxTokens` | `HINDSIGHT_RECALL_MAX_TOKENS` | `1024` | Max tokens in the recalled memory block. |
| `recallTypes` | — | `["world", "experience"]` | Memory types to retrieve. |
| `recallFallbackQueryTemplate` | — | project-context prompt | Query used when there's no specific prompt/task text available (interactive `sessionStart` with no queued prompt, or any `subagentStart`). |

---

### Auto-Retain

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `autoRetain` | `HINDSIGHT_AUTO_RETAIN` | `true` | Master switch for auto-retain. |
| `retainMode` | `HINDSIGHT_RETAIN_MODE` | `"full-session"` | `"full-session"` sends the full transcript per session (upserted by session ID). `"chunked"` sends sliding windows every N turns. |
| `retainEveryNTurns` | — | `10` | Retain fires every N turns. `1` = every turn. Higher values reduce API calls. |
| `retainContext` | — | `"copilot-cli"` | Label identifying the source integration. Useful when multiple integrations write to the same bank. |

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

With this config, running Copilot CLI in `~/projects/api` and `~/projects/frontend` stores and recalls memories separately. Bank IDs are derived from the working directory path.

## Troubleshooting

**Hooks not firing**: check that `~/.copilot/hooks/hindsight-copilot-cli.json` (or `.github/hooks/hindsight-copilot-cli.json` for repo scope) is valid JSON. Copilot CLI loads hook configuration when it starts, so restart the CLI after installing or updating hooks.

**No memories recalled**: Recall returns results only after something has been retained. Complete one Copilot CLI session first, then start a new one.

**Memory not being stored**: `retainEveryNTurns` defaults to `10` — the `agentStop` hook only fires a retain every 10 turns. While testing, add `"retainEveryNTurns": 1` to `~/.hindsight/copilot-cli.json`. The `sessionEnd` hook also forces a final retain when the session terminates.

**Debug mode**: Add `"debug": true` to `~/.hindsight/copilot-cli.json` to see what Hindsight is doing on each turn.
