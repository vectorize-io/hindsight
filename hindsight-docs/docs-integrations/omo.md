---
sidebar_position: 35
title: "OMO (oh-my-openagent) Persistent Memory with Hindsight | Integration Guide"
description: "Add persistent memory to oh-my-openagent (OMO) with Hindsight. Five lifecycle hooks automatically recall context before each prompt and retain conversations — no workflow changes required."
---

# OMO (oh-my-openagent)

[View Changelog →](/changelog/integrations/omo)

Persistent memory for [oh-my-openagent](https://github.com/code-yeongyu/oh-my-openagent) (OMO) using [Hindsight](https://vectorize.io/hindsight). Five lifecycle hooks automatically recall relevant context before each prompt and retain conversations after each turn — no changes to your OMO workflow required.

## Quick Start

:::tip Recommended: Hindsight Cloud
[Sign up free](https://ui.hindsight.vectorize.io/signup) for a Hindsight Cloud API key — no self-hosting, no local daemon to manage.
:::

From the `hindsight-integrations/omo/` directory:

```bash
# Hooks (global)
mkdir -p ~/.omo/hooks
cp hooks/hooks.json ~/.omo/hooks/hindsight-hooks.json

# Scripts + settings (global)
mkdir -p ~/.omo/plugins/hindsight/scripts
cp -r scripts/ ~/.omo/plugins/hindsight/scripts/
cp settings.json ~/.omo/plugins/hindsight/settings.json

# Rules (per-project — run from your project root)
mkdir -p .omo/rules
cp rules/hindsight-memory.md .omo/rules/hindsight-memory.md
```

Then set your API key:

```bash
export HINDSIGHT_API_TOKEN=hsk_your_key_here
```

Add env vars to OMO's allowlist in `~/.config/opencode/oh-my-openagent.jsonc`:

```jsonc
{
  "mcp_env_allowlist": [
    "HINDSIGHT_API_URL",
    "HINDSIGHT_API_TOKEN",
    "HINDSIGHT_BANK_ID"
  ]
}
```

Start a new OMO session — memory is live.

## Features

- **Auto-recall** — on every user prompt, queries Hindsight for relevant memories and injects them as `additionalContext` (invisible to the transcript, visible to OMO)
- **Auto-retain** — after each OMO response and sub-agent stop, stores the conversation transcript to Hindsight for future recall
- **Sub-agent capture** — `SubagentStop` hook captures learnings from OMO's delegated sub-agents (Claude Code, Codex, OpenCode)
- **Dynamic bank IDs** — supports per-project memory isolation based on the working directory
- **Session-level upsert** — uses the session ID as the document ID so re-running the same session updates rather than duplicates stored content
- **Cloud-first** — defaults to Hindsight Cloud with no local daemon to manage
- **Zero dependencies** — pure Python stdlib, no pip install required

## Architecture

The plugin uses five OMO hook events:

| Hook | Event | Purpose |
|------|-------|---------|
| `session_start.py` | `SessionStart` | Warm up — verify Hindsight is reachable |
| `recall.py` | `UserPromptSubmit` | **Auto-recall** — query memories, inject as `additionalContext` |
| `retain.py` | `Stop` | **Auto-retain** — extract transcript, POST to Hindsight (async) |
| `retain.py` | `SubagentStop` | **Sub-agent retain** — capture delegated sub-agent learnings |
| `session_end.py` | `SessionEnd` | Force final retain for short sessions |

On `UserPromptSubmit`, the hook reads the prompt, queries Hindsight for the most relevant memories, and outputs a `hookSpecificOutput.additionalContext` block. OMO prepends this to the conversation before sending it to the model:

```
<hindsight_memories>
Relevant memories from past conversations...
Current time - 2026-06-08 14:30

- Project uses FastAPI with asyncpg — not SQLAlchemy [world] (2026-06-07)
- Preferred testing framework: pytest with pytest-asyncio [experience] (2026-06-07)
</hindsight_memories>
```

On `Stop` and `SubagentStop`, the hook reads the session transcript, strips previously injected memory tags (to prevent feedback loops), and POSTs the conversation to Hindsight asynchronously.

## Connection Modes

### 1. Hindsight Cloud (recommended)

No setup beyond an API key:

```json
{
  "hindsightApiUrl": "https://api.hindsight.vectorize.io",
  "hindsightApiToken": "hsk_your_token"
}
```

### 2. Self-Hosted

Point at your own Hindsight instance:

```bash
export HINDSIGHT_API_URL=http://localhost:8888
```

Or in `~/.hindsight/omo.json`:

```json
{
  "hindsightApiUrl": "http://localhost:8888"
}
```

No API token is required for local instances.

## Configuration

Settings are loaded from three sources in order (later wins):

1. Plugin `settings.json` (cloud URL pre-set)
2. User config (`~/.hindsight/omo.json`)
3. `HINDSIGHT_*` environment variables

---

### Connection

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `hindsightApiUrl` | `HINDSIGHT_API_URL` | `https://api.hindsight.vectorize.io` | API endpoint. |
| `hindsightApiToken` | `HINDSIGHT_API_TOKEN` | — | API key for authentication. Required for Hindsight Cloud. |

---

### Memory Bank

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `bankId` | `HINDSIGHT_BANK_ID` | `"omo"` | The bank to read from and write to. All sessions share this bank unless `dynamicBankId` is enabled. |
| `bankMission` | `HINDSIGHT_BANK_MISSION` | OMO orchestrator prompt | Describes the agent's purpose. Sent when creating or updating the bank. |
| `retainMission` | — | extraction prompt | Instructions for Hindsight's fact extraction. |
| `dynamicBankId` | `HINDSIGHT_DYNAMIC_BANK_ID` | `false` | When `true`, derives a unique bank ID from `dynamicBankGranularity` fields. |
| `dynamicBankGranularity` | — | `["agent", "project"]` | Fields to combine for dynamic bank IDs. `"project"` = working directory, `"agent"` = agent name. |
| `bankIdPrefix` | — | `""` | Prefix prepended to all bank IDs. |

---

### Auto-Recall

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `autoRecall` | `HINDSIGHT_AUTO_RECALL` | `true` | Master switch for auto-recall. |
| `recallBudget` | `HINDSIGHT_RECALL_BUDGET` | `"mid"` | Search depth: `"low"` (fast), `"mid"` (balanced), `"high"` (thorough). |
| `recallMaxTokens` | `HINDSIGHT_RECALL_MAX_TOKENS` | `1024` | Max tokens in the recalled memory block. |
| `recallTypes` | — | `["world", "experience"]` | Memory types to retrieve. |
| `recallContextTurns` | `HINDSIGHT_RECALL_CONTEXT_TURNS` | `1` | Prior turns to include when building the recall query. `1` = latest prompt only. |
| `recallMaxQueryChars` | `HINDSIGHT_RECALL_MAX_QUERY_CHARS` | `800` | Max characters in the query sent to Hindsight. |
| `recallRoles` | — | `["user", "assistant"]` | Roles to include when building a multi-turn query. |
| `recallPromptPreamble` | — | built-in | Text placed above the recalled memories in the injected context block. |

---

### Auto-Retain

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `autoRetain` | `HINDSIGHT_AUTO_RETAIN` | `true` | Master switch for auto-retain. |
| `retainMode` | `HINDSIGHT_RETAIN_MODE` | `"full-session"` | `"full-session"` sends the full transcript per session. `"chunked"` sends sliding windows every N turns. |
| `retainEveryNTurns` | — | `10` | Retain fires every N turns. `1` = every turn. Higher values reduce API calls. |
| `retainOverlapTurns` | — | `2` | Extra turns included from the previous chunk (chunked mode only). |
| `retainRoles` | — | `["user", "assistant"]` | Roles to include in the retained transcript. |
| `retainTags` | — | `["{session_id}"]` | Tags attached to the stored document. `{session_id}` is replaced at runtime. |
| `retainMetadata` | — | `{}` | Arbitrary key-value metadata attached to the stored document. |
| `retainContext` | — | `"omo"` | Label identifying the source integration. |

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

With this config, running OMO in `~/projects/api` and `~/projects/frontend` stores and recalls memories separately. Bank IDs are derived from the working directory path (e.g. `omo::api`, `omo::frontend`).

## Multi-Bank Recall

Query additional banks alongside the primary one:

```json
{
  "recallAdditionalBanks": ["shared-team-knowledge"]
}
```

## Troubleshooting

**No memories recalled**: Recall returns results only after something has been retained. Complete one OMO session first, then start a new one to see memories.

**Memory not being stored**: `retainEveryNTurns` defaults to `10` — retain only fires every 10 turns. While testing, add `"retainEveryNTurns": 1` to `~/.hindsight/omo.json`.

**Cloud mode silently skipping**: If `hindsightApiUrl` points to Hindsight Cloud but no `hindsightApiToken` is set, hooks silently skip. Set `HINDSIGHT_API_TOKEN` or add it to `~/.hindsight/omo.json`.

**Debug mode**: Add `"debug": true` to `~/.hindsight/omo.json` to see what Hindsight is doing on each turn:

```
[Hindsight] Recalling from bank 'omo', query length: 42
[Hindsight] Injecting 3 memories
[Hindsight] Retaining to bank 'omo', doc 'sess-abc123', 2 messages, 847 chars
```
