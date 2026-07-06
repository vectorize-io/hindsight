---
sidebar_position: 40
title: "ZCode Persistent Memory with Hindsight | Integration Guide"
description: "Add persistent long-term memory to ZCode (Z.ai's GLM desktop coding agent) with Hindsight. Python hooks automatically recall context before each prompt and retain conversations — no MCP, no workflow changes."
---

# ZCode

Persistent memory for [ZCode](https://zcode.z.ai) — Z.ai's GLM desktop coding agent — using [Hindsight](https://vectorize.io/hindsight). ZCode embeds the Claude Code agent runtime, so Python hook scripts automatically recall relevant context before each prompt and retain conversations after each turn. No MCP server, no changes to your ZCode workflow.

## Quick Start

:::tip Recommended: Hindsight Cloud
[Sign up free](https://ui.hindsight.vectorize.io/signup) for a Hindsight Cloud API key — no self-hosting, no local daemon to manage.
:::

```bash
# Install the CLI
pip install hindsight-zcode

# Install the hooks (defaults to Hindsight Cloud)
hindsight-zcode install --api-url https://api.hindsight.vectorize.io --api-token your-api-key

# Restart ZCode — memory is live
```

The installer copies the hook scripts to `~/.zcode/hooks/hindsight/`, registers them in `~/.zcode/cli/config.json` (merged with any existing hooks), and creates `~/.hindsight/zcode.json` for your personal config. It never touches your Claude Code config at `~/.claude/settings.json`.

**Self-hosting alternative** — connect to a local `hindsight-embed` daemon by omitting the flags:

```bash
hindsight-zcode install
```

To uninstall:

```bash
hindsight-zcode uninstall
```

### Alternative: install as a ZCode plugin

ZCode can install Hindsight directly from a plugin marketplace — no `pip` step. The same hook scripts ship as a hooks-only Claude Code plugin (`hindsight-zcode`) in the Hindsight marketplace:

```
# In ZCode: add the Hindsight marketplace, then install the plugin
zcode plugins add-marketplace vectorize-io/hindsight
zcode plugins install hindsight-zcode
```

When installed this way, ZCode registers the hooks automatically (no config-file edit). Provide your Hindsight credentials via environment variables (`HINDSIGHT_API_URL`, `HINDSIGHT_API_TOKEN`) or by creating `~/.hindsight/zcode.json`:

```json
{
  "hindsightApiUrl": "https://api.hindsight.vectorize.io",
  "hindsightApiToken": "hsk_your_token"
}
```

## Features

- **Auto-recall** — before each prompt, queries Hindsight for relevant memories and injects them as additional context (visible to the model, not the transcript)
- **Auto-retain** — after each response, stores the turn to Hindsight for future recall
- **No MCP required** — plain Python hook scripts calling Hindsight's REST API; nothing to run alongside ZCode
- **Cross-tool memory** — the same Hindsight bank is shared across Claude Code, Cursor, and other Hindsight integrations, so memory follows you between tools
- **Dynamic bank IDs** — supports per-project memory isolation based on the working directory
- **Zero runtime dependencies** — the hook scripts are pure Python stdlib; the `pip install` only ships the one-time installer

## Architecture

ZCode embeds the Claude Code agent runtime and reads the standard Claude Code hook schema from its own config namespace, `~/.zcode/cli/config.json` (with `hooks.enabled: true`). The plugin wires three hook events:

| Hook | Event | Purpose |
|------|-------|---------|
| `session_start.py` | `SessionStart` | Warm up — verify Hindsight is reachable |
| `recall.py` | `UserPromptSubmit` | **Auto-recall** — query memories, inject as `additionalContext` |
| `retain.py` | `Stop` | **Auto-retain** — assemble the turn, POST to Hindsight |

On `UserPromptSubmit`, the hook reads the prompt, queries Hindsight for the most relevant memories, and emits a context block that ZCode injects before sending the turn to the model:

```
<hindsight_memories>
Relevant memories from past conversations...
Current time - 2026-03-27 09:14

- Project uses FastAPI with asyncpg — not SQLAlchemy [world] (2026-03-26)
- Preferred testing framework: pytest with pytest-asyncio [experience] (2026-03-26)
</hindsight_memories>
```

On `Stop`, the hook pairs the user prompt (captured at `UserPromptSubmit`) with the agent's response and POSTs the turn to Hindsight. ZCode does not provide a `SessionEnd` hook event, so retention rides `Stop` — every turn is stored as it completes.

## Connection Modes

### 1. External API (recommended)

Connect to a running Hindsight server (cloud or self-hosted) via `~/.hindsight/zcode.json`:

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

Default config ships in `~/.zcode/hooks/hindsight/settings.json`. For personal overrides that survive updates, create `~/.hindsight/zcode.json`. Most settings can also be overridden via environment variable.

**Loading order** (later entries win):

1. Built-in defaults
2. Plugin `settings.json` (at `~/.zcode/hooks/hindsight/settings.json`)
3. User config (`~/.hindsight/zcode.json`)
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
| `bankId` | `HINDSIGHT_BANK_ID` | `"zcode"` | The bank to read from and write to. All sessions share this bank unless `dynamicBankId` is enabled. |
| `bankMission` | `HINDSIGHT_BANK_MISSION` | coding assistant prompt | Describes the agent's purpose. Sent when creating or updating the bank. |
| `dynamicBankId` | `HINDSIGHT_DYNAMIC_BANK_ID` | `false` | When `true`, derives a unique bank ID from `dynamicBankGranularity` fields — useful for per-project isolation. |
| `agentName` | `HINDSIGHT_AGENT_NAME` | `"zcode"` | Agent name used in dynamic bank ID derivation. |

---

### Auto-Recall

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `autoRecall` | `HINDSIGHT_AUTO_RECALL` | `true` | Master switch for auto-recall. |
| `recallBudget` | `HINDSIGHT_RECALL_BUDGET` | `"mid"` | Search depth: `"low"` (fast), `"mid"` (balanced), `"high"` (thorough). |
| `recallMaxTokens` | `HINDSIGHT_RECALL_MAX_TOKENS` | `1024` | Token budget for the injected memory block. |

---

### Auto-Retain

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `autoRetain` | `HINDSIGHT_AUTO_RETAIN` | `true` | Master switch for auto-retain. |
| `retainEveryNTurns` | `HINDSIGHT_RETAIN_EVERY_N_TURNS` | `1` | Retain every N turns. Default `1` stores every turn on `Stop`. |

## Relationship to ZCode's built-in memory

ZCode ships its own local, per-project memory (`~/.zcode/cli/memories/`). Hindsight is complementary: it stores memory in a **cloud (or self-hosted) bank that is shared across tools** — the same bank powers Claude Code, Cursor, and other Hindsight integrations — so your context follows you between agents and machines rather than staying local to one ZCode project.
