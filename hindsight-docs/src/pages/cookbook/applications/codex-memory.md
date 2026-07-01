---
sidebar_position: 9
---

# Codex Memory


:::info Complete Application
This is a complete, runnable application demonstrating Hindsight integration.
[**View source on GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/tree/main/applications/codex-memory)
:::


Give [OpenAI Codex CLI](https://github.com/openai/codex) persistent memory across sessions using Hindsight. Three Python hook scripts automatically recall relevant context before each prompt and store conversations after each turn — no changes to your Codex workflow required.

## What This Demonstrates

- **Automatic recall** — Hindsight injects relevant memories before every Codex prompt via `UserPromptSubmit` hook
- **Session retention** — conversations are stored to Hindsight after each turn via `Stop` hook
- **Cross-session continuity** — Codex remembers project context, preferences, and decisions between separate sessions
- **Per-project isolation** — optional dynamic bank IDs keep memory scoped to each repository
- **Zero workflow changes** — install once; memory works silently in the background

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   OpenAI Codex CLI                  │
│                                                     │
│  SessionStart ──→ session_start.py ──→ warm up     │
│                                                     │
│  UserPromptSubmit                                   │
│      │                                              │
│      ▼                                              │
│  recall.py ──→ Hindsight API ──→ inject memories   │
│      │              │                               │
│      │         [relevant facts from past sessions]  │
│      ▼                                              │
│  Codex sees prompt + memory context                 │
│      │                                              │
│      ▼ (after response)                             │
│  Stop hook                                          │
│      │                                              │
│      ▼                                              │
│  retain.py ──→ Hindsight API ──→ store session     │
└─────────────────────────────────────────────────────┘
```

## Prerequisites

1. **OpenAI Codex CLI v0.116.0+** — hooks were added in this release
   ```bash
   npm install -g @openai/codex
   codex --version
   ```

2. **Python 3.9+**
   ```bash
   python3 --version
   ```

3. **Hindsight** — cloud or local
   - [Hindsight Cloud](https://hindsight.vectorize.io) (sign up for an API key)
   - Or run locally: `docker run -p 9077:9077 ghcr.io/vectorize-io/hindsight:latest`

## Quick Start

### 1. Install the Hindsight Codex plugin

```bash
git clone https://github.com/vectorize-io/hindsight.git
cd hindsight/hindsight-integrations/codex
./install.sh
```

This copies hook scripts to `~/.hindsight/codex/scripts/`, writes `~/.codex/hooks.json`, and enables `codex_hooks = true` in `~/.codex/config.toml`.

### 2. Configure your connection

Create `~/.hindsight/codex.json`:

```json
{
  "hindsightApiUrl": "https://api.hindsight.vectorize.io",
  "hindsightApiToken": "hsk_your_token_here",
  "bankId": "codex"
}
```

For a local Hindsight instance, set `"hindsightApiUrl": "http://localhost:9077"` and omit the token.

### 3. (Optional) Seed your memory bank

To see recall working immediately without waiting for sessions to build up:

```bash
pip install -r requirements.txt

# Point to your Hindsight instance
export HINDSIGHT_URL=https://api.hindsight.vectorize.io
export HINDSIGHT_API_KEY=hsk_your_token_here

python seed_memory.py
```

### 4. Start Codex

```bash
codex
```

Ask something like `what testing framework do I prefer?` — Hindsight will inject the relevant context before your prompt reaches the model.

## How It Works

### SessionStart hook — `session_start.py`

Runs when Codex starts. In daemon mode, this warms up the local Hindsight server so the first recall doesn't have a cold-start delay. In cloud mode, it verifies the connection is reachable.

### UserPromptSubmit hook — `recall.py`

Runs before every user message. Queries Hindsight with the current prompt (and optionally prior context turns), retrieves the most relevant memories, and injects them as `additionalContext`:

```
<hindsight_memories>
Relevant memories from past conversations (prioritize recent when conflicting)...
Current time - 2026-03-27 18:00

- User prefers pytest for Python testing [experience] (2026-03-20)
- Current project uses FastAPI + PostgreSQL [world] (2026-03-25)
</hindsight_memories>
```

Codex sees this context as part of the conversation before generating a response.

### Stop hook — `retain.py`

Runs after every Codex response. Reads the session transcript, strips previously injected memory tags (to prevent feedback loops), and POSTs the conversation to Hindsight. The session ID is used as the document ID, so re-running the same session upserts rather than duplicates.

## Configuration

Edit `~/.hindsight/codex.json` to override defaults:

| Key | Default | Description |
|-----|---------|-------------|
| `hindsightApiUrl` | `""` | Hindsight server URL (required) |
| `hindsightApiToken` | `null` | API token for Hindsight Cloud |
| `bankId` | `"codex"` | Memory bank name |
| `retainEveryNTurns` | `10` | Store memory every N turns (use `1` while testing) |
| `autoRecall` | `true` | Inject memories before each prompt |
| `autoRetain` | `true` | Store conversations after each turn |
| `dynamicBankId` | `false` | Use separate bank per project/directory |
| `dynamicBankGranularity` | `["agent", "project"]` | How to partition dynamic banks |
| `recallBudget` | `"mid"` | Token budget for recall: `"low"`, `"mid"`, `"high"` |
| `debug` | `false` | Print debug output to stderr |

Environment variable overrides are also supported: `HINDSIGHT_API_URL`, `HINDSIGHT_API_TOKEN`, `HINDSIGHT_BANK_ID`, `HINDSIGHT_DEBUG`.

## Core Files

| File | Description |
|------|-------------|
| `seed_memory.py` | Seeds sample developer facts for demo purposes |
| `requirements.txt` | Python dependencies for seed script |
| `~/.hindsight/codex/scripts/recall.py` | UserPromptSubmit hook — retrieves and injects memories |
| `~/.hindsight/codex/scripts/retain.py` | Stop hook — stores conversation to Hindsight |
| `~/.hindsight/codex/scripts/session_start.py` | SessionStart hook — warms up server |
| `~/.hindsight/codex.json` | Your personal config overrides |

## Common Issues

**Hooks not firing**

Check that `~/.codex/config.toml` contains:
```toml
[features]
codex_hooks = true
```

Re-run `install.sh` to write this automatically.

**No memories recalled on the first session**

Recall only returns results after something has been retained. Either:
- Run `seed_memory.py` to pre-populate facts, or
- Complete one session (Codex responds to at least one message) and then start a new session

**Memory not being stored**

`retainEveryNTurns` defaults to 10 — retain only fires every 10 turns. While testing, add `"retainEveryNTurns": 1` to `~/.hindsight/codex.json`.

**Debug mode**

Add `"debug": true` to `~/.hindsight/codex.json` to see what Hindsight is doing on each turn:
```
[Hindsight] Recalling from bank 'codex', query length: 42
[Hindsight] Injecting 3 memories
[Hindsight] Retaining to bank 'codex', doc 'sess-abc123', 2 messages, 847 chars
```

## Built With

- [OpenAI Codex CLI](https://github.com/openai/codex) — AI coding agent
- [Hindsight](https://hindsight.vectorize.io) — long-term memory for AI agents
- [Hindsight Codex plugin](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/codex) — hook scripts and installer
