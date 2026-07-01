---
sidebar_position: 8
---

# Claude Code Memory


:::info Complete Application
This is a complete, runnable application demonstrating Hindsight integration.
[**View source on GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/tree/main/applications/claude-code-memory)
:::


Give [Claude Code](https://docs.anthropic.com/en/docs/claude-code) persistent memory across sessions using the Hindsight memory plugin. Conversations are captured automatically; relevant context is recalled and injected before every prompt — no changes to your Claude Code workflow required.

## What This Demonstrates

- **Auto-recall** — every user prompt triggers a `UserPromptSubmit` hook that queries Hindsight and injects matching memories as `additionalContext`. Claude sees the memories; the user doesn't see them in the transcript.
- **Auto-retain** — every Claude response triggers a `Stop` hook that POSTs the conversation to Hindsight. Memories are extracted asynchronously.
- **Cross-session continuity** — Claude Code remembers project decisions, preferences, and prior conversations between separate sessions.
- **Per-project isolation** — optional dynamic bank IDs scope memory to each repository.
- **Zero install friction** — `claude plugin install` does the wiring. No git clone, no editing config files manually.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Claude Code                        │
│                                                     │
│  SessionStart ──→ session_start.py ──→ warm up     │
│                                                     │
│  UserPromptSubmit                                   │
│      │                                              │
│      ▼                                              │
│  recall.py ──→ Hindsight ──→ inject as             │
│      │            │            additionalContext    │
│      │       [relevant facts from past sessions]    │
│      ▼                                              │
│  Claude sees prompt + memory context                │
│      │                                              │
│      ▼ (after response)                             │
│  Stop hook                                          │
│      │                                              │
│      ▼                                              │
│  retain.py ──→ Hindsight ──→ store conversation    │
│                                                     │
│  SessionEnd ──→ session_end.py ──→ cleanup         │
└─────────────────────────────────────────────────────┘
```

Memory is stored against the bank ID configured in `~/.hindsight/claude-code.json` (defaults to `claude-code`). Across sessions, every prompt sees the cumulative context.

## Prerequisites

1. **Claude Code** — install from [the Claude Code docs](https://docs.anthropic.com/en/docs/claude-code).

2. **Hindsight** — pick one:

   - [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) (free tier available, no self-hosting)
   - Self-hosted: `pip install hindsight-all && export HINDSIGHT_API_LLM_API_KEY=your-openai-key && hindsight-api`

3. **An LLM key for memory extraction** — `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in your environment. Or use the no-key option below.

## Quick Start

### 1. Install the Hindsight plugin

```bash
claude plugin marketplace add vectorize-io/hindsight
claude plugin install hindsight-memory
```

### 2. Configure your LLM provider for memory extraction

Pick one:

```bash
# Option A: OpenAI (auto-detected)
export OPENAI_API_KEY="sk-your-key"

# Option B: Anthropic (auto-detected)
export ANTHROPIC_API_KEY="your-key"

# Option C: No key — use Claude Code's own model (personal/local use only)
export HINDSIGHT_LLM_PROVIDER=claude-code
```

### 3. (Optional) Connect to Hindsight Cloud

If you want memories stored in the cloud rather than a local daemon, create `~/.hindsight/claude-code.json`:

```json
{
  "hindsightApiUrl": "https://api.hindsight.vectorize.io",
  "hindsightApiToken": "hsk_your_token_here",
  "bankId": "claude-code"
}
```

Skip this step to use the auto-managed local daemon.

### 4. (Optional) Seed the bank so recall works on day one

```bash
cd applications/claude-code-memory
pip install -r requirements.txt

# For Hindsight Cloud:
export HINDSIGHT_URL=https://api.hindsight.vectorize.io
export HINDSIGHT_API_KEY=hsk_your_token_here

# Or for a local daemon:
export HINDSIGHT_URL=http://localhost:8888

python seed_memory.py
```

### 5. Start Claude Code

```bash
claude
```

Try a prompt like:

> What testing framework do I prefer?

If you ran the seed script, Hindsight will inject your seeded preferences ahead of the prompt and Claude will answer from that context. Otherwise, ask Claude something memorable in one session and confirm it remembers in the next.

## How It Works

### `UserPromptSubmit` hook → `recall.py`

Runs before every prompt reaches the model. Queries Hindsight with the current message, retrieves the most relevant memories, and injects them as `additionalContext`:

```
<hindsight_memories>
Relevant memories from past conversations (prioritize recent when conflicting)...
Current time - 2026-04-29 14:00

- User prefers pytest for Python testing [experience] (2026-04-15)
- Current project uses FastAPI + PostgreSQL [world] (2026-04-20)
</hindsight_memories>
```

Claude treats this as part of the conversation context. The memory block is stripped from the transcript before retain to prevent feedback loops.

### `Stop` hook → `retain.py`

Runs after every response (async, non-blocking). Reads Claude Code's session JSONL transcript, applies chunked retention with a sliding window, and POSTs to Hindsight. The session ID is the document ID, so re-running the same session upserts rather than duplicates.

### `SessionStart` / `SessionEnd` hooks

Health-check at session start (warming the local daemon if used). Cleanup at session end.

## Per-Project Memory

Override the bank ID per-project so each repo gets isolated memory:

```bash
# In a project directory:
mkdir -p .hindsight
echo '{"bankId": "claude-code-myproject"}' > .hindsight/claude-code.json
```

Or set the env var per session:

```bash
HINDSIGHT_BANK_ID=claude-code-myproject claude
```

## Core Files

| File | Description |
|------|-------------|
| `seed_memory.py` | Pre-populates a bank with sample developer facts so recall works immediately |
| `requirements.txt` | Just `hindsight-client` for the seed script |

The actual memory plugin is installed via the Claude Code marketplace — this directory is just the cookbook helper around it.

## Verifying Memory

After a few sessions, query the bank directly:

```bash
curl -s "$HINDSIGHT_URL/v1/default/banks/claude-code/memories/list" \
  -H "Authorization: Bearer $HINDSIGHT_API_KEY"
```

Or open the bank in the [Hindsight Cloud UI](https://ui.hindsight.vectorize.io).

## Common Issues

**Recall returns nothing on the first session**
- The bank has no memories yet. Use `seed_memory.py` for an instant demo, or run a few real sessions and let `Stop` populate it.

**"OPENAI_API_KEY not set" / "ANTHROPIC_API_KEY not set"**
- The local daemon needs an LLM key for fact extraction. Export one of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or set `HINDSIGHT_LLM_PROVIDER=claude-code` to use Claude Code's own model.

**Plugin not activating**
- Confirm `claude plugin list` shows `hindsight-memory` enabled.
- Check `~/.hindsight/claude-code.json` is valid JSON if you created one.

---

**Built with:**
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) — Anthropic's official CLI agent
- [hindsight-claude-code plugin](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/claude-code) — Hindsight memory hooks for Claude Code
- [Hindsight](https://github.com/vectorize-io/hindsight) — Long-term memory for AI agents
