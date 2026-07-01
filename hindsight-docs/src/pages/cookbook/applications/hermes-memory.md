---
sidebar_position: 13
---

# Hermes + Hindsight Memory


:::info Complete Application
This is a complete, runnable application demonstrating Hindsight integration.
[**View source on GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/tree/main/applications/hermes-memory)
:::


Give your Hermes agent persistent cross-session memory. Hermes ships a built-in Hindsight memory provider that automatically captures every conversation and recalls relevant context before each response — one setup wizard and you're done.

## What This Demonstrates

- **Three memory modes** — `hybrid` (auto-recall + explicit tools, default), `context` (auto-recall only, clean tool surface), `tools` (explicit tools only, model decides when to recall)
- **Automatic retention** — every conversation is stored after each turn via the `post_llm_call` hook
- **Automatic recall** — relevant memories are injected before each response via the `pre_llm_call` hook
- **Cloud and local backends** — Hindsight Cloud (managed) or local embedded daemon
- **Coherence control** — `prefetch_method`: `recall` (fast, raw facts) or `reflect` (LLM synthesizes coherent memory summaries first)

## Architecture

```
You → hermes (chat)
       │
       Hindsight memory provider (built-in)
       │
       ├─ pre_llm_call hook ──→ auto-recall ──→ inject relevant memories
       │                                              │
       │                                        Hindsight API
       │                                              │
       └─ post_llm_call hook ──→ auto-retain ──→ store conversation
```

The Hindsight memory provider is registered as part of Hermes's hook system. Every turn automatically retains and recalls.

## Prerequisites

1. **Hermes installed** — https://github.com/NousResearch/hermes-agent

2. **A Hindsight backend** — pick one:

   | Option | Setup |
   |--------|-------|
   | **Hindsight Cloud** (easiest) | [Sign up](https://ui.hindsight.vectorize.io/signup) for a free `hsk_...` token |
   | **Local embedded** (no external services) | Just an LLM API key (OpenAI, Anthropic, Groq, etc.) |

## Quick Start

**Step 1: Configure Hindsight memory**

```bash
hermes memory setup
# Select "hindsight" when prompted
# Choose "cloud" for Hindsight Cloud (easiest)
# Or "local" to run an embedded daemon
```

**Step 2: Disable Hermes's built-in memory tool** (important!)

Hermes ships with a markdown-based memory tool. Without disabling it, the LLM often prefers it over Hindsight:

```bash
hermes tools disable memory
```

**Step 3: Verify connectivity**

```bash
hermes memory status
```

Should show:
```
✓ Hindsight memory provider is connected and healthy
```

**Step 4: Start chatting**

```bash
hermes
```

Memory is active. Conversations are retained automatically.

Alternatively, skip the wizard and use a config file directly:

```bash
cp configs/hermes-cloud.example.json ~/.hermes/hindsight/config.json
# edit to fill in api_key
hermes memory status
```

## Try It Out

**Session 1:**
```
You: I prefer functional programming and I'm working on a Rust CLI tool
Hermes: Got it — happy to help. What's the first piece you're working on?
```

**Exit Hermes, restart it, then Session 2:**
```
You: What do you know about my current project?
Hermes: You're building a Rust CLI tool and you prefer functional programming.
```

Memory persists across sessions without any code changes.

## Memory Modes

The key differentiator for Hermes is choosing how memory and tools interact:

### `hybrid` (default)

Auto-recall injects memories before every turn, AND the three explicit tools (`hindsight_retain`, `hindsight_recall`, `hindsight_reflect`) are available to the model.

**Best for:** Most users, internal assistants, anyone evaluating Hindsight. The model has both automatic context and explicit control.

```json
{
  "memory_mode": "hybrid",
  "prefetch_method": "recall"
}
```

### `context`

Auto-recall only. Memories are injected automatically, but no Hindsight tools are exposed to the model.

**Best for:** Consumer-facing or customer-facing assistants where you want invisible personalization without visible tool clutter. Clean, minimal tool surface.

```json
{
  "memory_mode": "context",
  "prefetch_method": "recall"
}
```

### `tools`

Explicit tools only. No automatic injection. The model must call `hindsight_recall` or `hindsight_reflect` deliberately.

**Best for:** Agents that should reason about *when* to use memory, or when you want tighter prompt control and explicit tool call logging. Tradeoff: the model may not use memory if it's not well-prompted.

```json
{
  "memory_mode": "tools"
}
```

| Mode | Auto-recall | Explicit tools | Best for |
|------|-------------|---|---|
| `hybrid` | ✓ | ✓ | Default, most users |
| `context` | ✓ | — | Clean tool surface, consumer agents |
| `tools` | — | ✓ | Explicit control, reasoning about when to recall |

## Prefetch Method

Controls how memories are prepared before injection (applies to `hybrid` and `context` modes only):

### `recall` (default)

Injects raw retrieved memory facts directly. Fast, minimal latency.

```json
{
  "prefetch_method": "recall"
}
```

### `reflect`

LLM synthesizes a coherent summary across relevant memories before injecting. Slower (2-3s per turn), but better for:
- Complex planning or open-ended reasoning
- When memory coherence matters more than speed
- Agents that need a "mental model" rather than raw facts

```json
{
  "prefetch_method": "reflect"
}
```

**Example:** An assistant planning a multi-month project. With `reflect`, memories are synthesized into a coherent project roadmap before each response. With `recall`, raw facts are injected and the model must synthesize them on the spot.

## Configuration Reference

| Key | Default | Mode | Description |
|-----|---------|------|-------------|
| `mode` | `cloud` | all | `cloud` or `local` |
| `api_url` | `https://api.hindsight.vectorize.io` | cloud | Hindsight Cloud URL |
| `api_key` | — | cloud | `hsk_...` token from https://ui.hindsight.vectorize.io |
| `bank_id` | `hermes` | all | Memory bank name |
| `memory_mode` | `hybrid` | all | `hybrid`, `context`, or `tools` |
| `prefetch_method` | `recall` | all | `recall` or `reflect` |
| `autoRecall` | `true` | all | Auto-inject before responses |
| `autoRetain` | `true` | all | Auto-store after responses |
| `recallBudget` | `"mid"` | all | Recall effort: `low`, `mid`, `high` |
| `recallMaxTokens` | `4096` | all | Max tokens for recalled memories |
| `llm_provider` | — | local | LLM provider: `openai`, `anthropic`, `gemini`, `groq`, `minimax`, `ollama`, `lmstudio` |
| `llm_api_key` | — | local | API key for extraction LLM (omit for `ollama`) |
| `llm_model` | provider default | local | Model name override |

## Important: Disable Built-in Memory Tool

Hermes ships with a markdown-based memory tool (`memory_save`, `memory_search`). When both are available, LLMs tend to prefer the simpler, more visible tool.

**Disable it before using Hindsight memory:**

```bash
hermes tools disable memory
```

To re-enable later:

```bash
hermes tools enable memory
```

Check active tools with:

```bash
hermes                    # launch agent
/tools                    # in-chat command
```

Should show `hindsight_*` tools, not the `memory_*` tools.

## Inspect Memory

Check memory provider status:

```bash
hermes memory status
```

Query stored memories directly:

```bash
# For local daemon (embedded mode)
uvx hindsight-embed@latest -p hermes memory recall hermes "user preferences"

# Open web UI
uvx hindsight-embed@latest -p hermes ui
```

Or use the helper script:

```bash
./scripts/inspect-memory.sh
```

## Common Issues

**First launch takes a long time**

On the first `hermes` launch with embedded mode, Hindsight downloads ~3GB of dependencies (Python, sentence-transformers, etc.). Subsequent launches use cached packages. Check progress in `~/.hindsight/profiles/hermes.log`.

**Memories available next turn, not same turn**

Retention is asynchronous. A new fact extracted from the current exchange appears in the next session, not immediately. This is by design to avoid feedback loops.

**`hermes memory status` fails**

Check:
1. Is the API key correct? `cat ~/.hermes/.env | grep HINDSIGHT`
2. Is the API URL reachable? `curl https://api.hindsight.vectorize.io/health`
3. For local mode, is the daemon running? Check `~/.hindsight/profiles/hermes.log`

**Hermes still using built-in memory tool**

Run `hermes tools disable memory` before starting the agent. Verify with `/tools` inside the chat.

**Memory injection feels slow in `hybrid` mode**

Consider switching to `recall` instead of `reflect` for faster responses (at the cost of coherence). Or use `context` mode to avoid exposing both auto-recall and explicit tools.

## Built With

- [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) — AI agent platform with built-in Hindsight provider
- [Hindsight](https://vectorize.io/hindsight) — long-term memory for AI agents
