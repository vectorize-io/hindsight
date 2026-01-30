---
sidebar_position: 4
---

# OpenClaw

Local, long term memory for [OpenClaw](https://openclaw.ai) agents using [Hindsight](https://vectorize.io/hindsight).

This plugin integrates [hindsight-embed](https://vectorize.io/hindsight/cli), a standalone daemon that bundles Hindsight's memory engine (API + PostgreSQL) into a single command. Everything runs locally on your machine, reuses the LLM you're already paying for, and costs nothing extra. The plugin automatically manages the daemon lifecycle and provides hooks for seamless memory capture and recall.

## Quick Start

```bash
# 1. Configure your LLM provider
export OPENAI_API_KEY="sk-your-key"
openclaw config set 'agents.defaults.models."openai/gpt-4o-mini"' '{}'

# 2. Install and enable the plugin
openclaw plugins install @vectorize-io/hindsight-openclaw

# 3. Start OpenClaw
openclaw gateway
```

That's it! The plugin will automatically start capturing and recalling memories.

## How It Works

### Auto-Capture (Hooks)
Every conversation is **automatically stored** after each turn:
- Extracts facts, entities, and relationships
- Processes in background (non-blocking)
- Stores in PostgreSQL via embedded `hindsight-api`

### Auto-Recall (Before Agent Start)
Before each agent response, relevant memories are **automatically injected**:
- Relevant memories retrieved (up to 1024 tokens)
- Injected into context with `<hindsight_memories>` tags (JSON format with metadata)
- Agent seamlessly uses past context

## Why Auto-Recall?

Traditional memory systems give agents a `search_memory` tool - the model must decide when to call it. In practice, models don't use memory tools consistently. They lack reliable self-awareness about what they should remember to check.

Auto-recall solves this by injecting relevant memories automatically before every agent turn. Memories are formatted as JSON with full metadata:

```json
<hindsight_memories>
[
 {
    {
      "chunk_id": "openclawd_default-session_12",
      "context": "",
      "document_id": "default-session",
      "id": "5f55f684-e6f5-46e3-9f5c-043bdf005511",
      "mentioned_at": "2026-01-30T11:07:33.211396+00:00",
      "occurred_end": "2025-01-29T23:14:30+00:00",
      "occurred_start": "2025-01-29T23:14:30+00:00",
      "tags": [],
      "text": "Nicolò Boschi attended an OpenAI devday last year and found it cool. | When: 2025-01-30 | Involving: Nicolò Boschi",
      "type": "world"
    }
]
</hindsight_memories>
```

The agent sees past context automatically without needing to remember to remember. This approach trades token cost for reliability - but for conversational agents, spending 500 tokens on auto-injected context is better than ignoring 10,000 stored facts because the model didn't call a tool.

## Understanding OpenClaw Concepts

### Plugins
Extensions that add functionality to OpenClaw. This Hindsight plugin:
- Runs a background service (manages `hindsight-embed` daemon)
- Registers hooks (automatic event handlers)

### Hooks
Automatic event handlers that run without agent involvement:
- **`before_agent_start`**: Auto-recall - injects memories before agent processes message
- **`agent_end`**: Auto-capture - stores conversation after agent responds

Think of hooks as "forced automation" - they always run.

## Architecture

```
┌─────────────────────────────────────────┐
│  OpenClaw Gateway                         │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │  Hindsight Plugin                 │ │
│  │                                   │ │
│  │  • Service: Manages daemon       │ │
│  │  • Hook: before_agent_start      │ │
│  │    → Auto-recall (1024 tokens)   │ │
│  │  • Hook: agent_end               │ │
│  │    → Auto-capture                │ │
│  └───────────────────────────────────┘ │
└─────────────────────────────────────────┘
                  ↓
       uvx hindsight-embed
       • Daemon on port 8889
       • PostgreSQL (pg0://hindsight-embed)
       • Bank: 'openclaw' (isolated within shared database)
       • Fact extraction
```

**Database Architecture:** All banks share a single pg0 database instance (`pg0://hindsight-embed`). Bank isolation happens within the database via separate tables/schemas per bank ID. The 'openclaw' bank is automatically created when the plugin stores its first memory.

**Local-First Design:**
- **Your data stays local**: All conversations, facts, and relationships stored in PostgreSQL on your machine
- **No additional costs**: Reuses your configured LLM provider (OpenAI, Anthropic, Gemini, Groq, Ollama) - no separate memory API charges
- **No vendor lock-in**: Standard PostgreSQL storage, export anytime with `hindsight-embed memory export`
- **Works offline**: With Ollama, the entire stack (agent + memory + LLM) runs offline
- **Zero infrastructure setup**: No database deployment, connection strings, or credential management - everything handled automatically

## Installation

### Prerequisites

- **Node.js** 22+
- **OpenClaw** (Clawdbot) with plugin support
- **uv/uvx** for running `hindsight-embed`
- **LLM API key** (OpenAI, Anthropic, etc.)

### Setup

```bash
# 1. Configure your LLM provider
export OPENAI_API_KEY="sk-your-key"
openclaw config set 'agents.defaults.models."openai/gpt-4o-mini"' '{}'

# 2. Install and enable the plugin
openclaw plugins install @vectorize-io/hindsight-openclaw

# 3. Start OpenClaw
openclaw gateway
```

On first start, `uvx` will automatically download `hindsight-embed` (no manual installation needed).


## Configuration

Optional settings in `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "hindsight-openclaw": {
        "enabled": true,
        "config": {
          "daemonIdleTimeout": 0
        }
      }
    }
  }
}
```

**Options:**
- `daemonIdleTimeout` (number, default: `0`) - Seconds before daemon shuts down from inactivity (0 = never)
- `bankMission` (string, default: auto-generated) - Custom context for the memory bank. Defaults to: "You are an AI assistant helping users across multiple communication channels (Telegram, Slack, Discord, etc.). Remember user preferences, instructions, and important context from conversations to provide personalized assistance."
- `embedVersion` (string, default: `"latest"`) - hindsight-embed version to use (e.g., `"latest"`, `"0.4.2"`, or leave empty for latest). Use this to pin a specific version if latest is broken.

## Supported LLM Providers

The plugin auto-detects your configured provider and API key:

| Provider | Environment Variable | Model Example |
|----------|---------------------|---------------|
| OpenAI | `OPENAI_API_KEY` | `openai/gpt-4o-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic/claude-sonnet-4` |
| Gemini | `GEMINI_API_KEY` | `gemini/gemini-2.0-flash-exp` |
| Groq | `GROQ_API_KEY` | `groq/llama-3.3-70b` |
| Ollama | None needed | `ollama/llama3` |

Configure with:
```bash
export OPENAI_API_KEY="sk-your-key"
openclaw config set 'agents.defaults.models."openai/gpt-4o-mini"' '{}'
```

## Verification

**Check if plugin is loaded:**
```bash
openclaw plugins list | grep hindsight
# Should show: ✓ enabled │ Hindsight Memory │ ...
```

**Test auto-recall:**
Send a message on any OpenClaw channel (Telegram, Slack, etc.):
```
User: My name is John and I love pizza
Bot: Got it! I'll remember that.

User: What do I like to eat?
Bot: You love pizza!  # ← Used auto-recall
```

**View daemon logs:**
```bash
tail -f ~/.hindsight/daemon.log
```

**Check memories in database:**
```bash
uvx hindsight-embed@latest memory recall openclaw "pizza" --output json
```

## Inspecting Memories

The plugin uses `hindsight-embed` daemon which provides CLI commands for inspection:

**View daemon logs:**
```bash
uvx hindsight-embed@latest daemon logs
# Or follow logs in real-time:
tail -f ~/.hindsight/daemon.log
```

**Open web UI:**
```bash
uvx hindsight-embed@latest ui
# Opens browser to http://localhost:8890
# Browse memories, facts, entities, and relationships
```

**List memory banks:**
```bash
uvx hindsight-embed@latest bank list
# Shows all banks including 'openclaw'
```

**Query memories:**
```bash
# Search memories
uvx hindsight-embed@latest memory recall openclaw "user preferences" --output json

# View recent memories
uvx hindsight-embed@latest memory list openclaw --limit 10

# Export all memories
uvx hindsight-embed@latest memory export openclaw --output memories.json
```

**Inspect facts and entities:**
```bash
# List extracted facts
uvx hindsight-embed@latest fact list openclaw

# List entities
uvx hindsight-embed@latest entity list openclaw

# Show entity relationships
uvx hindsight-embed@latest entity graph openclaw
```

## Troubleshooting

**Plugin not loading?**
```bash
# Check plugin installation
openclaw plugins list | grep -i hindsight

# Reinstall if needed
openclaw plugins install @vectorize-io/hindsight-openclaw
```

**Daemon not starting?**
```bash
# Check daemon status
uvx hindsight-embed@latest daemon status

# Manually start
uvx hindsight-embed@latest daemon start

# View logs
tail -f ~/.hindsight/daemon.log
```

**No API key error?**
```bash
# Set in shell profile
echo 'export OPENAI_API_KEY="sk-your-key"' >> ~/.zshrc
source ~/.zshrc

# Verify
echo $OPENAI_API_KEY
```

**Memories not being stored?**
```bash
# Check gateway logs for auto-capture
tail -f /tmp/openclaw/openclaw-*.log | grep Hindsight

# Should see:
# [Hindsight Hook] agent_end triggered
# [Hindsight] Retained X messages for session ...
```
