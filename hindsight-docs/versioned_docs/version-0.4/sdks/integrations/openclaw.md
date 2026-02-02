---
sidebar_position: 4
---

# OpenClaw

Local, long term memory for [OpenClaw](https://openclaw.ai) agents using [Hindsight](https://vectorize.io/hindsight).

This plugin integrates [hindsight-embed](https://vectorize.io/hindsight/cli), a standalone daemon that bundles Hindsight's memory engine (API + PostgreSQL) into a single command. Everything runs locally on your machine, reuses the LLM you're already paying for, and costs nothing extra.

## Quick Start

**Step 1: Set up LLM for memory extraction**

Choose one provider and set its API key:

```bash
# Option A: OpenAI (uses gpt-4o-mini for memory extraction)
export OPENAI_API_KEY="sk-your-key"

# Option B: Anthropic (uses claude-3-5-haiku for memory extraction)
export ANTHROPIC_API_KEY="your-key"

# Option C: Gemini (uses gemini-2.5-flash for memory extraction)
export GEMINI_API_KEY="your-key"

# Option D: Groq (uses openai/gpt-oss-20b for memory extraction)
export GROQ_API_KEY="your-key"
```

**Step 2: Install the plugin**

```bash
openclaw plugins install @vectorize-io/hindsight-openclaw
```

**Step 3: Start OpenClaw**

```bash
openclaw gateway
```

The plugin will automatically:
- Start a local Hindsight daemon (port 8888)
- Capture conversations after each turn
- Inject relevant memories before agent responses

**Important:** The LLM you configure above is **only for memory extraction** (background processing). Your main OpenClaw agent can use any model you configure separately.

## How It Works

**Auto-Capture:** Every conversation is automatically stored after each turn. Facts, entities, and relationships are extracted in the background.

**Auto-Recall:** Before each agent response, relevant memories are automatically injected into the context (up to 1024 tokens). The agent uses past context without needing to call tools.

Traditional memory systems give agents a `search_memory` tool - but models don't use it consistently. Auto-recall solves this by injecting memories automatically before every turn.

## Configuration

### Plugin Settings

Optional settings in `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "hindsight-openclaw": {
        "enabled": true,
        "config": {
          "daemonIdleTimeout": 0,
          "embedVersion": "latest"
        }
      }
    }
  }
}
```

**Options:**
- `daemonIdleTimeout` - Seconds before daemon shuts down from inactivity (default: `0` = never)
- `embedVersion` - hindsight-embed version (default: `"latest"`)
- `bankMission` - Custom context for the memory bank (optional)

### LLM Configuration

The plugin auto-detects your LLM provider from these environment variables:

| Provider | Env Var | Default Model |
|----------|---------|---------------|
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-3-5-haiku-20241022` |
| Gemini | `GEMINI_API_KEY` | `gemini-2.5-flash` |
| Groq | `GROQ_API_KEY` | `openai/gpt-oss-20b` |

**Override with explicit config:**

```bash
export HINDSIGHT_API_LLM_PROVIDER=openai
export HINDSIGHT_API_LLM_MODEL=gpt-4o-mini
export HINDSIGHT_API_LLM_API_KEY=sk-your-key

# Optional: custom base URL (OpenRouter, Azure, vLLM, etc.)
export HINDSIGHT_API_LLM_BASE_URL=https://openrouter.ai/api/v1
```

**Example: Free OpenRouter model**

```bash
export HINDSIGHT_API_LLM_PROVIDER=openai
export HINDSIGHT_API_LLM_MODEL=xiaomi/mimo-v2-flash  # FREE!
export HINDSIGHT_API_LLM_API_KEY=sk-or-v1-your-openrouter-key
export HINDSIGHT_API_LLM_BASE_URL=https://openrouter.ai/api/v1
```

### External API (Advanced)

To use an existing Hindsight API server instead of the local daemon:

```bash
export HINDSIGHT_EMBED_API_URL=http://your-server:8000
export HINDSIGHT_EMBED_API_TOKEN=your-api-token  # Optional, if API requires auth

openclaw gateway
```

Useful for shared memory across multiple OpenClaw instances or production deployments.

## Inspecting Memories

### Check Configuration

View the daemon config that was written by the plugin:

```bash
cat ~/.hindsight/embed
```

This shows the LLM provider, model, and other settings the daemon is using.

### Check Daemon Status

```bash
# Check if daemon is running
uvx hindsight-embed@latest daemon status

# View daemon logs
tail -f ~/.hindsight/daemon.log
```

### Query Memories

```bash
# Search memories
uvx hindsight-embed@latest memory recall openclaw "user preferences"

# View recent memories
uvx hindsight-embed@latest memory list openclaw --limit 10

# Open web UI
uvx hindsight-embed@latest ui
```

## Troubleshooting

### Plugin not loading

```bash
openclaw plugins list | grep hindsight
# Should show: ✓ enabled │ Hindsight Memory │ ...

# Reinstall if needed
openclaw plugins install @vectorize-io/hindsight-openclaw
```

### Daemon not starting

```bash
# Check daemon status
uvx hindsight-embed@latest daemon status

# View logs for errors
tail -f ~/.hindsight/daemon.log

# Check configuration
cat ~/.hindsight/embed
```

### No API key error

Make sure you've set one of the provider API keys:

```bash
export OPENAI_API_KEY="sk-your-key"
# or
export ANTHROPIC_API_KEY="your-key"

# Verify it's set
echo $OPENAI_API_KEY
```

### Verify it's working

Check gateway logs for memory operations:

```bash
tail -f /tmp/openclaw/openclaw-*.log | grep Hindsight

# Should see on startup:
# [Hindsight] ✓ Using provider: openai, model: gpt-4o-mini

# Should see after conversations:
# [Hindsight] Retained X messages for session ...
# [Hindsight] Auto-recall: Injecting X memories
```
