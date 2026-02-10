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

# Option E: Claude Code (uses claude-sonnet-4-20250514, no API key needed)
export HINDSIGHT_API_LLM_PROVIDER=claude-code

# Option F: OpenAI Codex (uses o3-mini, no API key needed)
export HINDSIGHT_API_LLM_PROVIDER=openai-codex
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
- Start a local Hindsight daemon (port 9077)
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
          "apiPort": 9077,
          "daemonIdleTimeout": 0,
          "embedVersion": "latest"
        }
      }
    }
  }
}
```

**Options:**
- `apiPort` - Port for the openclaw profile daemon (default: `9077`)
- `daemonIdleTimeout` - Seconds before daemon shuts down from inactivity (default: `0` = never)
- `embedVersion` - hindsight-embed version (default: `"latest"`)
- `bankMission` - Custom context for the memory bank (optional)

### LLM Configuration

The plugin auto-detects your LLM provider from these environment variables:

| Provider | Env Var | Default Model | Notes |
|----------|---------|---------------|-------|
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` | |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-3-5-haiku-20241022` | |
| Gemini | `GEMINI_API_KEY` | `gemini-2.5-flash` | |
| Groq | `GROQ_API_KEY` | `openai/gpt-oss-20b` | |
| Claude Code | `HINDSIGHT_API_LLM_PROVIDER=claude-code` | `claude-sonnet-4-20250514` | No API key needed |
| OpenAI Codex | `HINDSIGHT_API_LLM_PROVIDER=openai-codex` | `o3-mini` | No API key needed |

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

Connect to a remote Hindsight API server instead of running a local daemon. This is useful for:

- **Shared memory** across multiple OpenClaw instances
- **Production deployments** with centralized memory storage
- **Team environments** where agents share knowledge

#### Plugin Configuration

Configure in `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "entries": {
      "hindsight-openclaw": {
        "enabled": true,
        "config": {
          "hindsightApiUrl": "https://your-hindsight-server.com",
          "hindsightApiToken": "your-api-token"
        }
      }
    }
  }
}
```

**Options:**
- `hindsightApiUrl` - Full URL to external Hindsight API (e.g., `https://mcp.hindsight.example.com`)
- `hindsightApiToken` - API token for authentication (optional, only if API requires auth)

#### Environment Variables (Alternative)

You can also configure via environment variables:

```bash
export HINDSIGHT_EMBED_API_URL=https://your-hindsight-server.com
export HINDSIGHT_EMBED_API_TOKEN=your-api-token  # Optional

openclaw gateway
```

**Note:** Plugin config takes precedence over environment variables.

#### Behavior

When external API mode is enabled:
- **No local daemon** is started (no hindsight-embed process)
- **Health check** runs on startup to verify API connectivity
- **All memory operations** (retain, recall, reflect) go to the external API
- **Faster startup** since no local PostgreSQL or embedding models are needed

#### Verification

Check OpenClaw logs for external API mode:

```bash
tail -f /tmp/openclaw/openclaw-*.log | grep Hindsight

# Should see on startup:
# [Hindsight] External API mode enabled: https://your-hindsight-server.com
# [Hindsight] External API health check passed
```

If you see daemon startup messages instead, verify your configuration is correct.

## Inspecting Memories

### Check Configuration

View the daemon config that was written by the plugin:

```bash
cat ~/.hindsight/profiles/openclaw.env
```

This shows the LLM provider, model, port, and other settings the daemon is using.

### Check Daemon Status

```bash
# Check if daemon is running
uvx hindsight-embed@latest -p openclaw daemon status

# View daemon logs
tail -f ~/.hindsight/profiles/openclaw.log
```

### Query Memories

```bash
# Search memories
uvx hindsight-embed@latest -p openclaw memory recall openclaw "user preferences"

# View recent memories
uvx hindsight-embed@latest -p openclaw memory list openclaw --limit 10

# Open web UI (uses openclaw profile's daemon)
uvx hindsight-embed@latest -p openclaw ui
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
# Check daemon status (note: -p openclaw uses the openclaw profile)
uvx hindsight-embed@latest -p openclaw daemon status

# View logs for errors
tail -f ~/.hindsight/profiles/openclaw.log

# Check configuration
cat ~/.hindsight/profiles/openclaw.env

# List all profiles
uvx hindsight-embed@latest profile list
```

### No API key error

Make sure you've set one of the provider API keys (or use a provider that doesn't require one):

```bash
# Option 1: OpenAI
export OPENAI_API_KEY="sk-your-key"

# Option 2: Anthropic
export ANTHROPIC_API_KEY="your-key"

# Option 3: Claude Code (no API key needed)
export HINDSIGHT_API_LLM_PROVIDER=claude-code

# Option 4: OpenAI Codex (no API key needed)
export HINDSIGHT_API_LLM_PROVIDER=openai-codex

# Verify it's set
echo $OPENAI_API_KEY
# or
echo $HINDSIGHT_API_LLM_PROVIDER
```

### Verify it's working

Check gateway logs for memory operations:

```bash
tail -f /tmp/openclaw/openclaw-*.log | grep Hindsight

# Should see on startup:
# [Hindsight] ✓ Using provider: openai, model: gpt-4o-mini
# or
# [Hindsight] ✓ Using provider: claude-code, model: claude-sonnet-4-20250514

# Should see after conversations:
# [Hindsight] Retained X messages for session ...
# [Hindsight] Auto-recall: Injecting X memories
```
