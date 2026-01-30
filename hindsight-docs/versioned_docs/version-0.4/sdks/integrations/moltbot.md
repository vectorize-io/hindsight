---
sidebar_position: 4
---

# Moltbot (Clawdbot)

Biomimetic long-term memory for [Moltbot](https://molt.bot) using [Hindsight](https://vectorize.io/hindsight). Automatically captures conversations and intelligently recalls relevant context.

## Quick Start

```bash
# 1. Install the plugin
npm install -g @vectorize-io/hindsight-moltbot-plugin

# 2. Configure your LLM provider
export OPENAI_API_KEY="sk-your-key"
clawdbot config set 'agents.defaults.models."openai/gpt-4o-mini"' '{}'

# 3. Enable the plugin
clawdbot plugins enable hindsight-memory

# 4. Start Moltbot
clawdbot gateway
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
- Injected into context with `<hindsight-context>` tags
- Agent seamlessly uses past context

## Understanding Moltbot Concepts

### Plugins
Extensions that add functionality to Moltbot. This Hindsight plugin:
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
│  Moltbot Gateway                        │
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
       • PostgreSQL (pg0)
       • Fact extraction
```

## Installation

### Prerequisites

- **Node.js** 22+
- **Moltbot** (Clawdbot) with plugin support
- **uv/uvx** for running `hindsight-embed`
- **LLM API key** (OpenAI, Anthropic, etc.)

### Setup

```bash
# 1. Install the plugin
npm install -g @vectorize-io/hindsight-moltbot-plugin

# 2. Configure your LLM provider
export OPENAI_API_KEY="sk-your-key"
clawdbot config set 'agents.defaults.models."openai/gpt-4o-mini"' '{}'

# 3. Enable the plugin
clawdbot plugins enable hindsight-memory

# 4. Start Moltbot
clawdbot gateway
```

On first start, `uvx` will automatically download `hindsight-embed` (no manual installation needed).


## Configuration

Optional settings in `~/.clawdbot/clawdbot.json`:

```json
{
  "plugins": {
    "entries": {
      "hindsight-memory": {
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
- `embedPort` (number, default: auto) - Port for embedded server
- `bankMission` (string, default: none) - Custom context for the memory bank

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
clawdbot config set 'agents.defaults.models."openai/gpt-4o-mini"' '{}'
```

## Verification

**Check if plugin is loaded:**
```bash
clawdbot plugins list | grep hindsight
# Should show: ✓ enabled │ Hindsight Memory │ ...
```

**Test auto-recall:**
Send a message on any Moltbot channel (Telegram, Slack, etc.):
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
uvx hindsight-embed memory recall moltbot "pizza" --output json
```

## Troubleshooting

**Plugin not loading?**
```bash
# Check plugin installation
npm list -g @vectorize-io/hindsight-moltbot-plugin

# Reinstall if needed
npm install -g @vectorize-io/hindsight-moltbot-plugin
clawdbot plugins enable hindsight-memory
```

**Daemon not starting?**
```bash
# Check daemon status
uvx hindsight-embed daemon status

# Manually start
uvx hindsight-embed daemon start

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
tail -f /tmp/clawdbot/clawdbot-*.log | grep Hindsight

# Should see:
# [Hindsight Hook] agent_end triggered
# [Hindsight] Retained X messages for session ...
```

## Development

```bash
# Clone repo
git clone https://github.com/vectorize-io/hindsight.git
cd hindsight/hindsight-integrations/moltbot

# Install dependencies
npm install

# Build
npm run build

# Run tests
npm test

# Install locally
npm run build && ./install.sh
```

## Requirements

- **Node.js** 22+
- **Moltbot** (Clawdbot) with plugin support
- **uv/uvx** for running `hindsight-embed`
- **LLM API key** (OpenAI, Anthropic, etc.)

## License

MIT

## Links

- [Hindsight Documentation](https://vectorize.io/hindsight)
- [Moltbot Documentation](https://docs.molt.bot)
- [GitHub Repository](https://github.com/vectorize-io/hindsight)
