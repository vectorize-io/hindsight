# Hindsight Memory for Moltbot

Memory slot plugin that replaces Moltbot's default Markdown-based memory with Hindsight's biomimetic memory system. Automatically captures conversations and provides intelligent recall via multi-strategy retrieval.

## How It Works

```
┌─────────────────────────────────────────┐
│  Clawdbot Gateway                       │
│  ┌─────────────────────────────────┐   │
│  │  Hindsight Plugin                │   │
│  │                                  │   │
│  │  HOOKS (auto-retention)          │   │
│  │  • Captures messages             │   │
│  │  • Runs automatically            │   │
│  │  • Background processing         │   │
│  │                                  │   │
│  │  SKILLS (memory_search)          │   │
│  │  • Agent tool for searching      │   │
│  │  • Optional, agent decides       │   │
│  │                                  │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
                  ↓
       uvx hindsight-embed (CLI)
       • daemon on port 8889
       • memory retain --async
       • memory recall --format json
                  ↓
          hindsight-api + pg0
```

**HOOKS** force memory storage (automatic)
**SKILLS** let agents search memories (on-demand)

## Installation

```bash
# 1. Set up LLM
clawdbot config set 'agents.defaults.models."openai/gpt-4o-mini"' '{}'
export OPENAI_API_KEY="sk-your-key"

# 2. Install plugin
cd hindsight-integrations/moltbot
./install.sh

# 3. Enable as memory slot (replaces default memory)
clawdbot plugins enable hindsight-memory
# Or manually set: plugins.slots.memory = "hindsight-memory"

# 4. Start (or restart if already running)
clawdbot gateway stop
clawdbot gateway
```

First start downloads `hindsight-embed` automatically via `uvx`.

**Note**: This plugin replaces Moltbot's default Markdown-based memory system. Your conversations will be stored in Hindsight's PostgreSQL database (via pg0) instead of `memory/*.md` files.

## What You Get

### Automatic Memory Capture
Every conversation is automatically stored with:
- Session context
- Background processing (`--async` flag)
- Fact extraction
- Entity resolution

### Memory Search Tool
Agents can call `memory_search` to find relevant context:
```
memory_search "what did the user say about breakfast?"
```

Returns relevant memories with scores and metadata.

## Configuration

Optional config in `~/.clawdbot/clawdbot.json`:

```json
{
  "plugins": {
    "entries": {
      "hindsight-memory": {
        "enabled": true,
        "config": {
          "bankMission": "Custom context for what the agent does",
          "daemonIdleTimeout": 0
        }
      }
    }
  }
}
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `bankMission` | string | - | Custom context/mission for the memory bank |
| `embedPort` | number | random | Port for embedded Hindsight server (auto-assigned if not specified) |
| `daemonIdleTimeout` | number | `0` | Seconds before daemon shuts down from inactivity (0 = never timeout) |

## Supported LLMs

Plugin auto-detects your configured provider:
- **OpenAI**: `export OPENAI_API_KEY="..."`
- **Anthropic**: `export ANTHROPIC_API_KEY="..."`
- **Gemini**: `export GEMINI_API_KEY="..."`
- **Groq**: `export GROQ_API_KEY="..."`
- **Ollama**: No key needed

## Troubleshooting

**No API key found?**
```bash
echo 'export OPENAI_API_KEY="sk-your-key"' >> ~/.zshrc
source ~/.zshrc
```

**Check plugin status:**
```bash
clawdbot plugins list | grep Hindsight
```

**Check daemon:**
```bash
uvx hindsight-embed daemon status
```

**View logs:**
```bash
tail -f ~/.hindsight/daemon.log
```

## Development

```bash
# Build
npm run build

# Deploy
cp -r dist ~/.clawdbot/extensions/hindsight-memory/

# Restart gateway
clawdbot gateway stop
clawdbot gateway
```

## Requirements

- Node.js 22+
- uvx (comes with uv)
- Clawdbot with plugin support
