# Hindsight Memory Plugin for Moltbot

Biomimetic long-term memory plugin for [Moltbot/Clawdbot](https://molt.bot) with automatic fact extraction, entity resolution, and multi-strategy retrieval.

## Features

- **Auto-retention**: Automatically captures conversation history
- **Fact extraction**: LLM-powered extraction of world facts and experiences
- **Entity resolution**: Normalizes and links entities across memories
- **Multi-strategy retrieval**: Combines semantic, BM25, graph, and temporal search
- **Embedded deployment**: Runs hindsight-embed locally (no external services needed)
- **LLM auto-detection**: Automatically uses your configured Moltbot LLM provider

## Quick Start (Recommended)

### 1. Configure OpenAI in Clawdbot

```bash
# Make sure you're using Node 22
source ~/.nvm/nvm.sh && nvm use 22

# Add OpenAI model to config
clawdbot config set 'agents.defaults.models."openai/gpt-4o-mini"' '{}'

# Set your API key (add to ~/.zshrc or ~/.bashrc to persist)
export OPENAI_API_KEY="sk-your-actual-openai-key-here"

# Verify model is configured
clawdbot models list
```

### 2. Run the Install Script

```bash
cd /path/to/hindsight-integrations/moltbot
./install.sh
```

### 3. Enable and Start

```bash
clawdbot plugins enable hindsight-memory
clawdbot start
```

That's it! The plugin will auto-install `hindsight-all` on first start.

---

## Manual Installation

### 1. Copy Plugin to Moltbot Extensions

```bash
cd /path/to/hindsight-integrations/moltbot
npm install && npm run build

# Deploy
rm -rf ~/.clawdbot/extensions/hindsight-memory
mkdir -p ~/.clawdbot/extensions/hindsight-memory
cp -r dist package.json clawdbot.plugin.json moltbot.plugin.json skills hooks README.md ~/.clawdbot/extensions/hindsight-memory/

# Install dependencies
cd ~/.clawdbot/extensions/hindsight-memory
npm install
```

### 2. Set API Key

The plugin automatically detects which LLM provider you have configured in Moltbot and uses it for Hindsight. You need to set the corresponding API key:

**For Anthropic:**
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
```

**For OpenAI:**
```bash
export OPENAI_API_KEY="your-openai-key"
```

**For Google Gemini:**
```bash
export GEMINI_API_KEY="your-gemini-key"
```

**For Groq:**
```bash
export GROQ_API_KEY="your-groq-key"
```

**For Ollama (no key needed):**
Configure an ollama provider in Moltbot config - no API key required.

### 3. Enable the Plugin

```bash
clawdbot plugins enable hindsight-memory
```

### 4. Start Moltbot

```bash
clawdbot start
```

On first start, `uvx` will automatically download and run `hindsight-embed` (no manual installation needed).

## Configuration

Edit your Moltbot configuration to customize the plugin:

```toml
[plugins.entries.hindsight-memory]
enabled = true

[plugins.entries.hindsight-memory.config]
# Optional: Custom mission/context for the memory bank
bankMission = "You are a helpful assistant that remembers user preferences and past conversations"

# Optional: Fixed port for hindsight-embed server (defaults to auto-assigned)
embedPort = 0
```

## Usage

Once enabled, the plugin automatically:
1. **Retains** every message in conversations (stored with session context)
2. **Provides** a `memory_search` skill for agents to recall relevant memories
3. **Maintains** a single memory bank per Moltbot instance

### Memory Search Skill

The plugin registers a `memory_search` skill that agents can use:

```typescript
{
  "tool": "hindsight.memory_search",
  "parameters": {
    "query": "What did the user say about their preferences?",
    "limit": 10
  }
}
```

## Requirements

- **Node.js**: 22.0.0 or higher
- **uvx**: Python tool runner (comes with uv/Python 3.11+)
- **Moltbot**: Latest version with plugin support
- **API Key**: For your configured LLM provider (or use Ollama locally)

The plugin uses `uvx hindsight-embed` which auto-downloads hindsight-embed on first run.

## Troubleshooting

### "No API keys found" Error

If you see this error:
```
No API keys found for Hindsight memory plugin.
Configured providers in Moltbot: anthropic
```

Make sure you've exported the required API key in your shell:
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

Or run Moltbot with the environment variable:
```bash
ANTHROPIC_API_KEY="your-key" clawdbot start
```

### hindsight-embed Installation Issues

The plugin uses `uvx hindsight-embed` which auto-downloads on first run. If you encounter issues:

1. Verify uvx is installed: `which uvx`
2. If not installed, install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Test manually: `uvx hindsight-embed --help`

### Plugin Not Loading

Check plugin status:
```bash
clawdbot plugins list
```

Look for the Hindsight Memory plugin status. If it shows "error", check the error message for details.

## Architecture

- **Plugin Entry**: `dist/index.js` - Main plugin registration and LLM detection
- **Embed Manager**: `dist/embed-manager.js` - Manages hindsight-embed subprocess
- **Client**: `dist/client.js` - HTTP client for Hindsight API
- **Skills**: `skills/hindsight/` - Memory search tool for agents
- **Hooks**: `hooks/retain-messages/` - Auto-retention on message events

## Development

### Build from Source

```bash
npm run build
```

### Watch Mode

```bash
npm run dev
```

### Deploy to Moltbot

```bash
npm run build
cp -r dist ~/.clawdbot/extensions/hindsight-memory/
```

## License

MIT - See LICENSE file in the Hindsight repository root
