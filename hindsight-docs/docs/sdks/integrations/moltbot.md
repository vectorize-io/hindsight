---
sidebar_position: 4
---

# Moltbot (Clawdbot)

Long-term memory integration for [Moltbot](https://molt.bot) (formerly Clawdbot), the multi-channel AI assistant. Give your Telegram, WhatsApp, Discord, Slack, and other channel bots biomimetic long-term memory with automatic retention and intelligent recall.

## Features

- **Multi-Channel Memory** - Works across all Moltbot channels (Telegram, WhatsApp, Discord, Slack, Signal, etc.)
- **Automatic Memory Capture** - Conversations are automatically retained via hooks with background processing
- **Memory Slot Replacement** - Replaces Moltbot's default Markdown-based memory with Hindsight's PostgreSQL backend
- **Intelligent Recall** - Multi-strategy retrieval (semantic, BM25, graph, temporal) with cross-encoder reranking
- **Agent Tool** - `memory_search` skill allows agents to proactively query long-term memories
- **Fact Extraction** - Automatically extracts facts, entities, and relationships from conversations
- **LLM Reuse** - Uses Moltbot's configured LLM (no separate API key required)
- **Zero Configuration** - Works out-of-the-box with `uvx hindsight-embed`

## Architecture

```
┌─────────────────────────────────────────┐
│  Moltbot Gateway                        │
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

### Prerequisites

- Node.js 22+
- `uv` and `uvx` installed
- Moltbot with plugin support

### Setup

```bash
# 1. Configure LLM for Moltbot
clawdbot config set 'agents.defaults.models."openai/gpt-4o-mini"' '{}'
export OPENAI_API_KEY="sk-your-key"

# 2. Install the plugin
cd hindsight-integrations/moltbot
./install.sh

# 3. Enable the plugin (sets memory slot)
clawdbot plugins enable hindsight-memory

# 4. Start or restart Moltbot gateway
clawdbot gateway stop
clawdbot gateway
```

On first start, `uvx` will automatically download `hindsight-embed` (no manual installation needed).

## How It Works

### Automatic Memory Capture (Hooks)

The plugin registers hooks that automatically capture conversations:

- **`tool_result_persist`** - After each tool execution
- **`command:new`** - Before starting a new session

When triggered, the hook:
1. Captures the current session messages
2. Formats them into a conversation transcript
3. Calls `uvx hindsight-embed memory retain` with `--async` flag
4. Queues for background processing

```typescript
// Hook automatically retains conversations
await client.retain({
  content: transcript,
  document_id: sessionId,
  metadata: {
    session_key: sessionKey,
    retained_at: new Date().toISOString(),
    message_count: sessionEntry.messages.length,
  },
});
```

### Memory Search Tool (Skill)

Agents can use the `memory_search` skill to query long-term memories:

**In conversation:**
```
User: What did I tell you about my favorite breakfast?
Agent: [uses memory_search "user favorite breakfast"]
Agent: You mentioned that you prefer pancakes with maple syrup!
```

**Tool definition:**
```yaml
---
name: memory_search
description: Search your long-term memory for relevant facts, experiences, and context
user-invocable: false
---
```

The skill handler calls Hindsight's recall API:

```typescript
const response = await client.recall({
  query: "user favorite breakfast",
  limit: 10,
});
```

### Memory Processing Pipeline

1. **Conversation** happens in Telegram/WhatsApp/Discord/etc.
2. **Hook triggered** after tool execution or session start
3. **Transcript created** from session messages
4. **Async retention** via `uvx hindsight-embed memory retain --async`
5. **Background processing**:
   - Fact extraction using configured LLM
   - Entity resolution and normalization
   - Relationship mapping
   - Vector embedding generation
   - Storage to PostgreSQL (via pg0)
6. **Recall available** via `memory_search` skill

## Configuration

Optional configuration in `~/.clawdbot/clawdbot.json`:

```json
{
  "plugins": {
    "entries": {
      "hindsight-memory": {
        "enabled": true,
        "config": {
          "bankMission": "Custom context for what the agent does"
        }
      }
    },
    "slots": {
      "memory": "hindsight-memory"
    }
  }
}
```

### Bank Mission

The `bankMission` config sets the memory bank's context, guiding what Hindsight should remember:

```json
{
  "config": {
    "bankMission": "You're a personal assistant - remember user preferences, important dates, project details, and task priorities."
  }
}
```

## Supported LLM Providers

The plugin auto-detects your Moltbot-configured LLM provider:

| Provider    | Environment Variable     | Configuration                                          |
|-------------|-------------------------|--------------------------------------------------------|
| **OpenAI**  | `OPENAI_API_KEY`        | `clawdbot config set 'agents.defaults.models."openai/gpt-4o-mini"' '{}'` |
| **Anthropic** | `ANTHROPIC_API_KEY`   | `clawdbot config set 'agents.defaults.models."anthropic/claude-sonnet-4"' '{}'` |
| **Gemini**  | `GEMINI_API_KEY`        | `clawdbot config set 'agents.defaults.models."gemini/gemini-2.0-flash"' '{}'` |
| **Groq**    | `GROQ_API_KEY`          | `clawdbot config set 'agents.defaults.models."groq/llama-3.3-70b"' '{}'` |
| **Ollama**  | (none)                  | `clawdbot config set 'agents.defaults.models."ollama/llama3.2"' '{}'` |

The plugin uses the same API key and model that Moltbot is configured to use.

## Verification

### Check Plugin Status

```bash
clawdbot plugins list | grep -i hindsight
```

Expected output:
```
│ Hindsight    │ hindsigh │ loaded   │ ~/.clawdbot/extensions/hindsight-memory/dist/index.js
│ Memory       │ t-memory │          │ Hindsight memory plugin for Moltbot - biomimetic long-term memory
```

### Check Memory Slot Configuration

```bash
clawdbot doctor
```

Expected output:
```
Plugins: Loaded: 2, Disabled: 27, Errors: 0
```

### Test Memory Retention

1. Send a message to your Moltbot via Telegram/WhatsApp:
   ```
   My favorite color is blue and I love pizza
   ```

2. Wait a few seconds for background processing

3. Ask the bot to recall:
   ```
   What's my favorite color?
   ```

The bot should retrieve the information from Hindsight's long-term memory.

### Check Daemon Logs

```bash
# Watch daemon process memories
tail -f ~/.hindsight/daemon.log

# Check retention happened
grep "Retained" ~/.hindsight/daemon.log
```

## Troubleshooting

### Plugin Not Loading

**Error**: `plugin not found: hindsight-memory`

**Fix**: Reinstall the plugin:
```bash
cd hindsight-integrations/moltbot
./install.sh
clawdbot plugins enable hindsight-memory
clawdbot gateway stop && clawdbot gateway
```

### No API Key Found

**Error**: `No API keys found for configured models`

**Fix**: Set environment variable and configure Moltbot:
```bash
echo 'export OPENAI_API_KEY="sk-your-key"' >> ~/.zshrc
source ~/.zshrc
clawdbot config set 'agents.defaults.models."openai/gpt-4o-mini"' '{}'
```

### Daemon Won't Start

**Error**: `Still waiting for daemon... (30s elapsed)`

**Fix**: Check daemon logs for errors:
```bash
tail -50 ~/.hindsight/daemon.log

# Kill existing daemon and restart gateway
uvx hindsight-embed daemon stop
clawdbot gateway stop && clawdbot gateway
```

### Memory Not Being Retained

**Issue**: Conversations aren't being saved

**Debug**:
1. Check hook is registered:
   ```bash
   clawdbot hooks list | grep hindsight
   ```

2. Check daemon is running:
   ```bash
   uvx hindsight-embed daemon status
   ```

3. Check gateway logs:
   ```bash
   tail -f /tmp/clawdbot/clawdbot-*.log | grep -i hindsight
   ```

## Development

### Build and Deploy

```bash
# Build TypeScript
npm run build

# Deploy to local extensions
cp -r dist ~/.clawdbot/extensions/hindsight-memory/

# Restart gateway
clawdbot gateway stop && clawdbot gateway
```

### File Structure

```
hindsight-integrations/moltbot/
├── src/
│   ├── index.ts              # Main plugin entry point
│   ├── client.ts             # Hindsight CLI wrapper
│   ├── embed-manager.ts      # Daemon lifecycle management
│   └── types.ts              # TypeScript type definitions
├── hooks/
│   └── retain-messages/      # Auto-retention hook
│       ├── HOOK.md
│       └── handler.ts
├── skills/
│   └── hindsight/            # memory_search skill
│       ├── SKILL.md
│       └── handler.ts
├── clawdbot.plugin.json      # Plugin manifest
├── package.json              # NPM package config
├── tsconfig.json             # TypeScript config
└── install.sh                # Installation script
```

## Advanced Usage

### Custom Bank Per Channel

You can configure different memory banks for different channels by modifying the plugin:

```typescript
// In src/index.ts
const channelBankMap = {
  'telegram': 'telegram-bot',
  'discord': 'discord-bot',
  'slack': 'slack-bot',
};

const bankId = channelBankMap[channel] || 'default';
client.setBankId(bankId);
```

### Query Hindsight Directly

While the plugin uses `uvx hindsight-embed`, you can query Hindsight directly:

```bash
# Retain a memory
uvx hindsight-embed memory retain default "User prefers dark mode" --document-id session-123 --async

# Recall memories
uvx hindsight-embed memory recall default "user preferences" --limit 5 --format json
```

### Monitor Memory Processing

```bash
# Watch daemon activity
tail -f ~/.hindsight/daemon.log | grep -E "Retained|Recall|Fact extraction"

# Check database
ls -lh ~/.pg0/
```

## Comparison with Default Memory

| Feature | Default Memory (memory-core) | Hindsight Memory |
|---------|------------------------------|------------------|
| Storage | Markdown files (`memory/*.md`) | PostgreSQL (pg0) |
| Search | Vector + BM25 on Markdown | Multi-strategy (semantic, BM25, graph, temporal) |
| Retention | Manual by agent | Automatic via hooks |
| Fact Extraction | None | LLM-based extraction |
| Entities | None | Resolution and normalization |
| Relationships | None | Graph-based linking |
| Reranking | None | Cross-encoder reranking |
| Human Readable | Yes (Markdown files) | No (database) |

## Resources

- **Moltbot Documentation**: https://docs.molt.bot
- **Hindsight GitHub**: https://github.com/vectorize-io/hindsight
- **Plugin Source**: `hindsight-integrations/moltbot/`
- **Issues**: https://github.com/vectorize-io/hindsight/issues

## Next Steps

- Explore [Hindsight CLI](../cli.md) for manual memory operations
- Learn about [LiteLLM integration](./litellm.md) for other frameworks
- Read about [Biomimetic Memory Architecture](../../developer/architecture.md)
