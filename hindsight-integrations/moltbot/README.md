# Hindsight Memory Plugin for Moltbot

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

## Documentation

For full documentation, configuration options, troubleshooting, and development guide, see:

**[Moltbot Integration Documentation](https://vectorize.io/hindsight/sdks/integrations/moltbot)**

## Links

- [Hindsight Documentation](https://vectorize.io/hindsight)
- [Moltbot Documentation](https://docs.molt.bot)
- [GitHub Repository](https://github.com/vectorize-io/hindsight)

## License

MIT
