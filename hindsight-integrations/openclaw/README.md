# Hindsight Memory Plugin for OpenClaw

Biomimetic long-term memory for [OpenClaw](https://openclaw.ai) using [Hindsight](https://vectorize.io/hindsight). Automatically captures conversations and intelligently recalls relevant context.

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

## Documentation

For full documentation, configuration options, troubleshooting, and development guide, see:

**[OpenClaw Integration Documentation](https://vectorize.io/hindsight/sdks/integrations/openclaw)**

## Links

- [Hindsight Documentation](https://vectorize.io/hindsight)
- [OpenClaw Documentation](https://openclaw.ai)
- [GitHub Repository](https://github.com/vectorize-io/hindsight)

## License

MIT
