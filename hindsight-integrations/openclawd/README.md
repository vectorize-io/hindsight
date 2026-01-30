# Hindsight Memory Plugin for OpenClawd

Biomimetic long-term memory for [OpenClawd](https://openclawd.com) using [Hindsight](https://vectorize.io/hindsight). Automatically captures conversations and intelligently recalls relevant context.

## Quick Start

```bash
# 1. Configure your LLM provider
export OPENAI_API_KEY="sk-your-key"
clawdbot config set 'agents.defaults.models."openai/gpt-4o-mini"' '{}'

# 2. Install and enable the plugin
clawdbot plugins install @vectorize-io/hindsight-openclawd

# 3. Start OpenClawd
clawdbot gateway
```

That's it! The plugin will automatically start capturing and recalling memories.

## Documentation

For full documentation, configuration options, troubleshooting, and development guide, see:

**[OpenClawd Integration Documentation](https://vectorize.io/hindsight/sdks/integrations/openclawd)**

## Links

- [Hindsight Documentation](https://vectorize.io/hindsight)
- [OpenClawd Documentation](https://openclawd.com)
- [GitHub Repository](https://github.com/vectorize-io/hindsight)

## License

MIT
