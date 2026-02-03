# Hindsight Memory Plugin for OpenClaw

Biomimetic long-term memory for [OpenClaw](https://openclaw.ai) using [Hindsight](https://vectorize.io/hindsight). Automatically captures conversations and intelligently recalls relevant context.

## Quick Start

```bash
# 1. Configure your LLM provider for memory extraction
# Option A: OpenAI
export OPENAI_API_KEY="sk-your-key"

# Option B: Claude Code (no API key needed)
export HINDSIGHT_API_LLM_PROVIDER=claude-code

# Option C: OpenAI Codex (no API key needed)
export HINDSIGHT_API_LLM_PROVIDER=openai-codex

# 2. Install and enable the plugin
openclaw plugins install @vectorize-io/hindsight-openclaw

# 3. Start OpenClaw
openclaw gateway
```

That's it! The plugin will automatically start capturing and recalling memories.

## Documentation

For full documentation, configuration options, troubleshooting, and development guide, see:

**[OpenClaw Integration Documentation](https://vectorize.io/hindsight/sdks/integrations/openclaw)**

## Development

To test local changes to the Hindsight package before publishing:

1. Add `embedPackagePath` to your plugin config in `~/.openclaw/openclaw.json`:
```json
{
  "plugins": {
    "entries": {
      "hindsight-openclaw": {
        "enabled": true,
        "config": {
          "embedPackagePath": "/path/to/hindsight-wt3/hindsight-embed"
        }
      }
    }
  }
}
```

2. The plugin will use `uv run --directory <path> hindsight-embed` instead of `uvx hindsight-embed@latest`

3. To use a specific profile for testing:
```bash
# Check daemon status
uvx hindsight-embed@latest -p openclaw daemon status

# View logs
tail -f ~/.hindsight/profiles/openclaw.log

# List profiles
uvx hindsight-embed@latest profile list
```

## Links

- [Hindsight Documentation](https://vectorize.io/hindsight)
- [OpenClaw Documentation](https://openclaw.ai)
- [GitHub Repository](https://github.com/vectorize-io/hindsight)

## License

MIT
