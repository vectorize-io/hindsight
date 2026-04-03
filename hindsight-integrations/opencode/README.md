# opencode-hindsight

Persistent long-term memory for [OpenCode](https://opencode.ai) via [Hindsight](https://hindsight.vectorize.io).

Auto-captures coding sessions, recalls relevant context via MCP, and supports backfilling historical sessions.

## Installation

Add the plugin to your OpenCode config:

```json
{
  "plugin": ["opencode-hindsight"]
}
```

Set an LLM API key for memory extraction:

```bash
export OPENAI_API_KEY="sk-your-key"
# or: ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY
```

Start OpenCode. The plugin activates automatically.

## How It Works

**Auto-retain**: When a session goes idle, the plugin reads the conversation from OpenCode's SQLite database and sends it to Hindsight's retain API. Each session uses a stable `document_id`, so re-runs are idempotent.

**MCP recall**: For on-demand recall during sessions, add Hindsight's built-in MCP server to your config:

```json
{
  "plugin": ["opencode-hindsight"],
  "mcp": {
    "hindsight": {
      "type": "remote",
      "url": "http://localhost:9077/mcp/opencode/"
    }
  }
}
```

The default port is `9077` (hindsight-embed daemon). If running the full Hindsight API server, the default is `8888`.

This gives the agent `retain`, `recall`, and `reflect` tools natively.

**Compaction memory**: When a session is compacted, the plugin recalls relevant memories from Hindsight and injects them into the compaction context, so they persist across session boundaries.

## Backfilling Historical Sessions

Import existing OpenCode session history:

```bash
pip install hindsight-client

# All sessions
python backfill/backfill.py opencode \
  --hindsight-url http://localhost:8888 \
  --bank-id opencode

# Filtered
python backfill/backfill.py opencode \
  --hindsight-url http://localhost:8888 \
  --bank-id opencode \
  --since 2026-03-01 \
  --project my-project
```

Import JSONL transcripts (Claude Code, custom agents, etc.):

```bash
python backfill/backfill.py jsonl \
  --hindsight-url http://localhost:8888 \
  --bank-id my-agent \
  --input "./transcripts/*.jsonl"
```

## Configuration

Create `~/.hindsight/opencode.json`:

```json
{
  "autoRetain": true,
  "autoRecall": true,
  "bankId": "opencode",
  "dynamicBankId": false,
  "debug": false
}
```

All settings can be overridden via `HINDSIGHT_*` environment variables. See the [full documentation](https://hindsight.vectorize.io/sdks/integrations/opencode) for all options.

## Connection Modes

1. **External API**: Set `hindsightApiUrl` to a running Hindsight server
2. **Local daemon**: Plugin auto-starts `hindsight-embed` via `uvx` (requires LLM API key)
3. **Existing server**: Plugin detects `hindsight-embed` already running on the configured port

## Development

```bash
# Install dependencies
bun install

# Run tests
bun test
```

## License

MIT
