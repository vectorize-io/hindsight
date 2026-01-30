---
sidebar_position: 2
---

# Local MCP Server

Hindsight provides a fully local MCP server that runs entirely on your machine with an embedded PostgreSQL database. No external server or database setup required.

This is ideal for:
- **Personal use with Claude Desktop** — Give Claude long-term memory across conversations
- **Development and testing** — Quick setup without infrastructure
- **Privacy-focused setups** — All data stays on your machine

## Quick Install

```bash
curl -fsSL https://hindsight.vectorize.io/get-mcp | bash -s -- \
  --app claude-desktop \
  --set HINDSIGHT_API_LLM_API_KEY=sk-...
```

This script will:
1. Install [uv](https://docs.astral.sh/uv/) if not already installed
2. Configure Claude Desktop to use the Hindsight MCP server
3. Set the provided environment variables in the MCP configuration

:::info Other MCP Applications
The quick install script currently supports Claude Desktop only. For other MCP-compatible applications (Cursor, Cline, etc.), follow the [Manual Configuration](#manual-configuration) steps below.
:::

## Manual Configuration

Add the following to your MCP client's configuration. For Claude Desktop:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

For other MCP clients, refer to their documentation for the configuration file location.

```json
{
  "mcpServers": {
    "hindsight": {
      "command": "uvx",
      "args": ["--from", "hindsight-api", "hindsight-local-mcp"],
      "env": {
        "HINDSIGHT_API_LLM_API_KEY": "sk-..."
      }
    }
  }
}
```

### With Custom Bank ID

By default, memories are stored in a bank called `mcp`. To use a different bank:

```json
{
  "mcpServers": {
    "hindsight": {
      "command": "uvx",
      "args": ["--from", "hindsight-api", "hindsight-local-mcp"],
      "env": {
        "HINDSIGHT_API_LLM_API_KEY": "sk-...",
        "HINDSIGHT_API_MCP_LOCAL_BANK_ID": "my-personal-memory"
      }
    }
  }
}
```

## Environment Variables

All standard [Hindsight configuration variables](/developer/configuration) are supported.

### Local MCP Specific

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HINDSIGHT_API_MCP_LOCAL_BANK_ID` | No | `mcp` | Memory bank ID to use |
| `HINDSIGHT_API_MCP_INSTRUCTIONS` | No | - | Additional instructions appended to both `retain` and `recall` tools |

### Customizing Tool Behavior

You can customize what gets stored by adding instructions to the tools. Re-run the install script with the additional `--set` flag:

```bash
curl -fsSL https://hindsight.vectorize.io/get-mcp | bash -s -- \
  --app claude-desktop \
  --set HINDSIGHT_API_LLM_API_KEY=sk-... \
  --set HINDSIGHT_API_MCP_INSTRUCTIONS="Also store every action you take, code you write, and files you modify."
```

These instructions are appended to the default tool descriptions, guiding Claude on when and how to use the memory tools.

## Available Tools

### retain

Store information to long-term memory. This is a **fire-and-forget** operation — it returns immediately while processing happens in the background.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | Yes | The fact or memory to store |
| `context` | string | No | Category for the memory (default: `general`) |

**Example:**
```json
{
  "name": "retain",
  "arguments": {
    "content": "User's favorite color is blue",
    "context": "preferences"
  }
}
```

**Response:**
```json
{
  "status": "accepted",
  "message": "Memory storage initiated"
}
```

### recall

Search memories to provide personalized responses.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Natural language search query |
| `max_tokens` | integer | No | Maximum tokens to return (default: 4096) |
| `budget` | string | No | Search depth: `low`, `mid`, or `high` (default: `low`) |

**Example:**
```json
{
  "name": "recall",
  "arguments": {
    "query": "What are the user's color preferences?",
    "max_tokens": 2048,
    "budget": "mid"
  }
}
```

## How It Works

The local MCP server:

1. **Starts an embedded PostgreSQL** (pg0) on an automatically assigned port
2. **Initializes the Hindsight memory engine** with local embeddings
3. **Connects via stdio** to Claude Code using the MCP protocol

Data is persisted in the pg0 data directory (`~/.pg0/hindsight-mcp/`), so your memories survive restarts.

## Troubleshooting

### "HINDSIGHT_API_LLM_API_KEY required"

Make sure you've set the API key in your MCP configuration:

```json
{
  "env": {
    "HINDSIGHT_API_LLM_API_KEY": "sk-..."
  }
}
```

### Slow startup

The first startup may take longer as it:
- Downloads the embedding model (~100MB)
- Initializes the PostgreSQL database

Subsequent starts are faster.

### Checking logs

Set `HINDSIGHT_API_LOG_LEVEL=debug` for verbose output:

```json
{
  "env": {
    "HINDSIGHT_API_LOG_LEVEL": "debug"
  }
}
```

Logs are written to stderr and visible in Claude Code's MCP server output.
