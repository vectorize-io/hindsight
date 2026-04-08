---
sidebar_position: 2
title: "Hindsight Local MCP Server | Persistent Memory for Claude"
description: "Run Hindsight as a local MCP server with embedded PostgreSQL — no external setup required. Ideal for Claude Code and Claude Desktop for long-term memory across conversations."
---

# Local MCP Server

Hindsight provides a local MCP server that runs entirely on your machine with an embedded PostgreSQL database. No external server or database setup required.

This is ideal for:
- **Personal use with Claude Code / Claude Desktop** — Give Claude long-term memory across conversations
- **Development and testing** — Quick setup without infrastructure
- **Privacy-focused setups** — All data stays on your machine

## How It Works

Running `hindsight-local-mcp` starts the full Hindsight API on `localhost:8888` with an embedded PostgreSQL database (pg0). You then connect your MCP client to it over HTTP.

- Starts an embedded PostgreSQL (pg0) automatically
- Runs database migrations on startup
- Exposes the full MCP endpoint at `http://localhost:8888/mcp/`
- Data persists in `~/.pg0/hindsight-mcp/` across restarts

## Setup

### 1. Start the server

```bash
HINDSIGHT_API_LLM_API_KEY=sk-... uvx --from hindsight-api hindsight-local-mcp
```

Or with Ollama (no API key needed):

```bash
HINDSIGHT_API_LLM_PROVIDER=ollama HINDSIGHT_API_LLM_MODEL=llama3.2 uvx --from hindsight-api hindsight-local-mcp
```

### 2. Configure your MCP client

**Claude Code:**

```bash
claude mcp add --transport http hindsight http://localhost:8888/mcp/
```

**Other MCP clients** — add an HTTP transport entry pointing to `http://localhost:8888/mcp/`.

## Bank Modes

The local server supports the same two modes as the hosted API:

### Multi-bank mode (default)

Use `http://localhost:8888/mcp/` — exposes all tools including bank management. Bank is selected per-request via the `bank_id` tool parameter or the `X-Bank-Id` header.

```bash
claude mcp add --transport http hindsight http://localhost:8888/mcp/
```

### Single-bank mode

Use `http://localhost:8888/mcp/<bank-id>/` — pins all tools to one bank, no `bank_id` parameter needed. This replaces the old `HINDSIGHT_API_MCP_LOCAL_BANK_ID` env var.

```bash
claude mcp add --transport http hindsight http://localhost:8888/mcp/my-bank/
```

## Available Tools

The local server exposes the full tool set (29 tools in multi-bank mode, 26 in single-bank mode):

**Core Memory**

| Tool | Description |
|------|-------------|
| `retain` | Store information to long-term memory with optional tags, metadata, and document association |
| `recall` | Search memories with natural language, configurable budget, type filters, and tag filters |
| `reflect` | Synthesize memories into a reasoned answer with optional structured output |

**Mental Models**

| Tool | Description |
|------|-------------|
| `list_mental_models` | List pinned reflections for a bank |
| `get_mental_model` | Get a specific mental model |
| `create_mental_model` | Create a new mental model with optional auto-refresh trigger |
| `update_mental_model` | Update a mental model's metadata |
| `delete_mental_model` | Delete a mental model |
| `refresh_mental_model` | Regenerate a mental model's content |

**Directives**

| Tool | Description |
|------|-------------|
| `list_directives` | List directives that guide memory processing |
| `create_directive` | Create a new directive |
| `delete_directive` | Delete a directive |

**Memory Browsing**

| Tool | Description |
|------|-------------|
| `list_memories` | Browse memories with filtering and pagination |
| `get_memory` | Get a specific memory by ID |
| `delete_memory` | Delete a specific memory |

**Documents**

| Tool | Description |
|------|-------------|
| `list_documents` | List ingested documents |
| `get_document` | Get a specific document |
| `delete_document` | Delete a document and its linked memories |

**Operations**

| Tool | Description |
|------|-------------|
| `list_operations` | List async operations with status filtering |
| `get_operation` | Check operation status and progress |
| `cancel_operation` | Cancel a pending or running operation |

**Tags & Bank Management**

| Tool | Description |
|------|-------------|
| `list_tags` | List unique tags used in a bank |
| `get_bank` | Get bank profile (name, mission, disposition) |
| `get_bank_stats` | Get bank statistics (multi-bank only) |
| `update_bank` | Update bank name or mission |
| `delete_bank` | Delete an entire bank and all its data |
| `clear_memories` | Clear memories without deleting the bank |
| `list_banks` | List all memory banks (multi-bank only) |
| `create_bank` | Create or configure a memory bank (multi-bank only) |

For detailed parameter documentation, see the [MCP Server reference](/developer/mcp-server#available-tools).

## Environment Variables

All standard [Hindsight configuration variables](/developer/configuration) are supported. Key ones for local use:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HINDSIGHT_API_LLM_API_KEY` | Yes* | — | API key for your LLM provider |
| `HINDSIGHT_API_LLM_PROVIDER` | No | `openai` | LLM provider (`openai`, `anthropic`, `ollama`, etc.) |
| `HINDSIGHT_API_LLM_MODEL` | No | `gpt-4o-mini` | Model name |
| `HINDSIGHT_API_DATABASE_URL` | No | `pg0://hindsight-mcp` | Override the database URL |
| `HINDSIGHT_API_PORT` | No | `8888` | Port to listen on |
| `HINDSIGHT_API_LOG_LEVEL` | No | `info` | Log level |

*Not required when using a local provider like Ollama.

## Troubleshooting

### Slow first startup

The first startup downloads the local embedding model (~100MB) and initializes the database. Subsequent starts are faster.

### Port already in use

Set a different port:

```bash
HINDSIGHT_API_LLM_API_KEY=sk-... HINDSIGHT_API_PORT=9000 uvx --from hindsight-api hindsight-local-mcp
```

Then update your MCP client URL to `http://localhost:9000/mcp/`.

### Checking logs

Set `HINDSIGHT_API_LOG_LEVEL=debug` for verbose output:

```bash
HINDSIGHT_API_LLM_API_KEY=sk-... HINDSIGHT_API_LOG_LEVEL=debug uvx --from hindsight-api hindsight-local-mcp
```
