---
sidebar_position: 5
---

# MCP Server

Hindsight includes a built-in [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that allows AI assistants to store and retrieve memories directly.

## Access

The MCP server is **enabled by default** and mounted at `/mcp` on the API server. Each memory bank has its own MCP endpoint:

```
http://localhost:8888/mcp/{bank_id}/
```

For example, to connect to the memory bank `alice`:
```
http://localhost:8888/mcp/alice/
```

To disable the MCP server, set the environment variable:

```bash
export HINDSIGHT_API_MCP_ENABLED=false
```

## Authentication

By default, the MCP endpoint is **open** for local development.

To enable authentication, set an API key:

```bash
export HINDSIGHT_API_TENANT_API_KEY=your-secret-key
```

When authentication is enabled, include your API key in the `Authorization` header:

### Claude Code

```bash
claude mcp add --transport http hindsight http://localhost:8888/mcp \
  --header "Authorization: Bearer your-secret-key" \
  --header "X-Bank-Id: my-bank"
```

### Claude Desktop

Add to `~/.claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "hindsight": {
      "url": "http://localhost:8888/mcp",
      "headers": {
        "Authorization": "Bearer your-secret-key",
        "X-Bank-Id": "my-bank"
      }
    }
  }
}
```

### Direct HTTP Request

```bash
curl -X POST http://localhost:8888/mcp \
  -H "Authorization: Bearer your-secret-key" \
  -H "X-Bank-Id: my-bank" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'
```

If the key is missing or invalid, requests will receive a `401 Unauthorized` response.

## Bank Selection

Specify the memory bank via:

1. **X-Bank-Id header** (recommended): `--header "X-Bank-Id: my-bank"`
2. **URL path**: `http://localhost:8888/mcp/my-bank/`
3. **Default**: Uses `HINDSIGHT_MCP_BANK_ID` env var (default: "default")

## Per-Bank Endpoints

Unlike traditional MCP servers where tools require explicit identifiers, Hindsight uses **per-bank endpoints**. The `bank_id` is part of the URL path, so tools don't need to specify which bank to use—it's implicit from the connection.

This design:
- **Simplifies tool usage** — no need to pass `bank_id` with every call
- **Enforces isolation** — each MCP connection is scoped to a single bank
- **Enables multi-tenant setups** — connect different users to different endpoints

---

## Available Tools

### retain

Store information to long-term memory.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | Yes | The fact or memory to store |
| `context` | string | No | Category for the memory (default: `general`) |

**Example:**
```json
{
  "name": "retain",
  "arguments": {
    "content": "User prefers Python over JavaScript for backend development",
    "context": "programming_preferences"
  }
}
```

**When to use:**
- User shares personal facts, preferences, or interests
- Important events or milestones are mentioned
- Decisions, opinions, or goals are stated
- Work context or project details are discussed

---

### recall

Search memories to provide personalized responses.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Natural language search query |
| `max_results` | integer | No | Maximum results to return (default: 10) |

**Example:**
```json
{
  "name": "recall",
  "arguments": {
    "query": "What are the user's programming language preferences?"
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "fact_abc123",
      "text": "User prefers Python over JavaScript for backend development",
      "type": "world",
      "context": "programming_preferences",
      "event_date": null
    }
  ]
}
```

**When to use:**
- Start of conversation to recall relevant context
- Before making recommendations
- When user asks about something they may have mentioned before
- To provide continuity across conversations

---

## Integration with AI Assistants

The MCP server can be used with any MCP-compatible AI assistant. See the [Authentication](#authentication) section above for Claude Code and Claude Desktop configuration examples.

Each user can have their own configuration pointing to their personal memory bank using either:
- The `X-Bank-Id` header (recommended)
- A bank-specific URL path like `/mcp/alice/`
