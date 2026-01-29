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

The MCP server can be used with any MCP-compatible AI assistant.

### Claude Desktop Configuration

To connect Claude Desktop to a specific memory bank:

```json
{
  "mcpServers": {
    "hindsight-alice": {
      "url": "http://localhost:8888/mcp/alice/"
    }
  }
}
```

Each user can have their own MCP server configuration pointing to their personal memory bank.
