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

By default, the MCP endpoint is **open** (no authentication required).

To enable authentication, configure the API key tenant extension:

```bash
export HINDSIGHT_API_TENANT_EXTENSION=hindsight_api.extensions.builtin.tenant:ApiKeyTenantExtension
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
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'
```

If the key is missing or invalid, requests will receive a `401 Unauthorized` response.

## Bank Selection

The memory bank is resolved in this priority order:

1. **URL path** (highest priority): `http://localhost:8888/mcp/my-bank/`
2. **X-Bank-Id header**: `--header "X-Bank-Id: my-bank"`
3. **Default**: Uses `HINDSIGHT_MCP_BANK_ID` env var (default: "default")

## Per-Bank Endpoints

Unlike traditional MCP servers where tools require explicit identifiers, Hindsight uses **per-bank endpoints**. The `bank_id` is part of the URL path, so tools don't need to specify which bank to use—it's implicit from the connection.

This design:
- **Simplifies tool usage** — no need to pass `bank_id` with every call
- **Enforces isolation** — each MCP connection is scoped to a single bank
- **Enables multi-tenant setups** — connect different users to different endpoints

## Two Modes

The MCP server operates in two modes depending on the URL:

| Mode | URL | Tools | bank_id |
|------|-----|-------|---------|
| **Single-bank** | `/mcp/{bank_id}/` | Memory + mental model tools | Implicit from URL |
| **Multi-bank** | `/mcp/` | All tools including bank management | Explicit `bank_id` parameter on each tool |

**Single-bank mode** (recommended) scopes all operations to the bank in the URL. Tools don't expose a `bank_id` parameter.

**Multi-bank mode** exposes all tools with an optional `bank_id` parameter, plus bank management tools (`list_banks`, `create_bank`).

---

## Available Tools

### retain

Store information to long-term memory.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | Yes | The fact or memory to store |
| `context` | string | No | Category for the memory (default: `general`) |
| `timestamp` | string | No | ISO 8601 timestamp for when the event occurred |

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
| `max_tokens` | integer | No | Maximum tokens to return (default: 4096) |

**Example:**
```json
{
  "name": "recall",
  "arguments": {
    "query": "What are the user's programming language preferences?"
  }
}
```

**When to use:**
- Start of conversation to recall relevant context
- Before making recommendations
- When user asks about something they may have mentioned before
- To provide continuity across conversations

---

### reflect

Generate thoughtful analysis by synthesizing stored memories with the bank's personality.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | The question or topic to reflect on |
| `context` | string | No | Optional context about why this reflection is needed |
| `budget` | string | No | Search budget: `low`, `mid`, or `high` (default: `low`) |

**Example:**
```json
{
  "name": "reflect",
  "arguments": {
    "query": "Based on my past decisions, what architectural style do I prefer?",
    "budget": "mid"
  }
}
```

**When to use:**
- When reasoned analysis is needed, not just fact retrieval
- Questions like "What should I do?" rather than "What did I say?"
- Synthesizing patterns across multiple memories

---

### create_mental_model

Create a mental model — a living document that stays current with your memories. Mental models are pre-computed reflections that get automatically refreshed as new memories are stored.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Human-readable name for the mental model |
| `source_query` | string | Yes | The query used to generate and refresh the model |
| `mental_model_id` | string | No | Custom ID (alphanumeric lowercase with hyphens). Auto-generated if not provided |
| `tags` | list[string] | No | Tags for organizing and filtering models |
| `max_tokens` | integer | No | Maximum tokens for model content (default: 2048) |

**Example:**
```json
{
  "name": "create_mental_model",
  "arguments": {
    "name": "Team Directory",
    "source_query": "Who works here and what do they do?",
    "tags": ["team", "people"]
  }
}
```

Content generation runs asynchronously. The response includes an `operation_id` to track progress.

---

### list_mental_models

List all mental models in a bank, optionally filtered by tags.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tags` | list[string] | No | Filter models by tags |

---

### get_mental_model

Retrieve a specific mental model by ID, including its full content.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mental_model_id` | string | Yes | The ID of the mental model to retrieve |

---

### update_mental_model

Update a mental model's metadata or settings.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mental_model_id` | string | Yes | The ID of the mental model to update |
| `name` | string | No | New name |
| `source_query` | string | No | New source query |
| `tags` | list[string] | No | New tags |
| `max_tokens` | integer | No | New max tokens |

---

### delete_mental_model

Permanently delete a mental model.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mental_model_id` | string | Yes | The ID of the mental model to delete |

---

### refresh_mental_model

Re-generate a mental model's content from the latest memories. Runs asynchronously.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mental_model_id` | string | Yes | The ID of the mental model to refresh |

---

### list_banks (multi-bank mode only)

List all available memory banks.

---

### create_bank (multi-bank mode only)

Create a new memory bank or retrieve an existing one.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `bank_id` | string | Yes | The ID for the new bank |
| `name` | string | No | Human-friendly name for the bank |
| `mission` | string | No | Mission describing who the agent is and what they're trying to accomplish |

---

## Integration with AI Assistants

The MCP server can be used with any MCP-compatible AI assistant. See the [Authentication](#authentication) section above for Claude Code and Claude Desktop configuration examples.

Each user can have their own configuration pointing to their personal memory bank using either:
- A bank-specific URL path like `/mcp/alice/` (recommended)
- The `X-Bank-Id` header
