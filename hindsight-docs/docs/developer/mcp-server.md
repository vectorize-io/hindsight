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
| **Single-bank** | `/mcp/{bank_id}/` | 27 tools (memory, mental models, directives, documents, operations, tags, bank management) | Implicit from URL |
| **Multi-bank** | `/mcp/` | All 30 tools including `list_banks`, `create_bank`, `get_bank_stats` | Explicit `bank_id` parameter on each tool |

**Single-bank mode** (recommended) scopes all operations to the bank in the URL. Tools don't expose a `bank_id` parameter.

**Multi-bank mode** exposes all tools with an optional `bank_id` parameter, plus bank management tools (`list_banks`, `create_bank`, `get_bank_stats`).

## Tool Metadata and Instructions

Hindsight can append deployment-specific guidance to the `retain` and `recall` MCP tool descriptions. Set `HINDSIGHT_API_MCP_INSTRUCTIONS` on the API server when clients should see local rules, such as which tags to use or which memories should be retained.

```bash
export HINDSIGHT_API_MCP_INSTRUCTIONS="Use project:<name> tags for project-specific memories."
```

MCP clients that read tool annotations also receive safety hints from the built-in tools:

- Read-only operations such as `recall`, `reflect`, `list_*`, and `get_*` are marked with `readOnlyHint: true`.
- Delete, clear, and invalidate operations are marked with `destructiveHint: true`.
- `openWorldHint` is `false` for the built-in tools because Hindsight operates on its configured memory store rather than the open internet.
- Write operations such as `retain`, `create_*`, `update_*`, `refresh_mental_model`, and `cancel_operation` are not marked destructive.

---

## Available Tools

### retain

Store information to long-term memory.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | Yes | The fact or memory to store |
| `context` | string | No | Category for the memory (default: `general`) |
| `timestamp` | string | No | ISO 8601 timestamp for when the event occurred |
| `tags` | list[string] | No | Tags for organizing and filtering this memory |
| `metadata` | object | No | Key-value metadata to attach (e.g., `{"source": "slack"}`) |
| `document_id` | string | No | Associate this memory with an existing document |

**Example:**
```json
{
  "name": "retain",
  "arguments": {
    "content": "User prefers Python over JavaScript for backend development",
    "context": "programming_preferences",
    "tags": ["user:alice", "preferences"]
  }
}
```

**When to use:**
- User shares personal facts, preferences, or interests
- Important events or milestones are mentioned
- Decisions, opinions, or goals are stated
- Work context or project details are discussed

---

### sync_retain

Store information to long-term memory and wait for completion. Unlike [`retain`](#retain) (which is asynchronous), `sync_retain` blocks until the memory is fully stored and immediately available for recall — useful for read-after-write flows where you query right after storing.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `content` | string | Yes | The fact or memory to store |
| `context` | string | No | Category for the memory (default: `general`) |
| `timestamp` | string | No | ISO 8601 timestamp for when the event occurred |
| `tags` | list[string] | No | Tags for organizing and filtering this memory |
| `metadata` | object | No | Key-value metadata to attach (e.g., `{"source": "slack"}`) |
| `document_id` | string | No | Associate this memory with an existing document |

**Example:**
```json
{
  "name": "sync_retain",
  "arguments": {
    "content": "User prefers Python over JavaScript for backend development",
    "context": "programming_preferences",
    "tags": ["user:alice", "preferences"]
  }
}
```

**When to use:**
- You need the memory queryable immediately after storing (read-after-write)
- A workflow step depends on the stored memory being available before continuing
- Otherwise prefer `retain` (asynchronous) to avoid blocking on storage

---

### recall

Search memories to provide personalized responses.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Natural language search query |
| `max_tokens` | integer | No | Maximum tokens to return (default: 4096) |
| `budget` | string | No | Search thoroughness: `low`, `mid`, or `high` (default: `high`) |
| `types` | list[string] | No | Filter by fact type: `world`, `experience`, `observation`. Defaults to all |
| `tags` | list[string] | No | Filter memories by tags. Omit for no filter |
| `tags_match` | string | No | `any` (default), `all`, `any_strict`, `all_strict`, or `exact`. With `exact`, pass `tags: []` to select the untagged/global scope |
| `tag_groups` | list[object] | No | Compound boolean tag filter. Mutually exclusive with `tags`; each leaf has its own `match` value |
| `query_timestamp` | string | No | ISO 8601 timestamp — recall as if asking at this point in time; anchors relative temporal expressions and recency scoring |
| `min_scores` | object | No | Optional per-stage score floors, e.g. `{"reranker": 0.5}`. Keys: `semantic`/`keyword` (retrieval-level cutoffs), `reranker`/`final` (post-ranking). All inclusive and AND-ed; omit for no filtering. Reranker scores aren't calibrated across queries — calibrate before use |
| `temporal_window` | object | No | An explicit `{"start": ISO, "end": ISO}` period to search over, used instead of reading dates out of the query text. Ranks memories dated inside the window higher; it does not drop the ones outside it, so it can't restrict results to a period |

**Example:**
```json
{
  "name": "recall",
  "arguments": {
    "query": "What are the user's programming language preferences?",
    "tags": ["preferences"],
    "budget": "high"
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
| `max_tokens` | integer | No | Maximum tokens in the response (default: 4096) |
| `response_schema` | object | No | JSON Schema for structured output. When provided, the response includes a `structured_output` field |
| `tags` | list[string] | No | Scope memories, observations, mental models, and tagged directives. Omitted tags leave memory retrieval unfiltered but load only untagged directives |
| `tags_match` | string | No | `any` (default), `all`, `any_strict`, `all_strict`, or `exact`. Untagged directives remain global in every mode |
| `include_trace` | boolean | No | Include `tool_trace` and `llm_trace` debugging output. Defaults to `false` to keep responses small |

The MCP tool forwards `tags_match` only when `tags` is present. Pass
`tags: []` with `tags_match: "exact"` to select the empty/global scope for raw
facts, observations, and mental models; directive loading also selects only
untagged directives.

**Example:**
```json
{
  "name": "reflect",
  "arguments": {
    "query": "Based on my past decisions, what architectural style do I prefer?",
    "budget": "mid",
    "tags": ["architecture"]
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
| `tags_match` | string | No | How the model's tags are matched against memories on refresh: `any`, `all`, `any_strict`, `all_strict`, or `exact`. See the note below on the default |
| `trigger` | object | No | Refresh policy — see [Trigger settings](#trigger-settings) |
| `max_tokens` | integer | No | Maximum tokens for model content (default: 2048) |
| `trigger_refresh_after_consolidation` | boolean | No | Legacy shorthand for `trigger.refresh_after_consolidation` |

`trigger` is the preferred form for refresh settings; `tags_match` and
`trigger_refresh_after_consolidation` remain as shorthands for existing
integrations. Setting the same field both ways is an error rather than one
silently winning.

:::warning Tagged models default to `all_strict`
When a mental model has `tags` but no explicit `tags_match`, its refresh matches memories with **`all_strict`** — a memory must carry **every** one of the model's tags to be included. If your memories use narrow, single-topic tags (e.g. `["project:status"]`) while the model is tagged broadly (e.g. `["projects", "mental-model"]`), the refresh filters out everything and the content comes back empty.

Pass `tags_match: "any"` (the same default that `recall` and `reflect` use) to match memories that carry *any* of the model's tags:

```json
{
  "name": "create_mental_model",
  "arguments": {
    "name": "Current projects",
    "source_query": "Which projects is the user currently working on?",
    "tags": ["projects", "mental-model"],
    "tags_match": "any"
  }
}
```
:::

**Example:**
```json
{
  "name": "create_mental_model",
  "arguments": {
    "name": "Team Directory",
    "source_query": "Who works here and what do they do?",
    "tags": ["team", "people"],
    "tags_match": "any"
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
| `trigger` | object | No | Refresh-policy fields to change — see [Trigger settings](#trigger-settings). This is a patch: omitted fields keep their current values |
| `tags_match` | string | No | Legacy shorthand for `trigger.tags_match` |
| `trigger_refresh_after_consolidation` | boolean | No | Auto-refresh after consolidation. Only set when you want to change this setting |

---

### Trigger settings

`trigger` carries a mental model's (or knowledge page's) refresh policy: **when**
it rebuilds itself and **what** it rebuilds from. It accepts every field the HTTP
API accepts, so anything you can configure through `PATCH /mental_models/{id}`
you can also configure from an agent over MCP.

| Field | Type | Description |
|-------|------|-------------|
| `mode` | string | `full` regenerates the content from scratch on each refresh; `delta` makes surgical edits, preserving unchanged sections byte-for-byte. Delta falls back to full when there is no existing content or the source query changed |
| `refresh_after_consolidation` | boolean | Rebuild after each memory consolidation |
| `refresh_cron` | string | UTC five-field cron, e.g. `0 3 * * *` for daily at 03:00 UTC. Only runs when the model is stale, so an unchanged scope costs no LLM call |
| `min_refresh_interval_seconds` | integer | Floor between two *automatic* refreshes. Triggers arriving sooner fold into one queued refresh, so a burst of retains costs one rebuild. Explicit refreshes ignore it |
| `fact_types` | list[string] | Which of `world`, `experience`, `observation` the refresh retrieves. Omit for all three |
| `exclude_mental_models` | boolean | Exclude **all** mental models from the refresh, so a model never reflects on its siblings |
| `exclude_mental_model_ids` | list[string] | Exclude specific mental models by ID |
| `tags_match` | string | How the model's tags select memories: `any`, `all`, `any_strict`, `all_strict`, `exact` |
| `tag_groups` | list[object] | Compound boolean tag expressions used *instead of* the model's flat tags |
| `include_chunks` | boolean | Whether the refresh's internal recall returns raw chunk text |
| `recall_max_tokens` | integer | Token budget for facts from the refresh's internal recall |
| `recall_chunks_max_tokens` | integer | Token budget for raw chunks from the refresh's internal recall |
| `response_schema` | object | JSON Schema for structured output, stored alongside the markdown under `reflect_response.structured_output` |
| `keep_trace` | boolean | Record how each refresh reached its result under `reflect_response.trace`. The only way to diagnose a cron- or consolidation-driven refresh after the fact |

`refresh_after_consolidation` and `refresh_cron` are **mutually exclusive** — a
model refreshes either after consolidation or on a schedule, never both. Setting
one clears the other.

**On create**, omitted fields take the engine defaults (`mode: full`, no
schedule, all fact types); for a knowledge page they take the page defaults
(`mode: delta`, observation-only, auto-refresh, siblings excluded).

**On update, `trigger` is a patch**: only the fields you send change, so putting
a model on a cron schedule keeps its `fact_types` and `mode`. Send an explicit
`null` to clear a nullable setting. Read the current policy back with
`get_mental_model` (detail `content` or `full`) if you want to inspect before
changing.

```json
{
  "name": "update_mental_model",
  "arguments": {
    "mental_model_id": "team-conventions",
    "trigger": { "refresh_cron": "0 3 * * *", "fact_types": ["observation"] }
  }
}
```

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

### clear_mental_model

Clear a mental model's content while keeping its definition. After clearing, call `refresh_mental_model` to rebuild it from the latest memories.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mental_model_id` | string | Yes | The ID of the mental model to clear |

---

### list_banks (multi-bank mode only)

List available memory banks, most recently written first. The response carries the
total number of matching banks alongside the page, so large deployments can be
walked with `offset`.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | No | Case-insensitive substring matched against bank ID and name |
| `limit` | integer | No | Maximum number of banks to return (default: 100) |
| `offset` | integer | No | Number of banks to skip (default: 0) |

---

### create_bank (multi-bank mode only)

Create a new memory bank or retrieve an existing one.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `bank_id` | string | Yes | The ID for the new bank |
| `name` | string | No | Human-friendly name for the bank |
| `mission` | string | No | Mission describing who the agent is and what they're trying to accomplish |

---

### list_directives

List directives in a bank. This management tool does not use reflect's
directive-isolation behavior: omitting `tags` lists every directive, including
tagged directives.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tags` | list[string] | No | Filter using `any` matching. When present, returns untagged/global directives plus directives sharing at least one tag. When omitted or empty, returns all directives |
| `active_only` | boolean | No | Only return active directives (default: `true`) |

---

### create_directive

Create a new directive in a bank.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Human-readable name for the directive |
| `content` | string | Yes | The directive content/instruction |
| `priority` | integer | No | Priority level (higher = more important) |
| `is_active` | boolean | No | Whether the directive is active (default: `true`) |
| `tags` | list[string] | No | Execution scope for the directive. Empty/omitted (default) means global; non-empty means reflect must use a matching tag scope |

---

### delete_directive

Delete a directive by ID.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `directive_id` | string | Yes | The ID of the directive to delete |

---

### list_memories

Browse stored memories with optional filtering and pagination.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `type` | string | No | Filter by fact type: `world`, `experience`, or `observation` |
| `q` | string | No | Search query to filter memories |
| `limit` | integer | No | Maximum number of results (default: 100) |
| `offset` | integer | No | Number of results to skip for pagination (default: 0) |

---

### get_memory

Retrieve a specific memory by ID.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `memory_id` | string | Yes | The ID of the memory to retrieve |

---

### list_documents

List documents that have been ingested into the memory bank.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `q` | string | No | Search query to filter documents |
| `limit` | integer | No | Maximum number of results (default: 100) |

---

### get_document

Retrieve a specific document by ID, including its metadata.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `document_id` | string | Yes | The ID of the document to retrieve |

---

### delete_document

Delete a document and all memories linked to it.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `document_id` | string | Yes | The ID of the document to delete |

---

### list_operations

List async operations (retain processing, mental model refresh, etc.) with optional status filtering.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `status` | string | No | Filter by status: `pending`, `running`, `completed`, `failed`, `cancelled` |
| `limit` | integer | No | Maximum number of results (default: 100) |

---

### get_operation

Get the status and details of an async operation.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `operation_id` | string | Yes | The ID of the operation to check |

---

### cancel_operation

Cancel a pending or running async operation.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `operation_id` | string | Yes | The ID of the operation to cancel |

---

### list_tags

List all unique tags used in a bank, optionally filtered by pattern.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `q` | string | No | Glob pattern to filter tags (e.g., `project:*`) |
| `limit` | integer | No | Maximum number of results (default: 100) |

---

### get_bank

Get information about a memory bank, including its name, mission, and disposition.

---

### get_bank_stats (multi-bank mode only)

Get statistics for a memory bank (node/link counts).

---

### update_bank

Update a memory bank's configuration. Updates the bank's name and/or any bank-level configuration fields — only provided fields are updated; omitted fields remain unchanged.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | No | Human-friendly display name for the bank |
| `mission` | string | No | **Deprecated** — alias for `config_updates.reflect_mission` |
| `config_updates` | object | No | Dictionary of configuration fields to update. Supports all bank-configurable fields (see below). Non-configurable or credential fields are rejected |

The `config_updates` object accepts any bank-configurable field by its Python field name, including:

- `reflect_mission` — mission/context for Reflect operations
- `retain_mission` — steers what gets extracted during `retain()`
- `retain_extraction_mode` — `concise` (default), `verbose`, `custom`, `verbatim`, or `chunks`
- `retain_custom_instructions` — custom extraction prompt (active when mode is `custom`)
- `retain_chunk_size` — target maximum characters for each content chunk
- `retain_structured_chunk_size` — maximum characters for a single JSONL line or conversation turn to keep whole
- `retain_chunk_batch_size` — number of chunks to process in parallel
- `enable_observations` — toggle observation consolidation after `retain()`
- `observations_mission` — controls observation synthesis rules
- `disposition_skepticism` — critical evaluation level (1–5)
- `disposition_literalism` — literal vs. abstract interpretation (1–5)
- `disposition_empathy` — emotional context consideration (1–5)
- `entity_labels` — controlled vocabulary for entity classification
- `entities_allow_free_form` — allow labels outside `entity_labels`
- `recall_include_chunks` — include raw chunks in recall results
- `recall_max_tokens` — max tokens for recall results
- `mcp_enabled_tools` — tool allowlist for this bank

---

### delete_bank

Permanently delete a memory bank and all its data (memories, documents, entities, mental models).

---

### clear_memories

Clear all memories from a bank without deleting the bank itself. Optionally filter by fact type to only clear specific kinds of memories.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `type` | string | No | Fact type to clear: `world`, `experience`, or `observation`. If not specified, clears all |

---

### get_knowledge_base_tree

Browse the knowledge base as a nested tree of folders and pages. Each page reports `is_stale`: `false` means it is provably up to date, `true` means the bank has been written to since the page last refreshed.

---

### search_knowledge_base

Find knowledge pages by relevance (hybrid BM25 + vector search over page names and content). Returns ranked pages with a snippet each.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | What to search for |
| `limit` | integer | No | Maximum pages to return, 1–50 (default: 10) |

---

### get_knowledge_page

Read a knowledge page as a markdown document (YAML frontmatter + synthesized body).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `page_id` | string | Yes | The ID of the page to read (a `kp-...` node id) |

---

### create_knowledge_folder

Create a folder in the knowledge base. Folders group pages and hold no content of their own.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Folder name |
| `parent_id` | string | No | Parent folder id (a `kf-...` node id). Omit to create at the top level |

---

### create_knowledge_page

Create a page — a living document whose content is synthesized from the bank's memories by running `source_query`. Content is generated asynchronously; use the returned `operation_id` to track completion.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Page name (unique within its folder) |
| `source_query` | string | Yes | The question this page answers and rebuilds itself from |
| `parent_id` | string | No | Parent folder id (a `kf-...` node id). Omit to create at the top level |
| `tags` | array | No | Tags scoping which memories the page is built from |
| `max_tokens` | integer | No | Maximum tokens for the generated content (default: 4096) |
| `trigger` | object | No | Refresh policy — see [Trigger settings](#trigger-settings). Omitted fields keep the knowledge-page defaults: `delta` rebuilds from consolidated observations after each consolidation, ignoring sibling pages |
| `refresh_after_consolidation` | boolean | No | Legacy shorthand for `trigger.refresh_after_consolidation` |

Set `trigger.refresh_cron` to move a page off consolidation-driven rebuilds and
onto a fixed UTC schedule — the two are mutually exclusive, so the cron clears
the auto-refresh while leaving the page's `mode` and `fact_types` alone.

---

### update_knowledge_node

Rename or move a folder/page, and/or update a page's options. Only the arguments you pass are changed. Changing `source_query` schedules an async refresh so the page rebuilds against the new question.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `node_id` | string | Yes | The folder (`kf-...`) or page (`kp-...`) to update |
| `name` | string | No | New name for the node |
| `parent_id` | string | No | Folder id to move the node into, or `"root"` to move it to the top level |
| `source_query` | string | No | Pages only — the new question the page answers |
| `tags` | array | No | Pages only — replacement tag list (pass `[]` to clear) |
| `max_tokens` | integer | No | Pages only — new maximum tokens for the generated content |
| `trigger` | object | No | Pages only — refresh-policy fields to change; see [Trigger settings](#trigger-settings). This is a patch: omitted fields keep their current values |
| `refresh_after_consolidation` | boolean | No | Pages only — legacy shorthand for `trigger.refresh_after_consolidation` |

---

### delete_knowledge_node

Delete a folder or page and its whole subtree. Each deleted page takes its backing mental model with it.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `node_id` | string | Yes | The folder (`kf-...`) or page (`kp-...`) to delete |

:::note
Exporting the knowledge base is deliberately not an MCP tool — it returns the whole
bank as a single markdown bundle. Use the HTTP endpoint
`GET /v1/default/banks/{bank_id}/knowledge-base/export` instead.
:::

---

## Integration with AI Assistants

The MCP server can be used with any MCP-compatible AI assistant. See the [Authentication](#authentication) section above for Claude Code and Claude Desktop configuration examples.

Each user can have their own configuration pointing to their personal memory bank using either:
- A bank-specific URL path like `/mcp/alice/` (recommended)
- The `X-Bank-Id` header
