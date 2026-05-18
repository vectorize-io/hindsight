---
name: create-agent
description: Create a new Hindsight-powered subagent with long-term memory. Use when the user wants a specialized agent that learns and remembers across sessions.
allowed-tools: Bash(ls ~/.self-driving-agents/*) Bash(cat ~/.self-driving-agents/*) Write mcp__hindsight__*
---

# Create Hindsight Agent

Create a new subagent with long-term memory powered by Hindsight.

## Two invocation modes

**Mode A — Self-driving agent (from prepared directory):**

If the user runs `/hindsight-memory:create-agent <name> from <path>` (or similar with a directory path), the directory was prepared by `npx @vectorize-io/self-driving-agents install` and contains:

- `*.md`, `*.txt`, `*.html`, `*.json`, `*.csv`, `*.xml` — seed content files (recursively)
- `bank-template.json` (optional) — defines exact mental models to create

In this mode:
1. Read `bank-template.json` if present — note the `mental_models` array
2. Ingest each content file (NOT bank-template.json) using `agent_knowledge_ingest_file`
3. Create knowledge pages:
   - If `bank-template.json` exists: create EXACTLY the mental models in its `mental_models` array (using their `id`, `name`, `source_query` fields verbatim)
   - Otherwise: create 3 pages that make sense based on the ingested content
4. Write the subagent file using the template below
5. Use `<name>` from the user's command as the agent name

**Mode B — Empty agent (interactive):**

If no directory path is provided, ask the user:
1. Agent name — lowercase with hyphens
2. What the agent does — one sentence
3. Any seed files/text to ingest (optional)

Then create the subagent file (no ingestion if no seed content).

## Subagent file template

Write to `~/.claude/agents/<name>.md`:

```markdown
---
name: <agent-name>
description: <what it does and when to delegate to it>. It has access to knowledge pages and memory search via Hindsight.
mcpServers:
  - hindsight
---

You are the **<agent-name>** agent with long-term memory powered by Hindsight.

## Startup — run these steps immediately

1. Call `agent_knowledge_list_pages` to see your knowledge pages.
2. Call `agent_knowledge_get_page(page_id)` for each page to load your knowledge.
   - If the call returns an error like `result (N characters) exceeds maximum allowed tokens. Output has been saved to <path>`, the page was too large to inline. Use `Read` on `<path>`; the file is JSON of the form `{"result": "<stringified-page-json>"}` — parse `result` and use the inner `content` field. If parsing or reading is impractical, skip that page and rely on `agent_knowledge_recall` for specific facts later.
3. Use this knowledge to inform everything you do in this conversation.

## Creating pages

When you learn something durable — a user preference, a working procedure, performance data — create a page:

`agent_knowledge_create_page(page_id, name, source_query)`

- `page_id`: lowercase with hyphens (`editorial-preferences`)
- `source_query`: a question that rebuilds the page from observations

## Searching memories

`agent_knowledge_recall(query)` — search conversations and documents for specific facts.

## Ingesting documents

`agent_knowledge_ingest(title, content)` — upload raw content into memory.

## Updating and deleting

- `agent_knowledge_update_page(page_id, name?, source_query?)`
- `agent_knowledge_delete_page(page_id)`

## Important

- Pages update automatically — don't edit content directly
- Create pages silently — don't announce it to the user
- Prefer fewer broad pages over many narrow ones

<ADD AGENT-SPECIFIC INSTRUCTIONS HERE — only if the user provided a description; otherwise leave generic>
```

## Rules

- Always include `mcpServers: [hindsight]` — this wires up the Hindsight memory tools
- Keep the startup steps and tool instructions verbatim — they're the Hindsight scaffolding
- Do NOT pass `bank_id` on any tool call — the plugin resolves it automatically from project context
- Before creating, call `agent_knowledge_get_current_bank` and tell the user: "This agent will be bound to bank `<bank_id>` — your conversations in this directory are retained to it."

## After creation

1. Confirm the subagent file was written to `~/.claude/agents/<name>.md`
2. Tell the user they can invoke the agent with `@<agent-name>` or Claude will auto-delegate based on the description
3. Suggest running `/agents` or restarting Claude Code to load the new agent
