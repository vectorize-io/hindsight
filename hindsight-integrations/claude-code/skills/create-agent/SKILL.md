---
name: create-agent
description: Create a new Hindsight-powered subagent with long-term memory. Use when the user wants a specialized agent that learns and remembers across sessions.
---

# Create Hindsight Agent

Create a new subagent with long-term memory powered by Hindsight.

## What to ask the user

1. **Agent name** — lowercase with hyphens (e.g. `code-reviewer`, `project-manager`)
2. **What the agent does** — one sentence for the description
3. **Any initial knowledge to seed** — files, docs, or context to ingest

## Create the subagent file

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

<ADD AGENT-SPECIFIC INSTRUCTIONS HERE — what it reviews, how it responds, what domain knowledge it applies>
```

## Rules

- Always include `mcpServers: [hindsight]` — this wires up the Hindsight memory tools
- Keep the startup steps and tool instructions verbatim — they're the Hindsight scaffolding
- Customize the description (used by Claude to decide when to delegate)
- Add agent-specific sections AFTER the Hindsight scaffolding (e.g. "## What I review for", "## My approach")
- Do NOT hardcode a `bank_id` — the plugin resolves it at runtime from config

## After creation

1. Confirm the file was written
2. If the user has seed content (files, docs, URLs), offer to ingest it using `agent_knowledge_ingest`
3. Tell the user they can invoke the agent with `@<agent-name>` or Claude will auto-delegate based on the description
4. Suggest restarting Claude Code or running `/agents` to load the new agent
