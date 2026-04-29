---
name: agent-knowledge
description: Your long-term knowledge pages. Read them at session start. Create new pages when you learn something worth remembering across sessions. Pages auto-update from your conversations via Hindsight.
---

# Agent Knowledge

You have knowledge pages that persist across sessions and auto-update from your conversations.

**How it works:** Your conversations are automatically retained into a Hindsight memory bank. The system extracts observations and uses them to keep your pages current. Each page has a "source query" — a question the system re-answers after every consolidation cycle to rebuild the page content. You create pages; the system maintains them.

## At session start

Call `agent_knowledge_list_pages` to see what pages exist, then `agent_knowledge_get_page` for each one you need.

## Reading

- `agent_knowledge_list_pages()` — list page IDs and names (no content)
- `agent_knowledge_get_page(page_id)` — read the full content of a page

## Creating pages

When you learn something durable — a user preference, a working procedure, performance data — create a page immediately.

`agent_knowledge_create_page(page_id, name, source_query)`

- `page_id`: lowercase with hyphens (`editorial-preferences`)
- `source_query`: a question that produces the page content from observations

Examples:

- `"What are the user's preferences for tone, length, and formatting?"`
- `"What content strategies have performed well or poorly? Include numbers."`
- `"What are the best practices for [topic], preferring our data over generic advice?"`

## Searching memories

`agent_knowledge_recall(query)` — search across all retained conversations and documents for specific facts.

Use when pages don't cover what you need.

## Ingesting documents

`agent_knowledge_ingest(title, content)` — upload raw content into memory. Never summarize before ingesting. Save large content to a file first, read it, then pass the full text.

## Updating and deleting

- `agent_knowledge_update_page(page_id, name?, source_query?)` — change what a page tracks
- `agent_knowledge_delete_page(page_id)` — remove a page

## Important

- Pages update automatically — don't edit content directly
- State preferences clearly in your responses so the system captures them
- Create pages silently — don't announce it to the user
- Prefer fewer broad pages over many narrow ones
