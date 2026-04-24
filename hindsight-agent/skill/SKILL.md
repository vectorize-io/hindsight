---
name: agent-knowledge
description: Manage your long-term knowledge. Read existing topic pages before acting. Create new topic pages when you discover a recurring concern worth tracking across sessions. Query memories for deeper research. The system automatically keeps pages up to date from your conversations.
---

# Agent Knowledge

Your knowledge is stored as topic pages that the system keeps updated automatically from your conversations. You **read** pages, **create** new ones when needed, and **recall** memories for deeper research. You never edit page content directly — the system handles that.

All commands use `hindsight-agent <agent-id>`. To find your agent ID:
1. Check if `{{AGENT_ID}}` is set above (setup bakes it in). If it's a placeholder or wrong, continue.
2. Your agent ID is usually your profile/agent name (e.g., your Hermes profile name or OpenClaw agent name).
3. If unsure, run `hindsight-agent list` to see all configured agents and pick the one that matches you.

## Mandatory startup sequence

Run these silently at the start of every session:

```bash
hindsight-agent pages list <your-agent-id>
```

Read the pages relevant to the current task. If the list is empty, that's fine — create pages as you learn things (see below).

## Reading pages

```bash
# List all pages (names + content)
hindsight-agent pages list {{AGENT_ID}}

# Read one specific page
hindsight-agent pages get {{AGENT_ID}} <page_id>
```

## Recalling memories

Use recall to search across all retained knowledge — conversations, reference documents, observations. This is useful when pages don't cover what you need, or when you want specific details.

```bash
# Search memories
hindsight-agent recall {{AGENT_ID}} "<natural language query>"

# Limit results
hindsight-agent recall {{AGENT_ID}} "<query>" -n 5

# Filter by type (world, experience, observation)
hindsight-agent recall {{AGENT_ID}} "<query>" --type observation
```

Use recall when:
- You need specific facts not covered by your pages
- You want to verify something before making a decision
- You're looking for evidence to support a recommendation
- You want to check what reference documents say about a topic

## Listing reference documents

```bash
hindsight-agent documents {{AGENT_ID}}
```

This shows what content has been retained into your memory — reference documents, conversation transcripts, etc. Use it to understand what knowledge is available for recall.

## Creating pages

When you discover a recurring topic worth tracking across sessions — user preferences, a procedure that works, performance data — create a page for it. Use your judgment.

```bash
hindsight-agent pages create {{AGENT_ID}} "<Page Name>" "<source_query>"
```

Or with a custom ID:
```bash
hindsight-agent pages create {{AGENT_ID}} "<Page Name>" "<source_query>" --id <page-id>
```

**The `source_query` is the key field.** It's a question the system will re-ask on every consolidation to rebuild the page content from your accumulated observations. Write it using the patterns below.

### Source query patterns

Use these patterns to write effective source queries:

**For best practices (combining reference docs with user feedback):**
```
What are the best practices for [topic], combining industry standards 
with what has actually worked for us? When our data contradicts general 
advice, prefer our data and note the deviation.
```

**For user preferences:**
```
What are the user's preferences for [topic], including explicit rules 
they've stated and patterns observed from their feedback and corrections?
```

**For performance/analytics:**
```
What [topic] strategies have performed well or poorly based on analytics 
and user feedback? Include specific numbers when available. What patterns 
emerge about what works vs what doesn't?
```

**For procedures:**
```
What is the current procedure for [topic]? Include steps, tools used, 
and any lessons learned from past attempts.
```

**When to create a page:**
- The user stated a durable preference or rule — do it immediately, don't wait
- You discovered a procedure that works and want to remember it
- You have performance data that should inform future decisions
- On your first session with no pages: create at least one broad page for core preferences

**When NOT to create a page:**
- One-off facts (just acknowledge and move on — the system retains the conversation)
- Things that are already covered by an existing page
- Agent internals, tool usage, or delivered content

## Updating a page's source query

If a page's scope needs to change — broader, narrower, or refocused — update its source_query. The system will re-synthesize the content on next consolidation.

```bash
hindsight-agent pages update {{AGENT_ID}} <page_id> --source-query "Updated question..."
```

You can also rename a page:
```bash
hindsight-agent pages update {{AGENT_ID}} <page_id> --name "Better Name"
```

## Deleting a page

If a page is redundant, outdated, or was a mistake — delete it:

```bash
hindsight-agent pages delete {{AGENT_ID}} <page_id>
```

Do this silently. Don't ask the user for permission to clean up your own knowledge.

## How pages stay current

1. Every conversation turn is automatically retained by the system
2. The system extracts observations from your conversations
3. After consolidation, pages re-run their source_query against the latest observations
4. Next time you read the page, the content reflects the latest user feedback

You don't need to update pages manually. Just acknowledge user feedback in one clear sentence so the retain pipeline captures it cleanly, and the page will update itself.

## Rules

- **Never edit page content directly** — the system synthesizes it from observations
- **Never ask the user about knowledge structure** — which pages exist, naming, organization. That's your decision, invisible to the user.
- **Create pages silently** — don't announce "I'm creating a page for X". Just do it.
- **Prefer fewer broader pages** — one "preferences" page is better than three narrow ones
- **Never surface errors about the knowledge system to the user** — if a command fails, handle it silently or retry. The user should not know about the infrastructure.
- **Use recall for ad-hoc research** — pages are for durable knowledge; recall is for one-off lookups
