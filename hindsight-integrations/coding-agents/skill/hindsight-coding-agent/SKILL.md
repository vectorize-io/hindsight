---
name: hindsight-coding-agent
description: Everyday use of this machine's Hindsight coding-agent memory — the plugin behind the 🧠 banner. Use when the user says "store/remember this in hindsight", asks what the memory or its pages are, or a memory needs correcting. For installing, configuring or debugging the plugin, use hindsight-coding-agent-setup instead.
---

<!-- GENERATED from README.md (its skill:begin regions) + skill-src/preamble.md.
     Edit those, then run: npm run skill:build -->

# Hindsight Coding-Agent Memory

This machine runs the `hindsight-coding-agents` plugin: long-term project memory for coding
sessions, backed by a Hindsight server. You (the agent) are already wired into it — this skill
explains what happens automatically and which tools you have. Installing, updating, configuration and
diagnosing problems live in the `hindsight-coding-agent-setup` skill.

## What happens automatically (no action needed)

- **Per-repo memory bank**: each repository resolves to a bank (shown in the session banner:
  `↳ memory bank “coding-agent::<repo>”`). Worktrees share the main repo's bank.
- **Ingestion builds itself**: on first open, the bank is seeded from recent commit messages and a
  read-only codebase survey; every session start, a background engine tops it up (new commits, new
  conversations) and keeps 5 pages current. There is NO ingest command to run.
- **Session synthesis**: by default, the first prompt of a session triggers one deep memory
  synthesis (`reflect`) injected into context. `autoInject` switches the source: `pages` (page
  search hits), `recall` (recalled observations), or `none` (nothing injected; the agent
  searches the pages first and reflects only when they are too shallow).
- **Write-back**: the session transcript is retained into the bank automatically at session end
  (per-turn on opencode). The user never needs to "save" a conversation.
- **Prime Agent 0.10+ is the exception**: it has no hooks, so nothing is injected. Call
  `hindsight_reflect` yourself first (`await mcp.call_tool("hindsight", "hindsight_reflect", …)`, with the server name your
  install chose if it is not `hindsight`);
  that call also starts the server that saves the session.

## Storing things deliberately

When the user says "store this in hindsight" / "remember this":

- The **current conversation** is captured automatically at session end — say so; no tool needed.
- An **external document, notes, or durable findings** → `hindsight_ingest_document(title, content)`.
- A **new feature/initiative being started** → `hindsight_capture_initiative(title, summary)`,
  right after the plan is agreed and before code is written.
- A **plan that materially changed** (goal, scope, or rationale — including mid-implementation) →
  call `hindsight_capture_initiative` again with `page_id` set to that initiative's page
  id, summarising the _current_ intent. Same page, updated plan — never a second page. Trivial
  course-corrections don't count.

If those two tools are not in your tool list, writing is turned off for this setup
(`writeTools: false`): don't try to call them, and tell the user that only the conversation itself
is saved.

## Retrieving

- `hindsight_search_pages(query)` — FIRST STOP for project questions (components,
  conventions, past decisions, initiatives). Server-side hybrid search, fast.
- `hindsight_read_page(page_id)` / `hindsight_list_pages` — read pages fully.
- `hindsight_reflect(query)` — deep reasoning over the whole memory for WHY questions and exact
  decided values; slower (seconds), use deliberately.
- Credit visibly whenever memory informs an answer: start that part with
  `🧠 From Hindsight memory (<page>): …` — and never credit memory that didn't contribute.

## Correcting wrong or stale memory

If you verify that something Hindsight served is wrong or outdated (the code, git, or an external
source contradicts it), FIX THE RECORD — don't just ignore it. Call
`hindsight_ingest_document` with:

- **title**: `Correction: <topic>` (e.g. `Correction: retry policy 4xx set`)
- **content**: (1) what memory claimed, (2) what is verifiably true now, (3) the evidence you
  checked (file/commit/output). Quote exact values verbatim.

Newer facts supersede older ones in retrieval, so one clear correction permanently outranks the
stale memory. Do this whenever you catch a wrong injected memory, a stale page claim, or
an outdated decision — silent disregard leaves the trap armed for the next session.
