---
title: "Claude Code Now Builds and Reads Its Own Knowledge Base"
authors: [benfrank241]
slug: "2026/08/13/knowledge-pages-coding-agents"
date: 2026-08-13T12:00
tags: [hindsight, knowledge-pages, claude-code, coding-agents, agent-memory, self-updating-docs]
description: "With Hindsight's Coding Agents integration, Claude Code surveys your repo, builds a self-healing wiki of it, keeps it current, and reads it before every task."
image: /img/blog/knowledge-pages-coding-agents.png
hide_table_of_contents: true
---

![Knowledge Pages for coding agents: Claude Code surveys your repo, builds a self-healing wiki from its memory, and reads it before every task](/img/blog/knowledge-pages-coding-agents.png)

Your coding agent is brilliant and amnesiac. It reasons about a task beautifully, then closes the session and forgets that this repo uses `pnpm`, that errors are typed Result values, and that you already ruled out a separate vector database last month. Every session relearns the codebase from scratch.

[Knowledge Pages](https://hindsight.vectorize.io/developer/knowledge-pages), shipped in Hindsight 0.9.0, fix that. They are living documents a memory bank writes about itself, and they rewrite themselves as the bank learns. The part that makes them click for coding: with the [Coding Agents integration](https://hindsight.vectorize.io/sdks/integrations/coding-agents), your agent **builds, maintains, and reads** those pages itself. Claude Code does it for you.

<!-- truncate -->

## TL;DR

- **Knowledge Pages** are self-maintaining documents synthesized from a bank's memory. Each answers one question ("What's our error-handling convention?") and heals itself as new memory lands.
- Install the Coding Agents integration and **Claude Code surveys your repo on first run**, seeding pages for architecture, conventions, key decisions, and active initiatives from your git history and past sessions.
- It **keeps them current** as you work (re-injecting the page roster every ~10 turns) and **records new work** as tracked pages via `hindsight_capture_initiative`.
- It **reads them back**: `hindsight_search_knowledge_pages` and `hindsight_read_knowledge_page` put the right page in front of the agent before it starts a task.
- One command: `npx @vectorize-io/hindsight-coding-agents install claude-code`. In our benchmark, memory cut Claude Code's corrections by **~57%**.

## What a Knowledge Page actually is

Start with the shape and the engine, because they are different things. The shape is a wiki: pages in folders, browsable, searchable, projectable to disk as markdown. The engine underneath is memory.

A page is a **projected view** over processed memory, the way a database view is not a table. Before a page is written, Hindsight has already extracted facts from your sessions, commits, and documents, deduplicated them, and reconciled their contradictions through consolidation. Your raw history stays the source of truth about *what was said*. The page is the reconciled truth about *what holds* right now. When a decision changes from X to Y, the page says Y, and can say why, instead of preserving both a paragraph apart.

That is why pages heal themselves rather than rot: they are the rendering, not the storage. Delete one and nothing is lost; it re-projects from memory on the next build.

## How your coding agent builds it for you

Here is the part that makes this automatic rather than a chore. Install the integration and, on the next session in a cold repo, an **exploration agent runs a read-only survey** under your own harness's CLI. It maps the codebase and seeds the initial pages: a component map, core concepts, conventions and patterns, key decisions and rationale, and active initiatives. From that moment on, Hindsight maintains them.

![The Knowledge view: the pages Claude Code seeded and now maintains — Architecture, Conventions, Decisions, Open initiatives — each synthesized from the repo's memory](/img/blog/knowledge-pages-tree.png)

You did not write any of these. The agent surveyed the repo, and Hindsight synthesized the pages from what it found. Each carries a green dot when it is freshly built, and the whole tree is searchable.

As you keep working, two things happen without you asking:

- **Maintenance.** Every ~10 user turns the integration refetches the pages and re-injects the current roster, so the agent is always working against the latest version (`pageRefreshEveryTurns`, default 10).
- **New work gets recorded.** When you finish brainstorming a feature and start building, the agent calls `hindsight_capture_initiative` to open a tracked page for it. Bug fixes and small chores are deliberately skipped; initiatives are not.

## A page is reconciled synthesis, not notes

Open one and it reads like something a careful teammate wrote, except nobody did.

![A rendered Knowledge Page — "System architecture" — synthesized from the repo's observations into clean prose and marked updated moments after the latest consolidation](/img/blog/knowledge-pages-rendered.png)

The "updated" timestamp is the tell: the page refreshed itself the moment new memory landed. Your architecture doc stays honest as the codebase evolves, with zero upkeep, and there is an Edit button for the rare time you want to steer it.

## Steering a page: a name and a question

You never author the body. A page is defined by a **name** and a **source query** — the question Hindsight re-asks after every consolidation to rebuild the page from memory.

![The page editor: a Name field and a "Source query" — the question that rebuilds this page from memory — plus optional tags](/img/blog/knowledge-pages-edit.png)

Change the question and the content re-synthesizes. This is also the manual lever behind the automatic behavior: the survey and `hindsight_capture_initiative` create pages with sensible questions for you, but you can add your own ("What is our deploy and rollback procedure?") and Hindsight keeps it answered from then on. You curate by asking better questions, not by writing and maintaining prose.

## The agent reads it, too

This is the difference between memory tools an agent *has* and memory it actually *uses*. Through the integration's MCP tools, Claude Code reads the knowledge base as part of its normal loop:

- `hindsight_search_knowledge_pages` — hybrid search to pull the right page for the task at hand.
- `hindsight_read_knowledge_page` — read a specific page in full.
- `hindsight_reflect` — [deeper reasoning](https://hindsight.vectorize.io/blog/2026/07/24/recall-vs-reflect) over the repo's full memory when a single page is too shallow.

So a session does not start cold. Before the agent writes a line, it can pull the conventions page and the relevant decisions, grounded in what the repo already established. It is the closest thing to a new hire who read all the docs and never forgets them.

## See the whole memory: the repo at a glance

Pages are one view. The bank's Home gives you the other: a **memory constellation** that plots every memory and the links between them, colored by how they connect — semantic, temporal, entity, and causal — beside the pages and the documents that fed them.

![The bank overview: a memory constellation of the repo's memories and links, colored by connection type, beside the Knowledge pages and recent documents](/img/blog/knowledge-pages-overview.png)

This is the same memory the agent recalls and reflects over, made visible. When an answer looks thin, the constellation tells you whether the memory is missing or just unretrieved.

## Setup

One command wires memory, the survey, and the page tools into Claude Code:

```bash
npx @vectorize-io/hindsight-coding-agents install claude-code
```

The same package supports the rest of the fleet — `install codex`, `cursor-cli`, `copilot-cli`, `opencode`, and more — or `install all` for every detected agent. Memory can live in [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup), a server you run, or a local daemon; you choose once.

## Why it is worth it

**Your agent gets more useful the longer you work in a repo.** The learnings from session one become the starting context for session fifty, instead of resetting every morning. In our 61-task benchmark, giving Claude Code memory cut the corrections it needed by about **57%**, at lower cost and faster wall time.

**The documentation writes and repairs itself.** The pages a human would never keep current, the agent keeps current for free, because they are a rendering of the memory it is already accumulating.

**You can read and trust what your agent knows.** An agent that silently accumulates memory is hard to debug. A folder of self-healing pages, plus the constellation, turns that opaque store into something you can open, review, and correct.

## Quick start

1. Install: `npx @vectorize-io/hindsight-coding-agents install claude-code`, and pick where memory lives.
2. Open Claude Code in a repo. The first session surveys it and seeds your pages.
3. Keep working. The agent reads the pages before tasks, records new initiatives, and Hindsight keeps everything current.
4. Browse or steer the pages anytime in the Control Plane, the [API](https://hindsight.vectorize.io/developer/api/knowledge-pages), or via `hindsight fs mount`.

Reasoning is not the hard part anymore. Continuity is. Read the [0.9.0 launch notes](https://hindsight.vectorize.io/blog/2026/08/06/hindsight-0-9-0), or install the integration and let your coding agent write its own documentation.

---

**Learn more:**
- [Coding Agents integration](https://hindsight.vectorize.io/sdks/integrations/coding-agents) — one command, native memory for Claude Code and the rest
- [Knowledge Pages developer guide](https://hindsight.vectorize.io/developer/knowledge-pages) — the concept and the model underneath
- [Give any Agent Plugins client memory](https://hindsight.vectorize.io/blog/2026/08/12/agent-plugins-persistent-memory) — the explicit, portable counterpart to this automatic integration
