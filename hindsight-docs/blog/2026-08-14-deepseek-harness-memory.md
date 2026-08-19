---
title: "Give DeepSeek Harness a Memory of Your Codebase"
authors: [benfrank241]
slug: "2026/08/14/deepseek-harness-persistent-memory"
date: 2026-08-14T15:00
tags: [hindsight, deepseek, deepseek-harness, coding-agents, agent-memory, persistent-memory]
description: "DeepSeek just launched Harness (dsh), an open-source coding agent built on plugins. Hindsight plugs in natively to give it long-term memory of your codebase."
image: /img/blog/deepseek-harness-memory-cover.png
hide_table_of_contents: true
---

![DeepSeek Harness with Hindsight: a coding agent that retains your project's rules and recalls them in future sessions](/img/blog/deepseek-harness-memory-cover.png)

Yesterday DeepSeek open-sourced [Harness](https://github.com/deepseek-ai/dsh) (`dsh`), a coding agent built on the [Cordis](https://github.com/deepseek-ai/dsh) plugin framework whose guiding principle is "everything is a plugin." It is MIT licensed, written in TypeScript, and positioned squarely as an open alternative to Claude Code. It passed 27,000 stars within hours of launch.

It is also, like every coding agent, completely amnesiac between sessions. Close the terminal, come back tomorrow, and it has forgotten that this repo uses `pgm` not `npm`, that PR titles must follow Conventional Commits, and the tie-break rule you talked through last week. Because Harness is plugin-first, we could fix that the same way it does everything else: with a plugin.

<!-- truncate -->

## TL;DR

- **DeepSeek Harness now has long-term memory**, via the [Hindsight Coding Agents integration](https://hindsight.vectorize.io/sdks/integrations/coding-agents).
- One command: `npx @vectorize-io/hindsight-coding-agents install dsh`. It installs as a **native Cordis plugin**, so there is no MCP server to run.
- It is **fully automatic**: a repo's git history and conversations flow into a memory bank in the background, and Harness recalls the relevant context when it starts a task.
- It also builds and reads **Knowledge Pages**: a self-healing wiki of your architecture, conventions, and decisions that future sessions start from.
- Memory lives in a Hindsight bank you control, [Cloud](https://ui.hindsight.vectorize.io/signup) or self-hosted, and it is shared across your other tools.

## Why a coding agent needs memory most

An agent that works in discrete sessions is great for focus and terrible for continuity. Every session re-learns your stack from scratch, rediscovers the same gotchas, and re-asks what you already answered. Most of a real fix is derivable from the code, but the last mile often hinges on a project-specific decision that is not in the code at all: a rounding rule, a retry allowlist, a naming convention. Those decisions live in git history and past conversations. Memory is what puts them back in front of the agent at the moment it starts working.

## Setup: one command

Harness treats everything as a plugin, so the integration is one too. Install the package and wire up `dsh`:

```bash
npx @vectorize-io/hindsight-coding-agents install dsh
```

That registers a Cordis plugin row that every `dsh` profile composes, using native tools, no MCP needed. Point memory at [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup), a server you run, or a local daemon; you choose once. (Harness writes its session logs as Zstandard-framed JSONL, which needs Node 22.15+.)

There is no capture command to remember. From the next session on, memory is automatic.

## Teach it once

In a normal working session, tell Harness the rules of the repo. Here it records two project conventions, and Hindsight retains them as durable memory:

![DeepSeek Harness storing two project rules to the repo's Hindsight memory](/img/blog/deepseek-harness-retain.png)

Those retentions land in a Hindsight bank scoped to the repo. You can watch them arrive in the Control Plane, tagged by the harness that produced them:

![The Hindsight Control Plane showing the retained conversation and conventions, tagged harness-dsh](/img/blog/deepseek-harness-memory.png)

Nothing here required a manual export. Harness worked, and the integration captured what mattered in the background.

## It remembers in the next session

Start a fresh session later and ask about the project's conventions. Instead of guessing, Harness recalls the rules it was taught, from memory:

![A new DeepSeek Harness session recalling the package-management and PR-title conventions from Hindsight memory](/img/blog/deepseek-harness-recall.png)

The package-management rule and the Conventional Commits requirement come back verbatim, because they are stored as reconciled memory rather than left to a context window that resets every session. The learnings from session one become the starting context for session fifty.

## Beyond recall: a self-healing wiki

The integration does more than retain and recall individual facts. On a cold repo it runs a read-only survey and seeds **Knowledge Pages** for architecture, conventions, and in-flight initiatives, then keeps them current as you work. Harness reads those pages before it starts, and records new work as tracked pages, so the documentation writes and repairs itself. You can read more in [Knowledge Pages for coding agents](https://hindsight.vectorize.io/blog/2026/08/13/knowledge-pages-coding-agents).

## The nice thing about a shared bank

Because memory lives in a Hindsight bank and not inside Harness, it is portable. The same bank that Harness fills is recalled by Claude Code, Codex, Cursor, and the other [ten coding agents](https://hindsight.vectorize.io/sdks/integrations/coding-agents) the integration supports. Teach one, and the rest of your fleet knows it too.

DeepSeek shipped a plugin-first coding agent. We shipped the plugin that gives it a memory. Install it with one command, or read the [integration guide](https://hindsight.vectorize.io/sdks/integrations/coding-agents) first.

---

**Learn more:**
- [Coding Agents integration](https://hindsight.vectorize.io/sdks/integrations/coding-agents) — one command, native memory for DeepSeek Harness and ten more
- [Knowledge Pages for coding agents](https://hindsight.vectorize.io/blog/2026/08/13/knowledge-pages-coding-agents) — the self-healing wiki your agent builds and reads
- [Hindsight on GitHub](https://github.com/vectorize-io/hindsight) — the open-source memory engine underneath
