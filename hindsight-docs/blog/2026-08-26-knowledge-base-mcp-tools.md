---
title: "Your Agent Can Now Manage Its Own Knowledge Base Over MCP"
authors: [benfrank241]
slug: "2026/08/26/knowledge-base-mcp-tools"
date: 2026-08-26T12:00
tags: [hindsight, knowledge-pages, mcp, agent-memory, knowledge-base, self-updating-docs]
description: "Hindsight 0.9.2 turns the knowledge base into a native MCP surface: seven agent-facing tools to browse, search, create, update, and prune its own wiki, over the same connection it already uses to retain and recall."
image: /img/blog/knowledge-base-mcp-tools.png
hide_table_of_contents: true
---

![The Hindsight knowledge base as a native MCP surface: seven agent-facing tools to read, search, and maintain a self-healing wiki over the same connection an agent uses to retain and recall](/img/blog/knowledge-base-mcp-tools.png)

A [Hindsight](https://hindsight.vectorize.io) knowledge base is the readable, self-healing wiki sitting on top of your agent's memory: pages that distill what the agent knows into something a human can skim, and that rebuild themselves as the underlying memory changes.

Until now there was an odd asymmetry. An agent could `retain` and `recall` over MCP all day, but to browse or maintain that wiki it had to drop down to a separate HTTP path. The memory was native; the knowledge base was a second integration.

Hindsight 0.9.2 closes that gap. The knowledge base is now a **native MCP surface**: seven agent-facing operations registered as tools, on the same connection the agent already uses to remember.

<!-- truncate -->

## TL;DR

- **Seven knowledge base operations are now MCP tools**, so an agent can browse the tree, search, read a page, create folders and pages, update nodes, and prune them without leaving MCP.
- They run over the **same connection as `retain` and `recall`**, with the same bank scoping and tenant auth, so there is no second integration to wire up.
- They are **annotated for safety**: reads are flagged read-only (clients can auto-approve them), the one delete is flagged destructive, and every tool is bounded to the agent's own memory.
- Operators can **restrict which tools a bank exposes**, so you can hand one bank full curation rights and another read-only access.
- **Each page's refresh trigger is visible on the tree**, so an agent can tell a self-maintaining page from a hand-written one before it decides to edit.
- `export_knowledge_base` **stays off MCP** on purpose: it returns the whole bank as one markdown bundle, which has no business in an agent's context window.

## The seven tools

The knowledge base breaks into reads, writes, and a single destructive operation. That split is not just documentation; it is encoded in each tool's MCP annotations, which is what lets a client treat them differently.

| Tool | Type | What it does |
| --- | --- | --- |
| `get_knowledge_base_tree` | read | Returns the folder and page tree, with each page's refresh trigger visible on it |
| `search_knowledge_base` | read | Searches across pages for the relevant ones |
| `get_knowledge_page` | read | Reads a single page's full content |
| `create_knowledge_folder` | write | Adds a folder to organize the tree |
| `create_knowledge_page` | write | Writes a new page |
| `update_knowledge_node` | write | Edits an existing page or folder |
| `delete_knowledge_node` | destructive | Removes a page or folder |

Reads carry a `readOnlyHint`, so an MCP client can group them and auto-approve the safe ones without a prompt on every call. `delete_knowledge_node` carries a `destructiveHint`, so a client can single it out for confirmation. And because every one of these tools only ever touches the agent's own bank, they are all marked closed-world: there is no reaching outside the memory the agent already owns.

## One connection, and the loop closes

The point is not really the seven tools. It is that they ride the connection the agent already has.

Before, an agent's relationship with its knowledge base was read-mostly and awkward: memory lived on MCP, the wiki lived on HTTP, and keeping them in sync meant maintaining two integrations. Now a single MCP session can `retain` a new fact, `recall` against it, notice the relevant page is stale, and `update_knowledge_node` to fix it, all in one place. The agent that writes the memory is the agent that curates the human-readable view of it.

That is the difference between a wiki that is *about* your agent and a wiki your agent actually *keeps*. A support agent can file a new runbook page the first time it solves a novel issue. A coding agent can correct an architecture page the moment it learns the old one is wrong. The knowledge base stops being a periodic export and becomes a living surface the agent tends as it works.

Here is that loop inside a single session:

1. The agent `recall`s to answer a question and gets a grounded response.
2. Working the task, it learns something the wiki does not reflect yet.
3. It calls `search_knowledge_base` to find the page that should hold it.
4. `get_knowledge_page` pulls the current text, and `update_knowledge_node` writes the correction in place.
5. The next session's `recall`, and the next human reader, both start from the fixed page.

No hand-off, no second integration, no export-edit-reimport. The same session that used the memory improved the record of it.

## Every page shows what rebuilds it

Knowledge pages come in two flavors: ones that rebuild themselves on a trigger as the underlying memory changes, and ones written by hand. Telling them apart used to mean opening a page to find out. In 0.9.2, `get_knowledge_base_tree` surfaces each page's refresh trigger right on the tree, so an agent can see, before it touches anything, whether a page maintains itself or waits to be edited.

That signal matters the moment an agent has write access. A page that refreshes on its own is usually one to leave alone and let the trigger update; a hand-written page is one the agent can safely correct with `update_knowledge_node`. Putting the trigger on the tree turns "should I edit this" from a guess into something the agent reads off the structure, which is exactly what you want when the thing doing the editing is a model.

## Safe by default, and scoped by the operator

Handing an agent write and delete access to its own documentation sounds like exactly the kind of thing you would want a guardrail on, and there are two.

The first is per-tool. The annotations above mean a client is never guessing whether a call is safe: reads announce themselves as read-only, the destructive delete announces itself as destructive, and a host can auto-approve the former while gating the latter. You get the convenience of unattended reads without signing a blank check on deletes.

The second is per-bank. An operator can restrict the set of tools a given bank exposes over MCP, applied to both the tool listing and the actual invocation.

Point a bank at the three read tools and you have given an agent a knowledge base it can consult but not change. Point another at all seven and you have a curator. The scope is set where it belongs, on the deployment, not in a prompt you have to trust the model to honor.

That is also why `export_knowledge_base` is not in the MCP set at all. It exists, but only over HTTP and the CLI, because it hands back the entire bank as a single markdown bundle. That is a great way to back up a knowledge base and a terrible thing to drop into a context window, so it stays out of the agent's reach by design.

## Who it's for

If you already run an agent against Hindsight over MCP, this is the upgrade that lets it own its documentation instead of just reading it. Coding agents get the tightest loop: survey the repo, write the pages, and correct them in place as the codebase moves, which is the pattern behind [Claude Code building and reading its own knowledge base](https://hindsight.vectorize.io/blog/2026/08/13/knowledge-pages-coding-agents). Support and research agents get a place to file what they learn so the next session starts from it.

And if you were holding off because wiring the knowledge base meant a second integration, that reason is gone. It is the same connection, the same auth, the same bank.

## Setup

The tools register automatically for MCP clients on 0.9.2. Connect your agent to Hindsight's MCP server as you already do for `retain` and `recall`, and the seven knowledge base tools appear alongside them, scoped to the bank you point at. To hand out narrower access, set a bank's enabled tools to the subset you want that bank to expose. The [MCP server guide](https://hindsight.vectorize.io/docs/developer/mcp-server) has the connection details, and the [knowledge pages docs](https://hindsight.vectorize.io/docs/developer/knowledge-pages) cover the pages themselves.

## Frequently asked questions

### What is a Hindsight knowledge base?

It is a human-readable, self-healing wiki built on top of your agent's memory. Pages distill what the agent knows into something you can skim, organized in a folder tree, and they rebuild themselves when the memory behind them changes. It is the view a person reads to understand what the agent has learned, and now also a surface the agent can maintain itself.

### How do the knowledge base tools relate to `retain` and `recall`?

They sit one layer up. `retain` and `recall` are how an agent writes and reads the raw memory itself: the facts, observations, and relationships. The knowledge base is the curated, human-readable wiki distilled from that memory, and these seven tools are how an agent tends it.

In practice the two work together: the agent retains what it learns, recalls it to act, and uses the knowledge base tools to keep the readable summary of it accurate. Same connection, same bank, two levels of abstraction.

### How is this different from the Claude Code knowledge base post?

That post covered the Coding Agents integration, where Claude Code builds and reads a repo wiki through a dedicated path. This is the underlying capability in the core server: the knowledge base is now a native MCP surface for *any* MCP client, with seven tools on the same connection used for `retain` and `recall`. The coding-agent experience is one thing you can build on top of it.

### Can I give an agent read-only access to the knowledge base?

Yes. Restrict the bank's enabled tools to the three reads (`get_knowledge_base_tree`, `search_knowledge_base`, `get_knowledge_page`). The agent can browse and search its wiki but cannot create, edit, or delete anything. The restriction applies to both the tool listing and any attempt to invoke a tool that is not exposed.

### Can an agent build a knowledge base from nothing?

Yes. With `create_knowledge_folder` and `create_knowledge_page` an agent can lay out a fresh tree and populate it, then keep it current with `update_knowledge_node` as it learns more. That is the coding-agent pattern: survey the repo once, write the initial pages, and correct them in place from then on, all over the same MCP connection.

### Why can't an agent export the whole knowledge base over MCP?

Because `export_knowledge_base` returns the entire bank as one markdown bundle, which is useful for a backup and actively harmful inside a context window. It stays available over HTTP and the CLI, where a bulk export belongs, and out of the agent's MCP tool set on purpose.

### Do the knowledge base tools respect multi-tenant boundaries?

Yes. They share the same bank scoping and tenant auth as the existing MCP tools, and each one is bounded to the agent's own memory. An agent can only see and change the knowledge base of the bank it is pointed at.

### Are the knowledge base tool calls audited?

Yes. Auditable MCP tool runs are wrapped with audit logging, so a create, update, or delete is recorded rather than silent. On 0.9.2, worker runs are also traced and join the caller's trace, so an async operation is visible end to end instead of vanishing once the call returns. You can see what an agent changed, not only that something changed.

---

**Learn more:**
- [Claude Code Builds and Reads Its Own Knowledge Base](https://hindsight.vectorize.io/blog/2026/08/13/knowledge-pages-coding-agents) — the coding-agent pattern this capability powers
- [MCP server guide](https://hindsight.vectorize.io/docs/developer/mcp-server) — connect an agent and scope its tools
- [What's new in Hindsight 0.9.2](https://hindsight.vectorize.io/blog/2026/08/24/version-0-9-2) — the release this shipped in
