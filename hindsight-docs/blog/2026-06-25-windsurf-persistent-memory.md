---
title: "Windsurf Persistent Memory: A Rule Instead of Hooks"
authors: [benfrank241]
slug: "2026/06/25/windsurf-persistent-memory"
date: 2026-06-25T12:00
tags: [windsurf, codeium, cascade, memory, persistent-memory, hindsight, coding-agents, mcp, tutorial]
description: "Give Windsurf's Cascade agent persistent memory with hindsight-windsurf: a remote MCP server and one always-on rule. Recall at task start, retain as you work."
image: /img/blog/windsurf-persistent-memory.png
hide_table_of_contents: true
---

![Windsurf Persistent Memory with Hindsight](/img/blog/windsurf-persistent-memory.png)

[Windsurf](https://windsurf.com) (formerly Codeium) puts an agent, Cascade, at the center of the editor. It reads your codebase, runs commands, edits across files, and holds a coherent plan within a session. What it doesn't do is carry anything across sessions. Close the editor, reopen it tomorrow, and Cascade is a fresh model again: the architectural decision you talked through last week and the convention you established on Tuesday are gone.

The `hindsight-windsurf` integration gives Cascade persistent long-term memory: durable [agent memory](https://vectorize.io/what-is-agent-memory) that survives across sessions. But Windsurf is interesting because it gets there a different way than the hook-based integrations for Claude Code or Cursor. Windsurf doesn't expose lifecycle hooks to third parties, so there's no place to bolt a `sessionStart` recall or a `stop` retain. Instead, the integration leans on two things Windsurf *does* support: **remote [Model Context Protocol](https://modelcontextprotocol.io) (MCP) servers** and **always-on workspace rules**.

<!-- truncate -->

## TL;DR

- Windsurf has no third-party lifecycle hooks, so memory is wired through MCP plus a rule, not hook scripts.
- `hindsight-windsurf init` connects the Hindsight **remote MCP server** (Cascade gets `recall` / `retain` / `reflect` tools) and writes one **always-on rule** to `.windsurf/rules/hindsight.md`.
- The rule tells Cascade to `recall` at the start of each task and `retain` durable facts as it works.
- No local daemon, no plugin scripts, no per-turn hooks; the MCP endpoint connects directly to Hindsight Cloud (or your self-hosted server).
- This is **model-driven memory**: the always-on rule is in every Cascade request, but the actual recall/retain calls are the agent's decision. That's the main tradeoff versus deterministic hook-based integrations.

---

## Why Windsurf Needs Persistent Memory

A new Cascade session starts with whatever it can see: your open files, the workspace, and any rules you've written in `.windsurf/rules/`. What it can't see is the past. The bug you traced through three files yesterday, the reason you picked one library over another, the naming convention you've been holding the line on. None of that survives the session boundary unless you wrote it down somewhere Cascade reads.

You can pin context manually with rules files. That works for the things you remember to record, not for the things you didn't realize mattered until later. Persistent memory closes that gap: durable facts get retained as you work, and relevant ones come back automatically the next time they're useful, without you curating a rules file by hand.

## How Windsurf Persistent Memory Works

Windsurf gives third parties two integration points, and `hindsight-windsurf` uses both.

**Remote MCP server.** Windsurf reads MCP servers from a single global config at `~/.codeium/windsurf/mcp_config.json`, and it supports *remote* servers via `serverUrl` with custom headers. So the integration points Cascade straight at the Hindsight MCP endpoint, with no local process and no daemon to manage:

```json
{
  "mcpServers": {
    "hindsight": {
      "serverUrl": "https://api.hindsight.vectorize.io/mcp/my-project/",
      "headers": { "Authorization": "Bearer hsk_..." }
    }
  }
}
```

That gives Cascade three tools: `recall` (search memory), `retain` (store a durable fact), and `reflect` (consolidate). The memory bank is encoded in the endpoint path, so a single config line scopes the whole connection to one bank.

**Always-on rule.** Windsurf applies any rule file under `.windsurf/rules/` whose frontmatter says `trigger: always_on` to *every* Cascade request in the workspace. The integration writes one dedicated file, `.windsurf/rules/hindsight.md`, that tells Cascade how and when to use those tools:

```markdown
---
trigger: always_on
---

<!-- Managed by hindsight-windsurf -->
You have persistent long-term memory through the Hindsight MCP server
(`recall`, `retain`, and `reflect` tools).

- At the start of each task, call `recall` with the user's request to load
  relevant decisions, preferences, and project context before you act.
  Use what's relevant and ignore the rest.
- When you learn a durable fact, such as an architectural decision, a user
  preference, a convention, or anything worth remembering across sessions,
  call `retain` to store it.
- Do not mention these memory operations unless the user asks about them.
```

The file carries a sentinel comment (`<!-- Managed by hindsight-windsurf -->`) so the integration owns it end to end, which lets it update or remove the rule idempotently without touching any other rule you've authored.

Put together: the MCP server makes memory *available* as tools, and the always-on rule makes Cascade *use* them. Recall before it acts, retain as it learns.

## Install

```bash
pip install hindsight-windsurf
cd your-project
hindsight-windsurf init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-project
```

`init` merges the `mcpServers` entry into `~/.codeium/windsurf/mcp_config.json` and writes the rule into `./.windsurf/rules/hindsight.md`. Reload Windsurf (or refresh MCP servers in Cascade) and the `hindsight` tools are live.

If `mcp_config.json` isn't plain JSON (say you've got comments, or some other tooling owns it), `init` won't clobber it. It prints the snippet for you to paste instead. You can also get that snippet anytime with `hindsight-windsurf init --print-only`.

Three commands cover the lifecycle:

| Command | Description |
| --- | --- |
| `hindsight-windsurf init` | Add the MCP server plus recall/retain rule |
| `hindsight-windsurf status` | Show whether the server and rule are configured |
| `hindsight-windsurf uninstall` | Remove the server and rule |

## Cloud or Self-Hosted

By default the integration points at [Hindsight Cloud](https://hindsight.vectorize.io) (`https://api.hindsight.vectorize.io`), which needs an API key. To run against a self-hosted server instead, pass `--api-url`. If it's an open local server, you can skip the token entirely:

```bash
hindsight-windsurf init --api-url http://localhost:8888 --bank-id my-project
```

Connection settings can also come from the environment:

| Setting | Env var | Default |
| --- | --- | --- |
| API URL | `HINDSIGHT_API_URL` | `https://api.hindsight.vectorize.io` |
| API token | `HINDSIGHT_API_TOKEN` | _(none; required for Cloud)_ |
| Bank id | `HINDSIGHT_WINDSURF_BANK_ID` | `windsurf` |

## Banks: Shared or Per-Project

The `--bank-id` you pass at init time is what scopes the memory. Leave it at the default (`windsurf`) and every project shares one bank, which is convenient and useful if your work spans repos that share conventions. Set a distinct `--bank-id` per project (or just use the project name) and each codebase gets its own isolated memory, so Cascade in repo A never recalls decisions from repo B.

Because the bank is baked into the MCP endpoint path, switching banks is a re-`init` away; there's no per-request bank juggling to think about.

## The Tradeoff: Model-Driven, Not Hook-Driven

This is worth being direct about, because it's the real difference between the Windsurf integration and the hook-based ones for Claude Code, Cursor, or the Cursor CLI.

Hook-based integrations are **deterministic**. A `sessionStart` hook recalls before the agent ever sees the prompt; a `stop` hook retains after every task, whether or not the model thought to. The recall and retain happen because the harness fires an event, not because the agent decided to.

Windsurf doesn't offer that surface to third parties, so `hindsight-windsurf` is **model-driven**. The always-on rule is injected into every Cascade request, so the instruction to use memory is always present, but the actual `recall` and `retain` calls are Cascade's decision. In practice modern models follow an always-on rule reliably, especially a short, concrete one like this. But it's an instruction, not a guarantee: a model can skip a `retain` on a task it didn't judge memorable, or answer from context without calling `recall` first.

If you want the agent to always pull memory for a given task, you can just ask ("check memory for how we handled auth before") and the `recall` tool is right there. And `reflect` lets you consolidate on demand. The honest framing: Windsurf trades the guarantees of hooks for the simplicity of a remote MCP server and one rule file. No local daemon, no plugin scripts, nothing to keep running.

## Setup

The fast path:

1. Sign up at [hindsight.vectorize.io](https://ui.hindsight.vectorize.io/signup). The free tier is enough to start.
2. Grab an API key from the dashboard.
3. Install and initialize in your project:
   ```bash
   pip install hindsight-windsurf
   cd your-project
   hindsight-windsurf init --api-token hsk_your_key --bank-id my-project
   ```
4. Reload Windsurf (or refresh MCP servers in Cascade), then run `hindsight-windsurf status` to confirm the server and rule are both in place.

From there, Cascade recalls relevant context at the start of a task and retains durable facts as it works, across every session, in the editor you're already using.

## Frequently Asked Questions

**Does Windsurf have built-in memory across sessions?**
No. Cascade holds context within a session, but a new session starts from scratch. Persistent memory across sessions comes from an integration like `hindsight-windsurf`.

**How is this different from a normal `.windsurf/rules` file?**
A static rules file holds whatever you hand-write. Hindsight stores durable facts as Cascade works and recalls the relevant ones later, so the context grows on its own instead of being curated by hand. The always-on rule just tells Cascade to use the memory tools.

**Does it work with self-hosted Hindsight?**
Yes. Pass `--api-url` to point at your own server; an open local server needs no token. Hindsight self-hosts with a single Docker command under an MIT license.

**Will memory recall slow Cascade down?**
Recall is the agent's call, not a per-prompt hook, so there's no fixed overhead on every turn. When Cascade does recall, a Hindsight Cloud query is typically well under a second.

## Further reading

- [What is agent memory?](https://vectorize.io/what-is-agent-memory): the foundational concepts behind recall, retention, and memory banks.
- [Best AI agent memory systems](https://vectorize.io/articles/best-ai-agent-memory-systems): how the major agent memory frameworks compare.
- [Cursor persistent memory](/blog/2026/06/12/cursor-persistent-memory): the hook-based sibling integration for the Cursor editor and CLI.
- [Cline persistent memory](/blog/2026/06/09/cline-persistent-memory): persistent memory for another VS Code coding agent.
