---
title: "Devin Desktop Persistent Memory (Formerly Windsurf)"
authors: [benfrank241]
slug: "2026/07/02/devin-desktop-persistent-memory"
date: 2026-07-02T13:00
tags: [hindsight, devin-desktop, devin, windsurf, codeium, memory, persistent-memory, mcp, tutorial]
description: "Add persistent memory to Devin Desktop (formerly Windsurf): a remote MCP server plus one always-on rule that recalls at task start and retains as you work."
image: /img/blog/devin-desktop-persistent-memory.png
hide_table_of_contents: true
---

![Devin Desktop Persistent Memory with Hindsight](/img/blog/devin-desktop-persistent-memory.png)

[Devin Desktop](https://devin.ai) is the editor Cognition rebranded from Windsurf (formerly Codeium) in June 2026. The name changed; the gap didn't. Devin reads your codebase and holds a plan within a session, but it carries nothing across sessions. Close the editor, reopen it tomorrow, and the agent is a fresh model again, with no memory of the decision you talked through last week or the convention you set on Tuesday.

The `hindsight-devin-desktop` integration adds persistent long-term memory to Devin. It's worth understanding *how* it gets there, because Devin Desktop doesn't expose lifecycle hooks to third parties. There's no place to bolt a `sessionStart` recall or a `stop` retain. Instead the integration uses two things the editor *does* support: **remote [Model Context Protocol](https://modelcontextprotocol.io) (MCP) servers** and **always-on workspace rules**.

<!-- truncate -->

## TL;DR

- Devin Desktop (formerly Windsurf) has no third-party lifecycle hooks, so memory is wired through MCP plus a rule, not hook scripts.
- `hindsight-devin-desktop init` connects the Hindsight **remote MCP server** (Devin gets `recall` / `retain` / `reflect` tools) and writes one **always-on rule** to `.devin/rules/hindsight.md`.
- The rule tells Devin to `recall` at the start of each task and `retain` durable facts as it works.
- No local daemon, no plugin scripts, no per-turn hooks. The MCP endpoint connects straight to [Hindsight Cloud](https://hindsight.vectorize.io) or your self-hosted server.
- This is **model-driven memory**: the rule rides in every request, but the actual recall/retain calls are Devin's decision. That's the main tradeoff versus deterministic hook-based integrations.

## Why Devin Desktop Needs Persistent Memory

A new Devin session starts with whatever it can see: your open files, the workspace, and any rules you've written in `.devin/rules/`. What it can't see is the past. The bug you traced through three files yesterday, the library you chose and why, the naming convention you've been holding the line on. None of that survives the session boundary unless you wrote it down somewhere Devin reads.

You can pin context by hand with rules files, and for stable facts that works. It doesn't help with the things you didn't know to record in advance. Persistent memory closes that gap: durable facts get retained as you work, and the relevant ones come back on their own next time.

That matters more for an editor you live in all day. A coding agent that reintroduces itself every morning isn't really an assistant. Memory is what turns a fresh-every-session model into one that builds on yesterday.

## How Devin Desktop Persistent Memory Works

Devin Desktop gives third parties two integration points, and `hindsight-devin-desktop` uses both.

**Remote MCP server.** Devin Desktop reads MCP servers from a single global config and supports *remote* servers via `serverUrl` with custom headers, so the integration points Devin straight at the Hindsight MCP endpoint with no local process to manage:

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

That gives Devin three tools: `recall` (search memory), `retain` (store a durable fact), and `reflect` (a synthesized, memory-grounded answer). The memory bank is encoded in the endpoint path, so one config line scopes the whole connection to a bank.

**Always-on rule.** Devin Desktop applies any rule file under `.devin/rules/` whose frontmatter says `trigger: always_on` to every request in the workspace. The integration writes one dedicated file, `.devin/rules/hindsight.md`, telling Devin how and when to use those tools:

```markdown
---
trigger: always_on
---

<!-- Managed by hindsight-devin-desktop -->
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

The file carries a sentinel comment (`<!-- Managed by hindsight-devin-desktop -->`) so the integration owns it end to end and can update or remove it idempotently without touching any other rule you've authored. Put together: the MCP server makes memory *available* as tools, and the always-on rule makes Devin *use* them.

## Install

```bash
pip install hindsight-devin-desktop
cd your-project
hindsight-devin-desktop init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-project
```

`init` merges the `mcpServers` entry into Devin Desktop's global MCP config and writes the rule into `./.devin/rules/hindsight.md`. Reload Devin Desktop (or refresh MCP servers) and the `hindsight` tools are live.

Three commands cover the lifecycle: `hindsight-devin-desktop init` adds the MCP server and the recall/retain rule, `status` shows whether both are configured, and `uninstall` removes them. If your MCP config isn't plain JSON (comments, or some other tool owns it), `init` won't clobber it. It prints the snippet to paste instead, which you can also get anytime with `hindsight-devin-desktop init --print-only`.

## Cloud or Self-Hosted

By default the integration points at Hindsight Cloud (`https://api.hindsight.vectorize.io`), which needs an API key from your dashboard. To run against your own server, pass `--api-url`. If it's an open local server, you can skip the token entirely:

```bash
hindsight-devin-desktop init --api-url http://localhost:8888 --bank-id my-project
```

Settings can also come from the environment: `HINDSIGHT_API_URL` (the API endpoint, defaulting to Cloud), `HINDSIGHT_API_TOKEN` (the bearer token, required for Cloud), and `HINDSIGHT_DEVIN_DESKTOP_BANK_ID` (the bank to scope memory to, defaulting to `devin-desktop`). Point two projects at the same bank to share memory, or give each its own bank for isolation.

## A Rebrand Detail Worth Knowing

Because Devin Desktop is a rebrand of Windsurf, a couple of on-disk paths still carry the old name, and the integration handles that so you don't have to. The global MCP config still lives under `~/.codeium/windsurf/` (that's Devin Desktop's data directory, unchanged by the rename), while the workspace rule now lives under `.devin/rules/`, with `.windsurf/rules/` kept as a legacy fallback. If you used the integration back when it was the Windsurf package, your existing rule keeps working and the new path takes precedence going forward.

## The Tradeoff: Model-Driven, Not Hook-Driven

This is worth being direct about, because it's the real difference between this integration and the hook-based ones for Claude Code or the Cursor CLI.

Hook-based integrations are **deterministic**. A `sessionStart` hook recalls before the agent ever sees the prompt; a `stop` hook retains after every task, whether or not the model thought to. The recall and retain happen because the harness fires an event, not because the agent decided to.

Devin Desktop doesn't offer that surface to third parties, so `hindsight-devin-desktop` is **model-driven**. The always-on rule is injected into every request, so the instruction to use memory is always present, but the actual `recall` and `retain` calls are Devin's decision. In practice modern models follow a short, concrete always-on rule reliably. But it's an instruction, not a guarantee: Devin can skip a `retain` on a task it didn't judge memorable, or answer from context without calling `recall` first. If you want memory pulled for a specific task, you can just ask ("check memory for how we handled auth"), and `reflect` is there to consolidate on demand. The honest framing: Devin Desktop trades the guarantees of hooks for the simplicity of a remote MCP server and one rule file, with no local daemon and nothing to keep running.

## Frequently Asked Questions

**Is Devin Desktop the same as Windsurf?**
Yes. Cognition rebranded the Windsurf editor (formerly Codeium) to Devin Desktop in June 2026. The `hindsight-devin-desktop` package is the maintained integration; it writes its rule to `.devin/rules/` and still reads the MCP config under `~/.codeium/windsurf/`, which is unchanged by the rebrand.

**Does Devin Desktop have built-in memory across sessions?**
No. A new session starts fresh. Persistent memory comes from an integration like `hindsight-devin-desktop` that gives Devin recall and retain over a memory layer.

**Will memory recall slow Devin down?**
Recall is the agent's call, not a per-prompt hook, so there's no fixed overhead on every turn. When Devin does recall, a Hindsight Cloud query is typically well under a second.

**Does it work with self-hosted Hindsight?**
Yes. Pass `--api-url` (or set `HINDSIGHT_API_URL`) to point at your server. For an open local server with no auth, omit the token.

## Further reading

- [What is agent memory?](https://vectorize.io/what-is-agent-memory): the foundational concepts behind recall, retention, and memory banks.
- [Best AI agent memory systems](https://vectorize.io/articles/best-ai-agent-memory-systems): how the major agent memory frameworks compare.
- [Cursor persistent memory](/blog/2026/06/12/cursor-persistent-memory): the hook-based sibling integration for the Cursor editor and CLI.
- [One memory for every AI tool](/blog/2026/04/07/one-memory-for-every-ai-tool): point Devin and your other agents at the same bank.
