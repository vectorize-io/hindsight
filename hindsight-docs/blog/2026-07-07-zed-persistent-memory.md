---
title: "Give Zed's AI Assistant a Persistent Memory"
authors: [benfrank241]
slug: "2026/07/07/zed-persistent-memory"
date: 2026-07-07T12:00
tags: [hindsight, zed, integration, memory, persistent-memory, mcp, tutorial]
description: "The Zed editor's AI assistant is sharp inside a task and a stranger between them. hindsight-zed gives it long-term memory that recalls and retains across sessions."
image: /img/blog/zed-persistent-memory.png
hide_table_of_contents: true
---

![Persistent memory for the Zed editor with Hindsight](/img/blog/zed-persistent-memory.png)

[Zed](https://zed.dev) is a fast, Rust-built editor with a genuinely good AI assistant in its Agent Panel. But like most coding agents, it is brilliant inside a single task and a stranger between them. Close the panel, start a new conversation tomorrow, and it has forgotten that this repo uses pnpm, that you settled on a repository pattern last week, and that you like your tests colocated.

`hindsight-zed` fixes that. One command wires Zed's Agent Panel to a shared [Hindsight](https://github.com/vectorize-io/hindsight) memory so the agent can recall relevant decisions before it answers and retain durable facts as it works, across every session.

<!-- truncate -->

## TL;DR

- `hindsight-zed init` registers the Hindsight **MCP server** in Zed and adds a rule telling the agent to use it.
- The agent gets three tools: `recall`, `retain`, and `reflect`.
- Recall runs at **query time** against your actual message, so it pulls context relevant to what you just asked, with no lag.
- Works with [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server. It is [agent memory](https://vectorize.io/what-is-agent-memory) that outlives the conversation.

## The problem: a great assistant with no yesterday

A coding assistant that forgets everything between sessions makes you the memory. You re-explain the stack, restate the conventions, and re-litigate decisions you already made, every time you open a fresh conversation. The model is capable; it just has no continuity.

Persistent memory closes that gap. The point of memory is that the things you have already told your agent, and the things it has already figured out, are still there next time. That is what turns a per-session tool into an assistant that actually knows your project.

## How Zed's memory works

Zed does not expose a pre-prompt hook, so there is nowhere to automatically inject context before a turn. What Zed does give you is two building blocks, and the integration uses both.

**MCP context servers.** Zed runs [Model Context Protocol](https://modelcontextprotocol.io) servers listed under `context_servers` in its `settings.json` and exposes their tools in the Agent Panel. `hindsight-zed` registers the Hindsight MCP server there, which hands the agent `recall`, `retain`, and `reflect` tools.

**A global instructions file.** Zed includes `~/.config/zed/AGENTS.md` in every agent conversation. The integration writes a short rule into that file, inside a fenced `<!-- HINDSIGHT:BEGIN -->` to `<!-- HINDSIGHT:END -->` block so it never touches your own rules. The rule tells the agent to call `recall` at the start of each task to load relevant decisions, preferences, and project context, and to call `retain` whenever it learns a durable fact worth keeping across sessions.

The result is recall that happens at query time, against the message you actually sent. Because it runs when you ask rather than on a fixed schedule, it can pull the memories relevant to this specific request. From your seat it is automatic: you type, the agent quietly checks its memory, then answers.

One transport note. Zed does not yet ship native HTTP MCP transport, so the server connects through the [`mcp-remote`](https://www.npmjs.com/package/mcp-remote) stdio bridge, run via `npx`. Because that bridge runs on Node.js, and the setup CLI is a Node tool too, Node.js is the only requirement. No Python needed.

## Install and quick start

```bash
npx hindsight-zed init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-memory
```

`hindsight-zed` is a zero-dependency Node CLI, so `npx` runs it with no global install. Prefer a persistent command? Run `npm install -g hindsight-zed` first.

`init` adds the `hindsight` MCP server to `~/.config/zed/settings.json` and the recall/retain rule to `~/.config/zed/AGENTS.md`. Restart Zed, open the Agent Panel, and the `hindsight` server should show a green dot.

That is the whole setup. Under the hood, the settings entry looks like this:

```jsonc
{
  "context_servers": {
    "hindsight": {
      "source": "custom",
      "command": "npx",
      "args": [
        "-y", "mcp-remote",
        "https://api.hindsight.vectorize.io/mcp/my-memory/",
        "--header", "Authorization: Bearer YOUR_HINDSIGHT_API_KEY"
      ]
    }
  }
}
```

If your `settings.json` uses comments (JSONC), `init` will not rewrite it. Instead it prints the exact `context_servers` entry for you to paste. You can see that snippet any time with `hindsight-zed init --print-only`.

## Commands

| Command | What it does |
| --- | --- |
| `hindsight-zed init` | Add the MCP server and the recall/retain rule |
| `hindsight-zed status` | Show whether the server and rule are configured |
| `hindsight-zed uninstall` | Remove the server and the rule |
| `hindsight-zed init --print-only` | Print the config to add manually |

Prefix any of these with `npx ` if you did not install globally.

## Cloud or self-hosted

For **Hindsight Cloud**, pass an API key from your dashboard. The API URL defaults to `https://api.hindsight.vectorize.io`, so there is nothing else to set.

For a **self-hosted** server, point at your own base URL with `--api-url http://localhost:8888`. An open local server needs no token.

Configuration resolves from flags, environment, or `~/.hindsight/zed.json` (written by `init`):

| Setting | Env var | Default |
| --- | --- | --- |
| API URL | `HINDSIGHT_API_URL` | `https://api.hindsight.vectorize.io` |
| API token | `HINDSIGHT_API_TOKEN` | none (required for Cloud) |
| Bank id | `HINDSIGHT_ZED_BANK_ID` | `zed` |

The bank is one isolated store. Point Zed and your other tools at the same bank id and they share one memory, which is the idea behind [one memory for every AI tool](/blog/2026/04/07/one-memory-for-every-ai-tool).

## Verify it works

Give the agent a durable fact in one conversation: "This repo uses pnpm, never npm." Start a **new** conversation and ask for something related, like "add the date-fns dependency." A memory-aware agent recalls the convention and reaches for pnpm without being reminded, because the fact was retained to Hindsight and recalled against your new request.

You can watch this happen from the other side too. Open your Hindsight bank and you will see the retained convention show up as a stored memory after the first conversation.

## Frequently asked questions

**Does the agent recall automatically?**
Recall is a tool the agent calls, and the global rule tells it to call `recall` at the start of every task. So in practice it runs on its own, and because it runs at query time it uses your actual message to find relevant memory.

**What do I need installed?**
Just Node.js (version 18.3 or newer). `hindsight-zed` is a zero-dependency Node CLI, and Zed's MCP bridge (`mcp-remote`) also runs on Node via `npx`. No Python required.

**Will it overwrite my Zed config?**
No. The rule lives inside a fenced `HINDSIGHT` block in `AGENTS.md`, and `init` leaves the rest untouched. If your `settings.json` has comments, `init` prints the snippet instead of rewriting the file.

**What does `reflect` do?**
Alongside `recall` and `retain`, the agent can call `reflect` to consolidate and reason over what it has stored, so memory improves as it accumulates rather than becoming a flat pile of notes.

## Further reading

- [What is agent memory?](https://vectorize.io/what-is-agent-memory): the concepts behind recall and retention.
- [Best AI agent memory systems](https://vectorize.io/articles/best-ai-agent-memory-systems): how the major memory frameworks compare.
- [Cursor persistent memory](/blog/2026/06/12/cursor-persistent-memory): the same idea for the other AI-first editor.
- [One memory for every AI tool](/blog/2026/04/07/one-memory-for-every-ai-tool): point Zed and your other agents at the same bank.
