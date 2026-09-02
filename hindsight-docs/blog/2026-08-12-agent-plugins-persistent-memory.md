---
title: "Give Any Agent Plugins Client Long-Term Memory"
authors: [benfrank241]
slug: "2026/08/12/agent-plugins-persistent-memory"
date: 2026-08-12T12:00
tags: [hindsight, agent-plugins, mcp, agent-memory, persistent-memory, vercel]
description: "Agent Plugins is the new vendor-neutral standard for packaging Skills and MCP servers. Hindsight ships one portable plugin that gives Codex, Cursor, Copilot, Kiro, and VS Code long-term memory."
image: /img/blog/agent-plugins-persistent-memory.png
hide_table_of_contents: true
---

![Hindsight as an Agent Plugin: one portable bundle brings recall, retain, and reflect to every compatible client](/img/blog/agent-plugins-persistent-memory.png)

On August 6, Vercel, OpenAI, Amazon, Cursor, and GitHub announced [Agent Plugins](https://agent-plugins.org): a vendor-neutral standard for packaging agent extensions so you build once and run across every compatible client. Until now, every tool wanted its own config format. A skill you wrote for one agent did not move to the next, and neither did the MCP servers you wired up.

Hindsight now ships as an Agent Plugin. One portable bundle gives any supported client — [Codex](https://openai.com/index/introducing-codex/), Cursor, GitHub Copilot, Kiro, and VS Code — the same long-term memory, with no per-tool integration to hand-roll.

<!-- truncate -->

## TL;DR

- **Agent Plugins** packages **Agent Skills + MCP servers** into a single distributable plugin. It launched with support in ChatGPT/Codex, Cursor, GitHub Copilot, Kiro, and VS Code.
- Hindsight ships **one** plugin for the standard instead of N per-IDE configs. It carries the same `recall` / `retain` / `reflect` memory.
- The bundle is three files: a manifest, an MCP server config, and a memory skill. It is **transport-only** — all memory logic stays server-side in Hindsight.
- Set two env vars (`HINDSIGHT_API_KEY`, optional `HINDSIGHT_BANK_ID`) and the agent calls memory tools automatically, guided by the bundled skill.
- Honest limitation: the standard covers Skills and MCP, not session hooks, so this is **explicit, tool-driven** memory. For fully automatic capture across coding agents, use the [Coding Agents integration](https://hindsight.vectorize.io/sdks/integrations/coding-agents) — it shares the same banks.

## What Agent Plugins actually is

For most of the last year, extending an agent meant learning each tool's dialect. MCP standardized how an agent talks to a server, but not how you *ship* an extension: the skill files, the server config, and the metadata still lived in tool-specific formats.

Agent Plugins closes that gap. It is a small, open spec — Vercel initiated the proposal and refined it with AWS, Cursor, GitHub, Microsoft, and OpenAI — that defines one folder layout for a plugin. A plugin bundles two things the ecosystem had already converged on:

- **Agent Skills**: reusable instructions that teach an agent *when* and *how* to do something.
- **MCP servers**: the connection to live tools and data.

The result is "build once, run anywhere" for agent extensions. At launch the standard is supported by **ChatGPT / Codex, Cursor, GitHub Copilot, Kiro, and VS Code**. That is exactly the surface where memory matters, and exactly where maintaining a separate integration per client stops being worth it.

## What Hindsight shipped

The Hindsight plugin follows the Agent Plugins `1.0.0` layout and nothing more:

```
agent-plugin/
├── plugin.json                       # manifest ($schema + name + metadata)
├── mcp.json                          # Hindsight MCP server (Streamable HTTP)
└── skills/
    └── hindsight-memory/
        └── SKILL.md                  # teaches the agent when to recall / retain / reflect
```

That is the whole thing. It is deliberately thin, because the memory engine already exists and is battle-tested — the plugin is just the standards-based front door to it.

- **`mcp.json`** connects the client to Hindsight's built-in [MCP server](https://hindsight.vectorize.io/developer/mcp-server) over Streamable HTTP. It is transport-only: retrieval, consolidation, and reasoning all happen server-side.
- **`skills/hindsight-memory/SKILL.md`** is loaded into the agent's context so it knows *when* to reach for memory, not merely that the tools exist. This is the part most integrations skip, and it is the difference between an agent that has memory tools and an agent that actually uses them at the right moment.

The entire plugin is MIT licensed and lives in the [Hindsight repo](https://github.com/vectorize-io/hindsight).

## The memory the agent gets

Through the MCP server, the agent gets Hindsight's full memory surface. The three tools it reaches for most:

| Tool | When | What it does |
|------|------|--------------|
| `recall` | Before answering, when past context could help | Semantic + keyword + graph + temporal retrieval over the bank |
| `retain` | After learning a durable, reusable fact | Stores the fact for future sessions |
| `reflect` | When a lookup is too shallow and you need synthesized reasoning | Disposition-aware reasoning over everything remembered |

Knowledge pages, mental models, documents, and tags are exposed through the same server. The bundled skill spells out the discipline: recall at the start of a task or when the user refers to earlier work, retain durable preferences and decisions rather than transient chatter, and reflect only when a single lookup is too shallow for the question.

## Configuration

The plugin reads two environment variables, interpolated into `mcp.json`:

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| API key | `HINDSIGHT_API_KEY` | — | Your `hsk_...` key, sent as `Authorization: Bearer`. Required for Hindsight Cloud. |
| Memory bank | `HINDSIGHT_BANK_ID` | `default` | The bank to read from and write to, sent as `X-Bank-Id`. Use one bank per user, project, or team. |

Bank scope is worth a beat. The `mcp.json` endpoint is per-bank, so the agent never passes a `bank_id` argument — isolation is enforced by the connection, not by the model remembering to scope its calls. That is the same [per-user isolation model](https://hindsight.vectorize.io/blog/2026/08/04/per-user-multi-tenant-agent-memory) our multi-tenant users rely on, and the same reason [one bank per project or team](https://hindsight.vectorize.io/blog/2026/07/16/bank-strategy-agent-memory) keeps context relevant instead of bleeding across concerns.

One portability note: most clients substitute `${VAR}`, but some — VS Code and Cursor — expect `${env:VAR}`. If your client does not interpolate at all, paste the literal key and bank id into `mcp.json`.

## Explicit tools vs. automatic capture

Here is the honest boundary. Agent Plugins `1.0.0` standardizes **Skills and MCP**, not session lifecycle hooks. So this plugin delivers **explicit, tool-driven** memory: the agent calls `recall` and `retain` when the skill tells it to, and that behavior is identical across every supported client.

What it does not do is fire on its own before every prompt or automatically capture a session as it ends. That fully automatic experience needs deeper, tool-specific wiring. For coding work, that is exactly what the [Coding Agents integration](https://hindsight.vectorize.io/sdks/integrations/coding-agents) provides: one package that installs native memory across opencode, Cline, Kilo, Cursor CLI, Codex CLI, GitHub Copilot CLI, Claude Code, Grok Build, and more. It builds a per-repo memory bank automatically from git history and past sessions, then injects the relevant context as the agent works — no tools for the model to remember to call.

The two approaches are not a fork. They read and write the **same Hindsight banks**, so context captured automatically by the Coding Agents integration is recalled by the portable plugin inside any Agent Plugins client, and vice versa. Pick portable-and-explicit or native-and-automatic per tool; the memory underneath is one store.

## Quick start

The fast path is [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup): sign up free, grab an API key, nothing to self-host.

1. Get your `hsk_...` key from [ui.hindsight.vectorize.io/connect](https://ui.hindsight.vectorize.io/connect).
2. Set the two environment variables the plugin reads:

   ```bash
   export HINDSIGHT_API_KEY="hsk_your_token"
   export HINDSIGHT_BANK_ID="my-project"   # optional; defaults to "default"
   ```

3. Install the plugin in your client through its plugin/MCP UI. Installation is client-specific by design — the spec leaves distribution to each tool.

Once it is installed, ask the agent something that depends on past context, or tell it a durable preference. It calls `recall` and `retain` automatically, guided by the skill.

**Self-hosting:** replace the host in `mcp.json` (`https://api.hindsight.vectorize.io`) with your own deployment's URL. A local server with the MCP endpoint open needs no API key.

## Why this matters

Standards are boring until they remove real work. A week ago, giving five different agents memory meant five integrations to build and maintain. Now it is one plugin that every compatible client loads the same way, and it stays in sync with the native integrations through shared banks.

Memory is the layer that turns a capable-but-amnesiac agent into one that gets more useful the longer you work with it. Agent Plugins makes that layer portable. If you are running Codex, Cursor, Copilot, Kiro, or VS Code, you can give it long-term memory today — [start here](https://ui.hindsight.vectorize.io/signup).

---

**Learn more:**
- [Agent Plugins standard](https://agent-plugins.org) — the spec and guides
- [Introducing Agent Plugins](https://vercel.com/blog/introducing-agent-plugins) — Vercel's announcement
- [Hindsight MCP Server reference](https://hindsight.vectorize.io/developer/mcp-server) — every tool the plugin exposes
