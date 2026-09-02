---
sidebar_position: 2
title: "Agent Plugins Persistent Memory with Hindsight | Integration Guide"
description: "Give any Agent Plugins client — Codex, Cursor, GitHub Copilot, Kiro, VS Code — long-term memory with Hindsight. One portable, standards-based plugin bundles the Hindsight MCP server and a memory skill for recall, retain, and reflect."
---

# Agent Plugins

Portable long-term memory for any [Agent Plugins](https://agent-plugins.org) client, powered by [Hindsight](https://vectorize.io/hindsight).

[Agent Plugins](https://agent-plugins.org) is the vendor-neutral open standard (developed with Amazon, Cursor, Microsoft, OpenAI, and Vercel) for packaging **Agent Skills + MCP servers** into a single distributable plugin. Instead of a separate integration per tool, Hindsight ships **one** plugin that every compatible client can load — at launch: **ChatGPT / Codex, Cursor, GitHub Copilot, Kiro, and VS Code**.

## Quick Start

:::tip Recommended: Hindsight Cloud
[Sign up free](https://ui.hindsight.vectorize.io/signup) for a Hindsight Cloud API key — no self-hosting, no local daemon to manage.
:::

1. Get your `hsk_...` API key from [ui.hindsight.vectorize.io/connect](https://ui.hindsight.vectorize.io/connect).
2. Set the environment variables the plugin reads:

   ```bash
   export HINDSIGHT_API_KEY="hsk_your_token"
   export HINDSIGHT_BANK_ID="my-project"   # optional; defaults to "default"
   ```

3. Install the plugin in your client (through its plugin/MCP UI, or by pointing it at the plugin directory — installation is client-specific per the standard).

Once installed, ask the agent something that depends on past context, or tell it a durable preference — it calls `recall` and `retain` automatically, guided by the bundled skill.

## What's in the plugin

The plugin is a thin, transport-only wrapper — all memory logic stays server-side in Hindsight. It follows the Agent Plugins `1.0.0` layout:

```
agent-plugin/
├── plugin.json                       # manifest ($schema + name + metadata)
├── mcp.json                          # Hindsight MCP server (Streamable HTTP)
└── skills/
    └── hindsight-memory/
        └── SKILL.md                  # teaches the agent when to recall / retain / reflect
```

- **`mcp.json`** connects the client to Hindsight's built-in [MCP server](/developer/mcp-server) over Streamable HTTP.
- **`skills/hindsight-memory/SKILL.md`** is loaded into the agent's context so it knows *when* to reach for memory, not just that the tools exist.

## Memory tools

Via the MCP server, the agent gets Hindsight's full memory surface. The three it reaches for most:

| Tool | When | What it does |
|------|------|--------------|
| `recall` | Before answering, when past context could help | Semantic + keyword + graph + temporal retrieval over the bank |
| `retain` | After learning a durable, reusable fact | Stores the fact for future sessions |
| `reflect` | When a lookup is too shallow and you need synthesized reasoning | Disposition-aware reasoning over everything remembered |

Additional tools (knowledge pages, mental models, documents, tags) are exposed too — see the [MCP Server reference](/developer/mcp-server).

## Configuration

The plugin reads two environment variables, interpolated into `mcp.json`:

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| API key | `HINDSIGHT_API_KEY` | — | Your `hsk_...` key. Sent as `Authorization: Bearer`. Required for Hindsight Cloud. |
| Memory bank | `HINDSIGHT_BANK_ID` | `default` | Bank to read from and write to (sent as `X-Bank-Id`). Use one bank per user, project, or team for isolation. |

:::note Env-var syntax varies by client
Most clients substitute `${VAR}`; some (VS Code, Cursor) use `${env:VAR}`. If your client doesn't interpolate, paste the literal key and bank id into `mcp.json`.
:::

**Self-hosting:** replace the host in `mcp.json` (`https://api.hindsight.vectorize.io`) with your deployment's URL. A local server with the MCP endpoint open needs no API key.

## Explicit tools vs. automatic capture

Agent Plugins `1.0.0` standardizes **Skills + MCP**, not session lifecycle hooks. This plugin therefore delivers **explicit, tool-driven** memory that works identically across every supported client.

For the fully automatic experience — recall injected before every prompt and transcripts retained on session end — use the native, hook-based integration built for your specific tool, such as [Claude Code](/sdks/integrations/claude-code) or [Codex](/sdks/integrations/codex). Both share the same Hindsight banks, so memory captured by the hook-based integration is recalled through the Agent Plugin, and vice versa.

## Troubleshooting

**No memories recalled**: `recall` returns results only after something has been retained. Retain a fact first, or seed the bank via the [API](/developer/api/quickstart).

**401 Unauthorized**: Check `HINDSIGHT_API_KEY` is set and your client is interpolating it into the `Authorization` header (see the env-var syntax note above).

**Wrong or empty memory**: Confirm `HINDSIGHT_BANK_ID` points at the bank you expect. Different tools writing to different banks won't share memory.

## Learn more

- [Agent Plugins standard](https://agent-plugins.org)
- [Hindsight MCP Server reference](/developer/mcp-server)
- [Hindsight Cloud sign-up](https://ui.hindsight.vectorize.io/signup)
