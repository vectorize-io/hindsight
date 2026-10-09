---
sidebar_position: 42
title: "Grok Bot Memory with Hindsight | Integration"
description: "Long-term memory for Grok Bot and Cursor. Each Bot keeps its own Hindsight memory bank plus a shared one, carries state between routine runs, and reads the banks your other AI tools write to."
---

# Grok Bot

Long-term memory for [Grok Bot](https://x.ai) and Cursor, shared across your Bots and your other AI tools.

Each Grok Bot keeps its own conversation and learned context, so one Bot cannot see what another worked out, and a scheduled routine starts every run from zero. [Hindsight](https://vectorize.io/hindsight) gives each Bot its own memory bank plus one shared bank for the whole team, and lets Bots read the banks your other agents (Claude Code, ChatGPT, Hermes, OpenClaw) already write to.

:::tip Hindsight Cloud (recommended)
[Sign up free](https://ui.hindsight.vectorize.io/signup). The plugin connects with OAuth, so there is no API key to copy and nothing to configure.
:::

## Install

:::info Marketplace listing in review
Hindsight is not in the Grok Bot or Cursor marketplace yet. Set it up by hand until the listing is approved. This page will be updated when it lands.
:::

### Grok Bot

Paste [`connect-prompt.md`](https://github.com/vectorize-io/hindsight/blob/main/hindsight-integrations/grok-bot/connect-prompt.md) to any Bot. It tells the Bot to add Hindsight as a custom connector, walks it through the OAuth sign-in, and then carries the same rules the skills carry: which bank to write to, when to recall, how to hand work to another Bot, and which tools never to call.

Name the Bot first. Its name becomes its bank id, so a Bot still called "Grok Bot" is asked for a name before setup continues.

### Cursor

Copy the plugin into Cursor's local plugin folder, then reload:

```bash
git clone https://github.com/vectorize-io/hindsight
mkdir -p ~/.cursor/plugins/local
cp -r hindsight/hindsight-integrations/grok-bot ~/.cursor/plugins/local/hindsight
```

Restart Cursor, or run **Developer: Reload Window**.

## Memory Banks

| Bank | Holds | Written by |
|---|---|---|
| `grok-bot::<bot-name>` | What one Bot learns in its own work. A Bot named "Sales Researcher" uses `grok-bot::sales-researcher`. | That Bot |
| `cursor::<project-name>` | What a Cursor agent learns in one project, named after the open workspace folder. The plugin installs into Cursor whenever you add it in Grok Bot. | Cursor agents in that project |
| `grok-bot::shared` | What every Bot should know, handoffs between Bots, and the "About the user" profile. | Any Bot |
| Your other banks | Memory from your other AI tools. | Those tools only; Bots read them |

The plugin connects to the root [MCP server](/developer/mcp-server), `https://api.hindsight.vectorize.io/mcp`, which reaches every bank in the organization you authorize.

## How It Works

In Grok Bot, a plugin's hooks never run: the Bot installs a plugin's MCP server and skills, and nothing else. So memory here is driven by skills, and each skill's description tells the Bot when to use it.

| Skill | Use |
|---|---|
| `memory-setup` | Connect, verify with `list_banks`, create the banks and the profile |
| `memory-context` | Recall relevant memory before starting any task |
| `memory-retain` | Save outcomes, decisions and preferences when a task finishes |
| `memory-reflect` | Answer questions about history and past decisions with `reflect` |
| `routine-memory` | Writes recall and retain steps into each routine it creates; reads the last run before a routine starts and records this run when it ends |
| `bot-handoff` | Keep a durable record of work passed between Bots, which Bots that were never messaged can also pick up |

The skills never call `delete_bank`, `clear_memories`, `invalidate_memory`, `delete_document` or `delete_mental_model`. If something should be deleted, the Bot tells you and you do it from the Hindsight dashboard.

## Self-Hosted Hindsight

Grok Bot needs a public HTTPS MCP endpoint with OAuth 2.1 discovery and dynamic client registration. Put the [`cloudflare-oauth-proxy`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/cloudflare-oauth-proxy) in front of your instance, check it with the [Meta Muse integration's](./meta-muse) preflight tool (Grok Bot connects the same way), then change the `url` in the plugin's `mcp.json`.

## Verifying Setup

Ask a Bot:

- "Set up Hindsight memory." → `memory-setup` lists your banks and creates the Bot's own
- "What do you remember about our vendor shortlist?" → `memory-context` recalls before answering
- "How do we usually handle renewals?" → `memory-reflect`

Then check the Hindsight dashboard: the Bot's bank and `grok-bot::shared` should appear, with memories tagged `source:grok-bot`.
