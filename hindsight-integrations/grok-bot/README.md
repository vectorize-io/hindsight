# Hindsight for Grok Bot

Long-term memory for [Grok Bot](https://x.ai) and Cursor, shared across your Bots and
your other AI tools.

Each Grok Bot keeps its own conversation and learned context, so one Bot cannot see what
another worked out, and a scheduled routine starts every run from zero. Hindsight gives
each Bot its own memory bank plus one shared bank for the whole team, and lets Bots read
the banks your other agents (Claude Code, ChatGPT, Hermes, OpenClaw) already write to.

## Install

This plugin is not in the Grok Bot or Cursor marketplace yet. The listing is still in
review. Until it is approved, set it up by hand.

### Grok Bot

Paste [`connect-prompt.md`](connect-prompt.md) to any Bot. It tells the Bot to add
Hindsight as a custom connector, walks it through the OAuth sign-in, and then carries
the same rules the skills carry: which bank to write to, when to recall, how to hand
work to another Bot, and which tools never to call.

Name the Bot first. Its name becomes its bank id, so a Bot still called "Grok Bot" is
asked for a name before setup continues.

### Cursor

Copy this directory into Cursor's local plugin folder, then reload:

```bash
git clone https://github.com/vectorize-io/hindsight
mkdir -p ~/.cursor/plugins/local
cp -r hindsight/hindsight-integrations/grok-bot ~/.cursor/plugins/local/hindsight
```

Restart Cursor, or run **Developer: Reload Window**. Cursor then loads the plugin's MCP
server and its skills from the copied folder.

### After the marketplace listing lands

Installing from the marketplace will replace both paths above: search for **Hindsight**
under **Connect apps** in Grok Bot, or find it in the Cursor marketplace. This README
will be updated when that happens.

## Configuration

There is nothing to configure in the plugin. It connects to Hindsight Cloud's MCP server
at `https://api.hindsight.vectorize.io/mcp`, and Grok Bot runs the OAuth flow against it
(discovery, dynamic client registration and PKCE). The connection reaches every bank in
the organization you authorize.

Memory is organized into banks:

| Bank | Holds | Written by |
|---|---|---|
| `grok-bot::<bot-name>` | What one Bot learns in its own work. A Bot named "Sales Researcher" uses `grok-bot::sales-researcher`. | That Bot |
| `cursor::<project-name>` | What a Cursor agent learns in one project, named after the open workspace folder. The plugin installs into Cursor whenever you add it in Grok Bot. | Cursor agents in that project |
| `grok-bot::shared` | What every Bot should know, handoffs between Bots, and the "About the user" profile. | Any Bot |
| Your other banks | Memory from your other AI tools. | Those tools only; Bots read them |

## How it works

In Grok Bot, a plugin's hooks never run: the Bot installs a plugin's MCP server and
skills, and nothing else. So memory here is driven by skills, and each skill's
description says when the Bot should use it.

| Skill | Use |
|---|---|
| `memory-setup` | Connect, verify with `list_banks`, create the banks and the profile |
| `memory-context` | Recall relevant memory before starting any task |
| `memory-retain` | Save outcomes, decisions and preferences when a task finishes |
| `memory-reflect` | Answer questions about history and past decisions with `reflect` |
| `routine-memory` | Writes recall and retain steps into each routine it creates; reads the last run before a routine starts and records this run when it ends |
| `bot-handoff` | Keep a durable record of work passed between Bots, which Bots that were never messaged can also pick up |

The skills never call `delete_bank`, `clear_memories`, `invalidate_memory`,
`delete_document` or `delete_mental_model`. If something should be deleted, the Bot tells
you and you do it from the Hindsight dashboard.

## Self-hosted Hindsight

Grok Bot needs a public HTTPS MCP endpoint with OAuth 2.1 discovery and dynamic client
registration. Put the [`cloudflare-oauth-proxy`](../cloudflare-oauth-proxy/) in front of
your instance, check it with `hindsight-muse-preflight` from the
[`meta-muse`](../meta-muse/) integration (Grok Bot connects the same way), then change
the `url` in `mcp.json`.

## Development

```bash
python -m pytest tests -v
```

The tests apply the Cursor Marketplace submission checklist to the manifests, and pin
every tool and parameter the skills use to Hindsight's MCP tool registry.
