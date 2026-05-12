---
sidebar_position: 5
title: "Right Agent Persistent Memory with Hindsight | Integration Guide"
description: "Add persistent memory to Right Agent — closed-box AI agents that run Claude Code inside OpenShell sandboxes. One Telegram bot per agent, with a per-chat Claude Code session over a shared, chat-tagged Hindsight bank. Hindsight is the native memory provider, selected during initialization."
---

# Right Agent

Persistent memory for [Right Agent](https://github.com/onsails/right-agent) using [Hindsight](https://hindsight.vectorize.io).

Right Agent runs [Claude Code](https://docs.anthropic.com/en/docs/claude-code) inside [OpenShell](https://github.com/NVIDIA/OpenShell) sandboxes. Each agent is its own Telegram bot — DMs, groups, and forum topics each get their own Claude Code session, all sharing one chat-tagged Hindsight bank.

Hindsight is the native, recommended memory provider — selected during `right init`. No plugin to install, no sandbox network policy to edit.

## Quick Start

Install Right Agent (see the [install guide](https://github.com/onsails/right-agent/blob/master/docs/INSTALL.md)), then run the init wizard:

```bash
right init
right up
```

`right init` walks you through picking a memory provider. Choose `hindsight` when prompted:

```
? memory provider:
> hindsight — hindsight cloud api (recommended)
  file — local MEMORY.md (no cloud dependency)
? hindsight api key: <paste your key, or press Enter to use HINDSIGHT_API_KEY at runtime>
? hindsight bank id (default: my-agent): <press Enter to accept>
```

Get an API key at [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup).

The agent auto-retains every turn and auto-recalls relevant context on every new message.

## Features

- **Auto-recall** — on every new user message, queries Hindsight for memories tagged with the current Telegram chat and injects them as system context.
- **Auto-retain** — after every turn, queues the conversation for retention. Asynchronous and resilient; never blocks the agent.
- **Explicit tools** — `memory_retain`, `memory_recall`, and `memory_reflect` MCP tools for when automatic isn't enough.
- **Per-chat tagging** — DMs, groups, and forum topics share one bank but each conversation only sees its own memories plus untagged globals.
- **Resilient writes** — local SQLite retain queue plus circuit breaker; pending writes replay automatically when Hindsight is reachable again.
- **No plugin, no policy edits** — Hindsight is built into Right Agent's MCP aggregator on the host. The sandbox needs no Hindsight-specific egress rule.
- **Bundled `rightmemory` skill** — tells the agent when to use each tool and what belongs in memory vs. in its identity files (`IDENTITY.md`, `USER.md`, `TOOLS.md`).

## How It Works

### Memory flow

- **On every new user message**, Right Agent recalls memories tagged with the current Telegram chat and injects them into the agent's system prompt under a `## Memory` section. Recalled content is wrapped as untrusted external context to defend against prompt injection from memory writes.
- **After every turn**, the conversation is queued for retention. If Hindsight is unreachable, the entry sits in a local SQLite queue and retries with backoff — the agent never blocks on memory writes.
- **The agent can also call retain, recall, and reflect directly** via the memory MCP tools when automatic behavior isn't enough.

### Per-chat scoping

One Hindsight bank per agent (set at `right init`, defaults to the agent name). Within that bank, every memory is tagged with the originating Telegram chat (`chat:<chat_id>`), and recall queries with `tags_match: "any"` — so a 1:1 DM, a group, and a forum topic each see their own memories plus any untagged globals. Group conversations don't bleed into 1:1s, but the agent keeps a shared baseline of facts that apply everywhere.

### Where the Hindsight client runs

Right Agent's MCP aggregator runs on the host — outside the sandbox — and is the only process that talks to the Hindsight API. Agents inside the sandbox call `memory_retain`, `memory_recall`, and `memory_reflect` over the aggregator's per-agent MCP endpoint, and the aggregator makes the Hindsight calls from the host. The sandbox network policy needs no Hindsight-specific egress rule, and Hindsight credentials never enter the sandbox.

## Memory Tools

The aggregator exposes three memory tools to every Hindsight-mode agent:

| Tool | Purpose |
|---|---|
| `mcp__right__memory_retain(content, context)` | Save a fact permanently with a short label (`"user preference"`, `"api format"`, etc.). |
| `mcp__right__memory_recall(query)` | Ranked semantic + keyword + graph search across the agent's memory. |
| `mcp__right__memory_reflect(query)` | Deep analysis across memories — synthesize patterns, compare past decisions. |

The bundled `rightmemory` skill tells the agent when to reach for each one and what belongs in memory rather than in the agent's identity files.

## Configuration

`right init` writes the Hindsight configuration into `~/.right/agents/<name>/agent.yaml`:

```yaml
memory:
  provider: hindsight
  api_key: your-api-key   # or omit to use HINDSIGHT_API_KEY env var
  bank_id: my-agent                    # defaults to the agent name
  recall_budget: 8000                  # optional, token budget for auto-recall
```

To change any of these later, edit the file and run `right restart <agent>`. To switch an existing agent between providers, run `right agent config <agent>` and re-run the wizard.

### Using an environment variable

If you'd rather keep the API key out of `agent.yaml`, leave `api_key` empty and set `HINDSIGHT_API_KEY` in the environment where `right up` runs:

```bash
export HINDSIGHT_API_KEY=your-api-key
right up
```

The aggregator reads `HINDSIGHT_API_KEY` at startup and uses it as the fallback for any agent whose `agent.yaml` doesn't set `memory.api_key`.

## Troubleshooting

**`Memory: degraded` shown in `/doctor` or in the Telegram chat.** The resilient client's circuit breaker tripped after repeated upstream errors. Pending retains are queued locally and replay when the circuit closes. Check Hindsight Cloud status and verify the API key is still valid.

**Memories not appearing after restart.** Verify `memory.bank_id` in `~/.right/agents/<name>/agent.yaml` matches the bank you previously wrote to. The default is the agent name; if you edited it manually, restarts pick up the new value and won't see memories under the old bank.

**`HINDSIGHT_API_KEY` ignored.** The `memory.api_key` field in `agent.yaml` takes precedence over the environment variable. Leave `api_key` empty (or remove it) to fall back to `HINDSIGHT_API_KEY`.

**Quota exhausted.** Right Agent surfaces a chat-level notice when Hindsight returns a quota error. Writes are dropped while quota is exhausted; reads keep working. Top up at [Hindsight Cloud](https://ui.hindsight.vectorize.io) and queued retains catch up on the next successful write.

**Switching from `file` to `hindsight` (or vice versa).** Run `right agent config <agent>` and re-run the wizard, then `right restart <agent>`. An existing local `MEMORY.md` is preserved but ignored while in Hindsight mode; switching back surfaces it again.

## Links

- [Right Agent on GitHub](https://github.com/onsails/right-agent)
- [Install guide](https://github.com/onsails/right-agent/blob/master/docs/INSTALL.md)
- [Hindsight Cloud signup](https://ui.hindsight.vectorize.io/signup)
