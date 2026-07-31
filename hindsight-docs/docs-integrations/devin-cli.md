---
sidebar_position: 42
title: "Devin CLI Persistent Memory with Hindsight | Integration"
description: "Add long-term memory to Devin CLI with Hindsight. Lifecycle hooks recall relevant context before every prompt and retain each session automatically, plus agent_knowledge_* MCP tools and memory-backed subagents."
---

# Devin CLI

Automatic long-term memory for [Devin CLI](https://docs.devin.ai/cli), powered by [Hindsight](https://vectorize.io/hindsight). This is a port of the [Claude Code plugin](./claude-code.md) — same hooks, same config schema, same MCP tools — adapted to Devin CLI's hook payloads and plugin system.

Memory is automatic from your seat: relevant context is injected before each prompt, and each session is retained without you asking.

## How It Works

Devin CLI's extensibility surface is close enough to Claude Code's that most of the plugin ports over unchanged:

- **Lifecycle hooks** — identical event names (`SessionStart`, `UserPromptSubmit`, `Stop`, `SessionEnd`), the same JSON-on-stdin / JSON-on-stdout protocol, and the same `hookSpecificOutput.additionalContext` injection mechanism. Recall runs on `UserPromptSubmit`; retain runs on `Stop` and `SessionEnd`.
- **MCP servers** — configured under `mcpServers` in `~/.config/devin/mcp_config.json`, the same schema Claude Code uses. The Hindsight MCP server exposes `agent_knowledge_*` tools for knowledge pages, recall, and document ingest.
- **Skills** — `create-agent` builds specialized subagents with their own memory banks; `setup` performs the one-time hook/MCP registration.

## Quick Start

```bash
# 1. Install the plugin
devin plugins install vectorize-io/hindsight#hindsight-integrations/devin-cli

# 2. Register hooks + the MCP server (one-time — see "Setup" below)
#    Easiest: start a Devin CLI session and ask it to set up Hindsight memory,
#    which invokes this plugin's `setup` skill.

# 3. Configure your LLM provider for memory extraction
export OPENAI_API_KEY="sk-your-key"        # or ANTHROPIC_API_KEY, etc.

# 4. Start a new session — hooks and the MCP server load at session start
devin
```

## Setup

Devin CLI's plugin system installs a plugin's skills, rules, and subagents cleanly, and it *tries* to wire up a plugin-shipped `hooks.json` / `mcp_config.json` too. But hook and MCP commands currently have no portable way to reference their own bundled scripts: there's no `${DEVIN_PLUGIN_ROOT}`-style substitution (unlike Claude Code's `${CLAUDE_PLUGIN_ROOT}`), and a relative path in a plugin's own `hooks.json` resolves against the *project* working directory at hook-run time, not the plugin's install directory.

So the plugin ships `scripts/install.py`, which writes absolute-path hook and MCP entries directly into Devin CLI's documented config locations:

- `~/.config/devin/config.json` (`"hooks"` key)
- `~/.config/devin/mcp_config.json` (`"mcpServers"` key)

Run it once after installing. It is idempotent, and re-running it after a plugin upgrade replaces the previous version's entries rather than leaving dead ones behind. If plugin path templating lands in a later CLI release, this step goes away.

## Features

- **Auto-recall** — queries Hindsight on every user prompt and injects relevant memories as context (invisible in the chat transcript, visible to the agent)
- **Auto-retain** — extracts and retains conversation content after each response, or every N turns
- **Knowledge tools** — `agent_knowledge_*` MCP tools for listing, reading, creating, updating, and deleting knowledge pages, plus search and document ingest
- **Subagents with memory** — the `create-agent` skill builds specialized subagents backed by their own banks
- **Daemon management** — auto-starts and stops a local `hindsight-embed` daemon, or connects to an external Hindsight server
- **Dynamic bank IDs** — per-agent, per-project, or per-session memory isolation

## How Retain Works

Claude Code's `Stop` hook receives a `transcript_path` — a JSONL file with the full conversation. Devin CLI's `Stop` hook does not; its stdin carries `stop_hook_active` plus the `session_id` and `prompt_id` present on every hook event.

Devin CLI does persist every session to a local SQLite database keyed by that same `session_id`, and that is what the plugin reads instead of a transcript file.

:::note
This database is an undocumented CLI internal — there is no public API or stability guarantee for it. The read path is written defensively: a missing file, a missing table, a renamed column, or an unparseable payload all degrade to an empty transcript rather than raising, so a future CLI release that changes the storage layer disables auto-retain instead of breaking your session. If auto-retain goes quiet after a Devin CLI upgrade, that is the first thing to check.
:::

## Configuration

The config schema, environment variables, and connection modes are identical to the [Claude Code plugin](./claude-code.md) — see that page for the full settings reference. The differences:

| | Claude Code | Devin CLI |
|---|---|---|
| User config file | `~/.hindsight/claude-code.json` | `~/.hindsight/devin-cli.json` |
| Default `agentName` / `retainContext` | `claude-code` | `devin-cli` |
| Default local daemon port | `9077` | `9078` |

The separate default port lets both plugins auto-manage their own local daemon on one machine without colliding. To share a single memory bank across Claude Code and Devin CLI, point both at the same `hindsightApiUrl` instead of running two local daemons.

## Source

[`hindsight-integrations/devin-cli`](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/devin-cli)
