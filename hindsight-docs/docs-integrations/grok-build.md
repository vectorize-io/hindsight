---
sidebar_position: 7
title: "Grok Build Persistent Memory with Hindsight | Integration"
description: "Add long-term memory to Grok Build with Hindsight. Automatically captures conversations and recalls relevant context across sessions — powered by the Claude Code plugin."
---

# Grok Build

Biomimetic long-term memory for [Grok Build](https://x.ai/cli) using [Hindsight](https://vectorize.io/hindsight). Automatically captures conversations and recalls relevant context across sessions — no changes to your workflow required.

:::tip Powered by the Claude Code plugin
Grok Build natively supports Claude Code plugins, including hooks, MCP servers, and skills. This integration uses the same [`hindsight-memory` plugin](/sdks/integrations/claude-code) that powers Claude Code — all features, configuration options, and knowledge tools are fully available in Grok Build.
:::

## Quick Start

:::tip Recommended: Hindsight Cloud
[Sign up free](https://ui.hindsight.vectorize.io/signup) for a Hindsight Cloud API key — no self-hosting, no local daemon to manage.
:::

```bash
# 1. Add the Hindsight marketplace and install the plugin
grok plugin marketplace add vectorize-io/hindsight
grok plugin install hindsight-memory

# 2. Configure your connection
mkdir -p ~/.hindsight
cat > ~/.hindsight/claude-code.json << 'EOF'
{
  "hindsightApiUrl": "https://api.hindsight.vectorize.io",
  "hindsightApiToken": "YOUR_API_KEY"
}
EOF

# 3. Start Grok Build — memory works automatically
grok
```

That's it! The plugin will automatically start capturing and recalling memories.

### Using a local daemon instead

If you prefer to run Hindsight locally, set an LLM API key and leave `hindsightApiUrl` empty:

```bash
export OPENAI_API_KEY="sk-your-key"
# or
export ANTHROPIC_API_KEY="your-key"
```

The plugin will auto-start a local `hindsight-embed` daemon.

## Features

- **Auto-recall** — on every user prompt, queries Hindsight for relevant memories and injects them as context (invisible to the chat transcript, visible to Grok)
- **Auto-retain** — after every response (or every N turns), extracts and retains conversation content for long-term storage
- **Knowledge tools** — Grok can read, write, and search its own memory via MCP tools (`agent_knowledge_recall`, `agent_knowledge_ingest`, etc.)
- **Subagent skill** — `/hindsight-memory:create-agent` scaffolds a subagent backed by an isolated memory bank
- **Dynamic bank IDs** — per-agent, per-project, or per-session memory isolation
- **Daemon management** — can auto-start/stop `hindsight-embed` locally or connect to an external Hindsight server

## Architecture

The plugin hooks into Grok Build's lifecycle events:

| Component | Trigger | Purpose |
|-----------|---------|---------|
| `session_start.py` | `SessionStart` hook | Health check — verify Hindsight is reachable |
| `recall.py` | `UserPromptSubmit` hook | **Auto-recall** — query memories, inject as `additionalContext` |
| `retain.py` | `Stop` hook | **Auto-retain** — extract transcript, POST to Hindsight (async) |
| `session_end.py` | `SessionEnd` hook | Cleanup — stop auto-managed daemon if started |
| `mcp_server.py` | MCP server | Exposes `agent_knowledge_*` tools — list/get/create/update/delete pages, recall, ingest |
| `create-agent` | Skill | Scaffolds a subagent file and seeds its memory bank |

## Configuration

The plugin reads configuration from `~/.hindsight/claude-code.json` — the same file used by Claude Code. If you run both tools and want different settings for each, you can also create `~/.hindsight/grok-build.json` which takes precedence when running under Grok Build.

**Loading order** (later entries win):
1. Built-in defaults
2. Plugin `settings.json` (ships with the plugin)
3. User config (`~/.hindsight/claude-code.json`)
4. Environment variables (`HINDSIGHT_*`)

For the full configuration reference — connection settings, LLM provider, memory bank, auto-recall, auto-retain, knowledge tools, subagents, and debug options — see the [Claude Code configuration docs](/sdks/integrations/claude-code#configuration).

### Separating Grok Build and Claude Code memory

If you use both Grok Build and Claude Code and want separate memory banks for each, override the agent name and retain context:

```json
{
  "hindsightApiUrl": "https://api.hindsight.vectorize.io",
  "hindsightApiToken": "YOUR_API_KEY",
  "agentName": "grok-build",
  "retainContext": "grok-build"
}
```

Save this to `~/.hindsight/claude-code.json` (applies to both tools) or create a separate `~/.hindsight/grok-build.json` (Grok Build only).

With `dynamicBankId` enabled, changing `agentName` to `"grok-build"` produces bank IDs like `grok-build::myproject` instead of `claude-code::myproject`, fully isolating memory between the two tools.

## Per-Project Memory

To give each project its own isolated memory bank:

```json
{
  "dynamicBankId": true,
  "dynamicBankGranularity": ["agent", "project"]
}
```

With this config, running Grok Build in `~/projects/api` and `~/projects/frontend` stores and recalls memories separately. Git worktrees of the same repo share a bank by default.

## Troubleshooting

**Plugin not listed**: Run `grok inspect` and check the Plugins section for `hindsight-memory`. If missing, re-run `grok plugin install hindsight-memory`.

**Hooks not firing**: Check the Hooks section in `grok inspect` output for `hindsight-memory`. Enable `"debug": true` in your config to see `[Hindsight]` messages in stderr.

**No memories recalled**: Memories need at least one retain cycle before they're available. Complete a full session first (say something, exit, start a new session).

**High latency on recall**: Use `"recallBudget": "low"` or reduce `recallMaxTokens` for faster responses.

**Debug mode**: Add `"debug": true` to your config file:

```
[Hindsight] Recalling from bank 'grok-build::myproject', query length: 42
[Hindsight] Injecting 3 memories
[Hindsight] Retaining to bank 'grok-build::myproject', doc 'sess-abc123', 2 messages, 847 chars
```

**State files**: Plugin state is stored at `~/.grok/plugins/data/hindsight-memory/state/`. Check `last_recall.json` to see what was most recently recalled.
