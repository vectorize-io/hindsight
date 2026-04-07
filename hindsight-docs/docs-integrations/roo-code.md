---
sidebar_position: 20
title: "Roo Code Persistent Memory with Hindsight | Integration"
description: "Add persistent long-term memory to Roo Code via Hindsight MCP. Auto-recalls relevant context before each task and retains learnings after — so every session builds on the last."
---

# Roo Code

Persistent long-term memory for [Roo Code](https://github.com/RooVetGit/Roo-Code) via [Hindsight](https://vectorize.io/hindsight). A one-command installer registers Hindsight's MCP server and injects custom rules that teach Roo to recall context before tasks and retain learnings after.

## Quick Start

```bash
# 1. Start Hindsight (self-hosted)
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-openai-key
hindsight-api

# 2. Install the integration into your project
python /path/to/hindsight-integrations/roo-code/install.py

# 3. Restart Roo Code — memory is active
```

Or with [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup):

```bash
python install.py --api-url https://api.hindsight.vectorize.io
```

## How It Works

Roo Code has two primary extensibility mechanisms: **MCP servers** for tools and **custom rules** for system-prompt injection. This integration uses both.

```
New task starts
  └─ Rules file instructs Roo to call hindsight_recall
       └─ Relevant memories injected into context automatically

Agent working…
  └─ Agent calls hindsight_retain for significant decisions/discoveries

Task ends
  └─ Rules file instructs Roo to call hindsight_retain with a summary
       └─ Summary stored for future sessions
```

The installer writes:
- **`.roo/mcp.json`** — registers Hindsight's `/mcp` endpoint as an MCP server, with `recall` and `retain` auto-approved
- **`.roo/rules/hindsight-memory.md`** — instructions injected into every Roo system prompt

## Installation Options

### Project-local install (default)

Writes to `.roo/` in the current directory — memory is scoped to this project:

```bash
python install.py
python install.py --api-url http://localhost:8888  # default
python install.py --project-dir /path/to/project
```

### Global install

Writes to `~/.roo/` — applies to all projects:

```bash
python install.py --global
```

## Configuration

The MCP entry written to `.roo/mcp.json`:

```json
{
  "mcpServers": {
    "hindsight": {
      "url": "http://localhost:8888/mcp",
      "timeout": 10000,
      "alwaysAllow": ["recall", "retain"]
    }
  }
}
```

To update the API URL after installation, re-run the installer or edit `.roo/mcp.json` directly.

## MCP Tools

Hindsight exposes two tools via its `/mcp` endpoint:

| Tool | Description |
|------|-------------|
| `recall` | Search memory for context relevant to a query |
| `retain` | Store content in memory immediately |

The rules file instructs Roo to call these automatically at task start and end. Agents can also call them explicitly mid-task.

## Verifying Setup

1. Start Hindsight and run the installer
2. Open Roo Code in your project
3. Check **Settings → MCP Servers** — `hindsight` should show as connected
4. Start a task — you should see `hindsight_recall` invoked in the tool call log

## Prerequisites

A running Hindsight instance:

**Self-hosted:**
```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-api-key
hindsight-api  # starts on http://localhost:8888
```

**Hindsight Cloud:** [Sign up](https://ui.hindsight.vectorize.io/signup) — no self-hosting required.
