---
sidebar_position: 7
title: "Windsurf Persistent Memory with Hindsight | Integration"
description: "Add long-term memory to Windsurf (Codeium) with Hindsight via MCP. One command wires the Hindsight MCP server into mcp_config.json plus an always-on recall/retain rule, so memory works automatically in Cascade."
---

# Windsurf

Long-term memory for [Windsurf](https://windsurf.com) (Codeium), powered by [Hindsight](https://vectorize.io/hindsight). One command connects Cascade to the Hindsight MCP server and adds a rule telling the agent to use it — so it recalls relevant memory at the start of a task and retains durable facts as it goes. Recall happens at query time against your actual message, and from your seat it's automatic.

## How It Works

Windsurf supports two things this integration uses:

- **MCP servers:** Windsurf runs MCP servers configured under `mcpServers` in `~/.codeium/windsurf/mcp_config.json` and surfaces their tools in Cascade. Remote servers connect via a `serverUrl` field with optional headers, so the Hindsight MCP endpoint connects directly — no bridge needed — giving the agent `recall` / `retain` / `reflect` tools.
- **Workspace rules** in `.windsurf/rules/`. A rule file with `trigger: always_on` frontmatter is included in every Cascade request in the workspace. The integration writes a small rule there telling the agent to recall first and retain what it learns.

## Setup

```bash
pip install hindsight-windsurf
cd your-project
hindsight-windsurf init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-memory
```

`init` adds the `hindsight` MCP server to `~/.codeium/windsurf/mcp_config.json` (Windsurf's single global MCP config) and writes the recall/retain rule to `./.windsurf/rules/hindsight.md`. Reload Windsurf (or refresh MCP servers in Cascade), and the `hindsight` server's tools become available.

Use a [Hindsight Cloud](https://hindsight.vectorize.io) key, or point at a self-hosted server with `--api-url http://localhost:8888` (no token needed for an open local server). If your `mcp_config.json` isn't plain JSON, `init` prints the entry to paste rather than rewriting the file — or run `hindsight-windsurf init --print-only` anytime.

## Commands

| Command | Description |
| --- | --- |
| `hindsight-windsurf init` | Add the MCP server + recall/retain rule |
| `hindsight-windsurf status` | Show whether the server + rule are configured |
| `hindsight-windsurf uninstall` | Remove the server + rule |

## Note

Recall and retain run through MCP tools the agent calls, guided by the always-on rule. This makes recall query-time precise (no lag), with the tradeoff that it relies on the agent following the "recall first" instruction rather than the editor enforcing it.

See the [package README](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/windsurf) for full configuration options.
