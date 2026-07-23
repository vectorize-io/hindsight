
# Devin Desktop

Long-term memory for [Devin Desktop](https://devin.ai) — the editor formerly known as Windsurf (Codeium) — powered by [Hindsight](https://vectorize.io/hindsight). One command connects Devin to the Hindsight MCP server and adds a rule telling the agent to use it — so it recalls relevant memory at the start of a task and retains durable facts as it goes. Recall happens at query time against your actual message, and from your seat it's automatic.

> **📝 Note**
>
Cognition rebranded Windsurf to Devin Desktop in June 2026. The MCP config still lives under `~/.codeium/windsurf/` — that's Devin Desktop's on-disk data directory and is unchanged by the rebrand. The workspace rule now lives under `.devin/rules/` (with `.windsurf/rules/` kept as a legacy fallback).
## How It Works

Devin Desktop supports two things this integration uses:

- **MCP servers:** Devin Desktop runs MCP servers configured under `mcpServers` in `~/.codeium/windsurf/mcp_config.json` and surfaces their tools to the agent. Remote servers connect via a `serverUrl` field with optional headers, so the Hindsight MCP endpoint connects directly — no bridge needed — giving the agent `recall` / `retain` / `reflect` tools.
- **Workspace rules** in `.devin/rules/`. A rule file with `trigger: always_on` frontmatter is included in every Devin request in the workspace. The integration writes a small rule there telling the agent to recall first and retain what it learns.

## Setup

```bash
pip install hindsight-devin-desktop
cd your-project
hindsight-devin-desktop init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-memory
```

`init` adds the `hindsight` MCP server to `~/.codeium/windsurf/mcp_config.json` (Devin Desktop's single global MCP config) and writes the recall/retain rule to `./.devin/rules/hindsight.md`. Reload Devin Desktop (or refresh MCP servers), and the `hindsight` server's tools become available.

Use a [Hindsight Cloud](https://hindsight.vectorize.io) key, or point at a self-hosted server with `--api-url http://localhost:8888` (no token needed for an open local server). If your `mcp_config.json` isn't plain JSON, `init` prints the entry to paste rather than rewriting the file — or run `hindsight-devin-desktop init --print-only` anytime.

## Commands

| Command | Description |
| --- | --- |
| `hindsight-devin-desktop init` | Add the MCP server + recall/retain rule |
| `hindsight-devin-desktop status` | Show whether the server + rule are configured |
| `hindsight-devin-desktop uninstall` | Remove the server + rule |

## Note

Recall and retain run through MCP tools the agent calls, guided by the always-on rule. This makes recall query-time precise (no lag), with the tradeoff that it relies on the agent following the "recall first" instruction rather than the editor enforcing it.

See the [package README](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/devin-desktop) for full configuration options.
