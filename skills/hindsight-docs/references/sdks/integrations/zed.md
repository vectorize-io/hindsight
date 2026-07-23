
# Zed

Long-term memory for the [Zed](https://zed.dev) editor's AI assistant, powered by [Hindsight](https://vectorize.io/hindsight). One command connects Zed's Agent Panel to the Hindsight MCP server and adds a rule telling the agent to use it — so it recalls relevant memory at the start of a task and retains durable facts as it goes. Recall happens at query time against your actual message, and from your seat it's automatic.

## How It Works

Zed has no pre-prompt hook, but it supports two things this integration uses:

- **MCP context servers:** Zed runs MCP servers configured under `context_servers` in `settings.json` and surfaces their tools in the Agent Panel. `hindsight-zed` registers the Hindsight MCP server there, giving the agent `recall` / `retain` / `reflect` tools.
- **A global instructions file** (`~/.config/zed/AGENTS.md`) that Zed includes in every conversation. The integration adds a small rule there telling the agent to recall first and retain what it learns.

Zed doesn't yet have native HTTP-MCP transport, so the server is connected through the [`mcp-remote`](https://www.npmjs.com/package/mcp-remote) stdio bridge (run via `npx`), which means Node.js must be installed.

## Setup

`hindsight-zed` is a zero-dependency Node CLI — Node.js is the only requirement (already needed for the `mcp-remote` bridge). Run it straight from npm with `npx`:

```bash
npx @vectorize-io/hindsight-zed init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-memory
```

Or install it globally for a persistent command:

```bash
npm install -g @vectorize-io/hindsight-zed
hindsight-zed init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-memory
```

`init` adds the `hindsight` MCP server to `~/.config/zed/settings.json` and the recall/retain rule to `~/.config/zed/AGENTS.md`. Restart Zed, open the Agent Panel, and the `hindsight` server should show a green dot.

Use a [Hindsight Cloud](https://hindsight.vectorize.io) key, or point at a self-hosted server with `--api-url http://localhost:8888` (no token needed for an open local server). If your `settings.json` has comments (JSONC), `init` prints the entry to paste rather than rewriting the file — or run `hindsight-zed init --print-only` anytime.

## Commands

| Command | Description |
| --- | --- |
| `hindsight-zed init` | Add the MCP server + recall/retain rule |
| `hindsight-zed status` | Show whether the server + rule are configured |
| `hindsight-zed uninstall` | Remove the server + rule |

## Note

Recall and retain run through MCP tools the agent calls, guided by the always-on rule. This makes recall query-time precise (no lag), with the tradeoff that it relies on the agent following the "recall first" instruction rather than the editor enforcing it.

See the [package README](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/zed) for full configuration options.
