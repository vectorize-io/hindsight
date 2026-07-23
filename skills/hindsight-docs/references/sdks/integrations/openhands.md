
# OpenHands

Long-term memory for [OpenHands](https://github.com/OpenHands/OpenHands) (formerly OpenDevin), powered by [Hindsight](https://vectorize.io/hindsight). One command connects OpenHands to the Hindsight MCP server and adds a recall/retain rule — so the agent recalls relevant memory at the start of a task and retains durable facts as it works.

## How It Works

OpenHands has **native Streamable-HTTP MCP support**, so the Hindsight MCP endpoint connects directly (no bridge):

```toml
[mcp]
shttp_servers = [
    {url = "https://api.hindsight.vectorize.io/mcp/my-project/", api_key = "hsk_..."}
]
```

OpenHands also loads `AGENTS.md` (and repo microagents) into the agent's context on every task — that's where the recall/retain rule lives, telling the agent to `recall` first and `retain` durable facts.

## Setup

```bash
pip install hindsight-openhands
cd your-project
hindsight-openhands init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-project
```

`init` merges the `[mcp]` entry into `./config.toml` and writes the rule into `./AGENTS.md`. Use a [Hindsight Cloud](https://hindsight.vectorize.io) key, or point at a self-hosted server with `--api-url http://localhost:8888` (no token needed for an open local server). If `config.toml` can't be parsed safely, `init` prints the snippet to paste instead — or run `hindsight-openhands init --print-only` anytime.

## Commands

| Command | Description |
| --- | --- |
| `hindsight-openhands init` | Add the MCP server + recall/retain rule |
| `hindsight-openhands status` | Show whether the server + rule are configured |
| `hindsight-openhands uninstall` | Remove the server + rule |

See the [package README](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/openhands) for full configuration options.
