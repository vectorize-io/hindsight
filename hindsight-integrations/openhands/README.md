# hindsight-openhands

Long-term memory for [OpenHands](https://github.com/OpenHands/OpenHands) (formerly OpenDevin), powered by [Hindsight](https://github.com/vectorize-io/hindsight).

`hindsight-openhands init` wires the Hindsight **MCP server** into OpenHands'
`config.toml` and adds a recall/retain rule to your project's `AGENTS.md`. The
agent then has `recall` / `retain` / `reflect` tools and — guided by the rule —
recalls relevant memory at the start of a task and retains durable facts as it
works.

## How it works

OpenHands has **native Streamable-HTTP MCP support**, so the Hindsight MCP
endpoint connects directly (no bridge):

```toml
[mcp]
shttp_servers = [
    {url = "https://api.hindsight.vectorize.io/mcp/my-project/", api_key = "hsk_..."}
]
```

OpenHands also loads `AGENTS.md` (and repo microagents) into the agent's context
on every task — that's where the recall/retain rule lives.

## Install

```bash
pip install hindsight-openhands
cd your-project
hindsight-openhands init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-project
```

`init` merges the `[mcp]` entry into `./config.toml` and writes the rule into
`./AGENTS.md`. Use [Hindsight Cloud](https://hindsight.vectorize.io), or a
self-hosted server with `--api-url http://localhost:8888` (no token needed for an
open local server).

> If `config.toml` can't be parsed safely, `init` prints the exact `[mcp]`
> snippet to paste instead of touching the file. `hindsight-openhands init
> --print-only` shows the snippet + rule anytime.

### Running the OpenHands Docker app?

The containerized OpenHands app loads MCP servers from its **UI settings**, not
from a project `config.toml`. So add the server in **Settings → MCP** as a
**Streamable HTTP** server (not SSE — Hindsight's endpoint is streamable HTTP):

- **URL:** `http://host.docker.internal:8888/mcp/<bank>/` (use `host.docker.internal`,
  not `localhost`, so the container can reach a Hindsight server on your host;
  launch the app with `--add-host host.docker.internal:host-gateway`)
- **API key:** your `hsk_...` for Cloud, or none for an open local server

The MCP tools load when a conversation starts. The `AGENTS.md` rule still applies
when you open the project as a repository.

## Commands

| Command | Description |
| --- | --- |
| `hindsight-openhands init` | Add the MCP server + recall/retain rule |
| `hindsight-openhands status` | Show whether the server + rule are configured |
| `hindsight-openhands uninstall` | Remove the server + rule |

## Configuration

| Setting | Env var | Default |
| --- | --- | --- |
| API URL | `HINDSIGHT_API_URL` | `https://api.hindsight.vectorize.io` |
| API token | `HINDSIGHT_API_TOKEN` | _(none; required for Cloud)_ |
| Bank id | `HINDSIGHT_OPENHANDS_BANK_ID` | `openhands` |

## Development

```bash
uv sync
uv run pytest tests -v -m 'not requires_real_llm'   # deterministic suite
uv run pytest tests -v -m requires_real_llm          # gated MCP-endpoint check
```

## License

MIT
