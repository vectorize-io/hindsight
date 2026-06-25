# hindsight-windsurf

Long-term memory for **Windsurf** (Codeium), powered by [Hindsight](https://github.com/vectorize-io/hindsight).

`hindsight-windsurf init` wires the Hindsight **MCP server** into Windsurf's
`~/.codeium/windsurf/mcp_config.json` and adds an always-on recall/retain rule to
`.windsurf/rules/hindsight.md`. Cascade then has `recall` / `retain` / `reflect`
tools and — guided by the rule — recalls relevant memory at the start of a task
and retains durable facts as it works.

## How it works

Windsurf supports two things this integration uses:

- **MCP servers** in `~/.codeium/windsurf/mcp_config.json` under `mcpServers`,
  including **remote servers** via `serverUrl` with headers — so the Hindsight
  MCP endpoint connects directly:

  ```json
  {
    "mcpServers": {
      "hindsight": {
        "serverUrl": "https://api.hindsight.vectorize.io/mcp/my-project/",
        "headers": { "Authorization": "Bearer hsk_..." }
      }
    }
  }
  ```

- **Workspace rules** in `.windsurf/rules/`. A rule file with `trigger: always_on`
  frontmatter is applied to every Cascade request in the workspace — that's where
  the recall/retain rule lives.

## Install

```bash
pip install hindsight-windsurf
cd your-project
hindsight-windsurf init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-project
```

`init` merges the `mcpServers` entry into `~/.codeium/windsurf/mcp_config.json`
(Windsurf's single global MCP config) and writes the rule into
`./.windsurf/rules/hindsight.md`. Reload Windsurf (or refresh MCP servers in
Cascade) and the `hindsight` tools are available.

Use a [Hindsight Cloud](https://hindsight.vectorize.io) key, or a self-hosted
server with `--api-url http://localhost:8888` (no token needed for an open local
server). If `mcp_config.json` isn't plain JSON, `init` prints the snippet to
paste instead of touching the file — or run `hindsight-windsurf init --print-only`
anytime.

## Commands

| Command | Description |
| --- | --- |
| `hindsight-windsurf init` | Add the MCP server + recall/retain rule |
| `hindsight-windsurf status` | Show whether the server + rule are configured |
| `hindsight-windsurf uninstall` | Remove the server + rule |

## Configuration

| Setting | Env var | Default |
| --- | --- | --- |
| API URL | `HINDSIGHT_API_URL` | `https://api.hindsight.vectorize.io` |
| API token | `HINDSIGHT_API_TOKEN` | _(none; required for Cloud)_ |
| Bank id | `HINDSIGHT_WINDSURF_BANK_ID` | `windsurf` |

## Development

```bash
uv sync
uv run pytest tests -v -m 'not requires_real_llm'   # deterministic suite
uv run pytest tests -v -m requires_real_llm          # gated MCP-endpoint check
```

## License

MIT
