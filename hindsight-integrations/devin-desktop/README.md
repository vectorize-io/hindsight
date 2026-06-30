# hindsight-devin-desktop

Long-term memory for **Devin Desktop** (the editor formerly known as Windsurf / Codeium), powered by [Hindsight](https://github.com/vectorize-io/hindsight).

`hindsight-devin-desktop init` wires the Hindsight **MCP server** into Devin Desktop's
`~/.codeium/windsurf/mcp_config.json` and adds an always-on recall/retain rule to
`.devin/rules/hindsight.md`. Devin then has `recall` / `retain` / `reflect`
tools and — guided by the rule — recalls relevant memory at the start of a task
and retains durable facts as it works.

> **Note:** Cognition rebranded Windsurf to Devin Desktop (June 2026). The MCP
> config path still lives under `~/.codeium/windsurf/` — that's Devin Desktop's
> on-disk data directory and is unchanged by the rebrand. The workspace rule now
> lives under `.devin/rules/` (with `.windsurf/rules/` kept as a legacy fallback).

## How it works

Devin Desktop supports two things this integration uses:

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

- **Workspace rules** in `.devin/rules/`. A rule file with `trigger: always_on`
  frontmatter is applied to every Devin request in the workspace — that's where
  the recall/retain rule lives.

## Install

```bash
pip install hindsight-devin-desktop
cd your-project
hindsight-devin-desktop init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-project
```

`init` merges the `mcpServers` entry into `~/.codeium/windsurf/mcp_config.json`
(Devin Desktop's single global MCP config) and writes the rule into
`./.devin/rules/hindsight.md`. Reload Devin Desktop (or refresh MCP servers) and
the `hindsight` tools are available.

Use a [Hindsight Cloud](https://hindsight.vectorize.io) key, or a self-hosted
server with `--api-url http://localhost:8888` (no token needed for an open local
server). If `mcp_config.json` isn't plain JSON, `init` prints the snippet to
paste instead of touching the file — or run `hindsight-devin-desktop init --print-only`
anytime.

## Commands

| Command | Description |
| --- | --- |
| `hindsight-devin-desktop init` | Add the MCP server + recall/retain rule |
| `hindsight-devin-desktop status` | Show whether the server + rule are configured |
| `hindsight-devin-desktop uninstall` | Remove the server + rule |

## Configuration

| Setting | Env var | Default |
| --- | --- | --- |
| API URL | `HINDSIGHT_API_URL` | `https://api.hindsight.vectorize.io` |
| API token | `HINDSIGHT_API_TOKEN` | _(none; required for Cloud)_ |
| Bank id | `HINDSIGHT_DEVIN_DESKTOP_BANK_ID` | `devin-desktop` |

## Development

```bash
uv sync
uv run pytest tests -v -m 'not requires_real_llm'   # deterministic suite
uv run pytest tests -v -m requires_real_llm          # gated MCP-endpoint check
```

## License

MIT
