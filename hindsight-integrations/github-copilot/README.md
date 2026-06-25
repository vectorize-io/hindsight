# hindsight-copilot

Long-term memory for **GitHub Copilot** in VS Code, powered by [Hindsight](https://github.com/vectorize-io/hindsight).

`hindsight-copilot init` wires the Hindsight **MCP server** into VS Code's
`.vscode/mcp.json` and adds a recall/retain rule to `.github/copilot-instructions.md`.
Copilot's agent mode then has `recall` / `retain` / `reflect` tools and — guided
by the rule — recalls relevant memory at the start of a task and retains durable
facts as it works.

## How it works

VS Code Copilot supports two things this integration uses:

- **MCP servers** in `.vscode/mcp.json` (agent mode), including **HTTP servers**
  with headers — so the Hindsight MCP endpoint connects directly:

  ```json
  {
    "servers": {
      "hindsight": {
        "type": "http",
        "url": "https://api.hindsight.vectorize.io/mcp/my-project/",
        "headers": { "Authorization": "Bearer hsk_..." }
      }
    }
  }
  ```

- **`.github/copilot-instructions.md`**, which Copilot applies to every chat in
  the workspace — that's where the recall/retain rule lives.

## Install

```bash
pip install hindsight-copilot
cd your-project
hindsight-copilot init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-project
```

`init` merges the `servers` entry into `./.vscode/mcp.json` and writes the rule
into `./.github/copilot-instructions.md`. Reload VS Code, open Copilot Chat in
**agent mode**, and start the `hindsight` MCP server from the chat's tools menu.

Use a [Hindsight Cloud](https://hindsight.vectorize.io) key, or a self-hosted
server with `--api-url http://localhost:8888` (no token needed for an open local
server). If `mcp.json` has comments, `init` prints the snippet to paste instead
of touching the file — or run `hindsight-copilot init --print-only` anytime.

## Commands

| Command | Description |
| --- | --- |
| `hindsight-copilot init` | Add the MCP server + recall/retain rule |
| `hindsight-copilot status` | Show whether the server + rule are configured |
| `hindsight-copilot uninstall` | Remove the server + rule |

## Configuration

| Setting | Env var | Default |
| --- | --- | --- |
| API URL | `HINDSIGHT_API_URL` | `https://api.hindsight.vectorize.io` |
| API token | `HINDSIGHT_API_TOKEN` | _(none; required for Cloud)_ |
| Bank id | `HINDSIGHT_COPILOT_BANK_ID` | `copilot` |

## Development

```bash
uv sync
uv run pytest tests -v -m 'not requires_real_llm'   # deterministic suite
uv run pytest tests -v -m requires_real_llm          # gated MCP-endpoint check
```

## License

MIT
