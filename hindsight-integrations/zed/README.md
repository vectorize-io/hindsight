# hindsight-zed

Long-term memory for the [Zed](https://zed.dev) editor's AI assistant, powered by
[Hindsight](https://github.com/vectorize-io/hindsight).

`hindsight-zed init` wires Zed's Agent Panel to the Hindsight **MCP server** and
adds a rule telling the agent to use it — so it recalls relevant memory at the
start of a task and retains durable facts as it goes. Recall happens at query
time against your actual message (no lag), and from your seat it's automatic.

## How it works

Zed has no pre-prompt hook, but it does support two things this integration uses:

- **MCP context servers** — Zed runs MCP servers configured under
  `context_servers` in `settings.json` and exposes their tools in the Agent
  Panel. We register the Hindsight MCP server there, giving the agent
  `recall` / `retain` / `reflect` tools.
- **A global instructions file** (`~/.config/zed/AGENTS.md`) that Zed includes in
  every conversation. We add a small rule there telling the agent to recall
  first and retain what it learns.

Zed doesn't yet have native HTTP-MCP transport, so the server is connected
through the [`mcp-remote`](https://www.npmjs.com/package/mcp-remote) stdio bridge
(run via `npx`). Because that bridge already runs on Node.js, this setup tool is
a Node CLI too — so **Node.js is the only requirement**.

## Install

No global install needed — run it straight from npm with `npx`:

```bash
npx @vectorize-io/hindsight-zed init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-memory
```

Or install it for a persistent command:

```bash
npm install -g @vectorize-io/hindsight-zed
hindsight-zed init --api-token YOUR_HINDSIGHT_API_KEY --bank-id my-memory
```

`init` adds the `hindsight` MCP server to `~/.config/zed/settings.json` and the
recall/retain rule to `~/.config/zed/AGENTS.md`. Restart Zed, open the Agent
Panel, and the `hindsight` server should show a green dot.

Use a [Hindsight Cloud](https://hindsight.vectorize.io) key, or point at a
self-hosted server with `--api-url http://localhost:8888` (no token needed for an
open local server).

> If your `settings.json` contains comments (JSONC), `init` won't rewrite it —
> it prints the exact `context_servers` entry for you to paste instead. Use
> `hindsight-zed init --print-only` any time to see the snippet without writing.

## Commands

| Command                           | Description                                   |
| --------------------------------- | --------------------------------------------- |
| `hindsight-zed init`              | Add the MCP server + recall/retain rule       |
| `hindsight-zed status`            | Show whether the server + rule are configured |
| `hindsight-zed uninstall`         | Remove the server + rule                      |
| `hindsight-zed init --print-only` | Print the config to add manually              |

(Prefix any of these with `npx ` if you didn't install globally.)

## What gets written

`~/.config/zed/settings.json`:

```jsonc
{
  "context_servers": {
    "hindsight": {
      "source": "custom",
      "command": "npx",
      "args": [
        "-y",
        "mcp-remote",
        "https://api.hindsight.vectorize.io/mcp/my-memory/",
        "--header",
        "Authorization: Bearer YOUR_HINDSIGHT_API_KEY",
      ],
    },
  },
}
```

`~/.config/zed/AGENTS.md` (inside a fenced `<!-- HINDSIGHT -->` block that leaves
the rest of the file untouched): a short rule telling the agent to `recall` at
the start of each task and `retain` durable facts.

## Configuration

| Setting   | Env var                 | Default                              |
| --------- | ----------------------- | ------------------------------------ |
| API URL   | `HINDSIGHT_API_URL`     | `https://api.hindsight.vectorize.io` |
| API token | `HINDSIGHT_API_TOKEN`   | _(none; required for Cloud)_         |
| Bank id   | `HINDSIGHT_ZED_BANK_ID` | `zed`                                |

These can also live in `~/.hindsight/zed.json` (written by `init`).

## Development

Requires Node.js ≥ 18.3. There are no dependencies to install.

```bash
node --test        # run the test suite
```

## License

MIT
