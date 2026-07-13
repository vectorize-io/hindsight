# hindsight-devin-desktop

Long-term memory for **Devin Desktop** (the editor formerly known as Windsurf / Codeium), powered by [Hindsight](https://github.com/vectorize-io/hindsight).

`hindsight-devin-desktop init` wires the Hindsight **MCP server** into Devin
Desktop's global `mcp_config.json` and writes always-on memory rules, so Devin
has `recall` / `retain` / `reflect` tools and — guided by the rules — recalls
relevant memory at the start of a task and retains durable facts as it works.

> **Note:** Cognition rebranded Windsurf to Devin Desktop (June 2026). The MCP
> config still lives under `~/.codeium/` (unchanged by the rebrand); Devin's own
> docs disagree on the exact path, so `init` writes **both**
> `~/.codeium/windsurf/mcp_config.json` and `~/.codeium/mcp_config.json`. The
> workspace rule lives under `.devin/rules/` (with `.windsurf/rules/` as a legacy
> fallback).

## Two-tier memory: global + per-project

Memory is split across two Hindsight **banks** (isolated memory scopes), so work
on one repo never bleeds into another:

- **Global bank** (`devin-desktop`) — your cross-project memory: preferences,
  coding style, who you are. Shared across every project.
- **Project bank** (`devin-desktop-<slug>`) — this repository's memory:
  architecture, decisions, conventions, bugs. The `<slug>` is derived from the
  repo's **git remote**, so it's stable across machines and identical for
  teammates who clone the same repo.

The MCP server runs in **multi-bank mode** (a single `serverUrl` ending in
`/mcp/`), and the committed per-project rule tells the agent which `bank_id` to
use — recall both banks at task start, retain project facts to the project bank
and user facts to the global bank.

## How it works

Devin Desktop supports two things this integration uses:

- **MCP servers** in the global `mcp_config.json` under `mcpServers`, including
  **remote servers** via `serverUrl` with headers — so the Hindsight MCP endpoint
  connects directly, in multi-bank mode:

  ```json
  {
    "mcpServers": {
      "hindsight": {
        "serverUrl": "https://api.hindsight.vectorize.io/mcp/",
        "headers": {
          "Authorization": "Bearer hsk_...",
          "X-Bank-Id": "devin-desktop"
        }
      }
    }
  }
  ```

  (`X-Bank-Id` is the fallback bank for any call that omits `bank_id`.)

- **Workspace rules** in `.devin/rules/` (per-project, committed) plus the global
  `~/.codeium/windsurf/memories/global_rules.md`. A rule with `trigger: always_on`
  is applied to every request — that's where the recall/retain routing lives.

## Install

```bash
pip install hindsight-devin-desktop
cd your-project
hindsight-devin-desktop init --api-token YOUR_HINDSIGHT_API_KEY
```

`init` (run inside a repo) derives the project bank from your git remote, merges
the `mcpServers` entry into Devin's config, writes `./.devin/rules/hindsight.md`
(**commit this** so teammates share the project bank), and adds a managed block
to your global rules. Then **open Devin Desktop and press Refresh in the MCP
panel** (editing the config doesn't hot-reload) — the `hindsight` tools then load.

Use a [Hindsight Cloud](https://hindsight.vectorize.io) key, or a self-hosted
server with `--api-url http://localhost:8888` (no token needed for an open local
server). Pass `--bank-id <id>` to set the project bank explicitly instead of
deriving it, or `--global-bank <id>` to change the cross-project bank. If
`mcp_config.json` isn't plain JSON, `init` prints the snippet to paste instead of
touching the file — or run `hindsight-devin-desktop init --print-only` anytime.

## Commands

| Command | Description |
| --- | --- |
| `hindsight-devin-desktop init` | Add the MCP server + memory rules (derives the project bank) |
| `hindsight-devin-desktop status` | Show resolved banks + whether the server/rules are configured |
| `hindsight-devin-desktop uninstall` | Remove the MCP server + memory rules |

## Configuration

| Setting | Env var | Default |
| --- | --- | --- |
| API URL | `HINDSIGHT_API_URL` | `https://api.hindsight.vectorize.io` |
| API token | `HINDSIGHT_API_TOKEN` | _(none; required for Cloud)_ |
| Global bank | `HINDSIGHT_DEVIN_DESKTOP_GLOBAL_BANK` | `devin-desktop` |
| Project bank | `HINDSIGHT_DEVIN_DESKTOP_BANK_ID` | _(derived from git remote)_ |

## Development

```bash
uv sync
uv run pytest tests -v -m 'not requires_real_llm'   # deterministic suite
uv run pytest tests -v -m requires_real_llm          # gated MCP-endpoint check
```

## License

MIT
