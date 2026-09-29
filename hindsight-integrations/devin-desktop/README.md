# hindsight-devin-desktop

Long-term memory for **Devin Desktop** (the editor formerly known as Windsurf / Codeium), powered by [Hindsight](https://github.com/vectorize-io/hindsight).

`hindsight-devin-desktop init` wires the Hindsight **MCP server** into Devin
Desktop and adds always-on recall/retain rules, so the agent has `recall` /
`retain` / `reflect` tools and — guided by the rules — recalls relevant memory at
the start of a task and retains durable facts as it works.

> **Note:** Cognition rebranded Windsurf to Devin Desktop (June 2026). This
> integration configures **both agents Devin Desktop ships** (see below), so
> memory works whichever one you use.

## Two agents, both covered

Devin Desktop runs two agents with **separate** configuration, so `init` wires
each:

| | **Cascade** (legacy Windsurf agent) | **Devin Local** (the successor agent) |
| --- | --- | --- |
| MCP server | `~/.codeium/windsurf/mcp_config.json` (`serverUrl`) | `~/.config/devin/config.json` (`url` + `transport` + `headers`) |
| Tool approval | automatic | pre-seeded allow-rule (`mcp__hindsight__*`) so tools don't prompt |
| Per-project rule | `.devin/rules/hindsight.md` | repo-root `AGENTS.md` |
| Global rule | `~/.codeium/windsurf/memories/global_rules.md` | `~/.config/devin/AGENTS.md` |
| Auto-recall | — (rule-driven) | **`SessionStart` hook** injects memory deterministically |
| Retain nudge | — | **`Stop` hook** forces a retain pass before the session ends |
| Visibility | **`post_mcp_tool_use` banner** (`hooks.json`) | native tool cards + rule narration |

Configuring one agent does **not** surface the server in the other, so the
integration writes both. All the file edits are surgical — dedicated files, or a
fenced managed block inside shared files (`AGENTS.md`, `global_rules.md`,
`hooks.json`). Paths above are macOS/Linux; on Windows the Devin Local config
lives under `%APPDATA%\devin\` (Cascade stays under `~/.codeium/windsurf\`).

### Hooks (Devin Local) — deterministic memory, always visible

The MCP tools + rules are *model-driven* — the agent recalls/retains because the
rule tells it to. For Devin Local, `init` also adds two hooks (opt out of both
with `--no-hooks`):

- **`SessionStart` auto-recall** — recalls project + global memory and injects it
  into the agent's context at the start of every session, so relevant memory
  loads **even if the model forgets to call `recall`**. It **always reports
  status** (loaded N / empty / unavailable) so memory use is never silent, and
  never blocks a session.
- **`Stop` retain-nudge** — before the agent stops, it forces one retain pass
  (the model decides *what* is durable and calls `retain`). Loop-guarded. Costs
  one extra turn per session; opt out with `--no-retain-hook`.

**Why not fully-automatic retain?** Devin's hooks can't hand a script the
conversation transcript, so a hook can't summarize-and-retain on its own — the
nudge is the closest deterministic option (guaranteed *trigger*, model authors
the content). Cascade can't do either hook (its hooks can't inject context), so
recall/retain there stay model-driven — but `init` adds a `post_mcp_tool_use`
banner (`show_output`) so Cascade **visibly shows** every `🧠 Hindsight: <tool>
used`.

## Two-tier memory: global + per-project

Memory is split across two Hindsight **banks** (isolated memory scopes), so work
on one repo never bleeds into another:

- **Global bank** (`devin-desktop`) — your cross-project memory: preferences,
  coding style, who you are. Shared across every project.
- **Project bank** (`devin-desktop-<slug>`) — this repository's memory:
  architecture, decisions, conventions. The `<slug>` is derived from the repo's
  **git remote**, so it's stable across machines and identical for teammates.

The MCP server runs in **multi-bank mode** (a single endpoint ending in `/mcp/`),
and the always-on rule tells the agent which `bank_id` to use — recall both banks
at task start, retain project facts to the project bank and user facts to the
global bank. An `X-Bank-Id` header names the global bank as the default.

### Opting out of the shared layer (`--no-global-bank`)

If you'd rather your preferences *not* follow you across repos, pass
`--no-global-bank` (local-only mode): everything — project facts **and** your
preferences — stays in that repo's project bank, nothing is shared, and the
global rule files aren't written. Good for keeping work and personal machines
separate, or when a shared profile isn't wanted.

## Install

```bash
pip install hindsight-devin-desktop
cd your-project
hindsight-devin-desktop init --api-token YOUR_HINDSIGHT_API_KEY
```

`init` (run inside a repo) derives the project bank from your git remote and
wires **both agents**: the MCP server entries, the per-project rules (**commit
`./.devin/rules/hindsight.md` and `./AGENTS.md`** so teammates share the project
bank), and the global rules. Then **activate the server** in whichever agent you
use (config isn't hot-reloaded):

- **Cascade** — open the MCP panel and press **Refresh**.
- **Devin Local** — open the **Devin MCP Marketplace**, find `hindsight` under
  **Installed**, and click **Connect**.

The `hindsight` tools then load and are used automatically. (Not sure which agent
you're on? Check the **agent selector** in the bottom-right of Devin Desktop.)

### Verify it's working

Start a session and you'll *see* memory is on — nothing is silent:

- **Devin Local** — the reply opens with a status line like
  `🧠 Hindsight preloaded 3 memories for this session` (or `no memory yet`, or
  `⚠️ memory unavailable this session`). Every hook run is also logged to
  `~/.hindsight/devin-hook.log`, so you can confirm the recall/retain hooks fired.
- **Cascade** — each `recall`/`retain` shows as a tool card, and expanding the
  `post-tool hooks` line reveals the `🧠 Hindsight: <tool> used` banner.
- `hindsight-devin-desktop status` lists every component as installed for both
  agents, and the banks it resolved.

Use a [Hindsight Cloud](https://hindsight.vectorize.io) key, or a self-hosted
server with `--api-url http://localhost:8888` (no token needed for an open local
server). Pass `--bank-id <id>` to set the project bank explicitly, or
`--global-bank <id>` to change the cross-project bank. Run
`hindsight-devin-desktop init --print-only` to see everything it would write
without touching a file.

## Commands

| Command | Description |
| --- | --- |
| `hindsight-devin-desktop init` | Wire both agents' MCP server + memory rules (derives the project bank) |
| `hindsight-devin-desktop status` | Show resolved banks + whether each agent is configured |
| `hindsight-devin-desktop uninstall` | Remove the MCP server + memory rules from both agents |

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
