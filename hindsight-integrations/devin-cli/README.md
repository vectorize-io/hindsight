# Hindsight Memory Plugin for Devin CLI

Biomimetic long-term memory for [Devin CLI](https://docs.devin.ai/cli) using [Hindsight](https://vectorize.io/hindsight). A port of [`hindsight-integrations/claude-code`](../claude-code/) — same hooks, same config schema, same MCP tools — adapted to Devin CLI's hook payloads and plugin system.

## Status

Functionally equivalent to the Claude Code plugin: auto-recall, auto-retain, `agent_knowledge_*` MCP tools, and a memory-backed subagent creator. Two things differ from the Claude Code plugin, both explained in detail further down:

1. **Setup is a two-step process** (`devin plugins install` + running this plugin's `setup` skill once) instead of one, because Devin CLI's plugin system can't yet template a bundled script's own install path into hooks it registers. See [Setup](#setup) and `scripts/install.py`.
2. **Retain reads Devin CLI's local session database instead of a transcript file**, because Devin CLI hooks don't expose a `transcript_path`. See [How retain works](#how-retain-works). This reads an undocumented internal of the CLI — see that section before relying on it in a context where silent retain failures would be costly.

## Quick Start

```bash
# 1. Install the plugin
devin plugins install vectorize-io/hindsight#hindsight-integrations/devin-cli

# 2. Register hooks + the MCP server (one-time; see "Setup" below for why this
#    extra step exists). Easiest way: start a Devin CLI session and ask it to
#    set up Hindsight memory — it'll invoke this plugin's `setup` skill. Or
#    run it directly — the cache path carries the plugin version, so select the
#    newest match instead of letting the shell expand to several paths (python3
#    would run the first and silently treat the rest as arguments):
python3 "$(ls -td ~/.local/share/devin/cli/plugins/cache/*/hindsight-memory/*/scripts/install.py | head -1)"

# 3. Configure your LLM provider for memory extraction
export OPENAI_API_KEY="sk-your-key"        # or ANTHROPIC_API_KEY, etc.
# ...or point at an external Hindsight server instead of a local daemon. This
# merges the key into ~/.hindsight/devin-cli.json — a plain `>` redirect would
# replace the whole file, dropping any bankId/token/debug settings already in it:
mkdir -p ~/.hindsight && python3 -c 'import json,pathlib,sys
p = pathlib.Path.home() / ".hindsight/devin-cli.json"
cfg = json.loads(p.read_text()) if p.exists() else {}
cfg["hindsightApiUrl"] = sys.argv[1]
p.write_text(json.dumps(cfg, indent=2) + "\n")' https://your-hindsight-server.com

# 4. Start a new Devin CLI session — hooks and the MCP server load at session start
devin
```

## Setup

Devin CLI's plugin system (3000.3.22+) can install a plugin's skills, rules, and subagents cleanly, and it *tries* to wire up a plugin-shipped `hooks.json` / `mcp_config.json` too. But as of this writing, hook and MCP server commands have no portable way to reference their own bundled scripts: there's no `${DEVIN_PLUGIN_ROOT}`-style substitution (unlike Claude Code's `${CLAUDE_PLUGIN_ROOT}`), and a relative path in a plugin's own hooks.json resolves against the *project* working directory at hook-run time, not the plugin's install directory — verified empirically against CLI 3000.3.22 (see `scripts/install.py`'s docstring for the exact test).

So this plugin ships `scripts/install.py`, which writes absolute-path hook and MCP server entries directly into Devin CLI's documented, stable config locations:

- `~/.config/devin/config.json` (`"hooks"` key)
- `~/.config/devin/mcp_config.json` (`"mcpServers"` key)

Run it once after installing (or after moving/updating the plugin — it's idempotent). The easiest way is to just ask Devin to do it — the `setup` skill (`/hindsight-memory:setup`) knows how to find and run it. If CLI plugin path templating gets fixed in a later release, this step goes away and `devin plugins install` alone will be enough — recheck `scripts/install.py`'s docstring against the current CLI changelog before assuming that's still necessary.

It only ever replaces entries it wrote itself; anything else in those files is left alone. If you already have an MCP server named `hindsight` that this plugin didn't write, it will say so and register nothing rather than overwrite it — rename yours, or re-run with `--force` to replace it.

## Features

- **Auto-recall** — on every user prompt, queries Hindsight for relevant memories and injects them as context (invisible to the chat transcript, visible to the agent)
- **Auto-retain** — after every response (or every N turns), extracts and retains conversation content to Hindsight for long-term storage
- **Knowledge tools** — MCP server exposing `agent_knowledge_*` tools for managing knowledge pages (list, get, create, update, delete), searching memories, and ingesting documents
- **Subagents with memory** — create specialized subagents with long-term memory via the `create-agent` skill (`/hindsight-memory:create-agent`)
- **Daemon management** — can auto-start/stop `hindsight-embed` locally or connect to an external Hindsight server
- **Dynamic bank IDs** — supports per-agent, per-project, or per-session memory isolation
- **Zero dependencies** — hooks are pure Python stdlib; the MCP server requires the `mcp` pip package

Config schema, env vars, and connection modes are identical to the [Claude Code plugin](../claude-code/README.md) — see that README for the full settings reference (bank/recall/retain/knowledge-tools tables). The only differences: the user config file is `~/.hindsight/devin-cli.json` instead of `~/.hindsight/claude-code.json`, `agentName`/`retainContext` default to `"devin-cli"`, and the local daemon defaults to port `9078` instead of `9077` (so both plugins can each auto-manage their own local daemon without colliding — point both at the same `hindsightApiUrl` instead if you want Claude Code and Devin CLI to share one memory bank).

## Architecture

| Component | Event/Transport | Purpose |
|-----------|-----------------|---------|
| `scripts/session_start.py` | `SessionStart` hook | Health check — verify Hindsight is reachable, pre-warm the daemon if not |
| `scripts/recall.py` | `UserPromptSubmit` hook | **Auto-recall** — query memories, inject as `additionalContext` |
| `scripts/retain.py` | `Stop` hook | **Auto-retain** — read the session's messages, POST to Hindsight |
| `scripts/session_end.py` | `SessionEnd` hook | Final retain + daemon cleanup |
| `scripts/mcp_server.py` | MCP stdio server (via `scripts/run_mcp.sh`) | **Knowledge tools** — `agent_knowledge_*` tools for pages, recall, ingest |
| `skills/create-agent/` | Skill | **Agent creation** — memory-backed subagent wizard |
| `skills/setup/` | Skill | **One-time setup** — registers hooks + MCP server (see [Setup](#setup)) |
| `scripts/install.py` / `scripts/uninstall.py` | — | The actual hook/MCP registration logic the `setup` skill runs |

### Library modules (`scripts/lib/`)

Shared with the Claude Code plugin almost verbatim — `client.py` (REST client), `config.py` (settings loader), `daemon.py` (`hindsight-embed` lifecycle), `bank.py` (bank ID derivation), `content.py` (transcript formatting), `state.py` (file-based state), `llm.py` (LLM provider auto-detection). One new module: `devin_transcript.py`.

`content.py` started as a byte-for-byte copy and has since diverged by a handful of hardening fixes found while reviewing this port — a channel-envelope regex that was unanchored, unguarded `.strip()` on caller-supplied queries, and a turn-boundary scan that counted tool-result messages as user turns. Each is a bug in the Claude Code plugin too; they are fixed here rather than there only to keep this PR to one integration.

## How retain works

Claude Code's `Stop` hook gets a `transcript_path` on stdin — a JSONL file with the full conversation. Devin CLI's `Stop` hook stdin only has `stop_hook_active` (plus a `session_id` and `prompt_id` present on every hook event as of CLI 3000.3.22). There's no `transcript_path` and no documented way to get one.

What Devin CLI *does* do is persist every session to a local SQLite database — `~/.local/share/devin/cli/sessions.db` (schema: `sessions`, `message_nodes`) — written live (WAL mode) as the conversation progresses, keyed by the exact same `session_id` the hooks receive. `scripts/lib/devin_transcript.py` reads it: `message_nodes` stores one row per message *as sent to the model in a given API call*, so the same system/rules/prior-turn messages get re-inserted verbatim on every subsequent turn (each row carries a stable `message_id` in its JSON payload); de-duplicating by first-seen `message_id` reconstructs the same linear, one-entry-per-turn transcript Claude Code's JSONL file represents. From there, retain.py hands the reconstructed message list to the same `lib/content.py` formatting logic the Claude Code plugin uses, unmodified.

**This is reading an undocumented internal of the CLI**, not a public API — there's no stability guarantee, and it was verified empirically (see `scripts/lib/devin_transcript.py`'s docstring) rather than from documentation. It degrades safely: if the database is missing, empty for a session, or has an unrecognized schema, `read_session_messages()` returns an empty list and retain/multi-turn-recall silently no-op rather than raising. If a future Devin CLI release changes this storage layer, retain will quietly stop working — set `"debug": true` in `~/.hindsight/devin-cli.json` and watch for `[Hindsight] No messages found for session` in a hook's stderr as the symptom, and recheck this file's docstring against the CLI's current internals before filing that as a plugin bug.

If Devin CLI ever adds a documented `transcript_path` (or a stable, public way to read a session's history) to hook stdin, switch to that instead — it would remove the undocumented-internals dependency entirely. Track this in the CLI's changelog for `hooks` / `lifecycle-hooks` changes.

## Subagents with Memory

Create specialized subagents that learn and build knowledge across sessions.

Just tell Devin:

> "Create a code review agent using the create-agent skill"

Devin will:
1. Ask for the agent name and description
2. Write the subagent file to `~/.config/devin/agents/<name>/AGENT.md`
3. Ingest any seed content you provide
4. Create initial knowledge pages

Unlike Claude Code subagents, Devin CLI custom subagents have no `mcpServers` frontmatter field — MCP servers are session-wide, so a subagent without a restrictive `allowed-tools` list already has access to the `hindsight` server's tools. See `skills/create-agent/SKILL.md` for the exact template.

### Knowledge Tools (MCP)

Same tool set as the Claude Code plugin:

| Tool | Description |
|------|-------------|
| `agent_knowledge_list_pages` | List all knowledge pages |
| `agent_knowledge_get_page` | Read a specific page |
| `agent_knowledge_create_page` | Create a new page with a source query |
| `agent_knowledge_update_page` | Update a page's name or source query |
| `agent_knowledge_delete_page` | Delete a page |
| `agent_knowledge_recall` | Search memories |
| `agent_knowledge_ingest` | Ingest text content |
| `agent_knowledge_ingest_file` | Ingest a file from disk |
| `agent_knowledge_get_current_bank` | Get the current bank ID |

## Troubleshooting

### Hooks/MCP server not active

Run `/hooks` and `/mcp` in a Devin CLI session. If nothing from this plugin shows up, you likely skipped the [Setup](#setup) step — run the `setup` skill (or `scripts/install.py` directly), then start a **new** session (hooks/MCP servers load at session start, not mid-session).

### Recall returning no memories

- Verify the Hindsight server is reachable: `curl http://localhost:9078/health` (local daemon) or your configured `hindsightApiUrl`.
- Set `"debug": true` in `~/.hindsight/devin-cli.json` and check the session's log under `~/.local/share/devin/cli/logs/` for `[Hindsight]` lines (hook stderr isn't shown in the TUI directly).

### Retain not storing anything

See [How retain works](#how-retain-works) above — this is the part of this plugin most likely to break on a future CLI release. Check for `[Hindsight] No messages found for session` in debug logs.

## License

MIT — see [LICENSE](./LICENSE).
