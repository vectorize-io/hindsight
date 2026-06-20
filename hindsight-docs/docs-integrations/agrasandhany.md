---
title: "Agrasandhany: Local Notes as Agent Memory with Hindsight | Integration"
description: "Turn the plain-text notes you already keep into long-term memory any agent can recall over MCP — your files stay yours. Agrasandhany syncs your notes vault into Hindsight with automatic change detection, folder-scoped recall, and living summaries."
---

# Agrasandhany

[Agrasandhany](https://github.com/yugandhar-maram/agrasandhany) (`agy`) turns the plain-text notes you already keep into long-term memory any agent can recall over MCP, powered by [Hindsight](https://hindsight.vectorize.io/).

You write notes for yourself — no schemas, no tagging, no curation. `agy` reads them as they are and keeps the memory in step with your files: your notes are the source of truth, and the memory only ever mirrors them.

## What you get

You keep writing notes the way you always have. `agy` turns them into memory your agents can reach:

- Every note becomes a searchable memory — its facts are retained, not just its text.
- Every folder becomes a scope, so you can ask for memories from one corner of your notes instead of all of them. A note in `projects/backend/` is scoped to `projects/backend` automatically.
- Drop a `.rr.toml` into a folder and that folder gets a living summary — an overview that rewrites itself as the notes inside it change.
- Any agent can reach the memory over MCP, so it isn't trapped in one app — your notes stay on your machine, and the recall goes wherever you do.

## How it stays in sync

A background daemon watches your notes folder — your *vault* — and keeps the memory in step:

```
vault (files) → daemon (sync) → Hindsight (memory)
                                       ↑
                                  MCP (agents)
```

You don't run anything extra — the daemon carries an embedded copy of Hindsight, so there's no separate memory service to set up. Each pass does only what changed: new and edited notes are re-read, deleted notes drop out, renames follow, and notes that haven't changed are skipped. Edit a note and its memory updates; delete it and the memory clears on the next pass.

## Use with your agent (MCP)

Point your agent at `agy`'s MCP server and it recalls scoped memories from the same notes the daemon keeps in sync. Add to your agent's MCP config (e.g. `.claude/settings.json`):

```json
{
  "mcpServers": {
    "agrasandhany": {
      "command": "uv",
      "args": ["run", "--project", "~/agrasandhany", "python", "-m", "agy.mcp"],
      "env": {
        "AGY_ENV_FILE": "~/agrasandhany/.env",
        "AGY_VAULT": "notes"
      }
    }
  }
}
```

## Get started

Setup is short — clone, start the daemon, point it at a notes folder:

```bash
git clone https://github.com/yugandhar-maram/agrasandhany.git ~/agrasandhany
cd ~/agrasandhany
uv run agy daemon start              # starts the daemon (embedded Hindsight + sync)
uv run agy vault add ~/notes --name notes   # register a vault; the daemon syncs it
uv run agy daemon status             # confirm services are running
```

You'll need [uv](https://docs.astral.sh/uv/getting-started/installation/) and an LLM provider — `agy` defaults to Claude Code, and any [Hindsight-supported provider](https://hindsight.vectorize.io/developer/configuration) works. For the full walkthrough — provider config, per-vault settings, reflections, and the complete CLI — see the agy docs:

- [Usage guide](https://github.com/yugandhar-maram/agrasandhany/blob/main/docs/usage.md) — install, daemon, vaults, MCP
- [Configuration](https://github.com/yugandhar-maram/agrasandhany/blob/main/docs/configuration.md) — environment variables and vault settings
- [Operations](https://github.com/yugandhar-maram/agrasandhany/blob/main/docs/operations.md) — CLI commands, logs, jobs, troubleshooting

---

`agy` is a community integration by [yugandhar-maram](https://github.com/yugandhar-maram), open source under the MIT license. Source, issues, and full documentation live at [github.com/yugandhar-maram/agrasandhany](https://github.com/yugandhar-maram/agrasandhany).
