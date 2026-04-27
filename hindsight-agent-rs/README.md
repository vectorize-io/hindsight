# hindsight-agent

Single-binary CLI for [Hindsight Wiki](https://github.com/vectorize-io/hindsight) — self-learning knowledge pages for AI agents.

Agents get a persistent wiki that evolves from their conversations. The agent reads wiki pages at session startup, creates new pages when it discovers recurring topics, and the system keeps pages updated automatically via async consolidation.

## Install

```bash
# From source
cargo install --path .

# Or build and copy
cargo build --release
cp target/release/hindsight-agent ~/.local/bin/
```

## Quick start

```bash
# 1. Set up an agent (one-time)
hindsight-agent setup my-agent \
  --bank-id my-bank \
  --harness hermes \
  --api-url http://localhost:8888

# 2. Create a wiki page
hindsight-agent wiki create my-agent user-prefs \
  "User Preferences" \
  "What are the user's preferences for tone, format, and content?"

# 3. Ingest a reference document
hindsight-agent ingest my-agent "Style Guide" -f style-guide.md

# 4. Search memories
hindsight-agent recall my-agent "what format does the user prefer"

# 5. List wiki pages
hindsight-agent wiki list my-agent
```

## How it works

```
User ↔ Agent conversation
    ↓
Harness plugin retains conversation → Hindsight bank (async)
    ↓
Consolidation extracts observations (background)
    ↓
Each wiki page re-runs its synthesis query against new observations
    ↓
Agent reads updated pages at next session startup
```

The agent decides **what** to track (creates pages with synthesis queries). The system handles **capture** (harness plugin) and **synthesis** (consolidation + page refresh).

## Commands

### `setup` — One-time agent onboarding

```bash
hindsight-agent setup <agent-id> \
  --bank-id <bank> \
  --harness hermes|openclaw \
  [--api-url <url>] \
  [--api-token <token>] \
  [--template <template.json>] \
  [--content <content-dir/>]
```

Creates the Hindsight bank, imports a template (optional), ingests reference docs (optional), configures the harness (creates Hermes profile or OpenClaw agent), and saves the agent config.

### `agents` — Manage agents

```bash
hindsight-agent agents list          # show all configured agents
hindsight-agent agents show <agent>  # show one agent's config
```

### `wiki` — Knowledge pages

```bash
hindsight-agent wiki list <agent>
hindsight-agent wiki get <agent> <page-id>
hindsight-agent wiki create <agent> <page-id> "<name>" "<synthesis-query>"
hindsight-agent wiki update <agent> <page-id> [--name "..."] [--source-query "..."]
hindsight-agent wiki delete <agent> <page-id>
```

Pages are created with opinionated defaults:
- `mode: delta` — only processes new observations per refresh
- `fact_types: [observation]` — synthesizes from observations only
- `exclude_mental_models: true` — pages don't feed into each other
- `refresh_after_consolidation: true` — auto-updates after each consolidation

### `recall` — Search memories

```bash
hindsight-agent recall <agent> "<query>" [-n 10] [--type observation]
```

### `ingest` — Upload documents

```bash
hindsight-agent ingest <agent> "<title>" -f document.md
hindsight-agent ingest <agent> "<title>" -c "inline content"
cat data.txt | hindsight-agent ingest <agent> "<title>"
```

### `documents` — List retained documents

```bash
hindsight-agent documents <agent>
```

### `retain` — Raw content retention

```bash
echo "content" | hindsight-agent retain <agent> [--document-id <id>]
hindsight-agent retain <agent> --input file.txt
```

## Config

Agent configs live at `~/.hindsight-agent/config.json`:

```json
{
  "agents": {
    "my-agent": {
      "bank_id": "my-bank",
      "api_url": "http://localhost:8888",
      "api_token": "hst_...",
      "harness": "hermes"
    }
  }
}
```

All commands resolve the agent ID to bank + API URL + token from this file. The agent never sees bank IDs.

## Connecting to Hindsight

```bash
# Local (default)
hindsight-agent setup my-agent --bank-id my-bank --harness hermes

# Self-hosted
hindsight-agent setup my-agent --bank-id my-bank --harness hermes \
  --api-url https://hindsight.internal.company.com

# Cloud
hindsight-agent setup my-agent --bank-id my-bank --harness hermes \
  --api-url https://api.hindsight.cloud \
  --api-token hst_your_token
```

Environment variables `HINDSIGHT_API_URL` and `HINDSIGHT_API_TOKEN` are also supported.

## Harness support

| Harness | Setup creates | Retain method |
|---------|---------------|---------------|
| **Hermes** | Profile, sets memory provider | `hindsight_agent` memory plugin (sync_turn + on_session_end) |
| **OpenClaw** | Agent, registers plugin | `hindsight-agent` plugin (reads config, POSTs on agent_end) |

The skill (`agent-knowledge`) is harness-agnostic — it uses `hindsight-agent` CLI commands that work identically across harnesses.
