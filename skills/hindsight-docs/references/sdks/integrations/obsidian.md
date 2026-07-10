
# Obsidian

Give your [Obsidian](https://obsidian.md) vault an agent that actually knows your notes, powered by [Hindsight](https://hindsight.vectorize.io). The plugin syncs your vault into a Hindsight bank and adds a chat panel whose answers are grounded on your notes — and cite them.

## Why this matters

Obsidian is where your knowledge lives. Vector-search plugins can find related notes, but they can't reason across them, remember what you asked, or keep a running synthesis as your vault grows. Hindsight adds that layer:

- **Recall** is faster and more accurate than text search.
- **Reflect** reasons over your whole vault to answer a question, and shows the notes it used.
- **Mental models** (on the roadmap) keep living summaries of your vault that refresh as you write.

## Source of truth stays in Obsidian

A hard rule of this integration: **Hindsight never becomes a second source of truth.** Sync is one-way (Obsidian → Hindsight), every answer cites the note it came from so you can **fix things at the source**, and chat conversations are **not** stored by default. Edit a note, and Hindsight reconverges on the next sync.

## What it does

- **Incremental vault sync** — notes are retained as Hindsight documents. Edits upsert, deletes remove. A content hash means unchanged notes are skipped.
- **Implicit scoping** — every note is auto-tagged on ingest with its **vault**, **folder** (and sub-folders), and **created/updated dates**. You never think about scope until you recall — then filter by any combination via Hindsight's `tag_groups`. Multiple vaults share one bank and stay separable by their `vault:` tag.
- **Same data from UI and API** — the Obsidian chat panel and your automations (n8n, Hermes, …) hit the same bank with the same tags, so they see the same scoped view.
- **Grounded chat** — a side panel that answers questions over your notes via Reflect, with collapsible **citations** (click to open the source note) and a **reasoning** disclosure.
- **Manual or automatic** — sync on every edit, or run _Sync vault now_ on demand.

### Scoping

Recall/reflect (from the UI or an API call) can filter by any combination of:

| Dimension            | Tag(s)                               |
| -------------------- | ------------------------------------ |
| Vault                | `vault:<name>`                       |
| Folder (+ ancestors) | `folder:Work`, `folder:Work/Clients` |
| Date                 | `created:2026-03`, `updated:2026-06` |

Your own frontmatter `tags`/`aliases` are carried through too.

## Installation

> ✨ **Recommended:** [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) — sign up free, get an API key, and skip self-hosting.

While the plugin is in beta it installs via [BRAT](https://github.com/TfTHacker/obsidian42-brat): add the repository [`vectorize-io/hindsight-obsidian`](https://github.com/vectorize-io/hindsight-obsidian) (the dedicated plugin repo BRAT installs from) and enable it in **Settings → Community plugins**.

**Self-hosting alternative** — run Hindsight locally:

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-openai-key
hindsight-api
```

## Configuration

Open **Settings → Hindsight**:

| Setting                   | Default                              | Description                                                                     |
| ------------------------- | ------------------------------------ | ------------------------------------------------------------------------------- |
| API URL                   | `https://api.hindsight.vectorize.io` | Hindsight server (use `http://localhost:8888` for self-hosted)                  |
| API key                   | —                                    | Hindsight Cloud API key                                                         |
| Bank name                 | `obsidian`                           | Shared bank for all vaults (separated by `vault:` tags)                         |
| Include / exclude folders | —                                    | Limit which notes sync                                                          |
| Sync on edit              | on                                   | Re-ingest notes automatically as you edit                                       |
| Default chat depth        | low                                  | Reflect budget for chat answers                                                 |
| Remember conversations    | **off**                              | When on, chat turns are stored in Hindsight (creates memory outside your vault) |

## Commands

- **Sync vault now** — full reconcile (ingest changed notes, prune deleted ones).
- **Ingest current note** — force-sync the active note.
- **Open chat** — open the grounded chat panel.
