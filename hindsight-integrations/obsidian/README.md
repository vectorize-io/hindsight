# Hindsight for Obsidian

Give your [Obsidian](https://obsidian.md) vault an AI agent that actually knows your notes, powered by [Hindsight](https://github.com/vectorize-io/hindsight).

The plugin syncs your vault into a Hindsight memory bank and adds a chat panel whose answers are **grounded on your notes — and cite them**.

## Source of truth stays in Obsidian

A hard rule of this plugin: **Hindsight is never a second source of truth.**

- Sync is **one-way**: Obsidian → Hindsight. Your vault is canonical.
- Every chat answer **cites the note it came from**, so you fix things at the source.
- Chat conversations are **not stored by default** (toggle in settings).
- Edit a note and Hindsight reconverges on the next sync.

## What it does

- **Incremental vault sync** — each note becomes a Hindsight document. Edits upsert, deletes remove. A content hash skips unchanged notes.
- **Implicit scoping** — every note is auto-tagged on ingest with its **vault**, **folder** (and sub-folders), and **created/updated dates**. You never think about scope until you recall — then filter by any combination (vault + folder + date) via Hindsight's `tag_groups`. Multiple vaults share one bank and stay separable by their `vault:` tag.
- **Same data from UI and API** — the Obsidian chat panel and your external automations (n8n, Hermes, etc.) hit the same bank with the same tags, so they see exactly the same scoped view.
- **Grounded chat** — a side panel that answers questions over your notes via Hindsight `reflect`. Each answer lists the **notes retrieved** (click to open) and a **reasoning** disclosure showing what each step queried. Scope a question with the **vault / folder** dropdowns above the ask bar, start a fresh thread with **New chat**, and flip on **Debug logging** to see the exact `reflect` request + retrieved notes in the console.
- **Manual or automatic** — sync on every edit, or run _Sync vault now_ on demand.
- **Always-visible sync status** — a live indicator in the status bar (and the chat header) shows how many notes are synced, when the last sync ran, and any pending edits — e.g. `Hindsight ✓ 412 notes · 2m ago`. A refresh button beside it (which spins while syncing) triggers a sync on click, so nothing happens invisibly.

### Scoping

| Dimension            | Tag(s)                               | Example recall filter    |
| -------------------- | ------------------------------------ | ------------------------ |
| Vault                | `vault:<name>`                       | only the Work vault      |
| Folder (+ ancestors) | `folder:Work`, `folder:Work/Clients` | everything under `Work/` |
| Date                 | `created:2026-03`, `updated:2026-06` | notes updated this month |

Your own frontmatter `tags`/`aliases` are carried through too.

## Installation

> ✨ **Recommended:** [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) — sign up free, get an API key, and skip self-hosting.

While in beta, install via [BRAT](https://github.com/TfTHacker/obsidian42-brat): add the repo [`vectorize-io/hindsight-obsidian`](https://github.com/vectorize-io/hindsight-obsidian) — the dedicated plugin repo BRAT installs from (see [Distribution](#distribution--maintainers)) — then enable it under **Settings → Community plugins**.

**Self-hosting alternative:**

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-openai-key
hindsight-api
```

## Configuration

Open **Settings → Hindsight**:

| Setting                   | Default                              | Description                                                                                    |
| ------------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------- |
| API URL                   | `https://api.hindsight.vectorize.io` | Hindsight server (use `http://localhost:8888` for self-hosted)                                 |
| API key                   | —                                    | Hindsight Cloud API key                                                                        |
| Bank name                 | `obsidian`                           | Shared bank for all your vaults (separated by `vault:` tags)                                   |
| Include / exclude folders | —                                    | Limit which notes sync                                                                         |
| Sync on edit              | on                                   | Re-ingest notes automatically as you edit                                                      |
| Default chat depth        | low                                  | Reflect budget for chat answers                                                                |
| Remember conversations    | **off**                              | When on, chat turns are stored in Hindsight (creates memory outside your vault)                |
| Prefix document IDs       | on                                   | Vault-prefixes ids so shared-bank vaults don't collide; turn off only for a single-vault setup |

## Commands

- **Sync vault now** — full reconcile (ingest changed notes, prune deleted ones).
- **Ingest current note** — force-sync the active note.
- **Open chat** — open the grounded chat panel.

## How it works

```
note created / edited ──▶ retain(documentId = note path)     (upsert; replaces prior version)
note renamed         ──▶ deleteDocument(old) + retain(new)
note deleted         ──▶ deleteDocument(path)
"Sync vault now"     ──▶ reconcile: ingest drifted notes, prune orphans

chat turn            ──▶ reflect(question) over the whole bank
                         └─ answer + citations (→ source notes) + reasoning
```

A local index (note path → content hash + mtime) means only changed notes are re-ingested.

## Development

```bash
npm install
npm run lint     # tsc --noEmit
npm test         # vitest
npm run build    # esbuild → main.js
```

To try it in a real vault, copy `main.js`, `manifest.json`, and `styles.css` into
`<vault>/.obsidian/plugins/hindsight/` and enable the plugin.

## Distribution & maintainers

> **Edit only this monorepo.** The dedicated repo is generated automatically on release — never commit to it directly.

| Repo                                                                                    | Role                                                                                                                                                                    |
| --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hindsight-integrations/obsidian/` (this monorepo)                                      | **Source of truth.** All code, tests, and the npm package (`@vectorize-io/hindsight-obsidian`) live here.                                                               |
| [`vectorize-io/hindsight-obsidian`](https://github.com/vectorize-io/hindsight-obsidian) | **Generated distribution repo** for BRAT + the Obsidian community store. The release workflow mirrors this folder here (via `git subtree`) and cuts the GitHub Release. |

**Why two repos?** Obsidian's BRAT and community store install from a repo's **latest GitHub Release**. We can't use this monorepo's releases — they pollute the core product's release list and steal the "Latest" badge, and BRAT reads a repo's _latest_ release (the core app), not a tag. So distribution releases go to a dedicated repo.

**Releasing an update** — just cut the monorepo release as usual:

```bash
./scripts/release-integration.sh obsidian <version>
```

The `Release Integration` workflow then automatically (see `.github/workflows/release-integration.yml` → "Mirror Obsidian plugin"):

1. publishes the npm package `@vectorize-io/hindsight-obsidian`,
2. mirrors this folder to the **root** of `vectorize-io/hindsight-obsidian` with `git subtree push --prefix=hindsight-integrations/obsidian`, and
3. cuts a GitHub Release there tagged with the bare version (e.g. `0.1.0`, matching `manifest.json`) so BRAT / the community store pick it up.

No manual second-repo step. The mirror needs a repo/org secret **`OBSIDIAN_DIST_TOKEN`** — a token with `contents: write` on `vectorize-io/hindsight-obsidian`.
