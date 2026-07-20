---
title: "Guide: Add Obsidian Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, obsidian, knowledge, memory]
description: "Add Obsidian memory with Hindsight by syncing your vault into a bank and chatting with an agent grounded on your notes, with every answer citing the note it came from."
image: /img/guides/guide-obsidian-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add Obsidian Memory with Hindsight](/img/guides/guide-obsidian-memory-with-hindsight.svg)

If you want **Obsidian memory with Hindsight**, the setup is the Hindsight plugin for Obsidian. It syncs your vault into a Hindsight bank and adds a chat panel whose answers are grounded on your notes — and cite them. That gives you an agent that actually knows your vault, instead of a plain text search that can't reason across notes or keep a running synthesis as your vault grows.

This is a good fit for Obsidian because your knowledge already lives there. Sync is one-way, Obsidian → Hindsight, so your vault stays canonical. Every answer cites the note it came from so you can fix things at the source, and chat conversations are not stored by default. Edit a note, and Hindsight reconverges on the next sync — Hindsight never becomes a second source of truth.

This guide walks through installing the plugin, pointing it at your Hindsight backend, understanding how vault sync and scoping work, and a quick verification flow so you can confirm chat is grounded on your notes. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. Install the plugin via [BRAT](https://github.com/TfTHacker/obsidian42-brat) from the repo `vectorize-io/hindsight-obsidian`, then enable it under **Settings → Community plugins**.
> 2. Open **Settings → Hindsight** and set the **API URL** and **API key** (Cloud), or point the API URL at your self-hosted server.
> 3. Run **Sync vault now** to ingest your notes as Hindsight documents.
> 4. Open the chat panel and ask a question — answers come back grounded, with citations.
> 5. Verify that an answer cites the note the fact actually lives in.

## Prerequisites

Before you start, make sure you have:

- Obsidian installed with an existing vault
- [BRAT](https://github.com/TfTHacker/obsidian42-brat) available to install beta plugins
- A reachable Hindsight backend, [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) or a self-hosted server

## Step 1: Install the plugin

While the plugin is in beta, install it via [BRAT](https://github.com/TfTHacker/obsidian42-brat). Add the repository [`vectorize-io/hindsight-obsidian`](https://github.com/vectorize-io/hindsight-obsidian) — the dedicated plugin repo BRAT installs from — then enable it under **Settings → Community plugins**.

If you prefer to self-host Hindsight rather than use Cloud, run a local server:

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-openai-key
hindsight-api
```

## Step 2: Point the plugin at Hindsight

Open **Settings → Hindsight** and configure the connection:

| Setting                   | Default                              | Description                                                                     |
| ------------------------- | ------------------------------------ | ------------------------------------------------------------------------------- |
| API URL                   | `https://api.hindsight.vectorize.io` | Hindsight server (use `http://localhost:8888` for self-hosted)                  |
| API key                   | —                                    | Hindsight Cloud API key                                                         |
| Bank name                 | `obsidian`                           | Shared bank for all vaults (separated by `vault:` tags)                         |
| Include / exclude folders | —                                    | Limit which notes sync                                                          |
| Sync on edit              | on                                   | Re-ingest notes automatically as you edit                                       |
| Default chat depth        | low                                  | Reflect budget for chat answers                                                 |
| Remember conversations    | **off**                              | When on, chat turns are stored in Hindsight (creates memory outside your vault) |

For Hindsight Cloud, leave the API URL at its default and paste your API key. For a self-hosted server, set the API URL to `http://localhost:8888` (or wherever your server runs).

## Step 3: Sync your vault

Run the **Sync vault now** command to do a full reconcile — it ingests changed notes and prunes deleted ones. Two other commands are available:

- **Sync vault now** — full reconcile (ingest changed notes, prune deleted ones).
- **Ingest current note** — force-sync the active note.
- **Open chat** — open the grounded chat panel.

With **Sync on edit** left on, notes are re-ingested automatically as you edit, so you rarely need to sync by hand.

## How the plugin uses memory

The plugin works at the two points Obsidian exposes: your notes and a chat panel.

- **Sync (vault → Hindsight):** each note becomes a Hindsight document. Edits upsert, deletes remove, and a content hash means unchanged notes are skipped. A local index of note path to content hash means only changed notes are re-ingested.
- **Grounded chat (over the bank):** the side panel answers questions over your notes via Reflect, with collapsible citations you can click to open the source note, and a reasoning disclosure showing what each step queried.

Because sync is one-way, your vault stays the source of truth. Every answer cites its source note so you fix things at the source, and chat conversations are not stored by default — edit a note and Hindsight reconverges on the next sync.

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Implicit scoping

Every note is auto-tagged on ingest with its **vault**, **folder** (and sub-folders), and **created/updated dates**. You never think about scope until you recall — then filter by any combination via Hindsight's `tag_groups`. Multiple vaults share one bank and stay separable by their `vault:` tag.

Recall and reflect, from the UI or an API call, can filter by any combination of:

| Dimension            | Tag(s)                               |
| -------------------- | ------------------------------------ |
| Vault                | `vault:<name>`                       |
| Folder (+ ancestors) | `folder:Work`, `folder:Work/Clients` |
| Date                 | `created:2026-03`, `updated:2026-06` |

Your own frontmatter `tags` and `aliases` are carried through too. Because the chat panel and your external automations hit the same bank with the same tags, they see the same scoped view.

## Verify that memory is working

A good test sequence is:

1. run **Sync vault now** so your notes are ingested
2. open the chat panel
3. ask a question whose answer lives in a specific note
4. check that the answer comes back grounded and lists the note it used
5. click the citation to confirm it opens the correct source note

For example:

- ask about a decision or convention recorded in one of your notes
- confirm the cited note is the one that actually contains it

If the answer surfaces the fact and cites the right note, the setup is working.

## Common mistakes

### Expecting a two-way sync

Sync is one-way, Obsidian → Hindsight. Your vault is canonical, so fix things by editing the note, not in Hindsight.

### Not syncing before you chat

The chat panel answers over what has been ingested. Run **Sync vault now** (or leave **Sync on edit** on) so the bank reflects your current notes.

### Expecting stored conversations by default

**Remember conversations** is off by default. Turn it on only if you want chat turns stored in Hindsight — that creates memory outside your vault.

### Assuming vaults are isolated banks

Multiple vaults share one bank by default and stay separable by their `vault:` tag. Filter by `vault:<name>` when you want a single vault's view.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — install with `pip install hindsight-all`, run `hindsight-api`, and point the plugin's API URL at it (for example `http://localhost:8888`).

### Does Hindsight become a second copy of my notes?

No. Sync is one-way and your vault stays the source of truth. Every answer cites its note so you fix things at the source, and Hindsight reconverges on the next sync.

### How is my vault scoped?

Every note is auto-tagged with its vault, folder (and ancestors), and created/updated dates, so you can filter recall and reflect by any combination.

### Are my chat conversations stored?

Not by default. **Remember conversations** is off unless you turn it on.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Obsidian integration docs](https://hindsight.vectorize.io/docs/integrations/obsidian)
