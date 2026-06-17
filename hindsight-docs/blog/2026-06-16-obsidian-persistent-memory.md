---
title: "Chat With Your Obsidian Vault: Grounded Answers That Cite Your Notes"
authors: [benfrank241]
slug: "2026/06/16/obsidian-persistent-memory"
date: 2026-06-16T12:00
tags: [obsidian, memory, persistent-memory, hindsight, second-brain, knowledge-management, tutorial]
description: "Give your Obsidian vault an AI agent that actually knows your notes. The Hindsight plugin syncs your vault into a memory bank and adds a chat panel whose answers are grounded on your notes and cite the source, with your vault staying the single source of truth."
image: /img/blog/obsidian-persistent-memory.png
hide_table_of_contents: true
---

![Obsidian Persistent Memory with Hindsight](/img/blog/obsidian-persistent-memory.png)

[Obsidian](https://obsidian.md) is where a lot of people keep their thinking. Years of notes, meeting records, research, half-finished ideas. The problem is that a vault that big stops being searchable in any useful way. You know you wrote something about a decision six months ago; you just can't find the note. Full-text search finds the word, not the idea, and it certainly can't reason across five notes to answer a question.

This post is a walkthrough of the new Hindsight plugin for Obsidian. It syncs your vault into a Hindsight memory bank and adds a chat panel that answers questions over your notes, and every answer **cites the note it came from**. Your vault stays the single source of truth; Hindsight is the reasoning layer on top of it.

## TL;DR

<!-- truncate -->

- Obsidian's built-in and vector-search plugins find related notes. They don't reason across your vault or keep a running synthesis as it grows.
- The Hindsight plugin **syncs your vault into a memory bank** (one note = one document) and adds a **grounded chat panel** backed by Hindsight's `reflect`.
- **Every answer cites its source notes.** Click a citation to open the note. A reasoning disclosure shows what each step queried.
- **Your vault is canonical.** Sync is one-way (Obsidian to Hindsight), and chat conversations are **not stored by default**. Hindsight never becomes a second source of truth.
- **Implicit scoping:** every note is auto-tagged with its vault, folder, and dates, so you can filter a question to "just the Work vault" or "notes updated this month" without any setup.
- Hindsight Cloud means no infrastructure. [Sign up free.](https://ui.hindsight.vectorize.io/signup)
- **Available now** in the [Obsidian community plugin store](https://community.obsidian.md/plugins/hindsight): one-click install.

:::tip Now in the Obsidian community store
The Hindsight plugin is **live in the [Obsidian community plugin store](https://community.obsidian.md/plugins/hindsight)** (`v0.1.2` at the time of writing). That means a one-click install from inside Obsidian, no manual steps, and it puts Hindsight one click away from Obsidian's millions of users. See [Installing](#installing) below.
:::

## The Problem: A Big Vault Is Hard to Reason Over

A small vault is easy. A few dozen notes, and search plus backlinks get you anywhere. The trouble starts when the vault gets large enough to be genuinely valuable, which is exactly when it gets hard to use.

Vector-search plugins help: they surface notes semantically related to what you typed, which beats keyword matching. But "show me related notes" is not the same as "answer my question." If the answer lives across three notes, a related-notes list hands you the three notes and leaves the synthesis to you. And nothing in that flow remembers what you asked or builds a running picture of your vault over time.

Hindsight adds the layer that's missing:

- **Recall** is faster and more accurate than text search.
- **Reflect** reasons over your whole vault to answer a question, and tells you which notes it used.
- **Mental models** (on the roadmap) will keep living summaries of your vault that refresh as you write.

## Source of Truth Stays in Obsidian

This is the design rule that shapes everything else, so it's worth stating plainly before the mechanics: **Hindsight never becomes a second source of truth.**

- **Sync is one-way.** Obsidian to Hindsight, never the reverse. Your vault is canonical; Hindsight is a derived index.
- **Every answer cites the note it came from.** When something is wrong or stale, you fix it in the note, at the source, not in some opaque memory store.
- **Chat conversations are not stored by default.** Asking a question doesn't create new memory floating outside your vault. (There's a toggle if you want it; it's off out of the box.)
- **Edit a note and Hindsight reconverges** on the next sync. The index always chases the vault, not the other way around.

If you've been burned by tools that quietly become a parallel copy of your data that drifts out of sync, this is the answer to that worry.

## How It Works

The plugin watches your vault and maps file events to Hindsight operations:

```
note created / edited ──▶ retain(documentId = note path)   (upsert; replaces prior version)
note renamed          ──▶ deleteDocument(old) + retain(new)
note deleted          ──▶ deleteDocument(path)
"Sync vault now"      ──▶ reconcile: ingest drifted notes, prune orphans

chat turn             ──▶ reflect(question) over the whole bank
                          └─ answer + citations (source notes) + reasoning
```

Each note becomes one Hindsight document, keyed by its path. A local index (note path to content hash plus mtime) means only notes that actually changed get re-ingested, so syncing a 400-note vault after editing one file re-sends one file, not four hundred.

There's always a visible sync indicator in the status bar (and the chat header) so nothing happens invisibly: something like `Hindsight ✓ 412 notes · 2m ago`, with a refresh button beside it that spins while syncing and triggers a sync on click.

## Grounded Chat With Citations

The chat panel is the part you'll actually use day to day. You open it as a side panel, ask a question in natural language, and Hindsight runs a `reflect` over your bank to produce an answer.

What makes it trustworthy is what comes attached to the answer:

- **Citations.** Each answer lists the notes it retrieved. Click one to open the source note in your vault. If the answer says you decided to move off Postgres in March, you can jump straight to the note that says so and confirm it.
- **Reasoning disclosure.** A collapsible view shows what each step of the reflect actually queried, so the answer isn't a black box.
- **Scope dropdowns.** Above the ask bar, vault and folder dropdowns let you narrow a question to part of your vault before you ask.
- **New chat** starts a fresh thread, and **Debug logging** (a setting) prints the exact `reflect` request and retrieved notes to the console if you want to see the plumbing.

Because the answer points back at real notes, the failure mode of "the AI confidently made something up" turns into "click the citation and check." The vault is right there.

## Implicit Scoping: Filter Without Setup

Every note is auto-tagged on ingest. You don't tag anything by hand; the plugin derives the tags from where the note lives:

| Dimension | Tag(s) | Example filter |
| --- | --- | --- |
| Vault | `vault:<name>` | only the Work vault |
| Folder (and ancestors) | `folder:Work`, `folder:Work/Clients` | everything under `Work/` |
| Date | `created:2026-03`, `updated:2026-06` | notes updated this month |

Your own frontmatter `tags` and `aliases` carry through too. The payoff: you never think about scope until the moment you want it, and then you filter by any combination (vault plus folder plus date) through Hindsight's `tag_groups`. A note under `Work/Clients/Acme` is reachable by a `folder:Work` filter, a `folder:Work/Clients` filter, or the vault filter, because folders tag their ancestors too.

Multiple vaults can share one bank and stay cleanly separable by their `vault:` tag, which is why the default bank name is just `obsidian` and `Prefix document IDs` is on by default (so same-named notes in different vaults don't collide).

## One Bank, UI and API

A detail that's easy to skip past but matters: the chat panel and any external automation hit the **same bank with the same tags**. The vault you sync from Obsidian is readable from an n8n workflow, a [Hermes](https://hindsight.vectorize.io/blog/2026/03/17/hermes-agent-memory) agent, a direct API call, or any other Hindsight integration, and they all see the same scoped view.

So your notes stop being trapped in one app. A nightly automation can reflect over `folder:Meetings updated:2026-06` and post a summary to Slack, using exactly the notes you'd see if you asked the same thing in the chat panel.

## Installing

The plugin is in the **Obsidian community store**, so installing it is one click from inside Obsidian:

1. Open **Settings → Community plugins → Browse**.
2. Search for **Hindsight**, click **Install**, then **Enable**.

> Want bleeding-edge beta builds before they hit the store? You can also track them via [BRAT](https://github.com/TfTHacker/obsidian42-brat) by adding the repository [`vectorize-io/hindsight-obsidian`](https://github.com/vectorize-io/hindsight-obsidian).

Then point it at Hindsight. The recommended path is **Hindsight Cloud**: [sign up free](https://ui.hindsight.vectorize.io/signup), grab an API key, and paste it into **Settings → Hindsight**. No server to run; sync and reflect happen against Cloud.

Prefer to self-host? Run Hindsight locally and set the API URL to your instance:

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-openai-key
hindsight-api  # http://localhost:8888
```

Then set **API URL** to `http://localhost:8888` in the plugin settings.

## Key Settings

Everything lives under **Settings → Hindsight**:

| Setting | Default | What it does |
| --- | --- | --- |
| API URL | `https://api.hindsight.vectorize.io` | Hindsight server; use `http://localhost:8888` for self-hosted. |
| API key | (none) | Your Hindsight Cloud key. |
| Bank name | `obsidian` | Shared bank for all your vaults (separated by `vault:` tags). |
| Include / exclude folders | (none) | Limit which notes sync. |
| Sync on edit | on | Re-ingest notes automatically as you edit. |
| Default chat depth | low | Reflect budget for chat answers; raise it for harder questions. |
| Remember conversations | **off** | When on, chat turns are stored in Hindsight (memory outside your vault). |
| Prefix document IDs | on | Vault-prefixes IDs so shared-bank vaults don't collide. Turn off only for a single-vault setup. |

And three commands from the command palette: **Sync vault now** (full reconcile), **Ingest current note** (force-sync the active note), and **Open chat**.

## Tradeoffs

**Sync depends on a content hash, not magic.** A note is re-ingested when its content changes. If you want a guaranteed fresh state across the whole vault (say after a big external edit or a git pull into the vault folder), run **Sync vault now** to reconcile rather than relying on per-edit sync.

**Reflect quality scales with `Default chat depth`.** The default is `low`, which is fast and fine for most lookups. For questions that need to pull across many notes, raise the budget; it searches more thoroughly at higher cost and latency.

**Chat memory is opt-in by design.** `Remember conversations` is off so that asking questions doesn't quietly accumulate state outside your vault. If you want the agent to remember the thread of a long working session, turn it on, but know that you're then creating memory that isn't one of your notes.

## Recap

| | Obsidian default | With Hindsight |
| --- | --- | --- |
| Find a note | Full-text / backlinks | Semantic recall, faster and more accurate |
| Answer a question across notes | You synthesize manually | Reflect answers and cites the notes |
| Trust the answer | n/a | Every answer links its source notes |
| Source of truth | Your vault | Still your vault (one-way sync) |
| Scope a query | Manual search operators | Auto tags: vault / folder / date |
| Reach from automations | Per-app | Same bank from n8n, Hermes, API |

## Next Steps

- **Hindsight Cloud:** [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io/signup)
- **Integration docs:** [Obsidian + Hindsight](/sdks/integrations/obsidian)
- **Install (Obsidian store):** [community.obsidian.md/plugins/hindsight](https://community.obsidian.md/plugins/hindsight)
- **Source:** [vectorize-io/hindsight/hindsight-integrations/obsidian](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/obsidian)
- **One memory for every tool:** [One Memory for Every AI Tool](https://hindsight.vectorize.io/blog/2026/04/07/one-memory-for-every-ai-tool)
