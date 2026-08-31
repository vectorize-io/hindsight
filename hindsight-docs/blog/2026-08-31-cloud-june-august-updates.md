---
title: "What's New in Hindsight Cloud: June–August Updates"
authors: [benfrank241]
slug: "2026/08/31/hindsight-cloud-june-august-updates"
date: 2026-08-31T12:00
tags: [hindsight-cloud, release, knowledge-pages, mental-models, ui, recall, curation]
description: "Three months of Hindsight Cloud in one place: a client-managed knowledge base, reversible memory curation, scheduled mental model refreshes, per-bank recall controls, and a rebuilt visual design."
image: /img/blog/hindsight-cloud-june-august-updates.png
hide_table_of_contents: true
---

![What's New in Hindsight Cloud: June to August updates across the knowledge base, curation, mental models, and recall controls](/img/blog/hindsight-cloud-june-august-updates.png)

It's been three months since the last Cloud roundup, and rather more than three months' worth of work has landed. The last update went out when 0.6.2 was current, so this one covers thirteen releases: **v0.7.0 through v0.9.2**.

The headline is a **client-managed knowledge base** you can browse and edit in the console. Underneath it: memories you can now edit and un-delete, mental models that refresh on a schedule you set, per-bank control over the recall pipeline, and a rebuilt visual design across the whole control plane.

<!-- truncate -->

- [**A knowledge base you can edit**](#a-knowledge-base-you-can-edit) — folder tree, page editor, page-level search, and MCP access.
- [**Reversible curation**](#reversible-curation) — edit a memory, invalidate it, or put it back.
- [**Mental models on a schedule**](#mental-models-on-a-schedule) — cron triggers, dry runs, and honest staleness.
- [**A new look**](#a-new-look) — the Hindsight palette, and cards you can actually see in light mode.
- [**Recall pipeline controls**](#recall-pipeline-controls) — switch stages off per bank.
- [**Constellation, animated and fullscreen**](#constellation-animated-and-fullscreen) — zoom, pan, and inline memory detail.
- [**Documents and entities**](#documents-and-entities) — tag filters, entity timelines, in-flight badges.
- [**Moving data between banks**](#moving-data-between-banks) — export and import without re-running the LLM.

## A knowledge base you can edit

The biggest addition is the **knowledge base**: a set of client-managed pages that live alongside your memories and are organized in a folder hierarchy you control.

![The knowledge base tree in the Hindsight control plane](/img/blog/knowledge-pages-tree.png)

The mental shift is worth stating up front. A knowledge page **is** a mental model, with a simplified, document-shaped configuration wrapped around it. So a page isn't a static file you maintain by hand. It's a standing answer, rebuilt in the background from the memories underneath it, that your application reads instead of paying for synthesis on the request path.

What you get in the console:

- **A folder tree** for organizing pages, rather than a flat list.
- **A page editor** with the rendered result alongside it.
- **Page-level search** across the knowledge base.
- **Refresh triggers visible on the tree**, so you can see how each page stays current without opening it.
- **Per-scope staleness** — a page reports whether *its own* scope has new material, instead of inheriting a single bank-wide watermark that marked everything stale at once.

![Editing a knowledge page in the console](/img/blog/knowledge-pages-edit.png)

Two things extend it beyond the console. `hindsight fs` projects the knowledge base onto your filesystem, so pages can be mirrored locally and managed from the CLI. And knowledge-base CRUD is now exposed as **native MCP tools**, which means an agent connected to Cloud can create, read, update, and search its own knowledge base rather than only reading from it.

## Reversible curation

Memory systems tend to be append-only, which is fine until something wrong gets in. You can now **edit, invalidate, and restore** individual memory units.

- **Edit** a memory's text, context, dates, fact type, or linked entities.
- **Invalidate** it, which moves it into a separate archive rather than deleting it.
- **Revert** an invalidated memory back into the bank.
- **Edits are tracked** with an `edited_at` timestamp, so a hand-corrected memory is distinguishable from an extracted one.

This is available through the API, the SDKs, and MCP, via a `PATCH` on the memory. It isn't surfaced in the console yet, so for now it's something you drive from code rather than by clicking.

The important word is *reversible*. Invalidating is a two-way door, which makes it safe to prune aggressively when an agent has learned something wrong.

## Mental models on a schedule

Mental models gained a third way to refresh. Alongside the existing manual and refresh-on-new-memories paths, a model can now refresh **on a cron schedule**.

The console presents this as a single **Refresh trigger** choice with three options: *Manual*, *On new memories*, or *On a schedule*. The cron field appears only when you pick the third. Schedules are standard 5-field UTC cron expressions, validated as you type, and the two automatic modes are mutually exclusive by design.

The scheduling UI does the thing schedule UIs usually skip:

- **A live preview** of what the expression means in plain language, with the next several runs listed in both UTC and your local time.
- **"Next refresh"** displayed next to "last refreshed" in the model list, the dashboard, and the detail dialog.

Alongside scheduling:

- **Dry-run refresh** builds a model without persisting it, with an optional trace, so you can see what a refresh *would* produce before committing to it. This one is API-side rather than a button in the console.
- **A minimum interval** between automatic refreshes stops a busy bank from re-synthesizing the same model continuously.
- **Consistent staleness reporting** — per-model staleness is now computed and described the same way everywhere it appears, which it previously wasn't.
- **Refresh operations record what they did**, so a refresh that ran but changed nothing is distinguishable from one that rewrote the document.

## A new look

The control plane got a design system overhaul, and it's more than a repaint.

The stock component-library greys were replaced with the **Hindsight palette**, applied through tokens rather than per-component edits, so every view inherits the new look at once. The most visible fix: in light mode the card color was previously identical to the page background, which meant **cards were effectively invisible**. Pages now sit at `#F3F5F9` with cards at white, and dark mode uses a blue-shifted set with the page at `#080C17` and cards at `#0F1724`.

Two smaller details worth calling out, because they're the kind of thing that quietly makes an interface feel better without anyone identifying why:

- **Letter tracking went from `0.025em` to `0`.** Inter reads wrong with positive tracking at body sizes.
- **Body copy now clears WCAG AA in both modes.** The muted foreground was retuned to `#525866`, measuring 5.3:1 against the page and 7.1:1 against a card.

## Recall pipeline controls

A recall runs several retrieval stages and then reranks what they produce. Not every bank needs all of them, and stages you don't need cost latency.

Bank configuration now includes a **Recall Pipeline** section with per-bank switches for the **temporal**, **graph**, and **reranking** stages. All three default to on, so nothing changes unless you opt out. Turning reranking off falls back to the fused ordering rather than failing.

This composes into something useful: a bank set to chunk-based extraction with observations off and these stages disabled behaves like a **conventional vector store**. That configuration ships as a ready-made `plain-retrieval` bank template, so you don't have to assemble it yourself.

Two related changes make recall easier to reason about:

- **Per-stage scores.** A recall result used to carry a single opaque `score`. It now carries a `scores` object breaking out `final`, `reranker`, `semantic`, and `text`, so you can see *why* something ranked where it did. The matching `min_scores` request parameter filters per stage instead of applying one global floor.
- **`prefer_observations`.** Recalling observations alongside raw facts could return the same information twice, once raw and once folded into the observation built from it. This opt-in flag drops any raw fact a returned observation lists as a source. It dedupes by provenance rather than similarity, and runs before truncation so the freed slots backfill and you still get a full result set.

If you want the full picture of what each stage does before switching any of them off, we published a [deep dive on the reranking stage](/blog/2026/08/28/cross-encoder-reranking-agent-memory) and one on [why multiple retrieval arms exist](/blog/2026/08/24/knowledge-graphs-vs-vector-search-agent-memory).

## Constellation, animated and fullscreen

The Constellation view, introduced in the last update, picked up two upgrades.

![The Constellation view: memories and entities as an interactive graph](/img/blog/constellation-view.png)

- **Animation and inline detail** — the graph animates, and clicking a memory opens its details instead of navigating away.
- **Fullscreen** — the graph takes over the viewport, with scroll to zoom, drag to pan, and hover to explore entity connections.

## Documents and entities

A cluster of navigation improvements across the data views:

- **Document tag filtering**, with facet chips unified across views so filtering works the same way everywhere.
- **Filter memories by linked entity**, plus an **entity timeline** for seeing what a bank knows about one entity over time.
- **In-flight retain badges** on documents currently being updated by a running retain operation.

## Moving data between banks

Bank-to-bank transfers are enabled on Cloud, in both directions:

- **Export and import documents between banks without re-running the LLM.** The extracted facts move with the documents, so you're not paying to re-derive what you already have.
- **Async export** for large banks, so an export runs as a tracked background operation instead of blocking on a single long request.

## Also shipped

Smaller changes worth knowing about:

- **Operations report progress as they run.** A long consolidation used to look identical whether it was healthy or stuck. Operations now record a coarse stage/processed/total snapshot mid-run, so a slow job is distinguishable from a frozen one.
- **Dry-run fact extraction** is a read-only API endpoint that previews what a retain would extract from a piece of text without writing anything to the bank: candidate facts and token usage, no entity resolution, embeddings, or persistence.
- **Observation scopes** can be enumerated, filtered, and visualized, and there's a `shared` scope keyword for observations that span scopes.
- **Audit log and observations** are overridable per bank, rather than being a single deployment-wide setting.
- **The API version is displayed in the sidebar**, which makes it obvious what you're actually running.

## Try it

Hindsight Cloud is the fastest way to run Hindsight without operating it yourself: managed Postgres, OAuth for MCP clients, billing, multi-org, and now a knowledge base your agents can manage themselves.

[Sign up at ui.hindsight.vectorize.io/signup](https://ui.hindsight.vectorize.io/signup) — the free tier is enough to try retain and recall against a real bank without entering a card.
