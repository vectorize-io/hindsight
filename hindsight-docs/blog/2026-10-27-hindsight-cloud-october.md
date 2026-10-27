---
title: "What's New in Hindsight Cloud: The API Is a Promise Now"
authors: [benfrank241]
slug: "2026/10/27/hindsight-cloud-october"
date: 2026-10-27T15:00
tags: [hindsight-cloud, release, memlake, mcp, reflect]
description: "Hindsight hit 1.0, most of Cloud moved onto a new storage engine, ChatGPT can connect as a client, and a bank's documents now travel in and out as a zip."
image: /img/blog/hindsight-cloud-october.png
hide_table_of_contents: true
---

![What's new in Hindsight Cloud: Hindsight 1.0, Memlake under the hood, the ChatGPT connector, and document import](/img/blog/hindsight-cloud-october.png)

The most important thing that happened to Hindsight Cloud in the last five weeks is something you cannot see in the console, and that is the point.

Hindsight reached **1.0**. After a year in production the architecture and the API are stable, and from here the API is a promise: what you build on Hindsight Cloud keeps working as Hindsight grows. Nothing changed in your code, your SDKs or your MCP setup. That is the whole announcement, and it is the one that matters most if you are shipping something on top of us.

Underneath it, a lot moved. Here is the rest.

<!-- truncate -->

## TL;DR

- **Hindsight 1.0.** The API is stable and stays that way.
- **Memlake runs most of Cloud now.** A new storage engine, nothing to turn on.
- **ChatGPT can connect to your banks** as a custom connector over MCP, and members can approve clients now, not just admins.
- **A bank's documents travel both ways.** Export to a zip, import it back into the same bank or a different one, billed below the retain rate.
- **Reflect shows its work**, and can rank a handbook above the chatter.
- **A bank answers to more than one name.**
- **Fresh this week:** every document carries a change history, and observations can group themselves by tags your client never sends.

## Memlake runs most of Cloud now

[Hindsight Memlake](https://hindsight.vectorize.io/blog/2026/10/13/memlake) is a new storage engine for agent memory, written in Rust, that keeps memory in object storage. It serves 100 million memories at under 100 ms p99, and every write is durable before it is acknowledged.

Most of Hindsight Cloud already runs on it. Vectorize operates it, so there is nothing to enable, no migration to schedule and no setting to find. If you want the detail on how it works, the launch post has it, and [the Memlake docs](https://docs.hindsight.vectorize.io/enterprise/memlake) cover running it in your own cloud or data center with Hindsight Enterprise.

This is the kind of change that only shows up as an absence: fewer slow queries on large banks, and headroom you do not have to think about.

## ChatGPT can connect to your banks

ChatGPT's custom connectors can now link to Hindsight Cloud over MCP. Add Hindsight as a connector, approve it once, and ChatGPT can retain and recall memories in your banks directly.

Alongside it, **who** can approve a client changed. Organization owners can now let plain **members** approve MCP connections from clients like Claude, ChatGPT and Cursor, where before that was limited to owners and admins. Turn it on under **Settings** then **Connected AI clients (MCP)**.

Two smaller fixes in the same area are worth knowing because they close real gaps. MCP connections now follow access: when someone leaves the organization, or loses the role that let them connect, their connections are revoked rather than quietly continuing to work. And signing in with SSO partway through an MCP connection now returns you to the approval screen instead of dumping you on the dashboard.

## A bank's documents travel both ways

Export has been able to produce a zip of a bank's documents for a while. Now you can put one back.

A `.zip` archive made by **Export**, from the same bank or a different one, can be imported into a bank's **Documents** view. The documents and the memories extracted from them are added without extracting facts again: the facts are re-embedded and re-linked in the bank you import into. Use it to restore a backup, or to copy documents from one bank to another.

What you will see:

- An **Import from zip** action next to **Export** in a bank's **Documents** view, for organization owners and admins
- A choice for a document that already exists: skip it, replace it, or keep both
- Your organization's import rate, shown before you confirm

**It is cheaper than retaining the same content again.** Imports bill per million tokens of document content at the import rate, which sits below the standard retain rate because no fact extraction runs, and is never higher than your organization's retain rate. It appears under **Retain** in the Billing page's **Pricing** card.

## Reflect shows its work

Two changes make a reflect answer easier to trust.

The **Based On** panel now shows the observations an answer cited, with their full text, plus the images and files attached to its evidence. You can see exactly what the answer drew on rather than inferring it.

And reflect now reads each document's **own metadata**. If you retain documents tagged with where they came from, say `{"doc_type": "kb"}` for published articles and `{"doc_type": "ticket"}` for support threads, your bank's mission can tell reflect how to weigh them: prefer the handbook over the chatter, unless something newer and more authoritative overrides it. Previously that distinction was invisible to reflect even when it was sitting right there in the document.

## A bank answers to more than one name

Add an **alias** to a bank and requests to either id reach the same memories. Promote an alias and it becomes the name the bank is **shown as** everywhere, with the original id still doing the work underneath.

This makes renaming painless. Point new clients at the new name while existing ones keep working on the old one, with no downtime and nothing to move. API keys scoped to a bank work with that bank's aliases too. Owners and admins manage them on the bank's **Configuration** tab.

## Finding things in a large bank

A run of smaller additions, all aimed at reading one slice of a bank instead of all of it:

- A **time filter** on the Documents list, with presets or a custom range, plus a **tag filter**
- **Tag filters on the knowledge base and entities.** The knowledge base tree and search, and the entity list, detail and graph, now take the same `tags` and `tags_match` filter recall uses, so you can show only the pages and entities for one user, team or topic
- Document **metadata** shown on each memory's details
- An advanced **search dialog** for knowledge base pages
- Causal links in bank stats, and pause, motion and SVG export on the constellation graph

The memories and documents APIs also filter and sort by time on the server now, so a client no longer has to pull a page and sort it itself.

## Fresh this week

These shipped in the engine in the last few days and are live on Cloud ahead of the docs catching up.

**Every document has a change history.** Observations in Hindsight are rebuilt, not versioned: when a fact is edited, appended to, re-retained or invalidated, the observations built on it are deleted and rebuilt. Until now that left no trace, so an observation you did not recognise was the end of the investigation. Now each document has a **History** tab in the console, and `GET /v1/default/banks/{bank_id}/documents/{document_id}/history`, listing every retain, append, reprocess, fact edit, invalidation, restore, delete and retag, and every consolidation run with the observations it created, updated and removed.

Three limits are worth knowing. It is **reconstructed from your bank's async operations**, so it reaches back only as far as those operations are kept, and a change that ran without one is not in it: a **synchronous retain**, sent with `async=false`, leaves no trace, and neither does a fact edit, delete or retag made while auto-consolidation is off. It **starts when the feature arrives**, so earlier changes are not in it. And it is ids and counts rather than diffs, with each id list capped at 100.

**Observations can group themselves.** A tag in `observation_scopes` can now be a glob, so `[["topic:*"]]` groups a memory by whatever `topic:` tags it happens to carry. That is the only way to do this when the tags are added during extraction and your client could not have known them when it sent the request. Banks also get `default_observation_scopes`, applied to every retain that sends none, so memories about the same subject consolidate together no matter which agent produced them, with no client change.

One detail in the matching is deliberate and worth repeating: a scope whose globs match nothing is **dropped**, not kept as an empty scope, because the empty scope is the untagged bank-wide one. A memory your pattern missed must not silently become visible to everyone.

**Three smaller ones.** Document search now covers a document's original text as well as its id, and tells you which of the two matched. Knowledge pages and mental models download as PDF or Markdown. And live API requests now take the next free model slot ahead of background work like async retain and consolidation, so a bank in the middle of consolidating no longer makes interactive recall feel slow.

## Two behaviour changes to know about

**A retain that returns no facts list is now an error.** When the model's reply came back without a top-level facts list, retain used to treat it as zero facts and report success. It now reports the malformed response as an error. If you have a dashboard counting successful retains, some of what it counted was never a retain, and you may see those surface as failures for the first time.

**Bank imports validate ids.** An import now rejects an invalid bank id, and ids using the reserved underscore prefix. Worth checking any scripted import before your next run.

## Learn more

- [Hindsight Memlake](https://hindsight.vectorize.io/blog/2026/10/13/memlake) on the storage engine behind most of Cloud
- [What's New in Hindsight Cloud](https://docs.hindsight.vectorize.io/cloud/whats-new) for the running list, with every item dated
- [MCP setup](https://docs.hindsight.vectorize.io/cloud/mcp) for connecting ChatGPT, Claude and Cursor
- [Billing](https://docs.hindsight.vectorize.io/cloud/billing) for the import rate and the rest of the pricing card
- [What's New in Hindsight Cloud: Memory That Reads Screenshots](https://hindsight.vectorize.io/blog/2026/09/21/hindsight-cloud-0-10-0) for the previous roundup
