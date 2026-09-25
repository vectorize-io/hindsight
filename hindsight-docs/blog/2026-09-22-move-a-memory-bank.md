---
title: "A Bank Is Now Something You Can Pick Up"
authors: [benfrank241]
slug: "2026/09/22/move-a-memory-bank"
date: 2026-09-22T15:00
tags: [hindsight, agent-memory, banks, transfer, migration, release]
description: "Clone a memory bank in one call, export and import it with an explicit scope, or rename it in place. What travels, what gets rebuilt, and where it will bite you."
image: /img/blog/move-a-memory-bank.png
hide_table_of_contents: true
---

![Moving a memory bank: what travels, what gets rebuilt, and what stays behind](/img/blog/move-a-memory-bank.png)

Picking a bank id used to be a one-way door. You chose `support` when you should have chosen `support-prod`, or you scoped one bank per customer when you wanted one per workspace, and the only fix was re-ingesting everything through a pipeline that costs real LLM money.

Hindsight 0.10.1 makes a bank portable. You can copy one, move one between instances, or rename one in place, and none of it re-extracts a single fact.

<!-- truncate -->

## TL;DR

- **Clone** a bank into a new id in one API call. It runs as a background operation, and no LLM is involved.
- **Export and import** a whole bank as a ZIP archive with an explicit scope: data, bank config, history.
- **Rename** a bank in place with `hindsight-admin`, which also moves its stored files to the new prefix.
- **Nothing derived travels.** Embeddings, database ids and the entity graph are rebuilt on arrival, so a moved bank is native to wherever it lands.
- **Three things will bite you**: a clone inherits the source's webhooks by default, a restore refuses to overwrite an existing bank, and rename won't touch a store-owned bank.

## Three ways to move a bank

**Clone** is the one-call option. `POST /v1/default/banks/{bank_id}/clone?target_bank_id=...` returns `202` with an operation id you poll. Under the hood it's an export and an import run back to back in one process, and the archive never touches disk: as the code puts it, stashing it in file storage "would only add a round trip and a blob to clean up." The read runs in a single transaction, so you get a point-in-time copy while the source stays writable. Without that, as the comment notes, the clone would be "a smear of whatever was being written meanwhile."

**Export and import** is the cross-instance path. `POST .../transfer/export` gives you an operation whose result carries a download URL; `POST .../transfer/import` takes the ZIP back as multipart. This is the supported way to change something you can't change in place on a populated bank, like your embedding model, your vector extension, or your text-search backend.

**Rename** is an operator tool, not an API. `hindsight-admin rename-bank --from old --to new` moves every row with that `bank_id` in a single transaction, then re-keys the bank's stored files to the new prefix and rebuilds its vector indexes concurrently. There's a `--dry-run`.

## What travels

The scope is three booleans, not an enum, and they mean exactly this:

| Flag | Default | Carries |
|---|---|---|
| `include_data` | on | Documents, chunks, facts, consolidated observations, attachments and their bytes, the curation archive, the operations log, **mental models and the knowledge-page tree** |
| `include_bank_config` | on | The bank row (config overrides, name, mission, disposition), directives, **webhooks** |
| `include_history` | off | Audit log and LLM request records |

The placement that surprises people is mental models and knowledge pages landing under *data* rather than config. They look like settings, but the code explains why they aren't:

> Mental models and knowledge pages sit under `data` rather than `bank_config` because they are a reading of the bank's facts, not a setting: a mental model's `based_on` evidence cites memory units by id, so carrying it without them would restore a synthesis whose grounding resolves to nothing.

This was actually reversed during development — the first version filed them under config — which tells you the distinction is easy to get wrong. The practical consequence: `include_data=false` with `include_bank_config=true` gives you a bank with its settings, directives and webhooks, and no mental models at all.

## Nothing derived travels

This is the part that makes the whole thing work.

Embeddings are never carried. Neither are database ids, the entity table, the co-occurrence graph, or memory links. What the archive holds is the source material — documents, their chunks, the extracted facts as text, entities named canonically rather than by id — and the importer replays it: re-embeds each fact with the **target** bank's model, re-resolves entities against the target, and rebuilds the graph.

No LLM is called. The importer's own description of itself is that it runs "exactly the steps retain runs after LLM extraction." So moving a bank costs embedding compute, not tokens, and the copy arrives native to its destination rather than carrying vectors from a model the target may not even run.

That's also why export and import is the documented answer to "I want to change my embedding model." You can't change it in place on a bank that's already full. You can move the bank through a new one.

There's one consequence worth knowing. Documents keep their ids; facts, observations and entities get fresh ones. Mental models and knowledge pages keep theirs, so the page tree still resolves. If you've stored a Hindsight fact id in your own database, it won't survive the trip.

## Where it will bite you

Four things I'd want to know before running any of this against something I cared about.

**A clone inherits the source's webhooks.** `include_bank_config` defaults to on, and webhooks live in bank config, so a clone made with defaults will call the source's webhook endpoints. The API docs say so directly: pass `include_bank_config=false`, or delete them on the copy, when they point at a per-bank consumer. Cloning production to get a debugging sandbox and then watching the sandbox fire production's webhooks is a bad afternoon.

**A restore refuses an existing target.** Both clone and restore write into a bank that must not already exist; the error tells you to delete it first or pick another id. That means promoting dev to prod on a schedule isn't a loop you can run — it's delete-then-restore, or a fresh bank id each time and repointing clients. Merge mode exists, but it works at document granularity with a conflict policy, and it doesn't dedupe facts, so re-importing the same archive can leave you with overlapping observations over the same facts.

**Rename is narrower than it looks.** It's PostgreSQL only. It refuses while the bank has pending or processing operations. It refuses outright on a store-owned bank, where memories live outside SQL — for those, the only way to change an id is clone to the new one and delete the old. And you have to stop clients first: one still calling the old id gets a 404, or, if it creates banks on demand, quietly starts a brand new empty bank under the old name.

**Narrowing the scope fails silently.** On import, the scope you ask for is intersected with what the archive actually contains. Asking for something the archive doesn't carry isn't an error — it's a no-op. Check the manifest rather than assuming your flags did anything.

## If you upgrade with attachments in a store-owned bank

0.10.1 ships one migration: attachments now belong to their document, replacing the separate `document_attachments` table. On a normal deployment it's cheap, because it doesn't move any bytes — a migrated row keeps the storage key it already had.

But a store-owned bank writes attachment rows without the document-attachment rows the backfill reads, so those attachments have no discoverable owner, and **the migration deletes them**. The blobs stay in file storage until the bank is deleted. If you run a store-owned deployment with attachments, know that before you upgrade, not after.

## Templates are a different feature

Hindsight has had [bank templates](https://hindsight.vectorize.io/blog/2026/04/10/templates-hub-hindsight-050) since 0.5.0, and it's easy to assume this is the same idea. It isn't.

A template is a hand-editable JSON manifest carrying config overrides, directives, and mental model *definitions* — the question a model answers, not its content. Importing one into a fresh bank regenerates those models by running reflect, which costs LLM calls. It upserts, so it applies to a bank that already exists.

A transfer is an opaque archive carrying the bank itself, content and all, with no LLM involved, into a bank that must not exist yet.

A template moves the recipe. A transfer moves the bank.

## Where this is available

Clone, export and import are on the HTTP API and in the Python, TypeScript and Go clients. Clone is in the control plane under Actions. The endpoints are live on Hindsight Cloud.

Rename is `hindsight-admin` only: no API, no MCP tool, no SDK method. The admin CLI also has `export-bank` and `import-bank`, though only the HTTP surface gives you the three-way scope.

One thing to note if you're scripting against it: there's no archive size cap and no task timeout, and a clone holds the payload and the archive in process memory. The operation's result reports the archive's byte size, which is the only signal you get about how large the thing was.

## Learn more

- [One Bank or Many? A Field Guide to Structuring Agent Memory](https://hindsight.vectorize.io/blog/2026/07/16/bank-strategy-agent-memory) on deciding what a bank should represent, which is now a reversible decision
- [Per-User Memory for AI Products](https://hindsight.vectorize.io/blog/2026/08/04/per-user-multi-tenant-agent-memory) on tenant isolation, and why a seed bank cloned with its config may not be what you want
- [How to Move Your Agent's Memory Off a Vector Database](https://hindsight.vectorize.io/blog/2026/07/28/migrate-agent-memory-off-vector-database) for moving memory in, rather than around
- [What's new in Hindsight 0.10.1](https://hindsight.vectorize.io/blog/2026/09/21/version-0-10-1) for the rest of the release, including the TypeSafe Jev reranker
