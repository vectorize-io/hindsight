---
title: "The Clutter You Inherit"
authors: [benfrank241]
slug: "2026/09/23/clutter-you-inherit"
date: 2026-09-23T15:00
tags: [hindsight, agent-memory, migration, consolidation, deduplication, observations]
description: "Importing a memory history carries its duplicates with it, and an import quietly tells the target bank the cleanup is already done. What Hindsight deduplicates, what it never will, and how to clean up after the fact."
image: /img/blog/clutter-you-inherit.png
hide_table_of_contents: true
---

![The clutter you inherit: what an import carries, what consolidation collapses, and what nothing ever merges](/img/blog/clutter-you-inherit.png)

Someone wrote up their move onto Hindsight recently, and one line stuck with me:

> The move preserved the history. It also preserved the clutter.

Going through what landed, [they found](https://praveenks.com/notes/from-honcho-to-hindsight/) "duplicate conclusions and leftover test records."

That is not a migration bug. It is the correct behaviour of an append-only memory system. But it is worth knowing exactly which layer is responsible for cleaning it up, because an import has a habit of switching that layer off.

<!-- truncate -->

## TL;DR

- **Nothing in Hindsight ever merges two duplicate raw facts.** Not at write time, not in a background sweep, not on import. Two documents saying the same thing in different words are two rows, permanently.
- **Consolidation is the only thing that collapses anything**, and it works one layer up, by folding related facts into a single observation.
- **An import marks imported facts as already consolidated**, so the target bank's consolidator skips them. Inherited clutter is not just carried across, it is frozen.
- **The fix is to import the documents and omit the observations**, letting the target bank consolidate from scratch.
- **Cleanup after the fact is real but manual.** Invalidation is well built and reversible. Finding what to invalidate is left entirely to you.

## Nothing ever merges two duplicate facts

Start with the part that surprises people, because everything else follows from it.

When two different documents state the same fact in different words, Hindsight stores two memory units. There is no text-equality check, no embedding-similarity check and no upsert-on-duplicate anywhere in the retain path. The only thing that stops a fact reaching the database is degenerate text, meaning punctuation-only or empty content with zero information in it.

What happens instead is linking, not merging. Facts whose embeddings sit above `semantic_link_min_similarity`, which defaults to 0.7, get an edge between them in the memory graph. Two facts that are 99% similar get a very strong edge and both stay fully recallable.

There is also no background job that goes looking for duplicates later. The graph maintenance sweep does two things, topping up links and pruning stranded entities, and merging is not one of them.

This is a deliberate design, not an oversight. Memory is append-only because the raw layer is the audit trail: it is what you point at when you want to know why the system believes something. A retain path that silently dropped a fact because it looked like one it already had would be a retain path you could not trust to have stored what you sent it.

## Consolidation is the only thing that collapses anything

The deduplication happens one layer up, asynchronously, and it produces something new rather than destroying anything.

Consolidation reads raw facts and folds related ones into **observations**. An observation carries a `proof_count`, which is not an increment counter but a count of distinct source memory ids, computed fresh each time sources are folded in. Seeing the same thing in ten places gives you one observation citing ten facts, not ten observations.

The prompt is explicit about preferring that shape:

> Do NOT create a near-duplicate sibling. One canonical observation with many source facts is always better than many siblings with one source fact each.

Behind the prompt there are two deterministic guards. A create whose normalised text exactly matches an observation the model was already shown gets dropped. And a create that lands within `consolidation_dedup_threshold` of an existing observation, 0.97 by default, goes to a focused merge-or-keep adjudication that folds sources, `proof_count` and temporal bounds into the twin instead of creating a sibling.

Two caveats worth carrying. That semantic dedup is Postgres-only and is skipped entirely on Oracle regardless of how you set the threshold, because the merge SQL uses `unnest` and `array_agg`. And consolidation never deletes the raw facts underneath. It stamps them `consolidated_at` and leaves them exactly where they are. The duplicates do not go away. They stop being the thing you read.

## Which is exactly what an import turns off

Here is the mechanism that turns "my migration brought clutter" into "my migration brought permanent clutter."

When an archive carrying observations is imported, the importer restores those observations and then marks their source facts as already consolidated. The comment in the code says what it is doing without dressing it up:

> Mark source facts consolidated so the target consolidator skips them.

And the observations themselves arrive unreconciled. The importer's own docstring:

> Inserted as-is: imported observations are NOT merged or deduplicated against observations that already exist in the target bank (unlike consolidation, which merges related observations). Importing into a bank that already has observations, or importing the same archive twice, can therefore produce overlapping observations over the same facts.

Put those together. The observation layer you inherit is whatever the source bank had, duplicates included, and the one process that would have rebuilt it properly has been told its work is already done. Consolidation will not revisit those facts later, because from where it sits they are finished. Nothing in the system will ever look at them again.

The docs give the answer for a whole-bank restore:

> Prefer importing observations into a fresh/empty bank, or omit `include_observations` and let the target consolidate the imported facts itself.

That second clause is the important one, and it generalises further than the docs state it. If you are importing a history you did not curate, and especially one that accumulated in a different system, do not bring its conclusions. Bring the documents and the facts, leave the observations behind, and pay for one consolidation pass on arrival. You get an observation layer built by your bank, under your mission, with the dedup guards actually running. That pass costs LLM calls, which is the real reason to think about it rather than a reason to skip it.

One related trap: a large import can trigger a mental model refresh per consolidation, because `mental_model_min_refresh_interval_seconds` defaults to 0. If your bank has knowledge pages, set a floor before a bulk import rather than after.

## The duplicates you make yourself

Not all of this is inherited. The same append-only property means a handful of ordinary mistakes manufacture duplicates locally, and they show up most often during exactly the kind of bulk ingest a migration involves.

**Omitting `document_id`.** Retain falls back to a fresh UUID per request, so re-running the same content produces another document and another complete set of facts. The docs are blunt about it: "If you omit `document_id`, Hindsight assigns a random UUID per request, so re-ingesting the same content will create duplicate memories." Passing a stable `document_id` is the single biggest lever you have, and it is the only thing that gives you idempotency across retains.

**Choosing `document_conflict=new-id` on a merge import.** It does what it says, importing the document again under a fresh id. It is occasionally what you want and it is a duplicate factory by design. The default, `skip`, has the opposite failure mode: it quietly declines to import rather than telling you it found a collision.

**Importing more than 250 distinct new entity names at once.** Hindsight merges surface variants within a single retain, so "Acme Corp" and "Acme Corp." become one entity. That pass is capped at 250 unique new names and above the cap it is skipped silently, with only a warning in the logs. A bulk history import is precisely the shape that exceeds it, so the same entity can land three times under three spellings when a smaller retain would have merged them.

## Cleaning up after the fact

Say the clutter is already in there. The tool you want is **invalidation**, and it is genuinely well built.

`PATCH /v1/default/banks/{bank_id}/memories/{memory_id}` with `{"state": "invalidated"}` does not set a flag. It moves the row out of `memory_units` into a separate archive table. The engine explains the reasoning:

> Invalidation keeps the recall hot-path clean by *moving* the row between tables rather than flagging it… Recall/consolidation/graph queries therefore need no state predicate.

So an invalidated fact is not filtered out of recall, it is structurally absent from it. Links are pruned, derived observations are recomputed, and causal edges are snapshotted onto the archive row because nothing else could recreate them. It is fully reversible: restoring re-inserts the row, rebuilds the search vector with your current backend, resets the consolidation timestamps so it gets reconsidered, and restores entity postings from the snapshot. The original text is readable the whole time.

The gap is not the removal. It is the finding.

The documented workflow, in full, is one sentence: cluster duplicates from `memories/list`, then invalidate them. And `memories/list` gives you `text ILIKE '%q%'`. There is no similarity search, no grouping, no duplicate detection, and no way to export embeddings to do it yourself. There is also no bulk invalidate, so a thousand duplicates is a thousand API calls, and the CLI cannot invalidate at all.

There is a near-duplicate finder in the development tools that clusters observations by cosine similarity with a suggested threshold around 0.92. It is not part of the shipped product, and its README is honest about why it works the way it does: it re-embeds everything locally because "The Hindsight API has no bulk export and does not expose embedding vectors."

Two sharp edges if you go cleaning. `clear_memories` with a type filter also deletes the curation archive for that type, irreversibly, which is not mentioned in the route description. And reprocessing a document resets the curation of every fact it produced, since extraction runs fresh from the original text. Reprocess first to fix systematic extraction problems, then curate the residue, not the other way round.

## Recall will not save you

It is tempting to assume ranking sorts this out. It mostly does not, and the defaults are the reason.

Reciprocal rank fusion deduplicates by document id, so it collapses the same fact found by several retrieval arms and does nothing about different ids holding the same content. The cross-encoder scores each candidate independently, which means near-identical texts get near-identical scores and cluster together at the top rather than crowding each other out. The per-arm candidate cap defaults to 0, meaning off. The TypeSafe reranker's tail cut defaults to off, and it is a relevance cut anyway, so ten copies of the right answer all pass it.

`prefer_observations` is the one lever that helps, and it defaults to `false`. Turned on, it drops any raw fact that a returned observation was consolidated from, and backfills the freed slots. Note what that is: **provenance** deduplication, not semantic deduplication. Two duplicate facts that were never consolidated into the same observation both survive it.

If you are querying a bank with a history behind it, request observations and set `prefer_observations`. You are reading the reconciled layer instead of the raw one, which is what that layer is for.

## What I would actually do

Migrating a history you did not curate:

1. **Bring the documents, not the conclusions.** Omit observations on import so the target consolidates them itself.
2. **Give every document a stable `document_id`** before you start, so re-running a failed batch is idempotent rather than duplicative.
3. **Import in batches under 250 new entity names**, or accept that surface variants will not merge.
4. **Set a mental model refresh floor** before the import, not after.
5. **Set your `retain_mission` first.** Systematic extraction problems are cheap to fix before ingest and expensive afterwards, because reprocessing discards curation.
6. **Query with `prefer_observations`** and let the reconciled layer do the work.
7. **Then curate the residue**, accepting that finding it is currently a script you write yourself.

The underlying point is one the same author made about scoping, and it applies to inherited clutter just as well:

> A project archive should help in that project, not quietly influence an unrelated conversation.

A history is worth moving. Its conclusions usually are not.

## Learn more

- [How to Move Your Agent's Memory Off a Vector Database](https://hindsight.vectorize.io/blog/2026/07/28/migrate-agent-memory-off-vector-database) for getting the history in
- [A Bank Is Now Something You Can Pick Up](https://hindsight.vectorize.io/blog/2026/09/22/move-a-memory-bank) on clone, transfer and rename in 0.10.1
- [The Consolidation Problem in Agent Memory](https://hindsight.vectorize.io/blog/2026/05/21/agent-memory-consolidation) on importance, merge, decay and eviction as a framework
- [One Bank or Many? A Field Guide to Structuring Agent Memory](https://hindsight.vectorize.io/blog/2026/07/16/bank-strategy-agent-memory) on keeping archives from bleeding into each other
