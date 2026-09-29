---
title: "Bring the Facts, Not the Beliefs"
authors: [benfrank241]
slug: "2026/09/23/bring-facts-not-beliefs"
date: 2026-09-23T15:00
tags: [hindsight, agent-memory, migration, consolidation, observations, import]
description: "When you migrate a memory history, its duplicate conclusions come with it. The fix is to import the source material and let your bank rebuild the beliefs itself."
image: /img/blog/bring-facts-not-beliefs.png
hide_table_of_contents: true
---

![Bring the facts, not the beliefs: import source material and let consolidation rebuild the conclusions](/img/blog/bring-facts-not-beliefs.png)

Someone wrote up their migration onto Hindsight recently, and one line stuck with me:

> The move preserved the history. It also preserved the clutter.

Going through what landed, [they found](https://praveenks.com/notes/from-honcho-to-hindsight/) "duplicate conclusions and leftover test records."

That is the normal experience of moving a memory history between systems, and it has a specific cause that is worth understanding, because the fix is a single decision you make at import time.

<!-- truncate -->

## TL;DR

- **Duplicate conclusions are a property of the history you are moving**, not of the move itself.
- **Hindsight reconciles duplicates one layer up**, by folding related facts into a single observation with a count of how many times the thing was independently seen.
- **Importing another system's conclusions skips that step.** Imported facts arrive marked as already consolidated, so your bank never rebuilds the beliefs for itself.
- **So bring the source material and leave the conclusions behind.** One consolidation pass on arrival gives you an observation layer built by your bank, under your mission.
- **Curate the residue afterwards** with invalidation, which is reversible and keeps the audit trail intact.

## Why the raw layer keeps every duplicate

Hindsight's raw memory layer is append-only on purpose. When two documents state the same thing in different words, you get two memory units, and neither one is quietly discarded.

That sounds like a gap until you consider the alternative. The raw layer is the audit trail. It is what you point at when someone asks why the system believes something. A retain path that dropped a fact because it resembled one already stored would be a retain path you could not trust to have kept what you sent it, and the resemblance judgement would be happening at the worst possible moment: before anything downstream has any context about which of the two is right.

So Hindsight relates duplicates instead of destroying them. Facts whose embeddings sit above `semantic_link_min_similarity`, 0.7 by default, get an edge between them in the memory graph. Two facts that are 99% similar end up strongly linked, and both remain in the record.

The reconciliation happens a layer up, where it has the context to be done well.

## Consolidation is where duplicates collapse

Consolidation reads raw facts and folds related ones into **observations**. This is the layer your agent should usually be reading.

An observation carries a `proof_count`: not an increment counter, but a count of distinct source facts, recomputed each time new sources fold in. Ten documents independently mentioning the same thing produce one observation citing ten facts, rather than ten near-identical rows competing with each other at recall time. The consolidation prompt is explicit about preferring that shape:

> Do NOT create a near-duplicate sibling. One canonical observation with many source facts is always better than many siblings with one source fact each.

Two deterministic guards sit behind the prompt. A new observation whose normalised text exactly matches one the model was already shown is dropped. And one that lands within `consolidation_dedup_threshold` of an existing observation, 0.97 by default, goes to a focused merge-or-keep adjudication that folds the sources, the proof count and the temporal bounds into the existing twin instead of creating a sibling. The adjudication is deliberately conservative: it is instructed to keep the two separate when they differ in a number, a named entity, a negation or a condition, because those are the differences that look like duplication and are not.

The raw facts stay where they are. Consolidation stamps them as consolidated and leaves the audit trail whole. The duplicates do not disappear; they stop being the thing you read.

This is the mechanism that cleans up a messy history. Which brings us to the one way to accidentally switch it off.

## The one decision that matters at import

A whole-bank archive can carry observations as well as documents and facts. Bringing them across feels like the thorough choice. It is usually the wrong one.

When an archive carrying observations is imported, the importer restores those observations and then marks their source facts as already consolidated. The comment in the code is direct about it:

> Mark source facts consolidated so the target consolidator skips them.

And the imported observations themselves arrive unreconciled:

> Inserted as-is: imported observations are NOT merged or deduplicated against observations that already exist in the target bank (unlike consolidation, which merges related observations). Importing into a bank that already has observations, or importing the same archive twice, can therefore produce overlapping observations over the same facts.

Put those together and the shape of the problem is clear. You inherit whatever conclusions the source system had reached, duplicates included, and the process that would have rebuilt them properly has been told its work is already finished. Those facts will not be revisited later, because from consolidation's point of view they are done.

The documentation gives the answer plainly:

> Prefer importing observations into a fresh/empty bank, or omit `include_observations` and let the target consolidate the imported facts itself.

That second clause is the one to internalise, and it matters most in exactly the case the docs do not spell out: a history that accumulated somewhere else, under another system's notion of what a conclusion is.

**Bring the documents and the facts. Leave the conclusions.** Pay for one consolidation pass on arrival and you get an observation layer built by your bank, under your mission, with the dedup guards actually running over your whole corpus at once. That pass costs LLM calls, which is a reason to plan it rather than a reason to skip it.

It is the same instinct as a database migration: you move the source of truth and rebuild the indexes on the other side, because an index built for someone else's query patterns is not worth carrying.

If you have already imported the conclusions, this is recoverable in a single call rather than a re-migration. The FAQ below has it.

## Set the ingest up before you run it

A migration is a bulk ingest, and a few things are much cheaper to get right beforehand than to repair afterwards.

**Give every document a stable `document_id`.** This is the single biggest lever available. With one, re-running a failed batch updates the existing document rather than creating another copy. Without one, retain assigns a fresh UUID per request, so the same content ingested twice becomes two documents and two complete sets of facts. The docs put it plainly: "If you omit `document_id`, Hindsight assigns a random UUID per request, so re-ingesting the same content will create duplicate memories."

**Settle your `retain_mission` first.** Extraction quality is a systematic property, and fixing it before ingest is cheap. Fixing it afterwards means reprocessing documents, which re-extracts from the original text and therefore discards any curation you had already done on the facts they produced. Get the mission right, then ingest, then curate.

**Watch the batch size for entity variety.** Hindsight merges surface variants within a single retain, so "Acme Corp" and "Acme Corp." resolve to one entity. That in-batch pass is capped at 250 unique new names and is skipped above the cap. A bulk import is exactly the shape that exceeds it, so batching an import to stay under that ceiling keeps variants merging that would otherwise land as separate entities.

**Set a mental model refresh floor.** `mental_model_min_refresh_interval_seconds` defaults to 0, so a large import can trigger a refresh per consolidation. If your bank has knowledge pages, set the floor before the import rather than discovering it during.

## Curating what is left

Some residue survives any migration: the leftover test records, the conclusions that were wrong before they were copied. The tool for those is **invalidation**, and it is worth knowing how well it behaves.

`PATCH /v1/default/banks/{bank_id}/memories/{memory_id}` with `{"state": "invalidated"}` does not set a flag to be filtered later. It moves the row out of the active table into a separate archive:

> Invalidation keeps the recall hot-path clean by *moving* the row between tables rather than flagging it… Recall/consolidation/graph queries therefore need no state predicate.

So an invalidated fact is structurally absent from recall rather than filtered out of it, which means no query pays for your cleanup. Links are pruned, derived observations are recomputed, and causal edges are snapshotted onto the archived row because nothing else could recreate them.

It is also fully reversible. Restoring re-inserts the row, rebuilds its search vector with your current backend, resets the consolidation timestamps so the fact gets reconsidered, and restores its entity postings from the snapshot. The original text stays readable the whole time. You can retire something you are not certain about without destroying it, which is the property you want when you are curating a history you did not write.

One note if you are cleaning in bulk: clearing memories by type also clears the curation archive for that type, so do your invalidation after any wholesale clearing, not before.

## A migration checklist

1. **Import documents and facts. Omit observations**, and let the target bank consolidate them itself.
2. **Assign a stable `document_id`** to everything before you start.
3. **Settle the `retain_mission`** before ingest, not after.
4. **Batch under 250 new entity names** so surface variants keep merging.
5. **Set a mental model refresh floor** if the bank has knowledge pages.
6. **Run consolidation and let it finish** before judging what landed.
7. **Query with `prefer_observations`** so recall returns the reconciled layer rather than the raw facts underneath it.
8. **Invalidate the residue**, knowing you can restore anything you retire.

The author of that migration write-up made a point about scope that applies just as well to conclusions:

> A project archive should help in that project, not quietly influence an unrelated conversation.

A history is worth moving. The beliefs another system formed about it usually are not, and rebuilding them is the cheapest part of the whole exercise.

## FAQ

**Does importing a bank cost LLM tokens?**
No. An import carries the source material and replays the deterministic half of retain: facts are re-embedded with the target bank's embedding model and entities are re-resolved against the target, with no extraction step. It costs embedding compute, not tokens, and it invents no new facts. The consolidation pass you run afterwards is the part that calls a model, which is the cost worth planning for.

**I already imported the observations. Can I fix it?**
Yes, and in one call. Clearing a bank's observations deletes them and resets the consolidation marker on every world and experience fact, specifically "so they get re-consolidated." Run `DELETE /v1/default/banks/{bank_id}/memories?type=observation`, or `hindsight bank clear-observations <bank_id>`, then trigger consolidation. Your facts are untouched, and the bank rebuilds its conclusions under your own mission. Do this before any invalidation work rather than after, since clearing by type also clears the curation archive for that type.

**Should I import into a fresh bank or merge into an existing one?**
A whole-bank restore requires a target that does not already exist, which is deliberate: merging an archive's config, mental models and webhooks into a populated bank would silently mix two sets of settings. Merge mode exists for the other case and works at document granularity, importing the archive's documents with a conflict policy for ids that already exist. If you are consolidating several histories into one bank, that is the path, and it is another reason to leave the source conclusions behind and let one consolidation pass reconcile everything together.

## Learn more

- [How to Move Your Agent's Memory Off a Vector Database](https://hindsight.vectorize.io/blog/2026/07/28/migrate-agent-memory-off-vector-database) for getting a history in
- [A Bank Is Now Something You Can Pick Up](https://hindsight.vectorize.io/blog/2026/09/22/move-a-memory-bank) on clone, transfer and rename in 0.10.1
- [The Consolidation Problem in Agent Memory](https://hindsight.vectorize.io/blog/2026/05/21/agent-memory-consolidation) on importance, merge, decay and eviction as a framework
- [One Bank or Many? A Field Guide to Structuring Agent Memory](https://hindsight.vectorize.io/blog/2026/07/16/bank-strategy-agent-memory) on keeping archives from bleeding into each other
