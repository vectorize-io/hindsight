---
title: "How We Made Retain's Peak Memory Flat, from 4 MB to 90 MB Documents"
authors: [benfrank241]
slug: "2026/08/27/retain-memory-budget"
date: 2026-08-27T12:00
tags: [hindsight, retain, memory, performance, engineering, deep-dive]
description: "How Hindsight 0.9.2 rebuilt the retain pipeline so peak memory stays flat from a 4 MB document to a 90 MB one, bounded by a budget instead of the input."
image: /img/blog/retain-memory-budget.png
hide_table_of_contents: true
---

![How Hindsight bounded retain's memory by a budget instead of by the document: windowed sizing, streamed chunking, and a streamed sub-batch split keep peak memory flat across document sizes](/img/blog/retain-memory-budget.png)

Retaining one large document into [Hindsight](https://hindsight.vectorize.io) used to cost memory proportional to the *document*, not to the work being done. Feed it a 90 MB body and peak allocation climbed with it, which is exactly the shape you do not want in a worker you are trying to size and pack. Hindsight 0.9.2 rebuilt the front half of the retain pipeline so peak memory stays flat from a 4 MB body to a 90 MB one, and added an explicit budget you can size a worker against.

The interesting part is where the memory actually went. It was not where anyone assumed.

<!-- truncate -->

## TL;DR

- Retaining a large document held state proportional to the **document**, not to the working set. Peak memory scaled with input size.
- The usual suspect (embeddings held as `list[float]`) was **not** the culprit. Measured, two *whole-document* operations dominated, both running before a single fact or embedding existed.
- Three fixes make peak memory **flat at any document size**: windowed token sizing, streamed chunking, and a streamed sub-batch split.
- The bound is now **explicit**: `HINDSIGHT_API_RETAIN_MEMORY_BUDGET_MB` (default `128`) is the number to size a worker against, whatever the document.
- The submitted body is the only thing still proportional to input, and it's the floor: it has to survive to be written as `documents.original_text`.
- All released in **0.9.2**.

## The misdiagnosis

When peak memory tracks input size, the reflex is to blame the biggest obvious structure. Here that was the embeddings: extracted facts carry vectors, vectors are `list[float]`, Python boxes every float, so surely that was the balloon.

We measured it with `tracemalloc` instead of guessing. The facts were never the problem. The streaming pipeline already bounds them to `retain_chunk_batch_size` chunks, roughly 1,700 facts, about 21 MB, and it holds there no matter how large the document is.

The real cost was two operations that run at the very start of a retain, before any fact or embedding exists, and each of which built a whole-document-sized structure in memory:

| Operation | 45 MB body, before | after |
| --- | --- | --- |
| **Sizing** (`count_tokens`) | 384.7 MB | 9.6 MB |
| **Chunking** (`split_text`) | 200.8 MB | 0.0 MB |

Sizing a 45 MB body allocated 385 MB, and chunking it another 200 MB, both before the pipeline had produced a single fact. That is where the peak came from.

## Fix 1: size the body a window at a time

Counting tokens to check a body against a batch budget was tokenizing the entire document and building one boxed Python `int` per token. For an 11.6 M-token body that is 11.6 M integer objects held at once, purely to produce a count.

`count_tokens_windowed()` sizes the body a megabyte at a time and keeps a running total, so it never holds more than one window's worth of tokens. Peak for sizing drops from 384.7 MB to 9.6 MB, and it is now flat at any document size.

There is a deliberate tradeoff hiding in "windowed": counting per window can miss a token that straddles a window boundary. It does, by about 45 tokens in 11.6 M. Every caller either compares the count against a batch budget or logs it, so a 0.0004% undercount is unobservable in practice, and that is the whole reason it is safe to window.

## Fix 2: stream the chunks instead of materializing them

Chunking used LangChain's `RecursiveCharacterTextSplitter`, which returns every chunk of the document at once, as a list. That list is the document again, cut up and held in memory alongside the original.

`iter_chunks()` streams chunks instead, yielding them one at a time, which includes a lazy re-implementation of exactly the splitter configuration retain relied on. Peak for chunking drops from 200.8 MB to effectively zero, flat at any size.

That re-implementation is worth a note, because a chunker is not something you want to quietly get *slightly* wrong: a shifted boundary would silently re-chunk every document you ever store. So **LangChain did not leave the codebase, only the runtime path.** It is kept as the reference that a differential test diffs our streaming splitter against, across every separator tier, so our boundaries are pinned to LangChain's own output. (This is why `langchain-core` and `langsmith` are still in the dependency list even though nothing imports them at request time.)

## Fix 3: stream the sub-batch split too

One whole-document structure was left. When a body is too large for a single retain, it is split into sub-batches, and the splitter returned all of them at once. Those slices *are* the document, cut up: 45.5 MB for a 45 MB body, held for the entire retain on top of the submitted body itself. Peak and live memory of the front half therefore still tracked the input, at two copies of it.

`iter_sub_batches()` yields them instead, screened, content-hashed, and flagged as each one arrives, so peak stops tracking the document:

| Peak (split + screening) | 16 MB body | 45 MB body | 90 MB body |
| --- | --- | --- | --- |
| **before** | 55.6 MB | 113.9 MB | 204.4 MB |
| **after** | 25.5 MB | 25.5 MB | 25.6 MB |

Live memory moves the same way, from 68.9 MB down to 23.5 MB on the 45 MB body, and flat. What is left scaling with the document is the submitted body itself, and that is the floor: it has to survive to be written as `documents.original_text`.

Making the loop stream did cost one invariant. `is_last` used to be trivial when you held the whole list; now it comes from a one-item lookahead in the generator, because the retain loop must know which sub-batch is last: the transactional-outbox callback fires inside the last sub-batch's transaction and nowhere else. That invariant is covered by tests added just before the change.

## The bound is now explicit

Rebuilding the internals makes peak memory flat; a config makes it a number you can plan around. `HINDSIGHT_API_RETAIN_MEMORY_BUDGET_MB` (default `128`) caps how much extracted-but-unwritten state a single retain may hold. Peak per retain is roughly this figure whatever the document, multiplied by your concurrent retain slots, which is exactly what you need to size a worker. Over budget, extraction waits for the write path to catch up rather than growing.

This replaces a bound that only looked like one. `HINDSIGHT_API_RETAIN_CHUNK_BATCH_SIZE` already capped how many chunks were in flight, but a *count* is only a memory bound if each item costs a predictable amount, and chunks do not: a chunk carries however many facts the extractor found in it. The megabyte budget is the honest bound. Set `RETAIN_MEMORY_BUDGET_MB` to `0` to restore the old count-only behavior.

## The result

Put together, the retain front half no longer scales with the document. Three whole-document structures became three streams:

| Fix | Change | 45 MB peak, before → after |
| --- | --- | --- |
| Windowed sizing | count a window at a time, not one boxed int per token | 384.7 MB → 9.6 MB |
| Streamed chunking | yield chunks, don't materialize the whole list | 200.8 MB → 0.0 MB |
| Streamed sub-batch split | yield screened sub-batches one at a time | 113.9 MB → 25.5 MB |

Peak memory is now flat across 4, 16, 45, and 90 MB bodies, held by an explicit budget instead of by the size of whatever someone happened to submit. Sizing and chunking are a touch faster than before as a bonus, since neither is building a giant throwaway structure anymore.

## Sizing a worker, concretely

Because peak is now a budget rather than a function of input, sizing a retain worker is arithmetic instead of guesswork:

- **Working set** ≈ `RETAIN_MEMORY_BUDGET_MB` × concurrent retain slots. At the default 128 MB and 4 slots, that is ~512 MB.
- **Plus the bodies in flight.** The one thing still proportional to input is each submitted body, which has to survive to be written as `documents.original_text`. Budget headroom for the largest bodies you actually accept, times your slots.
- **That is the whole model.** A worker sized this way holds whether your users retain a paragraph or a 90 MB export, and over budget the pipeline applies backpressure instead of climbing past your ceiling.

## Frequently asked questions

### Why did peak memory scale with the document before?

Two operations at the start of a retain each built a whole-document-sized structure before any fact existed: token sizing boxed one Python `int` per token, and chunking materialized every chunk as a list. On a 45 MB body that was 385 MB and 200 MB respectively. The facts and embeddings, the usual suspects, were already bounded and were never the cause.

### What does `HINDSIGHT_API_RETAIN_MEMORY_BUDGET_MB` actually do?

It caps the extracted-but-unwritten state one retain operation may hold, defaulting to 128 MB. Peak per retain is roughly that number regardless of document size, so you size a worker at budget × concurrent retains. Over budget, extraction applies backpressure and waits for the write path instead of growing. Set it to `0` to fall back to the previous count-only bound.

### Did you remove LangChain?

Only from the runtime path. Chunking no longer imports `RecursiveCharacterTextSplitter`; it uses a streamed re-implementation. LangChain is kept as the reference a differential test diffs our splitter against, tier by tier, so boundaries can't silently drift, which is why `langchain-core` and `langsmith` remain in the dependency list.

### Is any of this configurable per bank?

The memory budget and chunk-batch size are server-level knobs (`HINDSIGHT_API_RETAIN_MEMORY_BUDGET_MB`, `HINDSIGHT_API_RETAIN_CHUNK_BATCH_SIZE`), set via environment variables. They shipped in 0.9.2.

### How did you measure the numbers?

`tracemalloc`, reported as peak Python bytes allocated and order-independent, so a figure does not depend on which allocation happened to be live at the sampling instant. Each number is the front half of a retain (sizing, chunking, split, screening) on a body of the stated size, run in isolation.

### Does this change retain throughput or latency?

There is no throughput regression, and sizing and chunking are marginally faster now because neither builds a giant throwaway structure. This work changed the memory footprint of the retain front half, not the wall-clock of fact extraction, which is the expensive part and is unchanged.

### What about the memory the embeddings use?

It was already bounded before this work, which is why the embeddings were a red herring. Extracted facts and their vectors are held to `retain_chunk_batch_size` chunks, roughly 1,700 facts or about 21 MB, regardless of document size. The document-proportional peak came entirely from the two whole-document operations that ran before any fact existed.

### Do I need to change anything to get the flat memory?

No. The rebuilt sizing, chunking, and split are automatic in 0.9.2. `HINDSIGHT_API_RETAIN_MEMORY_BUDGET_MB` (default 128) is there if you want to tune the ceiling, but leaving it at the default already gives you flat, bounded peak.

### Does this affect the recall path too?

No, this is the retain (write) front half. Recall has its own budgeting, and 0.9.2 separately made recall skip over-budget facts instead of stopping, but the windowed sizing, streamed chunking, and streamed split described here are write-path only.

---

**Learn more:**
- [What's new in Hindsight 0.9.2](https://hindsight.vectorize.io/blog/2026/08/24/version-0-9-2) — the release this shipped in
- [How we built a 4-way parallel hybrid search](https://hindsight.vectorize.io/blog/2026/03/27/parallel-hybrid-search) — the other side of the pipeline, on the recall path
- [Hindsight on GitHub](https://github.com/vectorize-io/hindsight) — the open-source memory engine
