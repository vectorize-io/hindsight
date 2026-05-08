---
title: "How Hindsight Scales"
description: "A design analysis of how Hindsight's memory operations scale with data volume — what costs grow, what stays bounded, and why."
authors: [nicoloboschi]
date: 2026-05-08T12:00
tags: [scaling, performance, architecture, engineering]
image: /img/blog/how-hindsight-scales.png
hide_table_of_contents: true
---

Agent memory systems face a scaling problem that traditional databases don't. It's not just "can we store more data" — it's "does the system stay fast, accurate, and affordable as memories pile up over weeks, months, and years."

The challenge is that agent memory involves LLM calls, semantic search, graph traversal, and synthesis. Each has its own scaling curve. Some scale with input size. Some scale with the total number of stored memories. Some scale with query complexity. Understanding which axis each operation scales on is enough to predict the shape of costs and latency, even before you have exact numbers.

This post is a design analysis, not a benchmark report — performance curves are a future post. What we can explain here is why each operation has the scaling profile it does, what the worst-case bounds are, and which knobs control what tradeoffs. Failure modes and recovery behavior (queue backpressure, rate-limit degradation) are also out of scope — we'll cover those separately.

Five architectural decisions shape the scaling story:

- **Read-write asymmetry**: we pay the LLM cost at write time so reads are LLM-free.
- **Hierarchical knowledge compression**: raw facts → observations → mental models, where each tier compresses the one below it.
- **Parallel everything**: four-way recall, 32-way extraction, async consolidation.
- **Bounded traversal**: every operation has a hard worst-case ceiling controlled by configuration.
- **Local models where possible**: embeddings and reranking run locally by default, so recall has zero LLM API cost.

The rest of this post shows how these decisions play out in each operation.

<!-- truncate -->

## Retain — Cost Scales with Input, Not with What's Already Stored

Retain is the write path. Content comes in, gets chunked, facts are extracted by an LLM, embeddings are generated, and everything gets stored with entity, temporal, semantic, and causal links.

**The scaling thesis: retain cost is proportional to what you're ingesting, not to what's already stored.** A retain call that processes 10 chunks costs the same whether the bank has 100 or 1,000,000 existing facts.

Here's why. The pipeline is a streaming producer-consumer system. Content is split into chunks of ~3,000 characters, grouped into mini-batches, and processed through three phases:

1. **Phase 1 (read-heavy, outside transaction):** Entity resolution via trigram GIN scan, semantic ANN search to find similar existing facts. Runs on a separate connection to avoid holding row locks during slow reads.
2. **Phase 2 (write transaction):** Insert facts, create entity links, build temporal links (within 24-hour windows), semantic links (within-batch + pre-computed ANN), and causal links. Atomic per batch.
3. **Phase 3 (post-transaction, best-effort):** Final ANN pass across the full bank — finds semantic neighbors for newly inserted facts against the entire existing corpus.

LLM fact extraction — the dominant cost — is one call per chunk (`retain_chunk_size`, default ~3,000 chars), parallelized up to 32 concurrent extractions. Chunks are grouped into mini-batches (`retain_chunk_batch_size`, default 100) that bound memory usage regardless of input size. Embeddings are one per extracted fact. DB writes are linear with the number of facts. All of these scale with input volume, not bank size. Each fact creates at most 20 temporal links — a hard cap that prevents link storage from growing unboundedly.

The one exception is Phase 3's ANN pass, which queries the full bank to find semantic neighbors for new facts. But HNSW gives O(log N) per query, so this grows slowly even at large bank sizes.

**Delta retain** makes repeated ingestion cheaper still. If a document's content hash matches a previous version, unchanged chunks are skipped entirely — no LLM extraction, no embedding, no writes. For integrations that periodically re-sync documents, the cost after the first sync drops to only the changed chunks.

Fact extraction quality is independent of bank size — each chunk is processed in isolation. What does improve with scale is **link density**: more facts in the bank means more temporal neighbors, more semantic neighbors, and richer entity co-occurrence graphs. The graph gets more connected over time, which benefits graph-based retrieval downstream.

## Recall — Zero LLM Calls, O(log N) Retrieval

Recall is the read path. It runs four retrieval strategies in parallel, fuses them with Reciprocal Rank Fusion, and reranks with a cross-encoder. We covered the [full architecture in a previous post](/blog/2026/03/27/parallel-hybrid-search). Here we focus on what scales with what.

**The scaling thesis: recall makes zero LLM calls.** It's purely retrieval plus cross-encoder reranking. This is a deliberate architectural choice — we pay the LLM cost at retain time (fact extraction) so that recall is free of LLM API costs at any scale.

Here's what costs nothing, what costs O(log N), and what's bounded by configuration:

**Costs nothing (structurally zero):** LLM calls. There are none. The cross-encoder reranker runs locally by default (a small model on CPU), though it can also be configured to use external providers. Either way, the recall path itself never calls an LLM.

**Costs O(log N):** Semantic search. HNSW indexes give logarithmic query time. We use per-bank, per-fact-type partial indexes so the query planner hits exactly the right index for each query arm. BM25 via PostgreSQL GIN indexes also scales sub-linearly with corpus size.

**Bounded by configuration:** Graph traversal is capped by `budget` (LOW=100, MID=300, HIGH=1000 nodes explored). Temporal search is bounded to 5 BFS iterations with at most 10 neighbors expanded per source unit. These are hard ceilings — graph and temporal retrieval time is effectively constant regardless of how dense the link graph gets. `ef_search` (default 200) controls HNSW search thoroughness, trading recall quality for query speed.

The four strategies run in parallel — semantic + BM25 + temporal share a connection; graph runs independently per fact type. Total recall latency is the max of the slowest branch, not the sum. Cross-encoder inference and connection pool acquisition don't benefit from indexing, so they set the latency floor.

On the quality side, graph retrieval has more material to work with as the bank grows — denser entity co-occurrence and more semantic kNN paths give it more traversal options. Semantic search is the only strategy that can degrade slightly at very large scale (HNSW is approximate), mitigated by over-fetching and tunable `ef_search`. BM25 is lexical and stable. The ensemble effect means even if one strategy gets noisier, the other three compensate — RRF fusion is rank-based, so it handles mixed-quality inputs naturally.

## Consolidation — LLM-Bound Background Work

Consolidation runs after retain completes. It takes raw experience and world facts and synthesizes them into **observations** — consolidated knowledge that represents higher-level patterns and insights. Think of it as the system "thinking about" what it learned.

**The scaling thesis: consolidation cost scales linearly with the number of new memories, and it's LLM-bound** — roughly 80%+ of wall-clock time in our profiling. DB and embedding work is comparatively negligible.

The pipeline:

1. Fetch unconsolidated memories in batches
2. For each memory, run a recall to find related existing observations (parallel, DB-only recalls)
3. Group memories into sub-batches and make one LLM call per sub-batch
4. Execute the LLM's instructions: create, update, or delete observations
5. Generate embeddings for new/updated observations
6. Checkpoint: mark memories as consolidated

This runs asynchronously as a background worker — it never blocks user-facing operations. The batch architecture has built-in backpressure: `consolidation_max_memories_per_round` (default 100) caps how many memories a single round processes, and `consolidation_llm_batch_size` (default 8) controls how many memories go into each LLM call. Together these give a hard ceiling: a consolidation round will never make more than `max_memories_per_round ÷ llm_batch_size` LLM calls. Adaptive error handling bisects failed sub-batches (8→4→2→1) and retries, so one bad memory doesn't block the rest.

Consolidation quality **improves with scale** — more raw facts mean richer source material for observation synthesis. Scope isolation (tag-based) prevents cross-context contamination, and source fact tracking preserves provenance so every observation can be traced back to the facts it was synthesized from.

Over time, consolidation becomes sub-linear in a different sense: as the bank matures, more memories update existing observations rather than creating new ones. The LLM call count per batch stays constant, but the observation count grows slower than the memory count.

## Reflect — Bounded Reasoning Over Hierarchical Knowledge

Reflect is the synthesis operation. Given a question, it searches through a three-tier knowledge hierarchy — mental models, then observations, then raw facts — using an agentic LLM loop.

**The scaling thesis: reflect quality is decoupled from total memory count.** The hierarchical retrieval means reflect reasons over observations and mental models (which compress the raw corpus), not over all memories directly. The raw fact search is a targeted fallback, not a full-corpus scan.

The agent follows a forced tool-call sequence before entering free-form reasoning. Each iteration — including forced ones — is an LLM call where the model is constrained to use a specific tool:

1. **Search observations** (forced): LLM call with tool choice constrained to search observations. Calls recall internally, filtered to observation-type facts.
2. **Search raw facts** (forced): LLM call with tool choice constrained to recall experience and world facts.
3. **Search mental models** (forced, only when mental models exist in the bank): LLM call with tool choice constrained to search mental model embeddings.
4. **Reasoning iterations** (auto): The LLM decides whether to expand results, run additional searches, or synthesize a final answer.
5. **Final synthesis**: Forced text-only response with no tools.

Every iteration is an LLM call. The forced sequence is typically 2 steps (observations + raw facts), extending to 3 when the bank has mental models. After the forced sequence, the agent enters auto mode where the LLM chooses freely.

Hard ceilings prevent runaway cost: `reflect_max_iterations` (default 10) caps the total number of iterations, `reflect_max_context_tokens` (default 100,000) forces final synthesis if accumulated context grows too large, and `reflect_wall_timeout` (default 300s) is a wall-clock cutoff. With defaults, a reflect call will never make more than 12 LLM calls (2 forced + 10 auto iterations), never accumulate more than 100K tokens, and never run longer than 5 minutes.

As banks grow, observations absorb complexity. Instead of reflect needing to reason over thousands of raw facts, it reasons over the observations that summarize them. Mental models provide an even higher-level cache — pre-computed answers to common questions that reflect can reference without re-deriving.

## Mental Models — Refresh Cost Scales with New Information, Not Total Knowledge

Mental models sit at the top of the knowledge hierarchy. They're user-defined questions with pre-computed answers — pinned reflections that the system keeps up to date.

**The scaling thesis: delta refresh processes only new facts since the last refresh, so a mental model backed by 10,000 facts that sees 5 new ones costs the same to refresh as one backed by 100.** Refresh cost scales with the rate of new information, not the stock of existing knowledge. And reading a mental model is a direct database lookup by ID — no LLM, no vector search. The expensive work happens at refresh time, not at read time.

A mental model is defined by a **source query**, **tags** (scope filter), and a **trigger configuration**. The key trigger flag is `refresh_after_consolidation` — when enabled, the chain is automatic: retain → consolidation creates observations → mental models with matching tags refresh in the background.

Refresh runs in two modes. **Full refresh** runs a complete reflect cycle internally (forced searches + reasoning + synthesis), rewriting the entire document from scratch — simple and reliable, but it regenerates sections that haven't changed. **Delta refresh** narrows recall to facts created *after* the last refresh timestamp. Instead of synthesizing a new document, the LLM emits structured patch operations (append/insert/replace/remove blocks and sections) applied to the existing document structure. Sections not mentioned are untouched — zero prose drift on stable content, manually curated sections survive refreshes intact. Delta costs one extra LLM call for operation generation but processes a much smaller context. It falls back to full refresh automatically when the source query has changed, the document structure is malformed, or no new facts exist since the last refresh.

The background cost chain matters: N new memories → M observation updates → K mental model refreshes → K × multiple LLM calls each. For a bank with many mental models and frequent retains, this adds up. But consolidation-triggered refreshes are asynchronous — they run as background tasks, never blocking user-facing operations.

## Knobs and What They Trade Off

These are the configuration parameters that control scaling behavior, organized by what they trade off:

**Recall quality vs. latency:**
- `budget` — higher budget explores more graph nodes, finding more connections at the cost of traversal time
- `ef_search` — higher values make HNSW search more thorough, reducing approximation error at the cost of query time
- Semantic over-fetch multiplier — fetching more candidates from HNSW improves recall precision but increases the reranking workload

**Consolidation throughput vs. resource usage:**
- `consolidation_llm_batch_size` — larger batches mean fewer LLM calls but bigger prompts. Constrained by your provider's context window and rate limits
- `consolidation_max_memories_per_round` — higher limits process more memories per round but hold the worker slot longer

**Reflect depth vs. cost:**
- `reflect_max_iterations` — fewer iterations mean faster, cheaper reflects at the cost of less thorough reasoning
- `reflect_max_context_tokens` — lower ceiling forces earlier synthesis, trading depth for predictability

**Mental model freshness vs. background cost:**
- `refresh_after_consolidation` — enables automatic refresh but adds LLM calls after every consolidation round
- Full vs. delta refresh mode — delta is cheaper per refresh (only processes new facts) but can't restructure the document

**Horizontal scaling:**

Hindsight uses a broker-based architecture with two distinct process types that scale independently:

- **API processes** (`hindsight-api`) handle HTTP/MCP requests and write background tasks to a shared PostgreSQL `async_operations` table. Run as many instances as you need behind a load balancer — they share no state except the database.
- **Worker processes** (`hindsight-worker`) poll PostgreSQL for pending tasks and execute background operations (consolidation, mental model refresh). Multiple workers compete to claim tasks from the queue — no process-to-process communication required.

For single-instance development, `HINDSIGHT_API_WORKER_ENABLED=true` (the default) embeds the worker inside the API process. In production, disable it and run dedicated worker instances to separate background work from user-facing traffic.

Connection pooling (PgBouncer or equivalent) becomes important once you're past a handful of API instances. Workers are LLM-bound, so scaling workers past your LLM provider's rate limit won't help — add more workers only if you have rate limit headroom.
