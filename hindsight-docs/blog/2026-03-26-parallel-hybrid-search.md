---
title: "How We Built a 4-Way Hybrid Search System That Actually Runs in Parallel"
description: Sequential async queries were killing our retrieval latency. Here's how we built a true 4-way parallel hybrid search system with asyncio, CTEs, and RRF fusion — and cut end-to-end time by 64%.
authors: [hindsight]
date: 2026-03-26T12:00
tags: [engineering, retrieval, search, python, asyncio, performance]
image: /img/blog/blog-default.jpg
hide_table_of_contents: true
---

![How We Built a 4-Way Hybrid Search System That Actually Runs in Parallel](/img/blog/blog-default.jpg)

When we were designing Hindsight's memory retrieval, we had to confront one of the hardest challenges with AI data retrieval.

"Parallel" async code that wasn't actually parallel. You write four nice async functions, sprinkle in some awaits, and end up executing everything one after another. For a hybrid search stack with multiple retrieval strategies, that's unacceptable.

Here's how we built a 4-way hybrid search system that really does run in parallel, and what it took to make it fast in production — not just in benchmarks.

<!-- truncate -->

## Why Multi-Retrieval?

Most retrieval systems start with one approach — usually vector search — and call it a day. It works great in demos. Then real queries show up, and the cracks appear fast.

The core issue is that semantic similarity is not the same thing as relevance. Vector search finds things that are conceptually close in embedding space. But "close" is doing a lot of heavy lifting, and it breaks down in predictable ways depending on what the user is actually asking.

**Semantic search can't do exact matches.** Ask for a specific product SKU, an error code, a person's name, or an API endpoint, and vector embeddings will happily return things that are thematically related instead of the thing you asked for. The semantic "fuzziness" that makes vector search powerful becomes a liability when precision matters. A query for `HTTP 502 error` doesn't need a document about web servers in general — it needs the one that says "502."

**Keyword search can't do concepts.** Flip to BM25 and you get the opposite problem. It nails exact terms but falls over when the user phrases things differently than the document does. "How to fix" vs. "troubleshooting." "Car" vs. "automobile." If the query doesn't share tokens with the document, BM25 will never find it, no matter how relevant it is.

**Neither can follow relationships.** Ask "what happened after we changed the pricing model?" — a question about causation and sequence — and both semantic and keyword search will return documents that mention pricing changes. What they won't do is connect the pricing change to the downstream effects: the support tickets, the churn spike, the board discussion three weeks later. Those connections live in the relationships between memories, not in any single document's embedding or token set.

**Neither understands time.** "What was I working on last Tuesday?" is a simple question for a human. For a retrieval system, it requires parsing a natural-language date, resolving it to a range, and then finding memories bounded by that range. Standard vector search treats all documents as equally timeless. It will happily surface something from six months ago if the embeddings are close enough.

These aren't edge cases. In an agent memory system, they're the bread and butter. An agent might need the exact name of a library (keyword), the conceptual gist of a past conversation (semantic), the chain of events that led to a decision (graph), or what happened during a specific timeframe (temporal) — sometimes all in the same session.

A one-size-fits-all retriever forces you to pick which queries you're willing to get wrong. A hybrid system lets you stop choosing.

## The Problem: Four Very Different Retrieval Modes

Hindsight's memory system leans on four distinct retrieval strategies because different questions need different tools:

- **Semantic search** — vector similarity over embeddings for conceptual matches
- **BM25 keyword search** — classic full-text search for exact terms and phrases
- **Graph traversal** — walking relationships between memories (causal, temporal, entities)
- **Temporal search** — time-bounded recall with spreading activation over events

Each one has its own cost profile:

- Semantic search does vector similarity over an index
- BM25 uses PostgreSQL full-text indexes
- Graph traversal hops across relationship tables
- Temporal search parses natural-language dates, then runs range queries and neighbor walks

The naive implementation looks like this:

```python
# DON'T DO THIS – sequential in disguise
semantic_results = await retrieve_semantic(...)
bm25_results = await retrieve_bm25(...)
graph_results = await retrieve_graph(...)
temporal_results = await retrieve_temporal(...)
```

This is "async" but not concurrent: everything waits on the slowest step before starting the next.

## The Core Pattern: asyncio.gather with Real Independence

The first design decision was simple: treat each retrieval method as fully independent work. They share nothing except the connection pool. That means we can — and should — run them in parallel:

```python
async def _retrieve_parallel_hybrid(...) -> ParallelRetrievalResult:
    # All methods run independently in parallel
    semantic_result, bm25_result, graph_result, temporal_result = await asyncio.gather(
        run_semantic(),
        run_bm25(),
        run_graph(),
        run_temporal_with_extraction(),
    )
```

Each `run_*` function is responsible for:
- Acquiring a DB connection
- Executing queries
- Shaping results
- Recording timing metrics

Async only buys you concurrency if the underlying work is I/O-bound and independent. We leaned into that and designed the retrieval functions to be as self-contained as possible.

## Optimization 1: Combine Work with CTEs Instead of Extra Round-Trips

Parallelization across strategies doesn't excuse waste inside each one. For semantic and BM25, we often want results across multiple fact types. Instead of one query per fact type per method, we execute combined queries using common table expressions (CTEs) and window functions:

```sql
WITH semantic_ranked AS (
    SELECT id,
           text,
           embedding,
           fact_type,
           1 - (embedding <=> $1::vector) AS similarity,
           ROW_NUMBER() OVER (
               PARTITION BY fact_type
               ORDER BY embedding <=> $1::vector
           ) AS rn
    FROM memory_units
    WHERE bank_id = $2
      AND fact_type = ANY($3)  -- multiple fact types
      AND (1 - (embedding <=> $1::vector)) >= 0.3
),
bm25_ranked AS (
    SELECT id,
           text,
           fact_type,
           ts_rank_cd(search_vector, to_tsquery('english', $5)) AS bm25_score,
           ROW_NUMBER() OVER (
               PARTITION BY fact_type
               ORDER BY bm25_score DESC
           ) AS rn
    FROM memory_units
    WHERE bank_id = $2
      AND fact_type = ANY($3)
      AND search_vector @@ to_tsquery('english', $5)
)
SELECT * FROM (
    SELECT *, 'semantic' AS source FROM semantic_ranked WHERE rn <= $4
    UNION ALL
    SELECT *, 'bm25'    AS source FROM bm25_ranked    WHERE rn <= $4
) t;
```

The database does ranking, per-type limiting, and source tagging in one pass. That's fewer round-trips, less Python post-processing, and better cache locality on the DB side.

## Optimization 2: Treat the Connection Pool as a Shared Budget

Running four retrieval strategies in parallel is great until you starve the connection pool and everything stalls. To keep things healthy, we added explicit connection budgeting and timing:

```python
async def acquire_with_retry(pool, max_retries: int = 3, base_delay: float = 0.1):
    """Acquire a connection with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return await asyncio.wait_for(pool.acquire(), timeout=5.0)
        except asyncio.TimeoutError:
            if attempt >= max_retries - 1:
                raise
            await asyncio.sleep(base_delay * (2 ** attempt))
```

Each retrieval function records how long it waited for a connection:

```python
async def run_semantic() -> _TimedResult:
    start = time.time()
    acquire_start = time.time()
    async with acquire_with_retry(pool) as conn:
        conn_wait = time.time() - acquire_start
        results = await retrieve_semantic(conn, ...)
    return _TimedResult(results, time.time() - start, conn_wait)
```

That `conn_wait` metric ended up being one of the most useful signals for spotting pool pressure and mis-sized pools in production.

## Optimization 3: Parallelize Temporal Extraction Instead of Blocking on It

Temporal search has a sneaky bottleneck: natural-language date parsing. Libraries like `dateparser` are convenient but not cheap. The obvious implementation is:

```python
# DON'T DO THIS – everything waits on date parsing
temporal_constraint = extract_temporal_constraint(query_text)
temporal_results = await retrieve_temporal(..., temporal_constraint)
```

Now every retrieval path is effectively serialized behind "figure out what 'last Tuesday' means."

Instead, we treat temporal constraint extraction as just another chunk of work in the temporal branch:

```python
async def run_temporal_with_extraction() -> _TemporalWithConstraint:
    start = time.time()

    # Potentially slow date parsing
    extraction_start = time.time()
    tc = extract_temporal_constraint(query_text, ...)
    extraction_time = time.time() - extraction_start

    if tc is None:
        return _TemporalWithConstraint(
            results=[],
            total_time=time.time() - start,
            constraint=None,
            extraction_time=extraction_time,
        )

    async with acquire_with_retry(pool) as conn:
        results = await retrieve_temporal(conn, tc, ...)

    return _TemporalWithConstraint(
        results=results,
        total_time=time.time() - start,
        constraint=tc,
        extraction_time=extraction_time,
    )
```

Semantic, BM25, and graph retrieval don't care about temporal parsing and don't wait on it. Everything races together, and temporal only pays for its own complexity.

## Optimization 4: Batched Spreading Activation for Graph Walks

For temporal search, we often want "what was happening around this event" — which is essentially spreading activation over a time-aware subgraph. Doing one neighbor query per node explodes round-trips.

Instead, we work in batches:

```python
while frontier and budget_remaining > 0:
    # Take a batch from the frontier
    batch_ids = frontier[:batch_size]
    frontier = frontier[batch_size:]

    # Fetch all neighbors for this batch
    neighbors = await conn.fetch(
        """
        SELECT mu.*,
               ml.weight,
               ml.link_type,
               ml.from_unit_id
        FROM memory_links ml
        JOIN memory_units mu ON ml.to_unit_id = mu.id
        WHERE ml.from_unit_id = ANY($1::uuid[])
          AND ml.link_type IN ('temporal', 'causes', 'caused_by')
          AND ml.weight >= 0.1
        ORDER BY ml.weight DESC
        LIMIT $2
        """,
        batch_ids,
        batch_size * 10,
    )

    # ... update scores, frontier, and budget ...
```

With reasonable batch sizes, this shifts cost from O(nodes) queries to O(nodes / batch_size) and keeps latency predictable even for richer neighborhoods.

## Optimization 5: Indexes That Match How We Actually Query

Parallel work is pointless if each query slams into a full table scan. We tuned the schema to match our access patterns:

```sql
-- Vector similarity over embeddings
CREATE INDEX idx_memory_units_embedding
ON memory_units
USING ivfflat (embedding vector_cosine_ops)
WHERE embedding IS NOT NULL;

-- Full-text search
CREATE INDEX idx_memory_units_search_vector
ON memory_units USING gin(search_vector);

-- Temporal range queries
CREATE INDEX idx_memory_units_temporal_brin
ON memory_units USING brin (occurred_start, occurred_end);

-- Graph traversal fan-out
CREATE INDEX idx_memory_links_from_type_weight
ON memory_links (from_unit_id, link_type, weight DESC);
```

The goal is always the same: each retrieval function hits a well-supported index and gets out of the database quickly, so parallelism isn't just "four slow queries at once."

## What Parallel Actually Looks Like in Practice

With everything wired sequentially, retrieval times stacked:

```
Before (sequential)
-------------------
Semantic:   250 ms
BM25:       180 ms
Graph:      420 ms
Temporal:   340 ms   (includes ~120 ms date parsing)

Total:     1190 ms
```

With the 4-way parallel setup, total latency collapses to the slowest branch:

```
After (true parallel)
---------------------
Semantic:   250 ms ┐
BM25:       180 ms ├── all running concurrently
Graph:      420 ms ┤
Temporal:   340 ms ┘

Total:      420 ms  (bounded by graph branch)
```

Same work, ~64% less end-to-end time. Most of the engineering went into making that "same work" safe and observable.

## Observability: Making Parallel Retrieval Behavior Visible

To debug and tune this, we built a structured result object that carries both retrieved items and timings:

```python
@dataclass
class ParallelRetrievalResult:
    semantic:  list[RetrievalResult]
    bm25:      list[RetrievalResult]
    graph:     list[RetrievalResult]
    temporal:  list[RetrievalResult] | None

    timings: dict[str, float]              # per-branch totals
    temporal_constraint: tuple | None      # extracted time range, if any
    mpfp_timings: list[MPFPTimings]        # detailed graph traversal metrics
    max_conn_wait: float                   # worst connection wait across branches
```

This lets us answer questions like:
- Is one branch consistently dominating latency?
- Are we hitting connection pool limits (`max_conn_wait` spikes)?
- Is temporal extraction doing more work than its retrieval step?

Without this kind of breakdown, you're flying blind.

## Fusion: Letting the Retrieval Strategies Vote

Once all four retrieval strategies finish, we fuse their results so the agent sees a single ranked list instead of four disjoint ones. We use Reciprocal Rank Fusion (RRF) as the core mechanism:

```python
def fuse_results(parallel_result: ParallelRetrievalResult, k: int = 60):
    # Flatten results and tag sources for explainability
    semantic = parallel_result.semantic
    bm25     = parallel_result.bm25
    graph    = parallel_result.graph
    temporal = parallel_result.temporal or []

    for r in semantic:
        r.sources = ["semantic"]
    for r in bm25:
        r.sources = ["bm25"]
    for r in graph:
        r.sources = ["graph"]
    for r in temporal:
        r.sources = ["temporal"]

    fusion_scores: dict[str, float] = defaultdict(float)

    for source_results in [semantic, bm25, graph, temporal]:
        ranked = sorted(source_results, key=lambda r: r.score, reverse=True)
        for rank, result in enumerate(ranked, start=1):
            fusion_scores[result.id] += 1 / (k + rank)

    # Sort all unique results by fused score
    all_results = {r.id: r for r in (*semantic, *bm25, *graph, *temporal)}
    return sorted(all_results.values(), key=lambda r: fusion_scores[r.id], reverse=True)
```

RRF works well here because we don't have to normalize scores across very different scoring schemes — we just respect relative order within each method.

We chose RRF over alternatives for a specific reason: our four retrieval strategies produce incomparable scores (cosine similarity, BM25 tf-idf, graph hop weights, temporal decay). RRF is rank-based, so it sidesteps normalization entirely.

| Fusion method | How it works | When to use |
|---|---|---|
| **RRF** (our choice) | Rank-based, no score normalization | Mixed scoring schemes |
| CombSUM | Normalized score sum | When scores are comparable |
| Weighted average | Tuned per-source weights | When one source dominates |
| Cascade | Sequential filtering | Latency-first architectures |

## What We Learned Building This

A few principles that apply anywhere you're doing hybrid retrieval:

- **Async isn't magic.** You only get concurrency when tasks are independent, I/O-bound, and not competing for the same bottleneck.
- **Batch aggressively.** If you're looping and hitting the DB in each iteration, there's almost always a better, batched query.
- **Treat the connection pool as a global resource.** Time connection acquisition, use backoff, and design so no single branch can starve the pool.
- **Parallelize preprocessing too.** Things like temporal parsing belong inside their branch, not as a global pre-step everything waits on.
- **Invest in observability first.** You can't tune what you can't see; timings and per-branch metrics are non-negotiable.

## This Is What Powers Hindsight's Memory Recall

Every `recall_memory` call in Hindsight runs this stack. The 420ms bound is the latency budget an agent pays when it reaches into memory — and that budget holds even as memory banks grow into the tens of thousands of facts.

If you want to use this retrieval architecture without building it yourself, [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) runs it for you. The [recall docs](https://docs.hindsight.vectorize.io/docs/concepts/recall) cover the full technical reference, including how budget controls which strategies run and how reranking is applied.

## Building Parallel Hybrid Search That Holds Up in Production

A good parallel hybrid search system isn't just "we have four retrieval modes." It's:

- All four running truly in parallel
- Each one optimized internally for minimal round-trips
- Backed by indexes that reflect real query patterns
- Tied together by a fusion layer that can explain where results came from

In Hindsight, that combination turned a nice-on-paper design into something that can sit on the hot path of an agent's memory system without blowing the latency budget.
