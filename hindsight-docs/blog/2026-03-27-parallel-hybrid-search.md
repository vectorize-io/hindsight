---
title: "How We Built a 4-Way Hybrid Search System That Actually Runs in Parallel"
description: Sequential async queries were killing our retrieval latency. Here's how we built a true 4-way parallel hybrid search system with asyncio and RRF fusion — then evolved it further with connection sharing, cross-encoder reranking, and multiplicative boost scoring.
authors: [chrislatimer, nicoloboschi, benfrank241]
date: 2026-03-27T12:00
tags: [engineering, retrieval, search, python, asyncio, performance, deep-dive]
image: /img/blog/parallel-hybrid-search.png
hide_table_of_contents: true
---

![How We Built a 4-Way Hybrid Search System That Actually Runs in Parallel](/img/blog/parallel-hybrid-search.png)

When we were designing Hindsight's memory retrieval, we had to confront one of the hardest challenges with AI data retrieval.

"Parallel" async code that wasn't actually parallel. You write four nice async functions, sprinkle in some awaits, and end up executing everything one after another. For a hybrid search stack with multiple retrieval strategies, that's unacceptable.

Here's how we built a 4-way hybrid search system that really does run in parallel, how we evolved it to share connections and reduce round-trips, and how reranking ties it all together. Spoiler: the biggest bottleneck turned out not to be query speed — it was connection pool contention, and that reshaped the whole architecture.

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
- **Graph traversal** — expanding through precomputed link signals (entity co-occurrence, semantic kNN, causal chains)
- **Temporal search** — time-bounded recall with spreading activation over events

Each one has its own cost profile:

- Semantic search does vector similarity over an HNSW index
- BM25 uses PostgreSQL full-text indexes
- Graph traversal expands from seeds through precomputed link tables
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

## V1: asyncio.gather with Real Independence

Our first design decision was simple: treat each retrieval method as fully independent work. They share nothing except the connection pool. That means we can — and should — run them in parallel:

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

Each `run_*` function was responsible for:
- Acquiring its own DB connection
- Executing queries
- Shaping results
- Recording timing metrics

This was a solid improvement — total latency collapsed from the sum of all branches to the max of the slowest branch. But it had problems. Each branch grabbed its own connection from the pool (four connections per recall), temporal extraction blocked its branch even when there was no temporal component to the query, and semantic and BM25 ran separate queries against the same table.

We iterated from there.

## Evolution 1: Connection Sharing and Combined Queries

The 4-way parallel approach burned four connections per recall. Semantic and BM25 queries hit the same table with the same filters — the only difference is the scoring function. Running them separately meant two round-trips for what could be one.

We consolidated semantic + BM25 into a single query using per-fact-type `UNION ALL` arms. Each arm has its own `ORDER BY ... LIMIT`, so the query planner can use the per-fact-type partial HNSW indexes instead of falling back to a sequential scan:

```sql
-- Semantic arms: one per fact type, each hitting its partial HNSW index
(SELECT id, text, context, fact_type,
        1 - (embedding <=> $1::vector) AS similarity,
        NULL::float AS bm25_score,
        'semantic' AS source
 FROM memory_units
 WHERE bank_id = $2
   AND fact_type = 'world'
   AND embedding IS NOT NULL
   AND (1 - (embedding <=> $1::vector)) >= 0.3
 ORDER BY embedding <=> $1::vector
 LIMIT 100)

UNION ALL

-- BM25 arms: one per fact type
(SELECT id, text, context, fact_type,
        NULL::float AS similarity,
        ts_rank_cd(search_vector, to_tsquery('english', $4)) AS bm25_score,
        'bm25' AS source
 FROM memory_units
 WHERE bank_id = $2
   AND fact_type = 'world'
   AND search_vector @@ to_tsquery('english', $4)
 ORDER BY ts_rank_cd(search_vector, to_tsquery('english', $4)) DESC
 LIMIT $3)

UNION ALL
-- ... more arms for each fact type ...
```

We learned this the hard way: an earlier version used `ROW_NUMBER() OVER (PARTITION BY fact_type)` in a CTE, which looked cleaner but forced the planner into a full sequential scan — defeating the partial indexes entirely. The issue is that PostgreSQL's planner evaluates the CTE as a single unit: the window function needs to see all rows across all fact types before it can partition and rank them, so there's no way to push the `fact_type` filter down into the index scan. The `UNION ALL` approach is less elegant but lets each arm independently hit its index.

HNSW is approximate, so semantic arms over-fetch by 5x (minimum 100 results) and trim to the requested limit in Python. We also set `ef_search=200` globally on pool connections at init time to improve recall on sparse graphs.

We then took this further: since temporal retrieval also queries the same table and reuses the same connection, we run semantic + BM25 + temporal all on a single connection. The temporal constraint is extracted first (it's CPU work, no DB needed), and if a temporal component is detected, its query runs on the same connection immediately after:

```python
async def retrieve_all_fact_types_parallel(...) -> MultiFactTypeRetrievalResult:
    # Step 1: Extract temporal constraint (CPU work, no DB)
    temporal_constraint = extract_temporal_constraint(query_text, ...)

    # Step 2: Semantic + BM25 + Temporal on ONE connection
    async with acquire_with_retry(pool) as conn:
        semantic_bm25_results = await retrieve_semantic_bm25_combined(conn, ...)

        if temporal_constraint:
            temporal_results = await retrieve_temporal_combined(conn, ...)

    # Step 3: Graph retrieval per fact type in parallel (separate connections)
    graph_tasks = [run_graph_for_fact_type(ft) for ft in fact_types]
    graph_results = await asyncio.gather(*graph_tasks)
```

This reduced connection usage from four per recall to one (shared) + N (graph per fact type), while keeping graph traversal — the most expensive and independent strategy — fully parallel.

## Evolution 2: Reliable Connection Acquisition

Running retrieval strategies in parallel is great until you starve the connection pool and everything stalls. We wrapped connection acquisition with retry logic and exponential backoff:

```python
@asynccontextmanager
async def acquire_with_retry(pool, max_retries: int = 3):
    """Acquire a connection with retry logic for transient failures."""
    start = time.time()

    async def acquire():
        return await pool.acquire()

    conn = await retry_with_backoff(acquire, max_retries=max_retries)
    acquire_time = time.time() - start

    # Log slow acquisitions — indicates pool contention
    if acquire_time > 0.05:  # 50ms threshold
        pool_size = pool.get_size()
        pool_free = pool.get_idle_size()
        logger.warning(f"[DB POOL] Slow acquire: {acquire_time:.3f}s | size={pool_size}, idle={pool_free}")

    try:
        yield conn
    finally:
        await pool.release(conn)
```

The `retry_with_backoff` handles transient connection errors (interface errors, too-many-connections, deadlocks) with exponential backoff starting at 0.5s. The slow-acquire logging at 50ms turned out to be one of the most useful signals for spotting pool pressure in production — when that metric spikes, the pool needs resizing.

## Evolution 3: Link Expansion — Three Signals in One Query

Early graph retrieval used multi-hop BFS traversals, which meant O(hops) DB queries per traversal pattern. We replaced this with Link Expansion — a single-roundtrip approach that exploits the fact that all three graph signals (entity, semantic, causal) are precomputed and bounded at retain time.

At retain time, every new fact gets linked to related facts through three types of precomputed links:
1. **Entity links** — co-occurrence graph (bounded to 50 links per entity)
2. **Semantic links** — kNN graph (top-5 most similar facts, similarity >= 0.7)
3. **Causal links** — explicit chains (causes/caused_by/enables/prevents)

Because these links are bounded at write time, query-time expansion doesn't need fan-out caps or budget tracking. All three expansions fit in a single CTE query:

```python
class LinkExpansionRetriever(GraphRetriever):
    async def retrieve(self, pool, query_embedding_str, bank_id, fact_type, budget, ...):
        async with acquire_with_retry(pool) as conn:
            # Find semantic seeds (top-20 by embedding similarity)
            all_seeds = await _find_semantic_seeds(conn, query_embedding_str, ...)

            # Single CTE query expands all three signals at once
            entity_rows, semantic_rows, causal_rows = await self._expand_combined(
                conn, seed_ids, fact_type, budget
            )

        # Merge with additive scoring: entity + semantic + causal ∈ [0, 3]
        # Facts appearing in multiple signals accumulate higher scores
        for fid in all_ids:
            score_map[fid] = (
                entity_scores.get(fid, 0.0)     # tanh(count * 0.5) → [0, 1]
                + semantic_scores.get(fid, 0.0)  # similarity weight → [0.7, 1.0]
                + causal_scores.get(fid, 0.0)    # link weight → [0, 1]
            )
```

The CTE query issues all three expansions in a single roundtrip with a `source` discriminator column:

```sql
WITH entity_expanded AS (
    -- Entity co-occurrence: seeds → their precomputed entity-link neighbors
    SELECT mu.id, mu.text, ...,
           COUNT(DISTINCT ml.entity_id)::float AS score,
           'entity'::text AS source
    FROM memory_links ml
    JOIN memory_units mu ON mu.id = ml.to_unit_id
    WHERE ml.from_unit_id = ANY($1::uuid[])
      AND ml.link_type = 'entity'
      AND mu.fact_type = $2
    GROUP BY mu.id
    ORDER BY score DESC LIMIT $3
),
semantic_expanded AS (
    -- Semantic kNN: both outgoing and incoming directions
    SELECT id, text, ..., MAX(weight) AS score, 'semantic'::text AS source
    FROM (
        -- outgoing: seeds → their kNN at insert time
        SELECT mu.*, ml.weight FROM memory_links ml
        JOIN memory_units mu ON mu.id = ml.to_unit_id
        WHERE ml.from_unit_id = ANY($1::uuid[]) AND ml.link_type = 'semantic'
        UNION ALL
        -- incoming: facts inserted after seeds that found seeds as kNN
        SELECT mu.*, ml.weight FROM memory_links ml
        JOIN memory_units mu ON mu.id = ml.from_unit_id
        WHERE ml.to_unit_id = ANY($1::uuid[]) AND ml.link_type = 'semantic'
    ) sem_raw
    GROUP BY id, text, ... ORDER BY score DESC LIMIT $3
),
causal_expanded AS (
    -- Causal chains: explicit causes/enables/prevents links
    SELECT DISTINCT ON (mu.id) mu.*, ml.weight AS score, 'causal'::text AS source
    FROM memory_links ml
    JOIN memory_units mu ON ml.to_unit_id = mu.id
    WHERE ml.from_unit_id = ANY($1::uuid[])
      AND ml.link_type IN ('causes', 'caused_by', 'enables', 'prevents')
      AND ml.weight >= $4
    ORDER BY mu.id, ml.weight DESC LIMIT $3
)
SELECT * FROM entity_expanded
UNION ALL SELECT * FROM semantic_expanded
UNION ALL SELECT * FROM causal_expanded
```

This replaced multi-hop traversals with a single query per fact type. The key insight: by bounding link fan-out at write time (retain), we can do unbounded expansion at read time (recall) without worrying about query explosion.

## Evolution 4: Indexes That Match How We Actually Query

Parallel work is pointless if each query slams into a full table scan. We tuned the schema to match our access patterns:

```sql
-- Per-(bank, fact_type) partial HNSW indexes for vector similarity
-- Created per-bank at bank creation time
CREATE INDEX idx_mu_emb_{bank}_{fact_type}
ON memory_units USING hnsw (embedding vector_cosine_ops)
WHERE bank_id = '{bank_id}' AND fact_type = '{fact_type}';

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

The partial HNSW indexes per (bank, fact_type) were a key evolution. A single global HNSW index doesn't scale when you have many banks with different fact type distributions — the planner can't use it efficiently with the `WHERE` filters we need. Per-bank partials let each arm in the `UNION ALL` query hit exactly the right index.

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
    mpfp_timings: list[MPFPTimings]        # detailed graph retrieval metrics (seeds, edges, query time)
    max_conn_wait: float                   # worst connection wait across branches
```

This lets us answer questions like:
- Is one branch consistently dominating latency?
- Are we hitting connection pool limits (`max_conn_wait` spikes)?
- Is temporal extraction doing more work than its retrieval step?

Without this kind of breakdown, you're flying blind.

## Fusion: Letting the Retrieval Strategies Vote

Once all four retrieval strategies finish, we fuse their results so the agent sees a single ranked list instead of four disjoint ones. We use Reciprocal Rank Fusion (RRF) as the merge mechanism:

```python
def reciprocal_rank_fusion(result_lists: list[list[RetrievalResult]], k: int = 60) -> list[MergedCandidate]:
    """Merge multiple ranked result lists using RRF.

    RRF formula: score(d) = sum_over_lists(1 / (k + rank(d)))
    """
    rrf_scores = {}
    source_ranks = {}
    all_retrievals = {}

    source_names = ["semantic", "bm25", "graph", "temporal"]

    for source_idx, results in enumerate(result_lists):
        source_name = source_names[source_idx]
        for rank, retrieval in enumerate(results, start=1):
            doc_id = retrieval.id
            if doc_id not in all_retrievals:
                all_retrievals[doc_id] = retrieval
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
                source_ranks[doc_id] = {}

            rrf_scores[doc_id] += 1.0 / (k + rank)
            source_ranks[doc_id][f"{source_name}_rank"] = rank

    return [
        MergedCandidate(retrieval=all_retrievals[doc_id], rrf_score=score,
                        rrf_rank=rrf_rank, source_ranks=source_ranks[doc_id])
        for rrf_rank, (doc_id, score) in enumerate(
            sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True), start=1
        )
    ]
```

We chose RRF over alternatives for a specific reason: our four retrieval strategies produce incomparable scores (cosine similarity, BM25 tf-idf, graph hop weights, temporal decay). RRF is rank-based, so it sidesteps normalization entirely.

| Fusion method | How it works | When to use |
|---|---|---|
| **RRF** (our choice) | Rank-based, no score normalization | Mixed scoring schemes |
| CombSUM | Normalized score sum | When scores are comparable |
| Weighted average | Tuned per-source weights | When one source dominates |
| Cascade | Sequential filtering | Latency-first architectures |

But RRF is just the first stage. It gives us a single ordered list — it doesn't tell us which results are actually relevant to the query.

## Reranking: Cross-Encoder Scoring with Multiplicative Boosts

RRF produces a merged candidate list, but rank-based fusion can only combine relative orderings — it can't assess whether a result is actually a good answer. For that, we run a cross-encoder neural reranker over the top candidates.

The pipeline after RRF:

1. **Pre-filter**: Take the top 300 candidates by RRF score (reranking is expensive; we don't need to score everything)
2. **Cross-encoder scoring**: Run each (query, document) pair through a cross-encoder model that produces a relevance score
3. **Multiplicative boost scoring**: Adjust the cross-encoder score with recency and temporal proximity signals

The cross-encoder reranker prepends date information to each document so the model can reason about temporal relevance:

```python
async def rerank(self, query: str, candidates: list[MergedCandidate]) -> list[ScoredResult]:
    pairs = []
    for candidate in candidates:
        doc_text = candidate.retrieval.text
        if candidate.retrieval.context:
            doc_text = f"{candidate.retrieval.context}: {doc_text}"
        if candidate.retrieval.occurred_start:
            date_iso = candidate.retrieval.occurred_start.strftime("%Y-%m-%d")
            date_readable = candidate.retrieval.occurred_start.strftime("%B %d, %Y")
            doc_text = f"[Date: {date_readable} ({date_iso})] {doc_text}"
        pairs.append([query, doc_text])

    scores = await self.cross_encoder.predict(pairs)
    normalized_scores = [sigmoid(score) for score in scores]  # logits → [0, 1]
    # ... build ScoredResult objects ...
```

After cross-encoder scoring, we apply multiplicative boosts for recency and temporal proximity:

```python
_RECENCY_ALPHA: float = 0.2
_TEMPORAL_ALPHA: float = 0.2

def apply_combined_scoring(scored_results, now, recency_alpha=0.2, temporal_alpha=0.2):
    for sr in scored_results:
        # Recency: linear decay over 365 days → [0.1, 1.0]; neutral 0.5 if no date
        sr.recency = 0.5
        if sr.retrieval.occurred_start:
            days_ago = (now - sr.retrieval.occurred_start).total_seconds() / 86400
            sr.recency = max(0.1, min(1.0, 1.0 - (days_ago / 365)))

        # Temporal proximity: meaningful only for temporal queries; neutral otherwise
        sr.temporal = sr.retrieval.temporal_proximity if sr.retrieval.temporal_proximity is not None else 0.5

        recency_boost = 1.0 + recency_alpha * (sr.recency - 0.5)    # ∈ [0.9, 1.1]
        temporal_boost = 1.0 + temporal_alpha * (sr.temporal - 0.5)  # ∈ [0.9, 1.1]
        sr.combined_score = sr.cross_encoder_score_normalized * recency_boost * temporal_boost
```

Each alpha of 0.2 translates to a ±10% swing around the neutral point: `1 + 0.2 * (0 - 0.5) = 0.9` at worst, `1 + 0.2 * (0.5) = 1.1` at best. Combined, the two boosts can shift a result's final score by at most +21% or -19%.

The key design choice is **multiplicative** rather than additive boosts. This ensures recency and temporal signals always scale proportionally to the base relevance — a highly relevant old memory still beats a mediocre recent one, but between two equally relevant results, the more recent one wins.

Why 0.2 and not something larger? We calibrated against our benchmark suite. At 0.1 the boosts were invisible — recency made no difference even for explicitly time-scoped queries like "what did I work on last week." At 0.5, recency dominated too aggressively: a mediocre fact from yesterday would outrank a highly relevant one from a month ago. 0.2 was the sweet spot where time-scoped queries got the recency lift they needed without distorting results for time-agnostic queries.

The full recall pipeline, end to end:

1. **Retrieve**: Semantic + BM25 + temporal on a shared connection; graph in parallel per fact type
2. **Fuse (RRF)**: Merge the four result lists by rank, `k=60`
3. **Pre-filter**: Top 300 candidates by RRF score
4. **Rerank**: Cross-encoder neural scoring, sigmoid-normalized to `[0, 1]`
5. **Boost**: `combined_score = ce_normalized * recency_boost * temporal_boost`
6. **Return**: Final results sorted by combined score

## The Surprising Part

The thing that surprised us most wasn't any single optimization — it was how much of the final architecture was shaped by **connection pool pressure**, not query speed.

Our initial 4-way `asyncio.gather` design was correct in principle: the four retrieval strategies are independent. But in practice, each branch grabbing its own connection meant four connections held simultaneously per recall. Under load, that turned the connection pool into the bottleneck, and latency spiked not because queries were slow but because branches were waiting to acquire a connection. The `max_conn_wait` metric we added to debug this showed 200ms+ acquisition times — longer than some of the queries themselves.

That's what pushed us toward connection sharing: semantic + BM25 + temporal on a single connection, with only graph retrieval running independently. The total query time went up slightly (sequential within the shared connection), but end-to-end latency dropped because we stopped fighting the pool. The lesson generalizes: in async systems, the shared resource that gates concurrency is more important to optimize than the work itself.

## This Is What Powers Hindsight's Memory Recall

Every `recall_memory` call in Hindsight runs this stack: parallel hybrid retrieval, RRF fusion, cross-encoder reranking, and multiplicative boost scoring. The architecture holds even as memory banks grow into the tens of thousands of facts.

If you want to use this retrieval architecture without building it yourself, [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) runs it for you. The [recall docs](https://hindsight.vectorize.io/developer/retrieval) cover the full technical reference, including how budget controls which strategies run and how reranking is applied.

## Building Parallel Hybrid Search That Holds Up in Production

A good parallel hybrid search system isn't just "we have four retrieval modes." It's:

- Retrieval strategies running concurrently where it matters, sharing connections where it's efficient
- Each strategy optimized internally for minimal round-trips
- Backed by indexes that reflect real query patterns
- Tied together by RRF fusion, cross-encoder reranking, and a scoring formula that combines relevance with recency and temporal signals

In Hindsight, that combination turned a nice-on-paper design into something that can sit on the hot path of an agent's memory system without blowing the latency budget.
