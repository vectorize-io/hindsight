---
title: "How We Built Time-Aware Spreading Activation for Memory Graphs"
authors: [chrislatimer]
date: 2026-03-12T12:00
tags: [retrieval, graph, temporal, spreading-activation, memory, deep-dive]
image: /img/blog/spreading-activation-memory-graphs.png
---

![Hindsight — Stories, Not Rows](/img/blog/spreading-activation-memory-graphs.png)

Most retrieval systems treat time as a filter. We treat it as a dimension to traverse.

<!-- truncate -->

When an agent gets a question like "What happened during the product launch last March?", a naive "filter by March" query is nowhere near enough. The real answer lives in the story around that window: the prep work in January and February, the domino effects in April, the bugs that nearly killed the demo, the customer reactions that followed. We wanted retrieval that could reconstruct that story, not just return rows.

Temporal spreading activation is how we do it in [Hindsight](https://ui.hindsight.vectorize.io/signup): a time-anchored graph traversal that starts from a temporal window, then walks causal and temporal links to build a coherent sequence of events. The idea draws on [spreading activation theory from cognitive science](https://en.wikipedia.org/wiki/Spreading_activation) — originally proposed by Collins and Loftus in 1975 to model how human memory retrieval works by propagating activation through semantic networks — but adapted here for structured memory graphs with explicit temporal and causal edges.

## The problem: time is not just a WHERE clause

The simplest version of temporal search treats time as a range filter:

```sql
SELECT *
FROM memories
WHERE date BETWEEN '2024-03-01' AND '2024-03-31';
```

That's fine for "show me what we logged in March." It falls apart for "what happened around the launch?" because it misses:

- **Upstream causes** — the bug filed in February that forced a scope cut.
- **Prep work** — launch rehearsals, asset reviews, rollout planning in January.
- **Downstream consequences** — support volume spikes and customer feedback in April.
- **Parallel context** — competitor announcements or incidents in the same time band.

What we actually want is:

1. Use time to find good **entry points**.
2. From those, **walk the memory graph** along meaningful relationships.
3. Keep everything **anchored to the original temporal intent**.

## The core algorithm: temporal spreading activation

At a high level, the algorithm combines:

- **Temporal entry points** — memories that lie in (or overlap with) the target time window and are semantically relevant.
- **Spreading activation** — a controlled propagation of "attention" through the graph.
- **Multi-signal scoring** — temporal proximity, semantic similarity, and link strength all contribute.

### Step 1: find temporal-semantic entry points

We start by finding memories that overlap the query window and are semantically related to the question:

```python
entry_points = await conn.fetch("""
    SELECT
        id,
        text,
        occurred_start,
        occurred_end,
        mentioned_at,
        1 - (embedding <=> $1::vector) AS similarity
    FROM memory_units
    WHERE bank_id   = $2
      AND fact_type = $3
      AND embedding IS NOT NULL
      AND (
          -- events whose range overlaps the query window
          (occurred_start IS NOT NULL AND occurred_end IS NOT NULL
           AND occurred_start <= $5 AND occurred_end >= $4)
          OR
          -- events mentioned inside the window
          (mentioned_at IS NOT NULL AND mentioned_at BETWEEN $4 AND $5)
          OR
          -- partial overlap via start/end
          (occurred_start IS NOT NULL AND occurred_start BETWEEN $4 AND $5)
          OR
          (occurred_end   IS NOT NULL AND occurred_end   BETWEEN $4 AND $5)
      )
      AND (1 - (embedding <=> $1::vector)) >= $6  -- semantic threshold
    ORDER BY COALESCE(occurred_start, mentioned_at, occurred_end) DESC
    LIMIT 10
""")
```

A few deliberate choices here:

- We treat time as **interval overlap**, not a single timestamp. The `<=>` operator comes from [pgvector](https://github.com/pgvector/pgvector), which we use for cosine distance on embeddings stored directly in Postgres.
- We gate by **semantic similarity** so we're not just pulling random events from that month.
- When in doubt, `occurred_start`/`occurred_end` win over `mentioned_at` — *when something happened* is more important than *when it was talked about*.

### Step 2: score temporal proximity

Next, we estimate how well each entry point aligns with the temporal center of the window:

```python
total_days = (end_date - start_date).total_seconds() / 86400
mid_date   = start_date + (end_date - start_date) / 2

for ep in entry_points:
    if ep["occurred_start"] and ep["occurred_end"]:
        best_date = ep["occurred_start"] + (ep["occurred_end"] - ep["occurred_start"]) / 2
    elif ep["occurred_start"]:
        best_date = ep["occurred_start"]
    elif ep["occurred_end"]:
        best_date = ep["occurred_end"]
    else:
        best_date = ep["mentioned_at"]

    days_from_mid = abs((best_date - mid_date).total_seconds() / 86400)
    temporal_proximity = 1.0 - min(days_from_mid / (total_days / 2), 1.0)
```

This gives us a **temporal gradient**:

- Events right in the middle of the window score near **1.0**.
- Events on the edges fall toward **0.0**.
- Longer-running events that span the window get a reasonable central estimate.

We seed each entry point with both a semantic similarity and a temporal score.

### Step 3: spread through the memory graph selectively

From those seeds, we spread activation through the memory graph — but only along link types that make temporal sense for "what happened around X?":

```python
node_scores = {str(ep["id"]): (ep["similarity"], 1.0) for ep in entry_points}
frontier    = list(node_scores.keys())
visited     = set(frontier)

budget_remaining = budget - len(entry_points)
batch_size       = 20  # nodes per DB round-trip

while frontier and budget_remaining > 0:
    batch_ids = frontier[:batch_size]
    frontier  = frontier[batch_size:]

    neighbors = await conn.fetch("""
        SELECT
            mu.*,
            ml.weight,
            ml.link_type,
            ml.from_unit_id,
            1 - (mu.embedding <=> $1::vector) AS similarity
        FROM memory_links ml
        JOIN memory_units mu ON ml.to_unit_id = mu.id
        WHERE ml.from_unit_id = ANY($2::uuid[])
          AND ml.link_type IN ('temporal', 'causes', 'caused_by',
                               'enables', 'prevents')
          AND ml.weight >= 0.1
          AND mu.fact_type = $3
          AND (1 - (mu.embedding <=> $1::vector)) >= $4
        ORDER BY ml.weight DESC
        LIMIT $5
    """, query_emb_str, batch_ids, fact_type,
        semantic_threshold, batch_size * 10)
```

We deliberately follow:

- **temporal** — co-occurring or nearby events.
- **causes / caused_by** — upstream/downstream causal chains.
- **enables / prevents** — preconditions and blockers.

Everything else stays out of this traversal. If we don't constrain link types, the walk quickly turns into "six degrees of everything."

### Step 4: propagate scores with decay and causal boosts

As we walk, we combine three signals for each neighbor:

1. Its own **temporal proximity** to the window.
2. **Activation propagated** from the parent.
3. The **strength and type** of the edge.

```python
for n in neighbors:
    neighbor_id = str(n["id"])
    parent_id   = str(n["from_unit_id"])

    if neighbor_id in visited:
        continue

    # Parent temporal score (defaults to mid if missing)
    _, parent_temporal_score = node_scores.get(parent_id, (0.5, 0.5))

    # Compute neighbor's own temporal proximity
    neighbor_best_date = pick_best_date(n)  # same logic as before
    neighbor_temporal_proximity = compute_proximity(
        neighbor_best_date, mid_date, total_days
    )

    # Causal boosts
    link_type = n["link_type"]
    if link_type in ("causes", "caused_by"):
        causal_boost = 2.0
    elif link_type in ("enables", "prevents"):
        causal_boost = 1.5
    else:
        causal_boost = 1.0

    propagated_temporal = parent_temporal_score * n["weight"] * causal_boost * 0.7

    combined_temporal = max(neighbor_temporal_proximity, propagated_temporal)
```

Intuitively:

- **Causal chains stay "hot" longer** — if A causes B causes C, C can still have a high temporal score even if it's a bit further from the window, because it's on a causally important path.
- **Pure co-occurrence decays faster** — temporal links without causal weight fade with distance.
- **Anchoring to the window remains** — if a node is very close in time, its own proximity score dominates.

We also track semantic similarity along the way and can combine the two at ranking time (e.g., via a weighted sum or as separate dimensions in the fusion step).

### Step 5: keep it sane with batching and budgets

A traversal like this can easily get out of hand without guardrails, so we put two in place: **batched frontier processing** and a **node budget**.

**Batched frontier processing.** Instead of one DB query per node, we always work in chunks:

```python
batch_size = 20

while frontier and budget_remaining > 0:
    batch_ids = frontier[:batch_size]
    frontier  = frontier[batch_size:]

    neighbors = await conn.fetch(
        "... WHERE ml.from_unit_id = ANY($2::uuid[]) ...",
        query_emb_str, batch_ids, ...
    )
```

This turns O(nodes) queries into roughly O(nodes / batch_size) queries and plays much nicer with Postgres.

**Budget-constrained exploration.** We cap how many nodes we're allowed to touch:

```python
for n in neighbors:
    neighbor_id = str(n["id"])
    if neighbor_id in visited:
        continue

    visited.add(neighbor_id)
    budget_remaining -= 1

    if combined_temporal > 0.2:
        node_scores[neighbor_id] = (n["similarity"], combined_temporal)
        if budget_remaining > 0:
            frontier.append(neighbor_id)

    if budget_remaining <= 0:
        break
```

The effect:

- We get **predictable upper bounds** on cost per query.
- We explore the **best paths first**, because neighbors are ordered by link weight and filtered by semantic similarity.
- We avoid flooding the graph with **low-value nodes**.

## A concrete example: reconstructing a product launch

Suppose the time window is March 15–20.

**Entry points.** We might find:

- "Product launch event at Convention Center" (March 18, high similarity, temporal ≈ 0.95)
- "CEO keynote announcing new features" (March 18, temporal ≈ 0.95)
- "Press embargo lifted" (March 17, temporal ≈ 0.85)

**First spread.** From "Product launch event," causal and temporal links might pull in:

- "Final rehearsal completed" (March 16, `caused_by`)
- "Marketing materials distributed to partners" (March 14, `enables`)
- "First customer orders received" (March 19, `causes`)
- "Server scaling policy triggered" (March 18, `enables`)

These nodes get strong scores because they're both close in time and on important edges.

**Second spread.** From "Final rehearsal completed" we might see:

- "Bug fix for demo crash" (March 10, `caused_by`)
- "Rehearsal schedule finalized" (March 1, `enables`)

Even though March 1 is further out, the causal boosts keep it in play as part of the narrative. By the time we finish, we've reconstructed a story: **prep → launch → immediate aftermath**, with the key upstream and downstream events connected.

<!-- TODO: Add a diagram showing the graph traversal with entry points, first spread, and second spread. Show temporal window boundaries, causal vs. temporal edges with different colors. -->

## Handling incomplete temporal data in memory graphs

Real data isn't clean; a lot of memories only have partial timestamps. We handle that without derailing the scoring:

```python
def pick_best_date(row):
    if row["occurred_start"] and row["occurred_end"]:
        return row["occurred_start"] + (row["occurred_end"] - row["occurred_start"]) / 2
    if row["occurred_start"]:
        return row["occurred_start"]
    if row["occurred_end"]:
        return row["occurred_end"]
    if row["mentioned_at"]:
        return row["mentioned_at"]
    return None
```

If we have no temporal data at all, we can either:

- Assign a **low default temporal score** (so it can appear but rarely ranks high), or
- **Exclude it** from temporal spreading but leave it available to other retrieval paths.

Either way, the algorithm **degrades gracefully** instead of blowing up.

## Temporal spreading activation in a hybrid retrieval stack

Temporal spreading activation is just one leg of Hindsight's [hybrid retrieval system](https://docs.hindsight.vectorize.io/recall). In practice we run it alongside semantic search, [BM25](https://en.wikipedia.org/wiki/Okapi_BM25), and graph-only traversal:

```python
semantic_result, bm25_result, graph_result, temporal_result = await asyncio.gather(
    run_semantic(),
    run_bm25(),
    run_graph(),
    run_temporal_with_extraction(),
)
```

The temporal branch does two things:

1. **Extracts a time window** from natural language ("last March", "two weeks before the outage").
2. **Runs temporal spreading activation** within that window.

Because it runs in parallel, slow date parsing or a deeper temporal walk doesn't block other retrieval strategies. At fusion time, we can treat the temporal scores as either:

- A separate "time relevance" channel to combine with other signals, or
- A **priority boost** when the user's query is explicitly time-framed.

## Performance and limitations

In production, temporal spreading activation typically touches **30–80 nodes** per query with our default budget of 100. Latency adds roughly **15–40ms** on top of the base semantic search, depending on graph density and how many hops the causal chains require.

Compared to naive temporal filtering:

- **Recall improves significantly** for "story" queries — the kind where the answer spans multiple events across weeks or months.
- **Precision stays high** because the semantic similarity gate prevents irrelevant nodes from entering the walk, even when causal chains extend outside the time window.
- **Cost is bounded** — the budget cap means worst-case latency is predictable regardless of graph size.

There are limitations. The approach struggles when:

- **The graph is very sparse** — if few causal or temporal links exist, the traversal has nothing to follow and falls back to the same results as a time filter.
- **Time references are ambiguous** — "around the time we launched" requires accurate date extraction upstream; if the window is wrong, the entry points are wrong.
- **Causal links are noisy** — if the graph has many weak or incorrect causal edges, the boosts can pull in irrelevant nodes. We mitigate this with the `weight >= 0.1` threshold and semantic gating, but it's not perfect.

## Why time-aware graph traversal works for AI agents

The goal wasn't to invent a fancy algorithm for its own sake; it was to make time-framed questions behave the way you intuitively expect:

- You don't just get "things in March" — you get the **lead-up and fallout** that make that period meaningful.
- **Causal chains stay attached** even if they step outside the exact window.
- **Performance stays bounded** and predictable because of batching and budgets.

Most importantly, the agent no longer treats time as a dumb filter. It treats it as a structural axis of the memory graph — one that shapes how it walks the graph and how it reconstructs the story of what happened.

---

*Hindsight is an AI agent memory system that gives your agents persistent, structured memory with hybrid retrieval. [Learn about the architecture](https://docs.hindsight.vectorize.io/recall), or [sign up for Hindsight Cloud](https://ui.hindsight.vectorize.io/signup).*
