---
title: "Tuning Recall: Hindsight's Four Retrieval Arms, Explained"
authors: [benfrank241]
slug: "2026/08/27/tuning-recall-retrieval-arms"
date: 2026-08-27T12:00
tags: [hindsight, agent-memory, recall, retrieval, hybrid-search, reranking, configuration, performance]
description: "Every Hindsight recall runs four retrieval arms, fuses them, and reranks the result. Here's what each arm does, which knob controls it, and how to tune recall for the shape of your bank."
image: /img/blog/tuning-recall-retrieval-arms.png
hide_table_of_contents: true
---

![Hindsight's four recall retrieval arms: semantic, keyword, graph, and temporal, fused and reranked](/img/blog/tuning-recall-retrieval-arms.png)

When an agent calls `recall`, it looks like one query. It isn't. Hindsight runs **four independent retrieval arms** against the bank, fuses their rankings, and then reorders the survivors with a cross-encoder.

That design is why recall works well out of the box on banks of wildly different shapes. It's also why recall can feel slower or fuzzier than you expect on a bank whose content doesn't match the defaults, and why the fix is almost never "add more data."

This is the tuning guide: what each arm actually does, which knob controls it, and how to match the pipeline to the shape of your bank.

<!-- truncate -->

## TL;DR

- Recall runs four arms — **semantic**, **BM25 (keyword)**, **graph**, and **temporal** — fused with reciprocal rank fusion, then reranked by a cross-encoder.
- Semantic always runs. Graph, temporal, and the reranker have had per-bank on/off switches for a while, and as of the latest release **the keyword arm has one too**.
- Every arm has its own similarity threshold, deliberately calibrated independently. They are not interchangeable.
- On a large recall the **cross-encoder is the dominant cost**, not the arms. Measure before you start switching things off.
- The single cheapest win most deployments miss has nothing to do with the arms: it's `HINDSIGHT_API_QUERY_ANALYZER_LANGUAGES`.

## What actually happens during a recall

The pipeline, in order:

1. **Query analysis** — extracts temporal constraints from the query text (feeds the temporal arm).
2. **Four retrieval arms** run against the bank, each returning ranked candidates.
3. **Reciprocal rank fusion** merges the four rankings into one candidate pool.
4. **The reranker** scores the fused pool with a cross-encoder and produces the final order.

Each stage has a cost, and a bank whose content has no relational or temporal structure pays for arms it cannot use.

## Arm 1: semantic

An ANN scan over embeddings. This is the baseline retrieval, and it's the one arm with no off switch, because there's no recall without it.

The knob that matters is `HINDSIGHT_API_SEMANTIC_MIN_SIMILARITY` (default `0.3`): the minimum cosine similarity a candidate must reach to be returned at all.

Raise it and you get fewer, tighter results. Lower it and you widen the net at the cost of noise the reranker then has to sort out.

## Arm 2: BM25 (keyword)

Lexical retrieval over a text index, for the literal tokens embeddings represent poorly: identifiers, error codes, SKUs, function names.

Hindsight supports five BM25 backends. Two worth knowing: **pgroonga** handles multilingual and CJK content out of the box, and **ParadeDB's pg_search** is true BM25 and the only Citus-compatible option.

Two knobs:

- `HINDSIGHT_API_BM25_MIN_SCORE` (default `0`) gates out zero-score rows. This matters on backends like `vchord` whose operator ranks *every* document instead of pre-filtering to genuine term matches.
- `HINDSIGHT_API_ENABLE_TEXT_SEARCH` (default `true`) switches the arm off entirely, leaving pure vector search.

That last one is new. Semantic and BM25 share a single UNION query, so "off" had to mean the keyword half is never built rather than filtered away afterward. The gate is one line:

```python
tokens = tokenize_query(query_text) if enable_text_search else []
```

With no tokens, the arm skips its SQL, its query tokenization, its `pg_stats` term-selection round trip, and its bind parameters. It's read-path only: `search_vector` and its index are still maintained on write, so the flag is reversible without a reindex.

## Arm 3: graph

Traversal over the entities and links extracted from your facts. This is the arm that answers "what's connected to this?" rather than "what looks like this?"

The default algorithm is `link_expansion`: expansion from semantic seeds via entity co-occurrence, semantic kNN, and causal links, with a target latency under 100ms.

Knobs:

- `HINDSIGHT_API_GRAPH_SEED_MIN_SIMILARITY` (default `0.3`) — how similar a memory must be to *seed* a traversal. Independent of the main semantic threshold.
- `HINDSIGHT_API_LINK_EXPANSION_PER_ENTITY_LIMIT` (default `200`) — caps fanout per entity, which is what stops one hub entity from dominating.
- `HINDSIGHT_API_LINK_EXPANSION_TIMEOUT` (default `10` seconds).
- `HINDSIGHT_API_ENABLE_GRAPH_RETRIEVAL` (default `true`).

There's a related knob on the *write* side: `HINDSIGHT_API_SEMANTIC_LINK_MIN_SIMILARITY` (default `0.7`) controls how densely the semantic graph is built during retain. Changes to it are **not retroactive** — the new value applies to links created from that point on, and maintenance won't remove existing links below a raised threshold. To make an existing graph conform, rebuild it or re-ingest into a new bank.

## Arm 4: temporal

Date-aware retrieval, for "what did we decide last quarter?" It's fed by the query analysis stage, which extracts temporal constraints from the query text.

- `HINDSIGHT_API_TEMPORAL_SEMANTIC_MIN_SIMILARITY` (default `0.1`) — notably looser than the other gates, because temporal entry points are meant to cast wide and let the date constraint do the filtering.
- `HINDSIGHT_API_ENABLE_TEMPORAL_RETRIEVAL` (default `true`) — switching it off also skips the date-aware query analysis, since with no detected constraint there's nothing to filter on.

## The reranker

After RRF fusion, a cross-encoder rescores the candidate pool. This is the precision stage, and on a large recall **it's the dominant cost**.

- `HINDSIGHT_API_RERANKER_MAX_CANDIDATES` (default `300`) caps what gets reranked; RRF pre-filters the rest. Per-budget overrides exist for `low`, `mid`, and `high`.
- `HINDSIGHT_API_ENABLE_RERANKING` (default `true`) returns the RRF-fused ordering directly. Faster, less precise.

By default an unreachable reranker takes recall down with it. The **failover chain** fixes that: configure extra rerankers by index (`HINDSIGHT_API_RERANKER_1_PROVIDER`, and so on, contiguous from 1) and Hindsight tries them in order on timeout, connection error, HTTP error, or an unusable response. If you run a remote reranker in production, configure at least one fallback.

## The four toggles at a glance

| Setting | Environment variable | Turns off | Default |
|---|---|---|---|
| `enable_text_search` | `HINDSIGHT_API_ENABLE_TEXT_SEARCH` | The keyword arm, its tokenization and `pg_stats` lookup. Also drops the keyword arm from knowledge-page search. | `true` |
| `enable_temporal_retrieval` | `HINDSIGHT_API_ENABLE_TEMPORAL_RETRIEVAL` | The temporal arm and the date-aware query analysis feeding it. | `true` |
| `enable_graph_retrieval` | `HINDSIGHT_API_ENABLE_GRAPH_RETRIEVAL` | Entity and link traversal. | `true` |
| `enable_reranking` | `HINDSIGHT_API_ENABLE_RERANKING` | The cross-encoder; recall returns RRF order directly. | `true` |

All four are hierarchical: the environment variable sets the deployment default, and any bank overrides it through the config API or a bank template.

```bash
curl -X PUT "$HINDSIGHT_API_URL/v1/default/banks/my-bank" \
  -H "Authorization: Bearer $HINDSIGHT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"enable_text_search": false, "enable_graph_retrieval": false}'
```

Turning all four off reduces recall to a single vector query, which is the lowest-latency configuration Hindsight has.

## Breadth: the recall budget

Separately from *which* arms run, `budget` controls **how deep** each one digs. The recall request takes `low`, `mid`, or `high` (default `mid`), which maps to a `thinking_budget` used by every arm.

Two mapping functions:

- **`fixed`** (default) — `100` / `300` / `1000` items per retrieval method per fact type, independent of `max_tokens`.
- **`adaptive`** — a ratio of the request's `max_tokens` (`0.025` / `0.075` / `0.25`), clamped to `[20, 2000]`.

Use `adaptive` when callers vary `max_tokens` and you want retrieval breadth to scale with the requested output size. Use `fixed` when you want predictable cost per recall.

## Weighting: strategy boosts

If one arm is reliably the useful one for your bank, you don't have to switch the others off. `HINDSIGHT_API_RECALL_STRATEGY_BOOSTS` takes a comma-separated `strategy:level` list:

```bash
HINDSIGHT_API_RECALL_STRATEGY_BOOSTS=graph:high,bm25:low
```

Strategies are `semantic`, `bm25`, `graph`, `temporal`. Levels are `low` (mainly protects that source's candidates from being dropped before reranking), `medium`, and `high` (the source dominates the pool; only a strong direct match still outranks it). A strategy written without a level defaults to `medium`, and any strategy you omit keeps its normal weight.

The boost applies in two places — before the reranker cap, so favored candidates survive the budget, and after reranking, to nudge them up the final order.

This is usually the better first move than an off switch: it changes the ordering without giving up an arm's coverage entirely.

## Freshness

Reranking applies a recency adjustment, shaped by `HINDSIGHT_API_RECENCY_DECAY_FUNCTION`:

- **`linear`** (default) — fades from full freshness to a floor over `365` days.
- **`exponential`** — half-life based; neutral at `90` days, with a smooth fade rather than a cutoff.
- **`none`** — age never affects ranking.

For a bank of reference material that doesn't go stale, `none` is often more correct than the default.

## Tuning by bank shape

| Bank shape | Do this |
|---|---|
| Conversational memory, mixed content | Leave defaults alone |
| Code, logs, identifiers | Keep BM25 on; consider `bm25:medium` boost |
| Chunk-ingested plain retrieval | All four arms off + `retain_extraction_mode: chunks` |
| Reference docs that never go stale | `RECENCY_DECAY_FUNCTION=none` |
| Relationship-heavy (people, orgs, deals) | `graph:high` boost before touching anything else |
| Known single-language corpus | Set `QUERY_ANALYZER_LANGUAGES` (see below) |
| Latency-critical, precision-tolerant | Lower the reranker cap before disabling arms |

## Three cheap wins people miss

**1. Restrict the query analyzer's languages.** `HINDSIGHT_API_QUERY_ANALYZER_LANGUAGES` limits the locales `dateparser` considers. Locale auto-detection **dominates recall's CPU cost**, so restricting it to `en` (or `en,zh`) is often the single largest latency win available, and it costs you nothing if you know what language your queries are in. The tradeoff: explicit dates written in an unlisted locale will misparse rather than yield no constraint. Set it only when you actually know.

**2. Turn on the local reranker's free speedups.** `HINDSIGHT_API_RERANKER_LOCAL_BUCKET_BATCHING` sorts pairs by token length before batching to cut padding waste — 36 to 54% faster, and quality-identical by construction. On Apple Silicon, `HINDSIGHT_API_RERANKER_LOCAL_FP16` adds 27 to 36% at identical quality. Both default to off only to avoid regressions on hardware that doesn't support them.

**3. Configure a reranker fallback.** One line of config that converts "recall is down" into "recall is slightly slower."

## Pairing with the retain side

The recall toggles only remove work from the **read** path. If a bank is genuinely plain retrieval, configure ingestion to match or it keeps paying for LLM work that recall no longer uses:

- **`retain_extraction_mode: chunks`** skips LLM fact extraction and stores chunks as-is. It returns *before* any LLM queue or lock is acquired, so it removes the call rather than shortening it. That call is normally the dominant cost of ingestion.
- **`enable_observations: false`** skips consolidation, the other background LLM workload.

Configured together with the recall toggles, the bank behaves like a conventional vector store: chunks in, vector search out, no LLM on either path.

Which is worth pausing on. No extracted facts means no entities, no links, no temporal structure, and no mental models to reflect over. You've turned off the product. That's legitimate for a bank that really is plain retrieval, or to benchmark Hindsight against a baseline vector store on equal terms. It's not a general latency fix.

## A note on the thresholds

Five embedding-dependent gates — main semantic retrieval, graph seeds, temporal retrieval, semantic-link construction, and observation deduplication — are configured independently on purpose. They serve different precision/recall tradeoffs, and their defaults are calibrated for `BAAI/bge-small-en-v1.5`.

If you change embedding models, recalibrate all five. Cosine-similarity distributions shift between models even when both return normalized vectors, so a threshold that meant "closely related" under one model can mean something quite different under another.

## Frequently asked questions

### How many retrieval methods does Hindsight use?

Four: semantic (vector), BM25 (keyword), graph traversal, and temporal. Their rankings are fused with reciprocal rank fusion and then reranked by a cross-encoder.

### What is reciprocal rank fusion?

RRF merges several ranked lists into one by scoring each item from its *rank* in each list rather than its raw score. That's what lets Hindsight combine four arms whose scores live on completely different scales.

### Which recall setting should I change first?

Usually none of the arms. If latency is the problem, look at the reranker cap and `HINDSIGHT_API_QUERY_ANALYZER_LANGUAGES` first. If relevance is the problem, try a strategy boost before an off switch.

### Can I configure recall differently per bank?

Yes. All four toggles, the recall budget mapping, and the thresholds are hierarchical: environment variables set the deployment default and any bank overrides it through the config API or a bank template.

### Does disabling an arm delete any data?

No. The toggles are read-path only. Indexes and columns are still maintained on write, so every one of them is reversible without a reindex or re-ingest.

### Why does temporal retrieval use a much lower similarity threshold?

Temporal entry points are meant to cast a wide net and let the date constraint do the filtering, so `0.1` is deliberate rather than an oversight. The gates are tuned per task, not shared.

### What happens if my reranker goes down?

By default, recall fails with it. Configure a failover chain (`HINDSIGHT_API_RERANKER_<n>_PROVIDER`, contiguous from 1) and Hindsight falls through to the next member on timeout, connection error, HTTP error, or an unusable response.

### Is hybrid search always better than pure vector search?

Usually, but not always. Hybrid wins whenever queries contain literal tokens that embeddings represent poorly. It adds less when queries are purely conceptual, which is the case `enable_text_search` exists for.

## Learn more

- [Knowledge Graphs vs. Vector Search for Agent Memory](/blog/2026/08/24/knowledge-graphs-vs-vector-search-agent-memory) — why the graph and semantic arms both exist
- [Recall vs. Reflect](/blog/2026/07/24/recall-vs-reflect) — the two ways to read from a bank
- [Bank Strategy for Agent Memory](/blog/2026/07/16/bank-strategy-agent-memory) — deciding what becomes its own bank
- [How to Move Your Agent's Memory Off a Vector Database](/blog/2026/07/28/migrate-agent-memory-off-vector-database) — the migration path in
