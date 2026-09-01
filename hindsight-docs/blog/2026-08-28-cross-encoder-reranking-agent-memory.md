---
title: "Cross-Encoder Reranking: The Last Stage of Agent Memory Recall"
authors: [benfrank241]
slug: "2026/08/28/cross-encoder-reranking-agent-memory"
date: 2026-08-28T12:00
tags: [hindsight, agent-memory, recall, reranking, cross-encoder, retrieval, deep-dive]
description: "Retrieval hands the reranker a few hundred plausible memories. What happens next decides what your agent actually sees. Inside Hindsight's cross-encoder stage: score normalization, multiplicative boosts, candidate budgets, and how to make recall fail open."
image: /img/blog/cross-encoder-reranking-agent-memory.png
hide_table_of_contents: true
---

![Cross-encoder reranking in Hindsight: fused candidates scored jointly against the query, then adjusted by recency, temporal proximity, and evidence strength](/img/blog/cross-encoder-reranking-agent-memory.png)

Retrieval is the part everyone talks about. Embeddings, BM25, graph traversal, time windows: pick your arms, fuse the rankings, ship it.

But fusion doesn't tell you what's *relevant*. It tells you what several independent systems each thought was plausible, blended into one list. The stage that decides what your agent actually reads is the one that runs last, and it's the one that gets the least attention.

This is a look inside that stage in [Hindsight](https://hindsight.vectorize.io) — the cross-encoder reranker. How it scores, why the secondary signals are multiplicative instead of additive, the subtle failure mode that turns a reranker into a date sort, and how to configure it so an unreachable model degrades instead of taking recall down with it.

<!-- truncate -->

## TL;DR

- **A cross-encoder reads the query and the memory together.** Embeddings are computed independently and compared after the fact; a cross-encoder scores the pair jointly, which is why it's far more accurate and why it can't be an index.
- Because it costs one model call per candidate, it runs **last, on a capped set** — `HINDSIGHT_API_RERANKER_MAX_CANDIDATES`, default `300`.
- Scores are normalized **conditionally**: calibrated `[0, 1]` relevance from hosted rerankers is used as-is, local logits go through a sigmoid. Rank normalization is deliberately avoided so a weak top hit stays weak.
- Recency, temporal proximity, and evidence strength are **multiplicative boosts**, capped at roughly **+21% / −19%** combined, so they nudge the ranking without overriding relevance.
- A passthrough reranker creates a trap: identical scores would make the boosts the *only* signal, turning recall into a pure recency sort. Hindsight seeds the base score from the fusion rank to prevent it.
- Ending the reranker chain with an `rrf` member makes recall **fail open** instead of returning a 500 when your reranker is down.
- Everything here is released in **v0.9.2**.

## Why fusion isn't enough

Every retrieval arm in Hindsight produces a ranking, and those rankings get fused. The problem is that fusion is a *voting* operation. It rewards a memory that several arms independently liked, which is a decent proxy for relevance and a bad substitute for it.

The deeper issue is structural. Semantic retrieval compares a query embedding to memory embeddings, and those embeddings were computed **independently** — the memory was embedded at write time, long before your query existed.

That's what makes vector search fast: you can index it. It's also what makes it approximate. The model never got to look at the query and the memory at the same time.

A cross-encoder does exactly that. It takes a `(query, document)` pair and runs both through the model together, so attention flows across the boundary between them. It can notice that "the outage" in your query refers to the same incident the memory calls "the 502s on checkout," which no independent pair of embeddings can represent.

The cost is that it's not indexable. There's no precomputed vector to look up, because the score doesn't exist until the query arrives. You pay one model forward pass per candidate. That single property dictates the entire design of the stage: it has to run last, on a small set, with a hard budget.

## What the model actually sees

Before scoring, each candidate is assembled into a text the model can read. Three things happen in `rerank()`.

The memory's text is used as the document. If the memory carries context, it's prefixed:

```
{context}: {text}
```

And if the memory has a start date, that date is prepended in **two formats at once**:

```
[Date: June 05, 2022 (2022-06-05)] {doc_text}
```

The redundancy is intentional. Cross-encoders are language models, and they handle a written-out date and an ISO date differently. Supplying both gives the model a better chance of grounding a temporal query against the document without a separate date-handling path.

This matters more than it looks. It means temporal relevance gets two independent chances to influence the result: once inside the model, through the date in the text, and once outside it, through the temporal boost described below.

## Normalization: the calibration problem

Different rerankers return different kinds of numbers. Hosted APIs like Cohere and Jina return a calibrated relevance score already in `[0, 1]`. Local cross-encoder models return raw logits, which can be any real number.

Hindsight branches on that:

- **Already in `[0, 1]`** — used as-is.
- **Logits** — passed through a sigmoid.

What it pointedly does *not* do is rank normalization, and the reason is worth stating plainly. If you normalize by rank, the top candidate always scores 1.0.

But "best of a bad set" is not the same as "good," and an agent that treats them identically will confidently cite a memory that barely matched. As the source puts it, a top candidate scoring `0.007` should stay low rather than be inflated to `1.0`. Preserving absolute confidence is what lets a caller downstream tell the difference between a strong hit and the least-bad option.

## The three boosts, and why they multiply

The cross-encoder score is the primary relevance signal, but it isn't the only thing that matters. A highly relevant memory from three years ago may still lose to a slightly less relevant one from yesterday. Hindsight folds in three secondary signals:

```
recency_boost     = 1 + recency_alpha     * (recency    - 0.5)
temporal_boost    = 1 + temporal_alpha    * (temporal   - 0.5)
proof_count_boost = 1 + proof_count_alpha * (proof_norm - 0.5)

combined_score = CE_normalized * recency_boost * temporal_boost * proof_count_boost
```

Each signal is mapped to `[0, 1]` with `0.5` as neutral, then turned into a multiplier centered on `1.0`.

| Signal | Alpha | Per-signal range | Neutral when |
|---|---|---|---|
| Recency | `0.2` | ±10% | The memory has no usable date |
| Temporal proximity | `0.2` | ±10% | The query isn't temporal |
| Proof count | `0.1` | ±5% | The fact isn't an observation |

Combined, the two 0.2 signals bound the adjustment at roughly `(1 + α/2)² ≈ +21%` and `(1 − α/2)² ≈ −19%`.

**Why multiplicative rather than additive?** Because cross-encoder score calibration varies by model. An additive bonus of `+0.1` is a rounding error against one model's score distribution and a landslide against another's. A multiplier is proportional by construction: the influence of the secondary signals stays the same *relative* to the base score, whatever scale that base score happens to be on. Swap the reranker and the tuning still holds.

The bounds are the other half of the design. Capping the combined effect near ±20% means these signals reorder candidates that were already close, and can't promote an irrelevant memory over a relevant one. Recency breaks ties; it doesn't win arguments.

### Recency

Recency uses the memory's **effective time**, falling back through `occurred_start`, then `mentioned_at`, then `occurred_end` — the same coalesce order retrieval uses.

That fallback exists because plenty of real memories legitimately lack a start date. A conversational fact or an ongoing state might carry only a `mentioned_at`, and without the fallback every one of them would sit at a flat neutral 0.5 and lose recency ordering entirely.

Three decay curves are available:

| Function | Shape | Default parameter |
|---|---|---|
| `linear` (default) | Straight line from 1.0 today to a floor of 0.1 | 365-day window |
| `exponential` | `0.5 ** (days_ago / halflife)`, smooth asymptote, no hard cutoff | 90-day half-life |
| `none` | Always neutral | — |

The exponential curve's half-life is the age at which a memory is exactly neutral: younger memories get boosted, older ones penalized, with no cliff. It suits banks where relevance genuinely decays with age. The linear default suits banks where a memory from last year is still worth something.

One deliberate edge case: **future-dated memories clamp to maximum freshness** rather than being penalized. A scheduled meeting next week shouldn't rank below one from last month because a naive age calculation went negative.

### Proof count

For observations — facts consolidated from repeated evidence — the number of supporting proofs is a real relevance signal. Something observed 40 times is more load-bearing than something observed once.

It's normalized on a logarithmic curve:

```
proof_norm = clamp(0.5 + log(proof_count) / 10.0, 0, 1)
```

A proof count of 1 lands exactly on neutral. Growth is logarithmic because the difference between 1 and 10 proofs is meaningful and the difference between 100 and 110 isn't. By a proof count of about 150 the curve clamps at 1.0, which is the documented maximum of +5%.

For `world`, `experience`, and `opinion` facts, which don't carry proof counts, the signal is neutral and the multiplier collapses to exactly 1.0.

## The passthrough trap

Here's the subtle one, and it's the kind of bug that only shows up in a specific deployment shape.

Some deployments run no cross-encoder at all, using an RRF passthrough instead. In that mode every candidate receives an **identical** cross-encoder score.

Now look at the formula again. If `CE_normalized` is the same constant for every candidate, it stops differentiating anything, and the multiplicative boosts become the *only* remaining ranking signal. Recall silently degrades into a pure recency sort — newest first, regardless of whether the memory has anything to do with the query.

The system would look like it was working. It would return results. They'd just be wrong in a way no error surfaces.

The fix is to seed the base score from the fusion rank instead, mapping rank onto `[0.1, 1.0]`:

```
cross_encoder_score_normalized = 1.0 - (0.9 * rank / (n - 1))
```

Now the boosts modulate a meaningful base ordering rather than replacing it, and the passthrough behaves like what it claims to be: fusion order, gently adjusted.

There's a nice engineering detail in how this is triggered. The caller passes an explicit `is_passthrough` flag rather than the code detecting "all scores are identical," because that heuristic is too fragile — a real cross-encoder can legitimately tie scores, especially on small or synthetic result sets, and inferring passthrough from ties would corrupt genuine reranks.

## The candidate budget

Because the cross-encoder is the dominant cost of a large recall, the candidate set is capped. `HINDSIGHT_API_RERANKER_MAX_CANDIDATES` defaults to `300`; fusion pre-filters everything beyond that.

A flat cap isn't always right, so the budget can scale with the recall's own budget level:

| Variable | Purpose | Default |
|---|---|---|
| `HINDSIGHT_API_RERANKER_MAX_CANDIDATES` | Flat cap per recall | `300` |
| `HINDSIGHT_API_RERANKER_MAX_CANDIDATES_LOW` | Override for `budget=low` | `0` (use flat) |
| `HINDSIGHT_API_RERANKER_MAX_CANDIDATES_MID` | Override for `budget=mid` | `0` (use flat) |
| `HINDSIGHT_API_RERANKER_MAX_CANDIDATES_HIGH` | Override for `budget=high` | `0` (use flat) |
| `HINDSIGHT_API_RECALL_MAX_CANDIDATES_PER_SOURCE` | Cap per retrieval arm, applied before the global cap | `0` (disabled) |

The per-source cap solves a specific failure: one over-expanding arm filling the entire reranker budget on its own, crowding out every other arm's best hits before the cross-encoder ever sees them.

There's also `HINDSIGHT_API_RECALL_STRATEGY_BOOSTS`, which favors named arms as a `strategy:level` list such as `graph:high` or `graph:high,bm25:low`. It's applied in **two** places: before the reranker cap, so favored candidates survive the budget, and again after reranking, to nudge them up the final order. Two applications are necessary because those stages operate on different score scales, which is also why the setting takes a named level rather than a raw number.

## Making recall fail open

By default, a reranker that's unreachable takes recall down with it. The stage was a hard dependency, so a timeout on a self-hosted model became a 500 on the request.

The fix is a **failover chain**. The unindexed config is the primary (member 0), and extra members are numbered contiguously from 1:

```bash
# Primary: a self-hosted reranker that isn't always up.
export HINDSIGHT_API_RERANKER_PROVIDER=tei
export HINDSIGHT_API_RERANKER_TEI_URL=http://workstation:8081

# Member 1: a hosted reranker to fall back on.
export HINDSIGHT_API_RERANKER_1_PROVIDER=cohere
export HINDSIGHT_API_RERANKER_1_COHERE_API_KEY=...

# Member 2: last resort — keep the fusion order rather than fail.
export HINDSIGHT_API_RERANKER_2_PROVIDER=rrf
```

Members are tried in order on timeout, connection error, HTTP error, or an unusable response. **Ending the chain with `rrf` is what makes recall fail open**: results come back in fusion order, exactly as if no reranker were configured, instead of the request failing. Without an `rrf` member, or with every member down, the error still surfaces — which is the existing default behavior.

Four things are worth knowing before you rely on it:

- **Each member is read in isolation.** It inherits nothing from the primary and nothing from the shared provider keys, so spell out every setting with its own index. Unset settings fall back to built-in defaults, not to the primary's values.
- **There's no circuit breaker.** Members are retried on every request, so a member that's down costs its full timeout each time before the next is tried. Keep an unreliable primary's timeouts short.
- **Scores aren't comparable across providers.** Some return calibrated relevance, others logits, so a request served by a fallback can score differently from one served by the primary. Failovers are logged at `WARNING` — worth alerting on, since the chain is otherwise silent.
- **Indexed members are server-level only.** They're credential fields, never returned by the bank-config API and not per-bank configurable.

A member that fails to initialize at startup is logged rather than fatal, and retried on the next request that reaches it.

## Choosing a provider

Hindsight ships thirteen reranker providers: `local`, `tei`, `cohere`, `openrouter`, `zeroentropy`, `siliconflow`, `alibaba`, `google`, `flashrank`, `litellm`, `litellm-sdk`, `jina-mlx`, and `rrf`. The default is `local`, running `cross-encoder/ms-marco-MiniLM-L-6-v2` in-process.

Two local knobs are off by default despite being free wins on the right hardware:

- **`HINDSIGHT_API_RERANKER_LOCAL_FP16`** — 27–36% faster on MPS and quality-identical. It's off by default only because some CPUs lack native FP16 support.
- **`HINDSIGHT_API_RERANKER_LOCAL_BUCKET_BATCHING`** — sorts pairs by token length before batching to cut padding waste. 36–54% faster across models, and quality-identical *by construction*, since it changes only how work is grouped.

And one that's off for a much better reason. Apple Silicon's MPS backend is opt-in via `HINDSIGHT_API_RERANKER_LOCAL_ALLOW_MPS` because **MPS caches a distinct kernel and allocator pool per input shape and never releases it**. Reranking is a variable-length workload by nature, so memory grows without bound — idle instances have been observed at around 20 GB. CUDA and XPU don't have this problem and are still auto-selected.

If you run TEI, size its pool deliberately. TEI reserves one slot per text in a rerank request and rejects the whole request once its pool is full, so it needs at least `TEI_MAX_CONCURRENT × TEI_BATCH_SIZE` slots — 1024 at Hindsight's defaults, against TEI's own default of 512.

## Turning it off

Reranking is a stage, not a requirement. `HINDSIGHT_API_ENABLE_RERANKING=false` returns the fusion ordering directly: faster, and less precise. Like the other recall stage toggles it's hierarchical, so one bank can skip it via the config API without changing how the rest of the deployment recalls.

That's a reasonable trade for a bank used as plain retrieval over uniform chunks, where the arms mostly agree and there's little for a cross-encoder to correct. It's a bad trade for a bank of consolidated observations about recurring entities, which is exactly where fusion order is least trustworthy.

## FAQ

**What's the difference between a bi-encoder and a cross-encoder?**
A bi-encoder embeds the query and the document separately and compares the two vectors, which means documents can be embedded ahead of time and indexed. A cross-encoder feeds the query and document through the model together and outputs a relevance score directly. It's substantially more accurate and can't be precomputed, so it's used to rerank a shortlist rather than to search.

**Does reranking replace hybrid search?**
No. Reranking reorders candidates; it can't retrieve something the arms never returned. If a memory doesn't survive retrieval and fusion, no reranker will rescue it. The two solve different problems — see [Knowledge Graphs vs. Vector Search for Agent Memory](/blog/2026/08/24/knowledge-graphs-vs-vector-search-agent-memory).

**How many candidates should I rerank?**
The default of 300 is a reasonable balance. Raising it improves recall depth at a roughly linear cost in cross-encoder calls, which dominate a large recall's latency. If you need cheap low-budget recalls and thorough high-budget ones, use the per-budget overrides rather than moving the flat cap.

**Why is my top result scoring so low?**
Because Hindsight preserves absolute confidence instead of rank-normalizing. A low top score is genuine information: it means nothing in the bank matched your query well. Treat it as a signal that the memory doesn't exist yet, not as a scaling artifact.

**Can I use my own reranker?**
Yes. Any Cohere-compatible `/rerank` endpoint works through the `cohere` provider with a custom base URL, which covers Azure AI Foundry, Jina, Voyage, and self-hosted BGE deployments. Self-hosted models also run through `tei` or `litellm`.

**Does the reranker see my bank ID?**
Only if you turn that on. `HINDSIGHT_API_RERANKER_SEND_BANK_AS_HEADER` adds an `X-Hindsight-Bank-Id` header to remote reranker requests, and it's off by default because it transmits the bank ID to a third party. Enable it only for endpoints you trust.

## Learn more

- [Recall vs. Reflect: Two Ways to Query Agent Memory](/blog/2026/07/24/recall-vs-reflect) — where reranking sits in the wider query path
- [Knowledge Graphs vs. Vector Search for Agent Memory](/blog/2026/08/24/knowledge-graphs-vs-vector-search-agent-memory) — why multiple retrieval arms exist in the first place
- [Inside retain(): How One Sentence Becomes Memory](/blog/2026/07/13/inside-retain-agent-memory) — how the facts being reranked got there
- [Mental Models: A Deep Dive](/blog/2026/06/05/mental-models-deep-dive) — what Hindsight builds on top of consolidated observations

Hindsight is open source. The reranking stage lives in `hindsight-api-slim/hindsight_api/engine/search/reranking.py`, and every setting above is documented in the [configuration reference](https://hindsight.vectorize.io/docs/developer/configuration).
