---
title: "Freshness-Aware Memory: Knowing When a Belief Has Gone Stale"
authors: [benfrank241]
slug: "2026/06/17/freshness-aware-memory"
date: 2026-06-17T12:00
tags: [hindsight, memory, observations, freshness, reflect, agents]
description: "Most agent memory treats every stored fact as equally true forever. Hindsight computes a freshness trend for each belief from its evidence timestamps, tracks how far consolidation lags behind new memories, and uses both to decide when to trust a belief and when to verify it."
image: /img/blog/freshness-aware-memory.png
hide_table_of_contents: true
---

![Freshness-Aware Memory in Hindsight](/img/blog/freshness-aware-memory.png)

Most agent memory has a silent failure mode: it treats everything it ever stored as equally true, forever. A preference the user stated once in February outranks nothing. A decision they reversed last week still surfaces with full confidence. The retrieval layer returns the closest match by similarity and says nothing about whether that match still reflects reality.

That's fine for a demo and dangerous in production. An agent that confidently recommends the framework you abandoned six months ago isn't remembering, it's quoting a corpse.

This post is about how Hindsight addresses that: **freshness-aware memory**. Beliefs carry a computed sense of how current they are, and the reflect agent uses that signal to decide when to trust a belief outright and when to go verify it against the raw facts.

## TL;DR

<!-- truncate -->

- Hindsight consolidates raw facts into **observations**: deduplicated, evidence-grounded beliefs. Each observation knows which memories support it and when those memories were created.
- Every observation carries a **freshness trend**, computed from its evidence timestamps (not guessed by an LLM): `new`, `strengthening`, `stable`, `weakening`, or `stale`.
- Separately, the reflect agent gets a **consolidation-lag signal**: how many retained memories haven't been folded into observations yet, surfaced as `up_to_date`, `slightly_stale`, or `stale`.
- The reflect loop uses both. A stale belief, or a bank whose observations lag reality, triggers a drop to raw-fact `recall()` to verify before answering.
- The net effect: the agent trusts current beliefs directly and double-checks the ones that might be out of date, instead of treating all memory as equally reliable.

## Two Things "Stale" Can Mean

There are two independent ways a belief can be out of date, and Hindsight tracks them separately because they have different fixes.

1. **The belief itself is aging.** The evidence behind it is all old. Nobody has reinforced it lately. It might still be true, or the world might have moved on. This is a property of the observation, measured over time.
2. **The bank hasn't caught up.** You've retained new memories, but consolidation hasn't folded them into observations yet, so the observations you'd recall right now are a slightly old snapshot of what the bank actually knows. This is a property of the bank at this moment.

The first is about the *content* of a belief. The second is about the *pipeline* behind it. Hindsight computes both.

## Layer 1: The Per-Belief Freshness Trend

When Hindsight builds an observation, it attaches every supporting piece of evidence with the timestamp of the source memory. The trend is then computed directly from the distribution of those timestamps. It is not an LLM judgment call; it's arithmetic over the evidence dates, so it's deterministic and explainable.

There are five trends:

| Trend | What it means |
| --- | --- |
| `new` | All the supporting evidence is recent. A freshly formed belief. |
| `strengthening` | More, denser evidence lately than before. The belief is gaining support. |
| `stable` | Evidence is spread across time and continues to the present. A durable belief. |
| `weakening` | Evidence is mostly old and has gone sparse recently. Support is fading. |
| `stale` | No recent evidence at all. The belief may no longer apply. |

### How it's computed

The trend function splits an observation's evidence into time buckets relative to now, using two windows: a **recent** window (30 days by default) and an **old** boundary (90 days by default).

- Evidence newer than the recent cutoff is **recent**.
- Evidence older than the old cutoff is **old**.
- Everything in between is **middle**.

From there the logic is simple:

- **No recent evidence** → `stale`. Whatever this belief was built on, nothing has reinforced it lately.
- **Only recent evidence** (nothing old or middle) → `new`. The belief just formed.
- **Otherwise**, compare densities. Recent density is recent evidence per day in the recent window; older density is the rest of the evidence spread over the older period. The ratio decides it:
  - ratio above 1.5 → `strengthening`
  - ratio below 0.5 → `weakening`
  - in between → `stable`

Comparing *density* rather than raw counts matters: the recent window is short and the older period is long, so a belief with two mentions last week and three mentions over the prior two months reads as strengthening, not weakening, because the recent activity is denser.

### A belief over time

Picture a bank that's been learning about a user's stack. One observation: *"Prefers Postgres for primary storage."*

- **Week 1:** three facts in a few days all point at Postgres. Trend: `new`.
- **Month 2:** steady mentions keep arriving as they ship features on it. Trend: `stable`.
- **Month 5:** the team migrates to a managed service and stops talking about Postgres internals. New evidence dries up. Trend drifts to `weakening`, then `stale`.

Nobody wrote a rule that said "expire this after N days." The trend fell out of the evidence drying up. When the agent later recalls this belief, the `stale` marker is the cue that it should confirm against current facts before recommending a Postgres-specific approach.

## Layer 2: Consolidation Lag

Consolidation runs in the background after you retain. That's what keeps `retain()` fast: facts land immediately, and the engine folds them into observations shortly after. But it means there's a window where the observations in a bank are a slightly old view of what the bank has actually been told.

When the reflect agent searches observations, Hindsight tells it how far behind consolidation is, based on how many retained memories are still waiting to be consolidated:

| Pending memories | Signal |
| --- | --- |
| 0 | `up_to_date` |
| under 10 | `slightly_stale` |
| 10 or more | `stale` |

This is a different question from the per-belief trend. An observation can be perfectly `stable` on its own evidence while the bank as a whole is `stale` because a batch of fresh retains hasn't been processed yet. The agent needs both signals: one says "this belief is old," the other says "there may be newer information that hasn't become a belief yet."

## How Reflect Uses Freshness

[Reflect](/developer/reflect) answers questions with a hierarchical retrieval strategy, cheapest and most-synthesized first:

1. **[Mental models](/developer/api/mental-models)**: curated, high-level summaries.
2. **[Observations](/developer/observations)**: consolidated beliefs, with their freshness trend.
3. **[Raw facts](/developer/api/recall)**: ground truth, retrieved with `recall()`.

Freshness is what governs movement *down* that hierarchy. The agent's instructions are explicit about it: if the consolidated layer it's reading is marked stale, it does not stop there. It also calls `recall()` to verify against raw facts before committing to an answer. A fresh, relevant observation can answer a question directly. A stale one that's central to the answer gets checked against the underlying memories first.

That's the whole point of computing freshness: it's not a label for a dashboard, it's a routing decision. The agent spends its retrieval budget verifying exactly the beliefs that might be wrong, and trusts the ones that are clearly current.

## Why This Matters

Without freshness, every memory system eventually does this:

- **Recommends the abandoned choice.** The user moved from React to Vue. A flat memory store still has "loves React" sitting there with high similarity, and the agent suggests a React library.
- **Treats a one-off as a standing preference.** Something said once, months ago, never reinforced, surfaces with the same weight as a daily habit.
- **Misses that the picture changed.** New facts contradict an old belief, but the old belief still gets retrieved because nothing downgraded it.

Freshness-aware memory turns "what's the closest match?" into "what's the closest match, and should I still believe it?" The agent gets to reason about currency, not just relevance.

It also pairs with how Hindsight handles contradictions during [consolidation](https://hindsight.vectorize.io/blog/2026/05/21/agent-memory-consolidation): when a new fact reverses an old one, the observation is rewritten to capture the change rather than overwritten. Freshness handles the quieter case where nothing contradicts the belief, it just stops being reinforced and slowly ages out of relevance.

## Recap

| | Flat memory store | Freshness-aware memory |
| --- | --- | --- |
| Treats old and new facts | Equally | Trend per belief, computed from evidence dates |
| Knows a belief is fading | No | `weakening` / `stale` trend |
| Knows the index lags new writes | No | Consolidation-lag signal |
| Verifies before trusting | No | Reflect drops to raw facts when stale |
| Trend source | n/a | Arithmetic over timestamps, not an LLM guess |

## Next Steps

- **Observations and trends:** [How consolidation works](/developer/observations)
- **Reflect:** [The agentic retrieval loop](/developer/reflect)
- **Recall:** [Retrieving raw facts](/developer/api/recall)
- **Related reading:** [The Consolidation Problem in Agent Memory](https://hindsight.vectorize.io/blog/2026/05/21/agent-memory-consolidation)
- **Try it:** [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) or [self-host with one Docker command](https://hindsight.vectorize.io/developer/installation)
