---
title: "recall vs reflect: Search Your Agent's Memory, or Ask It"
authors: [benfrank241]
slug: "2026/07/24/recall-vs-reflect"
date: 2026-07-24T12:00
tags: [hindsight, agent-memory, recall, reflect, how-it-works]
description: "Hindsight gives your agent two ways to read its memory. recall retrieves the relevant facts (no LLM, instant). reflect reasons across them to synthesize an answer. When to use each."
image: /img/blog/recall-vs-reflect.png
hide_table_of_contents: true
---

![recall retrieves the relevant facts, reflect reasons across them to synthesize an answer](/img/blog/recall-vs-reflect.png)

Your agent writes to agent memory with `retain`. But there are two ways to read it back, and they are constantly confused because both take a query and return something useful. `recall` and `reflect` are not two names for the same thing. One is a search engine. The other is an analyst. Pick the wrong one and you either pay for an LLM call you did not need, or you get a pile of raw facts when you wanted an answer.

The cleanest way to tell them apart is a question, and it happens to be exactly how the code describes them: **use `recall` for "what did I say about X?" and `reflect` for "what should I do about X?"**

<!-- truncate -->

## TL;DR

- **`recall` retrieves.** It finds the memories relevant to a query and returns them as a ranked list of facts. No LLM. Sub-second. Cheap.
- **`reflect` reasons.** It is an agent that searches your memory across several levels and uses an LLM to synthesize an answer. Slower. Costs a model call.
- Ask yourself: do you want the *facts*, or a *conclusion drawn from the facts*? That is the whole decision.
- They are not rivals. `reflect` uses `recall` under the hood as its ground truth.

## recall: a search engine for your memory

`recall` is pure retrieval. You hand it a query, and it returns the memories that matter, ranked. What it does not do is think about them.

Under the hood it is more than a vector search. For each fact type, Hindsight runs several retrieval strategies in parallel: semantic vector search, BM25 keyword search, and graph activation over linked entities, plus a temporal pass when the query has a time element. It fuses those with Reciprocal Rank Fusion, reranks the survivors with a cross-encoder (local by default), and trims to your token budget. The result is a ranked list of facts, each with its type, timestamps, and entities.

The important thing for choosing between the two operations: **`recall` never calls an LLM.** The whole pipeline is embeddings plus a reranker, both local by default. That is why it comes back in well under a second and costs almost nothing. It is the operation you can afford to run on every single turn.

Two knobs shape it:

- **`budget`** (`low`, `mid`, `high`) controls how deep the search goes, roughly 100, 300, or 1000 units of graph traversal by default. Higher budget finds more, at slightly more work.
- **`max_tokens`** caps how much comes back. Hindsight returns facts until the budget is hit and stops before overflowing it.

You also get query-time filters: `tags` and `tags_match` to scope by label, and a `query_timestamp` to anchor relative dates like "last week" and bias recency. The full parameter list lives in the [recall API reference](/developer/api/recall).

Reach for `recall` when you want to load relevant context into a prompt before the model answers, or when you want to show a user the actual stored facts. It answers "what do I know about this?" and hands you the raw material.

## reflect: an analyst that reads your memory for you

`reflect` is a different animal. It does not just retrieve, it *reasons*. Give it a question and it runs an agentic loop: it decides what to look for, searches your memory across several levels, and then an LLM writes a synthesized answer grounded in what it found.

The retrieval it does is hierarchical, best source first:

1. **Mental models**: user-curated, self-refreshing summaries of topics your agent asks about often.
2. **Observations**: consolidated, evidence-backed knowledge distilled from many raw facts.
3. **Raw facts**: pulled via `recall`, as ground truth to verify against when the higher levels are stale.

So `reflect` is not a competitor to `recall`. It is built on top of it. `recall` is the bottom layer that keeps `reflect` honest.

Because there is an LLM in the loop, `reflect` can do things `recall` structurally cannot:

- **Synthesize.** It returns a written answer, not a list. "What should I do about X," "summarize what we learned this month," "what is this user's overall preference."
- **Structured output.** Pass a `response_schema` and it returns a validated object matching your JSON schema, so you can feed the conclusion straight into code.
- **Cited sources.** Set `include_based_on` and it returns a `based_on` block naming the exact memories, mental models, and directives it actually used. Those citations are validated against what was retrieved, so it cannot invent a source.

Its knobs mean something different too. `budget` here controls how many iterations the agent gets, so how thoroughly it explores before answering (`high` is deeper, and costs more). `max_tokens` limits only the length of the final written answer, not how much it retrieves along the way.

The tradeoff is the point: `reflect` makes one or more model calls, so it is slower and costs real tokens. Hindsight caches the stable prompt prefix across the loop to soften that, but a reflection is fundamentally more expensive than a lookup. You do not run it on every keystroke. Every option is documented in the [reflect API reference](/developer/api/reflect).

## The difference in one table

| | `recall` | `reflect` |
|---|---|---|
| Job | Retrieve relevant memories | Reason across memories to answer |
| LLM involved? | No, embeddings + reranker only | Yes, agentic loop with an LLM |
| Returns | A ranked list of facts | A synthesized answer (optionally structured) |
| Speed and cost | Sub-second, near-free | Slower, costs a model call |
| `budget` controls | Search depth | How thoroughly the agent explores |
| `max_tokens` controls | Size of the returned facts | Length of the final answer |
| Cited sources | The facts *are* the result | `include_based_on` returns validated sources |
| Best for | "What did I say about X?" | "What should I do about X?" |

## How to actually choose

The decision is almost always this: **do you want the facts, or a conclusion drawn from the facts?**

Use `recall` when:

- You are assembling context to put in front of the model before it answers. This is the common case, and it should be cheap and frequent.
- You want to display the raw memories to a user, or feed exact facts into deterministic logic.
- Latency and cost matter, which on a per-turn hot path they always do.

Use `reflect` when:

- You want an answer, not evidence. A summary, an analysis, a recommendation, a briefing.
- You need the output shaped for code, via `response_schema`.
- You want cited, verifiable reasoning across the whole memory, not just the top matching facts.
- The call is occasional, not on every turn. A daily digest, an onboarding summary, a "what do we know about this account" panel.

A healthy agent uses both. `recall` grounds ordinary turns, constantly and cheaply. `reflect` steps in when you actually need synthesis, and it leans on `recall` to stay grounded when it does. Search when you want the facts. Ask when you want the answer.

## Frequently asked questions

**Does `recall` use an LLM?**
No. Retrieval is embeddings plus a cross-encoder reranker, both local by default, which is why it returns in well under a second and costs almost nothing.

**Is `reflect` slower than `recall`?**
Yes. `reflect` runs an agentic loop with one or more LLM calls, so it is slower and costs real tokens. `recall` is a lookup. Use `reflect` when you need synthesis, not on every turn.

**Can `reflect` return structured output?**
Yes. Pass a `response_schema` and it returns an object validated against your JSON schema, ready to feed into code.

**Does `reflect` cite its sources?**
Yes. Set `include_based_on` and it returns the exact memories, mental models, and directives it used, validated against what was actually retrieved so it cannot invent a citation.

## Further reading

- [Inside retain()](/blog/2026/07/13/inside-retain-agent-memory): how those facts get written and consolidated in the first place.
- [One bank or many?](/blog/2026/07/16/bank-strategy-agent-memory): the boundary that both recall and reflect operate within.
- [Your 1M-token context window is not memory](/blog/2026/07/22/context-window-is-not-memory): why you retrieve a small slice instead of pasting everything.
