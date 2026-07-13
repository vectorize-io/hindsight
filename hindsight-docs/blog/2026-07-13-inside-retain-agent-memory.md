---
title: "Inside retain(): What Actually Happens When Your Agent Remembers"
authors: [benfrank241]
slug: "2026/07/13/inside-retain-agent-memory"
date: 2026-07-13T12:00
tags: [hindsight, agent-memory, retain, observations, knowledge-graph, how-it-works]
description: "Calling retain() is one line of code. Underneath, a sentence becomes extracted facts, resolved entities, a knowledge graph, and consolidated observations. Here is the whole write path."
image: /img/blog/inside-retain-agent-memory.png
hide_table_of_contents: true
---

![What happens when your agent calls retain: extract, resolve, connect, consolidate](/img/blog/inside-retain-agent-memory.png)

You give your agent a memory by calling one function: `retain()`. You hand it a sentence, and from then on the agent seems to just know the thing. That simplicity hides the interesting part. `retain()` does not just save a string. It keeps the original text and, on top of it, turns what you said into structured, connected, evidence-grounded knowledge that keeps getting refined as more comes in.

Here is the whole write path, one stage at a time, using a single sentence: *"Alice joined Google last spring and was thrilled about the research opportunities."*

<!-- truncate -->

## TL;DR

- `retain()` **extracts facts** from your content, including the reasoning and feeling behind them, not just the literal statement.
- It also **keeps the original text** (chunked if long), so the verbatim source stays available alongside the extracted memory.
- It **recognizes and resolves entities**, so "Alice" and "Alice Chen" become one person.
- It **connects** every fact into a knowledge graph by entity, time, meaning, and cause.
- It **grounds each fact in time** twice: when the event happened, and when you learned it.
- Then it **consolidates** related facts into durable **observations** that carry their own evidence, as a separate background step after the facts are stored.

## Stage 1: extraction that captures meaning, not words

Most memory systems store what was said. Hindsight extracts what it *means*. From our one sentence it pulls apart several distinct facts:

- **The core facts:** Alice joined Google; it happened last spring.
- **The feeling and significance:** she was thrilled; it mattered to her.
- **The reasoning:** she chose it for the research opportunities.

That last two are the difference between a memory that can answer "Where does Alice work?" and one that can answer "*Why* did Alice join Google?" A transcript can do the first. Only extracted meaning can do the second.

Extraction also preserves narrative instead of shredding it into disconnected shards. "Bob suggested one name, Alice wanted something unique, they landed on a third" stays a coherent story, so a later search returns the context, not three fragments that no longer make sense apart.

Every extracted fact is classified as one of two kinds, by **who is speaking**:

- **experience**: the bank's own agent acting or observing. "I recommended Python to Alice."
- **world**: a fact about someone or something else. "Alice works at Google."

This matters more than it sounds. When a user says "I bought a Tesla," that is a `world` fact about the *user*, not the agent's own experience. Hindsight gets this right when you give the bank a real `name` (so it knows who "the agent" is) and describe the speaker in each item's `context` when you retain a transcript.

Extraction is additive, not a replacement. Alongside the facts it pulls out, Hindsight also stores the **original text itself**, chunked if it is long, so the verbatim source stays available. Every memory can be traced back to exactly what was said, and you can retrieve the raw passage when you need the source rather than the distilled fact.

## Stage 2: entities, recognized and resolved

As it extracts, Hindsight identifies the **entities** that matter: people, organizations, places, products, and concepts. Alice and Google both become tracked entities.

The important part is **resolution**. "Alice," "Alice Chen," and "Alice C." collapse into one person; "Bob" and "Robert Chen" unify through nickname resolution. When a common name is ambiguous, Hindsight uses co-occurrence to disambiguate: a new "Alice" who shows up alongside "Google" and "Stanford" is probably the Alice you already know.

The payoff is that "What do I know about Alice?" returns *everything*, no matter which spelling each fact used. (We went deep on this in [entity resolution](/blog/2026/06/29/entity-resolution-agent-memory).)

## Stage 3: building the knowledge graph

Facts are not stored as an isolated list. Each one is wired into a **knowledge graph** through four kinds of connection:

- **Entity connections** link every fact that mentions the same entity. → "Tell me everything about Alice."
- **Time connections** link facts close together in time. → "What else happened around then?"
- **Meaning connections** link semantically similar facts even when the words differ. → "Anything related to this?"
- **Causal connections** track cause and effect explicitly. → "Why did this happen?" ("Alice felt burned out" ← caused by ← "she worked 80-hour weeks.")

These edges are what let recall follow a thread instead of just matching a vector. The graph is the difference between fetching a fact and reasoning across a memory.

## Stage 4: two kinds of time

Every fact gets grounded on **two** temporal axes, and keeping them separate is what makes time-aware recall work:

- **When it happened.** "Alice got married in June 2024" occurred in June 2024, even if you are told in January 2025.
- **When you learned it.** The moment you retained it.

Track only one and you lose something. Without event time, "What did Alice do in 2024?" cannot find a marriage you were told about in 2025. Without learned time, recency ranking and "what happened before her marriage?" break. Hindsight keeps both, so historical queries and recency both work.

## Stage 5: consolidation into observations (a background step)

The final stage is not part of the same inline work. Consolidation runs as a separate background step, where an engine turns accumulating facts into **observations**: deduplicated, evidence-grounded beliefs.

Three raw facts:

- "Alice prefers Python"
- "Alice dislikes verbose code"
- "Alice recommends type hints"

These consolidate into one durable observation: *"Alice is a Python-focused developer who values readability and simplicity."*

Crucially, that observation is **not** a summary the model invented. Each one references the specific memories that support it, with quotes, and carries a proof count. New evidence **refines** it rather than overwriting it, and its history is preserved. When two observations drift into saying the same thing, Hindsight reconciles the near-duplicates so recall stays clean instead of returning three versions of one belief. ([The consolidation problem](/blog/2026/05/21/agent-memory-consolidation) covers this in depth.)

Because consolidation runs in the background rather than inline, the facts land first and the observations catch up a moment later, which is why a fact you retain now is folded into the bank's observations shortly after.

## Why the write path is the whole game

It is tempting to think memory is a storage problem: dump the transcript into a vector database, embed it, move on. That gives you a searchable log, not a memory. A log has no sense of who Alice is across ten conversations, no causal thread, no distinction between a one-off remark and a settled belief, and no way to tell a 2024 event from a 2025 mention. Hindsight keeps that raw text too, so the source is never lost, but it does not stop there.

The write path is where all of that gets built. By the time `retain()` and its background consolidation are done, one sentence has become structured facts, resolved entities, a graph of connections, dual-timestamped grounding, and a set of evidence-backed observations. That is what recall gets to search later, and it is why the answers come back as knowledge instead of quotes.

## Steering what gets kept

You are not stuck with the defaults. A **retain mission** narrows extraction to what your bank cares about ("always keep technical decisions and architectural trade-offs; ignore greetings and logistics"), injected into the extraction prompt without replacing the logic. And an **extraction mode** (`concise`, `verbose`, or `custom`) trades speed for richness. Both are set on the bank config, so the pipeline adapts to your domain instead of the other way around.

## Frequently asked questions

**Are observations ready the instant retain finishes?**
Not quite. Extraction produces the facts, and consolidation then folds them into observations as a separate background step. So the structured facts come first and the consolidated observations follow shortly after, rather than in the same moment.

**Are observations just LLM summaries?**
No. Each observation is grounded in specific source memories with exact quotes and a proof count, and it is refined as evidence changes rather than regenerated from scratch.

**What if the same person is named three different ways?**
Entity resolution unifies them into one entity, so everything you know about them is retrievable regardless of which name a given fact used.

**Do I have to structure my input?**
No. You pass natural conversation or documents. The structure (facts, entities, graph edges, observations) is what `retain()` produces from it.

## Further reading

- [Retain: how Hindsight stores memories](/developer/retain): the reference for the write path.
- [Observations](/developer/observations): how consolidation builds evidence-grounded beliefs.
- [The consolidation problem in agent memory](/blog/2026/05/21/agent-memory-consolidation): why deduplication into observations matters.
- [Entity resolution in agent memory](/blog/2026/06/29/entity-resolution-agent-memory): how one person stays one person.
