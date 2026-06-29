---
title: "Entity Resolution in Agent Memory: One Person, Many Names"
authors: [benfrank241]
slug: "2026/06/26/entity-resolution-agent-memory"
date: 2026-06-26T12:00
tags: [hindsight, entity-resolution, agent-memory, memory, retrieval, knowledge-graph, deep-dive]
description: "How agent memory decides that Sarah, Sarah Chen, and she are one person, and why Hindsight resolves entities with a co-occurrence graph, not embeddings."
image: /img/blog/entity-resolution-agent-memory.png
hide_table_of_contents: true
---

![Entity Resolution in Agent Memory with Hindsight](/img/blog/entity-resolution-agent-memory.png)

Most agent memory systems reach for an embedding model or an LLM the moment they have to decide whether two mentions point at the same person. Hindsight doesn't. It resolves entities with three cheap, inspectable signals: how similar the names are, which other entities they show up next to, and how recently each was seen.

That choice sounds almost retro for 2026. It turns out to be the right one for memory, where a wrong merge is worse than a missed one, and where you want to be able to explain why the system thinks "Sarah" and "the new PM" are the same record.

This post is a walkthrough of how [agent memory](https://vectorize.io/what-is-agent-memory) does entity resolution, using Hindsight's implementation as a concrete reference. If you've ever wondered how a memory layer keeps "Sarah," "Sarah Chen," and "she" from fragmenting into three disconnected histories, this is the mechanism.

<!-- truncate -->

## TL;DR

- **Entity resolution** is deciding when two different mentions ("Sarah", "Sarah Chen", "she") refer to the same real-world entity, and linking them.
- Hindsight extracts entities with an LLM during fact extraction, but **resolves** them without embeddings or LLM judgment.
- Resolution is a weighted score over three signals: **name similarity** (0.5), **co-occurrence overlap** (0.3), and **temporal recency** (0.2). Clear `0.6` and the mention resolves to an existing entity; otherwise a new one is created.
- A **co-occurrence graph** (`entity_cooccurrences`) is the differentiator: who an entity appears next to disambiguates two people with the same name.
- The system is **deliberately conservative about merging** and leans on the graph at retrieval time instead of collapsing records it isn't sure about.
- You can make resolution reliable up front: pass a `context` string at retain time, or declare a known cast as **enum entity labels** so mentions map to a fixed vocabulary by exact match.

---

## What Entity Resolution Actually Means

Three terms get used interchangeably and shouldn't be. **Entity extraction** is pulling "Sarah" out of a sentence. **Entity resolution** is deciding that this "Sarah" is the same Sarah you stored last week. **Entity-based retrieval** is using that resolved identity to pull back everything related when the agent asks.

Memory needs all three, but resolution is the hard one. Extraction is a well-understood LLM task. Retrieval is a database join. Resolution is the judgment call in the middle, and it's where a memory layer quietly succeeds or fails.

Get it wrong in one direction and you over-merge: two different people named Alex collapse into one contradictory record, and the agent confidently tells a customer about an internal sprint. Get it wrong in the other direction and you under-merge: "Sarah" and "Sarah Chen" stay separate, so a recall about Sarah misses half her history. The whole job is staying out of both ditches.

If you're evaluating memory systems on more than retrieval scores, resolution behavior is worth probing directly. It's the part that decides whether your agent's memory of a person is coherent or fractured. (For the broader landscape, the [comparison of all major frameworks](https://vectorize.io/articles/best-ai-agent-memory-systems) is a good map.)

## The Obvious Approach, and Why Hindsight Skips It

The 2026-default instinct is to embed every entity mention and merge by cosine similarity, or to ask an LLM "are these the same person?" Both work, and both have real strengths: embeddings catch semantic equivalence that string matching misses, and an LLM can reason about context a score can't.

They also have failure modes that matter more in memory than in search. Embedding similarity is a black box: when it wrongly merges "Apple the customer" with "Apple the fruit reference," you can't easily see why, and you can't tune it without re-embedding. An LLM call per entity per fact is slow and non-deterministic, and it will, occasionally and unpredictably, decide two strangers are the same person.

Hindsight optimizes for a different property: **a wrong merge should be rare, cheap to debug, and never silent.** So resolution runs on signals an engineer can read off directly. The trade-off is honest, embeddings would catch a few equivalences this approach misses, but the misses are safe (a duplicate record), while the avoided errors are dangerous (a corrupted one).

## The Three Signals

When a new fact comes in, its extracted entities each get matched against existing entities in the same memory bank. For every candidate, Hindsight computes a weighted score:

```python
score  = name_similarity      * 0.5   # difflib SequenceMatcher, 0–1
score += cooccurrence_overlap * 0.3   # shared neighbors in the graph
score += temporal_proximity   * 0.2   # recency, within a 7-day window

if best_score > 0.6:
    resolve_to(best_candidate)        # same entity
else:
    create_new_entity()               # mint a fresh record
```

**Name similarity (weight 0.5)** uses Python's [`difflib.SequenceMatcher`](https://docs.python.org/3/library/difflib.html), a plain string-ratio comparison. "Sarah" against "Sarah" scores 1.0; "Sarah" against "Sarah Chen" lands near 0.7. Candidates themselves are fetched with PostgreSQL's [`pg_trgm`](https://www.postgresql.org/docs/current/pgtrgm.html) trigram index, so a large bank returns a handful of fuzzy matches per mention instead of a full scan.

**Co-occurrence overlap (weight 0.3)** is the interesting one, and the next section is entirely about it.

**Temporal proximity (weight 0.2)** nudges resolution toward entities seen recently. If a matching name was last seen two days ago, it earns most of the temporal weight; if it was last seen six months ago, it earns almost none. People you're actively talking about are more likely to be the people you mean.

The `0.6` threshold is the conservatism dial. A mention has to clear a real bar across these signals to merge into an existing record. When it doesn't, Hindsight creates a new entity rather than guessing, because a duplicate is recoverable and a bad merge is not.

## The Co-occurrence Graph Does the Disambiguation

Name similarity alone can't tell two people named Alex apart. The graph can.

Hindsight keeps an `entity_cooccurrences` table: every time two entities appear in the same fact, their pair count goes up. Over a few weeks of conversations, this builds a map of who shows up next to whom. That map is what carries the disambiguation load.

Here's the scenario that makes it concrete. Your bank has two Alexes. One is **Alex on the frontend team**, who co-occurs with "React," "Sprint 14," and "the staging deploy." The other is **Alex at Acme**, a customer, who co-occurs with "the renewal," "the SOC 2 review," and "Acme." Same name, two completely different neighborhoods in the graph.

A new fact lands: "Alex flagged a blocker on the Acme renewal." The mention "Alex" pulls both candidates by name. Name similarity is a tie, 1.0 for each. But the fact also mentions "Acme" and "the renewal," and the co-occurrence overlap with customer-Alex is high while the overlap with engineer-Alex is zero. The 0.3 graph weight breaks the tie, and the fact attaches to the right Alex.

No embedding could have done that from the names alone, because the names are identical. The signal that resolves it isn't what the entity *is*, it's what it's *near*. That's the part most "just embed it" designs leave on the table.

## How Resolved Entities Power Retrieval

Resolution would be academic if it didn't change what the agent recalls. It powers entity-link expansion, the entity arm of Hindsight's parallel [hybrid retrieval](/blog/2026/03/27/parallel-hybrid-search).

When a query surfaces a set of seed facts, Hindsight collects the entities in those facts and follows them out to every other fact that mentions the same entities, scoring candidates by how many entities they share with the seeds. Ask "what's the status of the Acme renewal," land on a few seed facts, and the expansion pulls in every other fact touching Acme-Alex, the renewal, and the SOC 2 review, even ones that never used the word "status."

This is why resolution quality compounds. Each correctly resolved mention thickens the links between facts, so retrieval reaches more of the relevant history with less prompting. It also runs alongside semantic, causal, and [time-aware](/blog/2026/03/12/spreading-activation-memory-graphs) links rather than replacing them, so a missed entity match is often covered by another strategy.

## How to Make Sure Entities Resolve

The resolver is conservative by design, which means it will sometimes leave two records separate when you'd rather it didn't. When resolution accuracy matters, you have two levers, and both act *before* the scoring runs, at extraction time. They're far more reliable than hoping a name similarity clears 0.6.

### Give the extractor context

Every retain call takes an optional `context` string alongside the content. It's woven directly into the extraction prompt, and it takes precedence over the model's defaults when attributing who said or did what.

This matters most for transcripts and multi-party text, where "I," "she," and "the customer" are ambiguous on their own. Tell the extractor who is in the room and the first-person statements get attributed to the right person instead of defaulting to the agent. The entities come out already pointed at the people you named, so resolution has far less to guess at.

```python
client.retain(
    content=transcript,
    context="Support call between agent Maria Lopez and customer John Doe (Acme).",
)
```

### Use entity labels in enum mode

If you already know the full cast, the people on a project, a product catalog, a support taxonomy, don't make the resolver rediscover it. Declare those values as an **entity label** in the bank's `entity_labels` config, and extraction is constrained to map every mention onto one of them.

Under the hood, a `value`-type label compiles to a Pydantic `Literal[...]` of your allowed values, so the model can't return anything off-list. And label entities resolve by **exact match**, not fuzzy scoring. "Sarah," "Sarah Chen," and "she" all collapse onto the one canonical value you defined, with the 0.6 threshold out of the picture entirely.

```json
{
  "entity_labels": {
    "attributes": [
      {
        "key": "person",
        "type": "value",
        "values": [
          { "value": "sarah-chen", "description": "Eng lead, frontend" },
          { "value": "alex-kim",   "description": "PM, growth" }
        ]
      }
    ]
  }
}
```

Enum mode is the most reliable resolution lever Hindsight offers. The catch is that it only works when the set is known and bounded: ideal for a fixed team or a finite catalog, wrong for an open-ended cast you discover as you go. For the open-ended case, the `context` field plus the co-occurrence graph is the right tool. (If you've already identified entities upstream, you can also pass them directly via the `entities` field on a retain call.)

## The Data Model, Briefly

Three tables carry the whole system, and they're worth seeing because they're refreshingly boring:

- **`entities`** holds one row per canonical entity: `canonical_name`, `mention_count`, `first_seen`, `last_seen`, and a `UNIQUE (bank_id, LOWER(canonical_name))` constraint that does case-insensitive dedup at the database level.
- **`unit_entities`** is the junction table mapping facts to the entities they mention, many-to-many.
- **`entity_cooccurrences`** is the graph: `(entity_id_1, entity_id_2, cooccurrence_count, last_cooccurred)`, with a `CHECK (entity_id_1 < entity_id_2)` so each pair is stored once.

No external graph database, no separate vector store for entities. The storage layer is the same embedded PostgreSQL that holds everything else, which keeps the self-hosting story to one Docker command.

## The Engineering Honesty Section

The parts that don't make it into a marketing diagram are where the real work lives.

**Labels resolve by exact match, on purpose.** Controlled-vocabulary entities like `pedagogy:scaffolding` skip fuzzy matching entirely, because trigram similarity would happily merge two distinct label values and quietly destroy a taxonomy. That edge case came from a real bug.

**Unicode lowercasing is a trap.** Python's `str.lower()` and PostgreSQL's `LOWER()` disagree on the Turkish dotted İ: Python produces a two-character result, Postgres a one-character one. If the application lowercases on one side and the database on the other, "İstanbul" resolves against itself and fails. Hindsight lets the database do the lowercasing in comparisons so both sides agree.

**Concurrent retains race.** Two workers can try to create the same entity at once. New entities go in with `INSERT ... ON CONFLICT DO NOTHING`, and a follow-up `SELECT` finds whichever row won, so a race produces one entity, not two or an error. Mention-count and co-occurrence updates are deferred and flushed in a fixed lock order to avoid deadlocks.

## What It Doesn't Do

Being honest about the boundaries matters more than the feature list.

Hindsight does not retroactively merge canonical entities. If "Sarah" and "Sarah Chen" end up as two records, a later co-mention strengthens the graph link between them, but it doesn't collapse them into one. There's also no entity-splitting operation: if two real people were wrongly merged early, the system has no built-in way to tear them apart. And there's no alias table, identity is emergent from names plus graph structure, not a curated mapping.

These are deliberate choices, not oversights. Retroactive merges and splits are exactly the operations that go catastrophically wrong in bulk, and the conservative path, prefer a duplicate, lean on the graph at read time, keeps the failure modes boring. If your use case needs aggressive canonicalization, that's a fair reason to weigh other systems; this one optimizes for not corrupting memory.

## How to Look at It Yourself

If you're running Hindsight, the entity tables are queryable directly: `mention_count` shows you which entities dominate a bank, and `entity_cooccurrences` shows you the neighborhoods. It's a good way to sanity-check what your agent actually thinks the world looks like. You can [self-host the whole thing](https://hindsight.vectorize.io) and inspect the graph, or start on the cloud free tier.

## Frequently Asked Questions

**Does Hindsight use embeddings for entity resolution?**
No. Embeddings power semantic retrieval elsewhere in the system, but entity resolution itself uses string similarity, a co-occurrence graph, and temporal recency. The goal is debuggable, conservative merging rather than maximum recall.

**How does it tell two people with the same name apart?**
By their neighbors. The co-occurrence graph records which entities appear together, so "Alex" near "Acme" resolves to a different record than "Alex" near "Sprint 14," even though the names are identical.

**What happens when it isn't sure?**
It creates a new entity. Resolution only merges when a weighted score clears `0.6`. A duplicate record is cheap to live with; a wrong merge corrupts the agent's memory of a person, so the system errs toward the safe mistake.

**How do I make sure the right entities get merged?**
Two levers, both at extraction time. Pass a `context` string when you retain (it tells the extractor who's involved and takes precedence for attribution), and, when the cast is known, declare those values as enum entity labels so mentions map onto a fixed vocabulary by exact match. Both are more reliable than leaning on name-similarity scoring after the fact.

**Can it merge two records later if it learns they're the same?**
Not automatically. Co-occurrence links strengthen between them, which helps retrieval reach both, but Hindsight does not retroactively collapse canonical entities, and it has no splitting operation either.

## Further reading

- [What is agent memory?](https://vectorize.io/what-is-agent-memory): the foundational concepts behind extraction, resolution, and recall.
- [Best AI agent memory systems](https://vectorize.io/articles/best-ai-agent-memory-systems): how the major frameworks compare.
- [How we solved memory conflicts in Hindsight](/blog/2026/02/09/resolving-memory-conflicts): what happens when resolved facts disagree.
- [Automatic entity tagging with entity labels](/blog/2026/06/02/entity-labels-automatic-memory-tagging): the controlled-vocabulary side of entities.
