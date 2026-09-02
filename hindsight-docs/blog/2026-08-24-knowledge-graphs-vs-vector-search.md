---
title: "Knowledge Graphs vs. Vector Search for Agent Memory"
authors: [benfrank241]
slug: "2026/08/24/knowledge-graphs-vs-vector-search-agent-memory"
date: 2026-08-24T12:00
tags: [hindsight, agent-memory, knowledge-graph, vector-search, retrieval, deep-dive]
description: "Should your agent's memory be a vector store or a knowledge graph? They solve different halves of the problem. Here is where each one wins, where each breaks, and why agent memory needs both."
image: /img/blog/knowledge-graphs-vs-vector-search.png
hide_table_of_contents: true
---

![Knowledge graphs vs. vector search for agent memory: two retrieval models that solve different halves of the problem](/img/blog/knowledge-graphs-vs-vector-search.png)

Almost every "add memory to your agent" guide reaches for a vector database on the first line. A smaller, louder camp says the opposite: real memory is a knowledge graph, and embeddings are a toy. Both camps are half right, and the half they are missing is the half that breaks in production.

The honest answer is that vector search and knowledge graphs solve *different halves* of the memory problem. If you pick one and ignore the other, you will pass the demo and fail the first hard question a real user asks. Here is where each one wins, where each one breaks, and why agent memory in particular needs both.

<!-- truncate -->

## TL;DR

- **Vector search** finds things that are *conceptually close*. It is unbeatable for paraphrase and synonyms, and it quietly fails on exact names, multi-hop questions, contradictions, and time.
- **Knowledge graphs** model *entities and the relationships between them*. They shine at "who," "why," and "how are these connected," and they cost you an extraction and entity-resolution step up front.
- **Agent memory is not document RAG.** It is about recurring entities, facts that change, and time. Those are exactly the queries pure vector search is worst at.
- The production answer is **hybrid**: run both, plus keyword and temporal, and fuse the results. That is what [Hindsight](https://hindsight.vectorize.io) does on every `recall`.

## What vector search is actually good at

Vector search embeds your text into a high-dimensional space and finds memories whose embeddings sit close to your query's. Its superpower is that it understands *meaning*, not just tokens. Ask "where does Alice work" and it will happily surface "Alice is a software engineer at Google," even though the query and the memory share almost no words. Synonyms, paraphrases, fuzzy conceptual matches: this is what embeddings are for, and nothing else does it as cheaply.

That same fuzziness is the problem. **Semantic similarity is not the same thing as relevance.** The failure modes are predictable:

- **Exact matches slip through.** Ask for `HTTP 502`, a specific SKU, an API endpoint, or a person's exact name, and vector search returns things that are *thematically* related instead of the one that actually says "502." The property that makes embeddings powerful, their willingness to blur, becomes a liability the moment precision matters.
- **There is no notion of an entity.** A vector index stores chunks, not people. It has no idea that "Alice," "Alice Chen," and "she" are the same person, so it cannot walk from Alice to her employer to her teammates. Every chunk is an island.
- **It cannot follow a chain.** "Why did Alice leave?" is a multi-hop question: it needs the events that led to a decision, in order. Cosine similarity ranks a flat pile of chunks; it does not traverse a path.
- **It has no clock.** "What was I working on last Tuesday?" requires resolving a date to a range and bounding the results. Standard vector search treats every memory as equally timeless and will cheerfully hand you something from six months ago because the embeddings were close.

None of these are edge cases. In an agent memory system, they are the bread and butter.

A fair objection: you do not need a graph to fix the first one. Most vector databases now run BM25 or full-text keyword search right next to their embeddings — [Vectorize](https://vectorize.io) included — and pairing semantic with keyword closes the exact-match gap cleanly. `HTTP 502` becomes a lexical hit instead of a fuzzy neighbor, and a proper name matches itself. That is the correct first move, and it is worth doing before reaching for anything heavier.

But keyword search is still *lexical*. It matches tokens, not meaning and not structure. BM25 can find the memory that says "502," yet it has no more idea than the vector index that "Alice," "Alice Chen," and "she" are one person, and it cannot trace why she left or who else was on her team. Adding keyword search fixes exact match and nothing else. Entities, multi-hop, contradictions, and time are still open, and those are the failure modes that actually call for a different data model.

## What a knowledge graph is actually good at

A knowledge graph stores memory as **entities and relationships**: Alice is a node, Google is a node, `works_at` is the edge between them. Instead of a bag of chunks, you get a structure you can traverse.

That structure is exactly what answers the questions vectors choke on:

- **Entities are first-class.** Resolve "Sarah," "Sarah Chen," and "she" into one node once, and every future memory about her attaches to the same place. (Hindsight does this with a co-occurrence graph rather than embeddings, which is [its own hard problem](https://hindsight.vectorize.io/blog/2026/06/29/entity-resolution-agent-memory).)
- **Relationships enable multi-hop.** "Who else was on the project Alice left?" is a two-hop traversal: Alice → project → people. A graph walks it; a vector index cannot express it.
- **Contradictions have a home.** When Alice changes jobs, you do not want two conflicting chunks with similar embeddings both ranking highly. In a graph, the new relationship supersedes the old one, and you can keep the history as a timeline instead of a contradiction.

The catch is that a graph is not free. Something has to *build* it: extract entities and relationships from raw text, resolve duplicates, and keep edges current. That extraction is usually an LLM step, entity resolution is genuinely difficult, and a brand-new graph is empty, so cold-start recall leans on other signals. A knowledge graph is powerful precisely because it is structured, and structure is work.

## At a glance

| | Vector search | Knowledge graph |
|---|---|---|
| **Best at** | Meaning, paraphrase, synonyms, fuzzy conceptual recall | Entities, relationships, multi-hop "who / why," change over time |
| **Struggles with** | Multi-hop chains, entity relationships, contradictions, time (exact terms are keyword/BM25's job) | Cold start, extraction cost, entity-resolution errors, free-form fuzz |
| **Setup cost** | Embed on write — cheap and immediate | Extract, resolve entities, and maintain edges |
| **Example it wins** | "Where does Alice work?" | "Why did Alice leave, and who else was on her team?" |

Read that table as a division of labor, not a scoreboard. Neither column is the winner, because neither column covers the other's row.

## Why agent memory tilts the answer

Here is the part the generic "vector DB vs. graph" debate misses: **agent memory is not document search.** When you retrieve over a static corpus of manuals, the documents do not change, entities barely matter, and time is irrelevant. Agent memory is the opposite on all three counts:

- It is **about recurring entities** — the same users, projects, and preferences show up again and again.
- Its facts **change constantly** — someone's role, a decision, a status. Yesterday's truth is today's contradiction.
- It is **deeply time-sensitive** — "last week," "before the migration," "what changed since."

Those three properties are precisely the ones pure vector search handles worst and a graph handles best. Which is why "just install a vector database" is [the wrong default for agent memory](https://hindsight.vectorize.io/blog/2026/05/12/case-against-external-vector-dbs-agent-memory) — not because embeddings are useless, but because they only cover the conceptual-similarity slice of a much larger job.

## The production answer is "and," not "vs"

Drop the versus. A real agent gets all four kinds of question in a single session: the exact name of a library (keyword), the gist of a past conversation (semantic), the chain of events behind a decision (graph), and what happened in a specific window (temporal). No single index answers all four, so Hindsight runs all four *in parallel* on every `recall` and fuses the results:

- **Semantic** — vector similarity over embeddings, for meaning and paraphrase.
- **Keyword** — BM25 full-text, for exact names, codes, and identifiers.
- **Graph** — traversal over precomputed links (entity co-occurrence, semantic neighbors, causal chains), for multi-hop and "why."
- **Temporal** — time-bounded recall with [spreading activation over events](https://hindsight.vectorize.io/blog/2026/03/12/spreading-activation-memory-graphs), for "when."

The four run concurrently, get merged with reciprocal-rank fusion, and a cross-encoder reranks the survivors so the final list is ordered by actual relevance rather than by whichever strategy shouted loudest. If you want the engineering underneath, we wrote up [how the 4-way parallel hybrid search works](https://hindsight.vectorize.io/blog/2026/03/27/parallel-hybrid-search).

The knowledge graph is not a competitor to the vector index here. It is one voice in a chorus, and it happens to be the voice that carries the questions embeddings cannot.

## So which should you use?

If you are building agent memory, the practical takeaway is short: you do not choose. Use vector search for conceptual recall, keyword for precision, a graph for entities and multi-hop reasoning, and a temporal index for time, and let a fusion step decide who was right for each query. Reach for a graph the moment your agent needs to answer "who," "why," or "how are these connected," and lean on vectors the moment it needs to match meaning instead of words.

The reason the "graph vs. vector" argument never resolves is that it is asking the wrong question. The right one is *which retrieval strategy wins for this query* — and the only good answer is to run all of them and fuse. Start free on [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup), where every recall is already hybrid.

## Frequently asked questions

### Is a knowledge graph better than vector search for agent memory?

Neither is strictly better; they answer different questions. Vector search wins on meaning and paraphrase, and a knowledge graph wins on entities, multi-hop reasoning, and how facts change over time. For agent memory specifically, where the same people and projects recur and yesterday's fact is often today's contradiction, the graph covers exactly the queries embeddings miss. So the strongest systems do not choose; they run both.

### Can you use a knowledge graph and vector search together?

Yes, and you generally should. That combination is called hybrid retrieval: run semantic (vector), keyword, graph, and temporal search in parallel, then fuse the ranked lists so the best result for each kind of query rises to the top. Hindsight runs all four on every `recall` and reranks the merged set with a cross-encoder, so a single query gets the strengths of each strategy without you having to pick one in advance.

### Do I need a dedicated graph database like Neo4j?

Not necessarily. A knowledge graph is a *data model*, not a specific product. Hindsight builds and traverses its graph inside PostgreSQL, right alongside the vector and full-text indexes, so you get graph traversal without standing up and syncing a second database. That is part of [the case against bolting on an external vector (or graph) DB](https://hindsight.vectorize.io/blog/2026/05/12/case-against-external-vector-dbs-agent-memory): the operational tax rarely earns its keep for agent-memory workloads.

### What is hybrid search?

Hybrid search combines multiple retrieval strategies instead of leaning on one. In Hindsight that means four running together — semantic similarity, BM25 keyword, graph traversal, and temporal search — fused with reciprocal-rank fusion and then reranked. It exists because no single index handles exact terms, loose concepts, entity relationships, and time equally well, and real queries demand all four.

### When should I reach for the graph instead of vectors?

Reach for the graph whenever the question is about entities or connections — "who," "why," "how are these related," or anything that requires more than one hop. Reach for vectors when the question is about meaning and the user might phrase it differently than the memory was written. In practice you let both run every time and let the fusion step decide which was right for this particular query.

### How is this different from GraphRAG?

GraphRAG-style approaches build a knowledge graph over a *static document corpus* to answer questions about those documents. Agent memory flips the inputs: the graph is built from an agent's own evolving observations about recurring entities — users, projects, decisions — and it is updated continuously as new turns arrive, not indexed once. The retrieval question is also live, asked mid-session rather than against a frozen archive, which is why time and contradiction handling matter so much more. The underlying idea (structure your knowledge as a graph) is shared; the workload (write-heavy, entity-centric, constantly changing) is what makes agent memory its own problem, and why the graph rides alongside vector, keyword, and temporal search rather than replacing them.

---

**Learn more:**
- [recall vs reflect](https://hindsight.vectorize.io/blog/2026/07/24/recall-vs-reflect) — search your agent's memory, or ask it to reason over it
- [How the 4-way parallel hybrid search works](https://hindsight.vectorize.io/blog/2026/03/27/parallel-hybrid-search) — the engineering behind fusing four strategies
- [Entity resolution in agent memory](https://hindsight.vectorize.io/blog/2026/06/29/entity-resolution-agent-memory) — how one person with many names becomes one node
