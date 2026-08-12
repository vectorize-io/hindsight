---
title: "Best Open-Source Agent Memory Systems (Self-Hosted, 2026)"
description: "A benchmark-grounded, fair comparison of the open-source agent memory systems you can self-host: Hindsight, Mem0, Graphiti/Zep, Letta, and Cognee."
authors: [benfrank241]
slug: "2026/08/11/open-source-agent-memory-systems"
date: 2026-08-11T12:00
tags: [hindsight, agent-memory, open-source, self-hosting, comparison, mem0, zep, letta, cognee]
image: /img/blog/open-source-agent-memory-systems.png
hide_table_of_contents: true
---

![Best open-source agent memory systems compared: Hindsight, Mem0, Graphiti/Zep, Letta, Cognee](/img/blog/open-source-agent-memory-systems.png)

"Open source" and "self-hostable" get used as synonyms, and they are not the same promise. Plenty of agent-memory tools that call themselves open source still expect you to stand up an external vector database, run a graph database next to it, or reach for a credit card at the tier you actually need in production.

If you're picking a memory layer you can run yourself, that gap matters more than a star count. Here is the honest version: what each open-source agent memory system really costs to run, how they're architected, and how they score when you measure instead of guess.

<!-- truncate -->

New to the category? Start with [what agent memory is](https://vectorize.io/what-is-agent-memory). This post is the open-source cut of the [full comparison of all major frameworks](https://vectorize.io/articles/best-ai-agent-memory-systems); here we only cover systems you can self-host.

Fair warning on the genre: most "best open-source memory" roundups are published by one of the vendors and, conveniently, rank that vendor first. This one is written by one too. So instead of asserting a winner, it leans on two things you can check yourself: **published benchmark numbers** and **what each system actually requires to run**.

## How to compare open-source agent memory

Five axes decide this, and only two of them show up on a landing page:

1. **Retrieval accuracy** — does it return the right memory, measured on a public benchmark, not asserted. See [how to evaluate an agent-memory system](/blog/2026/07/31/evaluate-agent-memory-system) for the full rubric.
2. **Self-hosting reality** — can you run it with your own infrastructure, and what does "your own infrastructure" include? One container, or a vector DB plus a graph DB plus a queue?
3. **Feature parity when self-hosted** — do you get the whole product, or a stripped core that nudges you toward the paid cloud?
4. **License** — permissive enough to actually deploy.
5. **Architecture** — vector, graph, or hybrid, and what that implies for the queries you can ask.

Weight them for your case. An audit-trail agent cares most about temporal modeling; a lean self-hoster cares most about how many databases land on their ops plate.

## The open-source memory systems at a glance

Numbers verified against GitHub and the published benchmarks in August 2026.

| System | GitHub ★ | License | Self-host reality | Storage | Strongest at |
|---|---|---|---|---|---|
| **Mem0** | ~63.0K | Apache-2.0 | OSS core; embedded store by default, external vector DB for production; hosted tier | Vector (+ optional graph) | Community + conversational personalization |
| **Cognee** | ~30.0K | Apache-2.0 | Embedded defaults, no mandatory cloud | Graph pipeline (SQLite/LanceDB/Kuzu) | Graph ECL pipelines |
| **Graphiti / Zep** | ~29.8K | Apache-2.0 | Graphiti self-hosts; Zep CE deprecated | Temporal knowledge graph (Neo4j/FalkorDB) | Temporal reasoning |
| **Letta** (ex-MemGPT) | ~24.2K | Apache-2.0 | Self-hostable server | Agent runtime + memory | Full agent runtime |
| **Hindsight** | ~19.6K | MIT | One Docker command, embedded PostgreSQL, no external DB | Single embedded Postgres, multi-strategy | Accuracy + deployment simplicity |

Two honest notes before the profiles. Hindsight is **not** the most-starred project here; Mem0, Cognee, and Graphiti all have larger communities, and that's worth something real. And every system in this table is permissively licensed, so "it's open source" is table stakes, not a differentiator. The differences that matter are accuracy and what it takes to run.

## The systems, one by one

### Mem0 — the biggest community

Mem0 is the most popular agent-memory project on GitHub by a wide margin (~63.0K stars), and popularity compounds: more integrations, more examples, more Stack Overflow answers when you're stuck. It's genuinely good at conversational personalization: memories scope by `user_id`, `agent_id`, and `run_id`, so recalling one person's preferences across sessions is the paved path, and you can start with the library, graduate to a self-hosted server, or use the hosted platform. The write side extracts facts from turns rather than storing raw transcript, which keeps the store from ballooning.

The trade-offs are less about a night-and-day accuracy gap and more about which benchmark you trust and the deployment shape. Mem0 publishes strong LongMemEval numbers, competitive with the top of the field, but LongMemEval is close to saturated now: the serious systems all cluster in the 90s, so it barely separates them. The sharper test is BEAM at 10 million tokens, where Hindsight is #1 and Mem0 doesn't publish a comparable result. On deployment, Mem0's open-source core runs with an embedded store by default, though production setups typically move to an external vector database, and the smoothest path nudges toward the hosted platform. If community size and a personalization-first design are your priorities, Mem0 is the obvious pick.

### Graphiti / Zep — the temporal specialist

This is the one to reach for when *when a fact was true* is the whole job. Graphiti, the open-source engine under Zep, models time natively: every edge in its knowledge graph carries validity metadata, so it answers "what was the customer's plan before the Q2 change?" cleanly. For compliance timelines and audit trails, that temporal depth is hard to match, and it's the axis where Graphiti genuinely leads.

Two caveats for self-hosters. **Zep Community Edition was deprecated in 2025**, so the free, open-source path to "self-host Zep with full features" is gone — what you self-host is Graphiti, the engine, not the whole product (enterprise BYOC aside). And Graphiti typically stores its graph in an external database (Neo4j or FalkorDB, though an embedded FalkorDB lite mode exists), so in most deployments you're running and operating that alongside your app.

### Letta — the agent runtime

Letta (formerly MemGPT) is the odd one out here, in a useful way: it's not really a drop-in memory *layer*, it's an **agent runtime** with memory built in. If you want the whole framework — the agent loop, tool use, and memory as one opinionated system — Letta is a strong, self-hostable choice with a real research lineage. Its memory model is worth knowing: agents hold editable **memory blocks** (a small, always-in-context core the agent rewrites itself) plus **archival memory** for the long tail, an idea straight from the MemGPT paper. If you already have an agent and just want to bolt on memory, it's heavier than you need, because you're adopting a runtime to get a memory feature.

### Cognee — the graph-pipeline option

Cognee's angle is an ECL (extract-cognify-load) graph pipeline with pluggable, embedded-by-default storage (SQLite, LanceDB, Kuzu), which makes for a genuinely low-friction local start with no mandatory cloud. If you like the idea of a configurable graph-construction pipeline over your data, it's worth a look.

The thing to bring your own skepticism to is the ranking. Several of the "best open-source memory" roundups at the top of Google are Cognee's own, and they place Cognee first without independent benchmark numbers. That's the gap this whole post is about: prefer a system that publishes a score on a benchmark you can reproduce over one that ranks itself in a blog it controls.

### Hindsight — accuracy and the simplest self-host

Full disclosure, this is our project, so weigh the claims against the sources rather than the byline. Hindsight's bet is the two axes above: retrieval accuracy and deployment simplicity.

On accuracy, it posts the highest published scores among these systems — **94.6% on LongMemEval-s**, and **#1 on [BEAM](/blog/2026/04/02/beam-sota)**, the benchmark that tests memory at 10 million tokens where context stuffing is impossible. Both are on public leaderboards with reproducible methodology, which is the point. It gets there by running [four retrieval strategies in parallel](/blog/2026/07/31/evaluate-agent-memory-system) (semantic, BM25 keyword, graph traversal, temporal) with cross-encoder reranking, rather than betting on one.

On deployment, it's a **single Docker command with embedded PostgreSQL** — no external vector database, no separate graph database, [no bring-your-own-datastore](/blog/2026/05/12/case-against-external-vector-dbs-agent-memory) at all. It's MIT-licensed, and the self-hosted build has full feature parity with the managed cloud; nothing is gated. Where it doesn't lead is community size and native temporal-validity modeling, which is exactly why Graphiti is in this list.

### Also worth knowing

**LangMem** is the sensible default if you're already committed to LangGraph; it's the in-ecosystem option rather than a standalone service. **Memobase** (~2.8K stars) focuses on user-profile memory and is worth a look for that shape. **Memary** exists but has been dormant since 2024, so treat it as a reference, not a dependency.

## What "self-hosted" really costs

The star counts hide the number that hits your ops budget: how many stateful services you end up running. Group the field by it.

- **One embedded store:** Hindsight keeps everything — memories, entities, graph, vectors — in a single embedded PostgreSQL. Cognee runs embedded too, across three stores (SQLite, LanceDB, Kuzu). Mem0 also ships an embedded default (on-disk Qdrant).
- **External database for production:** Graphiti typically runs on an external graph DB (Neo4j or FalkorDB), and Mem0 deployments usually move to an external vector store at scale. Those are real services to run, back up, and secure.
- **Cloud pressure:** Zep's full product is cloud now that CE is deprecated, and Mem0's smoothest path is its hosted platform. The open-source cores are real, but the paved road often leads off-prem.

None of this is disqualifying — a team already running Neo4j pays little marginal cost to add Graphiti. But "self-hosted agent memory" should mean you priced the whole deployment, external dependencies included, and know that [external datastores carry real cost](/blog/2026/07/28/migrate-agent-memory-off-vector-database) before you wire them in.

## When to pick each

- **Mem0** — you want the largest community and a personalization-first design.
- **Graphiti / Zep** — your agent's core job is temporal reasoning (audit trails, evolving records) and you're fine running an external graph database.
- **Letta** — you want a full, self-hostable agent runtime, not just a memory layer.
- **Cognee** — you want a configurable graph pipeline with embedded local defaults, and you'll benchmark it yourself.
- **Hindsight** — you want the highest published retrieval accuracy and the simplest deployment (one container, no external DB), and you don't need native fact-validity windows.
- **LangMem** — you're already all-in on LangGraph.

## Frequently asked questions

**Is Mem0 really open source?**
Yes, the Mem0 core is Apache-2.0 and self-hostable, and it runs with an embedded store by default. But the smoothest production path is its hosted platform, and at scale most deployments move to an external vector database, so "open source" and "no external services in production" are not the same thing here.

**Is Zep open source?**
Partly. Zep's open-source engine is Graphiti (Apache-2.0), which you can self-host. Zep the product is a cloud service, and Zep Community Edition (the self-hostable full product) has been deprecated. If you need the whole Zep feature set on your own infrastructure, that path is gone; what stays open is Graphiti, the engine.

**Which open-source agent memory is easiest to self-host?**
By the number of stateful services you have to run, the embedded-storage options win: Hindsight (one Docker container, embedded PostgreSQL, no external database) and Cognee (embedded SQLite/LanceDB/Kuzu defaults). Graphiti adds an external graph database, and Mem0 adds a vector store, both of which you then operate, back up, and secure.

**Which open-source agent memory is the most accurate?**
On LongMemEval, most serious systems now score in the 90s (Hindsight is at 94.6% on LongMemEval-s), so that benchmark barely separates them anymore. The sharper test is BEAM, which evaluates memory at 10 million tokens where context stuffing is impossible: Hindsight is #1 there, and most competitors don't publish a comparable result. When a roundup declares a winner, ask which benchmark, at what scale, and with what methodology.

**Do I need a vector database for agent memory?**
Not necessarily. Some store their graph in an external database (Graphiti), and some run a separate vector store in production (Mem0, though its default is embedded). Others, like Hindsight, keep everything in a single embedded PostgreSQL. That's why the deployment shape, not just the license, belongs in your comparison.

**What about LangMem, Memobase, and Memary?**
LangMem is the in-ecosystem choice if you're already on LangGraph. Memobase (~2.8K stars) is a smaller, user-profile-focused option. Memary has been dormant since 2024, so treat it as a reference rather than something to build on.

## The honest verdict

There's no universal winner, only the best fit for what you're optimizing. If you want the biggest community, Mem0. If you're building audit-grade temporal reasoning, Graphiti. If you want the whole agent framework, Letta. If you want a configurable graph pipeline, Cognee.

And if you're self-hosting and what you care about is retrieval accuracy plus a boring, single-container deployment you can stand up in a minute and forget, Hindsight is built for exactly that. It self-hosts with one Docker command and embedded PostgreSQL, MIT-licensed, or runs managed on [Hindsight Cloud](https://hindsight.vectorize.io) with the same feature set. Whichever you choose, choose it on a benchmark you can reproduce and a deployment you've actually priced, not on whose blog ranked whom first.

---

**Further reading:**
- [What Is Agent Memory?](https://vectorize.io/what-is-agent-memory) — foundational concepts
- [Best AI Agent Memory Systems in 2026](https://vectorize.io/articles/best-ai-agent-memory-systems) — all major frameworks, not just open source
- [How to Evaluate an Agent-Memory System](/blog/2026/07/31/evaluate-agent-memory-system) — the criteria in depth
- [The Case Against External Vector Databases](/blog/2026/05/12/case-against-external-vector-dbs-agent-memory) — why the deployment shape matters
