---
title: "The 10 Things to Look For in an Agent-Memory System"
description: "The 10 things that separate a real agent-memory system from a demo: retrieval, conflict resolution, freshness, security, self-hosting, observability, and more."
authors: [benfrank241]
slug: "2026/07/31/evaluate-agent-memory-system"
date: 2026-07-31T12:00
tags: [hindsight, agent-memory, evaluation, benchmark, beam, deep-dive]
image: /img/blog/evaluate-agent-memory-system.png
hide_table_of_contents: true
---

![The 10 things to look for in an agent-memory system: retrieval, conflicts, freshness, security, and more](/img/blog/evaluate-agent-memory-system.png)

Most agent-memory evaluations test one thing: given a query, did the system return the right chunk. Then the system ships, and the failures show up somewhere else entirely — a preference the agent never updated, a fact from six months ago it treated as current, a customer's card number sitting in a bank three tenants can read.

Retrieval accuracy is table stakes. It's necessary and nowhere near sufficient. If you're choosing an agent-memory system (or building one), the real question is how it behaves on the dimensions that don't show up in a quickstart demo. These are the ten things that actually separate a real system from a demo, plus a checklist you can run against any candidate.

<!-- truncate -->

New to the category? Start with [what agent memory is](https://vectorize.io/what-is-agent-memory) and why it's a distinct problem from stuffing more tokens into the [context window](/blog/2026/07/22/context-window-is-not-memory). This piece assumes you already know you need memory and just want to judge which system is any good.

## Why "it retrieved the right chunk" is not an agent-memory evaluation

An agent-memory system has a lifecycle, and retrieval is the last step of it:

1. **Write** — extract durable facts from a messy conversation or trajectory.
2. **Store** — resolve entities, dedupe, structure, and link the new facts to what is already known.
3. **Manage** — update facts that changed, retire facts that are wrong, keep track of when each was true.
4. **Read** — retrieve and rank the relevant facts for the current query.

Almost every demo evaluates step four in isolation: seed a bank with clean facts, ask a question, check the answer. But most production failures start in write and manage. The agent stored the transcript verbatim instead of the decision inside it. It created a second "auth service" entity instead of resolving it to the first. It kept a stale preference next to a fresh one and surfaced the wrong one. None of those are retrieval bugs. None of them show up if your eval only measures retrieval.

So the first rule of evaluating agent memory: **test the whole lifecycle, not just the read.** Everything below is organized around that.

## The 10 things to look for in an agent-memory system

Here is the short version as a scannable table: ten things to look for, what to test for each, and the red flag that tells you the system is weaker than its landing page.

| # | What to look for | How to test | Red flag |
|---|---|---|---|
| 1 | Retrieval beyond similarity | Multi-hop and entity/time-scoped queries, not just paraphrase lookups | Pure vector top-k, no reranking |
| 2 | Extraction & consolidation | Store the same fact 5 ways; count duplicates and invented facts | Raw transcript storage, growing duplicates |
| 3 | Entity resolution | Refer to one thing by three names across sessions | Three separate entities for one concept |
| 4 | Conflict resolution | Change a stored preference; ask again | Both old and new returned, no supersession |
| 5 | Freshness / temporal | "What did I say last week?"; superseded facts | No event dates, no valid-time model |
| 6 | Test-time learning | Teach a fact mid-session; use it in the same session | New info ignored until a re-index job runs |
| 7 | Security | Feed it a secret/PII; check what got stored and who can read it | Secrets stored verbatim, no scoping |
| 8 | Data ownership | Can you self-host it with your own database? | Cloud-only, or bring-your-own external DBs |
| 9 | Observability | Ask why a memory was recalled | Opaque; no trace of what matched and why |
| 10 | Multi-tenant scoping | Store as tenant A, recall as tenant B | Leakage across users with default config |

The rest of this section takes the capability rows one at a time. The operational rows (security, ownership, observability, scoping) get their own section after.

### Retrieval quality beyond vector similarity

Vector search answers one question well: what is semantically similar to this query. It's blind to a lot else. "What did the customer decide before the Q2 revision" is a temporal question. "Which service owns the token refresh" is an entity-and-relationship question. Cosine similarity over embeddings doesn't natively handle either.

The systems that hold up run more than one retrieval path and merge them. Hindsight, for example, runs four in parallel — semantic search, entity-based retrieval, temporal filtering, and graph traversal — then scores everything with a [cross-encoder reranker](/blog/2026/03/27/parallel-hybrid-search) before it reaches the agent. The bet is that no single method dominates across query types, so running several catches what any one would miss.

To test this, do not ask paraphrase questions. Ask **multi-hop** ones ("who introduced the person who owns the billing service") and **scoped** ones ("what changed about deployment in March"). Watch whether the system degrades to keyword-ish similarity or actually traverses relationships. If you want to understand the read path more deeply, the difference between [recall and reflect](/blog/2026/07/24/recall-vs-reflect) and how [spreading activation over a memory graph](/blog/2026/03/12/spreading-activation-memory-graphs) works are both worth reading.

### Extraction and consolidation

The write path decides what quality the read path even has access to. A system that stores raw chat logs is deferring all the hard work to retrieval time, where it is much harder. A system that [extracts atomic facts](/blog/2026/07/13/inside-retain-agent-memory) and [consolidates them](/blog/2026/05/21/agent-memory-consolidation) is building a clean substrate.

The test is blunt: state the same fact five different ways across five turns or sessions ("we use Postgres," "our database is Postgres," "storage is on Postgres," and so on). Then inspect the bank. A good system converges these into one fact with reinforced confidence. A weak one accumulates five near-duplicate rows, which will later surface as five noisy hits. While you are in there, check the opposite failure: did it **invent** any facts that were never stated? Consolidation that hallucinates is worse than no consolidation.

### Entity resolution

Entity resolution is the quiet capability that separates a knowledge base from a pile of strings. "The auth service," "our login system," and "the OAuth microservice" are one entity. If the system tracks them as three, every retrieval that touches that entity is fragmented, and the agent looks like it has amnesia.

Test it directly: refer to a single thing by several names across separate sessions, then ask a question that requires knowing they are the same. If you get partial answers that each only "know" one alias, [entity resolution](/blog/2026/06/29/entity-resolution-agent-memory) is weak. This is also where a lot of open-source demos quietly fall over, because resolution is expensive to do well.

### Conflict resolution and knowledge updates

Facts change. The customer moved. The team switched from weekly to biweekly standups. The stack dropped the ORM. A memory system that only ever appends will happily hand your agent both the old and the new fact and let the model guess.

The evaluation is a two-step: store a preference, then contradict it later, then ask. A system with real [conflict resolution](/blog/2026/02/09/resolving-memory-conflicts) supersedes the old value and returns the current one, ideally while keeping the history retrievable if you ask about the past explicitly. Academic benchmarks call this "knowledge updating," and it is one of the dimensions that separates systems that look good in a demo from systems that survive a long-lived user.

### Freshness and temporal reasoning

Related but distinct: can the system reason about *when*. Two things to probe. First, relative-time queries — "what did I mention last week" — which require [freshness-aware retrieval](/blog/2026/06/17/freshness-aware-memory) and real event dates on facts, not just a `created_at` on a row. Second, validity: does a fact that was true in Q1 and false in Q2 get modeled as having a lifespan.

Be honest with yourself about how much you need here. This is genuinely where Zep pulls ahead of most of the field — its knowledge graph carries explicit `valid_from` / `valid_to` metadata on every edge, and if your agent's core job is audit trails or compliance timelines, that depth is hard to match. Most systems, Hindsight included, do temporal filtering without modeling full fact-validity windows. Evaluate against your actual use case, not the most impressive one.

### Test-time learning

Some evals only measure recall of facts that were indexed in a prior batch job. Production is not batched. A user tells your agent something in turn 3 and expects it to matter in turn 9 of the same session. Test whether new information is usable immediately, or whether it only becomes retrievable after an offline consolidation pass. Both designs are valid — but you want to know which one you are buying, because it changes what your agent can do within a single conversation.

## The production dimensions most guides skip

Here is where the buyer's guides and decision-tree posts go quiet. These dimensions never show up in a benchmark score, and they are the ones that get escalated to you at 2 a.m.

### Data ownership and self-hosting

Agent memory is the most sensitive data in your stack. It's a durable record of everything your agent has seen: conversations, decisions, code, customer details. That's exactly the data you least want to hand to a third-party service you can't inspect or host.

Ask four concrete questions. Can you self-host it at all? What does self-hosting actually require — one container, or a vector DB plus a graph DB plus a queue you now operate? What is the license? And if you don't want to run infrastructure, is there a managed option that isn't feature-gated? Hindsight self-hosts with a single Docker command and embedded PostgreSQL, MIT-licensed, no external datastores — which is a deliberate design choice, because [external vector databases carry real costs](/blog/2026/05/12/case-against-external-vector-dbs-agent-memory) and teams increasingly want to [migrate off them](/blog/2026/07/28/migrate-agent-memory-off-vector-database). For teams that would rather not operate it, [Hindsight Cloud](https://hindsight.vectorize.io) is the managed option with full feature parity — no capabilities withheld from the self-hosted build, which is not something every vendor can say. Zep's community edition is deprecated, Mem0 expects an external vector store, and several hosted products cannot be self-hosted without an enterprise agreement. Those are not disqualifiers — they are trade-offs you should price in before, not after, you integrate.

### Security: keeping secrets and PII out of memory

If your write path stores raw conversation, then every API key, password, and PII string a user ever pastes is now in your memory bank, retrievable, and possibly shared. Two things to test. First, feed the system a secret and inspect what actually got persisted — is it scrubbed, or is it sitting there in plain text. Second, check the read side: can that memory surface for a user who should never see it.

This is a real evaluation, not a checkbox. A memory system without secret and PII scrubbing is a data-exfiltration surface waiting to happen, and it is the kind of thing that passes every retrieval benchmark while failing your security review.

### Latency, cost, and scale

Retrieval quality that takes two seconds and three LLM calls per query is a different product than one that takes 200 milliseconds. Measure write latency and read latency separately, under a bank size that resembles production, not the 20-fact demo. Ask what the cost model looks like when a bank has a million facts, and whether recall latency stays flat as it grows. [How a memory system scales](/blog/2026/05/08/how-hindsight-scales) — single table versus a sprawl of services, embedded versus external storage — determines whether your cost curve is linear or ugly.

### Observability and debuggability

When an agent recalls the wrong thing, can you find out why. A system that returns an opaque list of memories, with no trace of what matched, which strategy fired, or how it ranked, is one you can't debug in production. Look for a way to inspect the retrieval: what candidates were considered, how the reranker scored them, what got cut. You'll spend more time debugging recall than building it, so weight this heavily.

### Multi-tenant scoping and isolation

If you serve more than one user, the single most important architectural question is what is recallable by whom. Store a fact as tenant A and try to recall it as tenant B. If it leaks under default configuration, you have found a launch-blocking bug in the vendor, not in your code. Deciding how to slice this — one bank per user, per project, per org — is its own design problem; the [bank-strategy field guide](/blog/2026/07/16/bank-strategy-agent-memory) covers the trade-offs. Whatever the model, the isolation guarantee should be explicit and testable.

## How to read a benchmark without fooling yourself

Public benchmarks are useful context and a bad substitute for your own eval. The first trap is age. The benchmarks most people cite — LoCoMo, LongMemEval — were designed in the 32K-context era, when you physically could not fit a long history into one model call, so needing a memory system was the premise. With million-token windows, a naive "dump everything into context" baseline now scores competitively on them. A high number no longer distinguishes a real memory architecture from a context stuffer.

[BEAM](https://arxiv.org/pdf/2510.27246) ("Beyond a Million Tokens") is the one worth watching, because it tests at context lengths up to **10 million tokens**, where stuffing is physically impossible and only a system that can retrieve from a too-large pool survives. If you want a single public number to sanity-check retrieval at scale, look there rather than at the older datasets. (For the record, Hindsight currently tops BEAM's 10M tier at 64.1% — [the details are here](/blog/2026/04/02/beam-sota) — but treat that as a reason to trust the architecture, not as your evaluation.)

Whatever number you look at, read it critically: what context length and split it used, which model did the extraction and answering (the same memory system swings several points across LLMs), and whether the harness is reproducible. Then set it aside. **No public benchmark reflects your data, your query mix, or your failure costs. The only eval that predicts your production behavior is the one you build on your own domain — which is the next section.**

## Run the eval yourself

You do not need a research lab to do this well. You need a small, honest harness:

1. **Build a gold set from your domain.** Twenty to fifty real scenarios beat a thousand synthetic ones. Include multi-hop questions, an update ("I changed my mind about X"), a temporal question, and a secret you should never see stored.
2. **Test each dimension, not just retrieval.** Write, store, manage, read. Inspect the bank after writes, not only the answers after reads.
3. **Trace-review the failures.** For every wrong answer, find the stage that broke. Was the fact never extracted, mis-resolved, not superseded, or just not retrieved?
4. **Measure a few numbers.** Recall (did the needed evidence come back), ranking quality (was it near the top), and update success (old gone, new kept).
5. **Re-run on every change.** Model swap, config change, version bump — memory quality drifts, and a repeatable loop is the only way you notice.

## The evaluation checklist

Copy this. A candidate that cannot check most of these boxes is a demo, not a memory system.

| # | ✅ | What to look for |
|---|---|---|
| 1 | ☐ | Retrieval combines more than vector similarity (entity, temporal, graph) and reranks, not raw top-k |
| 2 | ☐ | Facts are extracted and deduplicated, not stored as raw transcript |
| 3 | ☐ | Entity resolution merges aliases across sessions |
| 4 | ☐ | Conflicting facts are superseded, with history retrievable |
| 5 | ☐ | Facts carry event dates; relative-time queries work |
| 6 | ☐ | New information is usable in the same session (or you know it isn't) |
| 7 | ☐ | Secrets and PII are scrubbed before storage |
| 8 | ☐ | Self-hostable with a license and datastore you accept |
| 9 | ☐ | Retrieval is inspectable — you can see why something was recalled |
| 10 | ☐ | Tenant isolation is explicit and you have tested a cross-tenant recall |

## Evaluate the whole system, not the demo

Three things to take away. Retrieval accuracy is the entry fee, not the evaluation — most failures live in the write and manage stages, so test the full lifecycle. The dimensions that never appear in a benchmark score — security, data ownership, observability, tenant isolation — are the ones that decide whether a system survives production. And a benchmark number is only as good as the split, model, and methodology behind it, so learn to interrogate it.

Weight the dimensions by your use case. An audit-trail agent should over-index on temporal validity; a multi-user support product should over-index on isolation and PII handling; a coding assistant should over-index on entity resolution and conflict updates. There is no single best system, only the best fit for what you are building.

If you want to run this checklist against something concrete, Hindsight is open source, MIT-licensed, and self-hosts in one Docker command — or runs managed on [Hindsight Cloud](https://hindsight.vectorize.io) with the same feature set — so you can benchmark it on your own data instead of trusting ours. Whatever you choose, choose it with an eval that tests the whole system — not the part that fits in a quickstart.

---

**Further reading:**
- [What Is Agent Memory?](https://vectorize.io/what-is-agent-memory) — foundational concepts
- [Best AI Agent Memory Systems in 2026](https://vectorize.io/articles/best-ai-agent-memory-systems) — full framework comparison
- [The Case Against External Vector Databases](/blog/2026/05/12/case-against-external-vector-dbs-agent-memory) — storage architecture trade-offs
- [Recall vs Reflect](/blog/2026/07/24/recall-vs-reflect) — the two ways to read memory back
