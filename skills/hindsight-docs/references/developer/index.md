

# Overview

## Why Hindsight?

AI agents forget everything between sessions. Every conversation starts from zero—no context about who you are, what you've discussed, or what the assistant has learned. This isn't just an implementation detail; it fundamentally limits what AI Agents can do.

**The problem is harder than it looks:**

- **Simple vector search isn't enough** — "What did Alice do last spring?" requires temporal reasoning, not just semantic similarity
- **Facts get disconnected** — Knowing "Alice works at Google" and "Google is in Mountain View" should let you answer "Where does Alice work?" even if you never stored that directly
- **AI Agents need to consolidate knowledge** — A coding assistant that remembers "the user prefers functional programming" should consolidate this into an observation and weigh it when making recommendations
- **Context matters** — The same information means different things to different memory banks with different personalities

Hindsight solves these problems with a memory system designed specifically for AI agents.

## What Hindsight Does

**Figure: What Hindsight Does.** An animated diagram on the docs site; its narration, step by step:

- **retain()**
  1. Your agent sends what happened: a conversation, a document, a transcript.
  2. The original text is stored as a document.
  3. It is split into chunks, so the exact passage can be handed back later.
  4. An LLM pulls out facts: world facts about others, and experience facts about what the agent itself did. The bank already knew Alice worked at Microsoft.
  5. Each fact is indexed four ways: by meaning, by its words, by the entities it links, and by when it happened.
  6. retain() is done. The rest happens in the background.
  7. Consolidation picks up the new facts and checks them against the observations the bank already holds. One disagrees: Microsoft or Google?
  8. It updates that observation instead of adding a second one: Alice moved from Microsoft to Google in March. Both facts stay as its sources, so the history is kept.
  9. When consolidation finishes, it queues a refresh for every mental model and page set to refresh after it that now has new memories…
  10. …and each one re-runs its question through reflect and is rewritten.
- **recall()**
  1. recall() finds the memories that matter for a query.
  2. Searches run at once, each through its own index: meaning, exact words and the entity graph. The time search only joins when the query names a date.
  3. The same indexes cover facts and observations, so both come back. They are merged and reranked; the old Microsoft fact falls below the cut.
  4. The agent gets ranked memories it can put straight into its prompt.
- **reflect()**
  1. reflect() answers a question by reasoning over everything in the bank.
  2. An agent loop decides what to look up. It starts with the most refined knowledge: mental models and knowledge pages.
  3. Then observations, searched through the same indexes as recall. If new facts are still waiting to be consolidated, they are marked stale.
  4. Then raw facts through recall, for the details the summaries leave out.
  5. When it needs the exact wording, it opens the chunk or document a fact came from.
  6. It stops when it has enough evidence, and writes an answer shaped by the bank’s mission and disposition. It can only cite what it found.
  7. The answer comes back with the memories it is based on.

**Your AI agent** stores information via `retain()`, searches with `recall()`, and reasons with `reflect()` — all interactions with its dedicated **memory bank**

## Key Components

### Memory Types

Hindsight organizes knowledge into a hierarchy of facts and consolidated knowledge:

| Type | What it stores | Example |
|------|----------------|---------|
| **Mental Model** | User-curated summaries for common queries | "Team communication best practices" |
| **Observation** | Automatically consolidated knowledge from facts | "User was a React enthusiast but has now switched to Vue" (captures history) |
| **World Fact** | Objective facts received | "Alice works at Google" |
| **Experience Fact** | Bank's own actions and interactions | "I recommended Python to Bob" |

During reflect, the agent checks sources in priority order: **Mental Models → Observations → Raw Facts**.

### Multi-Strategy Retrieval (TEMPR)

Four search strategies run in parallel:

**Figure: Multi-Strategy Retrieval (TEMPR).** An animated diagram on the docs site; its narration, step by step:

- **recall()**
  1. recall() gets a query. Nothing is decided yet about which kind of search fits it best, so every arm that applies runs.
  2. Each arm searches its own index: meaning (vectors), exact words (BM25), the entity graph, and time. “March 2026” becomes a date range; a query with no date skips the time arm.
  3. All four point into the same memories. Each arm returns its own ranked list, and facts and observations compete in every one. The mark shows how many arms found each.
  4. RRF fusion merges the lists by rank, not raw score: a memory found near the top by several arms beats one found by a single arm.
  5. The top candidates (up to 300) go to a cross-encoder, which reads the query and each memory together and scores how well they match.
  6. Small multiplicative boosts nudge the score: recent memories, memories inside the asked time range, and observations backed by more evidence.
  7. Results are packed best-first until max_tokens is used up. Only the memory text counts toward the budget.
  8. The agent gets a short, ranked list it can put straight into its prompt.

| Strategy | Best for |
|----------|----------|
| **Semantic** | Conceptual similarity, paraphrasing |
| **Keyword (BM25)** | Names, technical terms, exact matches |
| **Graph** | Related entities, indirect connections |
| **Temporal** | "last spring", "in June", time ranges |

### Observation Consolidation

After memories are retained, Hindsight automatically consolidates related facts into **observations** — deduplicated, evidence-grounded beliefs that the bank has built up across many memories:

- **Deduplication**: Overlapping facts are merged into a single durable observation instead of piling up as repeats
- **Evidence tracking**: Each observation references the source memories (with exact quotes) that support it, plus a proof count
- **Continuous refinement**: Observations are updated — not overwritten — when new evidence supports, contradicts, or extends them; history is preserved
- **Freshness awareness**: when newer memories have been retained but not yet consolidated, `reflect` treats the affected observations as stale and verifies them against raw facts before relying on them

### Mission, Directives & Disposition

Memory banks can be configured to shape how the agent reasons during `reflect`:

| Configuration | Purpose | Example |
|---------------|---------|---------|
| **Mission** | Natural language identity for the bank | "I am a research assistant specializing in ML. I prefer simplicity over cutting-edge." |
| **Directives** | Hard rules the agent must follow | "Never recommend specific stocks", "Always cite sources" |
| **Disposition** | Soft traits that influence reasoning style | Skepticism, literalism, empathy (1-5 scale) |

The **mission** tells Hindsight what knowledge to prioritize and provides context for reasoning. **Directives** are guardrails and compliance rules that must never be violated. **Disposition traits** subtly influence interpretation style.

These settings only affect the `reflect` operation, not `recall`.

## Clients & Languages

<ClientsGrid />

## Integrations

Browse all supported integrations in the Integrations Hub.

## Next Steps

### Getting Started
- [**Quick Start**](api/quickstart.md) — Install and get up and running in 60 seconds
- [**RAG vs Hindsight**](rag-vs-hindsight.md) — See how Hindsight differs from traditional RAG with real examples

### Core Concepts
- [**Retain**](retain.md) — How memories are stored with multi-dimensional facts
- [**Recall**](retrieval.md) — How TEMPR's 4-way search retrieves memories
- [**Reflect**](reflect.md) — How mission, directives, and disposition shape reasoning

### API Methods
- [**Retain**](api/retain.md) — Store information in memory banks
- [**Recall**](api/recall.md) — Search and retrieve memories
- [**Reflect**](api/reflect.md) — Agentic reasoning with memory
- [**Mental Models**](api/mental-models.md) — User-curated summaries for common queries
- [**Memory Banks**](api/memory-banks.md) — Configure mission, directives, and disposition
- [**Documents**](api/documents.md) — Manage document sources
- [**Operations**](api/operations.md) — Monitor async tasks

### Deployment
- [**Server Setup**](installation.md) — Deploy with Docker Compose, Helm, or pip
