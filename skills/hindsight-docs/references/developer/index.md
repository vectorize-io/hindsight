

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

```mermaid
graph LR
    subgraph app["<b>Your Application</b>"]
        Agent[AI Agent]
    end

    subgraph hindsight["<b>Hindsight</b>"]
        API[API Server]

        subgraph bank["<b>Memory Bank</b>"]
            direction TB
            MentalModels[Mental Models]
            Observations[Observations]
            MemEnt[Memories & Entities]
            Chunks[Chunks]
            Documents[Documents]

            MentalModels --> Observations --> MemEnt --> Chunks --> Documents
        end
    end

    Agent -->|retain| API
    Agent -->|recall| API
    Agent -->|reflect| API

    API --> bank
```

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

```mermaid
graph LR
    Q[Query] --> S[Semantic]
    Q --> K[Keyword]
    Q --> G[Graph]
    Q --> T[Temporal]

    S --> RRF[RRF Fusion]
    K --> RRF
    G --> RRF
    T --> RRF

    RRF --> CE[Cross-Encoder]
    CE --> R[Results]
```

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
- **Freshness trend**: Each observation carries a computed trend (stable / strengthening / weakening / stale) based on when its evidence arrived

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
