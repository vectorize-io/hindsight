---
sidebar_position: 1
slug: /
---

# Overview

## Why Hindsight?

AI assistants forget everything between sessions. Every conversation starts from zero—no context about who you are, what you've discussed, or what the agent has learned. This isn't just inconvenient; it fundamentally limits what AI agents can do.

**The problem is harder than it looks:**

- **Simple vector search isn't enough** — "What did Alice do last spring?" requires temporal reasoning, not just semantic similarity
- **Facts get disconnected** — Knowing "Alice works at Google" and "Google is in Mountain View" should let you answer "Where does Alice work?" even if you never stored that directly
- **Agents need opinions** — A coding assistant that remembers "the user prefers functional programming" should weigh that when making recommendations
- **Context matters** — The same information means different things to different agents with different personalities

Hindsight solves these problems with a memory system designed specifically for AI agents.

## What Hindsight Does

```mermaid
graph LR
    subgraph Clients
        A[Python Client]
        B[Node.js Client]
        C[CLI]
        D[AI Assistants]
    end

    subgraph Hindsight Server
        E[HTTP API]
        F[MCP API]
    end

    A --> E
    B --> E
    C --> E
    D --> F

    E --> G[Memory Engine]
    F --> G

    G --> H[(PostgreSQL + pgvector)]
```

**Store** conversations and documents → **Search** with multi-strategy retrieval → **Think** with personality-aware reasoning

## Architecture

```mermaid
graph TB
    subgraph Input
        I1[Raw Text]
        I2[Conversations]
        I3[Documents]
    end

    subgraph Ingestion
        E1[LLM Extraction]
        E2[Entity Resolution]
        E3[Graph Construction]
    end

    subgraph Storage
        S1[World Facts]
        S2[Agent Facts]
        S3[Opinions]
        S4[Entity Graph]
    end

    subgraph Retrieval
        R1[Semantic Search]
        R2[Keyword Search]
        R3[Graph Traversal]
        R4[Temporal Search]
        R5[RRF Fusion]
        R6[Cross-Encoder Rerank]
    end

    subgraph Output
        O1[Search Results]
        O2[Think Response]
    end

    I1 --> E1
    I2 --> E1
    I3 --> E1
    E1 --> E2
    E2 --> E3
    E3 --> S1
    E3 --> S2
    E3 --> S3
    E3 --> S4

    S1 --> R1
    S1 --> R2
    S4 --> R3
    S1 --> R4

    R1 --> R5
    R2 --> R5
    R3 --> R5
    R4 --> R5
    R5 --> R6
    R6 --> O1
    R6 --> O2
```

## Key Components

### Three Memory Networks

Hindsight separates memories by type for epistemic clarity:

| Network | What it stores | Example |
|---------|----------------|---------|
| **World** | Objective facts received | "Alice works at Google" |
| **Agent** | Agent's own actions | "I recommended Python to Bob" |
| **Opinion** | Formed beliefs + confidence | "Python is best for ML" (0.85) |

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

### Personality Framework (CARA)

Agents have Big Five personality traits that influence opinion formation:

| Trait | Low | High |
|-------|-----|------|
| **Openness** | Prefers proven methods | Embraces new ideas |
| **Conscientiousness** | Flexible, spontaneous | Systematic, organized |
| **Extraversion** | Independent | Collaborative |
| **Agreeableness** | Direct, analytical | Diplomatic, harmonious |
| **Neuroticism** | Calm, optimistic | Risk-aware, cautious |

The `bias_strength` parameter (0-1) controls how much personality influences opinions.

## Client-Server Interaction

```mermaid
sequenceDiagram
    participant C as Client
    participant A as Hindsight API
    participant DB as PostgreSQL

    C->>A: store("Alice works at Google")
    A->>A: Extract facts & entities
    A->>A: Build graph links
    A->>DB: Store memory units
    A-->>C: Success

    C->>A: search("What does Alice do?")
    A->>DB: 4-way parallel search
    A->>A: RRF fusion + rerank
    A-->>C: Ranked results

    C->>A: think("Tell me about Alice")
    A->>DB: Retrieve relevant memories
    A->>A: Generate with personality
    A-->>C: Response + sources
```

## Next Steps

- [Quick Start](/developer/api/quickstart) — Get up and running in 60 seconds

- [Architecture](./developer/architecture) — Deep dive into ingestion, storage, and graph construction
- [Retrieval](./developer/retrieval) — How TEMPR's 4-way search works
- [Personality](./developer/personality) — CARA framework and opinion formation
- [Ingest Data](./developer/api/ingest) — Store memories, conversations, and documents
- [Search Facts](./developer/api/search) — Multi-strategy retrieval
- [Think](./developer/api/think) — Personality-aware response generation
- [Server Deployment](./developer/server) — Deploy with Docker Compose, Helm, or pip
- [Development Guide](./developer/development) — Set up a local development environment
