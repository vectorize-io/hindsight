---
sidebar_position: 2
---

# RAG vs Memory

Traditional RAG (Retrieval-Augmented Generation) retrieves documents similar to a query. Hindsight provides structured memory with temporal reasoning, entity understanding, and belief formation.

## Capability Comparison

| Capability | RAG | Hindsight |
|------------|-----|-----------|
| **Search strategy** | Semantic similarity only | Semantic + keyword + graph + temporal |
| **Multi-hop reasoning** | Limited to retrieved chunks | Graph traversal across entity relationships |
| **Temporal queries** | Keyword matching ("spring") | Date parsing and range filtering |
| **Entity understanding** | None | Entity resolution, observations, co-occurrence |
| **Belief formation** | Stateless | Opinions with confidence scores that evolve |
| **Disposition** | None | 3 traits (skepticism, literalism, empathy) influence interpretation |

## Architecture Comparison

### RAG

| Step | Operation |
|------|-----------|
| 1 | Embed query |
| 2 | Vector similarity search |
| 3 | Return top-k chunks |
| 4 | Generate response |

Single retrieval strategy. No state between queries.

### Hindsight

| Step | Operation |
|------|-----------|
| 1 | Parse query (extract temporal expressions, entities) |
| 2 | Execute 4 parallel retrievals: semantic, BM25, graph, temporal |
| 3 | Fuse results with RRF |
| 4 | Rerank with cross-encoder |
| 5 | Apply disposition traits |
| 6 | Generate response |

Multiple retrieval strategies. Persistent state across sessions.

## Example Scenarios

### Multi-Hop Reasoning

**Stored facts:**
- "Alice is the tech lead on Project Atlas"
- "Project Atlas uses Kubernetes"
- "Kubernetes cluster had an outage Tuesday"

**Query:** "Was Alice affected by recent issues?"

| System | Result |
|--------|--------|
| RAG | Retrieves facts about Alice only (no semantic similarity to "issues") |
| Hindsight | Traverses Alice → Project Atlas → Kubernetes → outage via entity links |

### Temporal Queries

**Stored facts with timestamps:**
- March: "Alice started microservices migration"
- April: "Alice completed auth service"
- October: "Alice focusing on performance"

**Query:** "What did Alice do last spring?"

| System | Result |
|--------|--------|
| RAG | Returns all Alice facts regardless of date |
| Hindsight | Parses "last spring" → March-May, filters to that range |

### Entity Understanding

**Stored facts about a user across sessions:**
- "Pro subscription"
- "Mobile app crashes in settings"
- "Switched to annual billing"
- "Desktop app working fine"

**Query:** "What do you know about my account?"

| System | Result |
|--------|--------|
| RAG | Lists disconnected facts |
| Hindsight | Returns synthesized entity observations: subscription status, billing, known issues |

### Belief Evolution

**Week 1:** User struggles with async Python, succeeds with threads
**Week 3:** User asks about asyncio, implements async database calls

| System | Behavior |
|--------|----------|
| RAG | No memory of progression |
| Hindsight | Forms opinion "user prefers sync" (0.7) → updates to "user growing comfortable with async" (0.6) |

## When to Use Each

| Use Case | Recommended |
|----------|-------------|
| Document Q&A over static corpus | RAG |
| Search with no temporal requirements | RAG |
| AI assistants with persistent memory | Hindsight |
| Applications requiring entity tracking | Hindsight |
| Systems needing consistent disposition | Hindsight |
| Temporal queries ("last month", "in 2023") | Hindsight |
