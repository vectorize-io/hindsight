# DB Agent - Vector DB & RAG Design Guide

Use this file when the task involves vector databases, semantic retrieval, ANN tuning, or RAG.

## Core Position

- Vector DB is not a drop-in replacement for search engines or relational databases.
- Vector DB is usually an **index layer** for semantic retrieval.
- Canonical documents should live in an object store or primary database.
- Production retrieval systems are usually **hybrid**:
  - lexical retrieval for exactness and explainability
  - vector retrieval for semantic proximity
  - metadata filtering for governance and scope
  - reranking for precision

## Design Sequence

1. Define the retrieval problem
   - semantic discovery
   - exact term retrieval
   - compliance / tenant filtering
   - recommendation
   - agent context retrieval
2. Decide whether vector retrieval is justified
3. Design chunking and embedding strategy
4. Choose ANN index and benchmark it
5. Design filtering, reranking, and observability
6. Plan lifecycle:
   - re-embedding
   - re-indexing
   - hot/cold vector tiers
   - backup / restore

## Best Practices

### 1. Use Hybrid Retrieval by Default

- Keep lexical retrieval when:
  - exact term match matters
  - user expects deterministic ranking
  - compliance or audit filters must be explicit
  - explainability matters
- Good default pipeline:
  - filter scope
  - lexical retrieval
  - ANN retrieval
  - merge / dedupe
  - rerank
  - context compression

### 2. Govern Embeddings Like Schema

Store in metadata:

- embedding model name
- dimension
- normalization method
- chunking policy version
- preprocessing version
- creation timestamp

Plan for:

- rolling re-embedding
- shadow indexes
- side-by-side evaluation
- backfill and cutover

### 3. Chunking is a First-Class Design Decision

Choose chunking based on document type:

- contracts / policies: heading-aware chunks with overlap
- tickets / chats: conversation-turn boundaries
- code / config: syntax-aware or file-structure-aware chunking
- FAQs / short docs: smaller direct-answer chunks

Watch for:

- over-chunking -> weak context and low recall
- under-chunking -> noisy retrieval and token waste

### 4. Tune ANN with Real Data

Common ANN choices:

- HNSW: strong recall, high memory
- IVF: faster partitioned retrieval, recall tradeoffs
- PQ: smaller memory footprint, lower precision
- DiskANN: disk-oriented large-scale retrieval

Benchmark on production-like distributions, not synthetic random vectors.

Measure:

- recall@k
- latency p50 / p95 / p99
- memory footprint
- index build time
- update latency

### 5. Design Filtering Deliberately

Metadata filtering gets expensive with high-cardinality dimensions.

Check:

- whether filters are pushed down before ANN or after
- whether approximate filtering affects correctness
- whether tenant / user / region isolation needs separate partitions

Patterns:

- partition by tenant or region
- pre-segment indexes for extreme-cardinality filters
- maintain dedicated collections for very large tenants

### 6. Plan Writes and Lifecycle

High-write vector workloads need:

- ingestion buffers
- async indexing
- compaction
- periodic rebuild strategy
- separation of ingestion and search responsibilities when load is high

Consider:

- hot vectors in fast mutable index
- cold vectors in slower compacted tier

### 7. Observe Semantics, Not Just Infra

Track more than QPS and latency:

- recall drift
- semantic drift after model update
- similarity score distribution
- false positives
- empty-result rate
- prompt-token waste from irrelevant retrieval

Maintain:

- golden query sets
- offline evaluation pipelines
- before/after comparison when changing model or chunking

## Review Questions

- Why is vector retrieval needed here?
- Why is pure lexical retrieval insufficient?
- Why is pure vector retrieval insufficient?
- What is the source of truth?
- What invalidates the embeddings?
- How will re-embedding be rolled out safely?
- How are filters enforced?
- How is retrieval quality measured over time?
