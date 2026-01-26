# Performance

Hindsight is designed for high-performance semantic memory operations at scale. This page covers performance characteristics, optimization strategies, and best practices.

## Overview

Hindsight's performance is optimized across three key operations:

- **Retain (Ingestion)**: Batch processing with async operations for large-scale memory storage
- **Recall (Search)**: Sub-second semantic search with configurable thinking budgets
- **Reflect (Reasoning)**: Disposition-aware answer generation with controllable compute

## Design Philosophy: Optimized for Fast Reads

Hindsight is **architected from the ground up to prioritize read performance over write performance**. This design decision reflects the typical usage pattern of memory systems: memories are written once but read many times.

The system makes deliberate trade-offs to ensure **sub-second recall operations**:

- **Pre-computed embeddings**: All memory embeddings are generated and indexed during retention
- **Optimized vector search**: HNSW indexes enable fast approximate nearest neighbor search
- **Fact extraction at write time**: Complex LLM-based fact extraction happens during retention, not retrieval
- **Structured memory graphs**: Relationships and temporal information are resolved upfront

This means **Recall (search) operations are blazingly fast** because all the heavy lifting has already been done.

### Performance Comparison

| Operation | Typical Latency | Primary Bottleneck | Optimization Strategy            |
|-----------|----------------|-------------------|----------------------------------|
| **Recall** | 100-600ms | Re-ranker (on CPU) | Use GPU for re-ranking, or reduce budget |
| **Reflect** | 800-3000ms | LLM generation | Use faster LLM                   |
| **Retain** | 500ms-2000ms per batch | **LLM fact extraction** | Use high-throughput LLM provider |

Hindsight is designed to ensure your **application's read path (recall/reflect) is always fast**, even if it means spending more time upfront during writes. This is the right trade-off for memory systems where:

- Memories are retained in background processes or during low-traffic periods
- Memories are queried frequently in user-facing, latency-sensitive contexts
- The ratio of reads to writes is high (typically 10:1 or higher)

---

## Retain Performance

**Retain (write) operations are inherently slower** because they involve LLM-based fact extraction, entity recognition, temporal reasoning, relationship mapping, and embedding generation. **The LLM is the primary bottleneck for write latency.**

### Hindsight Doesn't Need a Smart Model

The fact extraction process is structured and well-defined, so smaller, faster models work extremely well. Our recommended model is `gpt-oss-20b` (available via Groq and other providers).

To maximize retention throughput:

1. **Use high-throughput LLM providers**: Choose providers with high requests-per-minute (RPM) limits and low latency
   - **Fast**: [Groq](https://groq.com) with `gpt-oss-20b` or other openai-oss models, self-hosted models on GPU clusters (vLLM, TGI)
   - **Slow**: Standard cloud LLM providers with rate limits

2. **Batch your operations**: Group related content into batch requests. The only limit is the HTTP payload size — Hindsight automatically splits large batches into smaller, optimized chunks under the hood, so you don't have to worry about it.

3. **Use async mode for large datasets**: Queue operations in the background

4. **Parallel processing**: For very large datasets, use multiple concurrent retention requests with different `document_id` values

### Throughput

Factors affecting throughput:
- Document size and complexity
- LLM provider rate limits (for fact extraction)
- Database write performance
- Available CPU/memory resources

---

## Recall Performance

### Budget

The `budget` parameter controls the search depth and quality. Choose based on query complexity — comprehensive questions that need thorough analysis benefit from higher budgets:

| Budget | Use Case |
|--------|----------|
| `low` | Quick lookups, real-time chat |
| `mid` | Standard queries, balanced performance |
| `high` | Comprehensive questions, thorough analysis |

### Optimization

1. **Appropriate budgets**: Use lower budgets for simple queries, higher for comprehensive reasoning
2. **Limit result tokens**: Set `max_tokens` to control response size (default: 4096)
3. **Include entities/chunks**: Use `include_entities` and `include_chunks` to retrieve additional context when needed — each has its own token budget

### Database Performance

Hindsight uses PostgreSQL with pgvector for efficient vector search:

- **Index type**: HNSW for approximate nearest neighbor search
- **Typical query time**: 10-50ms for vector search on 100K+ facts
- **Scalability**: Tested with millions of facts per bank

## Reflect Performance

### Performance Characteristics

| Component | Latency        | Description |
|-----------|----------------|-------------|
| Memory search | 100-600ms      | Based on budget (low/mid/high) |
| LLM generation | 500-2000ms     | Depends on provider and response length |
| **Total** | **600-2600ms** | Typical end-to-end latency |

### Optimization Strategies

1. **Budget selection**: Use lower budgets when context is sufficient
2. **Context provision**: Provide relevant `context` to reduce recall requirements and steer towards more focused answers

## Best Practices

### Operations
- **Use appropriate budgets**: Don't over-provision for simple queries; use higher budgets for comprehensive reasoning
- **Batch retain operations**: Group related content together for better efficiency
- **Cache frequent queries**: Cache at the application level for repeated queries
- **Profile with trace**: Use the `trace` parameter to identify slow operations

### Scaling
- **Horizontal scaling**: Deploy multiple API instances behind a load balancer with shared PostgreSQL
- **Concurrency**: 100+ simultaneous requests supported; memory search scales with CPU cores
- **LLM rate limits**: Distribute load across multiple API keys/providers (typically 60-500 RPM per key)

### Cost Optimization
- **Use efficient models**: `gpt-oss-20b` via Groq for retain — Hindsight doesn't need frontier models
- **Control token budgets**: Limit `max_tokens` for recall, use lower budgets when possible
- **Optimize chunks**: Larger chunks (1000-2000 tokens) are more efficient than many small ones

### Monitoring
- **Prometheus metrics**: Available at `/metrics` — track latency percentiles, throughput, and error rates
- **Key metrics**: `hindsight_recall_duration_seconds`, `hindsight_reflect_duration_seconds`, `hindsight_retain_items_total`
