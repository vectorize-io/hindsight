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

2. **Batch your operations**: Group related content into batch requests. Send as much data as you want in a single request — the only limit is the HTTP payload size.

3. **Use async mode for large datasets**: Queue operations in the background

4. **Parallel processing**: For very large datasets, use multiple concurrent retention requests with different `document_id` values

### Automatic Batch Optimization

**When using async retain, Hindsight automatically handles batch sizing for you.** You don't need to manually tune batch sizes or worry about optimal chunking.

How it works:
- **Send large batches**: Submit hundreds or thousands of items in a single async retain request
- **Automatic splitting**: Hindsight automatically splits large batches (>10,000 tokens) into optimized sub-batches
- **Parallel processing**: Sub-batches are processed concurrently in the background
- **Status tracking**: Parent operation aggregates status from all sub-batches
- **Token-based**: Batching uses tiktoken for accurate token counting, not character counts

Benefits:
- Send entire documents or datasets in one API call
- Let Hindsight optimize the processing strategy
- Track overall progress via the parent operation status
- No need to manually split data into small batches

### Throughput

Factors affecting throughput:
- Document size and complexity
- LLM provider rate limits (for fact extraction)
- Database write performance
- Available CPU/memory resources

---

## Tuning for Local & Small Environments

Hindsight's defaults are tuned for cloud LLM providers and multi-core servers. When you run it on a laptop, a single GPU box, or against a **local LLM server** (llama.cpp, vLLM, LM Studio, Ollama) with a small fixed slot pool, those defaults can saturate the backend, time out, or thrash the CPU. This section collects the knobs that matter for low-resource setups.

### LLM concurrency

The default `HINDSIGHT_API_LLM_MAX_CONCURRENT=32` assumes a cloud provider that can absorb dozens of parallel requests. A local server with a handful of slots cannot — Hindsight will fill every slot and **starve any other client sharing the endpoint** (your main agent, another app, or a second Hindsight operation).

```bash
export HINDSIGHT_API_LLM_MAX_CONCURRENT=2
```

A value of `2` lets retain and consolidation run concurrently without blocking each other. If the endpoint is **shared** with other clients (other applications, agents, or workflows hitting the same llama-server / vLLM / LM Studio instance), reserve slots for them by lowering further — leave at least one slot free per shared client.

You can also split the budget per operation so background work never crowds out live reads. The per-operation caps compose *on top of* the global cap:

```bash
# global=4, with retain/consolidation capped low so reflect always has headroom
export HINDSIGHT_API_LLM_MAX_CONCURRENT=4
export HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT=1
export HINDSIGHT_API_CONSOLIDATION_LLM_MAX_CONCURRENT=1
```

### Timeouts and retries

Small models on modest hardware generate tokens slowly, and the first request after startup pays a model-load cost. The default `HINDSIGHT_API_LLM_TIMEOUT=120` (seconds) can be too tight for a large local model on CPU — raise it to avoid spurious timeouts and wasted retries:

```bash
export HINDSIGHT_API_LLM_TIMEOUT=300        # allow slow local generation
export HINDSIGHT_API_LLM_MAX_RETRIES=2      # fail faster locally — retries rarely help a slow box
```

A local endpoint isn't rate-limited, so aggressive retry/backoff mostly adds latency on real failures. Lower retries and let genuine errors surface quickly.

### Smaller, faster models — and reasoning effort

Retain (fact extraction) is structured work that does not need a frontier model; reflect can use a lighter model still. On a constrained box, point each operation at the smallest model that holds up:

```bash
# Reflect on a small/fast model; retain on a slightly stronger structured-output model
export HINDSIGHT_API_REFLECT_LLM_MODEL=<small-fast-model>
export HINDSIGHT_API_RETAIN_LLM_MODEL=<structured-output-model>
```

If your model exposes a reasoning/thinking budget, keep it low (the default) — extra reasoning tokens are pure latency for the extraction and consolidation paths:

```bash
export HINDSIGHT_API_LLM_REASONING_EFFORT=low
```

Consolidation sends multiple facts to the LLM in a single call (default 8). On a small model with a limited context window, a large batch produces an oversized prompt and a long, error-prone response. Shrink the batch so each consolidation call stays small and reliable:

```bash
export HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE=2   # default 8; lower = smaller prompts, more calls
```

### Built-in llama.cpp tuning

The bundled `llamacpp` provider runs a llama.cpp server as a managed subprocess — no external server needed. Key knobs for small machines:

```bash
export HINDSIGHT_API_LLM_PROVIDER=llamacpp
export HINDSIGHT_API_LLM_MAX_CONCURRENT=2        # retain + consolidation without blocking
export HINDSIGHT_API_LLAMACPP_GPU_LAYERS=-1      # -1 = offload all layers to GPU; 0 = CPU only
export HINDSIGHT_API_LLAMACPP_CONTEXT_SIZE=8192  # lower to save RAM/VRAM; raise for big batches
export HINDSIGHT_API_LLAMACPP_EXTRA_ARGS="--n_threads 8"  # match physical cores on CPU-only boxes
# export HINDSIGHT_API_LLAMACPP_NO_GRAMMAR=true  # faster, but less reliable JSON output
```

See [Built-in llama.cpp](./configuration#built-in-llamacpp) for the full option list.

### Reranker on CPU

Recall's bottleneck on a machine without a GPU is the cross-encoder reranker. The local reranker has several CPU/Apple-Silicon knobs that are quality-neutral but materially faster:

```bash
# Apple Silicon (MPS): half precision is 27–36% faster, quality-identical
export HINDSIGHT_API_RERANKER_LOCAL_FP16=true

# Sort pairs by length before batching — 36–54% faster, quality-identical by construction
export HINDSIGHT_API_RERANKER_LOCAL_BUCKET_BATCHING=true

# Cap reranker parallelism so it doesn't thrash a small CPU under load (default 4)
export HINDSIGHT_API_RERANKER_LOCAL_MAX_CONCURRENT=2

# On macOS, force CPU if MPS/XPC causes instability
# export HINDSIGHT_API_RERANKER_LOCAL_FORCE_CPU=true
```

The biggest single win on CPU is reranking fewer candidates. By default Hindsight reranks up to 300 candidates per recall — shrink that pool to cut cross-encoder work proportionally:

```bash
export HINDSIGHT_API_RERANKER_MAX_CANDIDATES=100   # default 300; RRF pre-filters the rest
```

For a pure-CPU box that struggles with the cross-encoder, `flashrank` is a lighter ONNX-based reranker:

```bash
export HINDSIGHT_API_RERANKER_PROVIDER=flashrank
```

You can also reduce recall work directly: use a lower `budget` (`low`/`mid`) for everyday queries and reserve `high` for comprehensive reasoning. See [Recall Performance](#recall-performance) below.

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
3. **Include chunks**: Use `include_chunks` to retrieve the raw text that generated memories when you need additional context

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
- **Enable provider Batch API**: Set `HINDSIGHT_API_RETAIN_BATCH_ENABLED=true` with async retain to cut LLM fact-extraction costs by 50% (supported on OpenAI and Groq; results delivered within 24 hours)
- **Control token budgets**: Limit `max_tokens` for recall, use lower budgets when possible
- **Optimize chunks**: Larger chunks (1000-2000 tokens) are more efficient than many small ones

### Monitoring
- **Prometheus metrics**: Available at `/metrics` — track latency percentiles, throughput, and error rates
- **Key metrics**: `hindsight_recall_duration_seconds`, `hindsight_reflect_duration_seconds`, `hindsight_retain_items_total`
