

<PageHero title="FAQ" subtitle="Common questions and answers about Hindsight." />

**Contents**
- [What is Hindsight and how does it differ from RAG?](#what-is-hindsight-and-how-does-it-differ-from-rag)
- [Why use Hindsight instead of other solutions?](#why-use-hindsight-instead-of-other-solutions)
- [Supported clients, integrations, and LLM providers](#which-clients-and-languages-are-supported)
- [Which model should I use?](#which-model-should-i-use-with-hindsight)
- [Hosting and system requirements](#do-i-need-to-host-my-own-infrastructure)
- [What are "zombie" operations and how do I recover them?](#what-are-zombie-operations-and-how-do-i-recover-them)
- [How do I isolate user data?](#how-do-i-isolate-user-data)
- [Retain, recall, and reflect — what's the difference?](#whats-the-difference-between-retain-recall-and-reflect)
- [When should I use recall vs reflect?](#when-should-i-use-recall-vs-reflect)
- [When should I use mental models?](#when-should-i-use-mental-models)
- [Latency expectations](#whats-the-typical-latency-for-recall-operations)
- [Tags, metadata, and entity labels](#does-hindsight-support-metadata-filtering)
- [Controlling which memory types are recalled](#how-do-i-control-which-types-of-memories-are-recalled)
- [Recommended format for conversations](#what-is-the-recommended-format-for-retaining-conversations)

---

### What is Hindsight and how does it differ from RAG?

Hindsight is an agent memory system that provides long-term memory for AI agents using biomimetic data structures. Unlike traditional RAG (Retrieval-Augmented Generation), Hindsight:

- **Stores structured facts** instead of raw document chunks
- **Builds mental models** that consolidate knowledge over time
- **Uses graph-based relationships** between entities and concepts
- **Supports temporal reasoning** with time-aware retrieval
- **Enables disposition-aware reflection** for nuanced reasoning

For a detailed comparison, see [RAG vs Memory](developer/rag-vs-hindsight.md).

---

### Why use Hindsight instead of other solutions?

Hindsight is purpose-built for agent memory with unique advantages:

- **State-of-the-art accuracy**: Ranked #1 LongMemEval benchmarks for agent memory (see [details](https://benchmarks.hindsight.vectorize.io/))
- **Built on proven technology**: PostgreSQL - battle-tested, reliable, and widely understood
- **Cloud-native architecture**: Designed for modern cloud deployments with horizontal scalability
- **Flexible deployment**: Self-host or use Hindsight Cloud - works with any LLM provider
- **True long-term memory**: Builds mental models that consolidate knowledge over time, not just retrieval
- **Graph-based reasoning**: Understands relationships between entities and concepts for richer context
- **Production-ready**: Scales to millions of memories with 50-500ms recall latency
- **Developer-friendly**: Simple APIs (retain, recall, reflect), SDKs for Python/TypeScript/Go/Rust, integrations with LiteLLM/Vercel AI SDK

Unlike vector databases (just search) or RAG systems (document retrieval), Hindsight provides **living memory** that evolves with your users.

---

### Which clients and languages are supported?

<ClientsGrid />

---

### Which integrations are supported?

Browse all supported integrations in the Integrations Hub.

---

### Which LLM providers are supported?

- OpenAI
- Anthropic
- Google Gemini
- Vertex AI
- Groq
- Ollama
- LM Studio
- llama.cpp
- MiniMax
- DeepSeek
- Volcano Engine
- OpenRouter
- OpenAI Codex
- Claude Code
- AWS Bedrock
- OpenAI Compatible
- LiteLLM (100+)

See [Models](developer/models.md) for the full list of supported providers, recommended models, and configuration examples.

---

### Which model should I use with Hindsight?

The **[Model Leaderboard](https://benchmarks.hindsight.vectorize.io/)** benchmarks models across accuracy, speed, cost, and reliability for retain, reflect, and observation consolidation — it's the best place to find the right trade-off for your use case.

[](https://benchmarks.hindsight.vectorize.io/)

See [Models](developer/models.md) for the full list of supported and tested models, provider defaults, and configuration examples.

---

### Do I need to host my own infrastructure?

No! You have two options:

1. **Hindsight Cloud** - Fully managed service at [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io)
2. **Self-hosted** - Deploy on your own infrastructure using Docker or direct installation

See [Installation](developer/installation.md) for self-hosting instructions.

---

### What are the minimum system requirements for self-hosting?

For running the Hindsight API server locally:
- Python 3.11+
- 4GB RAM minimum (8GB recommended for production)
- LLM API key (OpenAI, Anthropic, etc.) or local LLM setup

See [Installation](developer/installation.md) for setup instructions.

---

### What are "zombie" operations and how do I recover them?

A **zombie operation** is a background task stuck in `processing` indefinitely because the worker that claimed it is gone — typically after a Docker container restart. The symptom is a `pending_consolidation` (or similar) counter that never decreases on `/banks/{bank_id}/stats`, even though the worker logs show plenty of free slots.

The root cause is almost always an unstable `HINDSIGHT_API_WORKER_ID`. By default the worker uses the container hostname as its identity, and Docker rotates that on every restart — so the new container has a different ID and won't recognize the old worker's claims as its own.

**Recover** with the admin CLI:

```bash
# If you know which worker is dead:
hindsight-admin decommission-worker <old-worker-id>

# Or, fleet-wide release across all workers:
hindsight-admin decommission-workers
```

**Prevent it** by setting `HINDSIGHT_API_WORKER_ID` to a stable value (Docker `-e HINDSIGHT_API_WORKER_ID=...`, or `--worker-id` on bare metal). The Helm chart already handles this — its StatefulSet wires the pod name automatically.

See [Admin CLI — Recovering stuck operations](developer/admin-cli.md#recovering-stuck-or-zombie-operations) for the full diagnosis and recovery flow.

---

### How do I isolate user data?

A **memory bank** is an isolated memory store (like a "brain") that contains its own memories, entities, relationships, and optional disposition traits (skepticism, literalism, empathy). Banks are completely isolated from each other with no data leakage.

There are two approaches for multi-user applications:

**1. Per-user memory banks** (recommended for most use cases)
- Create one bank per user (e.g., `bank_id="user-123"`)
- Easiest setup and strongest data isolation
- Perfect for per-user queries and personalization
- Each bank can have unique disposition traits and background context
- **Limitation**: Cannot perform cross-user analysis (e.g., "What is the most mentioned topic across all users?")

**2. Single bank with tags** (for applications needing aggregated insights)
- Use one bank for the entire application
- Tag memories with user identifiers during retain (e.g., `tags={"user_id": "user-123"}`)
- Filter by tags during recall/reflect for per-user queries
- **Advantage**: Enables both per-user AND cross-user queries (e.g., analyze specific users or aggregate across all users)

Choose per-user banks for simplicity and privacy, or single bank with tags if you need holistic reasoning across users. See [Memory Banks](developer/api/memory-banks.md) for management details.

---

### What's the difference between retain, recall, and reflect?

Hindsight has three core operations:

- **Retain**: Store data (facts, entities, relationships)
- **Recall**: Search and retrieve raw memory data based on a query
- **Reflect**: Use an AI agent to answer a query using retrieved memories

See [Operations](developer/api/operations.md) for API details.

---

### When should I use recall vs reflect?

**Use recall when:**
- You want raw facts to feed into your own reasoning or prompt
- You need maximum control over how memories are interpreted
- You're doing simple fact lookup (e.g., "What did Alice say about X?")
- Latency is critical — recall is significantly faster (50-500ms vs 1-10s)
- You want to build your own answer synthesis layer on top of retrieved memories

**Use reflect when:**
- You want a ready-to-use answer generated from memories (no extra LLM call needed)
- You need disposition-aware responses shaped by the bank's personality traits (skepticism, literalism, empathy)
- The query requires multi-step reasoning across facts, observations, and mental models
- You need structured output (via `response_schema`) from memory-grounded reasoning
- You want citations — reflect returns which memories, mental models, and directives informed the answer

**Key difference**: Recall returns data; reflect returns an answer. Recall gives you raw materials, reflect does the reasoning for you using the bank's disposition and an autonomous search loop.

```
recall("What food does Alice like?")
→ ["Alice loves sushi", "Alice prefers vegetarian options"]   # raw facts

reflect("What should I order for Alice?")
→ "I'd recommend a vegetarian sushi platter — Alice loves sushi and prefers vegetarian options."  # grounded answer
```

See [Recall](developer/api/recall.md) and [Reflect](developer/reflect.md) for full API details.

---

### When should I use mental models?

**Mental models** are consolidated knowledge patterns synthesized from individual facts over time. Use them when you need:

- Higher-level understanding beyond raw facts (e.g., "User prefers functional programming patterns")
- Long-term behavioral patterns (e.g., "Customer is price-sensitive but values quality")
- Context for AI agent reasoning during **reflect** operations

Mental models are automatically built during retain and used by reflect to provide richer, more contextual responses. See [Mental Models](developer/api/mental-models.md).

---

### What's the typical latency for recall operations?

Typical latencies:
- **Without reranking**: 50-100ms
- **With reranking**: 200-500ms (depends on reranker model and installation)

See [Performance](developer/performance.md) for tuning options.

---

### Does Hindsight support metadata filtering?

Yes — through **Tags**. Tags are string labels attached to memories at retain time and used as a visibility filter at recall/reflect time. Only memories tagged with a matching value are returned.

```python
# Tag memories at retain time
client.retain(bank_id="my-bank", items=[{
    "content": "...",
    "tags": ["user:alice"],
}])

# Filter by tag at recall time
client.recall(bank_id="my-bank", query="...", tags=["user:alice"])
```

See [Tags](developer/api/retain.md#tags-and-document_tags) for full details including document-level tagging.

**What about filtering by entities?**

Entities (people, places, concepts) extracted from memories are stored in the knowledge graph and drive graph-based retrieval — so querying "tell me about Alice" will naturally surface Alice-related memories without any manual filtering.

If you need explicit tag-based filtering on entity-like values, use **entity labels** with `tag: true`. Entity labels let you define a controlled vocabulary of `key:value` classifiers (e.g. `user:alice`, `topic:algebra`) extracted at retain time. Setting `tag: true` on a label group automatically writes each extracted label as a tag on the memory unit, making them available for standard `tags`/`tags_match` filtering:

```python
# Bank config: entity label group with tag: true
{
    "entity_labels": [{
        "key": "user",
        "type": "text",
        "tag": True,
        "description": "The user this memory belongs to"
    }]
}

# The label "user:alice" is extracted and also written as a tag
# Filter at recall time using the standard tags parameter
client.recall(bank_id="my-bank", query="...", tags=["user:alice"])
```

See [Entity Labels](developer/retain.md#entity-labels) for configuration details.

**What about document `metadata`?**

Document metadata (the `metadata` key-value pairs on a retain item) serves a different purpose. It is:
- **Included in the fact extraction prompt**, so the LLM can use it as additional context when extracting facts — for example, knowing the document title or source can improve accuracy.
- **Returned with every recalled memory** as-is, so your application can link memories back to source systems (e.g. a URL, thread ID, or ticket number) without extra lookups.

Metadata is not a filter — use tags when you need recall to be scoped to a subset of documents.

---

### How do I control which types of memories are recalled?

If your bank mixes different shapes of memory (e.g., concise rules and detailed procedures) and recall surfaces the wrong shape for a given query, use **entity labels** with `tag: true` to classify facts during retain and hard-filter them during recall.

1. Define a label group on the bank with `tag: true` and a controlled vocabulary (e.g., `rule` vs `procedure`)
2. Retain normally — the LLM classifies each extracted fact automatically
3. Pass `tags=["memory_type:rule"]` and `tags_match="any_strict"` at recall time to deterministically include only matching memories

This is a SQL-level filter applied before ranking, not a scoring signal — the excluded memories never enter the retrieval pipeline. This is more reliable than adjusting ranking weights, which only nudge continuous scores and cannot guarantee ordering.

See [Best Practices — Filtering by Memory Shape](best-practices.md#filtering-by-memory-shape-with-entity-labels) for a full walkthrough, or [Entity Labels](developer/api/memory-banks.md#entity-labels) for the configuration reference.

---

### What is the recommended format for retaining conversations?

Pass the **entire conversation as a single document** and upsert it as the conversation grows — Hindsight chunks it automatically, so you don't need to split it yourself.

**Preferred format: JSON array**

```json
[
  {"role": "user",      "content": "I moved to Berlin last month."},
  {"role": "assistant", "content": "How are you finding it?"},
  {"role": "user",      "content": "Love it, especially the food scene."}
]
```

Hindsight has internal chunking optimizations for the JSON array format, since it's the most common conversation shape.

**Alternative: prefixed plain text**

```
[2025-06-01T10:32:00Z] user: I moved to Berlin last month.
[2025-06-01T10:32:05Z] assistant: How are you finding it?
[2025-06-01T10:32:20Z] user: Love it, especially the food scene.
```

Adding a username and timestamp prefix to each message improves extraction quality — the LLM uses those signals to attribute facts correctly and reason about timing.

**Use a stable document ID to upsert:**

```python
await client.retain(
    bank_id="my-bank",
    documents=[{
        "id": "chat-session-abc123",  # stable ID enables upsert
        "content": conversation,       # full conversation so far
    }]
)
```

Re-retaining with the same `id` replaces the old document and its facts, so you won't accumulate duplicates as the conversation grows.

**Don't pre-summarize or pre-extract facts.** Hindsight does this automatically and needs the full conversation for context — a message like "yes, exactly" or "I'll go with option 2" is meaningless without the surrounding exchange.

---

## Still have questions?

Join our [Slack community](https://join.slack.com/t/hindsight-space/shared_invite/zt-3nhbm4w29-LeSJ5Ixi6j8PdiYOCPlOgg) or report issues on [GitHub](https://github.com/vectorize-io/hindsight/issues).
