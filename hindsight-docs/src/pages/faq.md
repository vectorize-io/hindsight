---
title: Frequently Asked Questions
description: Common questions and answers about Hindsight
hide_table_of_contents: true
---

# Frequently Asked Questions

### What is Hindsight and how does it differ from RAG?

Hindsight is an agent memory system that provides long-term memory for AI agents using biomimetic data structures. Unlike traditional RAG (Retrieval-Augmented Generation), Hindsight:

- **Stores structured facts** instead of raw document chunks
- **Builds mental models** that consolidate knowledge over time
- **Uses graph-based relationships** between entities and concepts
- **Supports temporal reasoning** with time-aware retrieval
- **Enables disposition-aware reflection** for nuanced reasoning

For a detailed comparison, see [RAG vs Memory](/developer/rag-vs-hindsight).

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

### Which LLM providers are supported?

Hindsight supports:
- **OpenAI**
- **Anthropic**
- **Google Gemini**
- **Groq**
- **Ollama** (local models)
- **LM Studio** (local models)
- **Any OpenAI-compatible provider** (Together AI, Fireworks, DeepInfra, etc.)
- **Any Anthropic-compatible provider**

**Using local models with Ollama:**
```bash
HINDSIGHT_API_LLM_PROVIDER=ollama
HINDSIGHT_API_LLM_MODEL=llama3.1
HINDSIGHT_API_LLM_BASE_URL=http://localhost:11434
```

**Using local models with LM Studio:**
```bash
HINDSIGHT_API_LLM_PROVIDER=lmstudio
HINDSIGHT_API_LLM_MODEL=your-model-name
HINDSIGHT_API_LLM_BASE_URL=http://localhost:1234/v1
```

Configure your provider using the `HINDSIGHT_API_LLM_PROVIDER` environment variable. See [Configuration](/developer/configuration) and [Models](/developer/models) for details.

---

### Do I need to host my own infrastructure?

No! You have two options:

1. **Hindsight Cloud** - Fully managed service at [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io)
2. **Self-hosted** - Deploy on your own infrastructure using Docker or direct installation

See [Installation](/developer/installation) for self-hosting instructions.

---

### What are the minimum system requirements for self-hosting?

For running the Hindsight API server locally:
- Python 3.11+
- 4GB RAM minimum (8GB recommended for production)
- LLM API key (OpenAI, Anthropic, etc.) or local LLM setup

See [Installation](/developer/installation) for setup instructions.

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

Choose per-user banks for simplicity and privacy, or single bank with tags if you need holistic reasoning across users. See [Memory Banks](/developer/api/memory-banks) for management details.

--- 
### What's the difference between retain, recall, and reflect?

Hindsight has three core operations:

- **Retain**: Store data (facts, entities, relationships)
- **Recall**: Search and retrieve raw memory data based on a query
- **Reflect**: Use an AI agent to answer a query using retrieved memories

See [Operations](/developer/api/operations) for API details.

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

See [Recall](/developer/api/recall) and [Reflect](/developer/reflect) for full API details.

---

### When should I use mental models?

**Mental models** are consolidated knowledge patterns synthesized from individual facts over time. Use them when you need:

- Higher-level understanding beyond raw facts (e.g., "User prefers functional programming patterns")
- Long-term behavioral patterns (e.g., "Customer is price-sensitive but values quality")
- Context for AI agent reasoning during **reflect** operations

Mental models are automatically built during retain and used by reflect to provide richer, more contextual responses. See [Mental Models](/developer/api/mental-models).

---


### What's the typical latency for recall operations?

Typical latencies:
- **Without reranking**: 50-100ms
- **With reranking**: 200-500ms (depends on reranker model and installation)

See [Performance](/developer/performance) for tuning options.



## Still have questions?

Join our [Slack community](https://join.slack.com/t/hindsight-space/shared_invite/zt-3nhbm4w29-LeSJ5Ixi6j8PdiYOCPlOgg) or report issues on [GitHub](https://github.com/vectorize-io/hindsight/issues).
