---
title: "Run Hindsight with Ollama: Local AI Memory, No API Keys Needed"
authors: [hindsight]
date: 2026-03-10T12:00
tags: [ollama, tutorial, python, memory, local, privacy, hindsight, llm, open-source]
image: /img/blog/run-hindsight-with-ollama.png
hide_table_of_contents: true
---

Running Hindsight with Ollama gives you a fully local AI memory system. No API keys, no cloud costs, no data leaving your machine. If you want persistent agent memory powered by open-source models on your own hardware, this tutorial walks through the complete setup.

<!-- truncate -->

## TL;DR

- Run Hindsight with Ollama entirely on your machine, with no API keys, no cloud, and no costs
- Ollama provides the LLM. Hindsight provides the memory engine. Two env vars connect them.
- `retain` and `recall` work great with local models
- `reflect` requires a model that supports tool calling (not all local models do)
- Expect slower inference than cloud APIs, but you get full privacy and zero per-token costs

---

## The Problem: Cloud LLMs Own Your Agent Memory

You want persistent memory for your AI agent. Hindsight handles that: fact extraction, knowledge graphs, semantic search, synthesis.

But it needs an LLM.

The default setup sends your data to OpenAI. That works. But:

- Every retain call extracts facts via the LLM. That's API cost per memory.
- Every reflect call runs multi-step reasoning. More tokens, more cost.
- Your users' memories transit through a third-party API.
- You need an API key, which means account setup, billing, rate limits.

For development, prototyping, or privacy-sensitive deployments, you want everything local.

[Ollama](https://ollama.com/) gives you that. Open-source models running on your hardware. No accounts, no keys, no network calls.

Running Hindsight with Ollama connects a production-grade memory engine to a local LLM. This tutorial shows you how.

---

## How Hindsight with Ollama Works

```
User input
     ↓
retain(content)     → Ollama extracts facts locally
     ↓
recall(query)       → Semantic search + Ollama reranking
     ↓
reflect(query)      → Ollama reasons over the knowledge graph
     ↓
Response
```

Same Hindsight memory engine. Same API. Just swap the LLM provider from `openai` to `ollama`. Everything else, including the knowledge graph, semantic search, and fact extraction, stays the same.

---

## Setting Up Hindsight with Ollama Step by Step

The full setup takes about ten minutes. You will install Ollama, pull a model, install Hindsight, and connect them with two environment variables.

### Step 1: Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Pull a model:

```bash
ollama pull gpt-oss:20b
```

> **Note:** OpenAI's `gpt-oss:20b` is an open-weight model that punches well above its size — it rivals much larger models on reasoning and supports tool calling out of the box. It needs ~16GB of RAM. For the full `gpt-oss:120b` (near o4-mini quality), you'll need ~80GB, but the 20b variant runs comfortably on most developer machines.

Verify it's running:

```bash
ollama list
```

You should see `gpt-oss:20b` in the output.

### Step 2: Install Hindsight

```bash
pip install hindsight-all
```

This includes:

- Embedded Postgres (no database setup)
- Fact extraction engine
- Semantic search with local embeddings
- Knowledge graph
- Synthesis engine

No external infrastructure needed. This is the same package used in the [OpenAI memory tutorial](/blog/2026/03/05/add-memory-to-openai-application) and the [CrewAI integration](/blog/2026/03/02/crewai), just pointed at a local LLM instead of a cloud API.

### Step 3: Start Hindsight with Ollama

Two environment variables:

```bash
export HINDSIGHT_API_LLM_PROVIDER=ollama
export HINDSIGHT_API_LLM_MODEL=gpt-oss:20b
export HINDSIGHT_API_LLM_MAX_CONCURRENT=1
export HINDSIGHT_API_ENABLE_OBSERVATIONS=false
```

Start the server:

```bash
hindsight-api
```

That's it. Hindsight detects Ollama at `localhost:11434` and uses `gpt-oss:20b` for fact extraction.

> **Note:** `LLM_MAX_CONCURRENT=1` prevents overloading your machine. Local models run one request at a time; parallel LLM calls queue up and compete for RAM/GPU. Cloud APIs handle concurrency on their end, but local models don't.

> **Note:** `ENABLE_OBSERVATIONS=false` disables observations (mental models). Observations run background LLM calls after each retain to consolidate facts into higher-level insights. With cloud APIs this is fast and useful, but with local models it adds significant inference time per retain. Disable them when getting started with Ollama. You can re-enable observations later once your setup is tuned.

You should see:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8888
```

### Step 4: Create a Bank and Store Memories

Install the client:

```bash
pip install hindsight-client
```

Create a memory bank:

```python
from hindsight_client import Hindsight

hindsight = Hindsight(base_url="http://localhost:8888")

hindsight.create_bank(
    bank_id="local-agent",
    name="Local Agent Memory",
    reflect_mission="Remember user preferences, facts, and conversation history.",
)
```

Store a memory:

```python
hindsight.retain(
    bank_id="local-agent",
    content="User: My name is Alice and I'm a software engineer in Portland. I prefer Python and functional programming patterns.\nAssistant: Nice to meet you, Alice!",
)
```

Ollama extracts facts from this locally. No data leaves your machine. Hindsight parses the conversation, identifies entities like "Alice" and "Portland," extracts relationships like "Alice is a software engineer," and stores them in the knowledge graph. All of this happens through Ollama running on your local hardware.

> **Gotcha:** Fact extraction with local models is slower than cloud APIs. Expect 15-20 seconds per retain call on Apple Silicon, longer on CPU-only machines. By default, `retain()` blocks until extraction is complete — your code waits while Ollama processes the content. If you need non-blocking ingestion, pass `async_=True` to `retain()` and the call returns immediately while extraction happens in the background.

### Step 5: Recall Memories

```python
memories = hindsight.recall(
    bank_id="local-agent",
    query="What programming languages does Alice prefer?",
    budget="low",
)

for memory in memories.results:
    print(memory.text)
```

Recall uses local embeddings by default, so no LLM call is needed for basic semantic search. This makes recall fast even when running Hindsight with Ollama on modest hardware. If you want to connect this to an [MCP-compatible AI client](/blog/2026/03/04/mcp-agent-memory), the recall results work the same way.

### Step 6: Reflect (With a Caveat)

```python
reflection = hindsight.reflect(
    bank_id="local-agent",
    query="What do you know about Alice?",
)

print(reflection.text)
```

Reflect is the most LLM-intensive operation in Hindsight. It traverses the knowledge graph, identifies relevant memories, and runs multi-step reasoning to synthesize a coherent answer. This is what makes Hindsight more than a simple vector store; reflect produces answers that combine information from multiple memories.

One important note: **reflect requires tool/function calling**, which not all local models support. `gpt-oss:20b` supports tool calling, so reflect works out of the box. If you use a different model (like Gemma 3), reflect may time out or return an error. Check [Ollama's model library](https://ollama.com/library) for models with tool calling support.

---

## Full Hindsight with Ollama Example

Save as `local_memory.py` to test the complete Hindsight with Ollama workflow:

```python
from hindsight_client import Hindsight

hindsight = Hindsight(base_url="http://localhost:8888")

# Create bank (idempotent, safe to run repeatedly)
hindsight.create_bank(
    bank_id="local-agent",
    name="Local Agent Memory",
    reflect_mission="Remember user preferences and important facts.",
)

# Store some memories (each retain call blocks while Ollama extracts facts)
print("Retaining memories...")
hindsight.retain(
    bank_id="local-agent",
    content="User: I'm Alice, a backend engineer at Acme Corp. I work mostly in Python and Go.",
)
print("First memory retained.")
hindsight.retain(
    bank_id="local-agent",
    content="User: I prefer dark mode, vim keybindings, and tabs over spaces.",
)
print("Second memory retained.")

# Recall
print("\nRecalling: 'What does Alice do for work?'")
memories = hindsight.recall(
    bank_id="local-agent",
    query="What does Alice do for work?",
    budget="low",
)

for r in memories.results:
    print(f"  → {r.text}")

print("\nDone. All processing happened locally.")
```

Run:

```bash
python local_memory.py
```

---

## Hindsight with Ollama: Pitfalls and Edge Cases

**1. First request is slow.** Ollama loads the model into memory on the first inference call. This can take 10-30 seconds. Subsequent calls are faster.

**2. Concurrent requests will queue.** Set `HINDSIGHT_API_LLM_MAX_CONCURRENT=1`. Local models can't parallelize like cloud APIs. Multiple concurrent retain calls will queue and process sequentially.

**3. Not all models support tool calling.** Reflect requires it. `gpt-oss:20b` supports tool calling out of the box. If you swap to a different model (like Gemma 3), reflect won't work.

**4. RAM matters.** `gpt-oss:20b` needs ~16GB of RAM. Add Hindsight's embedded Postgres and local embeddings, and you want 24GB+ total. For the full `gpt-oss:120b`, you need ~80GB (dedicated GPU). Use a smaller model on constrained hardware.

**5. Model quality varies.** Local models produce less precise fact extraction than GPT-4o or Claude. You may see more missed facts or less accurate entity resolution. For production workloads, cloud APIs still win on quality.

**6. Fresh Linux installs may need `zstd`.** The Ollama installer requires it for extraction. If you hit an error, run `sudo apt-get install zstd` (Debian/Ubuntu) or `sudo dnf install zstd` (RHEL/Fedora) first.

---

## Local Ollama vs. Cloud LLMs for Agent Memory

| | Ollama (Local) | OpenAI / Anthropic (Cloud) |
|---|---|---|
| **Cost** | Free after hardware | Per-token pricing |
| **Privacy** | Full, nothing leaves your machine | Data transits third-party API |
| **Speed** | 15-20s per retain (Apple Silicon) | 1-3s per retain |
| **Quality** | Good for 12b+ models | Best available |
| **Tool calling** | Model-dependent | Fully supported |
| **Reflect** | Requires specific models | Works with all providers |
| **Setup** | Ollama install + model pull | API key |
| **Concurrency** | Limited by hardware | Cloud-scaled |

**Use local when:** developing, prototyping, running in air-gapped environments, handling sensitive data, or avoiding per-token costs.

**Use cloud when:** you need speed, maximum quality, tool calling reliability, or production-scale concurrency.

You can also mix both approaches: develop locally with Hindsight and Ollama, then deploy to production with OpenAI or Anthropic. Same Hindsight API, same code, just change the env vars. Many teams use this pattern to keep development costs at zero while getting the best quality in production.

---

## Override Hindsight Ollama Defaults

Hindsight picks sensible defaults when running with Ollama, but you can override everything:

```bash
export HINDSIGHT_API_LLM_PROVIDER=ollama
export HINDSIGHT_API_LLM_MODEL=gpt-oss:120b       # upgrade from 20b
export HINDSIGHT_API_LLM_BASE_URL=http://192.168.1.50:11434/v1  # remote Ollama
export HINDSIGHT_API_LLM_MAX_CONCURRENT=2         # if your GPU can handle it
export HINDSIGHT_API_ENABLE_OBSERVATIONS=true      # re-enable once tuned
```

This also means you can run Ollama on a different machine (a GPU server, for example) and point Hindsight at it over the network. This is a common pattern for teams that want local privacy but better performance: run Ollama on a dedicated GPU box in your network, and run Hindsight on your application server.

---

## Recap: Running Hindsight with Ollama

Running Hindsight with Ollama takes two environment variables and about ten minutes of setup. Once connected, you get the same memory engine, the same knowledge graph, and the same retain/recall/reflect API that the cloud version provides, all running on your own hardware.

The key points to remember:

- `HINDSIGHT_API_LLM_PROVIDER=ollama` is the only switch you need
- `retain`, `recall`, and `reflect` all work with `gpt-oss:20b` — including tool calling
- For even better quality, `gpt-oss:120b` rivals OpenAI o4-mini but needs ~80GB of RAM
- Local inference is slower than cloud APIs, but you get full privacy and zero per-token costs
- The same code works with local or cloud providers, so you can swap by changing env vars

For teams that need data privacy, air-gapped deployments, or zero-cost development environments, Hindsight with Ollama is the fastest way to get persistent agent memory running locally.

---

## Next Steps

- **Try the full model** -- `gpt-oss:120b` for near o4-mini quality (needs ~80GB RAM / dedicated GPU)
- **Run Ollama on a GPU server** -- point `LLM_BASE_URL` at a remote machine for faster inference
- **Add the OpenAI chatbot loop** -- combine this with the [OpenAI memory tutorial](/blog/2026/03/05/add-memory-to-openai-application) for a fully local chatbot
- **Try MCP integration** -- connect Hindsight to [Claude, Cursor, or VS Code via MCP](/blog/2026/03/04/mcp-agent-memory)
- **Scale with LiteLLM** -- use [100+ model providers](/blog/2026/03/03/litellm) through a single interface
- **Inspect memories in the web UI** -- run the control plane at `localhost:9999` to browse extracted facts
- **Go to production with cloud** -- switch to `HINDSIGHT_API_LLM_PROVIDER=openai` when ready

Local memory is real memory. It just runs on your hardware.
