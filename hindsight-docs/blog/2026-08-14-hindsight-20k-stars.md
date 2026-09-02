---
title: "20,000 Stars: How Hindsight Got Here, Version by Version"
authors: [benfrank241]
slug: "2026/08/14/hindsight-20k-stars"
date: 2026-08-14T12:00
tags: [hindsight, agent-memory, open-source, milestone, changelog]
description: "Hindsight just crossed 20,000 GitHub stars in under ten months and 67 releases. Here's the timeline: every major capability we shipped, version by version."
image: /img/blog/hindsight-20k-stars.png
hide_table_of_contents: true
---

![Hindsight at 20,000 stars: a release timeline from the first open-source commit in December 2025 to v0.9.1](/img/blog/hindsight-20k-stars.png)

We open-sourced Hindsight in December 2025. Nine months and **67 releases** later, it just crossed **20,000 GitHub stars**. Across those releases we shipped **more than 900 tracked changes** — new capabilities, integrations, and hardening. The stars are the side effect. The releases are the story, so here is the real one: how an agent-memory engine went from a first public commit to what it is today, version by version.

<!-- truncate -->

## The timeline at a glance

| Version | When | Releases | The headline |
|---|---|---|---|
| **0.1.x** | Dec 2025 | 13 | Foundations: embedded Postgres, local MCP, `hindsight-embed` |
| **0.2.x** | Jan 2026 | 1 | Multi-bank memory and cross-bank MCP tools |
| **0.4.x** | Feb – Mar 2026 | 23 | The learning layer: observations, mental models, the knowledge graph |
| **0.5.x** | Apr 2026 | 7 | Faster retrieval, the Constellation view, template hub |
| **0.6 – 0.7.x** | May 2026 | 6 | Scaling the search layer, enterprise backends |
| **0.8.x** | Jun – Aug 2026 | 7 | Production hardening + a wave of coding-tool integrations |
| **0.9.x** | Aug 2026 | 2 | Knowledge Pages and memory for every coding agent |

## December 2025 — v0.1.x: the foundation

Thirteen releases in the first month set the core that has not changed since: **retain, recall, and reflect on embedded PostgreSQL**, no external vector database to run. We moved fast on the local ergonomics so anyone could try it in minutes.

Highlights:
- A **local MCP server** so any MCP client connects without a separate service, and **`hindsight-embed`** to run memory in-process.
- An **extensions system** for plugging in new operations, plus the first **graph-based retriever** to use relationships between memories.
- **Memory banks**, **memory tags**, **backup/restore**, and **multilingual** content support.
- Early breadth on models and providers: **LiteLLM**, **Cohere** embeddings/reranking, configurable embedding dimensions, and **Gemini 3 Pro / GPT-5.2**.

## January 2026 — v0.2.x: multi-bank memory

A focused release that made isolation real. **Multi-bank access** plus the MCP tools to work across banks meant per-user and per-project memory stopped being theoretical. It also added **Anthropic Claude** and **LM Studio**, custom entities on retain, structured `reflect` output, and the first **graph visualization** in the Control Plane.

## February to March 2026 — v0.4.x: the learning layer

This is the big one, and the release line the earlier draft of this post undersold. **Twenty-three releases and roughly 300 changes.** If 0.1 gave Hindsight a place to put memories, 0.4 gave it the ability to *learn* from them.

The learning layer:
- **Observations** — consolidated, deduplicated beliefs derived from raw facts — with configurable scopes, per-scope limits, history tracking, and batch consolidation.
- **Mental models** — standing answers that refresh themselves — with full MCP create/read/update/delete, tag-aware triggers, staleness signals, and a history diff view.
- A real **knowledge graph**: entity resolution, richer entity labels, and improved **causal link** detection, alongside temporal ordering.

Everything else scaled up around it:
- **Ingest anything**: PDFs, images, and Office documents, the Iris parser, and async batch retain via provider Batch APIs.
- **Retrieval at scale**: per-bank HNSW indexes, pgvectorscale (DiskANN) support, and a 40x speedup in observation recall on large banks.
- **Multi-tenancy and auth**: Bearer-token MCP, hierarchical config scopes, a Supabase tenant extension, and `consolidation.completed` / `retain.completed` **webhooks**.
- **The integration wave began**: Claude Code, Codex CLI, LangGraph, the Vercel AI SDK, Chat SDK, CrewAI, PydanticAI, AG2, Agno, Strands, and LlamaIndex, plus Go and AI SDK clients and native Windows support.

## April 2026 — v0.5.x: retrieval grows up

Seven releases, ~137 changes, focused on making recall faster and memory portable. The graph story consolidated onto a single **LinkExpansion retriever**, and a **3-phase retain pipeline** dramatically improved ingestion throughput under concurrency.

Highlights:
- The **Constellation view** — an interactive, zoomable canvas of the entity graph with heat-gradient coloring.
- **Delta mental-model refresh** that only re-reads memory created since the last run.
- **Bank template import/export** via a Template Hub: export a bank's config, mental models, and directives as a reusable manifest.
- **Local and open inference**: a built-in **llama.cpp** provider, plus OpenRouter and first-class DeepSeek support.
- New integrations: OpenAI Agents SDK, AutoGen, Paperclip, OpenCode, and Pipecat voice.

## May 2026 — v0.6.x and v0.7.x: scaling the search layer

Two minor lines, one theme: make retrieval fast and correct at real data sizes, and reach enterprise backends.

Highlights:
- **BM25 search backends**: ParadeDB `pg_search` (Citus-compatible) and PGroonga for better multilingual search, plus configurable tokenization.
- **Enterprise storage**: AlloyDB ScaNN indexing, an **Oracle Database** backend, and a read-replica option for recall traffic.
- Smarter scheduling: **bank-priority consolidation** and targeted consolidation by observation scope.
- More providers and models: z.ai, Fireworks, a litellmrouter with automatic fallback chains, and the Qwen3 reranker.
- Integrations: Dify, n8n, SmolAgents, AWS Bedrock AgentCore, Google ADK, Flowise, Roo Code, Vapi voice, and Gemini Spark.

## June to August 2026 — v0.8.x: built to run in production

The second-biggest line in the project's history: **seven releases and over 230 changes.** This is where Hindsight got operationally serious, and where the integration list exploded.

Production hardening:
- **Periodic background maintenance** that reconciles consolidation state and enforces retention across tenants on its own.
- **Durable, resumable progress snapshots** for long-running consolidation and batch retains.
- **Reversible memory curation** — edit, invalidate, and revert memory units — and **semantic deduplication** of near-duplicate observations.
- A **self-diagnosing health probe** that reports whether the event loop is blocked or the connection pool is exhausted.
- **Whole-bank and cross-bank export/import** (no re-running the LLM), plus prompt-prefix caching and Anthropic batch + prompt caching to cut cost.
- Security: **Memory Defense** SIEM enrichment and multi-LLM failover / round-robin.

The integration wave:
- **GitHub Copilot** (CLI and VS Code), **Aider**, **Zed**, **OpenHands**, **Continue.dev**, **Cursor** (plugin and CLI), **Cline**, **Windsurf**, **Composio**, **Zapier**, **Obsidian**, **Haystack**, the **Microsoft Agent Framework**, and more.

## August 2026 — v0.9.x: Knowledge Pages, and memory for every coding agent

The two most recent releases turn all of that infrastructure outward.

- **Knowledge Pages** turn a bank into a self-healing wiki it writes about itself: living documents synthesized from consolidated memory that refresh as the bank learns.
- **Hindsight Coding Agents** brings long-term memory to **ten coding agents** — Claude Code, Codex, Cursor CLI, opencode, Copilot CLI, and more — from one package, with per-bank toggles to tune temporal search, graph expansion, and reranking during recall.
- **0.9.1** followed a week later: roughly **9x faster temporal extraction** with the same results, **whole-bank transfers that carry the Knowledge Pages tree**, **xAI OAuth** for running the LLM on a SuperGrok subscription, and a database-independent liveness probe.

## The throughline

Read the timeline back and the pattern is clear. Nothing here is a prompt trick. It is databases, retrieval strategies, a learning layer, consolidation, ingestion, auth, and operational plumbing: the unglamorous infrastructure that makes memory accurate enough to measure and boring enough to run in production. More than nine hundred changes across sixty-seven releases, and the shape of the thing keeps getting sharper.

We should stay honest: Hindsight is not the most-starred project in the category, and the benchmarks are where we would rather compete anyway. If you have not tried it, start free on [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) or self-host in one command from [GitHub](https://github.com/vectorize-io/hindsight). Thank you to everyone who shipped a release, filed an issue, or built something on top. The next version is already in progress.

---

**Learn more:**
- [Full changelog](https://hindsight.vectorize.io/changelog) — every release in detail
- [Hindsight 0.9.0](https://hindsight.vectorize.io/blog/2026/08/06/hindsight-0-9-0) — Knowledge Pages and memory for every coding agent
- [The best open-source agent memory systems](https://hindsight.vectorize.io/blog/2026/08/11/open-source-agent-memory-systems) — an honest look at the whole category
