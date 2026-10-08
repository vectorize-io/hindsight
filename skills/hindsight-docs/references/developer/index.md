

{/* The page's own `# Overview` H1 is suppressed via hide_title: this is the site
    root, and the hero's headline is the H1 a visitor and a crawler should see.
    The sidebar and page metadata still read "Overview" from the frontmatter. */}

<HomeHero />

## Benchmarks

<HomeBenchmarks />

{/* Retrieval accuracy only. The coding-agent chart opens inside the coding-agents
    card above, where it is evidence for a path the reader has just been shown,
    and is explained in full on /sdks/integrations/coding-agents. */}

## Why Hindsight?

AI agents forget everything between sessions. Every conversation starts from zero—no context about who you are, what you've discussed, or what the assistant has learned. This isn't just an implementation detail; it fundamentally limits what AI Agents can do.

**The problem is harder than it looks:**

- **Simple vector search isn't enough** — "What did Alice do last spring?" requires temporal reasoning, not just semantic similarity
- **Facts get disconnected** — Knowing "Alice works at Google" and "Google is in Mountain View" should let you answer "Where does Alice work?" even if you never stored that directly
- **AI Agents need to consolidate knowledge** — A coding assistant that remembers "the user prefers functional programming" should consolidate this into an observation and weigh it when making recommendations
- **Context matters** — The same information means different things to different memory banks with different personalities

Hindsight solves these problems with a memory system designed specifically for AI agents.

{/* Rendered here rather than by the DocBreadcrumbs wrapper (which serves every
    other page): at the top it sat between the navbar and the hero band as a
    bordered strip and broke the full-bleed, and it lands better as the answer
    to the problem the section above just described. */}
<SkillBanner />

## How It Works

{/* The router: three kinds of visitor land on this page — one who has never
    heard of agent memory, one looking for the coding-agent plugin, one looking
    for the Hermes/OpenClaw plugin — and this is where each of them can see
    their own path. It opens "How It Works" because the cards answer "what does
    this look like for me" before the sections below answer "how". */}
<HomeFlow />

### Memory Types

Hindsight does not store conversations. It extracts what was said into typed
facts and then builds on them:

- **World fact** — an objective claim it was told. *"Alice works at Google."*
- **Experience fact** — something the bank itself did. *"I recommended Python to Bob."*
- **Observation** — a belief consolidated from many facts, with its evidence and
  its history. *"User was a React enthusiast, has now switched to Vue."*
- **Mental model** — a curated summary you write for a question you ask often.
- **Knowledge page** — a living document the bank writes about itself.

Facts are not a list. Each is linked to the entities it mentions and to the other
facts that share them, which is what makes "where does Alice work?" answerable
from two facts that were never stored together — the graph at the top of this
page is a sample bank's; hover a memory to see what it links to.

### Multi-Strategy Retrieval

`recall()` runs four searches **in parallel**, because no single one handles
every question:

- **Semantic** — meaning rather than wording, so "Alice's job" finds "Alice works as a software engineer"
- **Keyword (BM25)** — the names and technical terms an embedding blurs together; five pluggable Postgres backends, including one that works on a Citus cluster
- **Graph** — entity links, so a fact reachable in two hops comes back even when it shares no words with the query
- **Temporal** — time expressions parsed into a window, then filled by relevance *and spread across the range*, so "what happened in 2023?" is not all from December

Then the part that matters more than any single arm:

- **Fused by rank, not score** — a memory several strategies agree on wins, and no arm's scoring scale can dominate the others
- **Re-ranked by a cross-encoder** that reads the query and the memory together
- **Cut to a token budget, not a top-k** — you say how much context you can afford and Hindsight fills it, because agents budget in tokens and not in result counts

See [Recall](retrieval.md) for the strategies in full.

### Observation Consolidation

A background worker keeps turning raw facts into durable beliefs:

- **Deduplicated** — overlapping facts merge into one observation instead of piling up as repeats
- **Evidence-grounded** — each observation points at the memories that support it, with exact quotes and a proof count
- **Refined, not overwritten** — new evidence updates an observation and its history is kept, so you can see a belief change
- **Freshness-aware** — when memories have landed but not yet been consolidated, `reflect` treats the affected observations as stale and checks them against raw facts before trusting them

### Knowledge Pages

Observations answer one question at a time. A **knowledge page** is a living
document the bank writes about itself — "What are the components here?", "What's
our error-handling convention?" — rewritten incrementally as consolidation
produces new knowledge in its scope:

- **A wiki, not a blob** — pages live in a tree of folders, browsable and searchable, each answering one question
- **Built from observations** — synthesized from consolidated beliefs rather than raw conversation, so a page is not a transcript summary
- **Never self-citing** — a page never reads another page, so they cannot cite each other into a feedback loop
- **Real files when you want them** — `hindsight fs mount` projects the tree onto disk as ordinary markdown, so `grep`, an editor or an agent's file tools all work with no SDK

See [Knowledge Pages](knowledge-pages.md) for the full model.

### Reflect

`recall()` returns memories. `reflect()` returns an **answer**, by running an
agentic loop over the bank rather than a single query:

- **It gathers its own evidence** — the agent decides what it needs and calls its own tools, up to ten rounds, and cannot answer before it has retrieved something
- **It checks sources in priority order** — mental models and knowledge pages first, then observations, and only then raw facts, so it reads the distilled answer before the transcript
- **It cites what it used** — and only IDs it actually retrieved can be cited

What makes two banks answer the same question differently is their configuration:

- **Mission** — the bank's identity in plain language, which tells it what to prioritise. *"I am a research assistant specializing in ML. I prefer simplicity over cutting-edge."*
- **Directives** — hard rules it must never break. *"Never recommend specific stocks."*
- **Disposition** — skepticism, literalism and empathy on a 1–5 scale, shaping how it interprets what it finds

These shape `reflect` only. `recall` returns the same memories whoever is asking.

See [Reflect](reflect.md) for the loop in detail.

## Architecture Deep Dive

Everything above, end to end and in motion: what a document turns into on the
way in, what each of the three operations touches, and what the worker changes
behind them. Play it, or step through it at your own pace.

**Figure: What Hindsight Does.** An animated diagram on the docs site; its narration, step by step:

- **retain()**
  1. Your agent sends what happened: a conversation, a document, a transcript.
  2. The original text is stored as a document.
  3. It is split into chunks, so the exact passage can be handed back later.
  4. An LLM pulls out facts: world facts about others, and experience facts about what the agent itself did. The bank already knew Alice worked at Microsoft.
  5. Each fact is indexed four ways: by meaning, by its words, by the entities it links, and by when it happened.
  6. retain() is done. The rest happens in the background.
  7. Consolidation picks up the new facts and checks them against the observations the bank already holds. One disagrees: Microsoft or Google?
  8. It updates that observation instead of adding a second one: Alice moved from Microsoft to Google in March. Both facts stay as its sources, so the history is kept.
  9. When consolidation finishes, it queues a refresh for every mental model and page set to refresh after it that now has new memories…
  10. …and each one re-runs its question through reflect and is rewritten.
- **recall()**
  1. recall() finds the memories that matter for a query.
  2. Searches run at once, each through its own index: meaning, exact words and the entity graph. The time search only joins when the query names a date.
  3. The same indexes cover facts and observations, so both come back. They are merged and reranked; the old Microsoft fact falls below the cut.
  4. The agent gets ranked memories it can put straight into its prompt.
- **reflect()**
  1. reflect() answers a question by reasoning over everything in the bank.
  2. An agent loop decides what to look up. It starts with the most refined knowledge: mental models and knowledge pages.
  3. Then observations, searched through the same indexes as recall. If new facts are still waiting to be consolidated, they are marked stale.
  4. Then raw facts through recall, for the details the summaries leave out.
  5. When it needs the exact wording, it opens the chunk or document a fact came from.
  6. It stops when it has enough evidence, and writes an answer shaped by the bank’s mission and disposition. It can only cite what it found.
  7. The answer comes back with the memories it is based on.

## Integrations

**Coding agents** — [one install](../sdks/integrations/coding-agents.md) wires 20+
harnesses (Claude Code, Codex CLI, Cursor CLI, opencode, Copilot CLI and more) to
a per-repo memory bank.

**Personal agents** — [Hermes](../sdks/integrations/hermes.md) and
[OpenClaw](../sdks/integrations/openclaw.md) install Hindsight as their memory
provider: every turn recalls what matters before answering and retains what was
said afterwards. The Hermes desktop app configures it
[in Settings](../sdks/integrations/hermes-desktop.md), no terminal.

Everything else — frameworks, MCP servers, tools — is in the
Integrations Hub.

## Next Steps

### Getting Started
- [**Quick Start**](api/quickstart.md) — Install and get up and running in 60 seconds
- [**RAG vs Hindsight**](rag-vs-hindsight.md) — See how Hindsight differs from traditional RAG with real examples

### Core Concepts
- [**Retain**](retain.md) — How memories are stored with multi-dimensional facts
- [**Recall**](retrieval.md) — How the 4-way parallel search retrieves memories
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
