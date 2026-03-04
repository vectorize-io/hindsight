---
title: "The Open-Source MCP Memory Server Your AI Agent Is Missing"
authors: [hindsight]
date: 2026-03-04
tags: [mcp, memory, agents, docker, tutorial]
image: /img/blog/mcp-agent-memory.png
hide_table_of_contents: true
---

AI agents forget everything between sessions. Hindsight gives them persistent, structured memory via MCP. One Docker command to run the full stack locally. Connect any MCP-compatible client. Three core operations: `retain` (store), `recall` (search), `reflect` (reason) — plus mental models that auto-update as memories grow.

<!-- truncate -->

---

## TL;DR

- AI agents forget everything between sessions. Hindsight gives them persistent, structured memory via MCP.
- One Docker command to run the full stack locally. Connect any MCP-compatible client.
- Three core operations: `retain` (store), `recall` (search), `reflect` (reason). Plus mental models — living documents that auto-update as memories grow.
- Hindsight isn't a vector database. It extracts structured facts, resolves entities, builds a knowledge graph, and uses cross-encoder reranking to surface what actually matters.
- Open source: [github.com/vectorize-io/hindsight](https://github.com/vectorize-io/hindsight).

---

## The Problem

AI agents are stateless. Every session starts from zero.

You tell your coding assistant your tech stack, your deployment preferences, your team's conventions. Next session — gone. You explain the same architecture decisions, re-establish the same context, re-state the same constraints. Every time.

People work around this by pasting context into system prompts or maintaining notes they copy in manually. That works for a while, but it doesn't scale. It can't capture the kind of nuanced, evolving knowledge that accumulates over weeks of working with an agent — things like "this user prefers functional patterns," "their team uses PostgreSQL 16," or "they tried Redis caching last month and rolled it back."

What you actually want is for your agent to build up memory over time. Store what matters, retrieve it when relevant, and learn from the accumulation.

Hindsight is an open-source memory system designed for exactly this. It connects to any MCP-compatible agent and gives it persistent, structured long-term memory.

---

## The Approach

```
Your MCP Client  ──MCP (HTTP)──>  Hindsight API
(Claude, Cursor,                       │
 VS Code, etc.)                        ├── Memory Engine (retain/recall/reflect)
                                       ├── Fact Extraction + Entity Resolution
                                       ├── Embeddings + Cross-Encoder Reranking
                                       ├── Knowledge Graph Traversal
                                       └── PostgreSQL + pgvector
```

Your agent connects to Hindsight over MCP (Model Context Protocol). MCP is an open standard — any client that speaks it can use Hindsight as a memory backend.

When your agent stores a memory via `retain`, Hindsight doesn't just dump raw text into a vector database. It extracts structured facts, resolves entities ("Alice" and "my coworker Alice" are the same person), generates embeddings, and indexes everything for retrieval.

When your agent needs context via `recall`, Hindsight runs four retrieval strategies in parallel — semantic search, BM25 keyword matching, entity graph traversal, and temporal filtering — then reranks results with a cross-encoder. What comes back is the most relevant subset of your memories, not a raw dump.

This matters because naive RAG (embed text, cosine similarity, return top-k) breaks down when you have hundreds of memories spanning different topics and time periods. Hindsight's multi-strategy approach ensures that a question like "what did we decide about caching?" finds the right answer even when the memory uses different terminology.

---

## Implementation

### Step 1: Start Hindsight

The quickest way to run Hindsight is with Docker. One command gives you the full stack — API server, embedded PostgreSQL, local embedding models, and MCP endpoints:

```bash
docker run --rm -it --pull always -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_API_KEY=YOUR_LLM_API_KEY \
  -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

You'll need an LLM API key for Hindsight's internal processing (fact extraction, entity resolution, reflect). Hindsight supports multiple LLM providers — OpenAI, Anthropic, Gemini, Groq, or a local model via Ollama or LM Studio. Set the provider explicitly if you're not using OpenAI:

```bash
docker run --rm -it --pull always -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_PROVIDER=gemini \
  -e HINDSIGHT_API_LLM_API_KEY=YOUR_GEMINI_API_KEY \
  -e HINDSIGHT_API_LLM_MODEL=gemini-2.5-flash \
  -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

The `-v` flag persists your data across container restarts. Without it, memories are lost when the container stops. Port 8888 is the API and MCP endpoint; port 9999 is an optional admin UI for browsing memories.

Once running, the MCP endpoint is available at `http://localhost:8888/mcp/your_bank_id/` (replace `your_bank_id` with any name you like).

**Or use Hindsight Cloud** — skip Docker entirely. [Sign up for a free account](https://ui.hindsight.vectorize.io/signup), grab your API key, and connect via MCP:

```json
{
  "mcpServers": {
    "hindsight": {
      "type": "http",
      "url": "https://api.hindsight.vectorize.io/mcp/your_bank_id/",
      "headers": {
        "Authorization": "Bearer YOUR_API_KEY"
      }
    }
  }
}
```

Or with Claude Code:

```bash
claude mcp add --transport http hindsight \
  https://api.hindsight.vectorize.io/mcp/your_bank_id/ \
  --header "Authorization: Bearer YOUR_API_KEY"
```

### Step 2: Connect Your MCP Client

Hindsight works with any MCP-compatible client. Add the following JSON to your client's config file:

```json
{
  "mcpServers": {
    "hindsight": {
      "type": "http",
      "url": "http://localhost:8888/mcp/your_bank_id/"
    }
  }
}
```

Config file locations by client:

| Client | Config File |
|--------|-------------|
| Claude Desktop | `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows) |
| Cursor | `.cursor/mcp.json` (project) or `~/.cursor/mcp.json` (global) |
| VS Code | `.vscode/mcp.json` — uses `"servers"` instead of `"mcpServers"` |
| Windsurf | `~/.codeium/windsurf/mcp_config.json` — uses `"serverUrl"` instead of `"url"` |

**Claude Code** — use the CLI instead:

```bash
claude mcp add --transport http hindsight http://localhost:8888/mcp/your_bank_id/
```

Restart your client to pick up the changes.

### Step 3: Verify It's Working

Ask your agent:

> "What memory tools do you have available?"

It should list Hindsight's memory tools, including the three core operations (`retain`, `recall`, `reflect`), six mental model tools, and additional tools for browsing memories, managing documents, and bank administration.

---

## The Memory Tools

Once connected, your agent has access to three core operations and a set of mental model tools.

**Retain** — Store a memory:

Tell your agent something you want it to remember. It will call `retain` automatically based on the tool's built-in instructions, or you can be explicit:

> "Remember that I prefer TypeScript over JavaScript for all new projects."

Behind the scenes, Hindsight extracts structured facts, resolves entities, and indexes the memory for later retrieval. A single `retain` call on "Alice from engineering recommended we switch to Postgres 16 for the new JSONB features" produces:

- A fact: "Alice recommended switching to Postgres 16 for JSONB features"
- Entity resolution: "Alice" linked to "Alice from engineering"
- Temporal indexing: when this was mentioned
- Embeddings: for semantic search later

**Recall** — Search memories:

Your agent will proactively recall relevant context when you ask questions. You can also prompt it directly:

> "What do you know about my programming preferences?"

Recall runs four retrieval strategies in parallel — semantic search, keyword matching (BM25), graph traversal, and temporal filtering — then reranks the results with a cross-encoder. This is what makes it work better than a simple vector search.

**Reflect** — Synthesize insights:

Reflect goes deeper than recall. Instead of returning raw facts, it reasons across your memories using an LLM:

> "Based on what you know about me, what tech stack would you recommend for my next side project?"

This is useful for questions that require connecting dots across multiple memories.

**Mental Models** — Living documents:

Mental models are summaries that automatically stay up to date as new memories are added. Think of them as pre-computed reflections that refresh themselves:

> "Create a mental model called 'My Tech Stack' that tracks what languages, frameworks, and tools I use."

You can list, retrieve, update, and delete mental models. They're useful for maintaining an always-current view of a topic without running a full `reflect` every time.

---

## Memory Banks

The URL path controls which memory bank you're using. In the examples above, `/mcp/your_bank_id/` scopes all operations to that bank.

Banks are isolated stores — think of each one as a separate brain. You can run separate banks for different contexts:

- `my-project` for a specific project
- `team-knowledge` for shared team information
- One bank per user in a multi-agent system

Banks are created automatically on first use. To use a different bank, change the URL path:

```
http://localhost:8888/mcp/project-x/
```

If you want your agent to manage multiple banks in a single session, connect to the multi-bank endpoint at `/mcp/` instead. This adds `bank_id` as a parameter to every tool and includes additional bank management tools like `list_banks`, `create_bank`, and `get_bank_stats`.

---

## Pitfalls & Edge Cases

### Memory processing is async

When your agent calls `retain`, fact extraction and indexing happen in the background. If you store something and immediately try to recall it, it might not be there yet. Give it a few seconds for complex memories.

### Token limits on recall

By default, `recall` returns up to 4096 tokens of memory content. For banks with extensive history, some older or lower-relevance memories may be trimmed from the response. This is intentional — it keeps the context window manageable.

### Mental model creation is async too

When you create or refresh a mental model, the LLM-powered generation runs in the background. The initial call returns an `operation_id`. The content will be available shortly after — typically a few seconds, depending on how many memories need to be synthesized.

### LLM key is for Hindsight, not your agent

The `HINDSIGHT_API_LLM_API_KEY` is used by Hindsight internally for fact extraction, entity resolution, and reflect operations. It's separate from whatever LLM your agent uses. You can use a cheap, fast model here (Gemini Flash, Groq, etc.) — it doesn't need to be the same model powering your agent.

---

## When Hindsight Works Well

- You want structured memory, not just vector search over conversation logs
- You need memory that works across sessions, clients, and agents
- You want entity resolution, temporal awareness, and multi-strategy retrieval out of the box
- You're building agents that accumulate knowledge over time

---

## Recap

Hindsight gives any MCP-compatible agent persistent long-term memory. One Docker command to start, a few lines of JSON to connect.

The key insight is that memory isn't just storage and retrieval. Hindsight extracts structured facts from raw input, links entities, tracks temporal data, and uses cross-encoder reranking to surface the most relevant memories. That's what separates it from stuffing conversation logs into a vector database.

The MCP tools cover the full lifecycle: `retain` to store, `recall` to search with multi-strategy retrieval, `reflect` to synthesize insights, mental model tools for maintaining living documents that auto-update, and utility tools for browsing and managing memories, documents, and tags.

> **Want managed hosting?** [Hindsight Cloud](https://ui.hindsight.vectorize.io) runs the full stack for you — no Docker, no infrastructure. Sign up, grab an API key, and connect over HTTPS.

---

## Next Steps

- **Build up your memory**: Start using your agent normally. Tell it your preferences, your project context, your decisions. It will retain what matters.
- **Explore mental models**: Create living documents that auto-update as your memory grows. Try: `"Create a mental model that summarizes my project architecture."`
- **Try multi-bank setups**: Run separate banks for different projects or agents. Connect to `/mcp/` for multi-bank mode.
- **Use the SDK directly**: Beyond MCP, Hindsight has [Python](https://pypi.org/project/hindsight-client/) and [TypeScript](https://www.npmjs.com/package/@vectorize-io/hindsight-client) SDKs for integrating memory into your own applications.
- **Check out the docs**: Full API reference, SDK guides, and more at [hindsight.vectorize.io](https://hindsight.vectorize.io).
