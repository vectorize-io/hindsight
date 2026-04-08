---
title: "Give the Only Self-Improving AI Agent (Hermes) a Memory Upgrade It Deserves"
authors: [benfrank241]
date: 2026-03-17T12:00
tags: [hermes, agents, python, memory, tutorial, plugin]
image: /img/blog/hermes-agent-memory.png
---

![How to Add Persistent Memory to Hermes Agent](/img/blog/hermes-agent-memory.png)

:::warning Deprecated
The `hindsight-hermes` pip plugin described in this post is deprecated. Hermes now ships with a native Hindsight memory provider. See [Hindsight Is Now a Native Memory Provider in Hermes Agent](/blog/2026/04/06/hermes-native-memory-provider) for the current setup guide.
:::

[Hermes Agent](https://github.com/NousResearch/hermes-agent) is a self-improving AI agent with 40+ tools and a plugin system. Its built-in memory saves to local files. `hindsight-hermes` replaces it with structured fact extraction, entity resolution, and multi-strategy retrieval — via one pip install and three environment variables.

<!-- truncate -->

**TL;DR:**
- Hermes Agent's built-in memory is local file-based — no structure, no retrieval intelligence, no cross-machine sync
- `hindsight-hermes` is a pip-installable plugin that registers Hindsight retain/recall/reflect as native Hermes tools
- One `pip install`, three environment variables, disable the built-in `memory` tool, and you're done
- Works with [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) (zero infra) or self-hosted

## The problem: good memory, but it could go further

Hermes Agent has memory built in, and it's a reasonable design. The `memory` tool saves durable facts to `~/.hermes/` as persistent files, and the `session_search` tool lets the agent look back through past conversations. It works — the agent can store preferences, recall context, and carry knowledge across sessions.

But there's room to grow:

- **Structure.** Memories are stored as text. There's no entity resolution (connecting "Alice" with "my coworker Alice from engineering"), no relationship tracking, and no temporal awareness beyond session timestamps.
- **Retrieval.** Search is keyword-based. For simple lookups that's fine, but it struggles with questions that use different terminology than what was stored.
- **Locality.** Memories live on disk. Run Hermes on your laptop and your server — two separate brains with no way to share context.
- **Synthesis.** You can store and retrieve facts, but you can't ask "based on everything you know about this customer, what should we prioritize?" and get a reasoned answer.

Hindsight adds the layer on top: structured fact extraction, entity resolution, a knowledge graph, multi-strategy retrieval with cross-encoder reranking, and a `reflect` operation that synthesizes across all stored memories.

## Architecture: a plugin that registers three tools

```
User  ──>  Hermes Agent
              ├── Built-in tools (terminal, browser, files, ...)
              └── [hindsight] plugin
                    ├── hindsight_retain  ──>  Hindsight API  ──> fact extraction,
                    ├── hindsight_recall  ──>       │              entity resolution,
                    └── hindsight_reflect ──>       │              knowledge graph,
                                                   └──>           PostgreSQL + pgvector
```

`hindsight-hermes` hooks into Hermes's [plugin system](https://github.com/NousResearch/hermes-agent/blob/main/hermes_cli/plugins.py). When Hermes starts, it scans for packages with the `hermes_agent.plugins` entry point, finds `hindsight-hermes`, and calls its `register()` function. That registers three tools into Hermes's tool registry.

No forking Hermes. No patching config files. Just `pip install` and environment variables.

## Setting up Hindsight

You have two options: Hindsight Cloud (no setup) or self-hosted.

**Option A: Hindsight Cloud**

1. [Sign up at Hindsight Cloud](https://ui.hindsight.vectorize.io/signup)
2. Create a memory bank in the dashboard and copy your API key
3. Your base URL is `https://api.hindsight.vectorize.io`

**Option B: Self-hosted with Docker**

```bash
docker run --rm -it --pull always -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_API_KEY=YOUR_OPENAI_KEY \
  -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

Wait for the health check:

```bash
curl http://localhost:8888/health
# {"status":"healthy","database":"connected"}
```

The `-v` flag persists data across container restarts. Port 8888 is the API; port 9999 is the admin UI for browsing memories.

**Option C: Self-hosted with pip**

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=YOUR_OPENAI_KEY
hindsight-api
```

## Install hindsight-hermes

One pip install. The package auto-registers as a Hermes plugin via Python entry points — no config files, no manual plugin setup. The only requirement is that it's installed in the **same Python environment** as Hermes.

```bash
uv pip install hindsight-hermes --python $HOME/.hermes/hermes-agent/venv/bin/python
```

That's it. When Hermes starts, it discovers the package automatically and registers the three memory tools.

You can verify it's registered:

```bash
python -c "
import importlib.metadata
eps = importlib.metadata.entry_points(group='hermes_agent.plugins')
for ep in eps:
    print(f'{ep.name}: {ep.value}')
"
# Expected: hindsight: hindsight_hermes
```

## Configuration

Set these environment variables before launching Hermes:

```bash
# Required — where Hindsight is running
export HINDSIGHT_API_URL=http://localhost:8888

# Required — the memory bank (an isolated "brain" for this agent)
export HINDSIGHT_BANK_ID=my-agent

# Optional — only needed for Hindsight Cloud (https://api.hindsight.vectorize.io)
export HINDSIGHT_API_KEY=hsk_your-key-here

# Optional — recall budget: low (fast), mid (default), high (thorough)
export HINDSIGHT_BUDGET=mid
```

If neither `HINDSIGHT_API_URL` nor `HINDSIGHT_API_KEY` is set, the plugin silently skips registration — Hermes starts normally without the Hindsight tools.

### Disable Hermes's built-in memory

This is the step people miss. Hermes has its own `memory` tool that saves to `~/.hermes/`. If both are active, **the LLM defaults to the built-in one** — it's a single tool it already recognizes. Your Hindsight tools will sit unused.

```bash
hermes tools disable memory
```

This persists across sessions. Re-enable later with `hermes tools enable memory`.

## Using the memory tools

Launch Hermes:

```bash
hermes
```

Type `/tools` to verify. You should see the `[hindsight]` toolset:

```
[hindsight]
  * hindsight_recall     - Search long-term memory for relevant information.
  * hindsight_reflect    - Synthesize a thoughtful answer from long-term memories.
  * hindsight_retain     - Store information to long-term memory for later retrieval.
```

### Retain — store memories

Tell Hermes something to remember:

```
● Remember that my favourite programming language is Rust and I prefer dark mode.
```

You should see `⚡ hindsight` in the response — that confirms it called `hindsight_retain`, not the built-in memory tool.

Under the hood, Hindsight extracts structured facts ("User's favourite programming language is Rust"), resolves entities, generates embeddings, and indexes everything. You don't manage any of that.

### Recall — search memories

```
● What do you know about my programming preferences?
```

Recall runs four retrieval strategies in parallel — semantic search, BM25 keyword matching, entity graph traversal, and temporal filtering — then reranks results with a cross-encoder. This is what makes it work better than string matching over flat files.

### Reflect — synthesize across memories

```
● Based on what you know about me, suggest a colour scheme for my IDE.
```

Reflect doesn't return raw facts. It traverses the knowledge graph, reasons across everything in the bank, and produces a synthesized answer. Slower than recall, but far more useful for open-ended questions.

### Verify via the API

Confirm memories are stored by querying Hindsight directly:

```bash
curl -s http://localhost:8888/v1/default/banks/my-agent/memories/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "programming preferences", "budget": "low"}' | python3 -m json.tool
```

```json
{
  "results": [
    {
      "text": "User's favourite programming language is Rust.",
      "type": "world",
      "entities": ["user"]
    },
    {
      "text": "User prefers dark mode in all editors.",
      "type": "world",
      "entities": ["user"]
    }
  ]
}
```

## Pitfalls and edge cases

1. **Plugin not in `/tools`.** The most common cause: `hindsight-hermes` is installed in a different Python environment than Hermes. Entry points are per-environment. Run `python -c "import importlib.metadata; print(list(importlib.metadata.entry_points(group='hermes_agent.plugins')))"` from the Hermes venv to verify.

2. **LLM picks built-in memory.** Even with the plugin loaded, if both `memory` and `hindsight_retain` exist, the LLM chooses `memory`. Run `hermes tools disable memory`.

3. **Retain is asynchronous.** The API returns immediately; fact extraction happens in the background. If you retain and immediately recall in the same turn, the new facts may not be indexed yet. Design so recall happens on subsequent turns.

4. **Env vars are read once at startup.** Changing `HINDSIGHT_API_URL` or `HINDSIGHT_BANK_ID` after launch has no effect. Restart Hermes to pick up changes.


## Tradeoffs: Hindsight plugin vs. alternatives

| | **Hindsight plugin** | **Built-in memory** |
|---|---|---|
| **Storage** | PostgreSQL + pgvector | Local files (~/.hermes/) |
| **Structure** | Facts, entities, relationships | Raw text |
| **Retrieval** | Semantic + BM25 + graph, reranked | Basic search |
| **Synthesis** | reflect tool | None |
| **Cross-machine** | Yes | No |
| **Setup** | pip install + env vars | Built-in |

**Use the built-in memory** when you want zero setup and basic persistence.

**Use the Hindsight plugin** when you want structured retrieval, entity resolution, and memory that persists across machines.

## Recap

`hindsight-hermes` gives Hermes Agent persistent, structured long-term memory via a pip-installable plugin. No code changes, no config patches.

Hermes's plugin system uses standard Python entry points, so any pip package can register tools. `hindsight-hermes` injects retain, recall, and reflect — backed by Hindsight's multi-strategy retrieval, entity resolution, and knowledge graph.

The key practical detail: disable Hermes's built-in `memory` tool. Otherwise the LLM prefers it and your Hindsight tools go unused.

## Next steps

- **Build up memory over time.** Use Hermes normally — it will retain what matters and recall it when relevant.
- **Try reflect for synthesis.** Ask open-ended questions: "Based on everything you know about me, what kind of projects would I enjoy?"
- **Use per-user banks.** Set `HINDSIGHT_BANK_ID` per user for isolated memory per person.
- **Explore the MCP alternative.** Hermes supports MCP servers natively. You can connect Hindsight's MCP server directly (`http://localhost:8888/mcp`) instead of the plugin — no `hindsight-hermes` package needed. The tradeoff is that the plugin registers tools with Hermes-native schemas, while MCP tools need discovery.
- **Use manual registration for more control.** If you want to set tags, recall filters, or skip the plugin system, `hindsight-hermes` exposes `register_tools()` and `memory_instructions()` functions for direct use.
- **Read the docs.** Full integration reference at [hindsight.vectorize.io/sdks/integrations/hermes](https://hindsight.vectorize.io/sdks/integrations/hermes).
