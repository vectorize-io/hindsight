---
title: "Guide: Beyond RAG — Adding Agent Memory to LlamaIndex"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, llamaindex, rag, memory]
description: "LlamaIndex RAG retrieves what your documents say; Hindsight adds the agent memory layer that remembers what past sessions learned and what each user prefers."
image: /img/guides/guide-llamaindex-agent-memory-beyond-rag.svg
hide_table_of_contents: true
---

![Guide: Beyond RAG — Adding Agent Memory to LlamaIndex](/img/guides/guide-llamaindex-agent-memory-beyond-rag.svg)

LlamaIndex is best known for RAG: building an index over your documents and retrieving the passages that answer a question. That is retrieval over a fixed corpus of knowledge. **Agent memory** is a different thing — it is what happened in past sessions, which decisions were made, and what this particular user prefers. RAG answers "what do the docs say"; memory answers "what did we learn, and what does this user want."

These two capabilities are not competitors. They complement each other. A LlamaIndex agent with only RAG re-meets every user as a stranger and rediscovers the same context on every run. A LlamaIndex agent with only memory has continuity but no grounded knowledge base to draw on. Put both together and the agent recalls the durable, authoritative knowledge from your index *and* the evolving, personal context from prior conversations.

This guide explains where RAG stops and memory begins, then shows how Hindsight adds the memory layer to LlamaIndex through the two patterns the integration already ships: agent-driven tools (`HindsightToolSpec`) and automatic memory over LlamaIndex's `BaseMemory` interface. It is a conceptual and practical guide, not an install walkthrough — the [setup guide](https://hindsight.vectorize.io/guides/2026/05/04/guide-llamaindex-memory-with-hindsight) covers installation.

<!-- truncate -->

> **Quick answer**
>
> 1. Keep your LlamaIndex RAG index for grounded document knowledge.
> 2. Add Hindsight for agent memory — past sessions, decisions, preferences.
> 3. Use `HindsightMemory` (`BaseMemory`) for automatic recall/retain per turn.
> 4. Use `HindsightToolSpec` when the agent should decide when to store or search memory.
> 5. Run RAG and memory side by side: the index answers "what do the docs say," memory answers "what did we learn."

## RAG retrieves knowledge; memory retains experience

RAG and agent memory both "retrieve," which is why they get conflated, but they answer different questions.

**RAG** operates over a document index you built ahead of time. The corpus is authored and mostly static — product docs, a knowledge base, a codebase, a set of PDFs. When a query comes in, LlamaIndex embeds it, retrieves the most relevant chunks, and hands them to the model. The answer is grounded in *what the documents say*. If the documents don't change, the same query returns the same context tomorrow.

**Agent memory** operates over the stream of experience an agent accumulates while it runs. Nobody authors it up front — it is written as sessions happen. It captures what a user told the agent, which decisions were made and why, and preferences that only surface through interaction ("I prefer dark mode," "always use TypeScript," "we already tried that approach and it failed"). This is exactly the material a plain RAG index has no place to store, because it isn't in your documents — it *emerged* from usage.

Hindsight is built for that second job. It organizes memory as world facts (general knowledge), experience facts (what happened), and mental models (consolidated preferences and patterns) — structures a static document index does not provide.

## Where RAG stops and memory begins

A quick way to decide which layer a piece of information belongs to:

- **"What does the API reference say about pagination?"** → RAG. The answer lives in your documents.
- **"What did we decide about pagination last week?"** → memory. The answer lives in a past session.
- **"What is our refund policy?"** → RAG. It is authored knowledge.
- **"Does this user usually want the terse or the detailed explanation?"** → memory. It is a learned preference.

RAG stops at the edge of your corpus. The moment a useful fact is something the agent *learned* rather than something an author *wrote*, you are in memory territory. LlamaIndex gives you the retrieval engine for the first; Hindsight gives you the memory engine for the second.

## Adding the memory layer (tools + automatic BaseMemory)

The `hindsight-llamaindex` package exposes two complementary patterns. You can use either, or both together.

**Automatic memory (`HindsightMemory`)** implements LlamaIndex's `BaseMemory` interface, so memory happens without the agent thinking about it. On each turn, `aget(input)` recalls relevant memories from Hindsight and prepends them as a system message; `aput(message)` retains the message for future recall. You pass it to the agent as `memory=`:

```python
import asyncio
from hindsight_client import Hindsight
from hindsight_llamaindex import HindsightMemory
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

async def main():
    client = Hindsight(base_url="http://localhost:8888")

    memory = HindsightMemory.from_client(
        client=client,
        bank_id="user-123",
        mission="Track user preferences and project context",
    )

    agent = ReActAgent(tools=[], llm=OpenAI(model="gpt-4o"))
    response = await agent.run("Remember that I prefer dark mode", memory=memory)
    print(response)

asyncio.run(main())
```

**Agent-driven tools (`HindsightToolSpec`)** expose retain, recall, and reflect as tools the agent can choose to call. This fits when you want the model to decide *when* memory is worth writing or searching, rather than doing it on every turn:

```python
from hindsight_client import Hindsight
from hindsight_llamaindex import HindsightToolSpec
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

client = Hindsight(base_url="http://localhost:8888")

spec = HindsightToolSpec(
    client=client,
    bank_id="user-123",
    mission="Track user preferences",
)
tools = spec.to_tool_list()

agent = ReActAgent(tools=tools, llm=OpenAI(model="gpt-4o"))
```

You can narrow which tools are exposed with `spec.to_tool_list(spec_functions=["recall_memory", "reflect_on_memory"])`, or via the `create_hindsight_tools(...)` factory with `include_retain` / `include_recall` / `include_reflect` flags. Both patterns share the same bank, so a fact stored automatically is available to an explicit recall tool and vice versa. Full parameter tables are in [the LlamaIndex integration docs](https://hindsight.vectorize.io/docs/integrations/llamaindex).

## Connect LlamaIndex to Hindsight

Installation and first-run wiring are covered in the [LlamaIndex setup guide](https://hindsight.vectorize.io/guides/2026/05/04/guide-llamaindex-memory-with-hindsight). In short: `pip install hindsight-llamaindex`, point a `Hindsight` client at Hindsight Cloud or a local API, and pass a stable `bank_id`. Once that is in place, come back here for how memory sits alongside your RAG index.

## Using RAG and memory together

The clean architecture keeps the two layers separate and lets the agent draw on both:

- Your **LlamaIndex query engine / retriever** stays exactly as it is — grounded answers over your document index.
- **Hindsight memory** is layered on with `HindsightMemory` for automatic per-turn recall and retain.
- Optionally, expose an explicit `reflect` tool so the agent can reason over accumulated memory on demand.

The integration is designed for exactly this combination. You can attach automatic memory *and* a subset of tools at the same time — let `HindsightMemory` handle recall/retain while the tool spec exposes only `reflect`:

```python
from hindsight_llamaindex import create_hindsight_tools, HindsightMemory

# Automatic memory for context enrichment
memory = HindsightMemory.from_client(client=client, bank_id="user-123")

# Explicit reflect tool only; memory handles recall/retain
tools = create_hindsight_tools(
    client=client,
    bank_id="user-123",
    include_retain=False,
    include_recall=False,
    include_reflect=True,
)

agent = ReActAgent(tools=tools, llm=llm)
response = await agent.run("What should I prioritize?", memory=memory)
```

Your RAG tools live in that same `tools` list. The agent now retrieves grounded knowledge from the index and recalls learned context from memory in the same run — the docs tell it *what is true*, memory tells it *what this user and this project have decided*.

## Verify that memory is working

Because memory is about continuity across sessions, verify it across two runs, not one:

1. Run the agent once with a stable `bank_id` and state a preference or decision ("I prefer dark mode").
2. Start a fresh session with the *same* `bank_id`.
3. Ask something that depends on the earlier turn ("How should you format my output?").
4. Confirm `HindsightMemory` injects the earlier context without a manual tool call.

If the second run answers using the first run's detail, the memory layer is working. To confirm it is separate from RAG, ask a question your documents can't answer but a past session can — if the agent still gets it right, the answer came from memory.

## Common mistakes

### Treating RAG as memory

Stuffing conversation history or user preferences into your document index is fragile and blurs authored knowledge with emergent experience. Keep the index for documents and let Hindsight hold memory.

### Using a new bank ID per run

Continuity depends on reusing the same `bank_id`. A fresh bank each run gives the agent amnesia — recall and retain never hit the same store.

### Expecting memory to answer document questions

Memory holds what the agent learned, not your full corpus. "What does section 4 of the manual say" is a RAG question; don't expect memory to substitute for the index.

### Leaving the mission blank on a specialized agent

The bank `mission` guides fact extraction. For a domain-specific agent, a descriptive mission ("Track project decisions and coding preferences") produces sharper memories than the default.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works the same way — point the `Hindsight` client at your local API URL (e.g. `http://localhost:8888`). Cloud is just a hosted backend.

### Does adding memory replace my RAG pipeline?

No. They are separate layers. Your LlamaIndex retriever keeps answering document questions; Hindsight adds recall of past sessions and preferences on top.

### Should I use HindsightMemory or HindsightToolSpec?

Start with `HindsightMemory` for automatic per-turn recall and retain. Reach for `HindsightToolSpec` when the agent should explicitly decide when to store, search, or reflect — and combine both when you want automatic memory plus an on-demand `reflect` tool.

### Where do preferences and decisions actually live?

In the Hindsight bank keyed by your `bank_id`, organized as world facts, experience facts, and mental models — not in your document index.

## Next Steps

- Follow the [LlamaIndex setup guide](https://hindsight.vectorize.io/guides/2026/05/04/guide-llamaindex-memory-with-hindsight) for installation and first-run wiring
- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted memory backend
- Read [the full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow [the quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [LlamaIndex integration docs](https://hindsight.vectorize.io/docs/integrations/llamaindex)
