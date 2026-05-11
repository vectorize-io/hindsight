---
title: "Persistent Memory for AutoGen Agents with Hindsight"
authors: [DK09876]
date: 2026-04-06
tags: [autogen, integrations, agents, memory, python, microsoft, tutorial]
description: "AutoGen agents lose all state when a session ends. hindsight-autogen adds three tools — retain, recall, reflect — that give your agents persistent memory across sessions."
image: /img/blog/autogen-persistent-memory.png
hide_table_of_contents: true
---

![Persistent Memory for AutoGen Agents with Hindsight](/img/blog/autogen-persistent-memory.png)

AutoGen is Microsoft's open-source framework for building multi-agent systems: conversable agents, group chats, tool use, code execution. But when a session ends, every agent in the conversation forgets everything. `hindsight-autogen` fixes that by giving AutoGen agents persistent memory through three callable tools.

<!-- truncate -->

## TL;DR

- AutoGen agents have no built-in cross-session memory; state resets every run
- `hindsight-autogen` provides three `FunctionTool` instances for `AssistantAgent`: `hindsight_retain`, `hindsight_recall`, `hindsight_reflect`
- One pip install, pass `tools=[...]` to your agent, done
- Works with [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) or self-hosted

## The problem

AutoGen gives you `AssistantAgent` with chat history within a session. That's a message list; it doesn't extract facts, doesn't build knowledge over time, and disappears when the process exits.

For agents that serve repeat users or run across multiple sessions, you need more:

- A coding assistant that remembers your stack, preferences, and past decisions
- A multi-agent team where a coordinator retains knowledge from previous group chats
- A support agent that knows your account history across dozens of conversations

None of this works with in-session chat history. You need a system that extracts facts from conversations, builds knowledge over time, and retrieves relevant context semantically.

That's what Hindsight does. And `hindsight-autogen` wires it into AutoGen's tool system.

## Architecture

```
AutoGen AssistantAgent(tools=[...])
  └─ Hindsight FunctionTools (via create_hindsight_tools)
       ├─ hindsight_retain    → Hindsight retain
       │                        (fact extraction, entity resolution, knowledge graph)
       ├─ hindsight_recall    → Hindsight recall
       │                        (semantic + BM25 + graph + temporal retrieval)
       └─ hindsight_reflect   → Hindsight reflect
                                (synthesize a reasoned answer from all memories)
```

The tools are `FunctionTool` instances from `autogen_core.tools`, passed directly to `AssistantAgent(tools=[...])`. No subclassing, no custom agent types, just standard AutoGen tool use.

Under the hood, Hindsight extracts structured facts, identifies entities, builds a knowledge graph, and runs four parallel retrieval strategies with cross-encoder reranking.

## Step 1: Start Hindsight

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=YOUR_OPENAI_KEY
hindsight-api
```

Runs locally at `http://localhost:8888` with embedded Postgres, embeddings, and reranking.

Or use [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) and skip self-hosting.

## Step 2: Install the integration

```bash
pip install hindsight-autogen autogen-agentchat "autogen-ext[openai]"
```

`hindsight-autogen` pulls in `autogen-core` and `hindsight-client`. You also need `autogen-agentchat` for `AssistantAgent` and `autogen-ext[openai]` for the model client.

## Step 3: Create the bank and agent

Banks must exist before use. AutoGen agents are async, so wrap everything in `asyncio.run()`:

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from hindsight_client import Hindsight
from hindsight_autogen import create_hindsight_tools

async def main():
    client = Hindsight(base_url="http://localhost:8888")
    await client.acreate_bank("user-123", name="User 123 Memory")

    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    tools = create_hindsight_tools(
        client=client,
        bank_id="user-123",
        tags=["source:chat"],
        budget="mid",
    )

    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=tools,
        reflect_on_tool_use=True,
        system_message=(
            "You are a helpful assistant with long-term memory. "
            "Use hindsight_retain to store important facts the user shares. "
            "Use hindsight_recall to search memory before answering questions."
        ),
    )

    # Session 1: store preferences
    result = await agent.run(
        task="I'm a data scientist. I use Python, SQL, and VS Code with dark mode.",
    )

    # Wait for Hindsight to finish processing (fact extraction is async)
    await asyncio.sleep(3)

    # Session 2: recall from memory (same bank, memory persists)
    result = await agent.run(
        task="What IDE do I use?",
    )
    print(result.messages[-1].content)
    # → "You use VS Code with dark mode."

    # Clean up
    await client.aclose()
    await model_client.close()

asyncio.run(main())
```

Three tools, one bank. Memory persists across conversations because it's stored in Hindsight, not in the agent.

## Per-user memory banks

Parameterize `bank_id` for per-user isolation:

```python
def create_agent_for_user(user_id: str) -> AssistantAgent:
    tools = create_hindsight_tools(
        client=client,
        bank_id=f"user-{user_id}",
    )
    return AssistantAgent(
        name="assistant",
        model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
        tools=tools,
    )
```

Each bank is fully isolated; no cross-user data leakage.

## When to use this

- **Repeat-user agents** — Support bots, coding assistants, personal AI that should remember preferences and history across sessions
- **Multi-agent teams with shared memory** — A coordinator agent retains findings from group chats so future sessions start with context
- **Long-running workflows** — Agents that process data over days/weeks and need to accumulate knowledge incrementally
- **Personalization** — Any agent where "remembering the user" improves quality over time

## When NOT to use this

Be explicit: persistent memory isn't always the right tool.

- **In-session context only** — If your agent only needs to remember things within a single conversation, AutoGen's built-in chat history is simpler and has zero latency overhead. Don't add Hindsight just because you can.
- **Document search (RAG)** — If you need vector search over a document corpus, use a dedicated vector store. Hindsight is a memory system for facts learned over time, not a document store.
- **Ephemeral agents** — If each agent invocation is stateless by design (batch processing, one-shot tasks), persistent memory adds complexity without benefit.
- **Latency-critical hot paths** — Each memory operation adds a network round-trip. If sub-100ms response time matters more than personalization, skip it.

## Pitfalls and edge cases

**Bank must exist first.** Call `await client.acreate_bank(bank_id, name=...)` before the agent starts. If the bank doesn't exist, retain/recall will fail.

**Async processing delay.** After `hindsight_retain`, Hindsight processes content asynchronously, extracting facts, entities, embeddings. If you retain and immediately recall, the new memories may not be searchable yet. In practice, 1-3 seconds.

**Budget tuning.** Default `budget="mid"` balances speed and thoroughness. Use `"low"` for latency-sensitive agents, `"high"` for deep analysis. Budget controls how many retrieval strategies run and how much reranking happens.

**Reflect vs recall.** Use `hindsight_recall` for raw facts ("What IDE do I use?"). Use `hindsight_reflect` for synthesis ("Based on everything you know, what should I prioritize?"). Reflect is slower but produces reasoned answers that draw on the full knowledge graph.

## How this compares

**vs. AutoGen chat history:** Chat history stores raw messages in-session. It doesn't extract facts, doesn't generalize, and disappears when the conversation ends. Hindsight extracts structured facts, deduplicates, and retrieves only what's relevant — it compresses knowledge rather than accumulating tokens.

**vs. raw vector stores (Pinecone, Weaviate, Chroma):** A vector store gives you embedding similarity search. Hindsight runs four parallel retrieval strategies (semantic, BM25, graph traversal, temporal) with cross-encoder reranking, plus it extracts entities, resolves coreferences, and builds a knowledge graph. It's a memory engine, not a database. For independent benchmark results on what that architecture achieves at scale, see [Hindsight on BEAM](https://hindsight.vectorize.io/blog/2026/04/02/beam-sota).

**vs. other framework integrations:** If you're using LlamaIndex, LangGraph, CrewAI, or Pydantic AI instead of AutoGen, Hindsight has dedicated integrations for each: [LlamaIndex](/sdks/integrations/llamaindex), [LangGraph](/sdks/integrations/langgraph), [CrewAI](/sdks/integrations/crewai), [Pydantic AI](/sdks/integrations/pydantic-ai).

## Recap

- `hindsight-autogen` gives AutoGen agents persistent memory via `FunctionTool` instances passed to `AssistantAgent(tools=[...])`
- Three tools: `hindsight_retain` (store), `hindsight_recall` (search), `hindsight_reflect` (synthesize)
- Works with any AutoGen `AssistantAgent`, single agents or multi-agent teams
- Per-user banks for memory isolation, tags for scoping, budget for speed/depth tradeoff

## Next steps

- **Try it locally:** `pip install hindsight-all hindsight-autogen autogen-agentchat "autogen-ext[openai]"` and run the example above
- **Use Hindsight Cloud:** Skip self-hosting with a [free account](https://ui.hindsight.vectorize.io/signup)
- **Benchmark results:** [Why Hindsight leads on BEAM at 10M tokens](https://hindsight.vectorize.io/blog/2026/04/02/beam-sota)
- **Explore other integrations:** [LlamaIndex](/sdks/integrations/llamaindex), [LangGraph](/sdks/integrations/langgraph), [Pydantic AI](/sdks/integrations/pydantic-ai), [CrewAI](/sdks/integrations/crewai)
