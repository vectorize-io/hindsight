---
title: "OpenAI Agents Forget Everything Between Runs. Here's the Fix."
authors: [DK09876]
date: 2026-04-17
tags: [openai, integrations, agents, memory, python]
description: "OpenAI Agents SDK agents lose all state when a run ends. hindsight-openai-agents adds three tools and auto-injected memory instructions that give your agents persistent memory across sessions."
image: /img/blog/openai-agents-persistent-memory.png
hide_table_of_contents: true
---

![OpenAI Agents persistent memory with Hindsight](/img/blog/openai-agents-persistent-memory.png)

OpenAI's Agents SDK gives you a clean abstraction for building tool-using agents — define an `Agent`, give it tools, run it with `Runner.run()`. But when the run ends, the agent forgets everything. `hindsight-openai-agents` fixes that by giving OpenAI agents persistent memory through three callable tools and auto-injected memory instructions.

<!-- truncate -->

## TL;DR

- OpenAI Agents SDK has Sessions for conversation history but no cross-session semantic memory — state doesn't carry over between runs
- `hindsight-openai-agents` provides three `FunctionTool` instances for `Agent`: `hindsight_retain`, `hindsight_recall`, `hindsight_reflect`
- Or use `memory_instructions()` to auto-inject relevant memories into the system prompt on every run — no explicit recall tool calls needed
- One pip install, pass `tools=[...]` to your agent, done
- Works with [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) or self-hosted

---

## The problem

OpenAI Agents SDK gives you `Agent` with tool calling and `Runner.run()` for execution. It has a `Session` protocol (including `SQLiteSession`) for persisting conversation history — but that's a message log. It doesn't extract facts, doesn't build knowledge over time, and doesn't retrieve relevant context semantically.

For agents that serve repeat users or run across multiple sessions, you need more:

- A coding assistant that remembers your stack, preferences, and past decisions
- A customer support agent that knows your account history across dozens of conversations
- A personal AI that accumulates knowledge about you over weeks and months

None of this works with conversation history alone. You need a system that extracts facts from conversations, builds knowledge over time, and retrieves relevant context semantically.

That's what Hindsight does. And `hindsight-openai-agents` wires it into the Agents SDK's tool system.

---

## Architecture

```
OpenAI Agent(tools=[...])
  └─ Hindsight FunctionTools (via create_hindsight_tools)
       ├─ hindsight_retain    → Hindsight retain
       │                        (fact extraction, entity resolution, knowledge graph)
       ├─ hindsight_recall    → Hindsight recall
       │                        (semantic + BM25 + graph + temporal retrieval)
       └─ hindsight_reflect   → Hindsight reflect
                                (synthesize a reasoned answer from all memories)
```

The tools are `FunctionTool` instances compatible with the Agents SDK, passed directly to `Agent(tools=[...])`. No subclassing, no custom agent types — just standard tool use.

Under the hood, Hindsight extracts structured facts, identifies entities, builds a knowledge graph, and runs four parallel retrieval strategies with cross-encoder reranking.

---

## Step 1: Start Hindsight

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=YOUR_OPENAI_KEY
hindsight-api
```

Runs locally at `http://localhost:8888` with embedded Postgres, embeddings, and reranking. See the [self-hosting quickstart](/developer/api/quickstart) for full setup instructions.

Or skip self-hosting entirely — [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) is ready in seconds with no infrastructure to manage.

## Step 2: Install the integration

```bash
pip install hindsight-openai-agents openai-agents
```

`hindsight-openai-agents` pulls in `openai-agents` and `hindsight-client`. The `openai-agents` package provides `Agent`, `Runner`, and `FunctionTool`.

## Step 3: Create the bank and agent

Banks must exist before use. The Agents SDK is async-first, so wrap everything in `asyncio.run()`:

```python
import asyncio
from agents import Agent, Runner
from hindsight_client import Hindsight
from hindsight_openai_agents import create_hindsight_tools

async def main():
    client = Hindsight(base_url="http://localhost:8888")
    await client.acreate_bank("user-123", name="User 123 Memory")

    tools = create_hindsight_tools(
        client=client,
        bank_id="user-123",
        tags=["source:chat"],
        budget="mid",
    )

    agent = Agent(
        name="assistant",
        model="gpt-4o-mini",
        tools=tools,
        instructions=(
            "You are a helpful assistant with long-term memory. "
            "Use hindsight_retain to store important facts the user shares. "
            "Use hindsight_recall to search memory before answering questions."
        ),
    )

    # Session 1: store preferences
    result = await Runner.run(agent, input="I'm a data scientist. I use Python, SQL, and VS Code with dark mode.")

    # Wait for Hindsight to finish processing. After hindsight_retain is called,
    # Hindsight processes the content asynchronously — extracting facts, resolving
    # entities, generating embeddings, and updating the knowledge graph. This
    # typically takes 1-3 seconds depending on content length and server load.
    await asyncio.sleep(3)

    # Session 2: recall from memory (same bank, memory persists)
    result = await Runner.run(agent, input="What IDE do I use?")
    print(result.final_output)
    # → "You use VS Code with dark mode."

    # Clean up
    await client.aclose()

asyncio.run(main())
```

Three tools, one bank. Memory persists across runs because it's stored in Hindsight, not in the agent. Unlike AutoGen's `AssistantAgent`, OpenAI Agents always produce a text response after tool calls — no `reflect_on_tool_use` flag needed.

---

## Automatic memory with `memory_instructions()`

The tools approach above gives the agent full control over when to recall. But for most use cases, you want relevant memories injected automatically — no tool call required. That's what `memory_instructions()` does.

```python
from hindsight_openai_agents import create_hindsight_tools, memory_instructions

agent = Agent(
    name="assistant",
    model="gpt-4o-mini",
    instructions=memory_instructions(
        client=client,
        bank_id="user-123",
        base_instructions=(
            "You are a helpful assistant with long-term memory. "
            "Use hindsight_retain to store important facts the user shares."
        ),
    ),
    tools=create_hindsight_tools(
        client=client,
        bank_id="user-123",
        include_recall=False,  # recall handled by memory_instructions
    ),
)
```

`memory_instructions()` takes your static `base_instructions` and returns an async callable that composes them with recalled memories on every run. The Agents SDK's `Agent(instructions=...)` accepts `str | Callable | None`, so the callable slots in directly. The agent still uses `hindsight_retain` to store new information, but recall happens automatically — relevant memories are pre-loaded into the system prompt before the agent even starts reasoning.

This is the recommended approach for most use cases. It reduces latency (no recall tool round-trip), simplifies the agent's decision-making (it doesn't have to decide *when* to recall), and ensures context is always present.

---

## Per-user memory banks

Parameterize `bank_id` for per-user isolation:

```python
def create_agent_for_user(user_id: str) -> Agent:
    tools = create_hindsight_tools(
        client=client,
        bank_id=f"user-{user_id}",
    )
    return Agent(
        name="assistant",
        model="gpt-4o-mini",
        tools=tools,
    )
```

Each bank is fully isolated — no cross-user data leakage. For more patterns, see the [per-user memory cookbook](/cookbook/recipes/per-user-memory).

---

## Handoffs with shared memory

Handoffs are the Agents SDK's killer feature — one agent routes to another mid-conversation. Memory makes them better. When two agents share a `bank_id`, they share memory. The triage agent sees the user's history; the specialist can store and retrieve findings. No data copying between agents.

```python
from agents import Agent, Handoff

triage = Agent(
    name="triage",
    instructions=memory_instructions(
        client=client,
        bank_id="user-123",
        base_instructions="Route the user to the right specialist.",
    ),
    handoffs=[Handoff(agent=specialist)],
)

specialist = Agent(
    name="specialist",
    instructions=memory_instructions(
        client=client,
        bank_id="user-123",
        base_instructions="You are a domain expert.",
    ),
    tools=create_hindsight_tools(client=client, bank_id="user-123"),
)
```

Same `bank_id` means shared memory. The triage agent gets the user's full context injected via `memory_instructions()` before deciding where to route. The specialist picks up that same context and can store new findings with `hindsight_retain`. Everything stays in one memory bank — no copying, no syncing, no coordination code.

---

## When to use this

- **Repeat-user agents** — Support bots, coding assistants, personal AI that should remember preferences and history across sessions
- **Multi-agent handoffs with shared memory** — Agents in a handoff chain retain findings so downstream agents start with context
- **Long-running workflows** — Agents that process data over days/weeks and need to accumulate knowledge incrementally
- **Personalization** — Any agent where "remembering the user" improves quality over time

## When NOT to use this

- **In-session context only** — If your agent only needs to remember things within a single `Runner.run()` call, the built-in conversation handling is simpler and has zero latency overhead
- **Conversation history replay** — If you just need to persist and replay message logs across sessions, use the SDK's `Session` protocol (e.g., `SQLiteSession`). Hindsight is for semantic memory, not chat logs
- **Document search (RAG)** — Hindsight is a memory system for facts learned over time, not a document store
- **Ephemeral agents** — If each agent invocation is stateless by design (batch processing, one-shot tasks), persistent memory adds complexity without benefit
- **Latency-critical hot paths** — Each memory operation adds a network round-trip

---

## Pitfalls and edge cases

**Bank must exist first.** Call `await client.acreate_bank(bank_id, name=...)` before the agent starts. If the bank doesn't exist, retain/recall will fail.

**Async processing delay.** After `hindsight_retain`, Hindsight processes content asynchronously — extracting facts, entities, embeddings. If you retain and immediately recall, the new memories may not be searchable yet. In practice, 1-3 seconds.

**Budget tuning.** Default `budget="mid"` balances speed and thoroughness. Use `"low"` for latency-sensitive agents, `"high"` for deep analysis.

**Reflect vs recall.** Use `hindsight_recall` for raw facts ("What IDE do I use?"). Use `hindsight_reflect` for synthesis ("Based on everything you know, what should I prioritize?"). Reflect is slower but produces reasoned answers that draw on the full knowledge graph.

---

## How this compares

**vs. Sessions / SQLiteSession:** The Agents SDK's `Session` protocol persists conversation messages across runs. That's a message log — it doesn't extract facts, doesn't generalize, and grows linearly with every conversation. Hindsight extracts structured facts, deduplicates, and retrieves only what's relevant.

**vs. raw vector stores (Pinecone, Weaviate, Chroma):** A vector store gives you embedding similarity search. Hindsight runs four parallel retrieval strategies (semantic, BM25, graph traversal, temporal) with cross-encoder reranking, plus entity resolution and a knowledge graph. It's a memory engine, not a database. See [benchmark results](/blog/2026/04/02/beam-sota).

**vs. other framework integrations:** If you're using AutoGen, LlamaIndex, LangGraph, CrewAI, or Pydantic AI instead of OpenAI Agents SDK, Hindsight has dedicated integrations for each — see [AutoGen](/sdks/integrations/autogen), [LlamaIndex](/sdks/integrations/llamaindex), and [others](/sdks/integrations/crewai).

---

## Recap

- `hindsight-openai-agents` gives OpenAI agents persistent memory via `FunctionTool` instances passed to `Agent(tools=[...])`
- Three tools: `hindsight_retain` (store), `hindsight_recall` (search), `hindsight_reflect` (synthesize)
- `memory_instructions()` auto-injects relevant memories into the system prompt on every run — the recommended approach for most use cases
- Works with any model supported by the OpenAI Agents SDK
- Per-user banks for memory isolation, tags for scoping, budget for speed/depth tradeoff
- Shared `bank_id` across handoff agents gives you persistent, shared memory with zero coordination code

---

## Next Steps

- **Try it locally:** `pip install hindsight-all hindsight-openai-agents openai-agents` and run the example above
- **Use Hindsight Cloud:** Skip self-hosting with a [free account](https://ui.hindsight.vectorize.io/signup)
- **Quickstart:** Get a Hindsight server running in minutes with the [developer quickstart](/developer/api/quickstart)
- **Explore the cookbook:** [Memory patterns across agent frameworks](/cookbook)
- **Other integrations:** [AutoGen](/sdks/integrations/autogen), [LlamaIndex](/sdks/integrations/llamaindex), [LangGraph](/sdks/integrations/langgraph), [Pydantic AI](/sdks/integrations/pydantic-ai), [CrewAI](/sdks/integrations/crewai)
