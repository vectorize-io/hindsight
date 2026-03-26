---
title: "Teaching the Llama to Remember"
description: LlamaIndex agents reset memory every session. Learn how to add persistent cross-session memory using hindsight-llamaindex in 3 steps. Full code examples included.
authors: [DK09876]
date: 2026-03-26T12:00
tags: [llamaindex, integrations, agents, memory, python]
image: /img/blog/llamaindex-agent-memory.png
hide_table_of_contents: true
---

![Teaching the Llama to Remember](/img/blog/llamaindex-agent-memory.png)

You've built a LlamaIndex agent that answers questions brilliantly. Then the session ends and it forgets everything, your user's name, their preferences, the context you spent three turns establishing. The next session starts from zero.

That's the core limitation of LlamaIndex's built-in memory: it's session-scoped. `ChatMemoryBuffer` holds your conversation history while the agent is running, but the moment you restart, that context is gone. For agents that serve repeat users or run multi-session workflows, this isn't a minor inconvenience. It's a fundamental capability gap.

`hindsight-llamaindex` solves this. It implements LlamaIndex's native `BaseToolSpec` to give agents persistent, searchable memory that survives across sessions, so your agent remembers what it learned, regardless of when the session started.

**What you'll learn:**
- Why LlamaIndex agent memory resets and what persistent memory actually means
- How `hindsight-llamaindex` integrates with any LlamaIndex agent via the native tool system
- How to set up cross-session memory in three steps with full code examples
- When to use persistent memory and when to skip it

Works with [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) or self-hosted.

<!-- truncate -->

## What Is LlamaIndex Agent Memory?

LlamaIndex has two generations of memory tooling. The original `ChatMemoryBuffer` stores raw conversation history in-session and resets on restart. The newer `Memory` class adds pluggable `MemoryBlock` modules -- including `FactExtractionMemoryBlock` -- backed by SQLite for persistence across restarts.

That's a real improvement. But even the newer system has hard limits: retrieval is vector similarity only, there's no knowledge graph or entity resolution, and it's tightly coupled to the LlamaIndex ecosystem. If you switch agent frameworks, your memory layer goes with it.

"Persistent" memory, in the fullest sense, means a system that extracts facts from conversations, builds a knowledge graph over time, and retrieves relevant context using multiple strategies in parallel -- not just the most recent embedding match. It should work regardless of which agent framework you're running.

`hindsight-llamaindex` provides this via three native LlamaIndex tools: `retain_memory` (store facts), `recall_memory` (search memory), and `reflect_on_memory` (synthesize a reasoned answer from everything stored).

## The Problem with LlamaIndex's Built-In Memory

LlamaIndex's newer `Memory` API with SQLite persistence is a genuine step forward -- facts survive restarts, and `FactExtractionMemoryBlock` can pull structured information from conversations. For simple use cases, it's enough.

But it runs into hard architectural limits as soon as queries get complex:

- **Single retrieval strategy.** LlamaIndex Memory uses vector similarity only. There's no BM25 for exact matches, no graph traversal for relationship queries, no temporal search for date-bounded recall. "What was I working on last Tuesday?" returns whatever is semantically close -- not what actually happened on Tuesday.
- **No knowledge graph.** Facts are stored as embeddings, not as entities with relationships. The system can't connect "Priya owns the rate-limiting service" to "the rate-limiting blocker is resolved" without those facts appearing in the same conversation turn.
- **Framework coupling.** LlamaIndex Memory is designed for LlamaIndex. If you migrate to a different agent framework, your memory layer doesn't come with you.
- **No synthesis.** There's no equivalent of `reflect_on_memory` -- a query that reasons across everything stored and returns a synthesized answer, not just a ranked list of relevant chunks.

For agents that serve repeat users, you need all of this:

- A coding assistant that remembers your stack, your team structure, and your open work items
- A support agent that knows a user's history across dozens of sessions and can connect past issues to current ones
- A research assistant that builds on findings from previous runs and understands what's already been explored

That's what Hindsight does. And `hindsight-llamaindex` wires it into LlamaIndex's standard tool system without replacing anything you already have.

## How LlamaIndex Persistent Memory Works

```
LlamaIndex Agent (ReAct, FunctionCalling, etc.)
  └─ HindsightToolSpec (extends BaseToolSpec)
       ├─ retain_memory()     → Hindsight retain
       │                        (fact extraction, entity resolution, knowledge graph)
       ├─ recall_memory()     → Hindsight recall
       │                        (semantic + BM25 + graph + temporal retrieval)
       └─ reflect_on_memory() → Hindsight reflect
                                (synthesize a reasoned answer from all memories)
```

`HindsightToolSpec` extends LlamaIndex's [`BaseToolSpec`](https://docs.llamaindex.ai/en/stable/api_reference/tools/). Call `to_tool_list()` and you get standard `FunctionTool` instances. No monkey-patching, no custom wrappers. Any LlamaIndex agent that accepts tools can use it.

Under the hood, Hindsight extracts structured facts, identifies entities, builds a knowledge graph, and runs four parallel retrieval strategies with cross-encoder reranking.

## Step 1: Start Hindsight

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=YOUR_OPENAI_KEY
hindsight-api
```

Runs locally at `http://localhost:8888` with embedded Postgres, embeddings, and reranking.

Or use [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) and skip self-hosting entirely.

## Step 2: Install the Integration

```bash
pip install hindsight-llamaindex
```

Pulls in `llama-index-core` and `hindsight-client`.

## Step 3: Create the Bank and Agent

Banks must exist before use. Since LlamaIndex agents are async, wrap everything in `asyncio.run()` for scripts (or use top-level `await` in notebooks):

```python
import asyncio
from hindsight_client import Hindsight
from hindsight_llamaindex import HindsightToolSpec
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent

async def main():
    client = Hindsight(base_url="http://localhost:8888")
    await client.acreate_bank("user-123", name="User 123 Memory")

    spec = HindsightToolSpec(
        client=client,
        bank_id="user-123",
        tags=["source:chat"],
        budget="mid",
    )
    tools = spec.to_tool_list()

    agent = ReActAgent(
        tools=tools,
        llm=OpenAI(model="gpt-4o-mini"),
        system_prompt=(
            "You are a helpful assistant with long-term memory. "
            "Use retain_memory to store important facts. "
            "Use recall_memory to search memory before answering."
        ),
    )

    # Session 1
    await agent.run(
        "I'm a data scientist. I use Python, SQL, and VS Code with dark mode."
    )

    # Session 2 (new agent instance, same bank_id)
    agent = ReActAgent(tools=tools, llm=OpenAI(model="gpt-4o-mini"))
    response = await agent.run("What IDE do I use?")
    # → "You use VS Code with dark mode."
    print(response)

asyncio.run(main())
```

Three tools, one bank. Memory persists across agent instances because it's stored in Hindsight, not in the agent object.

## Convenience Factory

If you don't need full `BaseToolSpec` control:

```python
from hindsight_llamaindex import create_hindsight_tools

tools = create_hindsight_tools(
    client=client,
    bank_id="user-123",
    include_reflect=False,  # only retain + recall
)
```

The factory wraps `HindsightToolSpec` and returns a filtered list. Use `include_retain`, `include_recall`, and `include_reflect` to control which tools are exposed.

## Selective Tools and Per-User Memory Banks

LlamaIndex's `BaseToolSpec` natively supports selective export:

```python
spec = HindsightToolSpec(client=client, bank_id="user-123")

# Read-only: only recall and reflect
tools = spec.to_tool_list(spec_functions=["recall_memory", "reflect_on_memory"])
```

This is useful when different agents need different permissions. A research agent gets all three tools. A reporting agent gets read-only access.

For multi-user applications, parameterize `bank_id` for per-user isolation:

```python
def create_agent_for_user(user_id: str) -> ReActAgent:
    spec = HindsightToolSpec(
        client=client,
        bank_id=f"user-{user_id}",
    )
    return ReActAgent(
        tools=spec.to_tool_list(),
        llm=OpenAI(model="gpt-4o-mini"),
    )
```

Each bank is fully isolated. No cross-user data leakage.

## Global Configuration

Set defaults once, override per-call:

```python
from hindsight_llamaindex import configure

configure(
    hindsight_api_url="http://localhost:8888",
    api_key="YOUR_API_KEY",  # or HINDSIGHT_API_KEY env var
    budget="mid",
    tags=["source:llamaindex"],
)

# Now you can skip client= and url= arguments
tools = create_hindsight_tools(bank_id="user-123")
```

Per-call parameters override global config. Parameters you don't pass fall through to the global defaults.

## Real-World Use Cases

Persistent memory is most valuable in agents designed for ongoing, multi-session relationships. A few patterns that work well:

**Personal coding assistant.** A developer uses the same agent across dozens of sessions. After the first session, the agent knows their stack (Python, gRPC, Postgres), their team structure, and their open work items. Every subsequent session starts with full context rather than blank state. The developer never re-explains their environment.

**Customer support agent.** A support bot serves a user who contacts it five times over two months. Without persistent memory, each interaction starts with identity verification and problem re-explanation. With Hindsight, the agent recalls the user's account tier, their previous issue history, and the resolution steps already tried. Resolution time drops because the agent skips the orientation phase.

**Research assistant.** A knowledge-gathering agent works through a multi-week research project across many sessions. Each session picks up where the last left off; the agent recalls which sources have been reviewed, which hypotheses have been tested, and which questions remain open. The agent accumulates expertise rather than starting fresh.

In each case, the value is the same: the agent's context grows over time rather than resetting. The longer the relationship, the more useful the agent becomes.

## Common Pitfalls

**Bank must exist first.** Call `client.create_bank(bank_id, name=...)` before the agent starts. If the bank doesn't exist, retain/recall will fail with a 404-style error.

**Async processing delay.** After `retain_memory`, Hindsight processes content asynchronously. Extracting facts, entities, and embeddings takes 1-3 seconds. If you retain and immediately recall in the same session, the new memories may not be searchable yet.

**Budget tuning.** Default `budget="mid"` balances speed and thoroughness. Use `"low"` for latency-sensitive agents, `"high"` for deep analysis. Budget controls how many retrieval strategies run and how much reranking happens.

**Reflect vs. recall.** Use `recall_memory` for raw facts ("What IDE do I use?"). Use `reflect_on_memory` for synthesis ("Based on everything you know, what should I prioritize?"). Reflect is slower but produces reasoned answers that draw on the full knowledge graph.

**LlamaIndex agents are async.** `HindsightToolSpec` provides both sync and async tool methods. Async agents like [`ReActAgent`](https://docs.llamaindex.ai/en/stable/api_reference/agent/react/) use the async variants (`aretain`, `arecall`, `areflect`) directly.

## When NOT to Use This

Persistent memory isn't always the right tool.

- **In-session context only.** If your agent only needs to remember things within a single conversation, LlamaIndex's `ChatMemoryBuffer` is simpler and has zero latency overhead. Don't add Hindsight just because you can.
- **Document search (RAG).** If you need vector search over a document corpus, use LlamaIndex's built-in `VectorStoreIndex`. Hindsight is a memory system for facts learned over time, not a document store.
- **Ephemeral agents.** If each agent invocation is stateless by design -- batch processing, one-shot tasks -- persistent memory adds complexity without benefit.
- **Latency-critical hot paths.** Each memory operation adds a network round-trip. If sub-100ms response time matters more than personalization, skip it.

## How This Compares

| | `ChatMemoryBuffer` | LlamaIndex `Memory` (SQLite) | Raw vector store | Hindsight |
|---|---|---|---|---|
| **Scope** | In-session only | Persistent | Persistent | Persistent |
| **What it stores** | Raw conversation history | Extracted facts (flat) | Embeddings of arbitrary content | Extracted facts + entities + knowledge graph |
| **Retrieval** | Sequential context window | Vector similarity only | Vector similarity only | Semantic + BM25 + graph + temporal, with reranking |
| **Entity resolution** | No | No | No | Yes |
| **Synthesis** | No | No | No | Yes (`reflect_on_memory`) |
| **Framework coupling** | LlamaIndex only | LlamaIndex only | None | None |
| **Best for** | Single-session chat | Simple cross-session recall within LlamaIndex | Document search (RAG) | Long-term user/agent memory across frameworks |

**vs. LangGraph/LangChain:** If you're using LangGraph instead of LlamaIndex, see [`hindsight-langgraph`](https://docs.hindsight.vectorize.io/docs/sdks/integrations/langgraph) which offers tools, graph nodes, and a `BaseStore` adapter.

## FAQ

**Why isn't `recall_memory` returning results I just stored?**
Hindsight processes retained content asynchronously. After calling `retain_memory`, wait 1-3 seconds before calling `recall_memory`. In production, architect your workflow so retain and recall happen in separate sessions rather than back-to-back in the same one.

**What happens if I call `recall_memory` before the bank exists?**
The call will fail. Always call `client.create_bank(bank_id)` before initializing `HindsightToolSpec`. If you're not sure whether the bank exists, wrap the create call in a try/except and handle the conflict error.

**My agent isn't calling `retain_memory` automatically. Why?**
The agent decides when to call tools based on its system prompt. If your system prompt doesn't instruct the agent to store important facts, it won't do so reliably. Include explicit instruction: `"Use retain_memory to store any facts, preferences, or context the user shares."` Then test with a few turns to confirm the behavior.

**Can I use Hindsight with a non-ReAct LlamaIndex agent?**
Yes. `HindsightToolSpec` returns standard `FunctionTool` instances via `to_tool_list()`. Any LlamaIndex agent that accepts a tool list supports it, including `FunctionCallingAgent` and custom agent implementations that use LlamaIndex's tool abstractions.

## Recap

Session memory resets. Long-term memory doesn't have to.

`hindsight-llamaindex` fits into any LlamaIndex agent in three steps: install the package, create a bank, pass the tools. From that point, every fact your agent learns is stored, indexed, and retrievable in the current session and in every session that follows. The agent that served a user in January knows what it learned in January when that user returns in March.

That's the practical difference between a session-scoped tool and an agent with genuine memory. The more it's used, the more useful it becomes.

Try it now: `pip install hindsight-all hindsight-llamaindex` and run the example above. Or start with [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) if you'd rather skip self-hosting.

## Next Steps

- **Docs**: [LlamaIndex integration guide](https://docs.hindsight.vectorize.io/docs/sdks/integrations/llamaindex)
- **Other integrations**: [LangGraph](https://docs.hindsight.vectorize.io/docs/sdks/integrations/langgraph), [Pydantic AI](https://docs.hindsight.vectorize.io/docs/sdks/integrations/pydantic-ai), [CrewAI](https://docs.hindsight.vectorize.io/docs/sdks/integrations/crewai)
