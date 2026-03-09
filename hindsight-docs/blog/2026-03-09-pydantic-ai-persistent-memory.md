---
title: "Pydantic AI Persistent Memory: Add It in 5 Lines of Code"
authors: [hindsight]
date: 2026-03-09
tags: [memory, openai, anthropic, gemini, python, rust, agents, rag, vector, pydantic-ai, knowledge-graph]
image: /img/blog/pydantic-ai-persistent-memory.png
hide_table_of_contents: true
---

If you have built an AI agent with [Pydantic AI](https://ai.pydantic.dev/), you already know it handles typed outputs, dependency injection, and async workflows well. But there is one thing it does not do: remember anything between runs. Every call to `agent.run()` starts with a blank slate. Your agent has no idea what the user said yesterday, what preferences they shared, or what it already researched.

Adding **Pydantic AI persistent memory** does not require building a custom RAG pipeline or managing your own vector database. With the `hindsight-pydantic-ai` integration, you can wire long-term memory into any Pydantic AI agent in five lines of Python. This guide walks through the full setup, from installation to production-ready patterns.

<!-- truncate -->

## TL;DR

- Pydantic AI has no built-in persistent memory. Your agent starts from scratch every run.
- `hindsight-pydantic-ai` adds retain, recall, and reflect tools plus auto-injected memory instructions.
- Five lines of setup: create a client, call `create_hindsight_tools()`, and pass them to your Agent.
- `memory_instructions()` silently pre-loads relevant memories into the system prompt on every run, so the agent starts each conversation with context.
- Works with any model provider, including OpenAI, Anthropic, and Gemini.

---

## The Problem: Pydantic AI Has No Persistent Memory

[Pydantic AI](https://ai.pydantic.dev/) is a solid framework. Typed outputs, dependency injection, async-native design, and a clean tool API make it a popular choice for building Python-based agents. However, it ships with no memory layer at all.

Every `agent.run()` starts from zero. The agent does not know what the user said yesterday. It does not know their preferences. It does not know what it already researched. As a result, agents that interact with users over multiple sessions lose all accumulated context between runs.

You can pass `message_history` to continue a conversation within a single session. But that is chat history, not memory. Chat history does not generalize facts. It does not consolidate repeated information. And it grows linearly until it exceeds your context window and token limit.

Real agent memory means something different:

- Extracting structured facts from conversations
- Building a knowledge graph of entities and relationships
- Retrieving relevant context across days, weeks, and months
- Synthesizing coherent answers from scattered memories

That is what [Hindsight](https://hindsight.vectorize.io/) provides. If you have used Hindsight with other frameworks like [CrewAI](/blog/2026/03/02/crewai) or [OpenAI](/blog/2026/03/05/add-memory-to-openai-application), the concept is the same. And `hindsight-pydantic-ai` wires it directly into Pydantic AI's tool and instruction system, so you do not need to build any of this yourself.

---

## How Pydantic AI Persistent Memory Works with Hindsight

Before diving into code, it helps to understand what happens under the hood when you add Pydantic AI persistent memory through the Hindsight integration.

Hindsight is a memory engine that runs locally or in the cloud. When your agent stores a fact, Hindsight does not just save raw text. Instead, it extracts structured entities and relationships, builds a knowledge graph, and indexes everything for multi-strategy retrieval. That means when your agent searches memory later, it can find relevant information through semantic search, BM25 keyword matching, graph traversal, and temporal ranking, all combined.

The `hindsight-pydantic-ai` package connects this engine to Pydantic AI through two integration points:

```
Pydantic AI Agent
  |-- tools=[create_hindsight_tools(...)]
  |     |-- hindsight_retain  -> store facts to memory
  |     |-- hindsight_recall  -> search memory for relevant info
  |     |-- hindsight_reflect -> synthesize an answer from all memories
  |
  |-- instructions=[memory_instructions(...)]
        |-- auto-recalls relevant memories into the system prompt
```

**Tools** let the agent explicitly store and retrieve memories during a conversation. **Instructions** silently inject relevant memories before the agent even starts thinking. Both are optional. You can use one, the other, or both together depending on your use case.

The tools are async functions that call Hindsight's API directly. Since Pydantic AI is async-native, the closures use `aretain()`, `arecall()`, and `areflect()`. There are no thread-pool hacks or compatibility layers needed. If you prefer a protocol-based approach, you can also use [Hindsight's MCP memory server](/blog/2026/03/04/mcp-agent-memory) instead of direct tool integration.

---

## Setting Up Pydantic AI Persistent Memory in 5 Steps

The setup process takes about five minutes. You will install Hindsight, add the Pydantic AI memory integration package, wire the tools into your agent, and test that memories persist across sessions.

### Step 1: Install and Start Hindsight

First, install the Hindsight server and start it locally:

```bash
pip install hindsight-all
```

```bash
export HINDSIGHT_API_LLM_API_KEY=YOUR_OPENAI_KEY
hindsight-api
```

This runs locally at `http://localhost:8888`. It includes embedded Postgres, local embeddings, and local reranking. No external services are required beyond an LLM API key for entity extraction.

> **Note:** You can also use [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) and skip the self-hosted setup entirely. The cloud version provides the same API with managed infrastructure.

### Step 2: Install the Pydantic AI Memory Integration

Next, install the integration package:

```bash
pip install hindsight-pydantic-ai
```

You also need a model provider. For OpenAI:

```bash
pip install "pydantic-ai-slim[openai]"
```

Other providers work the same way. Swap in `[anthropic]` or `[google]` as needed. The Pydantic AI persistent memory integration is model-agnostic, so the memory layer works identically regardless of which LLM you choose. Hindsight supports [100+ LLM providers through LiteLLM](/blog/2026/03/03/litellm).

### Step 3: Add Persistent Memory Tools to Your Pydantic AI Agent

Here is where the five lines come in. Create a Hindsight client, generate the Pydantic AI persistent memory tools, and pass them to your agent:

```python
from hindsight_client import Hindsight
from hindsight_pydantic_ai import create_hindsight_tools
from pydantic_ai import Agent

client = Hindsight(base_url="http://localhost:8888")

agent = Agent(
    "openai:gpt-4o-mini",
    tools=create_hindsight_tools(client=client, bank_id="user-123"),
)
```

That is the full Pydantic AI persistent memory setup. The agent now has three tools:

- `hindsight_retain(content)` stores information to long-term memory
- `hindsight_recall(query)` searches memory and returns matching facts
- `hindsight_reflect(query)` synthesizes a reasoned answer from all relevant memories

The agent decides when to use each tool based on the conversation context. You do not need to call them manually.

### Step 4: Test Cross-Session Memory

Run two separate conversations to verify that Pydantic AI agent memory persists across sessions:

```python
import asyncio

async def main():
    # First conversation
    r1 = await agent.run(
        "Remember that I prefer functional programming patterns "
        "and I'm building a data pipeline in Python."
    )
    print(r1.output)

    # Later conversation -- agent recalls context
    r2 = await agent.run("What approach should I take for error handling?")
    print(r2.output)

asyncio.run(main())
```

In the first run, the agent stores the preferences via `hindsight_retain`. In the second run, the agent calls `hindsight_recall` to find relevant context. It then gives advice grounded in what it knows: functional patterns, Python, data pipelines.

This is the core benefit of Pydantic AI persistent memory: it works across runs, across days, and across process restarts. The memories live in Hindsight's knowledge graph, not in the agent's context window. Even if you restart your Python process completely, the agent picks up right where it left off.

### Step 5: Auto-Inject Memories with Pydantic AI Instructions

The tools above require the agent to decide to search memory on its own. Sometimes you want Pydantic AI persistent memory injected automatically, before the agent starts responding.

Pydantic AI's `instructions` parameter supports async callables that run on every `agent.run()`. This is a natural fit for memory injection:

```python
from hindsight_pydantic_ai import create_hindsight_tools, memory_instructions

agent = Agent(
    "openai:gpt-4o-mini",
    tools=create_hindsight_tools(client=client, bank_id="user-123"),
    instructions=[memory_instructions(client=client, bank_id="user-123")],
)
```

Now on every run, `memory_instructions` calls Hindsight's recall API and injects relevant memories into the system prompt. The agent starts every conversation with context about the user, without needing a tool call.

You can customize the query, result count, and prefix to control what gets injected:

```python
memory_instructions(
    client=client,
    bank_id="user-123",
    query="user preferences, history, and context",
    max_results=10,
    prefix="Here is what you know about this user:\n",
)
```

If recall fails or returns nothing, the instructions function returns an empty string. It never blocks the agent from responding.

---

## Advanced Pydantic AI Memory Configuration

### Selecting Which Memory Tools to Include

You do not always need all three persistent memory tools. `create_hindsight_tools` lets you pick which ones to include:

```python
# Read-only agent: can search memory but not write to it
tools = create_hindsight_tools(
    client=client,
    bank_id="user-123",
    include_retain=False,
    include_recall=True,
    include_reflect=True,
)

# Write-only agent: stores data but does not query
tools = create_hindsight_tools(
    client=client,
    bank_id="user-123",
    include_retain=True,
    include_recall=False,
    include_reflect=False,
)
```

This flexibility is useful in multi-agent architectures. For instance, you might have one Pydantic AI agent that gathers information and writes to persistent memory, while a separate agent reads memory to answer questions. Splitting read and write access keeps each agent focused on its role and prevents unintended memory writes from agents that should only consume context.

### Global Configuration for Multiple Agents

If you have multiple Pydantic AI agents sharing the same Hindsight persistent memory instance, use the global config instead of passing `client` everywhere:

```python
from hindsight_pydantic_ai import configure, create_hindsight_tools

configure(hindsight_api_url="http://localhost:8888", api_key="YOUR_KEY")

# No client needed: tools use the global config
agent1_tools = create_hindsight_tools(bank_id="agent-1")
agent2_tools = create_hindsight_tools(bank_id="agent-2")
```

An explicit `client=` parameter always takes priority over the global config. This lets you override on a per-agent basis when needed.

---

## Full Working Example of Pydantic AI Persistent Memory

Save this as `memory_agent.py` and run it to see Pydantic AI persistent memory in action:

```python
import asyncio

from hindsight_client import Hindsight
from hindsight_pydantic_ai import create_hindsight_tools, memory_instructions
from pydantic_ai import Agent

BANK_ID = "demo-user"


async def main():
    client = Hindsight(base_url="http://localhost:8888")
    await client.acreate_bank(bank_id=BANK_ID, name="Demo User Memory")

    agent = Agent(
        "openai:gpt-4o-mini",
        tools=create_hindsight_tools(client=client, bank_id=BANK_ID),
        instructions=[memory_instructions(client=client, bank_id=BANK_ID)],
    )

    print("--- Run 1: Teaching the agent ---")
    r1 = await agent.run(
        "Remember: I'm a backend engineer. I use Python and Rust. "
        "I prefer small, composable libraries over large frameworks."
    )
    print(f"Agent: {r1.output}\n")

    print("--- Run 2: Agent recalls context ---")
    r2 = await agent.run("Recommend a web framework for my next project.")
    print(f"Agent: {r2.output}\n")

    print("--- Run 3: Agent synthesizes ---")
    r3 = await agent.run("What do you know about my engineering philosophy?")
    print(f"Agent: {r3.output}")


asyncio.run(main())
```

Run it:

```bash
export OPENAI_API_KEY=YOUR_KEY
python memory_agent.py
```

Run it again after the first execution finishes. The agent remembers everything from the first session because Pydantic AI persistent memory stores facts in Hindsight, not in the process.

---

## Pydantic AI Persistent Memory: Pitfalls and Edge Cases

**Bank ID collisions.** Each `bank_id` is a separate persistent memory store. If two unrelated agents share a bank, their memories merge in unexpected ways. Use unique bank IDs per user, per agent, or per project.

**Instruction latency.** `memory_instructions` makes a recall API call on every `agent.run()`. For latency-sensitive applications, use `budget="low"` and a small `max_results`. Alternatively, skip automatic memory injection entirely and rely on the agent to call the recall tool only when needed. In most cases, the added latency from persistent memory lookups is 50-200ms depending on memory size and network conditions.

**Duplicate memories.** If the agent stores the same information multiple times, Hindsight deduplicates at the fact level. However, it is still better to give the agent clear guidance in the system prompt about when to store new facts versus when to skip.

**Async event loop conflicts.** The sync `create_bank()` method does not work inside `asyncio.run()` because it tries to create a nested event loop. Always use `await client.acreate_bank()` in async code. This is a common Python async pitfall, not specific to Hindsight or Pydantic AI persistent memory.

---

## Pydantic AI Memory: Tradeoffs and Alternatives

### When Pydantic AI Persistent Memory Makes Sense

Pydantic AI persistent memory with Hindsight works best for agents that interact with the same user or context across multiple sessions. Good use cases include personal assistants, support bots, research agents that accumulate knowledge, and any Python agent where remembering past interactions improves quality over time.

### When Not to Use It

Skip agent persistent memory for one-shot agents that never run again, stateless API handlers where each request is independent, or agents where you want full control over what goes into the prompt. In the last case, use the Hindsight Python client directly instead of the Pydantic AI integration.

### How It Compares to Alternatives

| Approach | Strengths | Weaknesses | Best For |
|---|---|---|---|
| **Hindsight + Pydantic AI** | Multi-strategy retrieval (semantic + BM25 + graph + temporal), structured fact extraction, synthesis engine | Requires running Hindsight server or using cloud | Multi-session agents needing deep memory |
| **Manual message_history** | Built into Pydantic AI, no extra dependencies | Does not generalize facts, grows until context window limit | Short single-session conversations |
| **Custom vector store + RAG** | Full control over embeddings and retrieval | You manage chunking, indexing, and retrieval yourself | Teams with existing vector infrastructure |
| **Mem0** | Another external memory option, easy API | Fewer retrieval strategies, no graph-based recall | Simpler memory needs without entity relationships |

The right choice depends on your requirements. For most Pydantic AI agents that need persistent memory across sessions with minimal setup, the Hindsight integration offers the fastest path to production.

---

## Recap: Pydantic AI Persistent Memory Integration

Adding Pydantic AI persistent memory does not require complex infrastructure or custom retrieval pipelines. The `hindsight-pydantic-ai` integration provides two functions that cover the full agent memory lifecycle:

- **`create_hindsight_tools()`** returns async tools that the agent calls to store and retrieve knowledge across sessions
- **`memory_instructions()`** auto-injects relevant memories on every run, so the agent starts with context and no tool call is needed

The integration is minimal by design. Two functions, no subclassing, no changes to your deps type. Just Pydantic AI tools and instructions that connect to a real memory engine.

Memories survive process restarts, build a knowledge graph over time, and compound in value as your agent learns more about each user. For Python developers building Pydantic AI agents that need persistent memory, this is the simplest path from stateless to stateful.

---

## Next Steps

- **Try it locally**: `pip install hindsight-all hindsight-pydantic-ai "pydantic-ai-slim[openai]"` and run the example above
- **Use Hindsight Cloud**: Skip self-hosting with a [free account](https://ui.hindsight.vectorize.io/signup)
- **Tag memories for scoping**: Use `tags` on retain and `recall_tags` on search to partition memories by project, environment, or topic
- **Combine tools and instructions**: Use `memory_instructions` for automatic context and tools for explicit store and retrieve during conversations
- **Read the Pydantic AI docs**: Learn more about [Pydantic AI tools and instructions](https://ai.pydantic.dev/tools/) to extend your agent further
- **Explore other integrations**: Add memory to [CrewAI agents](/blog/2026/03/02/crewai), [OpenAI apps](/blog/2026/03/05/add-memory-to-openai-application), or any framework via [MCP](/blog/2026/03/04/mcp-agent-memory)
- **Inspect the knowledge graph**: Run the Hindsight control plane or use the [cloud dashboard](https://ui.hindsight.vectorize.io/signup) to browse extracted facts, entities, and relationships
