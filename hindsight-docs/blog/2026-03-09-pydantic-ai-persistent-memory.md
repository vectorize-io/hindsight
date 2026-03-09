---
title: "Your Pydantic AI Agent Forgets You After Every Run. Fix It in 5 Lines."
authors: [hindsight]
date: 2026-03-09
tags: [memory, openai, anthropic, gemini, python, rust, agents, rag, vector]
image: /img/blog/pydantic-ai-persistent-memory.png
hide_table_of_contents: true
---

## TL;DR

<!-- truncate -->

- Pydantic AI has no built-in persistent memory -- your agent starts from scratch every run
- `hindsight-pydantic-ai` adds retain/recall/reflect tools and auto-injected memory instructions
- Five lines of setup: create a client, call `create_hindsight_tools()`, pass to your Agent
- Bonus: `memory_instructions()` silently pre-loads relevant memories into the system prompt on every run
- Works with any model provider (OpenAI, Anthropic, Gemini, etc.)

---

## The Problem: Stateless Agents

Pydantic AI is great. Typed outputs, dependency injection, async-native, clean tool API. But it has no memory layer.

Every `agent.run()` starts from zero. The agent doesn't know what the user said yesterday. It doesn't know their preferences. It doesn't know what it already researched.

You can pass `message_history` to continue a conversation, but that's chat history -- not memory. It doesn't generalize, doesn't consolidate, and it grows until it blows your context window.

Real memory means:

- Extracting facts from conversations
- Building a knowledge graph of entities and relationships
- Retrieving relevant context across days, weeks, months
- Synthesizing coherent answers from scattered memories

That's what Hindsight does. And `hindsight-pydantic-ai` wires it directly into Pydantic AI's tool and instruction system.

---

## Architecture

```
Pydantic AI Agent
  ├─ tools=[create_hindsight_tools(...)]
  │    ├─ hindsight_retain  → store facts to memory
  │    ├─ hindsight_recall  → search memory for relevant info
  │    └─ hindsight_reflect → synthesize an answer from all memories
  │
  └─ instructions=[memory_instructions(...)]
       └─ auto-recalls relevant memories into the system prompt
```

Two integration points, both optional:

1. **Tools** let the agent explicitly store and retrieve memories during a conversation.
2. **Instructions** silently inject relevant memories before the agent even starts thinking.

The tools are async functions that call Hindsight's API directly. No thread-pool hacks needed -- Pydantic AI is async-native, so the closures use `aretain()`, `arecall()`, and `areflect()`.

---

## Step 1 -- Start Hindsight

```bash
pip install hindsight-all
```

```bash
export HINDSIGHT_API_LLM_API_KEY=YOUR_OPENAI_KEY
hindsight-api
```

Runs locally at `http://localhost:8888`. Embedded Postgres, local embeddings, local reranking. No external services needed.

> **Note:** You can also use [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) and skip the self-hosted setup entirely.

---

## Step 2 -- Install the Integration

```bash
pip install hindsight-pydantic-ai
```

You also need a model provider. For OpenAI:

```bash
pip install "pydantic-ai-slim[openai]"
```

---

## Step 3 -- Add Memory Tools to Your Agent

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

That's it. The agent now has three tools:

- `hindsight_retain(content)` -- stores information to long-term memory
- `hindsight_recall(query)` -- searches memory and returns matching facts
- `hindsight_reflect(query)` -- synthesizes a reasoned answer from all relevant memories

The agent decides when to use them based on the conversation.

---

## Step 4 -- Try It

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

First run: the agent stores the preferences via `hindsight_retain`.

Second run: the agent calls `hindsight_recall` to find relevant context, then gives advice grounded in what it knows about you -- functional patterns, Python, data pipelines.

This works across runs, across days, across process restarts. The memories live in Hindsight, not in the agent's context window.

---

## Step 5 -- Auto-Inject Memories with Instructions

Tools require the agent to decide to search memory. Sometimes you want memories injected automatically, before the agent starts responding.

Pydantic AI's `instructions` parameter supports async callables that run on every `agent.run()`. This is a perfect fit:

```python
from hindsight_pydantic_ai import create_hindsight_tools, memory_instructions

agent = Agent(
    "openai:gpt-4o-mini",
    tools=create_hindsight_tools(client=client, bank_id="user-123"),
    instructions=[memory_instructions(client=client, bank_id="user-123")],
)
```

Now on every run, `memory_instructions` calls Hindsight's recall API and injects relevant memories into the system prompt. The agent starts every conversation with context about the user -- without using a tool call.

You can customize the query, result count, and prefix:

```python
memory_instructions(
    client=client,
    bank_id="user-123",
    query="user preferences, history, and context",
    max_results=10,
    prefix="Here is what you know about this user:\n",
)
```

If recall fails or returns nothing, the instructions function returns an empty string. It never blocks the agent.

---

## Selecting Which Tools to Include

You don't always need all three tools. `create_hindsight_tools` lets you pick:

```python
# Read-only agent -- can search memory but not write to it
tools = create_hindsight_tools(
    client=client,
    bank_id="user-123",
    include_retain=False,
    include_recall=True,
    include_reflect=True,
)

# Write-only agent -- stores data but doesn't query
tools = create_hindsight_tools(
    client=client,
    bank_id="user-123",
    include_retain=True,
    include_recall=False,
    include_reflect=False,
)
```

---

## Global Configuration

If you have multiple agents sharing the same Hindsight instance, use the global config instead of passing `client` everywhere:

```python
from hindsight_pydantic_ai import configure, create_hindsight_tools

configure(hindsight_api_url="http://localhost:8888", api_key="YOUR_KEY")

# No client needed -- tools use the global config
agent1_tools = create_hindsight_tools(bank_id="agent-1")
agent2_tools = create_hindsight_tools(bank_id="agent-2")
```

Explicit `client=` always takes precedence over the global config.

---

## Full Working Example

Save this as `memory_agent.py`:

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

Run it again. The agent remembers everything from the first session.

---

## Pitfalls and Edge Cases

**1. Bank ID collisions.** Each `bank_id` is a separate memory store. If two unrelated agents share a bank, their memories merge. Use unique bank IDs per user, per agent, or per project.

**2. Instruction latency.** `memory_instructions` makes a recall API call on every `agent.run()`. For latency-sensitive applications, use `budget="low"` and a small `max_results`. Or skip instructions entirely and rely on the agent to call the recall tool when needed.

**3. Duplicate memories.** If the agent stores the same information multiple times, Hindsight deduplicates at the fact level. But it's still better to give the agent clear guidance in the system prompt about when to store vs. when to skip.

**4. `acreate_bank` inside async.** The sync `create_bank()` method doesn't work inside `asyncio.run()` because it tries to create a nested event loop. Always use `await client.acreate_bank()` in async code.

---

## Tradeoffs and Alternatives

**When to use this:**

- Agents that interact with the same user/context across multiple sessions
- Personal assistants, support bots, research agents that accumulate knowledge
- Any Pydantic AI agent where "remembering" improves quality over time

**When not to use this:**

- One-shot agents that never run again
- Stateless API handlers where each request is independent
- Agents where you want full control over what goes into the prompt (use the Hindsight client directly instead of the integration)

**Alternatives:**

- **Manual message_history management**: Works for short-term context, but doesn't scale across sessions and doesn't extract structured facts.
- **Custom vector store + RAG**: You manage the embeddings, chunking, and retrieval yourself. Hindsight handles that plus entity graphs, temporal ranking, and synthesis.
- **Mem0 or similar**: Another external memory option. Hindsight differs with multi-strategy retrieval (semantic + BM25 + graph + temporal), structured fact extraction, and a synthesis engine (reflect).

---

## Recap

- `hindsight-pydantic-ai` gives Pydantic AI agents persistent, structured memory
- `create_hindsight_tools()` returns async tools that the agent calls to store and retrieve knowledge
- `memory_instructions()` auto-injects relevant memories on every run -- no tool call needed
- No thread-pool hacks or compatibility layers -- Pydantic AI is async, Hindsight's client is async
- Memories survive process restarts, build a knowledge graph, and compound over time

The integration is minimal by design. Two functions, no subclassing, no changes to your deps type. Just tools and instructions.

---

## Next Steps

- **Try it locally**: `pip install hindsight-all hindsight-pydantic-ai "pydantic-ai-slim[openai]"` and run the example above
- **Use Hindsight Cloud**: Skip self-hosting with a [free account](https://ui.hindsight.vectorize.io/signup)
- **Tag memories for scoping**: Use `tags` on retain and `recall_tags` on search to partition memories by project, environment, or topic
- **Combine tools and instructions**: Use `memory_instructions` for automatic context and tools for explicit store/retrieve during conversations
- **Inspect the knowledge graph**: Run the Hindsight control plane or use the cloud dashboard to browse extracted facts, entities, and relationships
