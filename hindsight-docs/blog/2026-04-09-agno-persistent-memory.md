---
title: "Agno Persistent Memory: Long-Term Memory for Agno Agents"
authors: [benfrank241]
date: 2026-04-09T09:00
tags: [agno, memory, persistent-memory, python, agents, hindsight, toolkit]
image: /img/blog/agno-persistent-memory.png
description: "Add persistent memory to Agno agents with Hindsight. HindsightTools plugs into Agno's native Toolkit pattern — retain, recall, and reflect across sessions with three lines of setup."
hide_table_of_contents: true
---

If you have built an agent with [Agno](https://docs.agno.com/), you know the framework handles multi-modal inputs, team coordination, and structured outputs well. What it does not handle is memory between runs. Every `agent.run()` starts with an empty context window. The agent has no idea what the user said last week, what preferences they shared, or what it has already researched.

Adding **Agno persistent memory** does not require building a custom RAG pipeline or maintaining your own vector store. The `hindsight-agno` package extends Agno's native `Toolkit` pattern to give your agents long-term memory — retain facts, recall them by semantic search, and synthesize coherent answers from accumulated knowledge. This guide covers the full setup, from installation through production patterns.

<!-- truncate -->

## TL;DR

- Agno has no built-in persistent memory. Every `agent.run()` starts from zero.
- `hindsight-agno` adds `HindsightTools` — a native Agno `Toolkit` with retain, recall, and reflect tools.
- Three lines of setup: install the package, create `HindsightTools`, pass to `Agent`.
- `memory_instructions()` pre-loads relevant memories into `Agent(instructions=[...])` on every run, so the agent starts each conversation with context.
- Per-user bank isolation works automatically via `user_id` or a custom `bank_resolver`.
- [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) skips local setup entirely — two lines of config and you are running.

---

## The Problem: Agno Has No Persistent Memory

[Agno](https://docs.agno.com/) is a capable agent framework. Its `Toolkit` pattern makes it easy to add tools, the team coordination primitives handle multi-agent workflows, and structured outputs and streaming work well out of the box. But Agno ships with no memory layer.

Every `agent.run()` starts from nothing. The agent does not know what the user mentioned in the previous session. It does not know their preferences, recurring questions, or what the agent itself already researched. If a user tells your agent something important today, that fact is gone tomorrow.

You can pass `messages` to continue a conversation within a single run. But that is session history, not memory. Session history does not extract structured facts. It grows linearly with each turn. It does not generalize or deduplicate. And it disappears entirely when the process exits.

Real agent memory is different:

- Extracting discrete facts from conversations and storing them durably
- Building a knowledge graph of entities and relationships
- Retrieving relevant context across days, weeks, and months via semantic search
- Synthesizing coherent answers from scattered, accumulated knowledge

That is what [Hindsight](https://hindsight.vectorize.io/) provides. The `hindsight-agno` package wires it directly into Agno's `Toolkit` system, so you do not build any of this yourself.

---

## How Agno Persistent Memory Works with Hindsight

The `hindsight-agno` integration connects to Agno at two points: tools and instructions.

```
Agno Agent
  |-- tools=[HindsightTools(...)]
  |     |-- retain_memory  -> store facts to long-term memory
  |     |-- recall_memory  -> search memory for relevant facts
  |     |-- reflect_on_memory -> synthesize an answer from memories
  |
  |-- instructions=[memory_instructions(...)]
        |-- auto-recalls relevant memories into the system prompt
```

**Tools** let the agent explicitly store and retrieve memories during a conversation. The agent decides when to call each tool based on context. **Instructions** inject relevant memories into the system prompt before the agent starts responding — no tool call required.

`HindsightTools` extends Agno's `Toolkit` base class directly, the same pattern used by `Mem0Tools` in the Agno ecosystem. This means it integrates natively: no wrappers, no compatibility shims, no changes to how you build your agent. You add it to `tools=[...]` exactly like any other Agno toolkit.

---

## Setting Up Agno Persistent Memory

### Step 1: Install Hindsight

Start with the Hindsight server:

```bash
pip install hindsight-all
```

```bash
export HINDSIGHT_API_LLM_API_KEY=YOUR_OPENAI_KEY
hindsight-api
```

This starts Hindsight locally at `http://localhost:8888` with embedded Postgres, local embeddings, and local reranking. The only external dependency is an LLM API key for entity extraction.

> **Prefer not to self-host?** Use [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) and skip this step entirely. The cloud version provides the same API — just configure `hindsight_api_url` and `api_key` as shown below.

### Step 2: Install the Agno Memory Integration

```bash
pip install hindsight-agno agno
```

### Step 3: Add HindsightTools to Your Agent

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from hindsight_agno import HindsightTools

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[HindsightTools(
        bank_id="user-123",
        hindsight_api_url="http://localhost:8888",
    )],
)
```

That is the complete Agno persistent memory setup. The agent now has three tools it can call:

- **`retain_memory`** — Store information to long-term memory
- **`recall_memory`** — Search long-term memory for relevant facts
- **`reflect_on_memory`** — Synthesize a reasoned answer from accumulated memories

The agent calls these tools autonomously based on the conversation. You do not invoke them directly.

### Step 4: Test Cross-Session Memory

Run two separate conversations to verify that memory persists across sessions:

```python
# First session — agent stores context
agent.print_response(
    "Remember that I prefer functional programming patterns "
    "and I am building a data pipeline in Python."
)

# Later session — agent recalls context
agent.print_response("What approach should I take for error handling?")
```

In the first run, the agent calls `retain_memory` to store the preferences. In the second run, it calls `recall_memory` to retrieve relevant facts before responding. The advice reflects what it knows: functional patterns, Python, data pipelines.

Restart the process and run the second call again. The agent still knows. Agno persistent memory is stored in Hindsight, not in the agent's context window.

### Step 5: Auto-Inject Memories with Memory Instructions

The tools above require the agent to decide when to search memory. For cases where you want Agno persistent memory injected automatically at the start of every run, use `memory_instructions`:

```python
from hindsight_agno import HindsightTools, memory_instructions

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[HindsightTools(
        bank_id="user-123",
        hindsight_api_url="http://localhost:8888",
    )],
    instructions=[memory_instructions(
        bank_id="user-123",
        hindsight_api_url="http://localhost:8888",
    )],
)
```

On every `agent.run()`, `memory_instructions` calls Hindsight's recall API and injects relevant memories into the system prompt. The agent starts each conversation with context — no tool call needed, no extra prompt from you.

You can customize the recall query, result count, and prefix:

```python
memory_instructions(
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
    query="user preferences, history, and context",
    max_results=10,
    prefix="Here is what you know about this user:\n",
)
```

If recall returns nothing or fails, `memory_instructions` returns an empty string. The agent runs normally.

---

## Advanced Agno Memory Configuration

### Per-User Bank Isolation

For agents that serve multiple users, `HindsightTools` resolves the bank ID dynamically so each user gets isolated Agno persistent memory. Resolution order:

1. `bank_resolver` — A callable `(RunContext) -> str` for custom logic
2. `bank_id` — A static bank ID passed to the constructor
3. `RunContext.user_id` — Automatic per-user banks

Using `user_id` from `RunContext`:

```python
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[HindsightTools(hindsight_api_url="http://localhost:8888")],
    user_id="user-123",  # Used as bank_id automatically
)
```

Using a custom resolver for team-based banks:

```python
def resolve_bank(ctx):
    return f"team-{ctx.user_id.split('-')[0]}"

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[HindsightTools(
        bank_resolver=resolve_bank,
        hindsight_api_url="http://localhost:8888",
    )],
)
```

### Selecting Which Memory Tools to Include

You do not always need all three tools. Use the `enable_*` flags to include only what your agent needs:

```python
# Read-only agent: recall and reflect, no storing
tools = HindsightTools(
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
    enable_retain=False,
    enable_recall=True,
    enable_reflect=True,
)

# Ingestion agent: stores data, does not search
tools = HindsightTools(
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
    enable_retain=True,
    enable_recall=False,
    enable_reflect=False,
)
```

This is useful in multi-agent setups: one agent accumulates knowledge, another answers questions from it. Splitting read and write access prevents unintended memory writes from agents that should only consume context.

### Global Configuration

If you have multiple Agno agents sharing the same Hindsight instance, configure once globally instead of repeating connection details on each toolkit:

```python
from hindsight_agno import configure, HindsightTools

configure(
    hindsight_api_url="http://localhost:8888",
    api_key="your-api-key",
    budget="mid",
    max_tokens=4096,
    tags=["env:prod"],
)

# No connection details needed per toolkit
agent1_tools = HindsightTools(bank_id="user-alice")
agent2_tools = HindsightTools(bank_id="user-bob")
```

An explicit `hindsight_api_url=` on `HindsightTools()` always takes priority over the global config.

### Hindsight Cloud Configuration

To use Hindsight Cloud instead of a local server:

```python
from hindsight_agno import configure

configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_your_token",
)
```

Or set the environment variable and skip the `configure()` call:

```bash
export HINDSIGHT_API_KEY=hsk_your_token
```

No daemon to manage. No local Postgres. The cloud server handles extraction, indexing, and retrieval.

---

## Full Working Example

Save this as `memory_agent.py` and run it to see Agno persistent memory in action:

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from hindsight_agno import HindsightTools, memory_instructions

BANK_ID = "demo-user"
HINDSIGHT_URL = "http://localhost:8888"

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[HindsightTools(
        bank_id=BANK_ID,
        hindsight_api_url=HINDSIGHT_URL,
    )],
    instructions=[memory_instructions(
        bank_id=BANK_ID,
        hindsight_api_url=HINDSIGHT_URL,
    )],
)

print("--- Run 1: Teaching the agent ---")
agent.print_response(
    "Remember: I am a backend engineer. I use Python and Rust. "
    "I prefer small, composable libraries over large frameworks."
)

print("\n--- Run 2: Agent recalls context ---")
agent.print_response("Recommend a web framework for my next project.")

print("\n--- Run 3: Agent synthesizes ---")
agent.print_response("What do you know about my engineering philosophy?")
```

Run it:

```bash
export OPENAI_API_KEY=YOUR_KEY
python memory_agent.py
```

Run it again. The agent remembers everything from the first execution because Agno persistent memory is stored in Hindsight, not in the process.

---

## Pitfalls

**Bank ID collisions.** Each `bank_id` is a separate memory store. If two unrelated agents share a bank, their memories merge in unexpected ways. Use unique bank IDs per user, per agent, or per project.

**No bank created yet.** Hindsight creates banks automatically on first retain. No manual setup is required.

**Memory instruction latency.** `memory_instructions` makes a recall API call on every `agent.run()`. For latency-sensitive applications, use `budget="low"` and a small `max_results`. Alternatively, skip automatic injection and rely on the agent to call the recall tool only when needed. In practice, recall adds 50–200ms depending on bank size and network conditions.

**Duplicate memories.** Hindsight deduplicates at the fact level, but it helps to give the agent guidance in the system prompt about when to store new facts versus when to skip.

---

## Recap

| | Agno default | With Hindsight |
|---|---|---|
| Memory across sessions | None | Automatic |
| Memory setup | None | `pip install hindsight-agno` |
| Recall mechanism | Not available | Semantic search via tool or instructions |
| Per-user isolation | No | Via `user_id` or `bank_resolver` |
| Hosting | N/A | Local or [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) |

`HindsightTools` follows Agno's native `Toolkit` pattern, the same interface used by `Mem0Tools`. There is no subclassing, no changes to your agent structure, and no new abstractions to learn. You add it to `tools=[...]` and it works.

---

## Next Steps

- **[Hindsight Cloud](https://ui.hindsight.vectorize.io/signup)** — Skip local setup with a free account
- **Try it locally**: `pip install hindsight-all hindsight-agno agno` and run the example above
- **Config reference**: [Agno integration docs](/sdks/integrations/agno)
- **Explore other integrations**: Add memory to [Pydantic AI agents](/blog/2026/03/09/pydantic-ai-persistent-memory), [LangGraph workflows](/blog/2026/02/25/langgraph-long-term-memory), or any framework via [MCP](/blog/2026/03/04/mcp-agent-memory)
- **Inspect the knowledge graph**: Use the [Hindsight Cloud dashboard](https://ui.hindsight.vectorize.io/signup) to browse extracted facts, entities, and relationships
