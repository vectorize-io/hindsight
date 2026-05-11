---
title: "Agno Persistent Memory: Long-Term Memory for Agno Agents"
authors: [benfrank241]
date: 2026-04-09T09:00
tags: [agno, integrations, agents, memory, persistent-memory, python, tutorial]
description: "Agno agents start each run fresh. hindsight-agno adds HindsightTools and memory_instructions so they can retain, recall, and reflect across sessions."
image: /img/blog/agno-persistent-memory.png
hide_table_of_contents: true
---

![Agno Persistent Memory: Long-Term Memory for Agno Agents](/img/blog/agno-persistent-memory.png)

If you have built an agent with [Agno](https://docs.agno.com/), you know the framework handles multimodal inputs, team coordination, and structured outputs well. What it does not handle is memory between runs. Every `agent.run()` starts with an empty context window. The agent has no idea what the user said last week, what preferences they shared, or what it has already researched.

Adding persistent memory does not require building a custom RAG pipeline or maintaining your own vector store. The `hindsight-agno` package extends Agno's native `Toolkit` pattern to give your agents long-term memory: retain facts, recall them by semantic search, and synthesize coherent answers from accumulated knowledge.

<!-- truncate -->

## TL;DR

- Agno has no built-in persistent memory. Every `agent.run()` starts from zero.
- `hindsight-agno` adds `HindsightTools`, a native Agno `Toolkit` with retain, recall, and reflect tools.
- Three lines of setup: install the package, create `HindsightTools`, pass it to `Agent`.
- `memory_instructions()` preloads relevant memories into `Agent(instructions=[...])` on every run, so the agent starts each conversation with context.
- Per-user bank isolation works automatically via `user_id` or a custom `bank_resolver`.
- [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) skips local setup entirely. Two lines of config and you are running.

## The Problem: Agno Has No Persistent Memory

[Agno](https://docs.agno.com/) is a capable agent framework. Its `Toolkit` pattern makes it easy to add tools, the team coordination primitives handle multi-agent workflows, and structured outputs and streaming work well out of the box. But Agno ships with no memory layer.

Every `agent.run()` starts from nothing. The agent does not know what the user mentioned in the previous session. It does not know their preferences, recurring questions, or what the agent itself already researched. If a user tells your agent something important today, that fact is gone tomorrow.

You can pass `messages` to continue a conversation within a single run. But that is session history, not memory. Session history does not extract structured facts, grows linearly with each turn, does not generalize or deduplicate, and disappears entirely when the process exits.

Real agent memory is different:

- Extracting discrete facts from conversations and storing them durably
- Building a knowledge graph of entities and relationships
- Retrieving relevant context across days, weeks, and months via semantic search
- Synthesizing coherent answers from scattered, accumulated knowledge

That is what [Hindsight](https://hindsight.vectorize.io/) provides. The `hindsight-agno` package wires it directly into Agno's `Toolkit` system, so you do not build any of this yourself.

## How Agno Persistent Memory Works with Hindsight

The `hindsight-agno` integration connects to Agno at two points: tools and instructions.

```text
Agno Agent
  |-- tools=[HindsightTools(...)]
  |     |-- retain_memory      -> store facts to long-term memory
  |     |-- recall_memory      -> search memory for relevant facts
  |     |-- reflect_on_memory  -> synthesize an answer from memories
  |
  |-- instructions=[memory_instructions(...)]
        |-- auto-recalls relevant memories into the system prompt
```

*Tools* let the agent explicitly store and retrieve memories during a conversation. *Instructions* inject relevant memories into the system prompt before the agent starts responding. No tool call required.

`HindsightTools` extends Agno's `Toolkit` base class directly, the same pattern used by `Mem0Tools` in the Agno ecosystem. You add it to `tools=[...]` exactly like any other Agno toolkit.

It helps to understand what each tool does and when the agent uses it:

- **`retain_memory`** is called when the agent encounters something worth keeping: a stated preference, a decision, a fact about the user or project.
- **`recall_memory`** returns a list of matching facts from past sessions. The agent uses it when it needs to look something up.
- **`reflect_on_memory`** synthesizes those facts into a coherent answer. Where `recall_memory` returns a list, `reflect_on_memory` reasons across all relevant memories to produce a single response. It is better for questions like "what do you know about my engineering style?" than raw fact retrieval.

## Setting Up Agno Persistent Memory

### Step 1: Start Hindsight

```bash
pip install hindsight-all
```

```bash
export HINDSIGHT_API_LLM_API_KEY=YOUR_OPENAI_KEY
hindsight-api
```

This starts Hindsight locally at `http://localhost:8888`. The only external dependency is an LLM API key for entity extraction.

Prefer not to self-host? Use [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) and skip this step entirely.

### Step 2: Install the Agno integration

```bash
pip install hindsight-agno agno
```

### Step 3: Add `HindsightTools` to your agent

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

That is the complete setup. The agent now has three tools:

- **`retain_memory`**: store information to long-term memory
- **`recall_memory`**: search long-term memory for relevant facts
- **`reflect_on_memory`**: synthesize a reasoned answer from accumulated memories

### Step 4: Test cross-session memory

```python
# First session: agent stores context
agent.print_response(
    "Remember that I prefer functional programming patterns "
    "and I am building a data pipeline in Python."
)

# Later session: agent recalls context
agent.print_response("What approach should I take for error handling?")
```

Restart the process and run the second call again. The agent still knows. Agno persistent memory is stored in Hindsight, not in the agent's context window.

### Step 5: Auto-inject memories with `memory_instructions`

For cases where you want memory injected automatically at the start of every run, use `memory_instructions`:

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

On every `agent.run()`, `memory_instructions` calls Hindsight's recall API and injects relevant memories into the system prompt. The agent starts each conversation with context. No tool call needed.

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

## Advanced Agno Memory Configuration

### Per-user bank isolation

For agents that serve multiple users, `HindsightTools` resolves the bank ID dynamically. Resolution order:

1. `bank_resolver`, a callable `(RunContext) -> str` for custom logic
2. `bank_id`, a static bank ID passed to the constructor
3. `RunContext.user_id`, for automatic per-user banks

```python
# Per-user banks from RunContext
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[HindsightTools(hindsight_api_url="http://localhost:8888")],
    user_id="user-123",
)

# Custom resolver for team-based banks
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

### Selecting which memory tools to include

```python
# Read-only agent: recall and reflect, no storing
tools = HindsightTools(
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
    enable_retain=False,
    enable_recall=True,
    enable_reflect=True,
)
```

Useful in multi-agent setups: one agent accumulates knowledge, another answers questions from it.

### Global configuration

```python
from hindsight_agno import configure, HindsightTools

configure(
    hindsight_api_url="http://localhost:8888",
    api_key="your-api-key",
    budget="mid",
    max_tokens=4096,
)

agent1_tools = HindsightTools(bank_id="user-alice")
agent2_tools = HindsightTools(bank_id="user-bob")
```

### Hindsight Cloud configuration

```python
from hindsight_agno import configure

configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_your_token",
)
```

No daemon to manage. No local Postgres. The cloud server handles extraction, indexing, and retrieval.

## Full Working Example

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

Run it twice. The agent remembers everything from the first execution.

## When Not to Use Agno Persistent Memory

Skip the memory layer for one-shot agents that never interact with the same user twice, stateless API handlers where each request is fully independent, or agents where you want complete control over what enters the prompt. In those cases, use the Hindsight Python client directly instead of the Agno integration.

## How It Compares to Alternatives

| Approach | Strengths | Weaknesses | Best For |
|---|---|---|---|
| **Hindsight + Agno** | Multi-strategy retrieval (semantic + BM25 + graph + temporal), structured fact extraction, synthesis | Requires Hindsight server or cloud | Multi-session agents needing deep memory |
| **Manual messages** | Built into Agno, no dependencies | Not persistent, grows to context window limit | Short single-session conversations |

## Pitfalls

**Bank ID collisions.** Each `bank_id` is a separate memory store. Use unique bank IDs per user, per agent, or per project to prevent memory bleed between unrelated agents.

**Memory instruction latency.** `memory_instructions` makes a recall API call on every `agent.run()`. For latency-sensitive applications, use `budget="low"` and a small `max_results`. Recall adds roughly 50 to 200ms depending on bank size and network conditions.

**Duplicate memories.** Hindsight deduplicates at the fact level, but giving the agent guidance in the system prompt about when to store new facts helps.

## Recap

| | Agno default | With Hindsight |
|---|---|---|
| Memory across sessions | None | Automatic |
| Memory setup | None | `pip install hindsight-agno` |
| Recall mechanism | Not available | Semantic search via tool or instructions |
| Per-user isolation | No | Via `user_id` or `bank_resolver` |
| Hosting | N/A | Local or [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) |

## Next Steps

- **Hindsight Cloud**: [Sign up free](https://ui.hindsight.vectorize.io/signup)
- **Try it locally**: `pip install hindsight-all hindsight-agno agno` and run the example above
- **Config reference**: [Agno integration docs](/sdks/integrations/agno)
- **Explore other integrations**: [Pydantic AI](/sdks/integrations/pydantic-ai), [LangGraph](/sdks/integrations/langgraph), [CrewAI](/sdks/integrations/crewai)
- **Inspect the knowledge graph**: Use the [Hindsight Cloud dashboard](https://ui.hindsight.vectorize.io/signup) to browse extracted facts, entities, and relationships
