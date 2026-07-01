---
sidebar_position: 10
---

# OpenAI Agents SDK with Long-Term Memory


:::tip Run this notebook
This recipe is available as an interactive Jupyter notebook.
[**Open in GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/blob/main/notebooks/10-openai-agents-memory.ipynb)
:::


Build an OpenAI agent that remembers user preferences and past interactions across conversations using Hindsight memory.

The agent gets **retain**, **recall**, and **reflect** tools and autonomously decides when to store or retrieve memories.

## Prerequisites

- Python 3.10+
- A [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) account **or** a self-hosted Hindsight instance
- An OpenAI API key

### Option A: Hindsight Cloud

Sign up at [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io/signup) and grab your API key from the dashboard.

### Option B: Self-Hosted (Docker)

```bash
export OPENAI_API_KEY=your-key

docker run --rm -it --pull always -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  -e HINDSIGHT_API_LLM_MODEL=gpt-4o-mini \
  -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

## Installation

```python
%pip install -q hindsight-openai-agents hindsight-client openai-agents python-dotenv
```

## Setup

Configure the Hindsight client. Set `HINDSIGHT_API_URL` and `HINDSIGHT_API_KEY` in your `.env` file or environment:

| | Hindsight Cloud | Self-Hosted |
|---|---|---|
| `HINDSIGHT_API_URL` | `https://api.hindsight.vectorize.io` | `http://localhost:8888` |
| `HINDSIGHT_API_KEY` | Your cloud API key | *(not required for local)* |

> **Note:** Self-hosted Hindsight does not require an API key by default. If you've configured authentication on your instance, provide the key accordingly.

> **Note:** You also need `OPENAI_API_KEY` set in your environment or `.env` file for the OpenAI Agents SDK.

```python
import os
from dotenv import load_dotenv

load_dotenv()

HINDSIGHT_API_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")
HINDSIGHT_API_KEY = os.getenv("HINDSIGHT_API_KEY")  # Required for Hindsight Cloud

from hindsight_client import Hindsight

client_kwargs = {"base_url": HINDSIGHT_API_URL}
if HINDSIGHT_API_KEY:
    client_kwargs["api_key"] = HINDSIGHT_API_KEY

client = Hindsight(**client_kwargs)
```

## Create the Memory Bank

Create an isolated memory bank for this demo. The `try/except` block ensures idempotency if the notebook is re-run.

```python
BANK_ID = "openai-agents-cookbook-demo"

# Clean up any leftover bank from a previous run, then create fresh
try:
    await client.adelete_bank(BANK_ID)
except Exception:
    pass

await client.acreate_bank(BANK_ID, name="OpenAI Agents Cookbook Demo")
print(f"Bank created: {BANK_ID}")
```

## Create the Agent

There are two patterns for giving an agent memory. We'll start with **Pattern 1: Tools only**.

Set up the OpenAI Agents SDK `Agent` with Hindsight memory tools. The `create_hindsight_tools()` factory returns retain, recall, and reflect tools that the agent can call autonomously.

> **Important:** Each call to `Runner.run()` is completely stateless — the SDK does not carry conversation history between runs (unless you use a `Session` object). This means on every new `Runner.run()` call, the agent has **zero context** about previous interactions. This is exactly why long-term memory matters: Hindsight gives the agent persistent knowledge that survives across independent runs.

```python
from agents import Agent, Runner
from hindsight_openai_agents import create_hindsight_tools, memory_instructions

# Pattern 1: Tools only — agent decides when to retain, recall, and reflect
tools = create_hindsight_tools(client=client, bank_id=BANK_ID)
print(f"Tools created: {[t.name for t in tools]}")

agent = Agent(
    name="memory_assistant",
    model="gpt-4o-mini",
    instructions=(
        "You are a helpful assistant with long-term memory. "
        "Use hindsight_retain to store important facts about the user. "
        "Use hindsight_recall to search your memory before answering questions. "
        "Use hindsight_reflect for thoughtful summaries that synthesize what you know."
    ),
    tools=tools,
)
```

## Conversation 1: Store Preferences

The agent should recognize important facts and store them using `hindsight_retain`.

```python
result = await Runner.run(
    agent,
    "Hi! I'm Alice. I'm a data scientist who works with Python and SQL. "
    "I prefer dark mode and use VS Code. Please remember all of this.",
)
print(f"Response: {result.final_output}")
```

```python
import asyncio

# Hindsight processes retained content asynchronously (extracting facts, entities, embeddings).
# The sleep gives the server time to finish before we recall.
await asyncio.sleep(3)
```

## Conversation 2: Recall Preferences

This is a brand-new `Runner.run()` call — the agent has no conversation history from Conversation 1. The only way it can answer is by calling `hindsight_recall` to search its long-term memory.

```python
result = await Runner.run(
    agent,
    "What IDE do I use? And what's my job?",
)
print(f"Response: {result.final_output}")
```

## Conversation 3: Reflect on Knowledge

The agent uses `hindsight_reflect` to synthesize a thoughtful answer from accumulated memories.

```python
result = await Runner.run(
    agent,
    "Based on everything you know about me, what tools and setup "
    "would you recommend for a new machine learning project?",
)
print(f"Response: {result.final_output}")
```

## Pattern 2: Auto-Inject Memories with `memory_instructions()`

The tools-only pattern above works great for open-ended conversations where the agent decides when to recall. But sometimes you **always** want relevant context injected into the system prompt automatically.

`memory_instructions()` accepts a `base_instructions` string (your static system prompt) and returns a single callable that combines that text with recalled memories on every agent turn — no explicit `hindsight_recall` tool call needed.

### When to use which pattern

| | **Pattern 1: Tools only** | **Pattern 2: Auto-inject + tools** |
|---|---|---|
| **How recall works** | Agent calls `hindsight_recall` explicitly | Memories auto-injected into system prompt |
| **Best for** | Open-ended conversations where recall isn't always needed | Scenarios where you always want prior context |
| **Agent tools** | retain + recall + reflect | retain + reflect (recall handled automatically) |
| **Trade-off** | Agent may forget to recall | Every turn pays the recall latency cost |

```python
# Pattern 2: Auto-inject memories into system prompt
agent_with_auto_memory = Agent(
    name="memory_assistant_auto",
    model="gpt-4o-mini",
    instructions=memory_instructions(
        client=client,
        bank_id=BANK_ID,
        base_instructions=(
            "You are a helpful assistant with long-term memory. "
            "Use hindsight_retain to store important facts about the user."
        ),
    ),
    tools=create_hindsight_tools(
        client=client,
        bank_id=BANK_ID,
        include_recall=False,  # recall handled by memory_instructions
    ),
)
```

### Demo: Auto-Memory Agent

The auto-memory agent already knows about Alice from the memories stored earlier — without needing to call `hindsight_recall` explicitly. The memories are injected into the system prompt automatically.

```python
# The auto-memory agent answers using injected context — no recall tool call needed
result = await Runner.run(
    agent_with_auto_memory,
    "What programming languages and tools do I use?",
)
print(f"Response: {result.final_output}")
```

## Cleanup

Delete the demo bank to free resources.

```python
await client.adelete_bank(BANK_ID)
await client.aclose()
print("Cleanup complete.")
```

## Key Takeaways

- **`Runner.run()` is stateless**: Each call starts fresh with no conversation history. Hindsight gives the agent persistent memory that survives across independent runs — this is different from the SDK's `Session` object, which tracks within-conversation history but resets between sessions.
- **Two patterns for recall**: Use tools-only (`include_recall=True`) when the agent should decide when to search memory, or use `memory_instructions()` to auto-inject relevant memories into every turn's system prompt.
- **Autonomous tool use**: The agent decides when to retain, recall, or reflect — no manual orchestration needed.
- **Per-user memory isolation**: Use `bank_id=f"user-{user_id}"` to give each user their own memory bank. You can also pass user-specific context via `Runner.run(..., context=my_context)` for request-scoped state.
- **Idempotent setup**: The `try/except` delete-then-create pattern makes notebooks safely re-runnable.
- **Works with any OpenAI model**: Set `model="gpt-4o"` or any other supported model on the `Agent`.