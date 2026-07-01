---
sidebar_position: 9
---

# AutoGen Assistant Agent with Long-Term Memory


:::tip Run this notebook
This recipe is available as an interactive Jupyter notebook.
[**Open in GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/blob/main/notebooks/09-autogen-assistant-agent.ipynb)
:::


Build an AutoGen assistant agent that remembers user preferences and past interactions across conversations using Hindsight memory.

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
%pip install -q hindsight-autogen hindsight-client autogen-agentchat python-dotenv "autogen-ext[openai]"
```

## Setup

Configure the Hindsight client. Set `HINDSIGHT_API_URL` and `HINDSIGHT_API_KEY` in your `.env` file or environment:

| | Hindsight Cloud | Self-Hosted |
|---|---|---|
| `HINDSIGHT_API_URL` | `https://api.hindsight.vectorize.io` | `http://localhost:8888` |
| `HINDSIGHT_API_KEY` | Your cloud API key | *(not required for local)* |

> **Note:** Self-hosted Hindsight does not require an API key by default. If you've configured authentication on your instance, provide the key accordingly.

> **Note:** You also need `OPENAI_API_KEY` set in your environment or `.env` file for the AutoGen model.

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
BANK_ID = "autogen-cookbook-demo"

# Clean up any leftover bank from a previous run, then create fresh
try:
    await client.adelete_bank(BANK_ID)
except Exception:
    pass

await client.acreate_bank(BANK_ID, name="AutoGen Cookbook Demo")
print(f"Bank created: {BANK_ID}")
```

## Create the Agent

Set up the AutoGen `AssistantAgent` with Hindsight memory tools. The `create_hindsight_tools()` factory returns retain, recall, and reflect tools that the agent can call autonomously.

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from hindsight_autogen import create_hindsight_tools

# Create Hindsight memory tools
tools = create_hindsight_tools(client=client, bank_id=BANK_ID)
print(f"Tools created: {[t.name for t in tools]}")

# Create the assistant agent with memory tools
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

agent = AssistantAgent(
    name="memory_assistant",
    model_client=model_client,
    tools=tools,
    system_message=(
        "You are a helpful assistant with long-term memory. "
        "Use hindsight_retain to store important facts about the user. "
        "Use hindsight_recall to search your memory before answering questions. "
        "Use hindsight_reflect for thoughtful summaries that synthesize what you know."
    ),
    reflect_on_tool_use=True,
)
```

## Conversation 1: Store Preferences

The agent should recognize important facts and store them using `hindsight_retain`.

```python
from autogen_agentchat.messages import TextMessage

result = await Console(
    agent.run_stream(
        task=TextMessage(
            content=(
                "Hi! I'm Alice. I'm a data scientist who works with Python and SQL. "
                "I prefer dark mode and use VS Code. Please remember all of this."
            ),
            source="user",
        )
    )
)
print(f"\nFinal response: {result.messages[-1].content}")
```

```python
import asyncio

# Hindsight processes retained content asynchronously (extracting facts, entities, embeddings).
# The sleep gives the server time to finish before we recall.
await asyncio.sleep(3)
```

## Conversation 2: Recall Preferences

The agent should search memory to answer questions about previously stored information.

```python
result = await Console(
    agent.run_stream(
        task=TextMessage(
            content="What IDE do I use? And what's my job?",
            source="user",
        )
    )
)
print(f"\nFinal response: {result.messages[-1].content}")
```

## Conversation 3: Reflect on Knowledge

The agent uses `hindsight_reflect` to synthesize a thoughtful answer from accumulated memories.

```python
result = await Console(
    agent.run_stream(
        task=TextMessage(
            content=(
                "Based on everything you know about me, what tools and setup "
                "would you recommend for a new machine learning project?"
            ),
            source="user",
        )
    )
)
print(f"\nFinal response: {result.messages[-1].content}")
```

## Cleanup

Delete the demo bank to free resources.

```python
await client.adelete_bank(BANK_ID)
await client.aclose()
await model_client.close()
print("Cleanup complete.")
```

## Key Takeaways

- **AutoGen + Hindsight**: Use `create_hindsight_tools()` to give any AutoGen `AssistantAgent` persistent long-term memory.
- **Autonomous tool use**: The agent decides when to retain, recall, or reflect -- no manual orchestration needed.
- **Per-user banks**: Use `bank_id=f"user-{user_id}"` for per-user memory isolation.
- **Idempotent setup**: The `try/except` delete-then-create pattern makes notebooks safely re-runnable.
- **reflect_on_tool_use**: Set `reflect_on_tool_use=True` on `AssistantAgent` so the agent produces a natural-language response after tool calls instead of returning raw tool output.