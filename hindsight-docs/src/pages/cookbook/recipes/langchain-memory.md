---
sidebar_position: 7
---

# LangChain Chatbot with Long-Term Memory


:::tip Run this notebook
This recipe is available as an interactive Jupyter notebook.
[**Open in GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/blob/main/notebooks/07-langchain-memory.ipynb)
:::


Add persistent long-term memory to a LangChain chatbot using Hindsight.
The model gets retain/recall/reflect tools and we handle the tool execution loop manually.

For multi-step agent loops managed by a framework (with state, branching, and checkpoints),
see the [LangGraph ReAct Agent](./06-langgraph-react-agent.ipynb) notebook.

## Prerequisites

- Python 3.10+
- A [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) account **or** a self-hosted Hindsight instance
- An OpenAI API key (or any LangChain-supported model)

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
!pip install hindsight-langgraph langchain-openai nest_asyncio python-dotenv -U
```

## Setup

Configure the Hindsight client. Set `HINDSIGHT_API_URL` and `HINDSIGHT_API_KEY` in your `.env` file or environment:

| | Hindsight Cloud | Self-Hosted |
|---|---|---|
| `HINDSIGHT_API_URL` | `https://api.hindsight.vectorize.io` | `http://localhost:8888` |
| `HINDSIGHT_API_KEY` | Your cloud API key | *(not required)* |

> **Note:** You also need `OPENAI_API_KEY` set in your environment or `.env` file for the LangChain model.

```python
import nest_asyncio
nest_asyncio.apply()

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

## Create Tools and Bind to Model

Create Hindsight memory tools and bind them to a ChatModel using `bind_tools()`.
This is pure LangChain — no LangGraph required.

The model decides which tools to call based on the conversation.
We handle tool execution in a simple loop (see below).

```python
from hindsight_langgraph import create_hindsight_tools
from langchain_openai import ChatOpenAI

BANK_ID = "langchain-demo"

tools = create_hindsight_tools(
    client=client,
    bank_id=BANK_ID,
    tags=["source:langchain"],
    budget="mid",
)

# bind_tools tells the model about available tools — pure LangChain, no LangGraph
model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

print(f"Model bound with {len(tools)} tools: {[t.name for t in tools]}")
```

## Create the Memory Bank

```python
await client.acreate_bank(BANK_ID, name="LangChain Demo")
print(f"Bank created: {BANK_ID}")
```

## Tool Execution Loop

When we invoke the model, it may call tools — sometimes multiple in sequence.
For example, it might call `hindsight_recall` first, then `hindsight_reflect` to synthesize.
We loop until the model responds with text instead of tool calls.

```python
from langchain_core.messages import HumanMessage, ToolMessage


async def chat_with_memory(message: str) -> str:
    """Chat that executes tool calls in a loop until the model responds with text."""
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant with long-term memory. "
            "Use hindsight_retain to store important facts. "
            "Use hindsight_recall to search memory before answering questions. "
            "Use hindsight_reflect for thoughtful summaries."
        )},
        {"role": "user", "content": message},
    ]

    # Loop: let the model call as many tools as it needs
    while True:
        response = await model.ainvoke(messages)

        if not response.tool_calls:
            break

        messages.append(response)
        for tool_call in response.tool_calls:
            tool = next(t for t in tools if t.name == tool_call["name"])
            result = await tool.ainvoke(tool_call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
            print(f"  [{tool_call['name']}] {result}")

    return response.content
```

## Store a Memory

The model recognizes important facts and calls `hindsight_retain` to store them.

```python
response = await chat_with_memory(
    "I'm a frontend developer who uses React and TypeScript. "
    "I prefer Vim keybindings. Please remember this."
)
print(f"\nAssistant: {response}")
```

```python
import asyncio

# Hindsight processes retained content asynchronously (extracting facts, entities, embeddings).
# The sleep gives the server time to finish before we recall.
await asyncio.sleep(3)
```

## Recall a Memory

In a new call, the model searches memory to answer a question about past context.

```python
response = await chat_with_memory("What programming languages do I use?")
print(f"\nAssistant: {response}")
```

## Reflect on Memories

The model calls `hindsight_reflect` to synthesize a thoughtful answer from stored knowledge.

```python
response = await chat_with_memory(
    "Based on what you know about me, suggest a good IDE setup."
)
print(f"\nAssistant: {response}")
```

## Cleanup

```python
await client.adelete_bank(BANK_ID)
print("Bank deleted.")
```

## LangChain vs LangGraph

This notebook shows a **manual tool-loop** pattern: we invoke the model, execute any tool calls ourselves, and loop until the model responds with text.
This works well for straightforward store/retrieve use cases where you want full control over the execution loop.

For **agentic workflows** — where the framework manages the tool execution loop,
supports branching, state checkpoints, and complex multi-step graphs — use a LangGraph agent instead.
See the [LangGraph ReAct Agent](./06-langgraph-react-agent.ipynb) notebook.