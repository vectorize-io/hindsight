---
sidebar_position: 21
---

# SmolAgents + Hindsight Memory


:::info Complete Application
This is a complete, runnable application demonstrating Hindsight integration.
[**View source on GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/tree/main/applications/smolagents-memory)
:::


Give your [SmolAgents](https://github.com/huggingface/smolagents) agents persistent long-term memory. Chat with an assistant multiple times and watch it remember what you told it in previous sessions.

## What This Demonstrates

- **Memory tools** — retain, recall, and reflect via `create_hindsight_tools()`
- **Auto-injected context** — relevant memories in every run via `memory_instructions()`
- **Persistent memory across sessions** — the agent remembers between script runs
- **Interactive chat loop** with the SmolAgents `ToolCallingAgent`

## Architecture

```
Session 1:
    You: "I'm a Python developer working on a FastAPI project"
    │
    ├─ memory_instructions() ──► recalls prior context (empty on first run)
    ├─ Agent decides to call hindsight_retain ──► stores the fact
    └─ Agent responds with acknowledgement

Session 2:
    You: "What do you know about me?"
    │
    ├─ memory_instructions() ──► injects "User is a Python developer..."
    ├─ Agent calls hindsight_recall ──► finds stored facts
    └─ Agent responds with everything it remembers
```

## Prerequisites

1. **Hindsight running**

   ```bash
   export OPENAI_API_KEY=your-key

   docker run --rm -it --pull always -p 8888:8888 -p 9999:9999 \
     -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
     -e HINDSIGHT_API_LLM_MODEL=o3-mini \
     -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
     ghcr.io/vectorize-io/hindsight:latest
   ```

2. **OpenAI API key** (for the SmolAgents model)

   ```bash
   export OPENAI_API_KEY=your-key
   ```

3. **Install dependencies**

   ```bash
   cd applications/smolagents-memory
   pip install -r requirements.txt
   ```

## Quick Start

### Interactive Chat

```bash
python personal_assistant.py
```

Example session:

```
Personal assistant ready (bank: personal-assistant)
Type 'quit' or 'exit' to stop.

You: I'm a Python developer and I love hiking on weekends
Assistant: Got it — I've stored that you're a Python developer who enjoys
hiking on weekends.

You: What do you know about me?
Assistant: From my memory: you're a Python developer who likes hiking on
weekends.

You: quit
```

Run it again — the agent still remembers:

```
You: What are my hobbies?
Assistant: Based on my memories, you enjoy hiking on weekends!
```

### Single Query

```bash
python personal_assistant.py "What do you remember about my preferences?"
```

### Reset Memory

```bash
python personal_assistant.py --reset
```

## How It Works

### 1. Create Memory Tools

`create_hindsight_tools()` returns SmolAgents `Tool` subclasses the agent can call:

```python
from hindsight_smolagents import create_hindsight_tools

tools = create_hindsight_tools(
    bank_id="personal-assistant",
    hindsight_api_url="http://localhost:8888",
)
# Returns: [HindsightRetainTool, HindsightRecallTool, HindsightReflectTool]
```

### 2. Pre-Inject Memories into Each Task

`memory_instructions()` performs a recall and returns a formatted string ready to prepend to whatever task you hand the agent:

```python
from hindsight_smolagents import memory_instructions

def make_task(user_query: str) -> str:
    prior_memories = memory_instructions(
        bank_id="personal-assistant",
        hindsight_api_url="http://localhost:8888",
        query=user_query,
        max_results=5,
    )
    if prior_memories:
        return f"{prior_memories}\n\nUser: {user_query}"
    return user_query
```

### 3. Wire Up the Agent

```python
from smolagents import LiteLLMModel, ToolCallingAgent

agent = ToolCallingAgent(
    tools=tools,
    model=LiteLLMModel(model_id="gpt-4o-mini"),
)

response = agent.run(make_task("What do you know about me?"))
```

## Core Files

| File | Description |
|------|-------------|
| `personal_assistant.py` | Complete working example with interactive chat and single-query modes |
| `requirements.txt` | Python dependencies |

## Customization

### Use Only Tools (No Pre-Injection)

Let the agent decide when to search memory:

```python
agent = ToolCallingAgent(
    tools=create_hindsight_tools(
        bank_id="my-bank",
        hindsight_api_url="http://localhost:8888",
    ),
    model=LiteLLMModel(model_id="gpt-4o-mini"),
)
agent.run("What do you remember about me?")
```

### Select Specific Tools

```python
tools = create_hindsight_tools(
    bank_id="my-bank",
    hindsight_api_url="http://localhost:8888",
    enable_retain=True,
    enable_recall=True,
    enable_reflect=False,
)
```

### Use a Different Model Provider

`LiteLLMModel` works with any [LiteLLM-supported provider](https://docs.litellm.ai/docs/providers). For example, Anthropic:

```python
from smolagents import LiteLLMModel

agent = ToolCallingAgent(
    tools=tools,
    model=LiteLLMModel(model_id="anthropic/claude-haiku-4-5-20251001"),
)
```

Or HuggingFace's hosted inference:

```python
from smolagents import InferenceClientModel

agent = ToolCallingAgent(
    tools=tools,
    model=InferenceClientModel(),
)
```

### Use `CodeAgent` Instead

`CodeAgent` writes Python code to call tools — useful for multi-step reasoning:

```python
from smolagents import CodeAgent

agent = CodeAgent(
    tools=tools,
    model=LiteLLMModel(model_id="gpt-4o-mini"),
)
```

## Common Issues

**"Connection refused"**
- Make sure Hindsight is running on `localhost:8888`

**"OPENAI_API_KEY not set"**
```bash
export OPENAI_API_KEY=your-key
```

**"No module named 'hindsight_smolagents'"**
```bash
pip install -r requirements.txt
```

---

**Built with:**
- [SmolAgents](https://github.com/huggingface/smolagents) - HuggingFace's lightweight agent framework
- [hindsight-smolagents](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/smolagents) - Hindsight memory tools for SmolAgents
- [Hindsight](https://github.com/vectorize-io/hindsight) - Long-term memory for AI agents
