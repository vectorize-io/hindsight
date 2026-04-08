---
sidebar_position: 13
title: "Strands Agents Persistent Memory with Hindsight | Integration"
description: "Add long-term memory to Strands Agents SDK agents with Hindsight. Retain, recall, and reflect tools using Strands' native @tool pattern for persistent memory across sessions."
---

# Strands Agents

Persistent memory tools for [Strands Agents SDK](https://github.com/strands-agents/sdk-python) agents via Hindsight. Give your agents long-term memory with retain, recall, and reflect — using Strands' native `@tool` pattern.

## Features

- **Native `@tool` Functions** - Tools are plain Python functions, compatible with `Agent(tools=[...])`
- **Memory Instructions** - Pre-recall memories for injection into agent system prompt
- **Three Memory Tools** - Retain (store), Recall (search), Reflect (synthesize) — include any combination
- **Simple Configuration** - Configure once globally, or pass a client directly

## Installation

```bash
pip install hindsight-strands
```

## Quick Start

```python
from strands import Agent
from hindsight_strands import create_hindsight_tools

tools = create_hindsight_tools(
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
)

agent = Agent(tools=tools)
agent("Remember that I prefer dark mode")
agent("What are my preferences?")
```

The agent now has three tools it can call:

- **`hindsight_retain`** — Store information to long-term memory
- **`hindsight_recall`** — Search long-term memory for relevant facts
- **`hindsight_reflect`** — Synthesize a reasoned answer from memories

## With Memory Instructions

Pre-recall relevant memories and inject them into the system prompt:

```python
from hindsight_strands import create_hindsight_tools, memory_instructions

tools = create_hindsight_tools(
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
)

memories = memory_instructions(
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
)

agent = Agent(
    tools=tools,
    system_prompt=f"You are a helpful assistant.\n\n{memories}",
)
```

## Selecting Tools

Include only the tools you need:

```python
tools = create_hindsight_tools(
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
    enable_retain=True,
    enable_recall=True,
    enable_reflect=False,  # Omit reflect
)
```

## Global Configuration

Instead of passing connection details to every call, configure once:

```python
from hindsight_strands import configure, create_hindsight_tools

configure(
    hindsight_api_url="http://localhost:8888",
    api_key="your-api-key",       # Or set HINDSIGHT_API_KEY env var
    budget="mid",                  # Recall budget: low/mid/high
    max_tokens=4096,               # Max tokens for recall results
    tags=["env:prod"],             # Tags for stored memories
    recall_tags=["scope:global"],  # Tags to filter recall
    recall_tags_match="any",       # Tag match mode: any/all/any_strict/all_strict
)

# Now create tools without passing connection details
tools = create_hindsight_tools(bank_id="user-123")
```

## Configuration Reference

### `create_hindsight_tools()`

| Parameter | Default | Description |
|---|---|---|
| `bank_id` | *required* | Hindsight memory bank ID |
| `client` | `None` | Pre-configured Hindsight client |
| `hindsight_api_url` | `None` | API URL (used if no client provided) |
| `api_key` | `None` | API key (used if no client provided) |
| `budget` | `"mid"` | Recall/reflect budget level (low/mid/high) |
| `max_tokens` | `4096` | Maximum tokens for recall results |
| `tags` | `None` | Tags applied when storing memories |
| `recall_tags` | `None` | Tags to filter when searching |
| `recall_tags_match` | `"any"` | Tag matching mode |
| `enable_retain` | `True` | Include the retain (store) tool |
| `enable_recall` | `True` | Include the recall (search) tool |
| `enable_reflect` | `True` | Include the reflect (synthesize) tool |

### `memory_instructions()`

| Parameter | Default | Description |
|---|---|---|
| `bank_id` | *required* | Hindsight memory bank ID |
| `client` | `None` | Pre-configured Hindsight client |
| `hindsight_api_url` | `None` | API URL (used if no client provided) |
| `api_key` | `None` | API key (used if no client provided) |
| `query` | `"relevant context about the user"` | Recall query for memory injection |
| `budget` | `"low"` | Recall budget level |
| `max_results` | `5` | Maximum memories to inject |
| `max_tokens` | `4096` | Maximum tokens for recall results |
| `prefix` | `"Relevant memories:\n"` | Text prepended before memory list |
| `tags` | `None` | Tags to filter recall results |
| `tags_match` | `"any"` | Tag matching mode |

### `configure()`

| Parameter | Default | Description |
|---|---|---|
| `hindsight_api_url` | Production API | Hindsight API URL |
| `api_key` | `HINDSIGHT_API_KEY` env | API key for authentication |
| `budget` | `"mid"` | Default recall budget level |
| `max_tokens` | `4096` | Default max tokens for recall |
| `tags` | `None` | Default tags for retain operations |
| `recall_tags` | `None` | Default tags to filter recall |
| `recall_tags_match` | `"any"` | Default tag matching mode |
| `verbose` | `False` | Enable verbose logging |

## Requirements

- Python >= 3.10
- strands-agents
- hindsight-client >= 0.4.0
- A running Hindsight API server
