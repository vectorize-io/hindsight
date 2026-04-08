---
sidebar_position: 9
title: "Agno Agent Persistent Memory with Hindsight | Integration Guide"
description: "Add persistent memory to Agno agents using Hindsight's retain, recall, and reflect tools. Plug into Agno's native Toolkit pattern for long-term memory across sessions."
---

# Agno

Persistent memory tools for [Agno](https://github.com/agno-agi/agno) agents via Hindsight. Give your agents long-term memory with retain, recall, and reflect — using Agno's native Toolkit pattern.

## Features

- **Native Toolkit** - Extends Agno's `Toolkit` base class, just like `Mem0Tools`
- **Memory Instructions** - Pre-recall memories for injection into `Agent(instructions=[...])`
- **Three Memory Tools** - Retain (store), Recall (search), Reflect (synthesize) — include any combination
- **Flexible Bank Resolution** - Static bank ID, `RunContext.user_id`, or custom resolver
- **Simple Configuration** - Configure once globally, or pass a client directly

## Installation

```bash
pip install hindsight-agno
```

## Quick Start

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

agent.print_response("Remember that I prefer dark mode")
agent.print_response("What are my preferences?")
```

The agent now has three tools it can call:

- **`retain_memory`** — Store information to long-term memory
- **`recall_memory`** — Search long-term memory for relevant facts
- **`reflect_on_memory`** — Synthesize a reasoned answer from memories

## With Memory Instructions

Pre-recall relevant memories and inject them into the system prompt:

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

## Selecting Tools

Include only the tools you need:

```python
tools = [HindsightTools(
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
    enable_retain=True,
    enable_recall=True,
    enable_reflect=False,  # Omit reflect
)]
```

## Bank Resolution

The bank ID is resolved in order:

1. **`bank_resolver`** — Custom callable `(RunContext) -> str`
2. **`bank_id`** — Static bank ID passed to constructor
3. **`run_context.user_id`** — Automatic per-user banks

```python
# Per-user banks from RunContext
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[HindsightTools(hindsight_api_url="http://localhost:8888")],
    user_id="user-123",  # Used as bank_id
)

# Custom resolver
def resolve_bank(ctx):
    return f"team-{ctx.user_id}"

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[HindsightTools(
        bank_resolver=resolve_bank,
        hindsight_api_url="http://localhost:8888",
    )],
)
```

## Global Configuration

Instead of passing connection details to every toolkit, configure once:

```python
from hindsight_agno import configure, HindsightTools

configure(
    hindsight_api_url="http://localhost:8888",
    api_key="your-api-key",       # Or set HINDSIGHT_API_KEY env var
    budget="mid",                  # Recall budget: low/mid/high
    max_tokens=4096,               # Max tokens for recall results
    tags=["env:prod"],             # Tags for stored memories
    recall_tags=["scope:global"],  # Tags to filter recall
    recall_tags_match="any",       # Tag match mode: any/all/any_strict/all_strict
)

# Now create toolkit without passing connection details
tools = [HindsightTools(bank_id="user-123")]
```

## Configuration Reference

### `HindsightTools()`

| Parameter | Default | Description |
|---|---|---|
| `bank_id` | `None` | Static Hindsight memory bank ID |
| `bank_resolver` | `None` | Callable `(RunContext) -> str` for dynamic bank ID |
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
- agno
- hindsight-client >= 0.4.0
- A running Hindsight API server
