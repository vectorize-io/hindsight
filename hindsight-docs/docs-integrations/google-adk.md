---
sidebar_position: 9
title: "Google ADK Persistent Memory with Hindsight | Integration"
description: "Add long-term memory to Google ADK agents with Hindsight. Implements ADK's BaseMemoryService for automatic retain on session end and recall on search_memory, plus explicit FunctionTool wrappers."
---

# Google ADK

Persistent long-term memory for [Google ADK](https://adk.dev/) agents via Hindsight. The `hindsight-google-adk` package gives you two complementary patterns:

- **`HindsightMemoryService`** — Implements ADK's `BaseMemoryService`. Pass it to `Runner(memory_service=...)` and sessions are automatically retained when they end; agents calling `search_memory` get matching results back from Hindsight.
- **`create_hindsight_tools(...)`** — Returns a list of ADK `FunctionTool`s (`hindsight_retain`, `hindsight_recall`, `hindsight_reflect`) the model can call directly inside a turn.

:::tip Recommended: Hindsight Cloud
[Sign up free](https://ui.hindsight.vectorize.io/signup) for a Hindsight Cloud API key. The integration points at production by default — no local server to manage.
:::

## Installation

```bash
pip install hindsight-google-adk
```

## Automatic Memory (BaseMemoryService)

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from hindsight_google_adk import HindsightMemoryService

memory = HindsightMemoryService.from_url(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",
)

agent = LlmAgent(name="assistant", model="gemini-2.0-flash")

runner = Runner(
    app_name="my-app",
    agent=agent,
    session_service=InMemorySessionService(),
    memory_service=memory,
)

# ... use runner.run_async(...) as normal. Memory is automatic.
```

When a session ends, `add_session_to_memory` retains all of its events to a Hindsight bank derived from `(app_name, user_id)`. When an agent calls `search_memory`, the integration runs a Hindsight recall against the same bank and returns the results as ADK `MemoryEntry` objects.

### Bank ID derivation

By default, each `(app_name, user_id)` pair gets its own bank: `"{app_name}::{user_id}"`. Override with `bank_id_template`:

```python
# Per-user, shared across apps
HindsightMemoryService.from_url(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",
    bank_id_template="user::{user_id}",
)

# Static bank (shared across all users)
HindsightMemoryService.from_url(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",
    bank_id_template="my-shared-bank",
)
```

## Explicit Tools (FunctionTool)

Give the agent direct retain / recall / reflect tools when you want it to decide when to call them mid-turn:

```python
from google.adk.agents import LlmAgent
from hindsight_google_adk import create_hindsight_tools

tools = create_hindsight_tools(
    bank_id="user-123",
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",
)

agent = LlmAgent(
    name="assistant",
    model="gemini-2.0-flash",
    tools=tools,
)
```

The agent gets three tools (toggle with `include_retain` / `include_recall` / `include_reflect`):

- **`hindsight_retain(content)`** — store information to long-term memory
- **`hindsight_recall(query)`** — search memory and return a numbered list of matches
- **`hindsight_reflect(query)`** — synthesize a coherent answer from memory

## Global Configuration

For app-wide defaults, call `configure(...)` once at startup. Subsequent `HindsightMemoryService.from_url()` / `create_hindsight_tools()` calls use it as a fallback:

```python
from hindsight_google_adk import configure

configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key=None,           # falls back to HINDSIGHT_API_KEY env var
    budget="mid",
    max_tokens=4096,
    bank_id_template="{app_name}::{user_id}",
)
```

## Configuration Reference

| Argument | Default | Description |
|---|---|---|
| `hindsight_api_url` | `https://api.hindsight.vectorize.io` | Hindsight API URL. Cloud by default. |
| `api_key` | `HINDSIGHT_API_KEY` env | Bearer token for Hindsight Cloud. |
| `bank_id_template` | `"{app_name}::{user_id}"` | Format string used to derive the bank id from ADK's `app_name` / `user_id`. |
| `budget` | `"mid"` | Recall budget level: `low`/`mid`/`high`. |
| `max_tokens` | `4096` | Max tokens for recall results. |
| `tags` | `None` | Tags added to every retained document. `app:<name>` and `user:<id>` are always added. |
| `recall_tags` | `None` | Tags appended to recall queries. `user:<id>` is always added. |
| `recall_tags_match` | `"any"` | Tag match mode: `any` / `all` / `any_strict` / `all_strict`. |
| `mission` | `None` | If set, the bank is created (idempotently) on first use with this fact-extraction mission. |
| `context` | `"google-adk"` | Source label attached to retained content. |

## Production Patterns

### Tagging memories per environment

```python
HindsightMemoryService.from_url(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",
    tags=["env:prod"],
    recall_tags=["env:prod"],
)
```

`app:` and `user:` tags are always added on top of these.

### Self-hosted Hindsight

```python
HindsightMemoryService.from_url(
    hindsight_api_url="http://localhost:8888",
)
```

No `api_key` needed for unauthenticated local servers.

### Combining memory service + tools

You can use both at once — `Runner(memory_service=HindsightMemoryService(...))` for automatic retain on session end, and `tools=create_hindsight_tools(...)` for mid-turn agent-driven recall. They share a bank when the bank ids align.

## Requirements

- Python 3.10+
- `google-adk>=2.0`
- `hindsight-client>=0.4.0`
