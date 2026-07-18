---
title: "Guide: Add Google ADK Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, google-adk, agents, memory]
description: "Add Google ADK memory with Hindsight using the hindsight-google-adk package, so agents automatically retain sessions and recall relevant memory through ADK's own memory service."
image: /img/guides/guide-google-adk-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add Google ADK Memory with Hindsight](/img/guides/guide-google-adk-memory-with-hindsight.svg)

If you want **Google ADK memory with Hindsight**, the cleanest setup is the `hindsight-google-adk` package. It implements ADK's `BaseMemoryService`, so you pass it to `Runner(memory_service=...)` and sessions are retained automatically when they end, while agents that call `search_memory` get matching results back from Hindsight. That gives ADK agents long-term memory across sessions instead of starting cold every time.

This fits ADK because ADK already has a memory service interface built in. Rather than bolting memory on from the outside, the integration plugs Hindsight directly into that interface. Memory is scoped per `(app_name, user_id)` by default, so each app and user gets its own isolated bank.

If you want the agent to decide when to store or recall mid-turn instead of relying on automatic session retention, the package also ships explicit `FunctionTool` wrappers. This guide covers both patterns, plus a quick verification flow. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-google-adk`.
> 2. Create `HindsightMemoryService.from_url(hindsight_api_url=..., api_key="hsk_...")`.
> 3. Pass it to `Runner(memory_service=memory)` — sessions retain automatically on end.
> 4. Agents calling `search_memory` recall from the same bank, keyed by `(app_name, user_id)`.
> 5. Verify that a later session recalls what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- Python 3.10+
- `google-adk>=2.0` installed and a working ADK agent
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server

## Step 1: Install the package

```bash
pip install hindsight-google-adk
```

This pulls in `hindsight-client>=0.4.0`, which the integration uses to talk to your Hindsight backend.

## Step 2: Wire up the memory service

The main pattern implements ADK's `BaseMemoryService`. Construct it with `from_url` and pass it to `Runner`:

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

For a self-hosted Hindsight server, point at your own URL and drop the API key:

```python
HindsightMemoryService.from_url(hindsight_api_url="http://localhost:8888")
```

No `api_key` is needed for unauthenticated local servers.

## How the memory service uses memory

The integration hooks into the two points ADK's memory service interface exposes:

- **Retain (on session end):** when a session ends, `Runner` calls `add_session_to_memory`, which retains all of the session's events to a Hindsight bank keyed by `(app_name, user_id)`.
- **Recall (on search):** when the agent — or any other call — invokes `search_memory(app_name, user_id, query)`, the integration runs a Hindsight recall against the same bank and returns the results as ADK `MemoryEntry` objects.

Failures are resilient: Hindsight errors in the `add_*` and `search_memory` methods are logged but never propagate to the `Runner`, so memory problems don't crash the agent.

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Per-user memory banks

By default, each `(app_name, user_id)` pair gets its own bank: `"{app_name}::{user_id}"`. That keeps one user's memory isolated from another's. Override the scheme with `bank_id_template`:

```python
# Per-user bank, shared across apps
HindsightMemoryService.from_url(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",
    bank_id_template="user::{user_id}",
)

# Static bank, shared across all users
HindsightMemoryService.from_url(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",
    bank_id_template="my-shared-bank",
)
```

Every retained event also carries `app:<app_name>` and `user:<user_id>` tags, and recall queries automatically include `user:<user_id>`, so users never see each other's memories.

## Optional: explicit retain / recall / reflect tools

If you want the model to decide when to store or recall mid-turn, add the explicit tools instead of (or alongside) the memory service:

```python
from google.adk.agents import LlmAgent
from hindsight_google_adk import create_hindsight_tools

tools = create_hindsight_tools(
    bank_id="user-123",
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",
)

agent = LlmAgent(name="assistant", model="gemini-2.0-flash", tools=tools)
```

The agent gets three tools (toggle with `include_retain` / `include_recall` / `include_reflect`):

- **`hindsight_retain(content)`** — store information to long-term memory
- **`hindsight_recall(query)`** — search memory and return matches
- **`hindsight_reflect(query)`** — synthesize a coherent answer from memory

You can use both patterns at once — the memory service for automatic retention on session end, and the tools for agent-driven recall mid-turn. They share a bank when the bank ids align.

## Verify that memory is working

A good test sequence is:

1. run an agent turn under a `Runner` with the `HindsightMemoryService` attached
2. let the agent record a decision, fact, or preference
3. end the session so its events are retained
4. start a new session with the same `app_name` and `user_id`
5. ask about the earlier detail so the agent calls `search_memory`

If the recalled results surface the earlier detail, the setup is working.

## Common mistakes

### Changing `app_name` or `user_id` between sessions

The bank is derived from `(app_name, user_id)` by default. If either value changes, the new session reads a different bank and won't see the earlier memory.

### Testing retain too early

Session events are retained when the session ends. If you check before the session completes, the content may not have been stored yet.

### Assuming memory is global

By default each `(app_name, user_id)` pair gets its own bank. That is usually what you want, but do not expect one user to recall another user's context unless you set a shared `bank_id_template`.

### Passing an API key to a local server

Hindsight Cloud needs an `api_key`. A self-hosted, unauthenticated server does not — just set `hindsight_api_url` to your local URL.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — pass its URL to `hindsight_api_url` and omit `api_key`.

### Do I have to use the explicit tools?

No. The `HindsightMemoryService` handles retain-on-session-end and recall-on-`search_memory` automatically. The `FunctionTool` wrappers are optional, for when you want the agent to control memory mid-turn.

### How is memory scoped?

Per `(app_name, user_id)` by default, using the template `"{app_name}::{user_id}"`. Override it with `bank_id_template`.

### Can I set app-wide defaults once?

Yes. Call `configure(...)` at startup and later `HindsightMemoryService.from_url()` / `create_hindsight_tools()` calls use it as a fallback.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Google ADK integration docs](https://hindsight.vectorize.io/docs/integrations/google-adk)
