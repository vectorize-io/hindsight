---
title: "Long-Term Memory for Google ADK Agents with Hindsight"
authors: [benfrank241]
slug: "2026/06/04/google-adk-persistent-memory"
date: 2026-06-04T12:00
tags: [integrations, google, adk, agents, memory, gemini, tutorial, hindsight]
description: "Add persistent long-term memory to Google ADK agents with Hindsight. Drop-in BaseMemoryService for automatic retain on session end and recall on search_memory — plus explicit retain/recall/reflect tools when you want them."
image: /img/blog/google-adk-persistent-memory.png
hide_table_of_contents: true
---

![Long-Term Memory for Google ADK Agents with Hindsight](/img/blog/google-adk-persistent-memory.png)

[Google's Agent Development Kit](https://adk.dev/) gives you a clean abstraction for building Gemini-powered agents — sessions, runners, tools, plugins, multi-agent composition. It has a memory interface too: `BaseMemoryService`, the contract any persistent-memory backend can implement. What it doesn't ship is a memory backend. That's left to you.

`hindsight-google-adk` is the drop-in implementation. Wire it into your `Runner`, and sessions are automatically retained when they end. The next time the agent calls `search_memory`, results come back from a Hindsight bank scoped to that user. No retrieval code to write, no vector store to manage, no schema decisions to make up front.

<!-- truncate -->

## TL;DR

- Google ADK ships a `BaseMemoryService` interface but no memory backend.
- `hindsight-google-adk` is a drop-in `BaseMemoryService` implementation.
- Pass `memory_service=HindsightMemoryService.from_url(...)` to your ADK `Runner` and sessions are automatically retained on close + recallable via `search_memory`.
- Or expose explicit `hindsight_retain` / `hindsight_recall` / `hindsight_reflect` tools the agent can call mid-turn.
- Bank IDs default to `{app_name}::{user_id}`, so memory is scoped per user out of the box.

---

## Why ADK Agents Need a Memory Layer

ADK gives you the agent loop, the tool framework, and the session-and-event infrastructure. By default, that infrastructure is ephemeral: when a session ends, its events are flushed. The next time the user comes back, the agent reads from a fresh `Session` with no record of the prior conversation.

`BaseMemoryService` is ADK's escape hatch — the interface that lets a long-term memory layer sit alongside the short-term session store. Implement `add_session_to_memory` and `search_memory` and the `Runner` will call them at the right moments. ADK ships an `InMemoryMemoryService` for development; for production, you implement (or install) something that actually persists.

The integration point is clean. The work is in the backend: fact extraction, deduplication, entity resolution, multi-strategy retrieval, ranking. That's what Hindsight does. Plugging it into ADK is one line.

---

## How It Works

`HindsightMemoryService` implements ADK's `BaseMemoryService` and hands off the heavy lifting to Hindsight. The handoff happens at three lifecycle points:

```text
Session ends
  └─ Runner calls add_session_to_memory(session)
       └─ HindsightMemoryService formats events as a document
            └─ Hindsight retains it to bank {app_name}::{user_id}
                 (fact extraction + entity resolution runs in the background)

Agent calls search_memory(query)
  └─ HindsightMemoryService calls Hindsight recall on the same bank
       └─ Results returned as ADK MemoryEntry objects
            (one per surfaced fact)

Code calls add_memory(...) with explicit MemoryEntry objects
  └─ Each entry is retained as its own document
```

Three things worth knowing about the implementation:

1. **Failures never break the agent.** Every retain and recall call is wrapped — if Hindsight is unreachable, the error is logged and the `Runner` continues. The agent runs without memory rather than crashing.
2. **Bank scoping is per-`(app_name, user_id)` by default.** Two users of the same app get separate banks; the same user across two apps also gets separate banks. Override via `bank_id_template`.
3. **Tags are added automatically.** Every retain carries `app:<name>` and `user:<id>` tags. Every recall filters by `user:<id>`. Cross-user contamination is impossible by default.

---

## Setup

### 1. Install

```bash
pip install hindsight-google-adk
```

Requires Python 3.10+, `google-adk>=2.0`, and `hindsight-client>=0.4.0`.

### 2. Pick a Hindsight Deployment

**Hindsight Cloud** is the fastest path — [sign up free](https://ui.hindsight.vectorize.io/signup), grab an API key, point your code at `https://api.hindsight.vectorize.io`. The integration defaults to the Cloud URL.

**Self-hosted**:

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=YOUR_OPENAI_KEY
hindsight-api  # starts at http://localhost:8888
```

### 3. Wire It Into Your Runner

```python
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from hindsight_google_adk import HindsightMemoryService

memory = HindsightMemoryService.from_url(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_your_token_here",
)

agent = LlmAgent(name="assistant", model="gemini-2.0-flash")

runner = Runner(
    app_name="my-app",
    agent=agent,
    session_service=InMemorySessionService(),
    memory_service=memory,
)
```

That's the integration. The rest is your normal ADK code — `runner.run_async(...)`, your tools, your prompts, your session lifecycle. The Runner takes care of calling `add_session_to_memory` when sessions end, and the agent gets `search_memory` for free.

---

## Explicit Tools: When You Want the Agent to Decide

The `BaseMemoryService` path is automatic — retention happens on session end, recall happens whenever the agent calls `search_memory`. That's a clean default, but sometimes you want the agent to make memory calls deliberately mid-turn: store a learning right after it happens, recall before pivoting topics, synthesize a coherent answer from many memories at once.

For that, the integration also exposes `create_hindsight_tools()`, which returns ADK `FunctionTool`s the model can call:

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

You get three tools, toggleable via `include_retain` / `include_recall` / `include_reflect`:

- **`hindsight_retain(content)`** — store information to long-term memory immediately, without waiting for session end.
- **`hindsight_recall(query)`** — search memory and return a numbered list of matches.
- **`hindsight_reflect(query)`** — synthesize a coherent answer from memory rather than returning raw facts. Slower, but much higher signal for "what's the state of X?" questions.

The two patterns compose. Run `Runner(memory_service=HindsightMemoryService(...))` for automatic session-end retention *and* `tools=create_hindsight_tools(...)` for mid-turn agent-driven calls. As long as the bank IDs line up, both paths read and write the same memory.

---

## Bank Scoping

The default `bank_id_template` is `"{app_name}::{user_id}"`. Most setups should leave it alone — every user in every app gets their own isolated bank, and the per-recall `user:<id>` tag filter is belt-and-suspenders.

Two cases where you'd change it:

**Memory shared across apps for the same user.** A reading-list agent and a calendar agent that both know about the user's preferences:

```python
HindsightMemoryService.from_url(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",
    bank_id_template="user::{user_id}",
)
```

**Static shared bank.** One bank for the whole app — useful for small teams or community-shared memory:

```python
HindsightMemoryService.from_url(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",
    bank_id_template="my-shared-bank",
)
```

The `app:` and `user:` tags are still added on retain, so even with a shared bank you can post-hoc slice by user if you need to.

---

## Configuration Reference

Pulled straight from the integration:

| Argument | Default | What it does |
|---|---|---|
| `hindsight_api_url` | `https://api.hindsight.vectorize.io` | Hindsight API URL (Cloud by default) |
| `api_key` | `HINDSIGHT_API_KEY` env | Bearer token for Hindsight Cloud |
| `bank_id_template` | `"{app_name}::{user_id}"` | Format string to derive the bank id |
| `budget` | `"mid"` | Recall budget: `low` / `mid` / `high` |
| `max_tokens` | `4096` | Max tokens in the recall response |
| `tags` | `None` | Tags added to every retain (on top of `app:` / `user:`) |
| `recall_tags` | `None` | Tags appended to recall queries (on top of `user:`) |
| `recall_tags_match` | `"any"` | Tag match mode: `any` / `all` / `any_strict` / `all_strict` |
| `mission` | `None` | If set, the bank is created on first use (idempotent) with this fact-extraction mission |
| `context` | `"google-adk"` | Source label attached to retained content |

### Global Config

For app-wide defaults, call `configure(...)` once at startup. Subsequent `HindsightMemoryService.from_url()` / `create_hindsight_tools()` calls use it as the fallback:

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

---

## Production Patterns

### Per-Environment Tagging

```python
HindsightMemoryService.from_url(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",
    tags=["env:prod"],
    recall_tags=["env:prod"],
)
```

`app:` and `user:` are still added on top. Now you can run dev/staging/prod against the same Hindsight project without polluting each other's recall.

### Bootstrapped Banks with a Mission

When a new user shows up, you don't always want to wait for the first retained session to define a memory bank's character. Pass a `mission` and the bank is created idempotently on first use with that fact-extraction prompt baked in:

```python
HindsightMemoryService.from_url(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",
    mission=(
        "Extract user preferences, ongoing tasks, and project context. "
        "Ignore casual chit-chat unless it reveals a stable preference."
    ),
)
```

### Self-Hosted Hindsight

```python
HindsightMemoryService.from_url(
    hindsight_api_url="http://localhost:8888",
)
```

No `api_key` needed against an unauthenticated local server.

### Recall Budget

`mid` is the default and the right starting point. Drop to `low` if your agent is latency-sensitive and you've got a tight first-response budget; bump to `high` for complex agents where deeper recall pays off (research, long-running planners, multi-document reasoning).

---

## Where ADK + Hindsight Pays Off

The integration earns its keep anywhere your ADK agent will see the same user more than once:

- **Long-running assistants.** A Gemini-powered personal assistant that remembers preferences, ongoing projects, and prior decisions across days and weeks.
- **Multi-agent setups.** Compose multiple ADK agents under one app; share a bank to give them a common picture of the user. The `app:` / `user:` tagging means you can still slice memory per sub-agent when you need to.
- **Customer-facing copilots.** Domain-specific agents (legal, medical, finance) where context — past matters, known conditions, account state — should persist across sessions.
- **Internal-tool agents.** An ADK agent that reads from your team's data, writes to a ticketing system, or handles intake. Past tickets and resolutions show up automatically in the next session.

Anywhere short-term session memory isn't enough, the `BaseMemoryService` slot is exactly where Hindsight goes.

---

## Production Notes

**Failure isolation.** Both `add_*` and `search_memory` swallow Hindsight exceptions and log them. A network blip or a temporary 5xx never crashes the `Runner`. Worst case, the agent runs without memory for that turn.

**Privacy and retention.** Per-user banks mean per-user deletion is trivial — `await client.adelete_bank(bank_id=f"{app_name}::{user_id}")`. The REST API serves the same operation under `DELETE /v1/default/banks/{bank_id}`. If you key banks by your internal user ID, right-to-be-forgotten requests are a one-call cleanup.

**Latency.** Recall on `search_memory` is a single Hindsight call, typically 50–300 ms depending on `budget` and memory size. Sessions retain asynchronously when they close, so retention never adds latency to the agent's response loop.

**Bank creation.** If you don't pass a `mission`, banks materialize on first retain. If you do pass a `mission`, the integration creates the bank idempotently on first use and tracks (in-process) which banks it's already initialized to avoid repeat create calls.

---

## Recap

ADK gives you the agent runtime and a `BaseMemoryService` slot. Hindsight gives you the memory backend that slots in. Together: persistent long-term memory for Gemini-powered agents with one line of integration code.

- **Automatic path:** `memory_service=HindsightMemoryService.from_url(...)` on the `Runner` — sessions retained on close, recall on `search_memory`.
- **Explicit path:** `create_hindsight_tools(...)` for agent-driven mid-turn retain / recall / reflect.
- **Both at once:** they share a bank when bank IDs align.

No retrieval code to write, no vector store to manage, no schema to design up front. The agent gets better the more it's used.

---

## Next Steps

- **Start with Hindsight Cloud**: [sign up](https://ui.hindsight.vectorize.io/signup), grab a key, point your `Runner` at it
- **Read the [Google ADK integration docs](https://hindsight.vectorize.io/integrations/google-adk)** for the full configuration reference
- **Pick a `bank_id_template`** that matches your user model — per-user is the safe default
- **Decide between automatic / explicit / both** — start with `BaseMemoryService` and add tools if and when the agent benefits from mid-turn calls
- **Tune `budget`** if you're latency-sensitive (`low`) or doing deep reasoning (`high`)
- **Browse the [full integration list](https://hindsight.vectorize.io/integrations/)** — Hermes, OpenAI Agents, n8n, Vapi, AgentCore, and 30+ others all share the same memory layer

---

**Further reading:**

- [What Is Agent Memory?](https://vectorize.io/what-is-agent-memory/) — foundational concepts
- [Hindsight Google ADK Integration docs](https://hindsight.vectorize.io/integrations/google-adk) — full configuration reference
- [Multi-Turn Agent Memory with AWS AgentCore](/blog/2026/05/01/agentcore-persistent-memory) — comparable runtime-adapter pattern
- [Best AI Agent Memory Systems in 2026](https://vectorize.io/articles/best-ai-agent-memory-systems/) — full landscape comparison
