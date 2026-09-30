---
title: "Why Your AWS Strands Agent Keeps Starting From Scratch (And How to Stop It)"
authors: [benfrank241]
date: 2026-03-25T12:00
tags: [strands, aws, agents, python, memory, tutorial]
image: /img/blog/strands-persistent-memory.png
hide_table_of_contents: true
---

![Why Your AWS Strands Agent Keeps Starting From Scratch (And How to Stop It)](/img/blog/strands-persistent-memory.png)

AWS Strands agents are stateless by default — every session starts cold. Here's how to add persistent long-term memory using `hindsight-strands`, so your agent remembers what matters across runs.

<!-- truncate -->

## TL;DR

- AWS Strands agents are stateless by default — every session starts cold
- `hindsight-strands` adds three native memory tools to any Strands agent: retain, recall, and reflect
- `memory_instructions()` pre-loads relevant memories directly into the system prompt before the agent starts
- Works with any Strands-compatible LLM backend: Bedrock, Anthropic, OpenAI
- You control what gets remembered, how much is recalled, and how memories are scoped with tags

---

## The Problem

AWS Strands gives you a clean, decorator-based way to build agents. Define your tools, wire up a model, and you're running. The problem: every conversation starts from scratch.

Ask your agent to remember your AWS region preference. It will. Ask it again next session — it won't have any idea what you're talking about. The `@tool` pattern handles the shape of your agent's capabilities, but it has nothing to say about persistence.

For most demo use cases this doesn't matter. For real applications — customer-facing agents, coding assistants, support bots — it's a fundamental limitation.

---

## The Approach

[Hindsight](https://github.com/vectorize-io/hindsight) is a memory layer for AI agents. It stores facts, extracts semantically relevant ones at query time, and can synthesize reasoned answers from accumulated context. The `hindsight-strands` package wraps this into native Strands tools, so your agent can manage its own memory the same way it uses any other capability.

Three tools get added to your agent:

- **`hindsight_retain`** — stores a piece of information to the memory bank
- **`hindsight_recall`** — searches memory semantically and returns relevant facts
- **`hindsight_reflect`** — synthesizes a reasoned answer from everything in memory

There's also a `memory_instructions()` helper that runs a recall query before the agent starts and injects the results into the system prompt — useful when you want the agent to be context-aware from the first message, not just when it explicitly calls a tool.

```
User session

┌─────────────────────────────────────────┐
│  System prompt                          │
│  + memory_instructions() results        │
├─────────────────────────────────────────┤
│  Agent (Strands)                        │
│  ├── hindsight_retain                   │
│  ├── hindsight_recall                   │
│  └── hindsight_reflect                  │
└─────────────────────────────────────────┘
          │                   ▲
        retain              recall
          │                   │
          ▼                   │
    ┌───────────────────────────────┐
    │        Hindsight bank         │
    └───────────────────────────────┘
```

---

## Implementation

### Install

```bash
pip install hindsight-strands
```

You'll also need a running Hindsight instance. Two options:

**Option 1 — Hindsight Cloud (no setup required)**

Sign up at [ui.hindsight.vectorize.io](https://ui.hindsight.vectorize.io) and grab your API URL and key from the dashboard. Pass them directly to `create_hindsight_tools()`.

> **Note:** Use Hindsight Cloud if you want to skip self-hosting entirely — free to get started.

**Option 2 — Self-hosted with Docker**

```bash
export OPENAI_API_KEY=sk-...

docker run --rm -it --pull always \
  -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

The API listens on port `8888`. The UI is available at `http://localhost:9999`. Hindsight also supports Anthropic, Gemini, Groq, and Ollama — swap `HINDSIGHT_API_LLM_PROVIDER` and the key accordingly.

### Basic setup

```python
from strands import Agent
from hindsight_strands import create_hindsight_tools

tools = create_hindsight_tools(
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
)

agent = Agent(tools=tools)
```

`bank_id` is how Hindsight partitions memory. Use something meaningful: a user ID, a session namespace, a project slug. The bank is created automatically on first write.

Now the agent has three new tools it can invoke. Run a session:

```python
agent("Remember that I'm deploying to us-east-1 and I prefer Bedrock Nova Pro for inference.")
```

The agent will call `hindsight_retain` with the relevant facts. Next session:

```python
agent("What model and region should I target for this deployment?")
```

It will call `hindsight_recall`, retrieve the stored preferences, and answer correctly — even in a fresh Python process.

### Pre-loading memory into the system prompt

Tool-based recall works well when the agent knows it needs to look something up. But sometimes you want the agent to already have context before the first message. That's what `memory_instructions()` is for:

```python
from strands import Agent
from hindsight_strands import create_hindsight_tools, memory_instructions

bank_id = "user-123"
api_url = "http://localhost:8888"

# Recall relevant context before the agent starts
memories = memory_instructions(
    bank_id=bank_id,
    hindsight_api_url=api_url,
    query="user preferences and environment setup",
)

agent = Agent(
    tools=create_hindsight_tools(bank_id=bank_id, hindsight_api_url=api_url),
    system_prompt=f"You are a helpful AWS infrastructure assistant.\n\n{memories}",
)
```

`memory_instructions()` runs a recall query synchronously and returns a formatted string you can embed anywhere in your prompt. The output looks like:

```
## Relevant memories

- User prefers Bedrock Nova Pro for inference tasks
- Default deployment region is us-east-1
- Project uses CDK for infrastructure
```

Combine both approaches: use `memory_instructions()` for ambient context, leave the tools in place for the agent to actively learn new things during the session.

### Global configuration

If you're instantiating agents across multiple modules, threading the `hindsight_api_url` and other params everywhere gets tedious. Use `configure()` once at startup:

```python
from hindsight_strands import configure, create_hindsight_tools

configure(
    hindsight_api_url="http://localhost:8888",
    api_key="YOUR_API_KEY",       # if auth is enabled
    budget="mid",                  # low / mid / high
    tags=["env:prod"],             # tag everything written this session
)

# Elsewhere, no need to pass connection details
tools = create_hindsight_tools(bank_id="user-123")
```

`budget` controls how many tokens are spent on recall — `low` is fast and cheap, `high` pulls more context. Default is `mid`.

### Tag-based scoping

Tags let you organize and filter memory across banks or environments:

```python
tools = create_hindsight_tools(
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
    tags=["env:prod", "team:platform"],        # applied to everything retained
    recall_tags=["env:prod"],                  # only recall memories with this tag
    recall_tags_match="all",                   # must match all tags (vs "any")
)
```

This is useful if you're sharing a bank across environments or teams and want to keep recall scoped.

### Selective tools

You don't have to enable all three tools.

A read-only agent is useful for serving end users who should be able to query memory but not modify it — for example, a customer-facing assistant that reads from a shared knowledge bank maintained by a separate process:

```python
tools = create_hindsight_tools(
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
    enable_retain=False,
    enable_recall=True,
    enable_reflect=True,
)
```

A write-only agent is useful for ingestion pipelines — a background agent that processes documents, support tickets, or logs and stores structured facts without exposing recall:

```python
tools = create_hindsight_tools(
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
    enable_retain=True,
    enable_recall=False,
    enable_reflect=False,
)
```

---

## Pitfalls & Edge Cases

**The agent won't retain unless it decides to.**
Memory tools are just tools — the agent calls them based on context. If you ask it to "remember X," it will. If you have a casual exchange and expect it to file something away, it probably won't. Use `memory_instructions()` to inject what matters rather than hoping the agent retains it autonomously.

**`bank_id` is your consistency primitive.**
If two sessions use different `bank_id` values, they have separate memory — full stop. Make sure user identity maps reliably to `bank_id` across your application. A common bug is using a session ID instead of a user ID, which silently gives every session a blank slate.

**Asyncio conflicts with Strands.**
Strands runs its own event loop. The `hindsight-strands` package handles this by executing Hindsight client calls in a `ThreadPoolExecutor` rather than `asyncio.run()`. If you're building on top of this or writing custom tooling, don't try to `await` Hindsight calls directly inside a Strands tool — you'll get "event loop already running" errors.

**`memory_instructions()` is synchronous and blocking.**
It runs a network call at agent initialization time. Don't call it inside a hot loop or on every request if the agent is being reconstructed frequently. Cache the result if the query is static and you're spinning up many agents.

**Reflect is expensive.**
`hindsight_reflect` synthesizes an answer from memory rather than just retrieving facts. It's more useful for open-ended queries but will use more tokens. If cost is a concern, leave it disabled and use recall instead.

---

## Tradeoffs & Alternatives

**When not to use this:**
If your sessions are fully self-contained and users expect no continuity, adding memory tooling introduces unnecessary complexity and latency. Not every agent needs persistent state.

**Conversation history vs. long-term memory:**
Strands manages within-session context automatically via its message history. `hindsight-strands` is for cross-session memory — facts that need to survive beyond a single conversation. Don't use it as a substitute for prompt history.

**Alternatives:**
- **DynamoDB / RDS**: You can roll your own persistence by writing session data to a database and loading it back. More control, more code, no semantic search.
- **Amazon Bedrock AgentCore Memory**: AWS's fully managed memory service for Bedrock agents. Handles both short-term (within-session) and long-term (cross-session) memory automatically. Tighter AWS integration but locks you into the Bedrock agent runtime — you can't use it with a Strands agent backed by a non-Bedrock provider.
- **LangChain memory primitives**: If you're already on LangChain, its memory abstractions are more mature. `hindsight-strands` exists because Strands has its own tool protocol that LangChain memory doesn't plug into.

---

## Recap

Strands agents are stateless by default. `hindsight-strands` adds three native memory tools — retain, recall, reflect — that let your agent manage its own long-term memory using the same `@tool` protocol it uses for everything else. `memory_instructions()` handles the case where you want context pre-loaded rather than retrieved on demand.

The mental model: memory banks are scoped by `bank_id`, tags let you filter across environments, and `budget` controls the recall depth. Everything else is just a Strands agent with extra tools.

---

## Next Steps

- [hindsight-strands on PyPI](https://pypi.org/project/hindsight-strands/)
- [Hindsight docs: Strands integration guide](https://hindsight.vectorize.io/sdks/integrations/strands)
- Run Hindsight locally with Docker and try the quick start above
- Explore `memory_instructions()` for pre-warming agents in customer-facing applications
- Check out the [CrewAI](https://hindsight.vectorize.io/blog/2026-03-02-crewai) and [LangGraph](https://hindsight.vectorize.io/blog/2026-03-24-langgraph-longterm-memory) integration posts for comparison on similar patterns
