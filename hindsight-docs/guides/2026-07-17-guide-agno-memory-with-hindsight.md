---
title: "Guide: Add Agno Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, agno, agents, memory]
description: "Add Agno memory with Hindsight using the hindsight-agno toolkit, so your agents recall relevant facts and retain what they learn across sessions."
image: /img/guides/guide-agno-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add Agno Memory with Hindsight](/img/guides/guide-agno-memory-with-hindsight.svg)

If you want **Agno memory with Hindsight**, the cleanest setup is the `hindsight-agno` toolkit. It extends Agno's native `Toolkit` base class and gives your agent three memory tools — retain, recall, and reflect — so the agent can store what it learns, search for relevant facts, and reason over its own memory across sessions.

This is a good fit for Agno because Agno agents are already tool-driven. Instead of bolting on a separate memory system, you add `HindsightTools` to the agent's `tools` list exactly like any other toolkit, and the model decides when to store or recall. You can also pre-recall memories and inject them into the system prompt with `memory_instructions`, so context is present before the first turn.

This guide walks through installing the toolkit, pointing it at your Hindsight backend, wiring per-user memory banks, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-agno`.
> 2. Set `HINDSIGHT_API_KEY` (Cloud) or point `hindsight_api_url` at your self-hosted server.
> 3. Add `HindsightTools(bank_id="user-123", hindsight_api_url="...")` to the agent's `tools` list.
> 4. The bank is resolved from `bank_id`, a custom resolver, or `RunContext.user_id`.
> 5. Verify that a later run recalls what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- Python >= 3.10 and `agno` installed
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server
- A model configured for your Agno agent (for example `OpenAIChat`)

## Step 1: Install the toolkit

Install the toolkit alongside Agno.

```bash
pip install hindsight-agno
```

`HindsightTools` extends Agno's native `Toolkit` base class, just like `Mem0Tools`, so it drops into an existing agent without changing how Agno works.

## Step 2: Point the toolkit at Hindsight

For Hindsight Cloud, set your API key as an environment variable (or pass `api_key=` directly):

```bash
export HINDSIGHT_API_KEY="hsk_..."
```

Then add the toolkit to your agent:

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from hindsight_agno import HindsightTools

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[HindsightTools(
        bank_id="user-123",
        hindsight_api_url="https://api.hindsight.vectorize.io",
        api_key="hsk_...",  # or set HINDSIGHT_API_KEY env var
    )],
)

agent.print_response("Remember that I prefer dark mode")
agent.print_response("What are my preferences?")
```

For a self-hosted Hindsight server, swap the URL to your local endpoint (for example `http://localhost:8888` when running with `./scripts/dev/start-api.sh`).

## Step 3: Configure once globally (optional)

Instead of passing connection details to every toolkit, configure them once:

```python
from hindsight_agno import configure, HindsightTools

configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="your-api-key",       # Or set HINDSIGHT_API_KEY env var
    budget="mid",                  # Recall budget: low/mid/high
    max_tokens=4096,               # Max tokens for recall results
    tags=["env:prod"],             # Tags for stored memories
    recall_tags=["scope:global"],  # Tags to filter recall
    recall_tags_match="any",       # Tag match mode: any/all/any_strict/all_strict
)

# Now create the toolkit without passing connection details
tools = [HindsightTools(bank_id="user-123")]
```

## How the toolkit uses memory

Adding `HindsightTools` gives the agent three tools it can call on its own:

- **`retain_memory`** — store information to long-term memory
- **`recall_memory`** — search long-term memory for relevant facts
- **`reflect_on_memory`** — synthesize a reasoned answer from memories

You can include only the tools you need by toggling `enable_retain`, `enable_recall`, and `enable_reflect`:

```python
tools = [HindsightTools(
    bank_id="user-123",
    hindsight_api_url="https://api.hindsight.vectorize.io",
    enable_retain=True,
    enable_recall=True,
    enable_reflect=False,  # Omit reflect
)]
```

If you want memory present before the first turn, use `memory_instructions` to pre-recall relevant memories and inject them into the system prompt:

```python
from hindsight_agno import HindsightTools, memory_instructions

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[HindsightTools(
        bank_id="user-123",
        hindsight_api_url="https://api.hindsight.vectorize.io",
    )],
    instructions=[memory_instructions(
        bank_id="user-123",
        hindsight_api_url="https://api.hindsight.vectorize.io",
    )],
)
```

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Per-user memory banks

The bank ID is resolved in this order:

1. **`bank_resolver`** — a custom callable `(RunContext) -> str`
2. **`bank_id`** — a static bank ID passed to the constructor
3. **`run_context.user_id`** — automatic per-user banks

That means you can give each user their own isolated memory by passing `user_id` to the agent, or share a bank across a team with a custom resolver:

```python
# Per-user banks from RunContext
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[HindsightTools(hindsight_api_url="https://api.hindsight.vectorize.io")],
    user_id="user-123",  # Used as bank_id
)

# Custom resolver
def resolve_bank(ctx):
    return f"team-{ctx.user_id}"

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[HindsightTools(
        bank_resolver=resolve_bank,
        hindsight_api_url="https://api.hindsight.vectorize.io",
    )],
)
```

## Verify that memory is working

A good test sequence is:

1. create an agent with `HindsightTools`
2. tell it something worth remembering
3. start a fresh run against the same bank
4. ask about the earlier fact

For example:

```python
agent.print_response("Remember that I prefer dark mode")
agent.print_response("What are my preferences?")
```

If the second response surfaces the earlier preference, the setup is working.

## Common mistakes

### Not providing a bank

If no `bank_resolver`, `bank_id`, or `user_id` resolves, the toolkit has no bank to read or write. Make sure at least one of the three is set.

### Expecting cross-bank recall

Each bank is isolated. Memory stored under one bank ID will not surface for a different one, so keep your bank strategy consistent across runs.

### Forgetting the API key

For Hindsight Cloud, set `HINDSIGHT_API_KEY` or pass `api_key=` to the toolkit or `configure()`. Without it, calls to Cloud will not authenticate.

### Disabling the tool you need

`enable_retain`, `enable_recall`, and `enable_reflect` default to `True`. If you turn one off, the agent loses that capability — for example disabling recall means it can store but never search.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — point `hindsight_api_url` at your own server instead of the Cloud URL.

### Does this change how I use Agno?

No. `HindsightTools` extends Agno's native `Toolkit`, so you add it to the agent's `tools` list like any other toolkit.

### How is memory scoped?

Per bank. The bank is resolved from `bank_resolver`, then `bank_id`, then `RunContext.user_id`.

### Can I control which tools the agent gets?

Yes. Use `enable_retain`, `enable_recall`, and `enable_reflect` to include only the tools you want.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Agno integration docs](https://hindsight.vectorize.io/docs/integrations/agno)
