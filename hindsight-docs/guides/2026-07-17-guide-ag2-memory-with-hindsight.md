---
title: "Guide: Add AG2 Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, ag2, agents, memory]
description: "Add AG2 memory with Hindsight using the hindsight-ag2 package, which registers retain, recall, and reflect tools on your AG2 agents so they remember across conversations."
image: /img/guides/guide-ag2-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add AG2 Memory with Hindsight](/img/guides/guide-ag2-memory-with-hindsight.svg)

If you want **AG2 memory with Hindsight**, the cleanest setup is the `hindsight-ag2` package. It registers three Hindsight-backed tools — retain, recall, and reflect — directly on your AG2 agents with a single call. That gives your agents long-term memory across conversations instead of forgetting everything the moment a chat ends.

This is a good fit for AG2 (the community AutoGen fork) because AG2 already has a tool-calling pattern built in. The tools are plain Python functions with `Annotated` type hints, so AG2 uses those hints to generate the schema the LLM sees. The agent decides when to store a fact, search past memory, or reason over everything it knows.

This guide walks through installing the package, registering the tools, sharing a memory bank across a GroupChat, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-ag2`.
> 2. Point it at Hindsight with `hindsight_api_url` (self-hosted) or an API key (Cloud).
> 3. Call `register_hindsight_tools(assistant, user_proxy, bank_id="my-bank")`.
> 4. The agent now has `hindsight_retain`, `hindsight_recall`, and `hindsight_reflect`.
> 5. Verify that a later conversation recalls what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- Python 3.10 or newer
- AG2 installed (`ag2 >= 0.9.0`)
- A running Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server

## Step 1: Install the package

Install the integration package.

```bash
pip install hindsight-ag2
```

This gives you `register_hindsight_tools` and `create_hindsight_tools`, which wire Hindsight's retain, recall, and reflect operations into AG2's tool system.

## Step 2: Register the tools on your agents

Register the Hindsight tools on both the assistant and the executing agent in one call:

```python
from autogen import AssistantAgent, UserProxyAgent, LLMConfig
from hindsight_ag2 import register_hindsight_tools

llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")

with llm_config:
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant with long-term memory.",
    )
    user_proxy = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
    )

# Register Hindsight memory tools on both agents
register_hindsight_tools(
    assistant, user_proxy,
    bank_id="my-bank",
    hindsight_api_url="http://localhost:8888",
)

result = user_proxy.initiate_chat(
    assistant,
    message="Remember that I prefer Python over JavaScript.",
)
```

The assistant can now use `hindsight_retain`, `hindsight_recall`, and `hindsight_reflect`.

## Step 3: Configure the connection

You can configure the connection once globally instead of passing it on every call:

```python
from hindsight_ag2 import configure

configure(
    hindsight_api_url="http://localhost:8888",
    api_key="your-key",       # or set HINDSIGHT_API_KEY env var
    budget="mid",              # low / mid / high
    max_tokens=4096,
    tags=["source:ag2"],       # default tags for retain
)
```

Constructor arguments override the global configuration, so you can set defaults once and adjust `budget`, `max_tokens`, or `tags` per tool set when needed.

## How the tools use memory

The integration provides three AG2-compatible tool functions backed by Hindsight's API:

- **`hindsight_retain(content)`** stores content. Hindsight extracts facts, entities, and relationships from the raw text.
- **`hindsight_recall(query)`** runs semantic search, BM25, graph traversal, and reranking, then returns a numbered list of matching memories.
- **`hindsight_reflect(query)`** synthesizes a reasoned answer from all relevant memories, using the bank's disposition traits.

Because these are plain Python functions with `Annotated` type hints, AG2 generates the tool schema the LLM sees automatically. The agent chooses when to call each one.

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Shared memory in a GroupChat

Multiple agents can share a single memory bank by registering the tools with the same `bank_id`:

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, LLMConfig
from hindsight_ag2 import register_hindsight_tools

llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")

with llm_config:
    researcher = AssistantAgent(name="researcher", system_message="You research topics.")
    writer = AssistantAgent(name="writer", system_message="You write content.")
    executor = UserProxyAgent(name="executor", human_input_mode="NEVER")

# All agents share the same memory bank
for agent in [researcher, writer]:
    register_hindsight_tools(agent, executor, bank_id="team-memory")

group_chat = GroupChat(agents=[researcher, writer, executor], messages=[])
manager = GroupChatManager(groupchat=group_chat)
```

Every agent that registers `bank_id="team-memory"` reads from and writes to the same shared memory, so a fact one agent learns is available to the others.

## Verify that memory is working

A good test sequence is:

1. register the tools and start a chat
2. tell the agent something to remember, so it calls `hindsight_retain`
3. end that conversation
4. start a fresh conversation with the same `bank_id`
5. ask the agent about what you told it earlier

For example:

- conversation one says "Remember that I prefer Python over JavaScript."
- conversation two asks "What language do I prefer?"

If the agent recalls the earlier preference, the setup is working.

## Common mistakes

### Using a different bank between runs

Memory is scoped to `bank_id`. If you change the bank between conversations, the second run will not recall what the first one stored.

### Forgetting to register on the executor too

`register_hindsight_tools` takes both the LLM-side agent and the executing agent. The LLM-side agent proposes the tool call and the executor runs it, so both need the tools registered.

### Not pointing at a running backend

The tools call a live Hindsight API. Make sure `hindsight_api_url` (or your Cloud credentials) points at a reachable server before running.

### Expecting recall without asking

The agent decides when to call the tools. If it never recalls, prompt it to look up what it knows, or rely on `reflect` to reason over stored memory.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — set `hindsight_api_url` to your server, for example `http://localhost:8888`.

### Can multiple agents share memory?

Yes. Register the tools on each agent with the same `bank_id` and they draw from one shared bank, which is how the GroupChat pattern works.

### Can I install only some of the tools?

Yes. `create_hindsight_tools` accepts `include_retain`, `include_recall`, and `include_reflect` so you can register only the tools you need.

### How do I control recall depth?

Set `budget` (low / mid / high) and `max_tokens`, either globally via `configure(...)` or per tool set via `create_hindsight_tools(...)`.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [AG2 integration docs](https://hindsight.vectorize.io/docs/integrations/ag2)
