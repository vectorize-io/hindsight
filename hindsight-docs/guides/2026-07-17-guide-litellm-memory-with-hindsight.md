---
title: "Guide: Add LiteLLM Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, litellm, agents, memory]
description: "Add LiteLLM memory with Hindsight using the hindsight-litellm package, so every completion recalls relevant memories before the LLM call and stores the conversation after."
image: /img/guides/guide-litellm-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add LiteLLM Memory with Hindsight](/img/guides/guide-litellm-memory-with-hindsight.svg)

If you want **LiteLLM memory with Hindsight**, the cleanest setup is the `hindsight-litellm` package. It sits in front of your LiteLLM completions, injects relevant memories into the prompt before each call, and stores the conversation afterward. That gives any LLM application long-term memory across sessions instead of forcing every new call to start from a blank slate.

This is a good fit for LiteLLM because LiteLLM already normalizes 100+ providers behind one `completion()` interface. The `hindsight-litellm` package adds a memory layer at the same seam: it works with OpenAI, Anthropic, Groq, Azure, AWS Bedrock, Google Vertex AI, and any other LiteLLM-supported provider, with just a few lines of setup. Memory is scoped per bank, so different agents or users stay isolated.

This guide walks through installing the package, pointing it at your Hindsight backend, setting your bank and memory mode, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-litellm`.
> 2. `hindsight_litellm.configure(hindsight_api_url="http://localhost:8888")` (add `api_key=...` for Cloud auth).
> 3. `hindsight_litellm.set_defaults(bank_id="my-agent", use_reflect=True)`.
> 4. `hindsight_litellm.enable()`, then call `hindsight_litellm.completion(...)` as usual.
> 5. Verify that a later call recalls what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- Python >= 3.10 and `litellm` >= 1.83.0
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server
- Credentials for at least one LLM provider that LiteLLM supports

## Step 1: Install the package

```bash
pip install hindsight-litellm
```

The package wraps LiteLLM's `completion()` so memory is recalled before each call and the conversation is stored after. It does not change how you talk to your LLM providers.

## Step 2: Configure and point at Hindsight

Use `configure()` for the static settings that don't change during a session, including the Hindsight API URL and (optionally) an API key:

```python
import hindsight_litellm

hindsight_litellm.configure(
    hindsight_api_url="http://localhost:8888",  # Hindsight API server URL
    api_key="your-api-key",                     # optional — for Hindsight authentication
    verbose=True,                               # optional — verbose logging and debug info
)
```

For [Hindsight Cloud](https://hindsight.vectorize.io), set the Cloud API URL and pass your key as `api_key`. For a self-hosted server, point `hindsight_api_url` at your instance (for example `http://localhost:8888`).

## Step 3: Set your bank and memory mode

Use `set_defaults()` for per-call defaults. A `bank_id` is required — it selects the memory bank, which is how memory stays isolated per agent or user:

```python
hindsight_litellm.set_defaults(
    bank_id="my-agent",   # required — memory bank ID
    use_reflect=True,     # reflect = synthesized context; False = raw recall
)
```

Choose a memory mode:

- **Recall mode** (`use_reflect=False`, the default): retrieves raw memory facts and injects them as a numbered list. Best when you need precise, individual memories.
- **Reflect mode** (`use_reflect=True`): synthesizes memories into a coherent context paragraph. Best for natural, conversational memory context.

You can also tune retrieval with defaults like `budget` (`"low"`, `"mid"`, `"high"`), `max_memories`, `max_memory_tokens`, and `fact_types`. See the [LiteLLM integration docs](https://hindsight.vectorize.io/docs/integrations/litellm) for the full list.

## Step 4: Enable and call completion

Enable the integration, then call `completion()` exactly as you would with LiteLLM:

```python
hindsight_litellm.enable()

response = hindsight_litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What did we discuss about AI?"}],
    hindsight_query="What do I know about AI discussions?",
)
```

When `inject_memories=True` (the default), you can pass `hindsight_query` to specify what to search for in memory. If you omit it, the last user message is used as the query.

Because LiteLLM normalizes providers, the same call works everywhere — just change `model`:

```python
messages = [{"role": "user", "content": "Hello!"}]

hindsight_litellm.completion(model="gpt-4o", messages=messages, hindsight_query="greeting")
hindsight_litellm.completion(model="claude-sonnet-4-20250514", messages=messages, hindsight_query="greeting")
hindsight_litellm.completion(model="groq/llama-3.1-70b-versatile", messages=messages, hindsight_query="greeting")
```

## How the integration uses memory

When you call `completion()`, the following happens automatically:

- **Memory retrieval (before):** Hindsight is queried for relevant memories based on the conversation (or your `hindsight_query`).
- **Prompt injection (before):** memories are injected into the prompt — by default into the system message.
- **LLM call:** the enriched prompt is sent to the LLM through LiteLLM.
- **Conversation storage (after):** the conversation is stored to Hindsight for future recall — async by default for performance.

You can also call the memory APIs directly without going through `completion()`: `recall()` for raw memories, `reflect()` for synthesized context, and `retain()` to store content. Each has an async variant (`arecall()`, `areflect()`, `aretain()`). For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Per-bank memory

Memory is scoped by the `bank_id` you set in `set_defaults()`. That keeps one agent's or user's memory isolated from another's. You can override the bank on a single call with the `hindsight_bank_id` kwarg, which is handy for multi-user applications where each user gets their own bank.

To shape what a bank learns and remembers over time, use `set_bank_mission()` to give it a mission — Hindsight uses that to build mental models for the bank.

## Verify that memory is working

A good test sequence is:

1. call `hindsight_litellm.completion(...)` and share a fact or preference in the message
2. let the conversation storage complete
3. call `completion()` again with a question about that earlier fact
4. confirm the answer reflects what you stored

For example:

- the first call mentions "I'm working on a machine learning project in Python"
- a later call asks "What projects am I working on?"

If the later response recalls the machine learning project, the setup is working. With `verbose=True`, you can also call `get_last_injection_debug()` to inspect exactly what was injected — the mode, whether memory was injected, the number of results, and the memory context itself.

## Common mistakes

### Forgetting to call `enable()`

`configure()` and `set_defaults()` only prepare the integration. You must call `enable()` before `completion()` will inject and store memory.

### Not setting a `bank_id`

`bank_id` is required in `set_defaults()`. Without it, there is no bank to recall from or store to.

### Testing recall before storage finishes

Conversation storage is async by default. If you check immediately, the previous conversation may not be stored yet. Set `sync_storage=True` in `configure()` when you need storage to block and surface errors immediately.

### Expecting the wrong memory shape

Recall mode injects raw facts as a list; reflect mode injects a synthesized paragraph. Pick the mode that matches how you want the LLM to use memory.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — point `hindsight_api_url` at your instance.

### Does this change how I use LiteLLM?

No. You call `hindsight_litellm.completion()` with the same `model` and `messages` you already use. The memory layer is added around the call.

### Which providers are supported?

Any provider LiteLLM supports — OpenAI, Anthropic, Groq, Azure, AWS Bedrock, Google Vertex AI, and more. You only change the `model` string.

### Can I use it without the LiteLLM callback path?

Yes. There are native client wrappers, `wrap_openai()` and `wrap_anthropic()`, that add the same memory behavior directly to those SDK clients.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [LiteLLM integration docs](https://hindsight.vectorize.io/docs/integrations/litellm)
