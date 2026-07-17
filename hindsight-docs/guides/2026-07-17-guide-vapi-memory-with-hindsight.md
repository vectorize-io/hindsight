---
title: "Guide: Add Vapi Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, vapi, voice, memory]
description: "Add Vapi memory with Hindsight using the hindsight-vapi webhook handler, so voice AI calls recall caller context at call start and retain the transcript when the call ends."
image: /img/guides/guide-vapi-memory-with-hindsight.svg
hide_table_of_contents: true
---

![Guide: Add Vapi Memory with Hindsight](/img/guides/guide-vapi-memory-with-hindsight.svg)

If you want **Vapi memory with Hindsight**, the cleanest setup is the `hindsight-vapi` webhook handler. A single handler recalls relevant memories at call start and injects them as `assistantOverrides`, then retains the full transcript when the call ends. That gives your Vapi voice assistant long-term memory across calls instead of forcing every call to start from scratch.

This is a good fit for Vapi because Vapi doesn't expose a per-turn hook, but it does fire server webhooks around each call. The handler uses both events: it recalls memory on the `assistant-request` webhook and merges it into the assistant config before the call begins, then retains the transcript on the `end-of-call-report` webhook after the call ends. Memory is scoped per bank, so you control whether a caller's history is isolated per user or shared.

This guide walks through installing the package, wiring it into an HTTP server, pointing it at your Hindsight backend, understanding the bank scoping options, and a quick verification flow so you can confirm memory is actually being used. Keep the [docs home](https://hindsight.vectorize.io/docs) and the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart) nearby while you work.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-vapi`.
> 2. Create a `HindsightVapiWebhook` with a `bank_id`, `hindsight_api_url`, and `api_key` (Cloud) or a local URL (self-hosted).
> 3. Serve `memory.handle(event)` from a `/webhook` route in your HTTP server.
> 4. Point Vapi's **Server URL** at that endpoint and enable the `assistant-request` and `end-of-call-report` events.
> 5. Verify that a later call remembers what an earlier one stored.

## Prerequisites

Before you start, make sure you have:

- A Vapi account with an assistant and access to the dashboard **Server URL** setting
- A reachable Hindsight backend, [Hindsight Cloud](https://hindsight.vectorize.io) or a self-hosted server
- An HTTP server you can point Vapi at (the examples below use FastAPI)

## Step 1: Install the package

Install `hindsight-vapi` into the environment that runs your webhook server.

```bash
pip install hindsight-vapi
```

The package provides a `HindsightVapiWebhook` handler you wire into any HTTP server. It does not run its own server — you serve it from a route you already control.

## Step 2: Wire the handler into your server

Create a webhook handler and serve it from a POST route. A FastAPI example using Hindsight Cloud:

```python
from fastapi import FastAPI, Request
from hindsight_vapi import HindsightVapiWebhook

app = FastAPI()
memory = HindsightVapiWebhook(
    bank_id="user-123",
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_your_token_here",
)

@app.post("/webhook")
async def vapi_webhook(request: Request):
    event = await request.json()
    response = await memory.handle(event)
    return response or {}
```

For a self-hosted Hindsight server, point `hindsight_api_url` at your local instance and omit `api_key`:

```python
memory = HindsightVapiWebhook(
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
)
```

If you run many webhooks, you can set connection details once with `configure()` instead of repeating them:

```python
from hindsight_vapi import configure

configure(
    hindsight_api_url="http://localhost:8888",
    api_key="hsk_...",
    recall_budget="mid",
)

# Now create webhooks without repeating connection details
memory = HindsightVapiWebhook(bank_id="user-123")
```

## Step 3: Point Vapi at your webhook

In the Vapi dashboard:

1. Go to **Settings → Server URL**
2. Point it at your webhook endpoint (for example, `https://your-domain.com/webhook`)
3. Enable the `assistant-request` and `end-of-call-report` event types

See [Vapi's server events docs](https://docs.vapi.ai/server-url) for details. Once the Server URL is set, memory is active for inbound calls.

## How the handler uses memory

Vapi doesn't expose a per-turn hook, so memory is injected **once per call** at call start:

- **Recall (before):** when Vapi fires the `assistant-request` webhook, the handler recalls memories (query = the caller's phone number) and returns them as `assistantOverrides` with a `<hindsight_memories>` system message. Vapi merges that into the assistant config before the call begins.
- **Retain (after):** when Vapi fires the `end-of-call-report` webhook, the handler retains the full transcript. Retain is fire-and-forget, so the webhook responds immediately.

Memory accumulates across calls. By the second or third call with the same caller, Hindsight surfaces relevant history automatically — previous decisions, account details, stated preferences.

For lower-level behavior, read [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall) and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Outbound calls

There is no `assistant-request` webhook for outbound calls, so recall can't happen automatically. Build the overrides at call-creation time instead with `build_assistant_overrides()`:

```python
overrides = await memory.build_assistant_overrides("Ben from Vectorize")
vapi.calls.create(
    assistant_id="...",
    assistant_overrides=overrides,
    customer={"number": "+15555550100"},
)
```

## Bank scoping

You choose how memory is scoped through the `bank_id`. Common patterns:

- **One bank per user** — scope by phone number (`user-+15551234567`) or your own account ID
- **Shared bank** — one bank for all callers, useful for small teams or shared memory
- **Per-assistant** — separate banks if you have multiple Vapi assistants with different personalities or scopes

Pick the pattern that matches how you want history to flow between callers and assistants.

## Verify that memory is working

The package ships an interactive webhook simulator so you can test without a real Vapi account:

```bash
python examples/interactive_webhook.py --bank demo-user
```

It supports commands like `:script` (guided demo), `:end <transcript>`, `:call <number>`, `:memories`, and `:quit`.

A good end-to-end test sequence with a real assistant is:

1. place a call and let the assistant record a decision or preference
2. end the call so the transcript is retained
3. place a second call from the same number
4. ask about the earlier decision

If the second call surfaces what the first one stored, the setup is working.

## Common mistakes

### Forgetting to enable both events

Recall runs on `assistant-request` and retain runs on `end-of-call-report`. If you only enable one, half of the loop is missing.

### Expecting mid-call recall

Memory is injected once per call at call start. Vapi has no per-turn hook, so recall does not refresh during the call.

### Expecting automatic recall on outbound calls

Outbound calls have no `assistant-request` webhook. Use `build_assistant_overrides()` at call-creation time instead.

### Testing retain too early

The transcript is retained on the `end-of-call-report` webhook. If you check before the call ends, it may not have been stored yet.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — set `hindsight_api_url` to your local instance (for example `http://localhost:8888`) and omit `api_key`.

### Does the package run its own server?

No. `HindsightVapiWebhook` is a handler you serve from your own HTTP route. You call `memory.handle(event)` and return the response.

### How is memory scoped?

By the `bank_id` you set. You can use one bank per user (for example scoped by phone number), a shared bank for all callers, or a bank per assistant.

### How do I test without a real Vapi account?

Use the interactive webhook simulator in the package `examples/` directory: `python examples/interactive_webhook.py --bank demo-user`.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Vapi integration docs](https://hindsight.vectorize.io/docs/integrations/vapi)
