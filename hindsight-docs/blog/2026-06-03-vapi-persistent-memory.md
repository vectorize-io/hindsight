---
title: "Voice Agents That Remember: Adding Memory to Vapi with Hindsight"
authors: [benfrank241]
slug: "2026/06/03/vapi-persistent-memory"
date: 2026-06-03T12:00
tags: [memory, voice, ai, vapi, phone, agents, webhook, tutorial, hindsight]
description: "Add persistent long-term memory to Vapi voice AI calls with one webhook. Recall caller history at call start, retain the transcript at call end — no per-turn integration needed."
image: /img/blog/vapi-persistent-memory.png
hide_table_of_contents: true
---

![Voice Agents That Remember: Vapi + Hindsight](/img/blog/vapi-persistent-memory.png)

[Vapi](https://vapi.ai) makes it remarkably easy to ship a production phone agent — managed STT, streaming TTS, telephony, transfers, and an LLM loop, all behind a single dashboard and API. The part it leaves to you is the memory.

Every Vapi call starts from zero. The caller who explained their account problem yesterday explains it again today. The customer who booked an appointment three weeks ago has to spell their name again. The agent that diagnosed a routing issue last Tuesday rediscovers it on Wednesday. Persistent memory across calls isn't a feature you can turn on inside Vapi — but it's a feature you can drop in with a webhook.

The `hindsight-vapi` integration adds long-term memory to any Vapi assistant with one HTTP endpoint. Memories are recalled at the start of every call and the transcript is retained at the end — no per-turn integration, no changes to your assistant config, no custom tool wiring.

<!-- truncate -->

## TL;DR

- Vapi voice agents have no built-in cross-call memory.
- `hindsight-vapi` is a webhook handler — one endpoint covers both directions.
- On `assistant-request`, it recalls memories for the caller and returns them as `assistantOverrides` (injected into the system prompt).
- On `end-of-call-report`, it retains the full transcript fire-and-forget.
- No per-turn hooks. No latency added mid-call. Memory is injected **once per call**, at call start.

---

## The Problem: Vapi Calls Start From Scratch

Vapi gives you a clean abstraction over the voice-agent stack — speech-to-text, an LLM loop, text-to-speech, call control, telephony providers. What it doesn't give you is a memory layer that survives between calls. Each call gets a fresh LLM context. The transcript is yours to keep, but the agent can't reach into it on the next call without you wiring something up.

You could solve this with a custom tool the agent calls during the conversation ("look up the caller's history before responding"), but that adds latency to the first turn, and the LLM has to remember to call it. You could prepend a summary to every assistant's system prompt, but that requires you to know which assistant a given caller will hit, and to keep the summary fresh as new calls land.

The cleaner solution is to use Vapi's existing server-webhook events. Vapi already fires an HTTP event at call start (`assistant-request`) and another at call end (`end-of-call-report`). If you handle those two, you have natural insertion points for recall and retention — without touching the assistant config, without adding mid-call latency, and without depending on the LLM remembering to call a tool.

That's what `hindsight-vapi` does.

---

## How It Works

Vapi doesn't expose a per-turn hook in its server architecture. So memory is injected **once per call** at call start, then the call runs as usual, and the transcript is retained at the end.

```text
Incoming call
  └─ Vapi fires "assistant-request" webhook
       └─ Recall memories (query = caller's phone number)
            └─ Return as assistantOverrides with <hindsight_memories> system message
                 └─ Vapi merges into assistant config before the call begins

Call runs normally
  (no per-turn hooks — LLM uses the injected memory throughout the call)

Call ends
  └─ Vapi fires "end-of-call-report" webhook
       └─ Retain full transcript (fire-and-forget — webhook responds immediately)
```

What actually happens on `assistant-request`:

1. **Recall.** The webhook pulls the caller's phone number out of `message.call.customer.number` and queries Hindsight for relevant memories. (If there's no phone number — uncommon, but possible with web-call assistants — it falls back to a generic "returning caller" query.)
2. **Inject.** The recalled memories are wrapped in a `<hindsight_memories>` block and returned as `assistantOverrides.model.messages` — a system message Vapi merges into the active assistant config before the LLM ever generates a token.
3. **Run the call.** The LLM has the caller's history in its context from word one.

What happens on `end-of-call-report`:

1. **Webhook responds immediately.** Vapi expects a quick 200; the integration returns one.
2. **Retention runs in the background.** The full transcript is sent to Hindsight via `asyncio.create_task` — fire-and-forget, no blocking, no risk of failing the webhook because retention is slow.

Memory accumulates across calls. By the second or third call from the same caller, recall surfaces real history — past decisions, account details, stated preferences — automatically.

---

## Setup: One Webhook, Done

The integration is a thin handler you mount on any HTTP server. Here's the FastAPI version end-to-end:

### 1. Install

```bash
pip install hindsight-vapi
```

### 2. Pick a Hindsight Deployment

**Hindsight Cloud** is the fastest path — [sign up free](https://ui.hindsight.vectorize.io/signup), grab an API key, point your code at `https://api.hindsight.vectorize.io`. No daemon to run.

**Self-hosted** is a single command:

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=YOUR_OPENAI_KEY
hindsight-api  # starts at http://localhost:8888
```

### 3. Wire the Webhook

```python
from fastapi import FastAPI, Request
from hindsight_vapi import HindsightVapiWebhook

app = FastAPI()

memory = HindsightVapiWebhook(
    bank_id="vapi-support",
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_your_token_here",
)

@app.post("/webhook")
async def vapi_webhook(request: Request):
    event = await request.json()
    response = await memory.handle(event)
    return response or {}
```

That's the whole integration. One handler covers both directions — recall on `assistant-request`, retain on `end-of-call-report`, everything else returns 200 OK.

### 4. Point Vapi at It

In the Vapi dashboard:

1. **Settings → Server URL** — paste your endpoint (e.g., `https://your-domain.com/webhook`)
2. Enable the `assistant-request` and `end-of-call-report` event types

See [Vapi's server-events docs](https://docs.vapi.ai/server-url) for tunneling tips during local dev (`ngrok`, `cloudflared`, etc.).

That's it. The next inbound call to that assistant goes through Hindsight on both ends.

---

## Outbound Calls

Vapi's `assistant-request` webhook only fires for **inbound** calls. For outbound, you need to build the memory overrides yourself at call-creation time:

```python
overrides = await memory.build_assistant_overrides("Ben from Vectorize")

vapi.calls.create(
    assistant_id="...",
    assistant_overrides=overrides,
    customer={"number": "+15555550100"},
)
```

`build_assistant_overrides()` runs the same recall logic as the inbound path and returns the same `assistantOverrides` payload — just on demand, with a query you provide (caller name, account ID, call topic). The retention side is unchanged: whenever Vapi delivers an `end-of-call-report` to your webhook (inbound or outbound), the handler retains the transcript automatically.

---

## Configuration

Everything is tunable on the constructor:

```python
HindsightVapiWebhook(
    bank_id="vapi-support",            # Required: memory bank to read/write
    hindsight_api_url="...",           # Hindsight API URL
    api_key="hsk_...",                 # Hindsight Cloud token
    recall_budget="mid",               # "low", "mid", or "high"
    recall_max_tokens=4096,            # Cap on tokens in the recall block
    enable_recall=True,                # Inject memories at call start
    enable_retain=True,                # Store transcript at call end
    memory_prefix="Relevant memories from past conversations:\n",
)
```

**`recall_budget`** is the lever to watch. Voice calls are latency-sensitive — but because memory injection happens **once at call start** rather than per turn, you've got more headroom than a streaming pipeline. `"mid"` is the right default for most assistants. Drop to `"low"` if your Vapi assistant config is already tight on the first-response budget; bump to `"high"` if your assistant handles complex multi-thread conversations where deeper recall pays off.

**Selective enable/disable** is useful for staged rollouts. Run recall-only first to surface what's already in the bank, then flip retention on once you're happy:

```python
memory = HindsightVapiWebhook(
    bank_id="vapi-support",
    enable_recall=True,
    enable_retain=False,   # Don't write yet — just observe
)
```

**Global configuration** for multiple assistants:

```python
from hindsight_vapi import configure

configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",
    recall_budget="mid",
)

# Per-assistant webhooks just need bank_id
support_memory = HindsightVapiWebhook(bank_id="vapi-support")
booking_memory = HindsightVapiWebhook(bank_id="vapi-booking")
```

---

## Bank Scoping: Who Shares Memory With Whom?

The `bank_id` is the lever that controls what each caller can see.

**Per-caller** — typical for personal-assistant / customer-support setups. Each phone number gets its own bank, no cross-contamination:

```python
caller_number = event["message"]["call"]["customer"]["number"]
memory = HindsightVapiWebhook(bank_id=f"user-{caller_number}")
```

**Per-assistant** — useful when one Vapi assistant handles a shared workload (e.g., a reception desk that books across multiple business lines):

```python
memory = HindsightVapiWebhook(bank_id="vapi-reception")
```

**Per-account** — your own internal customer or account ID, looked up from the phone number before instantiating the webhook handler. This is the right pattern when one customer might call from multiple numbers.

You don't have to pick once. A receptionist + per-customer setup is a perfectly reasonable mix — a shared reception bank for general info plus per-customer banks for individual histories — by running two webhooks against different paths.

---

## Where Vapi Memory Shines

The integration is well suited to anything where the caller relationship outlives the call:

- **Customer support phone lines** — the agent picks up where the last call left off. Past tickets, what worked, who was on the other end of the previous call, the resolution status.
- **Sales and qualification** — agents remember stated budgets, decision timelines, and which products the caller has already asked about. No more "tell me again what you're looking for."
- **Scheduling and booking** — the agent remembers the caller's preferences (morning vs. evening, in-person vs. video, recurring patterns) and can suggest the right slot without re-discovery.
- **Healthcare intake and follow-ups** — symptom history, medication notes, and prior intake answers carry across calls without manual note review (within whatever compliance posture you're already running).
- **Field-service dispatch** — repeat callers about ongoing issues skip the "what's your account number" intro and the agent already knows the open work order.

The common thread: the value of memory shows up the **second** time someone calls. That's also when it's most painful for them not to have it.

---

## Production Notes

A few things to know once you're past the prototype:

**Latency.** Recall on `assistant-request` runs before Vapi starts the call, so it doesn't add to per-turn latency once the call is live. Expect recall to add roughly 50–300 ms depending on `recall_budget` and memory size — and because it happens once at the start of the call rather than mid-conversation, the budget is forgiving.

**Privacy and retention.** If you need to delete memories for a specific caller (right-to-be-forgotten, account closure), the Hindsight Python client exposes `delete_bank` / `adelete_bank`, and the REST API serves the same operation under `DELETE /v1/default/banks/{bank_id}`. If you key banks by phone number or account ID, deletion is a single call.

**Failure isolation.** Both recall and retention swallow exceptions internally and log them — a failing Hindsight call never breaks the Vapi call. If recall errors, the assistant just runs without injected memories (no worse than not having the integration). If retention errors, you lose that one transcript; the next call still works fine.

**Multiple events.** The handler ignores event types it doesn't care about (call status changes, function calls, transcript-while-running, etc.) and returns `None`, which FastAPI converts to a 200 OK with empty body. You can mount your own logic for those events alongside the memory handler — they don't conflict.

---

## Recap

Vapi voice agents don't remember anything between calls — by default. Adding persistent memory doesn't require rebuilding your assistant, threading a tool into every conversation, or running a sidecar that watches the call. It requires one webhook handler that:

- **Recalls** the caller's history once at call start, injected into the system prompt via `assistantOverrides`
- **Retains** the full transcript once at call end, fire-and-forget

Two events. One endpoint. Memory that compounds across every call to your Vapi assistant.

---

## Next Steps

- **Try it on a free Hindsight Cloud account**: [sign up](https://ui.hindsight.vectorize.io/signup), grab a key, point your FastAPI webhook at the Cloud URL
- **Run the local example**: `python examples/interactive_webhook.py --bank demo-user` in `hindsight-integrations/vapi/` simulates Vapi events end-to-end (with `:script`, `:end <transcript>`, `:call <number>`, `:memories` commands)
- **Tune `recall_budget`**: start with `"mid"`, drop to `"low"` if you're optimizing first-response latency
- **Pick a bank-scoping strategy** that matches your customer model — per-caller, per-assistant, or per-account
- **Browse the [full integration list](https://hindsight.vectorize.io/integrations/)** — Hermes, OpenAI Agents, n8n, Paperclip, and 30+ others all have the same memory layer underneath

---

**Further reading:**

- [What Is Agent Memory?](https://vectorize.io/what-is-agent-memory/) — foundational concepts
- [Hindsight Vapi Integration docs](https://hindsight.vectorize.io/integrations/vapi) — full configuration reference
- [Best AI Agent Memory Systems in 2026](https://vectorize.io/articles/best-ai-agent-memory-systems/) — full landscape comparison
