
# Vapi

Persistent long-term memory for [Vapi](https://vapi.ai) voice AI calls via [Hindsight](https://vectorize.io/hindsight). A single webhook handler recalls relevant memories at call start (injected as `assistantOverrides`) and retains the full transcript when the call ends.

## Quick Start

> **💡 Hindsight Cloud (recommended)**
>
[Sign up free](https://ui.hindsight.vectorize.io/signup) — get an API key instantly, no infrastructure to run.
```bash
pip install hindsight-vapi
```

Wire it into any HTTP server. FastAPI example with Hindsight Cloud:

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

Point Vapi's **Server URL** at your webhook endpoint and memory is active.

**Self-hosting alternative:** [install Hindsight locally](../../developer/installation.md) and use `hindsight_api_url="http://localhost:8888"` (omit `api_key`).

## How It Works

Unlike the Pipecat integration (per-turn `FrameProcessor`), Vapi doesn't expose a per-turn hook, so memory is injected **once per call** at call start:

```
Incoming call
  └─ Vapi fires "assistant-request" webhook
       └─ Recall memories (query = caller's phone number)
            └─ Return as assistantOverrides with <hindsight_memories> system message
                 └─ Vapi merges into assistant config before the call begins

Call ends
  └─ Vapi fires "end-of-call-report" webhook
       └─ Retain full transcript (fire-and-forget — webhook responds immediately)
```

Memory accumulates across calls. By the second or third call with the same caller, Hindsight surfaces relevant history automatically — previous decisions, account details, stated preferences.

## Vapi Server URL Setup

In the Vapi dashboard:

1. Go to **Settings → Server URL**
2. Point it at your webhook endpoint (e.g., `https://your-domain.com/webhook`)
3. Enable the `assistant-request` and `end-of-call-report` event types

See [Vapi's server events docs](https://docs.vapi.ai/server-url) for details.

## Outbound Calls

There is no `assistant-request` webhook for outbound calls. Use `build_assistant_overrides()` at call-creation time:

```python
overrides = await memory.build_assistant_overrides("Ben from Vectorize")
vapi.calls.create(
    assistant_id="...",
    assistant_overrides=overrides,
    customer={"number": "+15555550100"},
)
```

## Configuration

```python
HindsightVapiWebhook(
    bank_id="user-123",              # Required: memory bank to use
    hindsight_api_url="...",         # Hindsight API URL
    api_key="hsk_...",               # API key (Hindsight Cloud)
    recall_budget="mid",             # "low", "mid", or "high"
    recall_max_tokens=4096,          # Max tokens for recall results
    enable_recall=True,              # Inject memories at call start
    enable_retain=True,              # Store transcript at call end
    memory_prefix="Relevant memories from past conversations:\n",
)
```

### Global Configuration

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

## Bank Scoping

Typical patterns for the `bank_id`:

- **One bank per user** — scope by phone number (`user-+15551234567`) or your own account ID
- **Shared bank** — one bank for all callers (useful for small teams or shared memory)
- **Per-assistant** — if you have multiple Vapi assistants with different personalities or scopes

## Prerequisites

A running Hindsight instance:

**Self-hosted:**
```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-api-key
hindsight-api  # starts on http://localhost:8888
```

**Hindsight Cloud:** [Sign up](https://ui.hindsight.vectorize.io/signup) — no self-hosting required.
