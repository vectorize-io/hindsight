---
title: "Guide: Voice Agents That Remember Callers Across Calls with Pipecat"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, pipecat, voice, memory]
description: "Give a Pipecat voice agent per-caller cross-call memory with Hindsight, so a returning caller never has to re-explain who they are."
image: /img/guides/guide-pipecat-voice-memory-across-calls.svg
hide_table_of_contents: true
---

![Guide: Voice Agents That Remember Callers Across Calls with Pipecat](/img/guides/guide-pipecat-voice-memory-across-calls.svg)

If you want a Pipecat voice agent to **remember callers across calls**, the trick is not a bigger prompt — it is a memory bank scoped to the caller. When the same person calls back next week, the agent should already know who they are, what they asked for last time, and what they prefer, instead of making them re-explain everything from scratch.

Pipecat handles the real-time audio and LLM orchestration; Hindsight handles the memory loop. A single `HindsightMemoryService` FrameProcessor sits between your user aggregator and LLM service, recalling relevant memories before each turn and retaining completed turns after. The piece most people miss is scoping: the bank ID has to be the *caller's* identity, not the call. Get that right and continuity across calls comes for free.

This is a strategy guide, not a fresh install. It assumes you have already wired `HindsightMemoryService` into a working pipeline (or will follow the setup guide linked below), and focuses on per-caller banks, what to retain, when to recall, and the voice-specific tradeoffs that decide whether recall helps or hurts.

<!-- truncate -->

> **Quick answer**
>
> 1. Derive a stable bank ID from the *caller*, not the call — e.g. the phone number or account ID.
> 2. Wire `HindsightMemoryService` between `user_aggregator` and `llm_service` with that `bank_id`.
> 3. Let recall run at the start of the call so the agent greets a returning caller with context.
> 4. Let retain fire-and-forget after each completed turn so the next call has this call's details.
> 5. Verify by calling twice from the same identity and confirming the second call remembers the first.

## Why voice agents forget returning callers

A stock voice pipeline is stateless between sessions. Each call spins up a fresh LLM context, so the moment a caller hangs up, everything they said is gone. The next call starts from zero: "Can I get your name and account number?" — again.

That is fine for a one-shot IVR, but it is exactly wrong for an agent that should feel like it knows the caller. The fix is not to stuff prior transcripts into the system prompt (they grow without bound and blow your latency budget). The fix is a memory store that recalls only what is relevant to the current turn and retains new facts as the call goes. Hindsight is that store, and `HindsightMemoryService` is the FrameProcessor that connects it to Pipecat.

## Scope a bank per caller

Memory in Hindsight is isolated per bank. If two callers share a bank, they leak into each other's context — the fastest way to make a voice agent feel broken. So the single most important decision here is: **what identity does `bank_id` map to?**

For cross-call memory, the bank must be tied to the person, not the session:

- **Phone number** is the natural key for telephony transports — it is stable across calls and available before the agent even answers.
- **Account or customer ID** is better when you have authenticated identity, since one person may call from several numbers.
- **Avoid** transient call/session IDs. A per-call bank means every call is a first call — the exact problem you are trying to solve.

```python
from hindsight_pipecat import HindsightMemoryService

# caller_number comes from your transport / telephony provider
memory = HindsightMemoryService(
    bank_id=f"caller-{caller_number}",   # stable per-caller key
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",  # or set HINDSIGHT_API_KEY env var
)
```

Because banks are strictly isolated, `caller-+15551234567` only ever recalls that caller's history. Pick the most stable identity you trust, and normalize it (consistent format) so the same person always resolves to the same bank.

## Recall at the start of a call, retain at the end

The recall-before / retain-after loop is what turns per-caller banks into continuity. Here is how it lands in the pipeline.

`HindsightMemoryService` acts on each `OpenAILLMContextFrame` (and the newer `LLMContextFrame`). On every turn it does two things:

1. **Retain** — any newly completed user+assistant turn pair is sent to Hindsight asynchronously, fire-and-forget, so nothing blocks the response path.
2. **Recall** — the latest user message is used as the search query, and the results are injected as a `<hindsight_memories>` system message before the LLM sees the context.

For cross-call memory, the important moment is the **first turn of a new call**. When a returning caller speaks, recall runs against their bank and surfaces what matters — their name, their open issue, their stated preferences — so the agent's first real reply already reflects who they are. You did not have to re-ask anything.

On the retain side, think about *what* a call should leave behind. You do not need to store every "um" and back-channel. The turn pairs Hindsight retains carry the substance: the caller's request, decisions made, preferences stated, commitments the agent made. Those are the things the next call should recall. If your calls are dense with throwaway chatter, tighten what you feed the agent rather than trying to filter after the fact — clean turns retain as clean memories.

```python
from pipecat.pipeline.pipeline import Pipeline

pipeline = Pipeline([
    transport.input(),
    stt_service,
    user_aggregator,
    memory,            # ← recall before LLM, retain after the turn
    llm_service,
    assistant_aggregator,
    tts_service,
    transport.output(),
])
```

The placement is load-bearing: `memory` must sit **after** `user_aggregator` (so it sees the assembled user context to query on) and **before** `llm_service` (so recalled context reaches the model before it generates). Put it after the LLM and recall can no longer influence the reply.

## Connect Pipecat to Hindsight

This guide assumes the integration is already installed and wired. If you have not done that yet, follow the setup guide first: [Add Pipecat Voice Agent Memory with Hindsight](https://hindsight.vectorize.io/guides/2026/05/04/guide-pipecat-memory-with-hindsight). It covers `pip install hindsight-pipecat`, pointing at Hindsight Cloud or a local server, and the base pipeline wiring. Come back here for the per-caller strategy.

## Voice-specific concerns: latency and recall budget

Voice is unforgiving about latency in a way text chat is not — a caller notices dead air. That shapes how you configure memory.

- **Retain is already non-blocking.** Completed turns are retained asynchronously, so storing memory never adds to the response path. You do not need to tune this for latency; you need to make sure it is enabled.
- **Recall is on the critical path.** Recall runs before the LLM, so its cost is part of time-to-first-token. This is where the recall budget matters. `HindsightMemoryService` exposes `recall_budget` (`"low"`, `"mid"`, `"high"`) and `recall_max_tokens`.

```python
memory = HindsightMemoryService(
    bank_id=f"caller-{caller_number}",
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",
    recall_budget="low",       # favor latency for real-time voice
    recall_max_tokens=2048,    # cap injected context so TTS starts fast
)
```

For a live phone call, start at `recall_budget="low"` and a modest `recall_max_tokens`. A voice agent rarely needs a wall of recalled context — it needs the two or three facts that make the caller feel known. Reserve `"mid"` or `"high"` for cases where you have measured headroom in your latency budget, or for non-real-time channels. If recall feels sluggish, it is almost always the budget set too aggressively for voice, not the store being slow.

You can also toggle the two halves independently with `enable_recall` and `enable_retain` if you want, for example, to retain silently during a rollout before turning recall on.

## Verify cross-call memory

The whole point is that a *second* call remembers the *first*. Test that directly:

1. Call in from a known identity (or run the text simulator with `--bank caller-demo`) and state a fact the agent should keep — an account detail, a preference, an open request.
2. End the call so the final turns retain.
3. Call in again from the same identity so the same `bank_id` resolves.
4. Ask a follow-up that depends on the earlier fact — do not restate it.
5. Confirm the agent answers using the first call's detail.

If the second call recalls the first, per-caller memory is working. If it does not: confirm both calls resolved to the identical `bank_id`, confirm `memory` sits before `llm_service`, and confirm retain completed on the first call (turn on debug logging). The `examples/interactive_chat.py` simulator in the package lets you run this loop without provisioning telephony:

```bash
python examples/interactive_chat.py --bank caller-demo
```

## Common mistakes

- **Using the call/session ID as the bank.** Every call becomes a first call. Scope to the caller (phone number or account), not the session.
- **Placing `memory` after the LLM.** Recalled context can no longer influence the reply. It must sit between `user_aggregator` and `llm_service`.
- **A recall budget too high for voice.** Aggressive recall adds to time-to-first-token and creates audible dead air. Start at `"low"` for live calls.
- **Not normalizing the identity.** If the same phone number sometimes resolves to a different string, it lands in a different bank and the caller looks new again.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works the same way — point `hindsight_api_url` at `http://localhost:8888` and drop the `api_key`. Cloud just saves you from running the backend.

### Should the bank be one per phone number?

Usually yes, when the number maps cleanly to one person. If you have authenticated identity, an account or customer ID is more robust, since one person may call from multiple numbers.

### Does retaining a call slow down the caller's experience?

No. Retain is fire-and-forget and runs asynchronously, so it never blocks the response path. Only recall is on the critical path, and you tune that with `recall_budget`.

### What actually gets remembered between calls?

The substance of completed turn pairs — requests, decisions, stated preferences, commitments. Recall then surfaces only the parts relevant to the current turn, rather than replaying the whole prior transcript.

## Next Steps

- Follow the [Pipecat setup guide](https://hindsight.vectorize.io/guides/2026/05/04/guide-pipecat-memory-with-hindsight) if you have not installed the integration yet
- Start with [Hindsight Cloud](https://hindsight.vectorize.io) for a hosted memory backend
- Read [the full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow [the quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [Pipecat integration docs](https://hindsight.vectorize.io/docs/integrations/pipecat)
