---
title: "Guide: Add Pipecat Voice Agent Memory with Hindsight"
authors: [benfrank241]
date: 2026-05-04T15:00:00Z
tags: [how-to, pipecat, voice-ai, memory]
description: "Add Pipecat voice agent memory with Hindsight using HindsightMemoryService to recall context before replies and retain completed turns across calls."
image: /img/guides/guide-pipecat-memory-with-hindsight.png
hide_table_of_contents: true
---

![Guide: Add Pipecat Voice Agent Memory with Hindsight](/img/guides/guide-pipecat-memory-with-hindsight.png)

If you want **Pipecat voice agent memory with Hindsight**, the cleanest setup is to insert `HindsightMemoryService` between the user aggregator and the LLM service in your pipeline. That lets Pipecat recall relevant context before each turn and retain completed turns after the response, which is exactly what a voice agent needs for continuity across calls.

This is more reliable than hoping the assistant will summarize its own history correctly. Hindsight handles the memory loop, while Pipecat keeps doing the real time audio and LLM orchestration it is already good at.

If you want the underlying reference open while you work, keep [the Pipecat integration docs](https://hindsight.vectorize.io/docs/integrations/pipecat), [the docs home](https://hindsight.vectorize.io/docs), [the quickstart guide](https://hindsight.vectorize.io/docs/quickstart), [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall), and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain) nearby.

<!-- truncate -->

> **Quick answer**
>
> 1. Install the Pipecat integration or plugin.
> 2. Point it at Hindsight Cloud or a local Hindsight API.
> 3. Wire memory into your Pipecat runtime with a stable bank ID.
> 4. Store one preference or project fact, then start a fresh run.
> 5. Confirm that recall brings the earlier context back automatically.

## Why this setup works

Voice agents need memory at the pipeline level, not only inside prompt text. `HindsightMemoryService` intercepts the conversation frame, recalls memories for the current user query, injects them into context, and retains complete turn pairs asynchronously so latency stays predictable.

## Prerequisites

- A Pipecat pipeline with user and assistant aggregators already working
- Python and `hindsight-pipecat` installed
- A stable bank ID per caller, user, or account

## Step 1: Install the integration

```bash
pip install hindsight-pipecat
```

## Step 2: Connect Pipecat to Hindsight

```python
from hindsight_pipecat import HindsightMemoryService

memory = HindsightMemoryService(
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
)
```

For [Hindsight Cloud](https://hindsight.vectorize.io), set `hindsight_api_url` to `https://api.hindsight.vectorize.io` and pass your API key.

## Step 3: Wire memory into your runtime

```python
from pipecat.pipeline.pipeline import Pipeline
from hindsight_pipecat import HindsightMemoryService

memory = HindsightMemoryService(
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
    recall_budget="mid",
    recall_max_tokens=4096,
)

pipeline = Pipeline([
    transport.input(),
    stt_service,
    user_aggregator,
    memory,
    llm_service,
    assistant_aggregator,
    tts_service,
    transport.output(),
])
```

## Step 4: Choose the right bank strategy

Use one bank per caller or account when the assistant should remember the same person across calls. Use a shared bank only when several users truly need the same memory context, such as a team queue. Voice systems feel wrong quickly when banks are too broad, so isolation matters here.

## Step 5: Verify that memory is working

1. Run a short conversation and state a preference or account detail that should matter later.
2. Start a second conversation with the same bank ID.
3. Ask a follow up question that depends on the earlier detail and confirm that the assistant recalls it.
4. If recall is missing, inspect where the memory processor sits in the pipeline and confirm it is before the LLM service.

If the second run can answer with details from the first run, your setup is working. If it cannot, turn on debug logging, check the configured bank ID, and confirm that the retain call actually completed.

## Common mistakes

- Placing the memory processor after the LLM, which prevents recalled context from influencing the response
- Using a transient call identifier instead of a stable user identifier for the bank
- Expecting low latency when recall is configured too aggressively for a real time voice use case

## FAQ

### Where should the memory processor go in the pipeline?

Place it between the user aggregator and the LLM so it can inject recalled context before generation.

### Does retain block the user turn?

No. Completed turns are retained asynchronously so the response path stays responsive.

### Should I use one bank per phone number?

Often yes, if the phone number maps cleanly to one user. Otherwise choose the stable account identifier you trust most.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted memory backend
- Read [the full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow [the quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Compare a related workflow in [team shared memory for AI coding agents](https://hindsight.vectorize.io/blog/team-shared-memory-ai-coding-agents)
