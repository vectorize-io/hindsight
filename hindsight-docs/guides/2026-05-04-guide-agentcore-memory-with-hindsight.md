---
title: "Guide: Add AgentCore Runtime Memory with Hindsight"
authors: [benfrank241]
date: 2026-05-04T15:00:00Z
tags: [how-to, agentcore, bedrock, memory]
description: "Add AgentCore Runtime memory with Hindsight using the runtime adapter, stable user bank IDs, and recall plus retain hooks across session churn."
image: /img/guides/guide-agentcore-memory-with-hindsight.png
hide_table_of_contents: true
---

![Guide: Add AgentCore Runtime Memory with Hindsight](/img/guides/guide-agentcore-memory-with-hindsight.png)

If you want **AgentCore Runtime memory with Hindsight**, the cleanest pattern is to wrap your Bedrock AgentCore handler with `HindsightRuntimeAdapter` and key memory to a stable user identity instead of the ephemeral runtime session ID. That gives AgentCore durable memory across session churn, which is the main gap teams hit when they move from demos to real user traffic.

This matters because AgentCore Runtime sessions are intentionally short lived. Without an external memory layer, the agent can keep losing context whenever the runtime environment turns over.

If you want the underlying reference open while you work, keep [the AgentCore Runtime integration docs](https://hindsight.vectorize.io/docs/integrations/agentcore), [the docs home](https://hindsight.vectorize.io/docs), [the quickstart guide](https://hindsight.vectorize.io/docs/quickstart), [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall), and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain) nearby.

<!-- truncate -->

> **Quick answer**
>
> 1. Install the AgentCore Runtime integration or plugin.
> 2. Point it at Hindsight Cloud or a local Hindsight API.
> 3. Wire memory into your AgentCore Runtime runtime with a stable bank ID.
> 4. Store one preference or project fact, then start a fresh run.
> 5. Confirm that recall brings the earlier context back automatically.

## Why this setup works

The runtime adapter gives you a clean recall, execute, retain loop around each turn. `before_turn()` fetches context, `run_turn()` wraps the whole exchange, and `after_turn()` stores the result. The critical design rule is simple: bank IDs must track stable user identity, not `runtimeSessionId`.

## Prerequisites

- An AgentCore Runtime handler that already receives a trusted user identifier
- Python and `hindsight-agentcore` installed
- A stable bank pattern, usually tenant plus user plus agent name

## Step 1: Install the integration

```bash
pip install hindsight-agentcore
```

## Step 2: Connect AgentCore Runtime to Hindsight

```python
import os
from hindsight_agentcore import configure

configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key=os.environ["HINDSIGHT_API_KEY"],
)
```

## Step 3: Wire memory into your runtime

```python
from hindsight_agentcore import HindsightRuntimeAdapter, TurnContext

adapter = HindsightRuntimeAdapter(agent_name="support-agent")

async def handler(event: dict) -> dict:
    context = TurnContext(
        runtime_session_id=event["sessionId"],
        user_id=event["userId"],
        agent_name="support-agent",
        tenant_id=event.get("tenantId"),
        request_id=event.get("requestId"),
    )

    result = await adapter.run_turn(
        context=context,
        payload={"prompt": event["prompt"]},
        agent_callable=run_my_agent,
    )
    return result
```

The default bank format follows `tenant:{tenant_id}:user:{user_id}:agent:{agent_name}`. That is exactly what you want for durable memory across Runtime session churn.

## Step 4: Choose the right bank strategy

Do not key memory to `runtimeSessionId`. That ID is ephemeral and will fragment memory immediately. Use a stable identity that survives across sessions, such as tenant plus user plus agent name. If several agent personas share the same account, give each persona its own suffix so memory stays coherent.

## Step 5: Verify that memory is working

1. Run one turn for a test user and store a preference or account detail.
2. Trigger another Runtime session for the same user.
3. Ask for the earlier detail and confirm that the adapter recalls it before the agent answers.
4. Repeat the test for a second user to confirm that banks stay isolated.

If the second run can answer with details from the first run, your setup is working. If it cannot, turn on debug logging, check the configured bank ID, and confirm that the retain call actually completed.

## Common mistakes

- Using a client supplied identifier that is not trusted, which can mix memory between users
- Keying the bank to `runtimeSessionId`, which breaks continuity by design
- Turning on reflect everywhere when recall would be faster and simpler for most turns

## FAQ

### Why should I avoid runtimeSessionId for memory?

Because AgentCore Runtime sessions are ephemeral. A memory bank tied to that ID dies with the session pattern you are trying to outgrow.

### When should I use reflect mode?

Use reflect selectively for harder reasoning steps. Keep normal turns on recall mode for lower latency.

### Can retention run in the background?

Yes. The integration supports async retention so user turns do not have to wait for retain to finish.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted memory backend
- Read [the full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow [the quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Compare a related workflow in [Agno persistent memory](https://hindsight.vectorize.io/blog/agno-persistent-memory)
