# hindsight-agentcore

Persistent memory for [Amazon Bedrock AgentCore Runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-sessions.html) agents using [Hindsight](https://hindsight.vectorize.io).

AgentCore Runtime sessions are explicitly ephemeral — they terminate on inactivity and reprovision fresh environments. This package adds durable cross-session memory so agents remember users, decisions, and learned patterns across any number of Runtime sessions.

## How It Works

```
AgentCore Runtime invocation
        │
        ▼
   before_turn()         ← Recall relevant memories from Hindsight
        │
        ▼
  Agent executes          ← Prompt enriched with prior context
        │
        ▼
   after_turn()          ← Retain output to Hindsight (async)
```

Memory is keyed to stable user identity — **not** the `runtimeSessionId`. Banks survive session churn.

Default bank format:
```
tenant:{tenant_id}:user:{user_id}:agent:{agent_name}
```

## Installation

```bash
pip install hindsight-agentcore
```

## Quick Start

```python
import os
from hindsight_agentcore import HindsightRuntimeAdapter, TurnContext, configure

configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key=os.environ["HINDSIGHT_API_KEY"],
)

adapter = HindsightRuntimeAdapter(agent_name="support-agent")


# Your AgentCore Runtime handler
async def handler(event: dict) -> dict:
    context = TurnContext(
        runtime_session_id=event["sessionId"],
        user_id=event["userId"],       # from validated auth — never client-supplied
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


async def run_my_agent(payload: dict, memory_context: str) -> dict:
    prompt = payload["prompt"]
    if memory_context:
        prompt = f"Past context:\n{memory_context}\n\nCurrent request: {prompt}"

    output = await call_bedrock(prompt)
    return {"output": output}
```

## Lower-Level Hooks

```python
# Manual recall → execute → retain
memory_context = await adapter.before_turn(context, query=user_message)

result = await run_my_agent(payload, memory_context=memory_context)

await adapter.after_turn(context, result=result["output"], query=user_message)
```

## Retrieval Modes

### Recall (default)

Fast multi-strategy retrieval (semantic + keyword + graph + temporal):

```python
from hindsight_agentcore import RecallPolicy

adapter = HindsightRuntimeAdapter(
    recall_policy=RecallPolicy(mode="recall", budget="mid", max_tokens=1500)
)
```

### Reflect

LLM-synthesized context for complex reasoning tasks:

```python
adapter = HindsightRuntimeAdapter(
    recall_policy=RecallPolicy(mode="reflect")
)
```

Use `reflect` selectively — it's slower. Reserve it for explicit planning steps or routing decisions.

## Async Retention

By default, `after_turn()` fires retention as a background task — the user turn is never delayed:

```python
configure(retain_async=True)   # default
configure(retain_async=False)  # await retention before returning
```

## Async and Long-Running Workflows

For jobs spanning multiple Runtime sessions, retain at start and completion:

```python
# Task start
await adapter.after_turn(
    context,
    result="Started QBR analysis for Acme Corp",
    query=task_description,
)

# ... long-running work across potentially multiple sessions ...

# Task completion
await adapter.after_turn(
    context,
    result=f"Completed QBR analysis. Finding: {summary}",
    query=task_description,
)
```

## Identity and Auth

**Never use `runtimeSessionId` as the bank ID.** Sessions expire. Memory must survive session churn.

Preferred identity sources (in order):
1. Validated user ID from AgentCore JWT/OAuth context
2. `X-Amzn-Bedrock-AgentCore-Runtime-User-Id` header
3. Application-supplied user ID in trusted server-side deployments

```python
context = TurnContext(
    runtime_session_id=event["sessionId"],
    user_id=jwt_claims["sub"],         # stable identity from validated token
    agent_name="support-agent",
    tenant_id=jwt_claims.get("tenant"),
)
```

## Configuration Reference

| Option | Env Variable | Default | Description |
|---|---|---|---|
| `hindsight_api_url` | `HINDSIGHT_API_URL` | Hindsight Cloud | Hindsight server URL |
| `api_key` | `HINDSIGHT_API_KEY` | — | API key for Hindsight Cloud |
| `recall_budget` | — | `"mid"` | Search depth: `low`, `mid`, `high` |
| `recall_max_tokens` | — | `1500` | Max tokens recalled |
| `retain_async` | — | `True` | Non-blocking retention |
| `timeout` | — | `15.0` | HTTP timeout in seconds |
| `tags` | — | `[]` | Tags applied to all retained memories |
| `verbose` | — | `False` | Log memory operations |

## Custom Bank Resolution

Override the default `tenant:user:agent` format:

```python
from hindsight_agentcore import TurnContext

def my_resolver(context: TurnContext) -> str:
    return f"acme:{context.user_id}:{context.agent_name}"

adapter = HindsightRuntimeAdapter(bank_resolver=my_resolver)
```

**Security rule:** The resolver must fail closed (`BankResolutionError`) rather than leak memory across users when identity is missing.

## Failure Modes

| Failure | Behavior |
|---|---|
| Hindsight unavailable | `before_turn()` returns `""`, agent continues |
| Recall timeout | Returns `""`, agent continues |
| Retain failure | Logged as warning, user turn unaffected |
| Bad bank resolution | Fails closed — no cross-user memory leakage |

## Deployment

**Hindsight Cloud** — [sign up](https://ui.hindsight.vectorize.io/signup), set `hindsight_api_url` to your Cloud endpoint.

**Self-hosted on AWS** — run Hindsight on ECS/EKS with RDS PostgreSQL (pgvector). Network path stays in your AWS account.

## Requirements

- Python 3.10+
- `hindsight-client>=0.4.0`
