---
sidebar_position: 1
---

# AgentCore Runtime + Hindsight Memory


:::info Complete Application
This is a complete, runnable application demonstrating Hindsight integration.
[**View source on GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/tree/main/applications/agentcore-memory)
:::


Give your Amazon Bedrock AgentCore Runtime agents persistent long-term memory. AgentCore Runtime sessions are explicitly ephemeral — they reprovision fresh environments on every invocation. This example adds durable memory so agents remember users and decisions across any number of Runtime sessions.

## What This Demonstrates

- **`run_turn()` wrapper** — recall before execution, retain after, in one call
- **Session-stable identity** — memory keyed to `user_id`, not the ephemeral `runtimeSessionId`
- **Async retention** — the user turn returns immediately; memory is stored in the background
- **Memory survives session churn** — each message in the interactive loop simulates a fresh Runtime invocation

## Architecture

```
Runtime invocation N               Runtime invocation N+1
      │                                    │
      ▼                                    ▼
 before_turn()                       before_turn()
 ──► recall from Hindsight           ──► finds memories from invocation N
      │                                    │
      ▼                                    ▼
 Agent executes                      Agent executes
 (prompt enriched with memory)       (prompt enriched with prior context)
      │                                    │
      ▼                                    ▼
 after_turn()                        after_turn()
 ──► retain output (async)           ──► retain output (async)
```

Bank ID format (memory survives session churn because it uses `user_id`):
```
tenant:{tenant_id}:user:{user_id}:agent:{agent_name}
```

## Prerequisites

1. **Hindsight running**

   ```bash
   export OPENAI_API_KEY=your-key

   docker run --rm -it --pull always -p 8888:8888 -p 9999:9999 \
     -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
     -e HINDSIGHT_API_LLM_MODEL=gpt-4o-mini \
     -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
     ghcr.io/vectorize-io/hindsight:latest
   ```

2. **OpenAI API key** (for the agent's LLM)

   ```bash
   export OPENAI_API_KEY=your-key
   ```

3. **Install dependencies**

   ```bash
   cd applications/agentcore-memory
   pip install -r requirements.txt
   ```

## Quick Start

### Interactive Chat

Each message simulates a fresh AgentCore Runtime invocation (new `runtimeSessionId`, same memory bank):

```bash
python support_agent.py
```

Example session:

```
Support agent ready (user: demo-user, agent: support-agent)
Each message = one Runtime invocation (new session ID, same memory).

You: My account number is 12345 and I'm on the Pro plan
[Hindsight] Retaining to bank 'tenant:demo-tenant:user:demo-user:agent:support-agent'
Agent: Got it! I've noted your account number 12345 and that you're on the Pro plan.

You: What plan am I on?
[Hindsight] Recalling from bank 'tenant:demo-tenant:user:demo-user:agent:support-agent'
Agent: You're on the Pro plan (account 12345).
```

The second message gets a brand-new `runtimeSessionId` — the agent would normally start cold. But Hindsight recalls the previous context.

### Single Query

```bash
python support_agent.py "What do you know about my account?"
```

### Reset Memory

```bash
python support_agent.py --reset
```

## How It Works

### 1. Configure Hindsight

```python
from hindsight_agentcore import configure

configure(
    hindsight_api_url="http://localhost:8888",
    api_key=None,    # set for Hindsight Cloud
    verbose=True,
)
```

### 2. Create the Adapter

```python
from hindsight_agentcore import HindsightRuntimeAdapter

adapter = HindsightRuntimeAdapter(agent_name="support-agent")
```

### 3. Define Your Agent Callable

```python
async def run_my_agent(payload: dict, memory_context: str) -> dict:
    prompt = payload["prompt"]
    if memory_context:
        prompt = f"Past context:\n{memory_context}\n\nCurrent request: {prompt}"
    output = await call_llm(prompt)
    return {"output": output}
```

### 4. Build a TurnContext and Run

```python
from hindsight_agentcore import TurnContext

context = TurnContext(
    runtime_session_id=event["sessionId"],   # fresh each invocation
    user_id=jwt_claims["sub"],               # stable — from validated token
    agent_name="support-agent",
    tenant_id=jwt_claims.get("tenant"),
)

result = await adapter.run_turn(
    context=context,
    payload={"prompt": event["prompt"]},
    agent_callable=run_my_agent,
)
```

## Core Files

| File | Description |
|------|-------------|
| `support_agent.py` | Complete working example simulating AgentCore Runtime invocations |
| `requirements.txt` | Python dependencies |

## Configuration

| Variable | Default | Description |
|---|---|---|
| `HINDSIGHT_URL` | `http://localhost:8888` | Hindsight server URL |
| `HINDSIGHT_API_KEY` | — | API key for Hindsight Cloud |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `MODEL` | `gpt-4o-mini` | LLM model name |
| `USER_ID` | `demo-user` | Stable user identity (never use session ID) |
| `TENANT_ID` | `demo-tenant` | Tenant identifier |

## Deploying to AgentCore Runtime

In a real AgentCore Runtime deployment, the handler is an AWS Lambda-style function:

```python
async def handler(event: dict) -> dict:
    # User identity from validated JWT/OAuth — never from client-supplied fields
    context = TurnContext(
        runtime_session_id=event["sessionId"],
        user_id=event["agentCoreContext"]["userId"],
        agent_name="support-agent",
        tenant_id=event["agentCoreContext"].get("tenantId"),
        request_id=event.get("requestId"),
    )

    result = await adapter.run_turn(
        context=context,
        payload={"prompt": event["prompt"]},
        agent_callable=run_my_agent,
    )
    return result
```

For Hindsight Cloud, set `HINDSIGHT_API_URL` and `HINDSIGHT_API_KEY` as Runtime secrets.

## Common Issues

**"Connection refused"**
- Make sure Hindsight is running on `localhost:8888`

**"OPENAI_API_KEY not set"**
```bash
export OPENAI_API_KEY=your-key
```

**"No module named 'hindsight_agentcore'"**
```bash
pip install -r requirements.txt
```

---

**Built with:**
- [Amazon Bedrock AgentCore Runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-sessions.html) - AWS managed ephemeral agent hosting
- [hindsight-agentcore](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/agentcore) - Hindsight memory adapter for AgentCore Runtime
- [Hindsight](https://github.com/vectorize-io/hindsight) - Long-term memory for AI agents
