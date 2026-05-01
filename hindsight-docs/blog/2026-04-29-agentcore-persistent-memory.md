---
title: "Multi-Turn Agent Memory with AWS AgentCore: Remember Across Sessions"
authors: [benfrank241]
date: 2026-04-29T16:00:00Z
tags: [integrations, aws, agentcore, bedrock, agents, memory, guide]
description: "Add persistent memory to AWS AgentCore agents with Hindsight. Agents remember context, decisions, and learnings across multi-turn conversations and sessions."
image: /img/blog/agentcore-persistent-memory.png
hide_table_of_contents: true
---

![Multi-Turn Agent Memory with AWS AgentCore: Remember Across Sessions](/img/blog/agentcore-persistent-memory.png)

AWS AgentCore (Amazon Bedrock Agents) excels at building intelligent agents that reason and act across multiple turns. But agents without persistent memory struggle with multi-session workflows. Hindsight now integrates with AgentCore as a runtime adapter that automatically captures agent context, decisions, and learnings—persisting them across sessions and enabling agents to build long-term institutional knowledge.

<!-- truncate -->

## Why Persistent Memory Matters for AgentCore

AgentCore agents are powerful within a single session, but they face a critical limitation:
- Each new session starts fresh, losing context from previous interactions
- Agents can't learn from past decisions or incorporate prior insights
- Users must re-explain context and history constantly
- Multi-turn workflows lose continuity when agents restart

Hindsight solves this with the **HindsightRuntimeAdapter**—a middleware that intercepts each agent turn, extracts relevant context, and stores it as persistent memory. Your agents now remember across sessions.

## How Hindsight Integrates with AgentCore

The integration is a single runtime adapter that hooks into AgentCore's turn lifecycle:

1. **Before Each Turn** - Recall relevant prior context
   - Agent retrieves memories related to the incoming request
   - Context is injected into the agent's reasoning
   - Agent builds on what it learned previously

2. **During the Turn** - Agent acts normally
   - Processes the user input with recall context
   - Makes decisions using internal reasoning
   - Produces outputs and tool calls as usual

3. **After Each Turn** - Persist new learning
   - Adapter captures agent decisions, outputs, and insights
   - New facts are stored in the memory bank
   - Tagged and indexed for future recall

All three operations happen transparently—your agent code stays unchanged.

## Setting Up Hindsight with AgentCore

First, install the Hindsight AgentCore integration:

```bash
pip install hindsight-agentcore
```

Then configure it globally and create an adapter:

```python
from hindsight_agentcore import HindsightRuntimeAdapter, TurnContext, configure
import os

# Configure Hindsight once (globally)
configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key=os.environ["HINDSIGHT_API_KEY"]
)

# Create the adapter
adapter = HindsightRuntimeAdapter(agent_name="my-support-agent")
```

In your AgentCore Runtime handler, create a TurnContext and call `run_turn`:

```python
# Inside your AgentCore event handler
context = TurnContext(
    runtime_session_id=event["sessionId"],
    user_id=event["userId"],
    agent_name="my-support-agent",
    tenant_id=event.get("tenantId")
)

result = await adapter.run_turn(
    context=context,
    payload={"prompt": user_message},
    agent_callable=my_agent_function
)
```

The adapter automatically recalls relevant context before execution and retains learnings after. Your agent function stays clean—it just receives the payload plus recalled memories.

## Real-World Use Cases

### Use Case 1: Customer Support Agent

An AgentCore agent that handles multi-session customer support with persistent case context:

```python
from hindsight_agentcore import HindsightRuntimeAdapter, TurnContext, configure
import os

# Configure once
configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key=os.environ["HINDSIGHT_API_KEY"]
)

adapter = HindsightRuntimeAdapter(agent_name="support-agent")

# Your LLM-backed agent function
async def support_agent(payload: dict, memory_context: str) -> dict:
    user_message = payload["prompt"]
    
    # Inject prior case context if available
    system_prompt = "You are a helpful support agent."
    if memory_context:
        system_prompt += f"\n\nPrior context about this customer:\n{memory_context}"
    
    # Call your LLM
    response = await llm.invoke(
        system_prompt=system_prompt,
        user_message=user_message
    )
    
    return {"output": response}

# When a customer returns days later
context = TurnContext(
    runtime_session_id=event["sessionId"],
    user_id=customer_id,
    agent_name="support-agent",
    tenant_id=account_id
)

result = await adapter.run_turn(
    context=context,
    payload={"prompt": "I need help with my invoice from last week"},
    agent_callable=support_agent
)
```

The agent automatically recalls the customer's issue, previous solutions, and account details—without the customer repeating anything.

### Use Case 2: Data Analysis Agent

An AgentCore agent that explores datasets and remembers discoveries for follow-up analysis:

```python
from hindsight_agentcore import HindsightRuntimeAdapter, TurnContext, configure
import os

configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key=os.environ["HINDSIGHT_API_KEY"]
)

adapter = HindsightRuntimeAdapter(agent_name="analytics-agent")

async def analytics_agent(payload: dict, memory_context: str) -> dict:
    query = payload["prompt"]
    
    # Build prompt with prior analysis context
    system = "You are a data analyst. Analyze trends carefully."
    if memory_context:
        system += f"\n\nPrevious analyses:\n{memory_context}"
    
    # Run analysis
    analysis = await llm.invoke(system_prompt=system, user_message=query)
    return {"output": analysis}

# Session 1: Initial Q1 analysis
context1 = TurnContext(
    runtime_session_id="session-1",
    user_id="analyst-1",
    agent_name="analytics-agent"
)
result1 = await adapter.run_turn(
    context=context1,
    payload={"prompt": "Analyze Q1 sales trends from our database"},
    agent_callable=analytics_agent
)

# Session 2 (days later): Compare Q2 to Q1
context2 = TurnContext(
    runtime_session_id="session-2",
    user_id="analyst-1",
    agent_name="analytics-agent"
)
result2 = await adapter.run_turn(
    context=context2,
    payload={"prompt": "How do Q2 trends compare to what you found in Q1?"},
    agent_callable=analytics_agent
)
```

The agent's analysis compounds—each new query automatically has access to prior findings, enabling comparative insights without manual context passing.

### Use Case 3: Code Review Agent

An AgentCore agent that learns codebase patterns and applies learnings across reviews:

```python
from hindsight_agentcore import HindsightRuntimeAdapter, TurnContext, configure
import os

configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key=os.environ["HINDSIGHT_API_KEY"]
)

adapter = HindsightRuntimeAdapter(agent_name="code-reviewer")

async def code_reviewer(payload: dict, memory_context: str) -> dict:
    pr_diff = payload["prompt"]
    
    system = """You are a code reviewer. Check for:
- Performance issues
- Security problems
- Style inconsistencies
- Design patterns"""
    
    # Inject codebase patterns from prior reviews
    if memory_context:
        system += f"\n\nCoding patterns in this repo:\n{memory_context}"
    
    review = await llm.invoke(system_prompt=system, user_message=pr_diff)
    return {"output": review}

# Review PRs across multiple sessions
for pr in pull_requests:
    context = TurnContext(
        runtime_session_id=f"review-{pr.id}",
        user_id="reviewer-team",
        agent_name="code-reviewer",
        tenant_id="engineering"
    )
    
    result = await adapter.run_turn(
        context=context,
        payload={"prompt": f"Review this PR:\n\n{pr.diff}"},
        agent_callable=code_reviewer
    )
    # Agent automatically recalls patterns from prior reviews
    # and applies them to this review
```

The agent builds institutional knowledge—improving review quality by recognizing and enforcing codebase patterns across all reviews.

## How the Adapter Works

The **HindsightRuntimeAdapter** orchestrates three phases around your agent:

```python
# 1. Before turn: Recall relevant context
memory_context = await adapter.before_turn(
    context=turn_context,
    query=user_message
)

# 2. Execute agent with recalled memories
result = await agent_callable(
    payload={"prompt": user_message},
    memory_context=memory_context  # Injected by adapter
)

# 3. After turn: Retain the output for future recall
await adapter.after_turn(
    context=turn_context,
    result=result["output"],
    query=user_message
)
```

The adapter integrates with **TurnContext** to key memories by:
- **runtime_session_id**: Session identifier for ephemeral tracking
- **user_id**: User who initiated the turn (memory scoping)
- **agent_name**: Which agent produced the memory
- **tenant_id**: Optional multi-tenant isolation
- **request_id**: Optional request tracing

These fields ensure memories are correctly scoped, isolated, and retrievable across sessions.

## Memory Scoping: Multi-Tenant Deployments

For deployments serving multiple users/tenants, TurnContext enforces proper isolation:

```python
from hindsight_agentcore import TurnContext

# Each user/tenant gets isolated memory
turn_context = TurnContext(
    runtime_session_id=runtime_session_id,  # AgentCore session tracking
    user_id=authenticated_user_id,          # Isolates memory by user
    agent_name="my-agent",                  # Identifies the agent
    tenant_id=customer_account_id,          # Multi-tenant isolation (optional)
    request_id=request_id                   # Tracing (optional)
)

result = await adapter.run_turn(
    context=turn_context,
    payload={"prompt": user_request},
    agent_callable=agent_function
)

# User A's memories never leak to User B
# Bank ID is derived from tenant_id:user_id:agent_name
```

The adapter uses **tenant_id** and **user_id** to construct isolated memory banks. Memories stored by one user are never recalled for another—critical for multi-tenant systems.

## Best Practices

**Scope Memory Carefully:** Use session_id and conversation_id to isolate user contexts. In multi-tenant systems, session_id is the primary isolation boundary.

**Tag for Organization:** Apply tags like `["customer", "account-123"]` or `["analysis", "q1-2026"]` to organize memories by context.

**Let the Adapter Handle It:** Don't manually call retain/recall. The adapter handles memory operations transparently during turn processing.

**Monitor Memory Growth:** In long-running systems, review stored memories periodically to ensure they remain relevant and don't accumulate noise.

**Test with Realistic Sessions:** Test multi-session workflows to verify the agent correctly recalls and applies prior context.

## Troubleshooting

**Agent Doesn't Recall Context:** Ensure turn_context is consistent across sessions for the same user/conversation. Check that session_id and conversation_id match expected values.

**Memory Feels Repetitive:** The agent may be storing overlapping facts. Review stored memories in the Hindsight Cloud dashboard and adjust what gets retained if needed.

**Performance Degradation:** In high-volume systems, memory recalls may add latency. Monitor recall performance and consider adjusting budget levels (low/mid/high) if needed.

**Multi-Tenant Isolation:** Verify TurnContext.session_id is always set and unique per user. Test that one user's memories don't appear in another user's recalls.

## Next Steps

- [Hindsight Cloud](https://hindsight.vectorize.io)
- [AWS Bedrock Agents Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html)
- [Hindsight AgentCore Integration README](/sdks/integrations/agentcore)
- [Hindsight Retain API](/developer/api/retain)
- [Hindsight Recall API](/developer/api/recall)
