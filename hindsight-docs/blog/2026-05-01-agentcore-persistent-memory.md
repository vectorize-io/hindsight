---
title: "Multi-Turn Agent Memory with AWS AgentCore: Remember Across Sessions"
authors: [benfrank241]
date: 2026-05-01T16:00:00Z
tags: [integrations, aws, agentcore, bedrock, agents, memory, guide]
description: "Add persistent memory to AWS AgentCore agents with Hindsight. Agents remember context, decisions, and learnings across multi-turn conversations and sessions."
image: /img/blog/agentcore-persistent-memory.png
hide_table_of_contents: true
---

![Multi-Turn Agent Memory with AWS AgentCore: Remember Across Sessions](/img/blog/agentcore-persistent-memory.png)

<!-- truncate -->

AWS AgentCore (Amazon Bedrock Agents) excels at building intelligent agents that reason and act across multiple turns. But agents without persistent memory struggle with multi-session workflows. Hindsight now integrates with AgentCore as a runtime adapter that automatically captures agent context, decisions, and learnings—persisting them across sessions and enabling agents to build long-term institutional knowledge.

## Why Your Agents Need Memory

Imagine a customer support agent. The first time a customer reaches out, the agent solves their issue. Days later, the same customer returns with a follow-up question. **Without memory, the agent forgets everything.** The customer has to re-explain their context, the agent re-diagnoses the problem, and you lose all the learning from the first interaction.

AgentCore sessions are ephemeral—they spin up to handle a request and tear down afterward. The next customer request starts with a fresh agent that knows nothing about what happened before. This works for stateless transactions, but it breaks for agents that should grow smarter over time.

Hindsight solves this by automatically capturing what your agent learns and making it available in the next session. Your agent remembers customer context, previous solutions, and patterns it has discovered—without you writing retrieval code or managing memory manually.

## How Memory Works in Your Agent

Adding Hindsight to your agent is straightforward. A thin adapter wraps your agent and automates three things:

1. **Before handling a request** - Recall what you've learned
   - Look up relevant memories from previous interactions
   - Inject that context into the agent's prompt
   - Agent uses it to make better decisions

2. **While handling the request** - Your agent acts normally
   - Processes the user input with the recalled context
   - Makes decisions and produces outputs
   - Everything else stays the same

3. **After responding** - Automatically remember what you learned
   - Extract key insights from the agent's output
   - Store them for future recall
   - Indexed and ready for next time

All of this happens automatically. You configure Hindsight once, and your agent starts remembering.

## Hindsight vs AgentCore Memory: When to Use Each

AWS AgentCore includes **AgentCoreMemory**, a built-in memory service. Both can persist agent state, but they solve different problems:

| Feature | AgentCoreMemory | Hindsight |
|---------|-----------------|-----------|
| **Service Type** | AWS-managed, AWS-native | Specialized memory system |
| **Operations** | Low-level (write records, query) | High-level (retain, recall, reflect) |
| **Retrieval** | Explicit API calls required | Automatic pre-turn recall |
| **Customization** | External pipeline required | Built-in (retention policies, extraction modes, observation synthesis) |
| **Knowledge** | Raw records | Consolidated observations, mental models, synthesized insights |

**Choose AgentCoreMemory if:**
- You want AWS-native control and AWS management workflows
- Your memory needs are simple (store/retrieve records)
- You prefer AWS's standard strategies and namespaces

**Choose Hindsight if:**
- You want specialized memory operations (retain, recall, reflect for synthesis)
- You need richer retrieval (observations, mental models, consolidated knowledge)
- You want automation to handle both reads and writes without explicit API calls
- You need built-in customization (extraction policies, retention rules, observation synthesis)

**In this integration:** Hindsight automates the full lifecycle—before each turn, the adapter automatically recalls memories; after each turn, it automatically retains insights. AgentCore's post-event retention requires explicit writes; read-side retrieval requires explicit API calls.

## Adding Memory to Your Agent

**Step 1: Install the integration**

```bash
pip install hindsight-agentcore
```

**Step 2: Configure Hindsight** (once, at startup)

```python
from hindsight_agentcore import HindsightRuntimeAdapter, TurnContext, configure
import os

configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key=os.environ["HINDSIGHT_API_KEY"]
)

adapter = HindsightRuntimeAdapter(agent_name="my-support-agent")
```

**Step 3: Wrap your agent's handler**

```python
# Your AgentCore event handler
context = TurnContext(
    runtime_session_id=event["sessionId"],
    user_id=event["userId"],
    agent_name="my-support-agent",
    tenant_id=event.get("tenantId")  # optional
)

result = await adapter.run_turn(
    context=context,
    payload={"prompt": user_message},
    agent_callable=my_agent_function
)
```

That's it. The adapter now automatically recalls memories before each request and stores new insights after. Your agent code doesn't change—it just gets smarter over time.

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

**Set user_id consistently:** The user_id field in TurnContext is your primary isolation boundary. Use a stable user identifier that remains the same across sessions for the same person.

**Add tenant_id for multi-tenant apps:** If you're building for multiple customers, use tenant_id to ensure memories never leak between customers.

**Start simple, add features later:** You don't need to optimize memory immediately. Configure Hindsight, add the adapter, and let it run. Review the Hindsight Cloud dashboard later to see what's being stored and refine as needed.

**Monitor in production:** In high-volume systems, periodically check the Hindsight dashboard to ensure the agent is storing relevant memories and not accumulating noise.

**Test with multiple sessions:** Create a test scenario where your agent handles a request, the session ends, a new session starts with the same user, and a follow-up request comes in. Verify that the agent recalls the prior context and uses it correctly.

## Where to Deploy Hindsight

Hindsight works with two deployment options—choose based on your infrastructure preferences:

**Hindsight Cloud** (Recommended for Getting Started)
- Fully managed by Hindsight
- No infrastructure to operate
- Automatic scaling and backups
- Authentication via API key
- Best for: Teams that want to focus on building, not running databases

**Self-Hosted Hindsight OSS** (For Advanced Deployments)
- Run Hindsight in your AWS account (EC2, ECS, Lambda)
- Database: PostgreSQL-compatible (RDS, Aurora)
- Full control over data residency and networking
- Integrate with your existing infrastructure
- Best for: Teams with strict data residency requirements or existing database infrastructure

Both options work identically with the AgentCore integration—the adapter doesn't care where Hindsight runs. Start with Hindsight Cloud to get up and running in minutes, then migrate to self-hosted if your needs change.

## Troubleshooting

**Agent isn't using prior context:** Make sure you're using the same user_id across sessions. If user_id changes between the first request and the follow-up, the agent will have different memory banks. Check your TurnContext creation logic.

**Memories aren't being stored:** Verify the agent is actually returning output (result must have an "output" key in the result dict). If your agent returns early or errors out, nothing gets stored. Check logs and agent output.

**I'm seeing old memories:** Review what's stored in the Hindsight Cloud dashboard. The agent may be storing things you don't expect. Once you understand what's stored, you can adjust retention policies or extraction instructions.

**Multi-user isolation isn't working:** If you're sharing a single adapter across multiple users, make sure each user has a unique user_id in their TurnContext. Memories are keyed by user_id, so if everyone has the same user_id, they'll share memories.

## Next Steps

- [Hindsight Cloud](https://hindsight.vectorize.io)
- [AWS Bedrock Agents Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html)
- [Hindsight AgentCore Integration README](/sdks/integrations/agentcore)
- [Hindsight Retain API](/developer/api/retain)
- [Hindsight Recall API](/developer/api/recall)
