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

Then configure it in your agent:

```python
from hindsight_agentcore import HindsightRuntimeAdapter
from hindsight_client import Hindsight

# Create Hindsight client
hindsight = Hindsight(
    base_url="https://api.hindsight.vectorize.io",
    api_key="your-hindsight-key"
)

# Wrap your AgentCore runtime with the adapter
adapter = HindsightRuntimeAdapter(
    hindsight_client=hindsight,
    bank_id="my-agent-memory",
    agent_name="my-agentcore-agent"
)

# In your agent's turn processing:
# Instead of: result = runtime.invoke(request)
# Use: result = adapter.run_turn(runtime, request, turn_context)
```

The adapter handles all memory operations automatically. Your agent logic stays clean.

## Real-World Use Cases

### Use Case 1: Customer Support Agent

An AgentCore agent that handles multi-session customer support with persistent case context:

```python
from hindsight_agentcore import HindsightRuntimeAdapter
from hindsight_client import Hindsight

hindsight = Hindsight(
    base_url="https://api.hindsight.vectorize.io",
    api_key="your-hindsight-key"
)

adapter = HindsightRuntimeAdapter(
    hindsight_client=hindsight,
    bank_id="support-cases",
    agent_name="support-agent"
)

# Customer calls back days later
# Agent automatically recalls the case context
result = adapter.run_turn(
    runtime=support_agent_runtime,
    request=user_message,
    turn_context=TurnContext(
        session_id=customer_id,
        conversation_id=case_number
    )
)

# Agent knows the history and continues helping
```

The agent remembers the customer's issue, previous solutions tried, and relevant account context—without the customer having to repeat everything.

### Use Case 2: Data Analysis Agent

An AgentCore agent that explores datasets and remembers discoveries for follow-up analysis:

```python
from hindsight_agentcore import HindsightRuntimeAdapter
from hindsight_client import Hindsight

hindsight = Hindsight(
    base_url="https://api.hindsight.vectorize.io",
    api_key="your-hindsight-key"
)

adapter = HindsightRuntimeAdapter(
    hindsight_client=hindsight,
    bank_id="data-analysis",
    agent_name="analytics-agent"
)

# Session 1: Initial analysis
result1 = adapter.run_turn(
    runtime=analytics_agent,
    request="Analyze Q1 sales trends",
    turn_context=TurnContext(session_id="session-1")
)

# Session 2 (days later): Follow-up analysis
result2 = adapter.run_turn(
    runtime=analytics_agent,
    request="How do Q2 trends compare to what you found in Q1?",
    turn_context=TurnContext(session_id="session-2")
)

# Agent recalls Q1 findings and makes comparative insights
```

The agent's analysis compounds over time—each new query builds on what was previously discovered.

### Use Case 3: Code Review Agent

An AgentCore agent that learns codebase patterns and applies learnings across reviews:

```python
from hindsight_agentcore import HindsightRuntimeAdapter
from hindsight_client import Hindsight

hindsight = Hindsight(
    base_url="https://api.hindsight.vectorize.io",
    api_key="your-hindsight-key"
)

adapter = HindsightRuntimeAdapter(
    hindsight_client=hindsight,
    bank_id="code-reviews",
    agent_name="code-reviewer"
)

# Review PRs over multiple sessions
for pr in pull_requests:
    result = adapter.run_turn(
        runtime=review_agent,
        request=f"Review PR {pr.number}: {pr.diff}",
        turn_context=TurnContext(
            session_id=f"review-session-{pr.number}",
            metadata={"repo": pr.repo, "pr": pr.number}
        )
    )
    # Agent recalls patterns from previous reviews
    # Applies codebase learnings to this review
```

The agent builds institutional knowledge of your codebase style, patterns, and conventions—improving review quality over time.

## How the Adapter Works

The **HindsightRuntimeAdapter** is a thin middleware layer:

```python
class HindsightRuntimeAdapter:
    def before_turn(self, request, turn_context):
        # Recall relevant memories based on context
        # Inject context into agent's internal state
        pass
    
    def run_turn(self, runtime, request, turn_context):
        # Recall prior context
        context = self.before_turn(request, turn_context)
        
        # Run agent normally
        result = runtime.invoke(request)
        
        # Persist learnings
        self.after_turn(result, turn_context)
        
        return result
    
    def after_turn(self, result, turn_context):
        # Extract agent outputs and decisions
        # Store as persistent memory
        pass
```

The adapter integrates with **TurnContext** to key memories by:
- **session_id**: Multi-tenant support (different users/sessions)
- **conversation_id**: Grouping related turns
- **metadata**: Custom tags and context

This ensures memories are correctly scoped and retrievable.

## Memory Scoping: Multi-Tenant Deployments

For deployments serving multiple users, TurnContext provides identity:

```python
from hindsight_agentcore import TurnContext

# Each user gets isolated memory
turn_context = TurnContext(
    session_id=user_id,           # Isolates memory by user
    conversation_id=thread_id,    # Groups related turns
    metadata={
        "user_tier": "premium",
        "department": "sales"
    }
)

result = adapter.run_turn(
    runtime=agent_runtime,
    request=user_request,
    turn_context=turn_context
)

# User A's memories never leak to User B
# Recall filters by session_id automatically
```

The adapter ensures memory is always scoped correctly—critical for multi-tenant systems.

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
