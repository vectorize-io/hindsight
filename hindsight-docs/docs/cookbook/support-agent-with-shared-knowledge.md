---
sidebar_position: 3
---

# Support Agent with Shared Knowledge

This pattern shows how to build a support agent that combines **per-user memory** with **shared product knowledge** (RAG), giving users personalized support while leveraging a single source of truth for documentation.

## The Problem

You're building a support agent that needs to:
- Remember each user's history, preferences, and past issues
- Access shared product documentation
- Keep user data completely isolated from other users

A naive approach would index product docs into each user's memory bank, but this is expensive and wasteful (N copies for N users).

## The Solution: Multi-Bank Architecture

Create separate memory banks for different concerns:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   User A Bank   │     │   User B Bank   │     │  Shared Docs    │
│                 │     │                 │     │     Bank        │
│  - Conversations│     │  - Conversations│     │                 │
│  - Preferences  │     │  - Preferences  │     │  - Product docs │
│  - Past issues  │     │  - Past issues  │     │  - FAQs         │
│  - Solutions    │     │  - Solutions    │     │  - Guides       │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                                 │
                           Agent queries
                           multiple banks
```

**Key benefits:**
- Product docs indexed once, shared by all users
- User memory is 100% isolated
- Simple mental model, no complex filtering

## Implementation

### 1. Set Up Memory Banks

Create three types of banks:

```python
from hindsight import HindsightClient

client = HindsightClient()

# Shared knowledge bank (created once)
shared_bank = client.create_bank(
    bank_id="product-docs",
    name="Product Documentation"
)

# Per-user banks (created when user signs up)
def create_user_bank(user_id: str):
    return client.create_bank(
        bank_id=f"user-{user_id}",
        name=f"Memory for {user_id}"
    )
```

### 2. Index Product Documentation

Index your product docs into the shared bank (do this once, or on doc updates):

```python
# Index product documentation
client.retain(
    bank_id="product-docs",
    content=[
        {
            "role": "document",
            "content": "# Pricing Tiers\n\nBasic: $10/mo...",
            "metadata": {"source": "pricing.md"}
        },
        {
            "role": "document",
            "content": "# Getting Started\n\nTo set up...",
            "metadata": {"source": "quickstart.md"}
        }
    ]
)
```

### 3. Store User Conversations

After each support interaction, retain it in the user's bank:

```python
def save_conversation(user_id: str, messages: list):
    client.retain(
        bank_id=f"user-{user_id}",
        content=messages  # [{"role": "user", "content": "..."}, ...]
    )
```

### 4. Query Multiple Banks at Support Time

When handling a user query, retrieve context from both banks:

```python
async def get_support_context(user_id: str, query: str):
    # Get user's personal context
    user_context = await client.recall(
        bank_id=f"user-{user_id}",
        query=query
    )

    # Get relevant product documentation
    docs_context = await client.recall(
        bank_id="product-docs",
        query=query
    )

    return {
        "user_history": user_context.results,
        "documentation": docs_context.results
    }
```

### 5. Build the Agent Prompt

Combine both contexts in your agent's prompt:

```python
def build_prompt(query: str, context: dict) -> str:
    return f"""You are a helpful support agent.

## User's History
{format_results(context["user_history"])}

## Product Documentation
{format_results(context["documentation"])}

## Current Question
{query}

Use the user's history to personalize your response and the documentation
for accurate product information. If you find a solution, remember it for
future reference.
"""
```

## Promoting Learnings to Shared Knowledge

When the agent discovers a solution that's not in the docs, you can optionally promote it to a "learnings" bank:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   User A Bank   │     │  Shared Docs    │     │    Learnings    │
│                 │     │     Bank        │     │      Bank       │
│  - Conversations│     │                 │     │                 │
│  - Preferences  │     │  - Product docs │     │  - Verified     │
│  - Past issues  │     │  - FAQs         │     │    solutions    │
│  - Solutions    │     │  - Guides       │     │  - Workarounds  │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                                 │
                           Agent queries
                           all three banks
```

```python
# Optional: Create a curated learnings bank
learnings_bank = client.create_bank(
    bank_id="support-learnings",
    name="Curated Support Learnings"
)

# After a successful resolution
def promote_learning(insight: str):
    client.retain(
        bank_id="support-learnings",
        content=[{
            "role": "system",
            "content": insight,
            "metadata": {"type": "verified_solution"}
        }]
    )
```

Then query three banks: user + docs + learnings.

## Complete Example

```python
from hindsight import HindsightClient

client = HindsightClient()

async def handle_support_request(user_id: str, query: str):
    # 1. Recall from user's memory
    user_recall = await client.recall(
        bank_id=f"user-{user_id}",
        query=query
    )

    # 2. Recall from shared docs
    docs_recall = await client.recall(
        bank_id="product-docs",
        query=query
    )

    # 3. Recall from learnings (optional)
    learnings_recall = await client.recall(
        bank_id="support-learnings",
        query=query
    )

    # 4. Build context for LLM
    context = f"""
User History:
{format_results(user_recall.results)}

Product Docs:
{format_results(docs_recall.results)}

Known Solutions:
{format_results(learnings_recall.results)}
"""

    # 5. Generate response with your LLM
    response = await llm.complete(
        system="You are a support agent...",
        context=context,
        query=query
    )

    # 6. Save the conversation to user's memory
    await client.retain(
        bank_id=f"user-{user_id}",
        content=[
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ]
    )

    return response
```

## When to Use This Pattern

**Good fit:**
- Support agents with shared documentation
- Multi-tenant applications with shared reference data
- Any scenario needing user isolation + shared knowledge

**Consider alternatives if:**
- You need cross-user learning (users benefiting from other users' solutions)
- Entity relationships must span across users and docs

