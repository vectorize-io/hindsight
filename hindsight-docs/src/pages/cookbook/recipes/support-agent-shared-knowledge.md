---
sidebar_position: 3
---

# Support Agent with Shared Knowledge


:::tip Run this notebook
This recipe is available as an interactive Jupyter notebook.
[**Open in GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/blob/main/notebooks/03-support-agent-shared-knowledge.ipynb)
:::


This pattern shows how to build a support agent that combines **per-user memory** with **shared product knowledge** (RAG), giving users personalized support while leveraging a single source of truth for documentation.

## The Problem

You're building a support agent that needs to:
- Remember each user's history, preferences, and past issues
- Access shared product documentation
- Keep user data completely isolated from other users

A naive approach would index product docs into each user's memory bank, but this is expensive and wasteful (N copies for N users).

## The Solution: Multi-Bank Architecture

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


```python
!pip install hindsight-client nest_asyncio openai python-dotenv -U
```

## 1. Set Up Memory Banks

Create three types of banks:


```python
# Jupyter notebooks already run an asyncio event loop. The hindsight client 
# uses loop.run_until_complete() internally, but Python doesn't allow nested 
# event loops by default. nest_asyncio patches this to allow nesting.
import nest_asyncio
nest_asyncio.apply()

import os
from dotenv import load_dotenv
from openai import OpenAI as OpenAIClient

# Load environment variables from .env file
# Copy .env.example to .env and fill in your values
load_dotenv()

# Configuration (override with env vars if set)
HINDSIGHT_API_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")
HINDSIGHT_UI_URL = os.getenv("HINDSIGHT_UI_URL", "http://localhost:9999")

from hindsight_client import Hindsight

client = Hindsight(base_url=HINDSIGHT_API_URL)
llm = OpenAIClient()  # Uses OPENAI_API_KEY from .env

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

## 2. Index Product Documentation

Index your product docs into the shared bank (do this once, or on doc updates):


```python
# Index product documentation - retain each doc separately
client.retain(
    bank_id="product-docs",
    content="# Pricing Tiers\n\nBasic: $10/mo, Pro: $25/mo, Enterprise: Contact us"
)

client.retain(
    bank_id="product-docs",
    content="# Getting Started\n\nTo set up your account, visit the dashboard and click 'New Project'"
)

# View the stored documents in the UI:
print(f"View documents: {HINDSIGHT_UI_URL}/banks/product-docs?view=documents")
```

## 3. Store User Conversations

After each support interaction, retain it in the user's bank:


```python
def save_conversation(user_id: str, messages: list):
    # Convert messages to string format
    content = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    client.retain(
        bank_id=f"user-{user_id}",
        content=content
    )
```

## 4. Query Multiple Banks at Support Time

When handling a user query, retrieve context from both banks:


```python
def get_support_context(user_id: str, query: str):
    # Get user's personal context
    user_context = client.recall(
        bank_id=f"user-{user_id}",
        query=query
    )

    # Get relevant product documentation
    docs_context = client.recall(
        bank_id="product-docs",
        query=query
    )

    return {
        "user_history": user_context.results,
        "documentation": docs_context.results
    }
```

## 5. Build the Agent Prompt

Combine both contexts in your agent's prompt:


```python
def format_results(results):
    """Format recall results for the prompt."""
    if not results:
        return "No relevant information found."
    return "\n".join([f"- {r.text}" for r in results])

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
        content=insight
    )
```

## Complete Example


```python
def format_results(results):
    if not results:
        return "No relevant information found."
    return "\n".join([f"- {r.text}" for r in results])

def handle_support_request(user_id: str, query: str):
    # 1. Recall from user's memory
    user_recall = client.recall(
        bank_id=f"user-{user_id}",
        query=query
    )

    # 2. Recall from shared docs
    docs_recall = client.recall(
        bank_id="product-docs",
        query=query
    )

    # 3. Recall from learnings (optional)
    learnings_recall = client.recall(
        bank_id="support-learnings",
        query=query
    )

    # 4. Build system prompt with context
    system_prompt = f"""You are a helpful support agent. Use the context below to answer the user's question.

## User's History
{format_results(user_recall.results)}

## Product Documentation
{format_results(docs_recall.results)}

## Known Solutions
{format_results(learnings_recall.results)}

Provide helpful, accurate responses based on the documentation. Reference the user's history when relevant."""

    # 5. Generate response using OpenAI
    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )
    assistant_response = response.choices[0].message.content

    # 6. Save the conversation to user's memory
    conversation = f"user: {query}\nassistant: {assistant_response}"
    client.retain(
        bank_id=f"user-{user_id}",
        content=conversation
    )

    return assistant_response

# Test the function
create_user_bank("bob")
print("User: How do I get started?")
result = handle_support_request("bob", "How do I get started?")
print(f"Assistant: {result}")
print(f"\nView user memory: {HINDSIGHT_UI_URL}/banks/user-bob?view=documents")
```

## When to Use This Pattern

**Good fit:**
- Support agents with shared documentation
- Multi-tenant applications with shared reference data
- Any scenario needing user isolation + shared knowledge

**Consider alternatives if:**
- You need cross-user learning (users benefiting from other users' solutions)
- Entity relationships must span across users and docs

## Cleanup

Delete the banks created during this notebook:


```python
import requests

# Delete all banks created in this notebook
for bank_id in ["product-docs", "support-learnings", "user-bob"]:
    response = requests.delete(f"{HINDSIGHT_API_URL}/v1/default/banks/{bank_id}")
    print(f"Deleted {bank_id}: {response.json()}")
```
