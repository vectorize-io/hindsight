---
sidebar_position: 2
---

# Per-User Memory


:::tip Run this notebook
This recipe is available as an interactive Jupyter notebook.
[**Open in GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/blob/main/notebooks/02-per-user-memory.ipynb)
:::


The simplest pattern: give your agent persistent memory for each user. The agent remembers past conversations, user preferences, and context across sessions.

## The Problem

Without memory, every conversation starts from scratch:

```
Session 1: "I prefer dark mode and use Python"
Session 2: "What's my preferred language?" → Agent doesn't know
```

## The Solution: One Bank Per User

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   User A Bank   │     │   User B Bank   │     │   User C Bank   │
│                 │     │                 │     │                 │
│  - Conversations│     │  - Conversations│     │  - Conversations│
│  - Preferences  │     │  - Preferences  │     │  - Preferences  │
│  - Context      │     │  - Context      │     │  - Context      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
   100% isolated          100% isolated           100% isolated
```

Each user gets their own memory bank. Complete isolation, simple mental model.


```python
!pip install hindsight-client nest_asyncio openai python-dotenv -U
```

## 1. Create a Bank When User Signs Up


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

def on_user_signup(user_id: str):
    client.create_bank(
        bank_id=f"user-{user_id}",
        name=f"Memory for {user_id}"
    )
    print(f"View bank: {HINDSIGHT_UI_URL}/banks/user-{user_id}?view=documents")
```

## 2. Manage Conversation Sessions

Use `document_id` to group messages belonging to the same conversation. When you retain with the same `document_id`, Hindsight replaces the previous version (upsert behavior), keeping the memory up-to-date as the conversation evolves.


```python
import uuid
import json

class ConversationSession:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.session_id = str(uuid.uuid4())  # Unique ID for this conversation
        self.messages = []

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def save(self, client: Hindsight):
        """Save the entire conversation. Replaces previous version if session_id exists."""
        # Convert messages to string format for retain
        content = "\n".join([f"{m['role']}: {m['content']}" for m in self.messages])
        client.retain(
            bank_id=f"user-{self.user_id}",
            content=content,
            document_id=self.session_id  # Same ID = upsert (replace old version)
        )
```

## 3. Recall Context Before Responding


```python
def get_context(user_id: str, query: str):
    result = client.recall(
        bank_id=f"user-{user_id}",
        query=query
    )
    return result.results
```

## 4. Complete Agent Loop


```python
def format_results(results):
    """Format recall results for the prompt."""
    if not results:
        return "No relevant memories found."
    return "\n".join([f"- {r.text}" for r in results])

def format_messages(messages):
    """Format conversation messages for the prompt."""
    return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

def handle_message(session: ConversationSession, user_message: str):
    # 1. Add user message to session
    session.add_message("user", user_message)

    # 2. Recall relevant context from past conversations
    context = client.recall(
        bank_id=f"user-{session.user_id}",
        query=user_message
    )

    # 3. Build system prompt with memory
    system_prompt = f"""You are a helpful assistant with memory of past conversations.

## What you remember about this user
{format_results(context.results)}

Respond helpfully and reference relevant memories when appropriate."""

    # 4. Generate response using OpenAI
    response = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            *[{"role": m["role"], "content": m["content"]} for m in session.messages]
        ]
    )
    assistant_response = response.choices[0].message.content

    # 5. Add assistant response to session
    session.add_message("assistant", assistant_response)

    # 6. Save the updated conversation (upserts based on session_id)
    session.save(client)

    print(f"User: {user_message}")
    print(f"Assistant: {assistant_response}\n")
    
    return assistant_response
```

## 5. Starting a New Conversation


```python
# Create the user's bank
on_user_signup("alice")

# Each new conversation gets a new session with a unique ID
session = ConversationSession(user_id="alice")

# Multiple exchanges in the same conversation
handle_message(session, "Hi! I'm working on a Python project")
handle_message(session, "Can you help me with async/await?")

# View the stored conversation in the UI.
# Each message updates the same document (via document_id), so you'll see
# the full conversation history in a single document rather than separate entries.
print(f"\nView documents: {HINDSIGHT_UI_URL}/banks/user-alice?view=documents")
```

## How Document ID Works

The `document_id` parameter is key to managing evolving conversations:

| Scenario | Behavior |
|----------|----------|
| First retain with `document_id="session_123"` | Creates new document |
| Retain again with same `document_id="session_123"` | **Replaces** previous version (upsert) |
| Retain with different `document_id="session_456"` | Creates separate document |
| Retain without `document_id` | Creates new document each time |

This upsert behavior means:
- You always retain the **full conversation** state
- Facts are re-extracted from the complete conversation
- No duplicate or stale facts from old versions
- Memory stays consistent as conversations evolve

## What Gets Remembered

Hindsight automatically extracts and connects:

- **Facts**: "User prefers Python", "User is building a CLI tool"
- **Entities**: People, projects, technologies mentioned
- **Relationships**: How entities relate to each other
- **Temporal context**: When things happened

You don't need to manually extract or structure this - just retain the conversations.

## When to Use This Pattern

**Good fit:**
- Chatbots and assistants
- Personal AI companions
- Any 1:1 user-to-agent interaction

**Consider adding shared knowledge if:**
- You have product docs or FAQs to reference
- Multiple users need access to the same information
- See the Support Agent with Shared Knowledge notebook

## Cleanup

Delete the banks created during this notebook:


```python
import requests

# Delete the user-alice bank
response = requests.delete(f"{HINDSIGHT_API_URL}/v1/default/banks/user-alice")
print(f"Deleted user-alice: {response.json()}")
```
