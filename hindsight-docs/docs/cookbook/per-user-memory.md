---
sidebar_position: 1
---

# Per-User Memory

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

## Implementation

### 1. Create a Bank When User Signs Up

```python
from hindsight import HindsightClient

client = HindsightClient()

def on_user_signup(user_id: str):
    client.create_bank(
        bank_id=f"user-{user_id}",
        name=f"Memory for {user_id}"
    )
```

### 2. Manage Conversation Sessions

Use `document_id` to group messages belonging to the same conversation. When you retain with the same `document_id`, Hindsight replaces the previous version (upsert behavior), keeping the memory up-to-date as the conversation evolves.

```python
import uuid

class ConversationSession:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.session_id = str(uuid.uuid4())  # Unique ID for this conversation
        self.messages = []

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    async def save(self, client: HindsightClient):
        """Save the entire conversation. Replaces previous version if session_id exists."""
        await client.retain(
            bank_id=f"user-{self.user_id}",
            content=self.messages,
            document_id=self.session_id  # Same ID = upsert (replace old version)
        )
```

### 3. Recall Context Before Responding

```python
async def get_context(user_id: str, query: str):
    result = await client.recall(
        bank_id=f"user-{user_id}",
        query=query
    )
    return result.results
```

### 4. Complete Agent Loop

```python
async def handle_message(session: ConversationSession, user_message: str):
    # 1. Add user message to session
    session.add_message("user", user_message)

    # 2. Recall relevant context from past conversations
    context = await client.recall(
        bank_id=f"user-{session.user_id}",
        query=user_message
    )

    # 3. Build prompt with memory
    prompt = f"""You are a helpful assistant with memory of past conversations.

## What you remember about this user
{format_results(context.results)}

## Current conversation
{format_messages(session.messages)}
"""

    # 4. Generate response
    response = await llm.complete(prompt)

    # 5. Add assistant response to session
    session.add_message("assistant", response)

    # 6. Save the updated conversation (upserts based on session_id)
    await session.save(client)

    return response
```

### 5. Starting a New Conversation

```python
# Each new conversation gets a new session with a unique ID
session = ConversationSession(user_id="alice")

# Multiple exchanges in the same conversation
await handle_message(session, "Hi! I'm working on a Python project")
await handle_message(session, "Can you help me with async/await?")

# Start a new conversation later (new session_id)
new_session = ConversationSession(user_id="alice")
await handle_message(new_session, "Different topic today...")
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
- See [Support Agent with Shared Knowledge](./support-agent-with-shared-knowledge)
