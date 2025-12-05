# hindsight-litellm

Universal LLM memory integration via LiteLLM. Add persistent memory to any LLM application with just a few lines of code.

## Features

- **Universal LLM Support** - Works with 100+ LLM providers via LiteLLM (OpenAI, Anthropic, Groq, Azure, AWS Bedrock, Google Vertex AI, and more)
- **Simple Integration** - Just configure, enable, and use `hindsight_litellm.completion()`
- **Automatic Memory Injection** - Relevant memories are injected into prompts before LLM calls
- **Automatic Conversation Storage** - Conversations are stored to Hindsight for future recall
- **Multi-User Support** - Entity ID scoping for isolated per-user memories
- **Session Management** - Group conversations into logical sessions
- **Direct Recall API** - Query memories manually without making LLM calls
- **Native Client Wrappers** - Alternative wrappers for OpenAI and Anthropic SDKs

## Installation

```bash
pip install hindsight-litellm
```

## Quick Start

```python
import hindsight_litellm

# Configure and enable memory integration
hindsight_litellm.configure(
    hindsight_api_url="http://localhost:8888",
    bank_id="my-agent",
    entity_id="user-123",  # Optional: for multi-user isolation
)
hindsight_litellm.enable()

# Use the convenience wrapper - memory is automatically injected and stored
response = hindsight_litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What did we discuss about AI?"}]
)
```

## Configuration Options

```python
hindsight_litellm.configure(
    # Required
    hindsight_api_url="http://localhost:8888",  # Hindsight API server URL
    bank_id="my-agent",                          # Memory bank ID

    # Optional - Multi-user and session management
    entity_id="user-123",          # User identifier for memory isolation
    session_id="session-abc",      # Session identifier for grouping
    api_key="your-api-key",        # Optional API key for authentication

    # Optional - Memory behavior
    store_conversations=True,      # Store conversations after LLM calls
    inject_memories=True,          # Inject relevant memories into prompts
    max_memories=10,               # Maximum memories to inject
    max_memory_tokens=2000,        # Maximum tokens for memory context
    recall_budget="mid",           # Recall budget: "low", "mid", "high"
    fact_types=["world", "agent"], # Filter fact types to inject

    # Optional - Advanced
    injection_mode="system_message",  # or "prepend_user"
    excluded_models=["gpt-3.5*"],     # Exclude certain models
    verbose=True,                     # Enable verbose logging
)
```

## Multi-Provider Support

Works with any LiteLLM-supported provider:

```python
import hindsight_litellm

hindsight_litellm.configure(
    hindsight_api_url="http://localhost:8888",
    bank_id="my-agent",
)
hindsight_litellm.enable()

# OpenAI
hindsight_litellm.completion(model="gpt-4o", messages=[...])

# Anthropic
hindsight_litellm.completion(model="claude-3-5-sonnet-20241022", messages=[...])

# Groq
hindsight_litellm.completion(model="groq/llama-3.1-70b-versatile", messages=[...])

# Azure OpenAI
hindsight_litellm.completion(model="azure/gpt-4", messages=[...])

# AWS Bedrock
hindsight_litellm.completion(model="bedrock/anthropic.claude-3", messages=[...])

# Google Vertex AI
hindsight_litellm.completion(model="vertex_ai/gemini-pro", messages=[...])
```

## Direct Recall API

Query memories manually without making an LLM call:

```python
from hindsight_litellm import configure, recall

configure(bank_id="my-agent", hindsight_api_url="http://localhost:8888")

# Query memories
memories = recall("what projects am I working on?", limit=5)
for m in memories:
    print(f"- [{m.fact_type}] {m.text}")

# Output:
# - [world] User is building a FastAPI project
# - [opinion] User prefers Python over JavaScript
```

### Async Recall

```python
from hindsight_litellm import arecall

memories = await arecall("what do you know about me?", limit=10)
```

## Native Client Wrappers

Alternative to LiteLLM callbacks for direct SDK integration:

### OpenAI Wrapper

```python
from openai import OpenAI
from hindsight_litellm import wrap_openai

client = OpenAI()
wrapped = wrap_openai(
    client,
    bank_id="my-agent",
    hindsight_api_url="http://localhost:8888",
    entity_id="user-123",
)

response = wrapped.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What do you know about me?"}]
)
```

### Anthropic Wrapper

```python
from anthropic import Anthropic
from hindsight_litellm import wrap_anthropic

client = Anthropic()
wrapped = wrap_anthropic(
    client,
    bank_id="my-agent",
    hindsight_api_url="http://localhost:8888",
)

response = wrapped.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Session Management

```python
from hindsight_litellm import configure, new_session, set_session, get_session

configure(bank_id="my-agent", hindsight_api_url="http://localhost:8888")

# Start a fresh conversation thread
session_id = new_session()
print(f"Started new session: {session_id}")

# Resume a previous conversation
set_session("previous-session-id")

# Get current session ID
current = get_session()
```

## Entity Management (Multi-User)

```python
from hindsight_litellm import configure, set_entity, get_entity

configure(bank_id="my-agent", hindsight_api_url="http://localhost:8888")

# Switch between users
set_entity("user-alice")
# ... Alice's conversations and memories

set_entity("user-bob")
# ... Bob's conversations and memories (isolated from Alice)
```

## Disabling and Cleanup

```python
from hindsight_litellm import disable, cleanup

# Temporarily disable memory integration
disable()

# Clean up all resources (call when shutting down)
cleanup()
```

## API Reference

### Main Functions

| Function | Description |
|----------|-------------|
| `configure(...)` | Configure global Hindsight settings |
| `enable()` | Enable memory integration with LiteLLM |
| `disable()` | Disable memory integration |
| `is_enabled()` | Check if memory integration is enabled |
| `cleanup()` | Clean up all resources |

### Configuration Functions

| Function | Description |
|----------|-------------|
| `get_config()` | Get current configuration |
| `is_configured()` | Check if Hindsight is configured |
| `reset_config()` | Reset configuration to defaults |

### Session/Entity Functions

| Function | Description |
|----------|-------------|
| `new_session()` | Generate and set a new session ID |
| `set_session(id)` | Set a specific session ID |
| `get_session()` | Get current session ID |
| `set_entity(id)` | Set entity ID for multi-user isolation |
| `get_entity()` | Get current entity ID |

### Recall Functions

| Function | Description |
|----------|-------------|
| `recall(query, ...)` | Synchronously query memories |
| `arecall(query, ...)` | Asynchronously query memories |

### Client Wrappers

| Function | Description |
|----------|-------------|
| `wrap_openai(client, ...)` | Wrap OpenAI client with memory |
| `wrap_anthropic(client, ...)` | Wrap Anthropic client with memory |

## Requirements

- Python >= 3.10
- litellm >= 1.40.0
- A running Hindsight API server

## License

MIT
