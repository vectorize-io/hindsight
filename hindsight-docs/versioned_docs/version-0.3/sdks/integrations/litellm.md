---
sidebar_position: 1
---

# LiteLLM

Universal LLM memory integration via [LiteLLM](https://github.com/BerriAI/litellm). Add persistent memory to any LLM application with just a few lines of code.

## Features

- **Universal LLM Support** - Works with 100+ LLM providers via LiteLLM (OpenAI, Anthropic, Groq, Azure, AWS Bedrock, Google Vertex AI, and more)
- **Simple Integration** - Just configure, enable, and use `hindsight_litellm.completion()`
- **Automatic Memory Injection** - Relevant memories are injected into prompts before LLM calls
- **Automatic Conversation Storage** - Conversations are stored to Hindsight for future recall
- **Two Memory Modes** - Choose between `reflect` (synthesized context) or `recall` (raw memory retrieval)
- **Direct Memory APIs** - Query, synthesize, and store memories manually
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
)
hindsight_litellm.enable()

# Use the convenience wrapper - memory is automatically injected and stored
response = hindsight_litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What did we discuss about AI?"}]
)
```

## How It Works

When you call `completion()`, the following happens automatically:

1. **Memory Retrieval** - Hindsight is queried for relevant memories based on the conversation
2. **Prompt Injection** - Memories are injected into the system message
3. **LLM Call** - The enriched prompt is sent to the LLM
4. **Conversation Storage** - The conversation is stored to Hindsight for future recall
5. **Response Returned** - You receive the response as normal

## Configuration Options

```python
hindsight_litellm.configure(
    # Required
    hindsight_api_url="http://localhost:8888",  # Hindsight API server URL
    bank_id="my-agent",                          # Memory bank ID

    api_key="your-api-key",        # Optional API key for authentication

    # Optional - Memory behavior
    store_conversations=True,      # Store conversations after LLM calls
    inject_memories=True,          # Inject relevant memories into prompts
    use_reflect=False,             # Use reflect API (synthesized) vs recall (raw memories)
    reflect_include_facts=False,   # Include source facts with reflect responses
    max_memories=None,             # Maximum memories to inject (None = unlimited)
    max_memory_tokens=4096,        # Maximum tokens for memory context
    recall_budget="mid",           # Recall budget: "low", "mid", "high"
    fact_types=["world", "agent"], # Filter fact types to inject

    # Optional - Bank Configuration
    bank_name="My Agent",          # Human-readable display name for the memory bank
    background="This agent...",    # Instructions guiding what Hindsight should remember

    # Optional - Advanced
    injection_mode="system_message",  # or "prepend_user"
    excluded_models=["gpt-3.5*"],     # Exclude certain models
    verbose=True,                     # Enable verbose logging and debug info
)
```

### Bank Configuration

The `background` and `bank_name` parameters configure the memory bank itself. When provided, `configure()` will automatically create or update the bank with these settings.

```python
hindsight_litellm.configure(
    hindsight_api_url="http://localhost:8888",
    bank_id="support-router",
    bank_name="Customer Support Router",
    background="""This agent routes customer support requests to the appropriate team.
    Remember which types of issues should go to which teams (billing, technical, sales).
    Track customer preferences for communication channels and past issue resolutions.""",
)
```

### Memory Modes: Reflect vs Recall

- **Recall mode** (`use_reflect=False`, default): Retrieves raw memory facts and injects them as a numbered list. Best when you need precise, individual memories.
- **Reflect mode** (`use_reflect=True`): Synthesizes memories into a coherent context paragraph. Best for natural, conversational memory context.

```python
# Recall mode - raw memories
hindsight_litellm.configure(
    bank_id="my-agent",
    use_reflect=False,  # Default
)
# Injects: "1. [WORLD] User prefers Python\n2. [OPINION] User dislikes Java..."

# Reflect mode - synthesized context
hindsight_litellm.configure(
    bank_id="my-agent",
    use_reflect=True,
)
# Injects: "Based on previous conversations, the user is a Python developer who..."
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

## Direct Memory APIs

### Recall - Query raw memories

```python
from hindsight_litellm import configure, recall

configure(bank_id="my-agent", hindsight_api_url="http://localhost:8888")

memories = recall("what projects am I working on?", budget="mid")
for m in memories:
    print(f"- [{m.fact_type}] {m.text}")
```

### Reflect - Get synthesized context

```python
from hindsight_litellm import configure, reflect

configure(bank_id="my-agent", hindsight_api_url="http://localhost:8888")

result = reflect("what do you know about the user's preferences?")
print(result.text)
```

### Retain - Store memories

```python
from hindsight_litellm import configure, retain

configure(bank_id="my-agent", hindsight_api_url="http://localhost:8888")

result = retain(
    content="User mentioned they're working on a machine learning project",
    context="Discussion about current projects",
)
```

### Async APIs

```python
from hindsight_litellm import arecall, areflect, aretain

# Async versions of all memory APIs
memories = await arecall("what do you know about me?")
context = await areflect("summarize user preferences")
result = await aretain(content="New information to remember")
```

## Native Client Wrappers

Alternative to LiteLLM callbacks for direct SDK integration.

### OpenAI Wrapper

```python
from openai import OpenAI
from hindsight_litellm import wrap_openai

client = OpenAI()
wrapped = wrap_openai(
    client,
    bank_id="my-agent",
    hindsight_api_url="http://localhost:8888",
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

## Debug Mode

When `verbose=True`, you can inspect exactly what memories are being injected:

```python
from hindsight_litellm import configure, enable, completion, get_last_injection_debug

configure(
    bank_id="my-agent",
    hindsight_api_url="http://localhost:8888",
    verbose=True,
)
enable()

response = completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's my favorite color?"}]
)

# Inspect what was injected
debug = get_last_injection_debug()
if debug:
    print(f"Mode: {debug.mode}")           # "reflect" or "recall"
    print(f"Injected: {debug.injected}")   # True/False
    print(f"Results: {debug.results_count}")
    print(f"Memory context:\n{debug.memory_context}")
```

## Context Manager

```python
from hindsight_litellm import hindsight_memory
import litellm

with hindsight_memory(bank_id="user-123"):
    response = litellm.completion(model="gpt-4", messages=[...])
# Memory integration automatically disabled after context
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

### Memory Functions

| Function | Description |
|----------|-------------|
| `recall(query, ...)` | Synchronously query raw memories |
| `arecall(query, ...)` | Asynchronously query raw memories |
| `reflect(query, ...)` | Synchronously get synthesized memory context |
| `areflect(query, ...)` | Asynchronously get synthesized memory context |
| `retain(content, ...)` | Synchronously store a memory |
| `aretain(content, ...)` | Asynchronously store a memory |

### Debug Functions

| Function | Description |
|----------|-------------|
| `get_last_injection_debug()` | Get debug info from last memory injection |
| `clear_injection_debug()` | Clear stored debug info |

### Client Wrappers

| Function | Description |
|----------|-------------|
| `wrap_openai(client, ...)` | Wrap OpenAI client with memory |
| `wrap_anthropic(client, ...)` | Wrap Anthropic client with memory |

## Requirements

- Python >= 3.10
- litellm >= 1.40.0
- A running Hindsight API server
