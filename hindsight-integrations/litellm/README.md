# hindsight-litellm

Universal LLM memory integration via LiteLLM. Add persistent memory to any LLM application with just a few lines of code.

## Features

- **Universal LLM Support** - Works with 100+ LLM providers via LiteLLM (OpenAI, Anthropic, Groq, Azure, AWS Bedrock, Google Vertex AI, and more)
- **Simple Integration** - Just configure, set defaults, enable, and use `hindsight_litellm.completion()`
- **Automatic Memory Injection** - Relevant memories are injected into prompts before LLM calls
- **Automatic Conversation Storage** - Conversations are stored to Hindsight for future recall (async by default for performance)
- **Two Memory Modes** - Choose between `reflect` (synthesized context) or `recall` (raw memory retrieval)
- **Direct Memory APIs** - Query, synthesize, and store memories manually
- **Native Client Wrappers** - Alternative wrappers for OpenAI and Anthropic SDKs
- **Debug Mode** - Inspect exactly what memories are being injected
- **Async Error Tracking** - Check for background operation failures with `get_pending_retain_errors()`

## Installation

```bash
pip install hindsight-litellm
```

## Quick Start

```python
import hindsight_litellm

# Step 1: Configure static settings
hindsight_litellm.configure(
    hindsight_api_url="http://localhost:8888",
    verbose=True,
)

# Step 2: Set defaults (bank_id is required)
hindsight_litellm.set_defaults(
    bank_id="my-agent",
    use_reflect=True,  # Use reflect for synthesized context
)

# Step 3: Enable memory integration
hindsight_litellm.enable()

# Step 4: Use with explicit hindsight_query (required when inject_memories=True)
response = hindsight_litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What did we discuss about AI?"}],
    hindsight_query="What do I know about AI discussions?",  # Required!
)
```

**Important:** When `inject_memories=True` (default), you must provide `hindsight_query` to specify what to search for in memory. This ensures intentional, focused memory queries.

## How It Works

Here's what happens under the hood when you call `completion()`:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. YOUR CODE                                                               │
│  ───────────────────────────────────────────────────────────────────────── │
│  response = hindsight_litellm.completion(                                   │
│      model="gpt-4o-mini",                                                   │
│      messages=[{"role": "user", "content": "Help me with my Python project"}]│
│  )                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. MEMORY RETRIEVAL (before LLM call)                                      │
│  ───────────────────────────────────────────────────────────────────────── │
│  # hindsight_litellm queries Hindsight for relevant memories                │
│                                                                             │
│  # If use_reflect=False (default) - raw memories:                           │
│  memories = hindsight.recall(query="Help me with my Python project")        │
│  # Returns: ["User prefers pytest", "User is building a FastAPI app", ...]  │
│                                                                             │
│  # If use_reflect=True - synthesized context:                               │
│  context = hindsight.reflect(query="Help me with my Python project")        │
│  # Returns: "The user is an experienced Python developer working on..."     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. PROMPT INJECTION                                                        │
│  ───────────────────────────────────────────────────────────────────────── │
│  # Memories are injected into the system message:                           │
│                                                                             │
│  messages = [                                                               │
│      {"role": "system", "content": """                                      │
│          # Relevant Memories                                                │
│          1. [WORLD] User prefers pytest for testing                         │
│          2. [WORLD] User is building a FastAPI app                          │
│          3. [OPINION] User likes type hints                                 │
│      """},                                                                  │
│      {"role": "user", "content": "Help me with my Python project"}          │
│  ]                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. LLM CALL                                                                │
│  ───────────────────────────────────────────────────────────────────────── │
│  # The enriched prompt is sent to the LLM                                   │
│  response = litellm.completion(model="gpt-4o-mini", messages=messages)      │
│                                                                             │
│  # LLM now has context and can give personalized responses like:            │
│  # "Since you're working on your FastAPI app, here's how to add tests       │
│  #  with pytest..."                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. CONVERSATION STORAGE (after LLM call)                                   │
│  ───────────────────────────────────────────────────────────────────────── │
│  # The conversation is stored to Hindsight for future recall                │
│  hindsight.retain(                                                          │
│      content="User: Help me with my Python project\n"                       │
│              "Assistant: Since you're working on FastAPI..."                │
│  )                                                                          │
│  # Hindsight extracts facts: "User asked about Python project help"         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  6. RESPONSE RETURNED                                                       │
│  ───────────────────────────────────────────────────────────────────────── │
│  # You receive the response as normal                                       │
│  print(response.choices[0].message.content)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

The memory injection and storage happen automatically - you just use `completion()` as normal.

## Configuration Options

The API is split into two functions for clarity:

### 1. `configure()` - Static Settings

Settings that typically don't change during a session:

```python
hindsight_litellm.configure(
    # Required
    hindsight_api_url="http://localhost:8888",  # Hindsight API server URL

    # Optional - Authentication
    api_key="your-api-key",        # API key for Hindsight authentication

    # Optional - Memory behavior
    store_conversations=True,      # Store conversations after LLM calls
    inject_memories=True,          # Inject relevant memories into prompts
    sync_storage=False,            # False = async storage (default, better performance)
                                   # True = sync storage (blocks, raises errors immediately)

    # Optional - Advanced
    injection_mode="system_message",  # How to inject: "system_message" or "prepend_user"
    excluded_models=["gpt-3.5*"],     # Exclude certain models from interception
    verbose=True,                     # Enable verbose logging and debug info
)
```

### 2. `set_defaults()` - Per-Call Defaults

Default values for per-call settings. These can be overridden on individual calls using `hindsight_*` kwargs:

```python
hindsight_litellm.set_defaults(
    # Required
    bank_id="my-agent",            # Memory bank ID

    # Optional - Memory retrieval
    budget="mid",                  # Budget level: "low", "mid", "high"
    fact_types=["world", "opinion"],  # Filter fact types to retrieve
    max_memories=10,               # Maximum memories to inject (None = unlimited)
    max_memory_tokens=4096,        # Maximum tokens for memory context
    include_entities=True,         # Include entity observations in recall

    # Optional - Reflect mode
    use_reflect=True,              # Use reflect API (synthesized) vs recall (raw memories)
    reflect_include_facts=False,   # Include source facts in debug info
    reflect_context="I am a delivery agent finding recipients.",  # Context for reflect reasoning
    reflect_response_schema={...}, # JSON Schema for structured reflect output

    # Optional - Debugging
    trace=False,                   # Enable trace info for debugging
    document_id="conversation-1",  # Document ID for grouping conversations
)
```

### 3. Per-Call Overrides

Override any default on individual calls using `hindsight_*` kwargs:

```python
response = hindsight_litellm.completion(
    model="gpt-4o-mini",
    messages=[...],
    hindsight_query="Where is Alice located?",      # REQUIRED when inject_memories=True
    hindsight_reflect_context="Currently on floor 3",  # Per-call reflect context override
    # hindsight_bank_id="other-bank",               # Override bank_id for this call
)
```

### Bank Configuration: mission

Use `set_bank_mission()` to configure what the memory bank should learn and remember (used for mental models):

```python
hindsight_litellm.set_bank_mission(
    mission="""This agent routes customer support requests to the appropriate team.
    Remember which types of issues should go to which teams (billing, technical, sales).
    Track customer preferences for communication channels and past issue resolutions.""",
    name="Customer Support Router",  # Optional display name
)
```


### Memory Modes: Reflect vs Recall

- **Recall mode** (`use_reflect=False`, default): Retrieves raw memory facts and injects them as a numbered list. Best when you need precise, individual memories.
- **Reflect mode** (`use_reflect=True`): Synthesizes memories into a coherent context paragraph. Best for natural, conversational memory context.

```python
# Recall mode - raw memories
hindsight_litellm.set_defaults(bank_id="my-agent", use_reflect=False)
# Injects: "1. [WORLD] User prefers Python\n2. [OPINION] User dislikes Java..."

# Reflect mode - synthesized context
hindsight_litellm.set_defaults(bank_id="my-agent", use_reflect=True)
# Injects: "Based on previous conversations, the user is a Python developer who..."

# Reflect with context - shapes LLM reasoning (not retrieval)
hindsight_litellm.set_defaults(
    bank_id="my-agent",
    use_reflect=True,
    reflect_context="I am a delivery agent looking for package recipients.",
)
```

## Multi-Provider Support

Works with any LiteLLM-supported provider:

```python
import hindsight_litellm

hindsight_litellm.configure(hindsight_api_url="http://localhost:8888")
hindsight_litellm.set_defaults(bank_id="my-agent")
hindsight_litellm.enable()

messages = [{"role": "user", "content": "Hello!"}]

# OpenAI
hindsight_litellm.completion(model="gpt-4o", messages=messages, hindsight_query="greeting")

# Anthropic
hindsight_litellm.completion(model="claude-3-5-sonnet-20241022", messages=messages, hindsight_query="greeting")

# Groq
hindsight_litellm.completion(model="groq/llama-3.1-70b-versatile", messages=messages, hindsight_query="greeting")

# Azure OpenAI
hindsight_litellm.completion(model="azure/gpt-4", messages=messages, hindsight_query="greeting")

# AWS Bedrock
hindsight_litellm.completion(model="bedrock/anthropic.claude-3", messages=messages, hindsight_query="greeting")

# Google Vertex AI
hindsight_litellm.completion(model="vertex_ai/gemini-pro", messages=messages, hindsight_query="greeting")
```

## Direct Memory APIs

### Recall - Query raw memories

```python
from hindsight_litellm import configure, set_defaults, recall

configure(hindsight_api_url="http://localhost:8888")
set_defaults(bank_id="my-agent")

# Query memories
memories = recall("what projects am I working on?", budget="mid")
for m in memories:
    print(f"- [{m.fact_type}] {m.text}")

# Output:
# - [world] User is building a FastAPI project
# - [opinion] User prefers Python over JavaScript
```

### Reflect - Get synthesized context

```python
from hindsight_litellm import configure, set_defaults, reflect

configure(hindsight_api_url="http://localhost:8888")
set_defaults(bank_id="my-agent")

# Get synthesized memory context
result = reflect("what do you know about the user's preferences?")
print(result.text)

# Output:
# "Based on our conversations, the user prefers Python for backend development..."

# With context to shape the response (doesn't affect retrieval)
result = reflect(
    query="what do I know about Alice?",
    context="I am a delivery agent looking for package recipients.",
)
```

### Retain - Store memories

```python
from hindsight_litellm import configure, set_defaults, retain, get_pending_retain_errors

configure(hindsight_api_url="http://localhost:8888")
set_defaults(bank_id="my-agent")

# Async retain (default) - fast, non-blocking
# Returns immediately; actual storage happens in background
result = retain(
    content="User mentioned they're working on a machine learning project",
    context="Discussion about current projects",
)
# result.success is True immediately (actual errors collected separately)

# Sync retain - blocks until complete, raises errors immediately
result = retain(
    content="Critical information that must be stored",
    context="Important data",
    sync=True,  # Block until storage completes
)

# Check for async retain errors (call periodically)
errors = get_pending_retain_errors()
if errors:
    for e in errors:
        print(f"Background retain failed: {e}")
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
from hindsight_litellm import configure, set_defaults, enable, completion, get_last_injection_debug

configure(hindsight_api_url="http://localhost:8888", verbose=True)
set_defaults(bank_id="my-agent", use_reflect=True)
enable()

response = completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's my favorite color?"}],
    hindsight_query="What is the user's favorite color?",
)

# Inspect what was injected
debug = get_last_injection_debug()
if debug:
    print(f"Mode: {debug.mode}")           # "reflect" or "recall"
    print(f"Injected: {debug.injected}")   # True/False
    print(f"Results: {debug.results_count}")
    print(f"Memory context:\n{debug.memory_context}")
    if debug.error:
        print(f"Error: {debug.error}")
```

## Context Manager

```python
from hindsight_litellm import hindsight_memory
import litellm

with hindsight_memory(bank_id="user-123"):
    response = litellm.completion(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}],
        hindsight_query="greeting context",
    )
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
| `configure(...)` | Configure static Hindsight settings (API URL, auth, storage options) |
| `set_defaults(...)` | Set defaults for per-call settings (bank_id, budget, reflect options) |
| `enable()` | Enable memory integration with LiteLLM |
| `disable()` | Disable memory integration |
| `is_enabled()` | Check if memory integration is enabled |
| `cleanup()` | Clean up all resources |

### Configuration Functions

| Function | Description |
|----------|-------------|
| `get_config()` | Get current static configuration |
| `get_defaults()` | Get current per-call defaults |
| `is_configured()` | Check if Hindsight is configured with a bank_id |
| `reset_config()` | Reset all configuration to defaults |
| `set_document_id(id)` | Convenience function to update document_id |
| `set_bank_mission(...)` | Set mission/instructions for a memory bank (for mental models) |

### Memory Functions

| Function | Description |
|----------|-------------|
| `recall(query, ...)` | Query raw memories (sync) |
| `arecall(query, ...)` | Query raw memories (async) |
| `reflect(query, ...)` | Get synthesized memory context (sync) |
| `areflect(query, ...)` | Get synthesized memory context (async) |
| `retain(content, sync=False, ...)` | Store a memory (async by default, use `sync=True` to block) |
| `aretain(content, ...)` | Store a memory (async) |

### Error Tracking Functions

| Function | Description |
|----------|-------------|
| `get_pending_retain_errors()` | Get and clear errors from background retain operations |
| `get_pending_storage_errors()` | Get and clear errors from background conversation storage |

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

## License

MIT
