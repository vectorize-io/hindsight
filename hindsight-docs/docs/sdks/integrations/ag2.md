---
sidebar_position: 8
---

# AG2

Persistent long-term memory for [AG2](https://ag2.ai) agents (community AutoGen fork). Give your agents retain/recall/reflect tools that persist across conversations.

[View Changelog →](/changelog/integrations/ag2)

## Features

- **Drop-in Tools** — `register_hindsight_tools()` registers retain, recall, and reflect in one line
- **AG2-native** — Uses `Annotated` type hints compatible with AG2's `@register_for_llm` / `@register_for_execution` pattern
- **GroupChat Support** — Multiple agents can share a single memory bank
- **Selective Tools** — Include only the tools you need (`include_retain`, `include_recall`, `include_reflect`)
- **Simple Configuration** — Configure once globally or override per tool set

## Installation

```bash
pip install hindsight-ag2
```

## Quick Start

```python
from autogen import AssistantAgent, UserProxyAgent, LLMConfig
from hindsight_ag2 import register_hindsight_tools

llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")

with llm_config:
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant with long-term memory.",
    )
    user_proxy = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
    )

# Register Hindsight memory tools on both agents
register_hindsight_tools(
    assistant, user_proxy,
    bank_id="my-bank",
    hindsight_api_url="http://localhost:8888",
)

# The assistant can now use hindsight_retain, hindsight_recall, hindsight_reflect
result = user_proxy.initiate_chat(
    assistant,
    message="Remember that I prefer Python over JavaScript.",
)
```

That's it. The assistant can now store and retrieve memories across conversations.

## How It Works

The integration provides three AG2-compatible tool functions backed by Hindsight's API:

| Tool | Hindsight | What happens |
|------|-----------|--------------|
| `hindsight_retain(content)` | `retain(bank_id, content, ...)` | Content is stored. Hindsight extracts facts, entities, and relationships from the raw text. |
| `hindsight_recall(query)` | `recall(bank_id, query, ...)` | Hindsight runs semantic search, BM25, graph traversal, and reranking. Returns a numbered list of matching memories. |
| `hindsight_reflect(query)` | `reflect(bank_id, query, ...)` | Hindsight synthesizes a reasoned answer from all relevant memories, using the bank's disposition traits. |

Tools are plain Python functions with `Annotated` type hints. AG2 uses these hints to generate the tool schema that the LLM sees.

## Configuration

### Global Configuration

```python
from hindsight_ag2 import configure

configure(
    hindsight_api_url="http://localhost:8888",
    api_key="your-key",       # or set HINDSIGHT_API_KEY env var
    budget="mid",              # low / mid / high
    max_tokens=4096,
    tags=["source:ag2"],       # default tags for retain
)
```

### Per-Tool Overrides

Constructor arguments override global configuration:

```python
from hindsight_ag2 import create_hindsight_tools

tools = create_hindsight_tools(
    bank_id="my-bank",
    hindsight_api_url="http://localhost:8888",
    budget="high",
    max_tokens=8192,
    tags=["team:alpha"],
)
```

## GroupChat with Shared Memory

Multiple agents can share a single memory bank in a GroupChat:

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, LLMConfig
from hindsight_ag2 import register_hindsight_tools

llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")

with llm_config:
    researcher = AssistantAgent(name="researcher", system_message="You research topics.")
    writer = AssistantAgent(name="writer", system_message="You write content.")
    executor = UserProxyAgent(name="executor", human_input_mode="NEVER")

# All agents share the same memory bank
for agent in [researcher, writer]:
    register_hindsight_tools(agent, executor, bank_id="team-memory")

group_chat = GroupChat(agents=[researcher, writer, executor], messages=[])
manager = GroupChatManager(groupchat=group_chat)
```

## Manual Registration

For full control over how tools are registered:

```python
from hindsight_ag2 import create_hindsight_tools

tools = create_hindsight_tools(
    bank_id="my-bank",
    hindsight_api_url="http://localhost:8888",
)
for tool_fn in tools:
    assistant.register_for_llm(description=tool_fn.__doc__)(tool_fn)
    user_proxy.register_for_execution()(tool_fn)
```

## API Reference

### Configuration

| Function | Description |
|----------|-------------|
| `configure(...)` | Set global connection and default settings |
| `get_config()` | Get current configuration |
| `reset_config()` | Reset configuration to None |

### create_hindsight_tools

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bank_id` | required | Hindsight memory bank ID |
| `client` | `None` | Pre-configured `Hindsight` client |
| `hindsight_api_url` | from config | Hindsight API URL |
| `api_key` | from config | API key |
| `budget` | `"mid"` | Recall/reflect budget (low/mid/high) |
| `max_tokens` | `4096` | Max tokens for recall results |
| `tags` | `None` | Tags applied when storing memories |
| `recall_tags` | `None` | Tags to filter when searching |
| `recall_tags_match` | `"any"` | Tag matching mode (any/all/any_strict/all_strict) |
| `retain_metadata` | `None` | Metadata dict for retain operations |
| `retain_document_id` | `None` | Document ID for retain (groups/upserts memories) |
| `recall_types` | `None` | Fact types to filter (world, experience, opinion, observation) |
| `recall_include_entities` | `False` | Include entity information in recall results |
| `reflect_context` | `None` | Additional context for reflect operations |
| `reflect_max_tokens` | `max_tokens` | Max tokens for reflect results |
| `reflect_response_schema` | `None` | JSON schema to constrain reflect output format |
| `reflect_tags` | `recall_tags` | Tags to filter memories used in reflect |
| `reflect_tags_match` | `recall_tags_match` | Tag matching for reflect |
| `include_retain` | `True` | Include the retain tool |
| `include_recall` | `True` | Include the recall tool |
| `include_reflect` | `True` | Include the reflect tool |

## Requirements

- Python >= 3.10
- ag2 >= 0.9.0
- A running Hindsight API server
