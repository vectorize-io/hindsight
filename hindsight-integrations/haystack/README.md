# hindsight-haystack

Haystack integration for [Hindsight](https://github.com/vectorize-io/hindsight) — persistent long-term memory for AI agents.

Provides Haystack `Tool` instances that give any Haystack `Agent` persistent memory via Hindsight's retain/recall/reflect APIs.

## Installation

```bash
pip install hindsight-haystack
```

## Quick Start

```python
from hindsight_client import Hindsight
from hindsight_haystack import create_hindsight_tools
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

client = Hindsight(base_url="http://localhost:8888")

tools = create_hindsight_tools(
    client=client,
    bank_id="user-123",
    mission="Track user preferences",
)

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
    tools=tools,
    system_prompt=(
        "You are a helpful assistant with long-term memory. "
        "Use retain_memory to store important facts. "
        "Use recall_memory to search memory before answering."
    ),
)

result = agent.run(messages=[ChatMessage.from_user("Remember that I prefer dark mode")])
print(result["messages"][-1].text)
```

## Automatic Memory with HindsightMemoryWrapper

For automatic recall and retain without relying on the agent to call tools:

```python
from hindsight_haystack import HindsightMemoryWrapper

toolset = HindsightMemoryWrapper(
    client=client,
    bank_id="user-123",
    mission="Track user preferences",
    auto_recall=True,   # Inject memories into system prompt before each turn
    auto_retain=True,    # Store user + assistant messages after each turn
)

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
    tools=toolset,
    system_prompt="You are a helpful assistant with long-term memory.",
)

# Use toolset.run() for automatic memory behavior
result = toolset.run(agent, messages=[ChatMessage.from_user("I prefer dark mode")])
```

## Selective Tools

```python
# Only retain + recall (no reflect)
tools = create_hindsight_tools(
    client=client,
    bank_id="user-123",
    include_reflect=False,
)
```

## Configuration

```python
from hindsight_haystack import configure

configure(
    hindsight_api_url="http://localhost:8888",
    api_key="your-api-key",
    budget="mid",
    tags=["source:haystack"],
    context="my-app",
    mission="Track user preferences",
)

# Now you can skip client= and url= arguments
tools = create_hindsight_tools(bank_id="user-123")
```

## Requirements

- Python 3.10+
- `haystack-ai >= 2.12.0`
- `hindsight-client >= 0.4.0`

## Documentation

- [Integration docs](https://hindsight.vectorize.io/sdks/integrations/haystack)
- [Hindsight API docs](https://docs.hindsight.vectorize.io)

