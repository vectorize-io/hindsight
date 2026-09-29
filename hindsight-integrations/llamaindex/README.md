# hindsight-llamaindex

LlamaIndex integration for [Hindsight](https://github.com/vectorize-io/hindsight) — persistent long-term memory for AI agents.

Provides two complementary patterns:

- **Tools** (`HindsightToolSpec`) — Agent-driven memory via LlamaIndex's `BaseToolSpec`. The agent decides when to retain/recall/reflect.
- **Memory** (`HindsightMemory`) — Automatic memory via LlamaIndex's `BaseMemory` interface. Messages are stored on every turn and recalled as context.

## Installation

```bash
pip install hindsight-llamaindex
```

## Quick Start: Agent Tools

```python
import asyncio
from hindsight_client import Hindsight
from hindsight_llamaindex import HindsightToolSpec
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent

async def main():
    client = Hindsight(base_url="http://localhost:8888")

    spec = HindsightToolSpec(
        client=client,
        bank_id="user-123",
        mission="Track user preferences",
    )
    tools = spec.to_tool_list()

    agent = ReActAgent(tools=tools, llm=OpenAI(model="gpt-4o"))
    response = await agent.run("Remember that I prefer dark mode")
    print(response)

asyncio.run(main())
```

## Quick Start: Automatic Memory

```python
from hindsight_client import Hindsight
from hindsight_llamaindex import HindsightMemory

client = Hindsight(base_url="http://localhost:8888")
memory = HindsightMemory.from_client(
    client=client,
    bank_id="user-123",
    mission="Track user preferences",
)

agent = ReActAgent(tools=tools, llm=llm, memory=memory)
```

## Configuration

```python
from hindsight_llamaindex import configure

configure(
    hindsight_api_url="http://localhost:8888",
    api_key="your-api-key",
    budget="mid",
    tags=["source:llamaindex"],
    context="my-app",
    mission="Track user preferences",
)
```

## Requirements

- Python 3.10+
- `llama-index-core >= 0.11.0`
- `hindsight-client >= 0.4.0`

## Documentation

- [Integration docs](https://hindsight.vectorize.io/sdks/integrations/llamaindex)
- [Hindsight API docs](https://docs.hindsight.vectorize.io)
