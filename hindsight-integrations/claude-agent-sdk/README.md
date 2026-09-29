# hindsight-claude-agent-sdk

Persistent memory tools and hooks for the [Claude Agent SDK](https://pypi.org/project/claude-agent-sdk/), powered by [Hindsight](https://github.com/vectorize-io/hindsight).

## Installation

```bash
pip install hindsight-claude-agent-sdk
```

## Quick Start

### Tools (explicit memory)

Give your Claude agent retain/recall/reflect tools so it can decide when to use memory:

```python
from claude_agent_sdk import query, ClaudeAgentOptions
from hindsight_claude_agent_sdk import create_hindsight_server

server = create_hindsight_server(
    bank_id="my-agent",
    hindsight_api_url="http://localhost:8888",
)

async for msg in query(
    prompt="Remember that I prefer dark mode. Then check what you know about me.",
    options=ClaudeAgentOptions(
        mcp_servers={"hindsight": server},
        allowed_tools=["mcp__hindsight__*"],
    ),
):
    print(msg)
```

### Hooks (automatic memory)

Auto-recall relevant memories before each prompt and auto-retain results after each session:

```python
from claude_agent_sdk import query, ClaudeAgentOptions
from hindsight_claude_agent_sdk import create_hindsight_server, create_memory_hooks

server = create_hindsight_server(bank_id="my-agent", hindsight_api_url="http://localhost:8888")
hooks = create_memory_hooks(bank_id="my-agent", hindsight_api_url="http://localhost:8888")

async for msg in query(
    prompt="Help me refactor the auth module.",
    options=ClaudeAgentOptions(
        mcp_servers={"hindsight": server},
        allowed_tools=["mcp__hindsight__*"],
        hooks=hooks,
    ),
):
    print(msg)
```

### Global configuration

```python
from hindsight_claude_agent_sdk import configure

configure(
    hindsight_api_url="http://localhost:8888",
    api_key="your-api-key",
    budget="mid",
)
```

## Hook Configuration

Fine-tune automatic memory behavior:

```python
from hindsight_claude_agent_sdk import MemoryHookConfig, create_memory_hooks

hooks = create_memory_hooks(
    bank_id="my-agent",
    hook_config=MemoryHookConfig(
        auto_recall=True,           # inject memories before each prompt
        auto_retain=True,           # save results after each session
        retain_on_tools=["Bash"],   # also retain notable Bash outputs
        recall_max_results=5,       # max memories to inject
        retain_tags=["source:my-app"],
    ),
)
```
