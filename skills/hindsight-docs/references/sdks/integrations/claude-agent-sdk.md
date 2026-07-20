
# Claude Agent SDK

Persistent long-term memory for Anthropic's [Claude Agent SDK](https://pypi.org/project/claude-agent-sdk/) via [Hindsight](https://vectorize.io/hindsight). Expose retain/recall/reflect as MCP tools so the agent decides when to use memory, or wire up hooks to inject relevant memories automatically before every turn.

## Quick Start

> **💡 Recommended: Hindsight Cloud**
>
[Sign up free](https://ui.hindsight.vectorize.io/signup) and grab an API key — no self-hosting required.
```bash
pip install hindsight-claude-agent-sdk
```

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

## Features

- **Memory Tools** — retain, recall, and reflect exposed as MCP tools the agent can call on its own
- **Automatic Hooks** — inject relevant memories into context before each turn and retain conversation content after, with no explicit tool calls
- **Per-Agent Banks** — isolate memory per agent or user with a `bank_id`
- **Cloud or Self-Hosted** — point at Hindsight Cloud or your own Hindsight deployment

## Learn More

- Claude Agent SDK cookbook recipe
- [Source on GitHub](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/claude-agent-sdk)
