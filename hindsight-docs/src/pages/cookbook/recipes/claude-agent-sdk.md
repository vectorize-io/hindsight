---
sidebar_position: 12
---

# Claude Agent SDK with Persistent Memory


:::tip Run this notebook
This recipe is available as an interactive Jupyter notebook.
[**Open in GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/blob/main/notebooks/claude-agent-sdk.ipynb)
:::


Build a Claude agent that remembers across sessions using Hindsight memory tools and automatic hooks.

## Features
- In-process MCP server with retain, recall, and reflect tools
- Automatic memory hooks that inject context before each prompt
- Auto-retain agent results for future sessions
- Knowledge that compounds over repeated runs

## Prerequisites
- **Claude Code CLI** installed and authenticated (`npm install -g @anthropic-ai/claude-code && claude auth login`, or set `ANTHROPIC_API_KEY`)
- An LLM API key for Hindsight (OpenAI, Gemini, etc.)
- Hindsight running locally via Docker (see setup below)
- Alternatively, a [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) account (no Docker needed)

:::note
The Claude Agent SDK runs the Claude Code CLI as a subprocess. You need the CLI installed **and** authenticated — either via `claude auth login` or by setting `ANTHROPIC_API_KEY` in your environment.
:::

## Start Hindsight Locally

Before running this notebook, start Hindsight in a terminal:

```bash
export LLM_API_KEY="your-llm-api-key"

docker run --rm -it --pull always -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_API_KEY=$LLM_API_KEY \
  -e HINDSIGHT_API_LLM_MODEL=gpt-4o-mini \
  -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

## 1. Install Dependencies


```python
!pip install -q hindsight-claude-agent-sdk nest-asyncio
```

## 2. Configure Environment


```python
import nest_asyncio
nest_asyncio.apply()

import os
import getpass

# Anthropic API key (used by Claude Agent SDK)
if not os.getenv("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter your Anthropic API key: ")

# Hindsight connection (defaults to local self-hosted instance)
# For Hindsight Cloud, set HINDSIGHT_API_URL and HINDSIGHT_API_KEY env vars
HINDSIGHT_API_URL = os.getenv("HINDSIGHT_API_URL", "http://localhost:8888")
HINDSIGHT_API_KEY = os.getenv("HINDSIGHT_API_KEY", None)  # optional, for Hindsight Cloud
BANK_ID = "claude-agent-demo"

print(f"Hindsight API: {HINDSIGHT_API_URL}")
print(f"Using API key: {'yes' if HINDSIGHT_API_KEY else 'no (self-hosted)'}")
print(f"Bank ID: {BANK_ID}")
```

## 3. Create a Memory Bank


```python
from hindsight_client import Hindsight

hindsight = Hindsight(base_url=HINDSIGHT_API_URL, api_key=HINDSIGHT_API_KEY)

# Create a dedicated bank for this demo (safe to re-run)
try:
    hindsight.create_bank(
        bank_id=BANK_ID,
        name="Claude Agent Demo",
        mission="Remember user preferences, decisions, and project context for a software development assistant.",
    )
    print(f"Bank '{BANK_ID}' created.")
except Exception:
    print(f"Bank '{BANK_ID}' already exists, continuing.")
```

## 4. Set Up Memory Tools

Create an in-process MCP server with retain, recall, and reflect tools:


```python
from hindsight_claude_agent_sdk import create_hindsight_server

server = create_hindsight_server(
    bank_id=BANK_ID,
    hindsight_api_url=HINDSIGHT_API_URL,
    api_key=HINDSIGHT_API_KEY,
    tags=["source:claude-agent-sdk-demo"],
)

print("Hindsight MCP server created with tools: hindsight_retain, hindsight_recall, hindsight_reflect")
```

## 5. Run Agent with Explicit Memory Tools

The agent decides when to store and retrieve memories:


```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions

async def run_agent(prompt: str, system: str = None):
    """Run a Claude agent with Hindsight memory tools."""
    options = ClaudeAgentOptions(
        mcp_servers={"hindsight": server},
        allowed_tools=["mcp__hindsight__*"],
        model="claude-sonnet-4-6",
        permission_mode="bypassPermissions",
    )
    if system:
        options.system_prompt = system

    result_text = None
    async for msg in query(prompt=prompt, options=options):
        if hasattr(msg, "result"):
            result_text = msg.result
    return result_text


# Store some preferences
result = asyncio.get_event_loop().run_until_complete(
    run_agent(
        "Store the following into memory using the retain tool:\n"
        "- I prefer Python with type hints and async/await patterns\n"
        "- My team uses pytest for testing with pytest-asyncio\n"
        "- We follow conventional commits (feat:, fix:, chore:)\n"
        "- Our API framework is FastAPI with Pydantic v2 models"
    )
)
print("Agent result:", result)
```

## 6. Recall Memories in a New Session

Simulate a fresh session — the agent has no conversation history, but can recall from memory:


```python
# New session — no prior context
result = asyncio.get_event_loop().run_until_complete(
    run_agent(
        "What testing framework does my team use? "
        "Search your memory first before answering.",
        system="You are a helpful coding assistant. Always check memory before answering questions about the user.",
    )
)
print("Agent result:", result)
```

## 7. Reflect for Deeper Synthesis

Use reflect when you need reasoned analysis across all stored memories:


```python
result = asyncio.get_event_loop().run_until_complete(
    run_agent(
        "Use the reflect tool to synthesize everything you know about my development stack and preferences.",
    )
)
print("Agent result:", result)
```

## 8. Add Automatic Memory Hooks

Hooks inject memory automatically — no explicit tool calls needed:


```python
from hindsight_claude_agent_sdk import create_memory_hooks, MemoryHookConfig

hooks = create_memory_hooks(
    bank_id=BANK_ID,
    hindsight_api_url=HINDSIGHT_API_URL,
    api_key=HINDSIGHT_API_KEY,
    hook_config=MemoryHookConfig(
        auto_recall=True,       # inject relevant memories before each prompt
        auto_retain=True,       # save agent results after each session
        recall_max_results=5,   # limit injected memories
    ),
)

print("Memory hooks created: auto-recall on UserPromptSubmit, auto-retain on Stop")
```

## 9. Run Agent with Hooks

Now the agent gets relevant memories injected as system context automatically:


```python
async def run_with_hooks(prompt: str):
    """Run a Claude agent with both tools and hooks."""
    result_text = None
    async for msg in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            mcp_servers={"hindsight": server},
            allowed_tools=["mcp__hindsight__*"],
            hooks=hooks,
            model="claude-sonnet-4-6",
            permission_mode="bypassPermissions",
            system_prompt="You are a helpful coding assistant.",
        ),
    ):
        if hasattr(msg, "result"):
            result_text = msg.result
    return result_text


# The agent receives past memories automatically — no tool call needed
result = asyncio.get_event_loop().run_until_complete(
    run_with_hooks("Write a sample pytest test for a FastAPI endpoint, using my team's preferred patterns.")
)
print("Agent result:", result)
```

The agent received your team's testing preferences via auto-recall before it even started working. And its result was auto-retained for future sessions.

## 10. Run Again to See Knowledge Compound

Each session adds to the knowledge base. Run the agent again with a related prompt:


```python
result = asyncio.get_event_loop().run_until_complete(
    run_with_hooks("What commit message format should I use for this test file I just created?")
)
print("Agent result:", result)
```

The agent recalls the conventional commits preference from earlier — even though it was stored in a completely different session.

## 11. Auto-Retain Tool Outputs (Optional)

You can also auto-retain outputs from specific tools like Bash:


```python
hooks_with_bash = create_memory_hooks(
    bank_id=BANK_ID,
    hindsight_api_url=HINDSIGHT_API_URL,
    api_key=HINDSIGHT_API_KEY,
    hook_config=MemoryHookConfig(
        auto_recall=True,
        auto_retain=True,
        retain_on_tools=["Bash"],  # remember Bash command outputs
        retain_tags=["source:cli"],
    ),
)

print("Hooks created with Bash output retention enabled")
```

## Cleanup

Delete the bank created during this notebook:


```python
hindsight.delete_bank(bank_id=BANK_ID)
print(f"Deleted bank '{BANK_ID}'.")
```
