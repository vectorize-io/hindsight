# hindsight-omnigent

Persistent memory for [Omnigent](https://github.com/omnigent-ai/omnigent) agents via Hindsight.
Exposes Hindsight's **retain**, **recall**, and **reflect** operations as Omnigent
`type: function` tools — plain Python callables your agent declares in its YAML and the LLM
can call directly.

Omnigent invokes function tools with **no session context**, so the Hindsight bank and
connection are configured once per agent process (via `configure()` or `HINDSIGHT_*` env
vars) rather than per call.

## Features

- **Omnigent function tools** — `retain` / `recall` / `reflect` referenced by dotted path
- **Drop-in YAML** — `tools_yaml()` emits a ready-to-paste `tools:` block with correct schemas
- **Configure once** — set the URL, key, and bank globally or through env vars
- **System-prompt injection** — `memory_instructions()` pre-recalls memories for the agent prompt

## Installation

```bash
pip install hindsight-omnigent
```

## Quick start

> **Recommended: [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup)** — free tier, no self-hosting required.

**1. Configure Hindsight** in the Python module that hosts your agent's tools:

```python
# my_agent/tools.py
from hindsight_omnigent import configure

# Re-exports retain / recall / reflect so Omnigent can resolve them by dotted path.
from hindsight_omnigent.tools import recall, reflect, retain  # noqa: F401

configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",   # or set HINDSIGHT_API_KEY
    bank_id="user-123",  # or set HINDSIGHT_BANK_ID
)
```

**2. Declare the tools** in your `agent.yaml`. Generate the block with:

```python
from hindsight_omnigent import tools_yaml
print(tools_yaml())
```

```yaml
name: memory_agent
prompt: You are a helpful assistant with long-term memory. Use the Hindsight tools to remember and recall facts about the user.
executor:
  harness: claude-sdk
tools:
  hindsight_retain:
    type: function
    description: Store information in long-term memory for later retrieval.
    callable: hindsight_omnigent.tools.retain
    parameters: {"type": "object", "properties": {"content": {"type": "string", "description": "The information to store in long-term memory."}}, "required": ["content"]}
  hindsight_recall:
    type: function
    description: Search long-term memory for relevant information.
    callable: hindsight_omnigent.tools.recall
    parameters: {"type": "object", "properties": {"query": {"type": "string", "description": "The search query to find relevant memories."}}, "required": ["query"]}
  hindsight_reflect:
    type: function
    description: Synthesize a reasoned answer from long-term memories.
    callable: hindsight_omnigent.tools.reflect
    parameters: {"type": "object", "properties": {"query": {"type": "string", "description": "The question to reflect on using stored memories."}}, "required": ["query"]}
```

**3. Run it:**

```bash
omnigent run path/to/agent.yaml
```

> The `callable` paths must be importable from where Omnigent runs the agent. Importing
> `my_agent.tools` triggers the `configure()` call above; alternatively, skip `configure()`
> entirely and set `HINDSIGHT_API_KEY` / `HINDSIGHT_BANK_ID` in the agent's `os_env`.

## Configuration via environment variables

`configure()` is optional — the tools read these on first use:

| Variable             | Description                                  |
| -------------------- | -------------------------------------------- |
| `HINDSIGHT_API_URL`  | Hindsight API URL (default: Hindsight Cloud) |
| `HINDSIGHT_API_KEY`  | API key for authentication                   |
| `HINDSIGHT_BANK_ID`  | Memory bank the tools read/write             |

## `configure()` reference

| Parameter           | Default                                                | Description                                |
| ------------------- | ------------------------------------------------------ | ------------------------------------------ |
| `hindsight_api_url` | `HINDSIGHT_API_URL` env, else Hindsight Cloud          | Hindsight API URL                          |
| `api_key`           | `HINDSIGHT_API_KEY` env                                | API key for authentication                 |
| `bank_id`           | `HINDSIGHT_BANK_ID` env                                | Memory bank the tools read/write           |
| `budget`            | `"mid"`                                                | Recall/reflect budget level (low/mid/high) |
| `max_tokens`        | `4096`                                                 | Max tokens for recall results              |
| `tags`              | `None`                                                 | Tags applied when storing memories         |
| `recall_tags`       | `None`                                                 | Tags to filter when searching              |
| `recall_tags_match` | `"any"`                                                | Tag matching mode                          |
| `client`            | `None`                                                 | Pre-built Hindsight client (overrides URL/key) |

## Seeding the prompt

Omnigent has no pre-turn context hook, so to start the agent already knowing what Hindsight
remembers, pre-recall and splice it into the prompt:

```python
from hindsight_omnigent import memory_instructions

context = memory_instructions(query="what we know about the user", max_results=5)
prompt = f"You are a helpful assistant.\n\n{context}" if context else "You are a helpful assistant."
```

## Selecting tools

Generate only the tools you need:

```python
tools_yaml(enable_retain=True, enable_recall=True, enable_reflect=False)
```

## Requirements

- Python >= 3.10
- hindsight-client >= 0.4.0
- A running Hindsight API server (or Hindsight Cloud)
- [Omnigent](https://github.com/omnigent-ai/omnigent) (Python >= 3.12) to run the agent

> **Note:** Omnigent is an early-stage (alpha) project and its agent spec is still evolving.
> This integration targets the `type: function` tool contract in Omnigent 0.1.x.
