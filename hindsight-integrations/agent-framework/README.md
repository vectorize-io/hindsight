# hindsight-agent-framework

Persistent long-term memory for [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) via [Hindsight](https://github.com/vectorize-io/hindsight).

Microsoft Agent Framework is the successor to Semantic Kernel. This integration plugs Hindsight in as a **context provider**, so every agent run automatically **recalls** relevant memories into the agent's context and **retains** the conversation afterward — no MCP, and no tools the model has to remember to call.

## Installation

```bash
pip install hindsight-agent-framework
```

> ✨ **Recommended:** [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) — sign up free, get an API key, and skip self-hosting.

## Usage

```python
from agent_framework.openai import OpenAIChatClient
from hindsight_agent_framework import HindsightProvider

agent = OpenAIChatClient().as_agent(
    name="assistant",
    instructions="You are a helpful assistant.",
    context_providers=[HindsightProvider(bank_id="user-123")],
)

session = agent.create_session()
await agent.run("Remember that I prefer vegetarian food.", session=session)
# ...later, even in a new process:
await agent.run("Suggest a recipe.", session=session)  # recalls the preference
```

Set your API key once via the `HINDSIGHT_API_KEY` environment variable, or pass `api_key=`/`hindsight_api_url=` to `HindsightProvider`. For self-hosting:

```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-openai-key
hindsight-api  # http://localhost:8888
```

```python
HindsightProvider(bank_id="user-123", hindsight_api_url="http://localhost:8888")
```

## How It Works

| Hook | Behavior |
| --- | --- |
| `before_run` | Recall memories relevant to the user's message and inject them as a `## Memories` block in the agent's instructions. |
| `after_run` | Retain the user input + agent response so future runs build on them. |

Memories live in a Hindsight **bank** (one per user/agent/session — you choose). Recall and retain are best-effort: a memory hiccup never blocks the agent.

## Configuration

`HindsightProvider(bank_id, ...)` accepts: `client`, `hindsight_api_url`, `api_key`, `budget` (low/mid/high), `max_tokens`, `context`, `tags`, `recall_tags`, `recall_tags_match`, `mission` (creates the bank with a fact-extraction persona), `auto_recall`, `auto_retain`, `source_id`. You can also set process-wide defaults via `configure(...)`.

## Development

```bash
uv sync
uv run ruff check .
uv run pytest tests -v          # unit tests
uv run pytest tests -v -m requires_real_llm   # e2e (needs a live Hindsight server)
```
