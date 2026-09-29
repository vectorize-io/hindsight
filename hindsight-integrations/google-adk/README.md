# hindsight-google-adk

Persistent long-term memory for [Google ADK](https://adk.dev/) agents via [Hindsight](https://vectorize.io/hindsight).

The package gives you two complementary patterns:

- **`HindsightMemoryService`** — Implements ADK's `BaseMemoryService`. Pass it to `Runner(memory_service=...)` and sessions are automatically retained on completion; agents calling `search_memory` get results back from Hindsight.
- **`create_hindsight_tools(...)`** — Returns a list of ADK `FunctionTool` instances (`hindsight_retain`, `hindsight_recall`, `hindsight_reflect`) the model can call inside a turn.

## Installation

```bash
pip install hindsight-google-adk
```

## Quick Start: Automatic Memory

```python
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from hindsight_google_adk import HindsightMemoryService

memory = HindsightMemoryService.from_url(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",
)

agent = LlmAgent(name="assistant", model="gemini-2.0-flash")

runner = Runner(
    app_name="my-app",
    agent=agent,
    session_service=InMemorySessionService(),
    memory_service=memory,
)
```

When a session ends, `Runner` calls `add_session_to_memory`, which retains the session's events to a Hindsight bank keyed by `(app_name, user_id)`. When the agent (or another call) invokes `search_memory(app_name, user_id, query)`, the integration runs a Hindsight recall and returns the results as ADK `MemoryEntry` objects.

## Quick Start: Explicit Tools

```python
from google.adk.agents import LlmAgent
from hindsight_google_adk import create_hindsight_tools

tools = create_hindsight_tools(
    bank_id="user-123",
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",
)

agent = LlmAgent(name="assistant", model="gemini-2.0-flash", tools=tools)
```

The agent now has three tools (toggle with `include_retain` / `include_recall` / `include_reflect`):

- `hindsight_retain(content)` — store information to long-term memory
- `hindsight_recall(query)` — search memory and return matches
- `hindsight_reflect(query)` — synthesize a coherent answer from memory

## Global Configuration

For shared defaults, call `configure(...)` once at startup:

```python
from hindsight_google_adk import configure

configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key=None,                      # falls back to HINDSIGHT_API_KEY env var
    bank_id_template="{app_name}::{user_id}",
    budget="mid",
    max_tokens=4096,
)
```

Subsequent `HindsightMemoryService.from_url()` and `create_hindsight_tools()` calls use the global config as a fallback.

## Bank ID Derivation

By default, each `(app_name, user_id)` pair gets its own bank: `"{app_name}::{user_id}"`. Override with `bank_id_template`:

```python
# Per-user bank shared across apps
HindsightMemoryService.from_url(..., bank_id_template="user::{user_id}")

# Static bank shared across all users
HindsightMemoryService.from_url(..., bank_id_template="my-shared-bank")
```

## Configuration Reference

| Argument | Default | Description |
|---|---|---|
| `hindsight_api_url` | `https://api.hindsight.vectorize.io` | Hindsight API URL (Cloud by default). |
| `api_key` | `HINDSIGHT_API_KEY` env | Bearer token for Hindsight Cloud. |
| `bank_id_template` | `"{app_name}::{user_id}"` | Format string for deriving the bank id. |
| `budget` | `"mid"` | Recall budget: `low`/`mid`/`high`. |
| `max_tokens` | `4096` | Max tokens for recall results. |
| `tags` | `None` | Extra tags added to retains. `app:` and `user:` are always added. |
| `recall_tags` | `None` | Extra tags appended to recall queries. `user:` is always added. |
| `recall_tags_match` | `"any"` | Tag match mode: `any` / `all` / `any_strict` / `all_strict`. |
| `mission` | `None` | If set, the bank is created (idempotent) on first use with this fact-extraction mission. |
| `context` | `"google-adk"` | Provenance label attached to retained content. |

## Memory Scoping with Tags

Each retained session/event carries `app:<app_name>` and `user:<user_id>` tags. Recall queries automatically include `user:<user_id>` so users never see each other's memories. Pass `tags=[...]` to extend the retain set or `recall_tags=[...]` to filter recall further.

## Connection Modes

### Hindsight Cloud (recommended)

```python
HindsightMemoryService.from_url(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",
)
```

### Self-hosted

```python
HindsightMemoryService.from_url(hindsight_api_url="http://localhost:8888")
```

## Error Handling

All `add_*` and `search_memory` methods are resilient: Hindsight failures are logged but never propagate to the `Runner`. The explicit tools raise `HindsightError` on failure so the agent can react.

## Requirements

- Python 3.10+
- `google-adk>=2.0`
- `hindsight-client>=0.4.0`

## License

MIT
