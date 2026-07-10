
# Composio

Persistent memory for [Composio](https://composio.dev) agents via Hindsight. Exposes Hindsight's
**retain**, **recall**, and **reflect** operations as Composio in-process custom tools your agent
can call directly.

The Hindsight memory bank for each call is the Composio session's `user_id`, so a single
registered tool set isolates memory per user automatically.

## Features

- **Composio Custom Tools** — registered via `composio.experimental.tool()` and bound to a session
- **Per-user memory** — the session `user_id` maps to the Hindsight `bank_id` (with a configurable fallback bank)
- **Three Memory Tools** — Retain (store), Recall (search), Reflect (synthesize) — include any combination
- **Simple Configuration** — pass a client/URL directly, or `configure()` once globally

## Installation

```bash
pip install hindsight-composio
```

## Quick Start

> **💡 Recommended: Hindsight Cloud**
>
[Sign up free](https://ui.hindsight.vectorize.io/signup) and grab an API key — no self-hosting required.
```python
from composio import Composio
from hindsight_composio import register_hindsight_tools

composio = Composio()  # uses COMPOSIO_API_KEY

tools = register_hindsight_tools(
    composio,
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",  # or set HINDSIGHT_API_KEY env var
)

session = composio.create(
    user_id="user-123",  # becomes the Hindsight bank_id
    experimental={"custom_tools": tools},
)

# Pass session.tools() to your agent/LLM as usual.
```

The session now has three tools the agent can call:

- **`HINDSIGHT_RETAIN`** — Store information to long-term memory
- **`HINDSIGHT_RECALL`** — Search long-term memory for relevant facts
- **`HINDSIGHT_REFLECT`** — Synthesize a reasoned answer from memories

## Bank selection

The bank is resolved per call:

1. The session's `user_id` (recommended — one tool set, isolated per user).
2. `default_bank` (passed to `register_hindsight_tools` or `configure`) when a call has no `user_id`.

If neither is available, the tool raises an error.

```python
tools = register_hindsight_tools(
    composio,
    hindsight_api_url="https://api.hindsight.vectorize.io",
    default_bank="shared",  # used only when a session has no user_id
)
```

### Self-hosting (local development)

If you're running Hindsight locally with `./scripts/dev/start-api.sh`, swap the URL:

```python
tools = register_hindsight_tools(
    composio,
    hindsight_api_url="http://localhost:8888",
)
```

See the [installation guide](../../developer/installation.md) for self-hosting setup.

## Selecting tools

Include only the tools you need:

```python
tools = register_hindsight_tools(
    composio,
    hindsight_api_url="https://api.hindsight.vectorize.io",
    enable_retain=True,
    enable_recall=True,
    enable_reflect=False,  # omit reflect
)
```

## Global configuration

Instead of passing connection details every time, configure once:

```python
from hindsight_composio import configure, register_hindsight_tools

configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="your-api-key",       # or set HINDSIGHT_API_KEY env var
    default_bank="shared",         # fallback bank when no user_id
    budget="mid",                  # recall/reflect budget: low/mid/high
    max_tokens=4096,               # max tokens for recall results
    tags=["env:prod"],             # tags for stored memories
    recall_tags=["scope:global"],  # tags to filter recall
    recall_tags_match="any",       # any/all/any_strict/all_strict
)

tools = register_hindsight_tools(composio)
```

## Configuration Reference

### `register_hindsight_tools()`

| Parameter           | Default    | Description                                      |
| ------------------- | ---------- | ------------------------------------------------ |
| `composio`          | _required_ | The `Composio` instance (provides the decorator) |
| `client`            | `None`     | Pre-configured Hindsight client                  |
| `hindsight_api_url` | `None`     | API URL (used if no client provided)             |
| `api_key`           | `None`     | API key (used if no client provided)             |
| `default_bank`      | `None`     | Bank used when a call has no Composio `user_id`  |
| `budget`            | `"mid"`    | Recall/reflect budget level (low/mid/high)       |
| `max_tokens`        | `4096`     | Maximum tokens for recall results                |
| `tags`              | `None`     | Tags applied when storing memories               |
| `recall_tags`       | `None`     | Tags to filter when searching                    |
| `recall_tags_match` | `"any"`    | Tag matching mode                                |
| `enable_retain`     | `True`     | Include the retain (store) tool                  |
| `enable_recall`     | `True`     | Include the recall (search) tool                 |
| `enable_reflect`    | `True`     | Include the reflect (synthesize) tool            |

### `configure()`

| Parameter           | Default                                                | Description                                   |
| ------------------- | ------------------------------------------------------ | --------------------------------------------- |
| `hindsight_api_url` | Hindsight Cloud (`https://api.hindsight.vectorize.io`) | Hindsight API URL                             |
| `api_key`           | `HINDSIGHT_API_KEY` env                                | API key for authentication                    |
| `default_bank`      | `None`                                                 | Fallback bank when a session has no `user_id` |
| `budget`            | `"mid"`                                                | Default recall/reflect budget level           |
| `max_tokens`        | `4096`                                                 | Default max tokens for recall                 |
| `tags`              | `None`                                                 | Default tags for retain operations            |
| `recall_tags`       | `None`                                                 | Default tags to filter recall                 |
| `recall_tags_match` | `"any"`                                                | Default tag matching mode                     |

## Requirements

- Python >= 3.10
- composio >= 0.13.1, < 1
- hindsight-client >= 0.4.0
- A running Hindsight API server (or Hindsight Cloud)

> **📝 Note**
>
Composio's custom-tools API is currently experimental. This integration targets the Composio
0.13.x SDK (`from composio import Composio`) and intentionally excludes the in-progress `1.0.0`
rewrite, whose API differs.
