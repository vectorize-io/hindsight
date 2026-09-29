---
sidebar_position: 2
---

# Programmatic API (Python)

The `hindsight-all` Python package lets your code spawn and manage a local Hindsight daemon without deploying any server infrastructure. It bundles the Hindsight API server, embedded PostgreSQL, and the Python client into one install — `pip install hindsight-all` and you can start a fully-functional Hindsight instance from a few lines of Python.

The daemon runs as a **separate OS process** on `127.0.0.1` (not in your Python process memory). Your code talks to it over HTTP via the bundled `HindsightClient`.

If you already have a Hindsight server running and just need a client, use [Python Client (hindsight-client)](./python.md) instead.

## How it works

`hindsight-all` exposes two primary APIs:

- **`HindsightServer`** — explicit lifecycle. Use it as a context manager when you want deterministic startup/shutdown (e.g. in tests).
- **`HindsightEmbedded`** — auto-managed. Starts a daemon on first use, reuses it across calls, shuts it down after an idle timeout. Easiest for application code that doesn't want to think about lifecycle.

Both end up talking to the same underlying daemon via the same `HindsightClient` HTTP interface — the difference is only how the server process is managed.

## Installation

```bash
pip install hindsight-all
```

The `hindsight-all` wheel bundles `hindsight-api-slim`, `hindsight-client`, and `hindsight-embed` as dependencies, so one `pip install` gets you everything.

## `HindsightServer` — explicit lifecycle

Use `HindsightServer` as a context manager when you want the server to start immediately, run for the duration of a block, and shut down cleanly afterwards. Ideal for tests and short-lived scripts.

```python
import os
from hindsight import HindsightServer, HindsightClient

with HindsightServer(
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    llm_api_key=os.environ["OPENAI_API_KEY"],
) as server:
    client = HindsightClient(base_url=server.url)

    client.retain(bank_id="my-bank", content="Alice works at Google")
    results = client.recall(bank_id="my-bank", query="What does Alice do?")
    for r in results:
        print(r.text)

    answer = client.reflect(bank_id="my-bank", query="Tell me about Alice")
    print(answer.text)
# Server is stopped here
```

## `HindsightEmbedded` — auto-managed

`HindsightEmbedded` is the simplest way to use Hindsight in Python. It automatically manages a background daemon for you — starts on first use, stays alive across calls, shuts down after an idle timeout.

```python
from hindsight import HindsightEmbedded
import os

# Server starts automatically on first call
client = HindsightEmbedded(
    profile="myapp",                        # Profile for data isolation
    llm_provider="openai",
    llm_model="gpt-4o-mini",
    llm_api_key=os.environ["OPENAI_API_KEY"],
)

# Use immediately - no manual server management needed
client.retain(bank_id="my-bank", content="Alice works at Google")
results = client.recall(bank_id="my-bank", query="What does Alice do?")

# Server continues running (auto-stops after idle timeout)
# Or explicitly stop it:
client.close(stop_daemon=True)
```

### What's a Profile?

A profile is an isolated Hindsight environment. Each profile gets its own embedded PostgreSQL database (stored in `~/.pg0/instances/hindsight-embed-{profile}/`) and its own API server. Use different profiles to separate environments (dev/prod), applications, or users.

### When to use which

| Use case | Pick |
|---|---|
| Tests, short-lived scripts, deterministic startup/shutdown | `HindsightServer` (context manager) |
| Long-running application, auto-start on first use, don't want to manage lifecycle | `HindsightEmbedded` |
| Existing Hindsight server running elsewhere | [`hindsight-client`](./python.md) directly |

## API namespaces

Both `HindsightEmbedded` and `HindsightClient` expose organized API namespaces for bank management, mental models, directives, and memories:

```python
from hindsight import HindsightEmbedded
import os

embedded = HindsightEmbedded(
    profile="myapp",
    llm_provider="openai",
    llm_api_key=os.environ["OPENAI_API_KEY"],
)

# Core operations
embedded.retain(bank_id="test", content="Hello")
results = embedded.recall(bank_id="test", query="Hello")

# Bank management
embedded.banks.create(bank_id="test", name="Test Bank", mission="Help users")
embedded.banks.set_mission(bank_id="test", mission="Updated mission")
embedded.banks.delete(bank_id="test")

# Mental models
embedded.mental_models.create(
    bank_id="test",
    name="User Preferences",
    content="User prefers dark mode"
)
models = embedded.mental_models.list(bank_id="test")

# Directives
embedded.directives.create(
    bank_id="test",
    name="Response Style",
    content="Be concise and friendly"
)
directives = embedded.directives.list(bank_id="test")

# List memories
memories = embedded.memories.list(bank_id="test", type="world", limit=50)
```

API namespaces ensure the daemon is running before each call, so daemon crashes are handled gracefully:

```python
# ✅ GOOD - Uses API namespace (daemon restarts handled)
embedded.banks.create(bank_id="test", name="Test")

# ❌ BAD - Direct client access (daemon crashes NOT handled)
client = embedded.client
client.create_bank(bank_id="test", name="Test")  # Fails if daemon crashed
```

For the full reference of retain/recall/reflect methods and their options (which work the same regardless of how you obtain the client) see the [Python Client page](./python.md).
