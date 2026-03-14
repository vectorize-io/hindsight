---
name: integrate
description: Build a new Hindsight integration for a target AI framework. Handles research, implementation, packaging, docs, and marketing. Invoke with the framework name, e.g. /integrate autogen
argument-hint: "[framework-name]"
---

# Build a Hindsight Integration for $ARGUMENTS

You are building an official Hindsight integration for the **$ARGUMENTS** framework. Follow this workflow end-to-end.

## Phase 1: Research the target framework

### 1a. Check if they already have a memory/storage abstraction

Search the $ARGUMENTS docs and source code for:
- An existing memory, storage, or long-term memory interface (e.g. CrewAI has `Storage`, LangChain has `BaseMemory`)
- An external memory provider registry or plugin system
- Any existing Hindsight integration (if so, note what it covers and what's missing)

If there is an existing abstraction, your integration is an **adapter** that implements their interface using the Hindsight client.

If there is no memory abstraction, look for:
- Vector database integrations or storage backends — these often have a similar write/read interface you can follow
- A plugin or middleware system where you can hook in

### 1b. Study existing Hindsight integrations for patterns

Read these for reference (pick the most relevant 1-2):
- `hindsight-integrations/crewai/` — adapter pattern (implements CrewAI's Storage interface)
- `hindsight-integrations/litellm/` — wrapper pattern (wraps LiteLLM callbacks)
- `hindsight-integrations/pydantic-ai/` — adapter pattern
- `hindsight-integrations/langchain/` — adapter pattern (implements LangChain's BaseMemory)
- `hindsight-integrations/ai-sdk/` — JS/TS wrapper pattern (Vercel AI SDK)

Key patterns to carry forward:
- Thread-local client + dedicated thread pool when the framework uses its own event loop (see crewai `_compat.py`)
- Config via environment variables with sensible defaults
- Clean error types in a separate `errors.py`

## Phase 2: Implementation

### 2a. Create the integration package

```
hindsight-integrations/$ARGUMENTS/
├── hindsight_$ARGUMENTS/
│   ├── __init__.py          # Public API exports
│   ├── storage.py           # or adapter.py — the main integration class
│   ├── config.py            # Config dataclass, env var loading
│   ├── errors.py            # Integration-specific errors
│   └── _compat.py           # Async/sync compatibility if needed
├── tests/
│   ├── test_unit.py         # Unit tests with mocked client
│   └── test_manual.py       # Manual integration test (requires running Hindsight)
├── pyproject.toml
└── README.md
```

For JS/TS integrations, use the equivalent structure with `package.json`.

### 2b. Features checklist — every integration MUST support:

- [ ] **retain** — store memories with all parameters (content, metadata, tags, source, etc.)
- [ ] **recall** — retrieve memories with all parameters (query, limit, filters, tags, temporal, etc.)
- [ ] **reflect** — disposition-aware reasoning (optional but preferred)
- [ ] **Official Hindsight client** — use `hindsight-client` (Python) or the official TS/Rust client, NOT raw HTTP
- [ ] **Cloud token auth** — support `HINDSIGHT_API_TOKEN` for Hindsight Cloud
- [ ] **Self-hosted URL** — support `HINDSIGHT_API_URL` for self-hosted deployments
- [ ] **Dynamic banks** — allow bank_id to be set per-request or per-user, not just globally
- [ ] **Dynamic parameters** — allow retain/recall params (e.g. tags, filters) to vary per-call, for example based on the current user making the request

### 2c. Implementation guidelines

- Use the `hindsight-client` package (Python, currently v0.4.x). Import as `from hindsight_client import HindsightClient`
- Support both sync and async usage if the target framework needs it
- If the framework runs its own async event loop, use a thread-local client + dedicated thread pool (see crewai `_compat.py` for the pattern)
- All config should work via environment variables with optional constructor overrides
- Don't over-abstract — keep the adapter thin and direct

### 2d. Tests

- Unit tests with mocked Hindsight client (test the adapter logic, not Hindsight itself)
- A manual integration test that connects to a real Hindsight instance
- Use `uv run pytest tests/ -v` to run

## Phase 3: Documentation & Marketing

### 3a. README

Write a clear, practical README with:
- What this integration does (1-2 sentences)
- Prerequisites (Hindsight account or self-hosted instance)
- Installation (`pip install hindsight-$ARGUMENTS` or similar)
- Quick start example (minimal working code, <20 lines)
- Configuration reference (env vars, constructor params)
- Full example with dynamic banks/tags
- Link to Hindsight docs

**No slop.** No "unleash the power of" or "seamlessly integrate". Just clear instructions and working code.

### 3b. Official integration on our repo

- Copy the README content into the Hindsight docs at `hindsight-docs/docs/integrations/$ARGUMENTS.md`
- Add it to the docs sidebar configuration
- Make sure the docs page has the same quick start example

### 3c. Cookbook example

Create a cookbook example at `hindsight-docs/docs/cookbooks/$ARGUMENTS-example.md` (or as a Jupyter notebook / standalone script) showing a realistic use case:
- A real-world scenario (not just "hello world")
- Uses dynamic banks or tags
- Shows retain + recall in a meaningful flow

### 3d. Blog post

Write a blog post draft following the established pattern (see `hindsight-docs/blog/` for examples):
- Practical, technical, not marketing fluff
- Shows the problem → solution → working code
- Save as a draft for review

## Key references

- Hindsight Python client: `hindsight-clients/python/`
- Existing integrations: `hindsight-integrations/`
- Hindsight API docs: https://docs.hindsight.vectorize.io
- Hindsight Cloud signup: https://ui.hindsight.vectorize.io/signup
