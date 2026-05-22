# Hindsight Pipecat Integration

Persistent long-term memory for [Pipecat](https://github.com/pipecat-ai/pipecat) voice AI pipelines via [Hindsight](https://vectorize.io/hindsight). A single `FrameProcessor` slots between your user context aggregator and LLM service — recalling relevant memories before each turn and retaining conversation content after.

## Quick Start

```bash
pip install hindsight-pipecat
```

> ✨ **Recommended: [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup)** — free tier, no self-hosting required. Sign up and grab an API key in under a minute.

```python
from pipecat.pipeline.pipeline import Pipeline
from hindsight_pipecat import HindsightMemoryService

memory = HindsightMemoryService(
    bank_id="user-123",
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",  # or set HINDSIGHT_API_KEY env var
)

pipeline = Pipeline([
    transport.input(),
    stt_service,
    user_aggregator,
    memory,           # ← add between user_aggregator and LLM
    llm_service,
    assistant_aggregator,
    tts_service,
    transport.output(),
])
```

### Self-hosting (local development)

If you're running Hindsight locally with `./scripts/dev/start-api.sh`, point at your local server instead:

```python
memory = HindsightMemoryService(
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
)
```

See the [Hindsight installation guide](https://hindsight.vectorize.io/developer/installation) for self-hosting setup.

## How It Works

```
New turn starts
  └─ OpenAILLMContextFrame arrives
       ├─ Retain previous complete turn (user+assistant) — fire-and-forget
       └─ Recall relevant memories for current user query
            └─ Inject as <hindsight_memories> system message
                 └─ Forward enriched context to LLM
```

On each `OpenAILLMContextFrame`:

1. **Retain** — any new complete user+assistant turn pairs are sent to Hindsight asynchronously (non-blocking)
2. **Recall** — the latest user message is used as the search query; results are injected as a system message before the LLM sees the context
3. **Forward** — the enriched context frame is pushed downstream

Memory accumulates across calls. By the third or fourth turn, recall starts surfacing useful context that the pipeline didn't have to re-establish.

## Prerequisites

A running Hindsight instance:

**Self-hosted:**
```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-api-key
hindsight-api  # starts on http://localhost:8888
```

**Hindsight Cloud:** [Sign up](https://ui.hindsight.vectorize.io/signup) — no self-hosting required.

## Configuration

```python
HindsightMemoryService(
    bank_id="user-123",               # Required: memory bank to use
    hindsight_api_url="...",          # Hindsight API URL
    api_key="hsk_...",                # API key (Hindsight Cloud)
    recall_budget="mid",              # "low", "mid", or "high"
    recall_max_tokens=4096,           # Max tokens for recall results
    enable_recall=True,               # Inject memories before LLM
    enable_retain=True,               # Store turns after each exchange
    memory_prefix="Relevant memories from past conversations:\n",
)
```

### Global configuration

```python
from hindsight_pipecat import configure

configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",  # Hindsight Cloud (default)
    api_key="hsk_...",
    recall_budget="mid",
)

# Now create services without repeating connection details
memory = HindsightMemoryService(bank_id="user-123")
```

## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```
