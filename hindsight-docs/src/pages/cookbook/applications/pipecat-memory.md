---
sidebar_position: 18
---

# Pipecat Memory


:::info Complete Application
This is a complete, runnable application demonstrating Hindsight integration.
[**View source on GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/tree/main/applications/pipecat-memory)
:::


Give [Pipecat](https://github.com/pipecat-ai/pipecat) voice AI pipelines persistent memory across conversations using Hindsight. A single `HindsightMemoryService` frame processor slots between your user context aggregator and LLM service — automatically recalling relevant context before each turn and retaining conversation content after.

## What This Demonstrates

- **Automatic recall** — Hindsight injects relevant memories into the LLM context before each turn
- **Session retention** — conversation turns are stored to Hindsight after each exchange
- **Cross-session continuity** — the pipeline remembers user preferences, past conversations, and context between sessions
- **Non-blocking retention** — memory writes are fire-and-forget, so they never delay the voice response
- **Drop-in integration** — one processor added to an existing pipeline, no other changes required

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                    Pipecat Pipeline                    │
│                                                        │
│  transport.input()                                     │
│      │                                                 │
│      ▼                                                 │
│  STT service  (speech → TranscriptionFrame)            │
│      │                                                 │
│      ▼                                                 │
│  LLMUserContextAggregator  (builds LLMContext)         │
│      │                                                 │
│      ▼                                                 │
│  HindsightMemoryService  ← recall → inject             │
│      │          └─ retain previous turn (async)        │
│      ▼                                                 │
│  LLM service  (context → TextFrame)                    │
│      │                                                 │
│      ▼                                                 │
│  LLMAssistantContextAggregator                         │
│      │                                                 │
│      ▼                                                 │
│  TTS service  (text → audio)                           │
│      │                                                 │
│      ▼                                                 │
│  transport.output()                                    │
└────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Python 3.10+**

2. **Pipecat** — install via pip with your desired extras (STT, TTS, LLM services)

3. **Hindsight** — cloud or local
   - [Hindsight Cloud](https://hindsight.vectorize.io) (sign up for an API key)
   - Or run locally: `pip install hindsight-all && hindsight-api`

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Hindsight

For local Hindsight:
```bash
pip install hindsight-all
export HINDSIGHT_API_LLM_API_KEY=your-openai-key
hindsight-api  # starts on http://localhost:8888
```

For Hindsight Cloud:
```bash
export HINDSIGHT_URL=https://api.hindsight.vectorize.io
export HINDSIGHT_API_KEY=hsk_your_token_here
```

### 3. (Optional) Seed your memory bank

To see recall working immediately without waiting for conversations to build up:

```bash
python seed_memory.py
```

### 4. Add to your Pipecat pipeline

```python
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.aggregators.llm_response import LLMUserContextAggregator, LLMAssistantContextAggregator
from hindsight_pipecat import HindsightMemoryService

context = OpenAILLMContext(
    messages=[{"role": "system", "content": "You are a helpful voice assistant."}]
)
user_aggregator = LLMUserContextAggregator(context)
assistant_aggregator = LLMAssistantContextAggregator(context)

memory = HindsightMemoryService(
    bank_id="pipecat",
    hindsight_api_url="http://localhost:8888",  # or your Cloud URL
    # api_key="hsk_...",                        # for Hindsight Cloud
)

pipeline = Pipeline([
    transport.input(),
    stt_service,
    user_aggregator,
    memory,               # ← inject here
    llm_service,
    assistant_aggregator,
    tts_service,
    transport.output(),
])
```

## How It Works

### On each `OpenAILLMContextFrame`

When a new user turn arrives as an `OpenAILLMContextFrame`:

1. **Retain** — if the context contains a complete previous turn (user message + assistant response) that hasn't been retained yet, it's sent to Hindsight asynchronously. This is fire-and-forget — it never delays the voice response.

2. **Recall** — the latest user message is used as the search query. Hindsight retrieves the most relevant memories from past conversations.

3. **Inject** — recalled memories are added as a `<hindsight_memories>` system message at the top of the LLM context. If a memory message from a previous turn already exists, it's replaced rather than accumulated.

4. **Forward** — the enriched context frame is pushed downstream to the LLM service.

### Memory format in context

```
<hindsight_memories>
Relevant memories from past conversations:
1. User's name is Alex and they prefer concise responses
2. User has a morning routine starting at 7am with news briefing
3. User prefers metric units (Celsius, kilometers) over imperial
</hindsight_memories>
```

The LLM sees this context before generating each response. Over multiple sessions, the pipeline accumulates a growing picture of the user's preferences and history.

## Configuration

```python
HindsightMemoryService(
    bank_id="pipecat",          # Required: memory bank to use
    hindsight_api_url="...",    # Hindsight API URL
    api_key="hsk_...",          # API key (Hindsight Cloud)
    recall_budget="mid",        # "low", "mid", or "high"
    recall_max_tokens=4096,     # Max tokens for recall results
    enable_recall=True,         # Inject memories before LLM
    enable_retain=True,         # Store turns after each exchange
)
```

## Core Files

| File | Description |
|------|-------------|
| `seed_memory.py` | Seeds sample user facts for demo purposes |
| `requirements.txt` | Python dependencies |
| `install.py` | Integration installer |
| `hindsight_pipecat/memory.py` | `HindsightMemoryService` frame processor |

## Common Issues

**No memories recalled on first conversation**

Recall returns results only after something has been retained. Either:
- Run `seed_memory.py` to pre-populate facts, or
- Complete one full turn (user speaks, assistant responds) and then start a new session

**Retention seems delayed**

Retain is asynchronous (fire-and-forget) — it runs after the pipeline has already forwarded the frame. Facts retained in one turn are available for recall starting from the next turn.

**Pipeline not recalling across sessions**

Check that you're using the same `bank_id` across all pipeline instances. Each unique `bank_id` is an isolated memory store.

## Built With

- [Pipecat](https://github.com/pipecat-ai/pipecat) — real-time voice AI pipeline framework
- [Hindsight](https://hindsight.vectorize.io) — long-term memory for AI agents
- [hindsight-pipecat](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/pipecat) — Pipecat frame processor
