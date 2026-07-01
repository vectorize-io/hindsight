---
sidebar_position: 2
---

# Agno + Hindsight Memory


:::info Complete Application
This is a complete, runnable application demonstrating Hindsight integration.
[**View source on GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/tree/main/applications/agno-memory)
:::


A personal assistant built with [Agno](https://github.com/agno-agi/agno) that remembers what you tell it across sessions using [Hindsight](https://github.com/vectorize-io/hindsight) for persistent long-term memory.

## What This Demonstrates

- **Native Toolkit pattern** — retain, recall, and reflect registered as Agno tools via `HindsightTools`
- **Auto-injected context** — pre-recalled memories in the system prompt via `memory_instructions()`
- **Per-user memory banks** — automatic bank isolation using `user_id`
- **Persistent memory across sessions** — the agent remembers between script runs

## Prerequisites

1. **Hindsight running**

   ```bash
   export OPENAI_API_KEY=your-key

   docker run --rm -it --pull always -p 8888:8888 -p 9999:9999 \
     -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
     -e HINDSIGHT_API_LLM_MODEL=o3-mini \
     -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
     ghcr.io/vectorize-io/hindsight:latest
   ```

2. **OpenAI API key** (for Agno's LLM)

   ```bash
   export OPENAI_API_KEY=your-key
   ```

3. **Install dependencies**

   ```bash
   cd applications/agno-memory
   pip install -r requirements.txt
   ```

## Quick Start

### Interactive Chat

```bash
python personal_assistant.py
```

Example session:

```
Personal assistant ready (bank: personal-assistant)
Type 'quit' or 'exit' to stop.

You: I'm a Python developer and I love hiking on weekends
Assistant: I've noted that! You're a Python developer who enjoys weekend hiking.

You: What do you know about me?
Assistant: From my memory, I know that you're a Python developer and you
love hiking on weekends.

You: quit
```

Run it again — the agent still remembers:

```
You: What are my hobbies?
Assistant: Based on my memories, you enjoy hiking on weekends!
```

### Single Query

```bash
python personal_assistant.py "What do you remember about my preferences?"
```

### Reset Memory

```bash
python personal_assistant.py --reset
```

## How It Works

1. **`HindsightTools`** registers three Agno tools the agent can call:
   - `retain_memory` — store information to long-term memory
   - `recall_memory` — search long-term memory for relevant facts
   - `reflect_on_memory` — synthesize a reasoned answer from memories

2. **`memory_instructions()`** pre-recalls relevant memories and injects them into the system prompt, so the agent starts every conversation with context.

3. The agent decides when to call retain/recall/reflect based on the conversation.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `BANK_ID` | `personal-assistant` | Hindsight memory bank ID |
| `HINDSIGHT_URL` | `http://localhost:8888` | Hindsight API URL |
| `MODEL` | `openai:gpt-4o-mini` | Agno model string |

## Files

| File | Description |
|------|-------------|
| `personal_assistant.py` | Complete working example with interactive chat and single-query modes |
| `requirements.txt` | Python dependencies |
