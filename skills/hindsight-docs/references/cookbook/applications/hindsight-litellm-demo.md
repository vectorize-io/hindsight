---
sidebar_position: 3
---

# Memory Approaches Comparison Demo


:::info Complete Application
This is a complete, runnable application demonstrating Hindsight integration.
[**View source on GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/tree/main/applications/hindsight-litellm-demo)
:::


Interactive Streamlit app comparing three memory approaches for LLM applications:

1. **No Memory** - Each query is independent (baseline)
2. **Full Conversation History** - Pass entire conversation (truncated to simulate context limits)
3. **Hindsight Memory** - Intelligent semantic memory retrieval

This demo showcases how Hindsight's semantic memory outperforms traditional approaches, especially as conversations grow longer.

## Quick Start

```bash
# 1. Set your OpenAI API key
export OPENAI_API_KEY=your-key

# 2. Start Hindsight server
docker run -d -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_PROVIDER=openai \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  ghcr.io/vectorize-io/hindsight:latest

# 3. Run the demo
./run.sh
```

Then open http://localhost:8501 in your browser.

## What This Demo Shows

### The Problem with Traditional Approaches

| Approach | How it Works | Limitation |
|----------|--------------|------------|
| **No Memory** | Each query standalone | Forgets everything between messages |
| **Full History** | Pass all messages to LLM | Token limits cause truncation - loses early context |
| **Hindsight** | Semantic retrieval of relevant facts | Retrieves what's relevant regardless of when it was said |

### Key Insight

After 5-10 messages, watch the **Full Conversation History** column start losing early context due to truncation (artificially set to 4 messages to demonstrate this quickly). Meanwhile, **Hindsight Memory** can still recall facts from the beginning because it uses semantic retrieval rather than sequential history.

## Testing the Demo

1. **Introduce yourself**:
   - "Hi, I'm Sarah, a data scientist at Netflix"
   - "I prefer Python and love machine learning"

2. **Have several exchanges** about different topics

3. **Test recall**:
   - "What programming language should I use?"
   - "What do you know about me?"

Watch how the three columns respond differently as the conversation grows.

## Features

- **Side-by-side comparison** of all three approaches
- **Debug panels** showing what context each approach uses
- **Memory explorer** to search Hindsight memories directly
- **Configurable settings** for history truncation, max memories, etc.
- **Multi-provider support** via LiteLLM (OpenAI, Anthropic, Groq)

## Prerequisites

- Python 3.10+
- Hindsight server running (Docker recommended)
- At least one LLM API key (OpenAI recommended)

## Setup

### Using run.sh (Recommended)

```bash
# Set API key
export OPENAI_API_KEY=your-key

# Start Hindsight, then run:
./run.sh
```

The script will check and install dependencies automatically.

### Manual Setup

```bash
# Install dependencies
pip install streamlit litellm

# Install Hindsight packages
pip install hindsight-client hindsight-litellm

# Run the app
streamlit run app.py
```

### Starting Hindsight Server

```bash
docker run -d -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_PROVIDER=openai \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  ghcr.io/vectorize-io/hindsight:latest

# Verify it's running
curl http://localhost:8888/health
```

## Configuration

### Sidebar Options

**Model Selection:**
- Provider: OpenAI, Anthropic, Groq
- Model: Various models per provider
- Custom model ID support

**Full History Config:**
- Max Messages to Keep (default: 4 to demonstrate truncation)

**Hindsight Config:**
- API URL (default: http://localhost:8888)
- Bank ID and Entity ID for memory isolation
- Max Memories to retrieve
- Recall Budget (low/mid/high)

**Generation Settings:**
- Temperature
- Max Tokens
- System Prompt

## Supported Models

### OpenAI
- gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo

### Anthropic
- claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022
- claude-3-opus-20240229, claude-3-sonnet-20240229

### Groq
- groq/llama-3.1-70b-versatile, groq/llama-3.1-8b-instant
- groq/mixtral-8x7b-32768

## Environment Variables

```bash
# Required
export OPENAI_API_KEY=sk-...

# Optional (for other providers)
export ANTHROPIC_API_KEY=sk-ant-...
export GROQ_API_KEY=gsk_...

# Optional
export HINDSIGHT_URL=http://localhost:8888
```

## Troubleshooting

### Hindsight server not responding

```bash
# Check if running
curl http://localhost:8888/health

# Start with Docker
docker run -d -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_PROVIDER=openai \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  ghcr.io/vectorize-io/hindsight:latest
```

### hindsight-litellm not installed

```bash
pip install hindsight-litellm
```

### API key errors

Make sure the appropriate API key is set:
```bash
export OPENAI_API_KEY=your-key
```

## Related

- [Hindsight](https://github.com/vectorize-io/hindsight) - Memory infrastructure for AI applications
- [hindsight-litellm](https://github.com/vectorize-io/hindsight/tree/main/hindsight-integrations/litellm) - LiteLLM integration package

## License

MIT
