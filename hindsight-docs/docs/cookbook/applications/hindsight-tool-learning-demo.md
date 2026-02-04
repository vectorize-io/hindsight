---
sidebar_position: 4
---

# Tool Learning Demo


:::info Complete Application
This is a complete, runnable application demonstrating Hindsight integration.
[**View source on GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/tree/main/applications/hindsight-tool-learning-demo)
:::


An interactive Streamlit demo showing how Hindsight helps LLMs learn which tool to use when tool names are ambiguous.

## The Problem

When building AI agents with tool/function calling, tool names and descriptions aren't always clear. An LLM might randomly select between similarly-named tools, leading to incorrect behavior.

## The Scenario

This demo simulates a **customer service routing system** with two channels:

| Tool | Description (What the LLM sees) | Actual Purpose (Hidden) |
|------|--------------------------------|------------------------|
| `route_to_channel_alpha` | "Routes to channel Alpha for appropriate request types" | Financial issues (refunds, billing, payments) |
| `route_to_channel_omega` | "Routes to channel Omega for appropriate request types" | Technical issues (bugs, features, errors) |

The descriptions are **intentionally vague**! Without prior knowledge, the LLM must guess which channel handles what.

## The Solution: Learning with Hindsight

With Hindsight memory:
1. **Store routing feedback** about which channel handles which request type
2. **Retrieve learned knowledge** when making routing decisions
3. **Consistently route correctly** based on past experience

## Quick Start

### Prerequisites

1. **Hindsight Server** running (Docker):
```bash
docker run -d -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_PROVIDER=openai \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  -e HINDSIGHT_API_LLM_MODEL=gpt-4o-mini \
  ghcr.io/vectorize-io/hindsight:latest
```

2. **OpenAI API Key**:
```bash
export OPENAI_API_KEY=your-key-here
```

### Run the Demo

```bash
./run.sh
```

Or manually:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to Use the Demo

### Step 1: Test Without Memory (Baseline)

1. Select a **Financial Request** (e.g., "I need a refund...")
2. Click **Route Request**
3. Observe: The "Without Hindsight" column may route incorrectly

### Step 2: Route First Customer and Learn

1. Route a customer → Both LLMs route simultaneously
2. Feedback is automatically stored to Hindsight
3. Wait ~5 seconds for Hindsight to index the memory

### Step 3: Test With Memory

1. Select another request (financial or technical)
2. Click **Route Request**
3. Observe: The "With Hindsight" column should now route correctly!

### Step 4: View Statistics

- See accuracy comparison between "Without Memory" vs "With Hindsight"
- Review test history to see the improvement over time

## Demo Features

- **Side-by-side comparison**: See routing results with and without memory
- **Pre-defined test requests**: Financial and technical scenarios
- **Custom requests**: Enter your own customer requests
- **Memory Explorer**: Query stored routing knowledge directly
- **Live statistics**: Track accuracy improvement

## Key Insight

> Even when tool names and descriptions don't reveal their purpose, Hindsight allows the LLM to **learn from experience** which tool to use for which type of request.

This is especially valuable for:
- Enterprise systems with legacy tool names
- Multi-tenant systems where tools have generic names
- Agents that need to learn organization-specific workflows

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Model | gpt-4o-mini | LLM model for routing decisions |
| Temperature (No Memory) | 0.7 | Randomness for baseline tests |
| Hindsight API URL | http://localhost:8888 | Hindsight server URL |

## Files

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `run.sh` - Launch script with dependency checking
- `README.md` - This file
