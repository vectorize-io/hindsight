---
sidebar_position: 1
---

# Chat Memory App


:::info Complete Application
This is a complete, runnable application demonstrating Hindsight integration.
[**View source on GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/tree/main/applications/chat-memory)
:::


A demo chat application that uses Groq's `qwen/qwen3-32b` model with Hindsight for persistent per-user memory.

## Features

- 🧠 **Persistent Memory**: Each user gets their own memory bank that remembers conversations
- 🚀 **Fast AI**: Powered by Groq's high-speed inference
- 🎯 **Per-User Context**: Isolated memory per user with automatic context retrieval
- 💬 **Real-time Chat**: Instant responses with memory-augmented context

## Setup

### 1. Start Hindsight API

First, start the Hindsight API server using Docker:

```bash
export GROQ_API_KEY=your_groq_api_key_here

# Start Hindsight with Groq as the LLM provider
docker run --rm -it --pull always -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_PROVIDER=groq \
  -e HINDSIGHT_API_LLM_API_KEY=$GROQ_API_KEY \
  -e HINDSIGHT_API_LLM_MODEL="openai/gpt-oss-20b" \
  -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

- **API**: http://localhost:8888
- **Control Plane UI**: http://localhost:9999

### 2. Configure Environment

Copy your Groq API key to the environment file:

```bash
# Update .env.local with your Groq API key
echo "GROQ_API_KEY=your_groq_api_key_here" > .env.local
echo "HINDSIGHT_API_URL=http://localhost:8888" >> .env.local
```

If you don't have one, you can get a free Groq API key here: https://console.groq.com/home

### 3. Install Dependencies

```bash
npm install
```

### 4. Run the App

```bash
npm run dev
```

Open http://localhost:3000 in your browser.

## How It Works

1. **User Identity**: Each browser session gets a unique user ID
2. **Memory Bank Creation**: First message creates a personal memory bank in Hindsight
3. **Context Retrieval**: Before responding, relevant memories are retrieved
4. **Memory Augmented Response**: Groq generates responses with memory context
5. **Conversation Storage**: Each conversation is stored for future context

## Architecture

```
User Message
     ↓
Next.js API Route (/api/chat)
     ↓
Hindsight.recall() → Get relevant memories
     ↓
Groq API → Generate response with memory context
     ↓
Hindsight.retain() → Store conversation
     ↓
Response to User
```

## Memory Bank Structure

Each user gets their own isolated memory bank with:
- **Name**: "Chat Memory for [userId]"
- **Background**: Conversational AI assistant context
- **Disposition**: Empathetic (4), Low Skepticism (2), Balanced Literalism (3)

## Try It Out

1. **First Conversation**: Tell the assistant about yourself
   - "Hi! I'm a software engineer from San Francisco. I love Python and machine learning."

2. **Second Conversation**: Ask what it remembers
   - "What do you know about me?"
   - "What programming languages do I like?"

3. **Context Building**: Continue sharing preferences
   - "I prefer VS Code over other editors"
   - "I'm working on a React project"

4. **Memory Verification**: Visit the Hindsight Control Plane at http://localhost:9999 to see stored memories

## Development

- **Groq Model**: Uses `qwen/qwen3-32b` for fast, high-quality responses
- **Memory Storage**: Automatic conversation retention with context categorization
- **Memory Retrieval**: Semantic search with 2048 token budget for relevant context
