---
sidebar_position: 3
---

# Chat Memory App (Hindsight Cloud)


:::info Complete Application
This is a complete, runnable application demonstrating Hindsight integration.
[**View source on GitHub →**](https://github.com/vectorize-io/hindsight-cookbook/tree/main/applications/chat-memory-cloud)
:::


A demo chat application with persistent per-user memory powered by [Hindsight Cloud](https://hindsight.vectorize.io). Supports OpenAI or Groq as the LLM provider. No local Hindsight server required.

## Features

- 🧠 **Persistent Memory**: Each user gets their own memory bank that remembers conversations
- ☁️ **Hindsight Cloud**: Memory stored in the cloud — no Docker setup needed
- 🔀 **Selectable LLM**: Choose between OpenAI (GPT-4o) or Groq (Qwen 32B)
- 🎯 **Per-User Context**: Isolated memory per user with automatic context retrieval
- 💬 **Real-time Chat**: Instant responses with memory-augmented context

## Setup

### 1. Get API Keys

- **Hindsight** — Sign up at https://hindsight.vectorize.io
- **OpenAI** — https://platform.openai.com/api-keys
- **Groq** (alternative) — Free at https://console.groq.com/home

### 2. Configure Environment

Edit `.env.local` with your API keys and preferred provider:

```bash
# LLM Provider: "openai" or "groq"
LLM_PROVIDER=openai

# OpenAI (required if LLM_PROVIDER=openai)
OPENAI_API_KEY=sk-your-key-here

# Groq (required if LLM_PROVIDER=groq)
GROQ_API_KEY=gsk_your-key-here

# Hindsight Cloud
HINDSIGHT_API_URL=https://api.hindsight.vectorize.io
HINDSIGHT_API_KEY=hsk_your-key-here
```

You can also override the model with `LLM_MODEL` (defaults to `gpt-4o` for OpenAI, `qwen/qwen3-32b` for Groq).

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
2. **Memory Bank Creation**: First message creates a personal memory bank in Hindsight Cloud
3. **Context Retrieval**: Before responding, relevant memories are recalled
4. **Memory Augmented Response**: LLM generates responses with memory context
5. **Conversation Storage**: Each conversation is retained for future context

## Architecture

```
User Message
     ↓
Next.js API Route (/api/chat)
     ↓
Hindsight Cloud recall() → Get relevant memories
     ↓
OpenAI or Groq → Generate response with memory context
     ↓
Hindsight Cloud retain() → Store conversation
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

4. **Memory Verification**: Log in to the [Hindsight dashboard](https://hindsight.vectorize.io) to see stored memories

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | LLM provider: `openai` or `groq` |
| `LLM_MODEL` | auto | Model override (defaults: `gpt-4o` / `qwen/qwen3-32b`) |
| `OPENAI_API_KEY` | — | Required when using OpenAI |
| `GROQ_API_KEY` | — | Required when using Groq |
| `HINDSIGHT_API_URL` | `https://api.hindsight.vectorize.io` | Hindsight API endpoint |
| `HINDSIGHT_API_KEY` | — | Your Hindsight API key |
