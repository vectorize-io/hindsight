---
title: "Give Your OpenAI App a Memory in 5 Minutes"
authors: [hindsight]
date: 2026-03-05
tags: [memory, openai, python, docker, rag, llm, vector, embedding]
image: /img/blog/add-memory-to-openai-application.png
hide_table_of_contents: true
---

Build a ChatGPT-style chatbot with persistent memory using the OpenAI SDK and Hindsight. Three API calls — `retain()`, `recall()`, `reflect()` — and your app remembers users across restarts, no vector database or RAG pipeline required.

<!-- truncate -->

## TL;DR

You'll build a ChatGPT-style chatbot that:

- Remembers facts across restarts
- Recalls relevant context automatically
- Synthesizes long-term knowledge on demand

All with three API calls:

- `retain()` — store memories
- `recall()` — retrieve relevant ones
- `reflect()` — synthesize across everything

No vector database. No embedding pipeline. No RAG plumbing.

Copy, paste, run.

---

## The Problem: Your Chatbot Has Amnesia

You build a chatbot with OpenAI:

```python
messages = [{"role": "system", "content": "You are a helpful assistant."}]
```

It works perfectly.

Until you restart the process.

Now it remembers nothing.

You can serialize `messages` to disk. But then:

- Context windows fill up
- Token costs explode
- You start truncating history
- The assistant forgets early decisions

We ran into this building Jerri, our internal AI project manager at Vectorize.

Jerri lives in Slack. It ingests meeting transcripts, tracks action items, and answers "what did we decide about X?" Without persistent memory, every session started from zero.

That's not memory. That's stateless autocomplete.

What you actually need:

1. Store facts as they happen
2. Retrieve only what's relevant
3. Synthesize when necessary

That's what we're building.

---

## Architecture

Here's the entire loop:

```
User message
     ↓
recall(query)      ← pull relevant memories
     ↓
OpenAI completion  ← inject memory into system prompt
     ↓
retain(exchange)   ← store conversation
     ↓
Response
```

Three calls. Same pattern Jerri runs in production.

---

## Step 1 — Start the Memory Layer

Install:

```bash
pip install hindsight-all
```

Start the server:

```bash
export HINDSIGHT_API_LLM_API_KEY=YOUR_OPENAI_KEY

hindsight-api
```

It runs locally at `http://localhost:8888`.

It includes:

- Embedded Postgres
- Fact extraction
- Semantic search
- Knowledge graph
- Synthesis engine

No external infrastructure.

> **Prefer not to self-host?** [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) gives you the same API with no setup — just swap `base_url` for your Cloud endpoint.

---

## Step 2 — Baseline Chat (No Memory)

Let's start with the broken version:

```python
from openai import OpenAI

openai = OpenAI()
messages = [{"role": "system", "content": "You are a helpful assistant."}]

while True:
    user_input = input("You: ")
    if user_input in ("quit", "exit"):
        break

    messages.append({"role": "user", "content": user_input})

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})

    print(reply)
```

Works great.

Restart it.

Ask: "What's my name?"

Blank stare.

---

## Step 3 — Add `retain()`

Create a memory bank:

```python
from hindsight_client import Hindsight

hindsight = Hindsight(base_url="http://localhost:8888")

hindsight.create_bank(
    bank_id="chatbot",
    name="Chatbot Memory",
    reflect_mission="Remember user preferences and important facts.",
)
```

Now retain every exchange:

```python
hindsight.retain(
    bank_id="chatbot",
    content=f"User: {user_input}\nAssistant: {reply}",
)
```

That's it.

Hindsight extracts facts, identifies entities, builds relationships, and stores them in a knowledge graph. You don't manage any of that.

---

## Step 4 — Add `recall()`

Before calling OpenAI, retrieve relevant memories:

```python
memories = hindsight.recall(
    bank_id="chatbot",
    query=user_input,
    budget="low",
)

memory_context = "\n".join(r.text for r in memories.results)
```

Inject into the system prompt:

```python
system_prompt = "You are a helpful assistant."

if memory_context:
    system_prompt += "\n\nRelevant past context:\n" + memory_context
```

Now:

1. Tell it your name
2. Restart
3. Ask again

It remembers. Because `recall` injects relevant past facts into the prompt.

---

## Step 5 — Add `reflect()` for Synthesis

`recall` returns facts. `reflect` returns reasoning.

For questions like:

- "What do you know about me?"
- "Summarize our conversations."
- "What patterns do you see?"

Use reflect:

```python
reflection = hindsight.reflect(
    bank_id="chatbot",
    query=user_input,
)

memory_context = reflection.text
```

Reflect traverses the knowledge graph, runs an LLM reasoning chain, and synthesizes across memories.

In Jerri, reflect powers weekly summaries, sprint reviews, and cross-meeting analysis. In your chatbot, it handles "step back and think" queries.

---

## Full Working Example

Copy this into `chat.py`:

```python
from openai import OpenAI
from hindsight_client import Hindsight

openai = OpenAI()
hindsight = Hindsight(base_url="http://localhost:8888")

hindsight.create_bank(
    bank_id="chatbot",
    name="Chatbot Memory",
    reflect_mission="Remember user preferences and key facts.",
)

SYSTEM_PROMPT = "You are a helpful assistant with long-term memory."

SYNTHESIS_KEYWORDS = [
    "summarize",
    "what do you know about me",
    "what have we talked about",
]


def get_memory_context(user_input):
    if any(k in user_input.lower() for k in SYNTHESIS_KEYWORDS):
        reflection = hindsight.reflect(
            bank_id="chatbot",
            query=user_input,
        )
        return reflection.text

    memories = hindsight.recall(
        bank_id="chatbot",
        query=user_input,
        budget="low",
    )
    return "\n".join(r.text for r in memories.results)


def main():
    conversation = []

    print("Chat with memory. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")
        if user_input in ("quit", "exit"):
            break

        memory_context = get_memory_context(user_input)

        conversation.append({"role": "user", "content": user_input})

        system = SYSTEM_PROMPT
        if memory_context:
            system += "\n\nRelevant context:\n" + memory_context

        messages = [{"role": "system", "content": system}] + conversation

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )

        reply = response.choices[0].message.content
        conversation.append({"role": "assistant", "content": reply})

        print(f"\nAssistant: {reply}\n")

        hindsight.retain(
            bank_id="chatbot",
            content=f"User: {user_input}\nAssistant: {reply}",
        )


if __name__ == "__main__":
    main()
```

Run:

```bash
export OPENAI_API_KEY=YOUR_KEY
python chat.py
```

Restart it. It still remembers.

---

## Production Lessons (From Building Jerri)

**1. Retain after responding.** Otherwise the assistant remembers questions but not answers.

**2. Use `budget="low"` for chat loops.** Sub-second latency. Upgrade only when needed.

**3. One bank per user in multi-user apps.** Otherwise memories leak across users.

**4. Set a mission.** Fact extraction quality depends heavily on it.

**5. Start by retaining everything.** Optimize later.

---

## When to Use This Pattern

**Use it if:**

- You need cross-session memory
- You want synthesis across time
- You don't want to build RAG infra

**Don't use it if:**

- You only need single-session context
- You're storing structured database records

This solves the space between "chat history" and "knowledge base."

---

## The Pattern to Remember

- `retain` — after responding
- `recall` — before responding
- `reflect` — when synthesizing

That's the loop.

That's what powers Jerri across weeks of meetings.

That's what you just built in 15 minutes.

---

## Next Steps

- **Add per-user banks** with unique `bank_id` per user
- **Use tags** for scoped memory (`tags` on retain, `tags_match` on recall)
- **Add structured JSON output** to reflect with `response_schema`
- **Inspect memories in the web UI** at `localhost:9999` via Docker
- **Try [hosted Hindsight](https://ui.hindsight.vectorize.io/signup)** instead of self-hosting

Persistent memory turns a chatbot into an agent.

Now yours remembers.
