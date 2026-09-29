---
title: "I Built a Chatbot That Never Forgets — In 80 Lines of Python"
authors: [benfrank241]
date: 2026-03-17T12:00
tags: [streamlit, tutorial, python, memory, chatbot, web-ui]
slug: python-chatbot-memory-streamlit
image: /img/blog/streamlit-chatbot-memory.png
---

![I Built a Chatbot That Never Forgets — In 80 Lines of Python](/img/blog/streamlit-chatbot-memory.png)

Build a web chatbot with persistent memory using Streamlit and Hindsight. ~80 lines of Python, no frontend framework. Memory survives restarts, and a sidebar shows what the agent remembers.

<!-- truncate -->

## TL;DR

- Build a web chatbot with persistent memory using [Streamlit](https://streamlit.io/) and [Hindsight](https://ui.hindsight.vectorize.io)
- ~80 lines of Python. No frontend framework, no JavaScript, no build step.
- Memory survives browser refreshes and server restarts
- Sidebar shows what the agent remembers — recalled facts and synthesized reflections

---

## The Problem: Terminal Chatbots Don't Ship

You built a chatbot with memory using [OpenAI and Hindsight](/blog/2026/03/05/add-memory-to-openai-application). It works in a terminal:

```python
user_input = input("You: ")
```

That's fine for a demo. But when you want to share it with your team, "clone this repo and run `python chat.py`" doesn't cut it.

You need a web UI. That usually means React, a backend API, WebSocket plumbing, and a deployment pipeline. For an internal tool or prototype, that's weeks of work you don't need.

[Streamlit](https://streamlit.io/) gives you a web app in pure Python. Chat components, session management, and a built-in server. No frontend build.

The catch: Streamlit re-runs your entire script on every interaction. Every button click, every message — top to bottom. That clashes with stateful operations like initializing API clients and maintaining conversation history.

This tutorial solves that. You'll build a Python chatbot with persistent memory and a sidebar that shows what the agent remembers, all in one file.

---

## Architecture: How Persistent Chatbot Memory Works

```
Browser (Streamlit UI)
     ↓
st.chat_input → user message
     ↓
recall(query)       ← pull relevant memories from Hindsight
     ↓
OpenAI completion   ← inject memories into system prompt
     ↓
retain(exchange)    ← store the conversation
     ↓
st.chat_message → display response
     ↓
st.sidebar → show recalled facts + reflect button
```

Three layers:

- **`st.session_state`** — per-tab, ephemeral conversation history (lost on browser close)
- **[Hindsight](https://hindsight.vectorize.io)** — persistent memory across restarts (facts, entities, [knowledge graph](/blog/2026/03/12/spreading-activation-memory-graphs))
- **OpenAI** — generates responses with memory-augmented context

---

## Step 1 — Bare Streamlit Chat (No Memory)

Start with a working chat UI. No memory, no Hindsight.

```python
import streamlit as st
from openai import OpenAI

st.title("Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

openai = OpenAI()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Say something"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages += st.session_state.messages

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    reply = response.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
```

Run it:

```bash
pip install streamlit openai
streamlit run app.py
```

Open `http://localhost:8501`. You have a chat UI.

Refresh the browser tab. Conversation survives (it's in `st.session_state`).

Close the tab and reopen. Gone.

Restart the Streamlit server. Gone.

That's the gap we're filling.

---

## Step 2 — Add Persistent Memory with Hindsight

Install the dependencies:

```bash
pip install hindsight-all hindsight-client
```

Start the Hindsight server in a separate terminal:

```bash
export HINDSIGHT_API_LLM_API_KEY=YOUR_OPENAI_KEY
hindsight-api
```

> **Note:** You can also use [Hindsight Cloud](https://ui.hindsight.vectorize.io) instead of self-hosting — just change the `base_url` to `https://api.hindsight.vectorize.io` and add your API key.

Now wire Hindsight into the chat. The key pattern: use [`@st.cache_resource`](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource) to initialize the client once, not on every re-run.

```python
import streamlit as st
from openai import OpenAI
from hindsight_client import Hindsight

st.title("Chat with Memory")

BANK_ID = "streamlit-chatbot"
SYSTEM_PROMPT = "You are a helpful assistant with long-term memory."


@st.cache_resource
def get_hindsight():
    client = Hindsight(base_url="http://localhost:8888")
    client.create_bank(
        bank_id=BANK_ID,
        name="Streamlit Chatbot",
        mission="Remember user preferences, facts, and conversation history.",
    )
    return client


@st.cache_resource
def get_openai():
    return OpenAI()


hindsight = get_hindsight()
openai_client = get_openai()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Say something"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Recall relevant memories
    with st.spinner("Remembering..."):
        memories = hindsight.recall(bank_id=BANK_ID, query=prompt, budget="low")
        memory_context = "\n".join(r.text for r in memories.results)

    system = SYSTEM_PROMPT
    if memory_context:
        system += "\n\nRelevant context from memory:\n" + memory_context

    messages = [{"role": "system", "content": system}] + st.session_state.messages

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    reply = response.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)

    # Retain the exchange
    hindsight.retain(
        bank_id=BANK_ID,
        content=f"User: {prompt}\nAssistant: {reply}",
    )
```

Tell the chatbot your name. Restart the Streamlit server. Ask "What's my name?"

It remembers. Because `recall` pulled the fact from Hindsight and injected it into the system prompt.

**What happens under the hood:** When you call `retain`, Hindsight extracts structured facts from the natural language content — entities, relationships, timestamps — and stores them in a knowledge graph backed by embedded Postgres with [pgvector](https://github.com/pgvector/pgvector). When you call `recall`, it runs semantic search over those facts and returns the most relevant ones for your query. You don't write schemas, queries, or extraction logic. The [OpenAI API](https://platform.openai.com/docs/api-reference) handles the LLM calls for both extraction and generation.

---

## Step 3 — Add a Sidebar to Inspect Chatbot Memory

The sidebar turns this from a chatbot demo into a memory debugging tool. You can see what the agent recalled, how many facts it has stored, and run reflect to get a synthesis.

Here's the final, complete `app.py`:

```python
import streamlit as st
from openai import OpenAI
from hindsight_client import Hindsight

st.set_page_config(page_title="Chat with Memory", layout="wide")
st.title("Chat with Memory")

BANK_ID = "streamlit-chatbot"
SYSTEM_PROMPT = "You are a helpful assistant with long-term memory."


@st.cache_resource
def get_hindsight():
    client = Hindsight(base_url="http://localhost:8888")
    client.create_bank(
        bank_id=BANK_ID,
        name="Streamlit Chatbot",
        mission="Remember user preferences, facts, and conversation history.",
    )
    return client


@st.cache_resource
def get_openai():
    return OpenAI()


hindsight = get_hindsight()
openai_client = get_openai()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_recall" not in st.session_state:
    st.session_state.last_recall = []

# ── Sidebar: Memory Panel ──────────────────────────────────
with st.sidebar:
    st.header("Memory")

    memory_list = hindsight.list_memories(bank_id=BANK_ID, limit=1)
    st.metric("Stored facts", memory_list.total)

    if st.session_state.last_recall:
        st.subheader("Last recalled")
        for fact in st.session_state.last_recall:
            st.caption(fact)
    else:
        st.caption("No memories recalled yet.")

    st.divider()
    reflect_query = st.text_input("Ask memory a question", key="reflect_input")
    if st.button("Reflect"):
        if reflect_query:
            with st.spinner("Reflecting..."):
                reflection = hindsight.reflect(
                    bank_id=BANK_ID, query=reflect_query
                )
            st.subheader("Reflection")
            st.write(reflection.text)
        else:
            st.warning("Enter a question first.")

# ── Main: Chat ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Say something"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Recall
    with st.spinner("Remembering..."):
        memories = hindsight.recall(bank_id=BANK_ID, query=prompt, budget="low")
        recalled_texts = [r.text for r in memories.results]
        st.session_state.last_recall = recalled_texts
        memory_context = "\n".join(recalled_texts)

    system = SYSTEM_PROMPT
    if memory_context:
        system += "\n\nRelevant context from memory:\n" + memory_context

    messages = [{"role": "system", "content": system}] + st.session_state.messages

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    reply = response.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)

    # Retain
    hindsight.retain(
        bank_id=BANK_ID,
        content=f"User: {prompt}\nAssistant: {reply}",
    )

    st.rerun()
```

Run:

```bash
export OPENAI_API_KEY=YOUR_KEY
streamlit run app.py
```

The sidebar shows:

- **Stored facts** — total memory count, updated on each page load
- **Last recalled** — the specific facts Hindsight found for the most recent query
- **Reflect** — a text input and button that runs `reflect()` and displays the synthesis

This is useful for debugging ("why did the agent say that?") and for understanding how memory evolves over time.

---

## Pitfalls and Edge Cases

**1. Streamlit re-runs your entire script on every interaction.** Every button click, every chat message, every widget change triggers a full top-to-bottom re-execution. Without `@st.cache_resource`, you'd create a new Hindsight client and call `create_bank` on every keystroke. The cache decorator ensures initialization happens once per server process.

**2. `st.session_state` is not persistent memory.** Session state lives in the Streamlit server's memory, scoped to a browser tab. Close the tab, lose the state. Restart the server, lose the state. Hindsight is the persistent layer. Don't store anything in `st.session_state` that you can't afford to lose — conversation display history is fine, but critical data should go through `retain`.

**3. Blocking calls freeze the UI.** `retain()`, `recall()`, and especially `reflect()` are synchronous HTTP calls. While they execute, the Streamlit UI is unresponsive. Wrap them in `st.spinner` so users see feedback. For production, consider running retain in a background thread — the user doesn't need to wait for fact extraction to complete before seeing the next response.

**4. One bank per user if deploying to multiple people.** The example uses a hardcoded `BANK_ID`. If two people use the same Streamlit app, they share memories. For multi-user deployments, derive `bank_id` from a session identifier or authentication — `st_experimental_user` (Streamlit's auth API) or a query parameter.

**5. `st.rerun()` is necessary after chat messages.** Without it, the sidebar won't update with the latest recalled facts until the next interaction. The `st.rerun()` at the end of the chat handler triggers a re-execution that refreshes the sidebar with the new recall results.

**6. Conversation history grows without bound.** `st.session_state.messages` accumulates every message. For long sessions, this means increasingly large payloads to OpenAI. Trim the list or switch to a sliding window. Hindsight handles long-term context — you don't need to keep the full conversation in the prompt.

---

## Tradeoffs: Streamlit Chatbot vs. Other Approaches

**Streamlit vs. Gradio**

[Gradio](https://www.gradio.app/) has dedicated `gr.ChatInterface` that handles streaming and message history automatically. Less boilerplate for a basic chat. But Streamlit's sidebar, layout options, and widget library are stronger for building a full tool around the chat — like the memory panel here.

**Streamlit vs. Custom React**

If you need authentication, routing, real-time streaming, or fine-grained UI control, build a real frontend. Streamlit is for internal tools, demos, and prototypes where speed-to-working-app matters more than UI polish.

**When Streamlit is right:**

- Internal tools for your team
- Prototyping agent interactions
- Demos that need a URL, not a terminal
- Memory debugging and introspection

**When it's not:**

- Customer-facing production apps
- Apps that need real authentication and authorization
- High-concurrency deployments (Streamlit is single-threaded per session)

---

## Recap

- `@st.cache_resource` for client initialization — runs once, not per interaction
- `st.session_state` for ephemeral display state, Hindsight for persistent memory
- `recall` before responding, `retain` after responding — same loop as the terminal version
- Sidebar gives you memory introspection for free — recalled facts, stored count, reflect on demand

Streamlit handles the UI. Hindsight handles the memory. OpenAI handles the generation. Each does one thing.

---

## Next Steps

- **Add streaming** with `st.write_stream` and OpenAI's `stream=True` for real-time token output
- **Derive `bank_id` from user identity** using Streamlit's authentication or a query parameter
- **Add a "Clear Memory" button** that calls `hindsight.delete_bank()` and recreates it
- **Show the knowledge graph** — use `include_entities=True` on recall and render entity connections
- **Deploy with Streamlit Community Cloud** — add `OPENAI_API_KEY` and `HINDSIGHT_API_URL` as secrets
- **Try [Hindsight Cloud](https://ui.hindsight.vectorize.io)** for deployment without self-hosting the memory server
- **Customize agent reasoning** — use [disposition traits](/blog/2026/03/13/disposition-aware-agents) to make your chatbot more empathetic, skeptical, or literal

A chatbot with memory is useful. A chatbot with memory you can inspect is a development tool.
