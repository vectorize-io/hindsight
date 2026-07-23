---
title: "Guide: LangGraph Short-Term State vs Long-Term Memory with Hindsight"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, langgraph, memory]
description: "LangGraph's checkpointer keeps short-term state inside a thread; Hindsight adds long-term memory that carries knowledge across threads, sessions, and users."
image: /img/guides/guide-langgraph-state-vs-long-term-memory.svg
hide_table_of_contents: true
---

![Guide: LangGraph Short-Term State vs Long-Term Memory with Hindsight](/img/guides/guide-langgraph-state-vs-long-term-memory.svg)

LangGraph's checkpointer gives your graph **short-term state**: it persists the messages and values of a single thread so a conversation survives a crash, a pause, or a human-in-the-loop interrupt. But that state is scoped to one thread. Start a new thread — or serve a different user — and the graph begins from nothing. The checkpointer never carries what it learned in thread A into thread B.

**Long-term memory** is the missing layer. Hindsight stores facts, preferences, and experiences that outlive any single thread, so an agent can recall in a fresh session what a user told it last week. The two are complementary, not competing: the checkpointer owns the *working set* of the current run, and Hindsight owns the *durable knowledge* that should follow the user everywhere.

This guide explains what the checkpointer remembers and what it forgets, how short-term state and long-term memory compose, why the `BaseStore` adapter is the natural bridge between them, and a decision guide for which layer to reach for. It builds on the existing [LangGraph setup guide](https://hindsight.vectorize.io/guides/2026/05/04/guide-langgraph-memory-with-hindsight) and the [LangGraph integration docs](https://hindsight.vectorize.io/docs/integrations/langgraph) — keep both nearby.

<!-- truncate -->

> **Quick answer**
>
> 1. The checkpointer persists **thread state** — one conversation, one run — and forgets it across threads.
> 2. Hindsight persists **long-term memory** — facts and preferences that cross threads, sessions, and users.
> 3. Wire both: keep the checkpointer for working state, add Hindsight for durable knowledge.
> 4. The `BaseStore` adapter (`HindsightStore`) is the cleanest bridge — LangGraph's native cross-thread store, backed by Hindsight.
> 5. Verify by asking a *new thread* to recall something stored in an *old thread*.

## What the checkpointer remembers (and forgets)

A LangGraph checkpointer snapshots your graph's state after each step and keys it by `thread_id`. That is what makes short-term state useful:

- A conversation survives process restarts, so you can resume mid-run.
- Human-in-the-loop interrupts work because the paused state is durable.
- The full message history of *this thread* is available to *this thread*.

What the checkpointer does **not** do is share anything across threads. Each `thread_id` is an island. If a user tells your agent "I prefer dark mode" in one thread and comes back tomorrow on a new thread, the checkpointer for the new thread starts empty — it has no way to reach into the old thread's state. It also isn't a knowledge store: it holds raw message history, not distilled facts you can search semantically.

That is the boundary. The checkpointer is excellent at *"where was I in this conversation"* and has nothing to say about *"what do I know about this user across all conversations."*

## Short-term state vs long-term memory

Here is the split, layer by layer:

| | LangGraph checkpointer (short-term) | Hindsight (long-term) |
|---|---|---|
| **Scope** | One `thread_id` | Across threads, sessions, and users |
| **Holds** | Raw graph state / message history | Distilled facts, preferences, experiences |
| **Keyed by** | Thread | Bank (per user, tenant, or thread) |
| **Retrieval** | Load the whole thread state | Semantic recall of relevant facts |
| **Lifetime** | The conversation | Durable, follows the user |
| **Best at** | Resuming a run, interrupts | Continuity across runs |

They compose cleanly because they answer different questions. In a single graph run, the checkpointer supplies the working set — the messages so far — while a Hindsight recall injects the durable facts that make the agent feel like it already knows you. After the run, a Hindsight retain distills what was worth remembering into long-term memory, ready for the next thread.

The [LangGraph integration](https://hindsight.vectorize.io/docs/integrations/langgraph) offers three patterns for adding that long-term layer:

- **Tools** — `create_hindsight_tools()` exposes retain, recall, and reflect as LangChain `@tool` functions. Agent-controlled: the model decides when to remember or look something up. Works with plain LangChain too.
- **Nodes** — `create_recall_node()` and `create_retain_node()` sit around your LLM node. The recall node injects matching memories before the model runs; the retain node stores messages after. Automatic, deterministic placement in the graph.
- **BaseStore adapter** — `HindsightStore` implements LangGraph's native cross-thread `BaseStore` interface, backed by Hindsight.

## The BaseStore adapter as the bridge

LangGraph already distinguishes these two layers in its own API. The **checkpointer** is short-term thread state; the **store** (`BaseStore`) is LangGraph's official home for *cross-thread* long-term memory. `HindsightStore` slots straight into that slot, so Hindsight becomes the durable memory backing LangGraph's own long-term abstraction.

You pass both to `compile()` — checkpointer for short-term, store for long-term — and they coexist:

```python
from hindsight_client import Hindsight
from hindsight_langgraph import HindsightStore

client = Hindsight(base_url="http://localhost:8888")
store = HindsightStore(client=client)

graph = builder.compile(checkpointer=checkpointer, store=store)

# Cross-thread long-term memory, keyed by namespace
await store.aput(("user", "123", "prefs"), "theme", {"value": "dark mode"})
results = await store.asearch(("user", "123", "prefs"), query="theme preference")
```

Namespace tuples map to Hindsight bank IDs with `.` as the separator (`("user", "123")` becomes bank `user.123`), and banks are auto-created on first access. That is exactly why the store is the natural bridge: your `("user", user_id)` namespace stays stable across every thread, so a value written in one thread is searchable in any later thread for the same user.

A few properties of `HindsightStore` to know before you lean on it:

- **Async-only.** Use `aput`, `aget`, `asearch`, `abatch`, `adelete`, `alist_namespaces`. The sync variants raise `NotImplementedError`.
- **`get()` is recall-based.** There is no direct key lookup — the key becomes a recall query and only exact `document_id` matches return, so items outside the top recall results can look missing. Prefer `asearch` for semantic retrieval.
- **`list_namespaces` is session-scoped.** It tracks only namespaces written via `aput()` in the current process; after a restart it returns empty even though the data persists in Hindsight.
- **`delete` is a no-op.** Hindsight's memory model is append-oriented — superseding facts is handled automatically during retain — so `adelete()` logs but does not remove data.

If you want tighter prompt control or agent-driven memory instead, the nodes and tools patterns are still available; the store is simply the option that matches LangGraph's own long-term-memory shape.

## Connect LangGraph to Hindsight

This guide is about the *state vs memory* distinction, not installation. For the full install and wiring walkthrough — `pip install hindsight-langgraph`, connecting the client, adding recall/retain nodes, and resolving dynamic bank IDs — follow the [LangGraph setup guide](https://hindsight.vectorize.io/guides/2026/05/04/guide-langgraph-memory-with-hindsight). Once that is in place, the rest of this guide applies directly.

## Which layer to use when

Reach for the layer that matches the question you're answering:

- **Use the checkpointer alone** when a workflow is single-session and self-contained — a one-off task, a form-filling flow, a run where nothing needs to be remembered next time. Short-term state is all you need.
- **Add Hindsight recall/retain nodes** when you want automatic, deterministic long-term memory around every run: recall the user's durable facts before the agent thinks, retain what's worth keeping after it responds.
- **Add Hindsight tools** when the agent should decide when to remember or look something up, or when you're on plain LangChain without a graph.
- **Use `HindsightStore`** when you specifically want LangGraph's native cross-thread store semantics — namespaced, semantic-searchable long-term memory — living alongside the checkpointer.

In practice most production graphs use *both* a checkpointer and a Hindsight layer: short-term state for the current conversation, long-term memory so the next conversation isn't a stranger.

## Verify long-term memory across threads

The whole point is cross-thread continuity, so test that directly:

1. Run the graph on **thread A** for a test user and store a durable fact — e.g. "I prefer dark mode" — via a retain node, tool call, or `store.aput`.
2. Start a **new thread B** for the *same user* (same bank ID / namespace, new `thread_id`).
3. Ask a question in thread B that depends on the thread A fact.
4. Confirm recall surfaces it — the checkpointer for thread B was empty, so if the answer is correct, it came from long-term memory.
5. Repeat with a *different* user and confirm the fact does **not** leak — bank isolation is strict.

If thread B answers from thread A's fact and a different user stays isolated, short-term state and long-term memory are composing correctly.

## Common mistakes

### Expecting the checkpointer to carry knowledge across threads

The checkpointer is per-`thread_id` by design. If you need continuity across threads or users, that is Hindsight's job, not the checkpointer's.

### Reusing one bank ID for every user

If every request maps to the same bank, one user's facts surface for another. Resolve the bank ID per user (for example `bank_id_from_config="user_id"`, or a `("user", user_id)` namespace) so memory stays scoped.

### Changing the thread key on the second run but expecting continuity

A new `thread_id` is a fresh checkpoint — that's expected. Continuity comes from the *bank/namespace* staying stable, not the thread. Keep the bank key constant across runs for the same user.

### Expecting `HindsightStore.adelete()` to remove data

`delete` is a no-op. Hindsight supersedes facts automatically during retain; don't build logic that depends on hard deletes through the store.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — point the client at your own URL (for example `http://localhost:8888`). [Hindsight Cloud](https://hindsight.vectorize.io) is just a hosted backend option.

### Does Hindsight replace the checkpointer?

No. They do different jobs. Keep the checkpointer for short-term thread state and add Hindsight for long-term memory; run both together.

### Should I use nodes, tools, or the store?

Use nodes for automatic recall/retain around the graph, tools for agent-controlled memory (or plain LangChain), and `HindsightStore` when you want LangGraph's native cross-thread store semantics. See the [decision guide above](#which-layer-to-use-when).

### Why did recall return nothing for a key I stored?

`HindsightStore.get()` is recall-based, not a direct key lookup — only exact `document_id` matches return, and items outside the top recall results can look missing. Use `asearch` with a semantic query instead.

## Next Steps

- Follow the [LangGraph setup guide](https://hindsight.vectorize.io/guides/2026/05/04/guide-langgraph-memory-with-hindsight) to install and wire the integration
- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted memory backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [LangGraph integration docs](https://hindsight.vectorize.io/docs/integrations/langgraph)
