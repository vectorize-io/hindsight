---
title: "Guide: Add LangGraph Persistent Memory with Hindsight"
authors: [benfrank241]
date: 2026-05-04T15:00:00Z
tags: [how-to, langgraph, langchain, memory]
description: "Add LangGraph persistent memory with Hindsight using tools, recall and retain nodes, or the BaseStore adapter so agents remember users across runs."
image: /img/guides/guide-langgraph-memory-with-hindsight.png
hide_table_of_contents: true
---

![Guide: Add LangGraph Persistent Memory with Hindsight](/img/guides/guide-langgraph-memory-with-hindsight.png)

If you want **LangGraph persistent memory with Hindsight**, the cleanest setup is to add Hindsight tools or memory nodes to your graph and resolve a stable bank ID for each user or thread. That gives LangGraph agents continuity across runs instead of treating every graph execution like a fresh start.

Hindsight is a good fit here because the integration supports three patterns. You can expose retain, recall, and reflect as tools, add pre built recall and retain nodes around your LLM node, or use the BaseStore adapter when you want LangGraph native storage semantics.

If you want the underlying reference open while you work, keep [the LangGraph integration docs](https://hindsight.vectorize.io/docs/integrations/langgraph), [the docs home](https://hindsight.vectorize.io/docs), [the quickstart guide](https://hindsight.vectorize.io/docs/quickstart), [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall), and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain) nearby.

<!-- truncate -->

> **Quick answer**
>
> 1. Install the LangGraph integration or plugin.
> 2. Point it at Hindsight Cloud or a local Hindsight API.
> 3. Wire memory into your LangGraph runtime with a stable bank ID.
> 4. Store one preference or project fact, then start a fresh run.
> 5. Confirm that recall brings the earlier context back automatically.

## Why this setup works

LangGraph already has a clear execution graph, so memory placement is explicit. A recall node can run before your agent node, a retain node can run after it, and dynamic bank IDs from `RunnableConfig` make per user memory practical without hardcoding identifiers inside the graph.

## Prerequisites

- A working LangGraph or LangChain app
- Python and `hindsight-langgraph` installed
- A stable user or thread identifier that you can map to a Hindsight bank

## Step 1: Install the integration

```bash
pip install hindsight-langgraph
```

## Step 2: Connect LangGraph to Hindsight

```python
from hindsight_client import Hindsight

client = Hindsight(base_url="http://localhost:8888")
```

For [Hindsight Cloud](https://hindsight.vectorize.io), set the base URL to `https://api.hindsight.vectorize.io` and pass your API token through the client configuration you already use in your app.

## Step 3: Wire memory into your runtime

```python
from hindsight_client import Hindsight
from hindsight_langgraph import create_recall_node, create_retain_node
from langgraph.graph import StateGraph, MessagesState, START, END

client = Hindsight(base_url="http://localhost:8888")
recall = create_recall_node(client=client, bank_id_from_config="user_id")
retain = create_retain_node(client=client, bank_id_from_config="user_id")

builder = StateGraph(MessagesState)
builder.add_node("recall", recall)
builder.add_node("agent", agent_node)
builder.add_node("retain", retain)
builder.add_edge(START, "recall")
builder.add_edge("recall", "agent")
builder.add_edge("agent", "retain")
builder.add_edge("retain", END)

graph = builder.compile()
```

If you prefer tool calling, `create_hindsight_tools()` is the quickest option. If you want native LangGraph storage patterns, `HindsightStore` is the better fit.

## Step 4: Choose the right bank strategy

Resolve bank IDs from `RunnableConfig` whenever you have a stable user or tenant key. That keeps memory attached to the correct person across graph runs. If you are building an internal assistant for one team, a shared bank can make sense, but most production graphs should scope memory by user, tenant, or thread.

## Step 5: Verify that memory is working

1. Run the graph once and store a preference or project fact for a test user.
2. Invoke the graph again with the same `user_id` in `configurable`.
3. Ask a question that depends on the earlier fact and confirm that recall surfaces it.
4. Repeat the same test with a different `user_id` to confirm isolation.

If the second run can answer with details from the first run, your setup is working. If it cannot, turn on debug logging, check the configured bank ID, and confirm that the retain call actually completed.

## Common mistakes

- Binding tools in plain LangChain but forgetting to run the tool execution loop
- Using a different runtime key on the second run, which silently creates a new bank
- Choosing one shared bank for all users when the app really needs user scoped memory

## FAQ

### Should I use tools, nodes, or BaseStore?

Use tools when you want agent controlled memory calls, nodes when you want automatic recall and retain around the graph, and BaseStore when you want LangGraph native store patterns.

### Can this work with plain LangChain?

Yes. The tools pattern works in LangChain too, but you need to handle tool execution yourself.

### How should I scope banks?

Per user is the safest default. Add tenant or thread context when your app needs stronger isolation.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted memory backend
- Read [the full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow [the quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Compare a related workflow in [Agno persistent memory](https://hindsight.vectorize.io/blog/agno-persistent-memory)
