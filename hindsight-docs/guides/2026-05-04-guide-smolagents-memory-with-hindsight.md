---
title: "Guide: Add SmolAgents Persistent Memory with Hindsight"
authors: [benfrank241]
date: 2026-05-04T15:00:00Z
tags: [how-to, smolagents, agent-frameworks, memory]
description: "Add SmolAgents persistent memory with Hindsight using native Tool subclasses, optional memory instructions, and stable bank IDs for repeat runs."
image: /img/guides/guide-smolagents-memory-with-hindsight.png
hide_table_of_contents: true
---

![Guide: Add SmolAgents Persistent Memory with Hindsight](/img/guides/guide-smolagents-memory-with-hindsight.png)

If you want **SmolAgents persistent memory with Hindsight**, the cleanest setup is to create Hindsight tools for retain, recall, and reflect, then optionally pre inject recalled memories into the system prompt with `memory_instructions()`. That gives SmolAgents long term memory without changing the rest of the agent loop.

This pattern fits SmolAgents well because the integration uses native Tool subclasses. You keep the familiar CodeAgent flow while adding a durable memory layer behind it.

If you want the underlying reference open while you work, keep [the SmolAgents integration docs](https://hindsight.vectorize.io/docs/integrations/smolagents), [the docs home](https://hindsight.vectorize.io/docs), [the quickstart guide](https://hindsight.vectorize.io/docs/quickstart), [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall), and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain) nearby.

<!-- truncate -->

> **Quick answer**
>
> 1. Install the SmolAgents integration or plugin.
> 2. Point it at Hindsight Cloud or a local Hindsight API.
> 3. Wire memory into your SmolAgents runtime with a stable bank ID.
> 4. Store one preference or project fact, then start a fresh run.
> 5. Confirm that recall brings the earlier context back automatically.

## Why this setup works

SmolAgents is designed around tools, so Hindsight can drop in cleanly. The agent can call memory tools directly when needed, while a simple instruction string can front load the most relevant context at the start of a run.

## Prerequisites

- A working SmolAgents agent, such as `CodeAgent`
- Python and `hindsight-smolagents` installed
- A stable bank ID for the same user, project, or assistant across runs

## Step 1: Install the integration

```bash
pip install hindsight-smolagents
```

## Step 2: Connect SmolAgents to Hindsight

```python
from hindsight_smolagents import configure

configure(
    hindsight_api_url="http://localhost:8888",
    budget="mid",
    max_tokens=4096,
)
```

You can skip global configuration and pass `hindsight_api_url` directly into `create_hindsight_tools()` if you prefer.

## Step 3: Wire memory into your runtime

```python
from smolagents import CodeAgent, HfApiModel
from hindsight_smolagents import create_hindsight_tools, memory_instructions

tools = create_hindsight_tools(
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
)

memories = memory_instructions(
    bank_id="user-123",
    hindsight_api_url="http://localhost:8888",
)

agent = CodeAgent(
    tools=tools,
    model=HfApiModel(),
    system_prompt=f"You are a helpful assistant.

{memories}",
)
```

## Step 4: Choose the right bank strategy

Use one bank per user when the same person should be remembered across tasks. Use one bank per project when a single user switches between unrelated contexts. The important part is that both the tools and the optional memory instructions use the same bank key.

## Step 5: Verify that memory is working

1. Ask the agent to remember a preference or reusable project fact.
2. Run the agent again and ask for that detail.
3. Confirm that recall finds the earlier memory, either via injected context or a tool call.
4. If you test with multiple users, switch bank IDs and verify that memories stay isolated.

If the second run can answer with details from the first run, your setup is working. If it cannot, turn on debug logging, check the configured bank ID, and confirm that the retain call actually completed.

## Common mistakes

- Passing one bank ID into the tools and a different one into `memory_instructions()`
- Using only the tools but expecting automatic prompt injection
- Leaving recall on broad shared banks when the application really needs user isolation

## FAQ

### Do I need to use memory instructions?

No. They are optional. Use them when you want context injected automatically before the agent starts reasoning.

### Can I use only recall and retain?

Yes. `create_hindsight_tools()` lets you include only the tools you need.

### Is this limited to CodeAgent?

No. The integration follows the SmolAgents tool model, so the same memory tools can fit other agents that accept tools.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted memory backend
- Read [the full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow [the quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Compare a related workflow in [Agno persistent memory](https://hindsight.vectorize.io/blog/agno-persistent-memory)
