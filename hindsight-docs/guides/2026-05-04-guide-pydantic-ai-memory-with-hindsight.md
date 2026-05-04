---
title: "Guide: Add Pydantic AI Persistent Memory with Hindsight"
authors: [benfrank241]
date: 2026-05-04T15:00:00Z
tags: [how-to, pydantic-ai, agent-frameworks, memory]
description: "Add Pydantic AI persistent memory with Hindsight using async memory tools and auto injected instructions so agents remember users and workflow context."
image: /img/guides/guide-pydantic-ai-memory-with-hindsight.png
hide_table_of_contents: true
---

![Guide: Add Pydantic AI Persistent Memory with Hindsight](/img/guides/guide-pydantic-ai-memory-with-hindsight.png)

If you want **Pydantic AI persistent memory with Hindsight**, the cleanest pattern is to give the agent Hindsight tools and add `memory_instructions()` so relevant memories are injected before each run. That gives a Pydantic AI agent long term memory without forcing you into thread pool workarounds or hand rolled recall logic.

This integration is appealing because it is async native from end to end. Retain, recall, and reflect all use async paths directly, which keeps the setup simple in production services.

If you want the underlying reference open while you work, keep [the Pydantic AI integration docs](https://hindsight.vectorize.io/docs/integrations/pydantic-ai), [the docs home](https://hindsight.vectorize.io/docs), [the quickstart guide](https://hindsight.vectorize.io/docs/quickstart), [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall), and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain) nearby.

<!-- truncate -->

> **Quick answer**
>
> 1. Install the Pydantic AI integration or plugin.
> 2. Point it at Hindsight Cloud or a local Hindsight API.
> 3. Wire memory into your Pydantic AI runtime with a stable bank ID.
> 4. Store one preference or project fact, then start a fresh run.
> 5. Confirm that recall brings the earlier context back automatically.

## Why this setup works

Pydantic AI agents already accept tools and instructions as first class concepts. Hindsight fits both: memory tools give the agent explicit actions, and `memory_instructions()` gives it recalled context before a run starts. That combination covers both automatic and agent driven memory usage.

## Prerequisites

- A working Pydantic AI agent
- Python and `hindsight-pydantic-ai` installed
- A bank ID strategy that stays stable across runs for the same user or project

## Step 1: Install the integration

```bash
pip install hindsight-pydantic-ai
```

## Step 2: Connect Pydantic AI to Hindsight

```python
from hindsight_client import Hindsight

client = Hindsight(base_url="http://localhost:8888")
```

You can also call `configure()` once and create tools without passing a client each time.

## Step 3: Wire memory into your runtime

```python
from hindsight_client import Hindsight
from hindsight_pydantic_ai import create_hindsight_tools, memory_instructions
from pydantic_ai import Agent

client = Hindsight(base_url="http://localhost:8888")

agent = Agent(
    "openai:gpt-4o",
    tools=create_hindsight_tools(client=client, bank_id="user-123"),
    instructions=[memory_instructions(client=client, bank_id="user-123")],
)

result = await agent.run("What do you remember about my preferences?")
print(result.output)
```

If you want tools only, drop `memory_instructions()`. If you want automatic injection with no tools, keep only the instructions.

## Step 4: Choose the right bank strategy

Per user banks are the safest default for assistants that follow an individual. Per workflow or project banks make more sense when one user operates several unrelated systems. Avoid rotating bank IDs per request, because that makes the agent look stateless even when the integration is correct.

## Step 5: Verify that memory is working

1. Run the agent once and ask it to remember a preference or operating rule.
2. Run it again with the same bank ID and ask for that detail.
3. Check whether the answer reflects the earlier memory before any explicit tool call is made.
4. If not, inspect the instruction output and confirm that recall was using the expected bank.

If the second run can answer with details from the first run, your setup is working. If it cannot, turn on debug logging, check the configured bank ID, and confirm that the retain call actually completed.

## Common mistakes

- Adding tools but forgetting to include memory instructions when you expected automatic recall
- Using a different bank ID in tools and instructions
- Overriding the global configuration in one place and forgetting that per call arguments win

## FAQ

### Do I need both tools and memory instructions?

No. Use both when you want automatic injection plus explicit memory actions. Use one or the other when that fits your agent design better.

### Why is async support important here?

Because it keeps the integration simple inside async apps and avoids awkward sync wrappers around memory calls.

### Can I filter recalled memories by tag?

Yes. The integration supports `recall_tags` and `recall_tags_match` so you can narrow memory scope.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted memory backend
- Read [the full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow [the quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Compare a related workflow in [Agno persistent memory](https://hindsight.vectorize.io/blog/agno-persistent-memory)
