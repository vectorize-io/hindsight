---
title: "Guide: Add LlamaIndex Persistent Memory with Hindsight"
authors: [benfrank241]
date: 2026-05-04T15:00:00Z
tags: [how-to, llamaindex, agent-frameworks, memory]
description: "Add LlamaIndex persistent memory with Hindsight using HindsightMemory for auto recall and retain, or HindsightToolSpec when agents need explicit tools."
image: /img/guides/guide-llamaindex-memory-with-hindsight.png
hide_table_of_contents: true
---

![Guide: Add LlamaIndex Persistent Memory with Hindsight](/img/guides/guide-llamaindex-memory-with-hindsight.png)

If you want **LlamaIndex persistent memory with Hindsight**, the smoothest setup is to use `HindsightMemory` for automatic recall and retain, then switch to `HindsightToolSpec` only when you want the agent to decide explicitly when memory tools should run. That gives LlamaIndex agents continuity across sessions without turning every workflow into manual memory plumbing.

This is a strong fit for LlamaIndex because the integration supports both patterns cleanly. You can stay high level with `BaseMemory`, or expose retain, recall, and reflect tools when you want more control.

If you want the underlying reference open while you work, keep [the LlamaIndex integration docs](https://hindsight.vectorize.io/docs/integrations/llamaindex), [the docs home](https://hindsight.vectorize.io/docs), [the quickstart guide](https://hindsight.vectorize.io/docs/quickstart), [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall), and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain) nearby.

<!-- truncate -->

> **Quick answer**
>
> 1. Install the LlamaIndex integration or plugin.
> 2. Point it at Hindsight Cloud or a local Hindsight API.
> 3. Wire memory into your LlamaIndex runtime with a stable bank ID.
> 4. Store one preference or project fact, then start a fresh run.
> 5. Confirm that recall brings the earlier context back automatically.

## Why this setup works

LlamaIndex already has a memory abstraction, so Hindsight can attach at the right layer. `HindsightMemory` recalls context before the agent runs and retains messages after the turn, while the tool spec pattern gives you explicit tool control when that fits the agent design better.

## Prerequisites

- A working LlamaIndex agent or workflow
- Python and `hindsight-llamaindex` installed
- A stable bank ID that represents the user, project, or assistant you want to persist over time

## Step 1: Install the integration

```bash
pip install hindsight-llamaindex
```

## Step 2: Connect LlamaIndex to Hindsight

```python
from hindsight_client import Hindsight

client = Hindsight(base_url="http://localhost:8888")
```

For [Hindsight Cloud](https://hindsight.vectorize.io), use the cloud API URL and authenticate the client the same way you do in other Python integrations.

## Step 3: Wire memory into your runtime

```python
import asyncio
from hindsight_client import Hindsight
from hindsight_llamaindex import HindsightMemory
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

async def main():
    client = Hindsight(base_url="http://localhost:8888")
    memory = HindsightMemory.from_client(
        client=client,
        bank_id="user-123",
        mission="Track user preferences and project context",
    )

    agent = ReActAgent(tools=[], llm=OpenAI(model="gpt-4o"))
    response = await agent.run("Remember that I prefer dark mode", memory=memory)
    print(response)

asyncio.run(main())
```

If you want explicit memory tools instead, create `HindsightToolSpec` and pass its tools into the agent.

## Step 4: Choose the right bank strategy

Use per user banks for assistants that follow one person across sessions. Use per project banks when the same user works across multiple unrelated domains. The key is consistency: whatever key you choose, keep it stable so `aput()` and `aget()` hit the same bank over time.

## Step 5: Verify that memory is working

1. Run the agent once and store a test preference or project fact.
2. Start a new session with the same bank ID and ask for that detail.
3. Confirm that `HindsightMemory` injects the earlier context without needing a manual tool call.
4. If recall fails, verify that the bank was auto created and that the same bank ID was reused.

If the second run can answer with details from the first run, your setup is working. If it cannot, turn on debug logging, check the configured bank ID, and confirm that the retain call actually completed.

## Common mistakes

- Using a new bank ID for each run, which destroys continuity
- Choosing the tool spec pattern when automatic memory would be simpler
- Leaving the mission blank when the agent has a specialized domain and would benefit from better extraction guidance

## FAQ

### Should I start with HindsightMemory or HindsightToolSpec?

Start with `HindsightMemory` if you want automatic memory behavior. Switch to `HindsightToolSpec` when your agent should decide when to store or search memory.

### Does HindsightMemory keep a local buffer too?

Yes. The local chat buffer is separate from the long term bank, so each new session can start fresh while still recalling past memories.

### What should the bank mission say?

Describe the facts and context you want remembered, such as project decisions, user preferences, or recurring workflow details.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted memory backend
- Read [the full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow [the quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Compare a related workflow in [Agno persistent memory](https://hindsight.vectorize.io/blog/agno-persistent-memory)
