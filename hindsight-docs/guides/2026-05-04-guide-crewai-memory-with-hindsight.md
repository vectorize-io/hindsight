---
title: "Guide: Add CrewAI Persistent Memory with Hindsight"
authors: [benfrank241]
date: 2026-05-04T15:00:00Z
tags: [how-to, crewai, multi-agent, memory]
description: "Add CrewAI persistent memory with Hindsight using ExternalMemory, HindsightStorage, and optional per agent banks so repeated crew runs build on earlier work."
image: /img/guides/guide-crewai-memory-with-hindsight.png
hide_table_of_contents: true
---

![Guide: Add CrewAI Persistent Memory with Hindsight](/img/guides/guide-crewai-memory-with-hindsight.png)

If you want **CrewAI persistent memory with Hindsight**, the cleanest approach is to plug `HindsightStorage` into CrewAI's `ExternalMemory` interface and let CrewAI search memory before tasks while Hindsight stores task outputs after they complete. That gives recurring crews real continuity instead of treating every kickoff like a blank slate.

This is especially useful for research, planning, and operational crews that repeat the same job over time. Hindsight preserves facts, entities, and relationships from earlier runs, so later runs can build on what the crew already learned.

If you want the underlying reference open while you work, keep [the CrewAI integration docs](https://hindsight.vectorize.io/docs/integrations/crewai), [the docs home](https://hindsight.vectorize.io/docs), [the quickstart guide](https://hindsight.vectorize.io/docs/quickstart), [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall), and [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain) nearby.

<!-- truncate -->

> **Quick answer**
>
> 1. Install the CrewAI integration or plugin.
> 2. Point it at Hindsight Cloud or a local Hindsight API.
> 3. Wire memory into your CrewAI runtime with a stable bank ID.
> 4. Store one preference or project fact, then start a fresh run.
> 5. Confirm that recall brings the earlier context back automatically.

## Why this setup works

CrewAI already exposes a storage boundary through `ExternalMemory`, so the integration is simple and predictable. `search()` maps to Hindsight recall, `save()` maps to retain, and optional per agent banks let you decide how much memory should be shared across the crew.

## Prerequisites

- CrewAI installed and a crew you can run end to end
- Python and `hindsight-crewai` installed
- A clear memory plan, one shared crew bank or isolated per agent banks

## Step 1: Install the integration

```bash
pip install hindsight-crewai
```

## Step 2: Connect CrewAI to Hindsight

```python
from hindsight_crewai import configure

configure(
    hindsight_api_url="http://localhost:8888",
    budget="mid",
    max_tokens=4096,
)
```

If you use [Hindsight Cloud](https://hindsight.vectorize.io), swap the API URL for `https://api.hindsight.vectorize.io` and include your API key.

## Step 3: Wire memory into your runtime

```python
from crewai.memory.external.external_memory import ExternalMemory
from crewai import Agent, Crew, Task
from hindsight_crewai import HindsightStorage

crew = Crew(
    agents=[Agent(role="Researcher", goal="Find information", backstory="...")],
    tasks=[Task(description="Research AI trends", expected_output="Report")],
    external_memory=ExternalMemory(
        storage=HindsightStorage(bank_id="my-crew")
    ),
)

crew.kickoff()
```

If you want explicit reasoning over memory, add `HindsightReflectTool` to the agents that need it. That is useful when a crew should synthesize memory before planning a new step.

## Step 4: Choose the right bank strategy

Start with one shared crew bank if agents collaborate on the same deliverable. Turn on `per_agent_banks=True` when each role should retain its own narrower memory, for example when a researcher and a writer should not mix every retained detail. Shared memory is better for coordination, isolated memory is better for specialization.

## Step 5: Verify that memory is working

1. Run a crew once and have it produce an output that contains a memorable fact or decision.
2. Kick off the same crew again with a related task.
3. Check that the crew recalls the earlier output or preference without being reminded manually.
4. If you enabled per agent banks, inspect behavior carefully because automatic task start search still queries the base bank by default.

If the second run can answer with details from the first run, your setup is working. If it cannot, turn on debug logging, check the configured bank ID, and confirm that the retain call actually completed.

## Common mistakes

- Expecting per agent search isolation without creating separate storage instances when CrewAI does not pass the agent into `search()`
- Using a bank mission that is too vague to guide memory extraction well
- Testing a second run with a different bank ID and assuming recall is broken

## FAQ

### Can each CrewAI agent have its own bank?

Yes. Set `per_agent_banks=True` or provide a custom `bank_resolver` if you want finer control.

### What does CrewAI store automatically?

CrewAI calls `save()` after task completion, and the Hindsight integration retains that output for future recall.

### When should I add the reflect tool?

Use it when an agent needs a synthesized view of memory rather than a raw recall result.

## Next Steps

- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted memory backend
- Read [the full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow [the quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Compare a related workflow in [team shared memory for AI coding agents](https://hindsight.vectorize.io/blog/team-shared-memory-ai-coding-agents)
