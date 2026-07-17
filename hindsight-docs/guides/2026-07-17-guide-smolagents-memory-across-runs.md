---
title: "Guide: Give SmolAgents Memory That Persists Across Runs"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, smolagents, memory]
description: "Give SmolAgents cross-run memory with Hindsight so a code-executing agent accumulates what worked and what failed instead of starting cold on every run."
image: /img/guides/guide-smolagents-memory-across-runs.svg
hide_table_of_contents: true
---

![Guide: Give SmolAgents Memory That Persists Across Runs](/img/guides/guide-smolagents-memory-across-runs.svg)

SmolAgents run in discrete runs. Each `agent.run(...)` spins up, reasons, executes code, and returns — and then everything it learned evaporates. The distinct problem this guide solves is **memory that persists across runs**: giving a code-executing agent a durable store so it accumulates task-specific facts, remembers which approaches worked, and avoids the failures it already hit — instead of rediscovering all of that on every fresh run.

Hindsight fits this cleanly because the SmolAgents integration exposes memory as native Tool subclasses. The agent can call `hindsight_retain` to store what it learned during a run, and a new run can start by recalling everything prior runs discovered. Point every run at the same stable bank ID and the memory carries forward. This is not a setup rehash — a full setup guide already exists, and this one focuses on the cross-run accumulation pattern on top of it.

This guide covers why per-run agents start cold, what is actually worth retaining from a run, how to recall prior runs at the start of a new one, and how to wire the drop-in Tool subclasses so the loop closes itself.

<!-- truncate -->

> **Quick answer**
>
> 1. `pip install hindsight-smolagents` and connect it to Hindsight (see the setup guide).
> 2. Use a **stable bank ID** for the same agent/user/project across every run.
> 3. Add `create_hindsight_tools(bank_id=...)` so the agent can retain and recall.
> 4. Start each run by injecting prior runs with `memory_instructions()`.
> 5. Verify that a later run answers with facts a earlier run stored.

## Why per-run agents start cold

A SmolAgents agent has memory *within* a run — its reasoning steps, tool observations, and intermediate code state live in the agent loop. But that state is scoped to the run. The next `agent.run(...)` starts with an empty slate.

For a code-executing agent that is expensive. In run one it might discover that a library needs a specific version pin, that an API returns paginated results, or that a particular approach throws and a different one works. In run two, none of that exists. The agent re-explores, re-fails, and re-derives the same facts — burning tokens and wall-clock time on knowledge it already had.

Cross-run memory breaks that cycle. Instead of the run boundary erasing what the agent learned, a durable store carries the useful residue forward: the facts, the working approaches, and the dead ends worth avoiding.

## What to retain from a run

Not everything from a run is worth keeping. The transcript is noisy; the value is in the durable lessons. Good candidates to `hindsight_retain`:

- **Task-specific facts** the agent had to discover — schema shapes, required version pins, endpoint quirks, credentials locations (never secrets themselves).
- **Approaches that worked** — "parsing the CSV with `pandas.read_csv(..., sep=';')` succeeded where the default separator failed."
- **Approaches that failed** — "calling the endpoint without pagination truncates at 100 rows," so the next run doesn't repeat it.
- **Decisions and conventions** the agent or user settled on during the run.

Let the agent call `hindsight_retain` when it reaches one of these moments, or retain a distilled summary at the end of a run. The point is to store the *lesson*, not the raw step-by-step trace. Hindsight extracts and consolidates facts on ingest, so short, specific statements retain best. For the lower-level behavior, see [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain).

## Recall prior runs at the start of a new one

A new run should begin already knowing what earlier runs learned. There are two ways to bring that context in, and they complement each other:

- **Front-load with `memory_instructions()`.** Before the run starts, pre-recall the most relevant memories and inject them into the system prompt. The agent begins with prior lessons in context, so it plans around them from step one:

  ```python
  from hindsight_smolagents import create_hindsight_tools, memory_instructions
  from smolagents import CodeAgent, HfApiModel

  BANK = "code-agent-project-x"

  memories = memory_instructions(
      bank_id=BANK,
      query="prior approaches, failures, and task facts for this project",
  )

  agent = CodeAgent(
      tools=create_hindsight_tools(bank_id=BANK),
      model=HfApiModel(),
      system_prompt=f"You are a coding agent.\n\n{memories}",
  )
  ```

- **On-demand with `hindsight_recall`.** Because recall is also a tool, the agent can search memory mid-run when it hits a question the front-loaded context didn't cover. That keeps early context tight while still giving the agent access to the full store.

Using the **same `bank_id` on every run** is what makes this work — the bank is the thread that ties runs together. For the lower-level behavior, see [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall).

## Connect SmolAgents to Hindsight

Install and connect once — this is covered in full in the setup guide, so this guide won't repeat it:

```bash
pip install hindsight-smolagents
```

Point it at Hindsight Cloud or a local server, then move on. See **[Guide: Add SmolAgents Persistent Memory with Hindsight](https://hindsight.vectorize.io/guides/2026/05/04/guide-smolagents-memory-with-hindsight)** for the complete install and connection walkthrough, including `configure()` and self-hosting.

## Drop-in memory tools your agent can call

The integration ships three native `Tool` subclasses, so the agent uses memory the same way it uses any other tool. Create them with the factory:

```python
from hindsight_smolagents import create_hindsight_tools

tools = create_hindsight_tools(bank_id="code-agent-project-x")
```

That gives the agent:

- **`hindsight_retain`** — store a fact, lesson, or decision to long-term memory.
- **`hindsight_recall`** — search long-term memory for relevant prior facts.
- **`hindsight_reflect`** — synthesize a reasoned answer from stored memories.

You can include only the tools you need. For a cross-run accumulation loop, retain and recall are the essentials; reflect is useful when you want a synthesized answer rather than raw facts:

```python
tools = create_hindsight_tools(
    bank_id="code-agent-project-x",
    enable_retain=True,
    enable_recall=True,
    enable_reflect=False,
)
```

If you'd rather instantiate tools individually, `HindsightRetainTool` and `HindsightRecallTool` take the same `bank_id`. The [integration docs](https://hindsight.vectorize.io/docs/integrations/smolagents) list every parameter, including tags for scoping and recall budget.

## Verify memory persists across runs

The test is simple: prove that run two knows something only run one could have learned.

1. In run one, let the agent solve a task and call `hindsight_retain` with a concrete lesson — for example, "the export endpoint paginates at 100 rows, so always pass `page`."
2. Let the run finish. The retain is stored to the bank.
3. Start run two with the **same `bank_id`**, and either front-load with `memory_instructions()` or ask a question that should trigger `hindsight_recall`.
4. Ask the agent about that pagination behavior, or watch whether it plans around it without rediscovering it.

If run two answers using the fact from run one, cross-run memory is working. If it starts cold, check that both runs used the same bank ID and that the retain call actually completed.

## Common mistakes

### Using a different bank ID per run

The bank is what ties runs together. If each run generates a fresh bank ID, nothing persists. Pin one stable ID for the agent, user, or project.

### Retaining the raw transcript

Storing the full step-by-step trace buries the lesson in noise. Retain short, specific facts and decisions instead — that's what recall surfaces well later.

### Adding tools but never injecting context

Giving the agent `hindsight_recall` lets it search on demand, but it won't automatically start each run with prior context unless you also inject `memory_instructions()`. Use both for a reliable warm start.

### Testing recall before retain completes

If you check run two before run one's retain finished, the memory may not be stored yet. Let the first run complete before asserting the second one remembers.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — point the tools at your local API URL. See the setup guide for connection details.

### How does an agent decide when to retain?

The agent calls `hindsight_retain` like any tool, so you can prompt it to store lessons at natural moments, or retain a distilled summary yourself at the end of a run. Store the lesson, not the raw trace.

### Does this slow down every run?

Front-loading with `memory_instructions()` adds one recall before the run and keeps injected context bounded by `max_results` and `max_tokens`. On-demand `hindsight_recall` only runs when the agent chooses to call it.

### Is this limited to CodeAgent?

No. The tools follow the SmolAgents tool model, so any agent that accepts tools can use the same cross-run memory pattern.

## Next Steps

- Start with the full [SmolAgents setup guide](https://hindsight.vectorize.io/guides/2026/05/04/guide-smolagents-memory-with-hindsight)
- Try [Hindsight Cloud](https://hindsight.vectorize.io) for a hosted memory backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [SmolAgents integration docs](https://hindsight.vectorize.io/docs/integrations/smolagents)
