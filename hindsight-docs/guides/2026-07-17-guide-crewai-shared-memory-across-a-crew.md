---
title: "Guide: Shared Memory Across a CrewAI Crew"
authors: [benfrank241]
date: 2026-07-17
tags: [how-to, crewai, multi-agent, memory]
description: "Decide when a CrewAI crew should share one Hindsight memory bank versus giving each agent its own isolated bank, and how memory flows between agents."
image: /img/guides/guide-crewai-shared-memory-across-a-crew.svg
hide_table_of_contents: true
---

![Guide: Shared Memory Across a CrewAI Crew](/img/guides/guide-crewai-shared-memory-across-a-crew.svg)

If you run a multi-agent CrewAI crew, the real question is not whether to add memory but how to scope it. **Shared memory across a CrewAI crew** means every agent — the researcher, the analyst, the writer — reads and writes to one Hindsight bank, so a finding produced by one agent is visible to the next. The alternative is per-agent isolation, where each role keeps its own bank and specializes without cross-contamination.

This is a strategy decision, not an install step. CrewAI already exposes a storage boundary through `ExternalMemory`, and Hindsight's `HindsightStorage` plugs into it: `search()` maps to recall at the start of each task, `save()` maps to retain after each task completes. What you choose for `bank_id` and `per_agent_banks` decides whether the crew behaves like one team with a shared notebook or a set of specialists with private files.

This guide covers when a shared crew bank helps, when per-agent isolation is better, how memory actually flows between agents during a kickoff, and a hybrid pattern that gives you both. If you have not wired up the integration yet, do that first with the [CrewAI setup guide](https://hindsight.vectorize.io/guides/2026/05/04/guide-crewai-memory-with-hindsight), then come back here to choose a scoping strategy.

<!-- truncate -->

> **Quick answer**
>
> 1. Default to one shared crew bank: `HindsightStorage(bank_id="my-crew")`.
> 2. Every agent then reads and writes the same memory — a researcher's findings are visible to the writer.
> 3. Switch to `per_agent_banks=True` when roles should specialize without mixing every retained detail.
> 4. For a hybrid, keep the shared bank for coordination and give one agent a private `HindsightStorage` instance as scratch.
> 5. Verify by having agent A store a fact in run one and agent B recall it in run two.

## Why crews need shared memory

A crew is a team, and teams work best when knowledge does not evaporate between hand-offs. In a single kickoff, CrewAI runs tasks in sequence: the researcher gathers findings, then the writer summarizes them. Within that one run, CrewAI passes task output down the chain. But across runs — the second, third, and tenth time you kick off the same crew — that continuity is gone unless something persists it.

Shared memory is what turns a crew from a one-shot pipeline into an institution that accumulates knowledge. When all agents write to one bank, the crew builds a shared body of facts, entities, and relationships that every role can draw on. The researcher does not re-research a topic it already covered last week; the writer recalls the tone and conclusions the crew settled on earlier. This is the natural fit for research, planning, and operational crews that repeat the same job over time.

Hindsight makes the shared bank more than a transcript dump. On retain, it extracts facts, entities, and relationships from raw task output; on recall, it runs semantic search, BM25, graph traversal, and reranking to surface what is relevant to the current task. So a shared crew bank is not just "everyone sees everything" — it is "everyone sees what matters right now."

## One shared crew bank vs per-agent banks

The choice comes down to whether your agents collaborate on one deliverable or specialize on separate concerns.

**Use one shared crew bank when:**

- Agents work toward the same output and need each other's context (a researcher feeding a writer).
- You want institutional or crew memory — decisions, conventions, and findings that outlive any single role.
- Coordination matters more than specialization.

```python
storage = HindsightStorage(bank_id="research-crew")
crew = Crew(agents=[researcher, writer], tasks=[...],
            external_memory=ExternalMemory(storage=storage))
```

**Use per-agent banks when:**

- Each role should retain its own narrower memory and not absorb every detail from other roles.
- Mixing memory would add noise — for example a fact-checking agent whose bank should stay clean of the writer's drafts.
- Specialization matters more than coordination.

```python
storage = HindsightStorage(bank_id="research-crew", per_agent_banks=True)
# Researcher -> "research-crew-researcher"
# Writer     -> "research-crew-writer"
```

With `per_agent_banks=True`, each agent's `save()` lands in a bank derived from its role. If you need a different naming scheme, pass a `bank_resolver` — a `(bank_id, agent) -> bank_id` function — for full control.

Shared memory is better for coordination; isolated memory is better for specialization. Most crews start with one shared bank and only split when a specific role clearly needs a clean, narrow store.

## How memory flows between agents

Understanding the flow matters because CrewAI's automatic memory calls behave differently from what you might assume.

During a kickoff, CrewAI calls `search()` at the start of each task and `save()` after each task completes. With a single shared bank, this is straightforward: every `save()` writes to `my-crew`, and every `search()` reads from `my-crew`. The researcher's retained findings are recalled when the writer's task begins, so memory flows forward through the crew automatically.

There is one important subtlety with `per_agent_banks=True`. CrewAI's `search()` method does not receive the agent parameter, so the automatic task-start recall queries the **base bank**, not the per-agent bank — while `save()` still writes to the per-agent bank. In other words, per-agent mode isolates writes but not the automatic reads. If you genuinely need per-agent search isolation, create a separate `HindsightStorage` instance per agent rather than relying on `per_agent_banks`.

For explicit, agent-controlled memory flow, add the `HindsightReflectTool` to the agents that need it. Instead of raw recall snippets, an agent calling this tool gets a synthesized, disposition-aware answer over all relevant memories — useful when a researcher should check "what do we already know?" before starting, or a writer should recall prior conclusions before drafting. The reflect tool takes its own `bank_id`, so you can point one agent at the shared bank and another at a private one within the same crew.

## Connect CrewAI to Hindsight

If you have not installed and wired up the integration, follow the dedicated setup guide — it covers `pip install hindsight-crewai`, `configure(...)` for Cloud or self-hosted, and wiring `HindsightStorage` into `ExternalMemory`:

- [Guide: Add CrewAI Persistent Memory with Hindsight](https://hindsight.vectorize.io/guides/2026/05/04/guide-crewai-memory-with-hindsight)

Once the crew runs with memory at all, the rest of this guide is about choosing how to scope it.

## A hybrid: shared context plus private scratch

You do not have to pick one extreme. A common pattern is a shared crew bank for coordination plus a private bank for one agent that needs a scratch space.

The shared bank holds the crew's institutional memory: findings, decisions, and conventions every role should see. Then, for an agent that generates a lot of intermediate reasoning you do not want polluting the shared store — say an analyst working through drafts — give that agent a second, private `HindsightStorage` (or a private-bank reflect tool) it writes to on its own.

```python
# Shared crew memory: everyone coordinates through this bank
shared = HindsightStorage(bank_id="research-crew",
                          mission="Track research findings and crew decisions.")

# Private scratch for the analyst, kept out of the shared bank
analyst_scratch = HindsightReflectTool(bank_id="research-crew-analyst-scratch")
analyst = Agent(role="Analyst", goal="...", backstory="...",
                tools=[analyst_scratch])

crew = Crew(agents=[researcher, analyst, writer], tasks=[...],
            external_memory=ExternalMemory(storage=shared))
```

The crew coordinates through the shared bank via `ExternalMemory`, while the analyst reasons over its own private bank through the reflect tool. Setting a `mission` on the shared bank helps Hindsight organize what it retains around the crew's actual purpose.

## Verify that shared memory is working

The point of shared memory is that one agent's output shows up in another agent's context on a later run. Test exactly that:

1. Run the crew once and have an early agent (the researcher) produce an output with a memorable fact or decision.
2. Let the run finish so CrewAI's `save()` retains that output to the shared bank.
3. Kick off the crew again with a related task aimed at a different agent (the writer).
4. Confirm the writer recalls the earlier fact without being reminded manually.

For example: run one has the researcher analyze the benefits of Rust and store the findings. Run two changes the task to "compare Rust with Go" — if the crew brings the earlier Rust analysis back on its own, shared memory is working.

If the second run cannot recall the first, turn on `verbose=True`, confirm both runs used the same `bank_id`, and check that the retain call actually completed.

## Common mistakes

### Assuming per-agent banks isolate recall too

`per_agent_banks=True` isolates `save()` writes, but CrewAI's automatic `search()` still queries the base bank because it does not pass the agent. For real per-agent search isolation, use a separate `HindsightStorage` instance per agent.

### Splitting into per-agent banks too early

Most crews benefit from shared memory first. Isolate a role only when its bank clearly needs to stay clean and narrow — not by default.

### Changing the bank ID between runs

Recall only brings back earlier context when both runs use the same `bank_id`. Testing run two with a different bank ID looks like broken recall but is really a scoping mismatch.

### A mission that is too vague

If you set a bank `mission`, make it specific enough to guide extraction. A vague mission gives Hindsight little to organize memory around.

## FAQ

### Do I need Hindsight Cloud?

No. A self-hosted Hindsight server works too — point `configure(hindsight_api_url=...)` at your local API (for example `http://localhost:8888`) and drop the API key.

### Should the whole crew share one bank by default?

Usually yes. Start with one shared bank for coordination, and split into per-agent banks only when a specific role needs isolated, narrower memory.

### How does one agent's memory reach another agent?

Through the shared bank. With a single `bank_id`, every agent's `save()` writes there and every task-start `search()` reads from there, so findings flow forward automatically. Add `HindsightReflectTool` for explicit, agent-controlled recall.

### Can I mix shared and isolated memory in one crew?

Yes. Use a shared bank via `ExternalMemory` for coordination and give a specific agent a private `HindsightStorage` instance or a private-bank reflect tool as scratch.

## Next Steps

- Set up the integration first with the [CrewAI setup guide](https://hindsight.vectorize.io/guides/2026/05/04/guide-crewai-memory-with-hindsight)
- Start with [Hindsight Cloud](https://hindsight.vectorize.io) if you want a hosted memory backend
- Read the [full Hindsight docs](https://hindsight.vectorize.io/docs)
- Follow the [quickstart guide](https://hindsight.vectorize.io/docs/quickstart)
- Review [Hindsight's recall API](https://hindsight.vectorize.io/docs/api/recall)
- Review [Hindsight's retain API](https://hindsight.vectorize.io/docs/api/retain)
- Read the [CrewAI integration docs](https://hindsight.vectorize.io/docs/integrations/crewai)
