---
title: "Reduce Hindsight Consolidation Memory Fan-Out Safely"
authors: [benfrank241]
date: 2026-04-28T14:00:00Z
tags: [consolidation, memory, operations, guide]
description: "Reduce Hindsight consolidation memory fan-out by tuning recall budget, source fact limits, and FlashRank memory settings on large banks in production."
image: /img/guides/guide-reduce-hindsight-consolidation-memory-fan-out.png
hide_table_of_contents: true
---

![Reduce Hindsight Consolidation Memory Fan-Out Safely](/img/guides/guide-reduce-hindsight-consolidation-memory-fan-out.png)

If you need to **reduce Hindsight consolidation memory fan out**, the recent defaults are a real improvement. Consolidation used to be able to amplify memory use during internal recall, especially on large banks where source fact hydration and reranker behavior could keep RSS higher than expected. The new defaults make that path much more bounded. Keep [the configuration guide](https://hindsight.vectorize.io/sdks/developer/configuration), [the observations guide](https://hindsight.vectorize.io/sdks/developer/observations), [the installation guide](https://hindsight.vectorize.io/sdks/developer/installation), and [the docs home](https://hindsight.vectorize.io) nearby while you tune it.

<!-- truncate -->

## The quick answer

- Consolidation recall now defaults to a low budget, which reduces how many candidate rows each recall arm tries to pull in.
- Source facts inside consolidation are now capped by default, instead of staying unlimited.
- FlashRank now defaults to a bounded CPU memory arena setting, which helps prevent monotonically growing RSS after consolidation work completes.

## Why consolidation used to fan out

Consolidation is not just a simple merge pass. It can trigger internal recall work so the system can find related observations, hydrate source facts, and decide what to update or combine. On large banks, that can get expensive fast if the recall budget is wide, source facts are effectively unlimited, and the reranker runtime keeps memory arenas hot.

That is the pattern this update is addressing. It does not remove consolidation. It narrows the expensive parts so large banks stay more predictable.

## Use the new bounded defaults first

The new baseline is intentionally conservative:

```bash
export HINDSIGHT_API_CONSOLIDATION_RECALL_BUDGET=low
export HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS=4096
export HINDSIGHT_API_RERANKER_FLASHRANK_CPU_MEM_ARENA=false
```

Those defaults reduce candidate fan out, cap how much source evidence gets pulled into the prompt, and stop ONNX Runtime from holding onto an ever growing CPU arena after consolidation batches finish.

## Tune up only when you have a reason

If you move beyond the defaults, do it on purpose.

- Raise `HINDSIGHT_API_CONSOLIDATION_RECALL_BUDGET` only when low recall is clearly missing useful related observations.
- Raise `HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS` only when the LLM needs more supporting evidence to make stable updates.
- Review `HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_ROUND` and `HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE` in [the configuration guide](https://hindsight.vectorize.io/sdks/developer/configuration) if you want to trade throughput against peak memory pressure.

The point is to keep the expensive path narrow by default, then widen one lever at a time if the bank actually needs it.

## Check the rest of the deployment shape too

Consolidation tuning only solves consolidation. If RSS still looks bad, compare it against the wider deployment:

- are you running the full image instead of slim?
- is the worker colocated with the API on the same small host?
- is PostgreSQL sharing the same memory envelope?
- are you using local reranking when an external reranker would fit better?

That is why [the installation guide](https://hindsight.vectorize.io/sdks/developer/installation) and [the services guide](https://hindsight.vectorize.io/sdks/developer/services) still matter here. Consolidation fan out is one contributor, not the entire footprint story.

## A simple operating playbook

A sane production playbook looks like this:

1. Start with the new defaults.
2. Watch RSS during large consolidation rounds.
3. Only tune one knob at a time.
4. Re-test on the same bank shape.
5. Keep notes on which change actually moved the needle.

That discipline matters because memory problems often feel mysterious when several variables change at once.

## FAQ

### Does low recall budget hurt normal user recall quality?

No. This setting is specifically for the internal recall pass inside consolidation, not the general recall path users call directly.

### Why cap source facts at 4096 tokens?

Because unlimited source fact hydration was one of the worst memory amplifiers on large banks. A cap makes the prompt cost much more predictable.

### Should I turn the FlashRank CPU memory arena back on?

Usually no. Leave it off unless you have measured a real need and are comfortable trading bounded RSS for a different allocation pattern.

## Next Steps

- [Hindsight Cloud](https://hindsight.vectorize.io)
- [the configuration guide](https://hindsight.vectorize.io/sdks/developer/configuration)
- [the observations guide](https://hindsight.vectorize.io/sdks/developer/observations)
- [the installation guide](https://hindsight.vectorize.io/sdks/developer/installation)
- [the docs home](https://hindsight.vectorize.io)
