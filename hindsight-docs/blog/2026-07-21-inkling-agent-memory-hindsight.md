---
title: "Mira Murati's Inkling Just Dropped. We Made It Your Agent's Memory in 5 Minutes."
authors: [benfrank241]
slug: "2026/07/21/inkling-agent-memory-hindsight"
date: 2026-07-21T12:00
tags: [hindsight, inkling, thinking-machines, agent-memory, open-weights, tutorial]
description: "Thinking Machines open-sourced Inkling. We made it an AI agent's long-term memory in four env vars, and it resolved 'last week' to a real date on the first try. Here is how."
image: /img/blog/inkling-agent-memory-hindsight.png
hide_table_of_contents: true
---

![Wiring Thinking Machines' Inkling into Hindsight as an agent memory engine](/img/blog/inkling-agent-memory-hindsight.png)

On July 15, Thinking Machines Lab shipped [Inkling](https://thinkingmachines.ai/news/introducing-inkling/), Mira Murati's first open-weights model. A 975-billion-parameter mixture-of-experts, Apache 2.0, multimodal, a one-million-token context window. The entire internet immediately did the same thing with it: started fine-tuning.

We did something the launch post did not suggest. We pointed it at the least glamorous, most underrated job in the whole agent stack: **being the brain behind your agent's long-term memory.** No fine-tuning. Four environment variables. And within about eight seconds it took a messy sentence, pulled out clean facts, and quietly resolved the phrase "last week" into an actual calendar date.

Here is exactly how to do it, and what happened when we tried.

<!-- truncate -->

## TL;DR

- Inkling runs behind any OpenAI-compatible endpoint (NVIDIA, Together, Fireworks, Baseten, Hugging Face).
- Hindsight, our open-source agent-memory system, lets you swap in that endpoint as its extraction and reasoning model with **four env vars**.
- You do not need to self-host the 975B weights. A free NVIDIA API key is enough to follow along.
- In our test, Inkling extracted structured facts cleanly, resolved a relative date on its own, and produced a coherent reflection, all out of the box.
- Total cost of the experiment: a few hundred tokens. Basically free.

## Wait, why would you use a frontier model for *memory*?

Because "memory" is not one job, it is two, and both are quietly hard.

When your agent finishes a turn, something has to read the raw conversation and decide what is actually worth remembering: which facts, about which entities, with what timestamps. That is the **retain** step, and it is a structured-extraction problem. Later, when the agent asks "what do we know about X," something has to reason across everything stored and synthesize an answer. That is **reflect**.

Hindsight handles the storage, search, entity resolution, and consolidation itself. But those two steps, extraction and reflection, call out to an LLM. Which LLM you choose determines how good your agent's memory actually is. So the real question is not "can Inkling chat," it is "can Inkling turn a sentence into structured, dated, entity-linked facts." Let us find out.

## Four environment variables. That is the whole integration.

Hindsight talks to any OpenAI-compatible endpoint, so wiring in Inkling is configuration, not code:

```bash
export HINDSIGHT_API_LLM_PROVIDER=openai
export HINDSIGHT_API_LLM_BASE_URL=https://integrate.api.nvidia.com/v1
export HINDSIGHT_API_LLM_API_KEY=nvapi-your-key
export HINDSIGHT_API_LLM_MODEL=thinkingmachines/inkling
```

Then start the server:

```bash
pip install hindsight-api
hindsight-api        # serves http://localhost:8888
```

That is it. Embeddings and reranking run locally by default, and the database is an embedded Postgres, so the only thing leaving your machine is the extraction and reflection calls to Inkling. On boot you will see Hindsight verify the connection:

```
Connection verified: openai/thinkingmachines/inkling
```

### Getting a key for free

Inkling is served on several providers, but the free one is NVIDIA's. Grab a key at [build.nvidia.com/thinkingmachines/inkling](https://build.nvidia.com/thinkingmachines/inkling), which comes with free credits. Together AI and Fireworks also host it with free signup credits if you prefer. You do not need the raw Hugging Face weights, and you almost certainly do not want to try to run a 975B model on your laptop.

One flag worth setting for a model like this:

```bash
export HINDSIGHT_API_LLM_STRICT_SCHEMA=false
```

This tells Hindsight to use its schema-in-prompt extraction mode rather than demanding native grammar-enforced JSON. More on why that matters below.

## The test: teach it one messy sentence

We threw Inkling a deliberately tangled sentence, the kind of thing a real agent hears all the time:

```bash
hindsight memory retain inkling-test \
  "Ben is a staff engineer at Vectorize in Toronto. He prefers Rust for
   performance-critical services and uses Postgres with pgvector. Last week
   he decided to drop the Kafka pipeline in favor of a simpler cron-based ingester."
```

Retain finished in about **8 seconds**. Then we asked what Hindsight had actually stored:

```bash
hindsight memory recall inkling-test "what tech stack and decisions does Ben prefer?"
```

```
[WORLD] Ben is a staff engineer at Vectorize in Toronto, prefers Rust for
        performance-critical services, and uses Postgres with pgvector.
        | Involving: Ben
[WORLD] Last week, Ben decided to drop the Kafka pipeline in favor of a
        simpler cron-based ingester. | When: on July 14, 2026 | Involving: Ben
```

## The part that made us do a double take

Look at the second fact again: **`When: on July 14, 2026`**.

We never gave Inkling a date. We wrote "last week." Inkling read the message timestamp, did the arithmetic, and pinned the decision to an absolute date, which is exactly what you want, because "last week" is useless to an agent three months from now. Temporal grounding is one of the things cheaper models routinely botch, and Inkling did it unprompted on the first try.

It also cleanly **split one run-on sentence into two facts** (a stable profile fact and a dated decision) and resolved eight entities, correctly recognizing that both facts were about the same Ben:

```
Total entities: 8   (Ben ×2, Vectorize, Toronto, Rust, Postgres, pgvector, Kafka, ...)
```

## Reflect: can it reason across what it stored?

Extraction is half the job. We asked Hindsight to synthesize:

```bash
hindsight memory reflect inkling-test \
  "Summarize Ben's engineering preferences and recent architectural decisions."
```

Inkling came back (in about 11 seconds) with a structured, accurate summary that used the resolved date and even editorialized correctly:

> On July 14, 2026, Ben decided to drop the Kafka pipeline in favor of a more straightforward cron-based ingester. This change reflects his inclination towards simplicity and efficiency in architectural choices.

Nobody told it Ben likes simplicity. It inferred that from a Kafka-to-cron swap. That is the kind of connective reasoning that makes a memory feel like it understands you rather than just storing you.

## Why this works out of the box

Two reasons, and one honest caveat.

First, Inkling is **post-trained**, not a raw base checkpoint. Thinking Machines trained it on chat, agentic code, and tool use, so it follows instructions and emits well-formed output without you begging. That is why it produced valid structured facts even in Hindsight's soft schema mode, where the JSON shape is described in the prompt rather than grammar-enforced by the API.

Second, the memory task plays to a big model's strengths. Extraction and temporal grounding reward broad world knowledge and careful reading, and reflection rewards synthesis. Inkling has all three.

The caveat: Inkling is **not on our [model leaderboard](https://benchmarks.hindsight.vectorize.io) yet**, and one tidy anecdote is not a benchmark. Our current top pick for the retain step is the much smaller `gpt-oss-20b`, which is faster and cheaper for high-volume memory writes. Inkling's 8-to-11-second calls are the tax you pay for a 975B reasoning model. So the honest framing is: Inkling is a *remarkably* capable memory brain, and probably overkill for cranking through thousands of retains, but genuinely impressive for quality-sensitive work.

## When you would actually reach for this

- **You are already building on Inkling** and want your agent's memory to speak the same model. One provider, one bill.
- **Quality over throughput.** Long, ambiguous, temporally messy inputs where you want extraction to be right, not just fast.
- **You want a fully swappable stack.** Open memory (Hindsight, MIT), an open model (Inkling, Apache 2.0), and the freedom to change either without a rewrite.

For firehose-volume memory writes, keep a small fast model on retain and save Inkling for reflect. Hindsight lets you point each at a different endpoint if you want.

## The five-minute version

1. Get a free key at [build.nvidia.com/thinkingmachines/inkling](https://build.nvidia.com/thinkingmachines/inkling).
2. Set four env vars (`PROVIDER=openai`, the NVIDIA `BASE_URL`, your key, `MODEL=thinkingmachines/inkling`), plus `STRICT_SCHEMA=false`.
3. `pip install hindsight-api` and run it.
4. `retain`, `recall`, `reflect`. Watch it turn "last week" into a real date.

Mira Murati's team built Inkling as a base for fine-tuning. It turns out it is also a shockingly good memory engine with zero fine-tuning at all. Give your agent a brain it can actually remember with.

## Further reading

- [The fully open agent memory stack](/blog/2026/07/17/hermes-hindsight-open-stack): pairing an open model with open memory, end to end.
- [Inside retain()](/blog/2026/07/13/inside-retain-agent-memory): what Hindsight does with each fact the model extracts.
- [Hindsight model leaderboard](https://benchmarks.hindsight.vectorize.io): which LLMs score best on the memory task.
