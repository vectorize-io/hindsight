---
title: "Hermes Writes Its Own Skills. Memory Makes Them Compound."
authors: [benfrank241]
slug: "2026/07/08/hermes-self-improving-memory"
date: 2026-07-08T12:00
tags: [hermes, agents, memory, hindsight, self-improving, agent-memory]
description: "Hermes self-improves by writing its own skills. But skills only compound if the agent remembers them and the context that made them matter. That is where memory comes in."
image: /img/blog/hermes-self-improving-loop.png
hide_table_of_contents: true
---

![Hermes writes its own skills, and memory makes them compound](/img/blog/hermes-self-improving-loop.png)

[Hermes Agent](https://github.com/NousResearch/hermes-agent) is the rare agent that actually self-improves, and it does it in a concrete way: it writes its own skills. Finish a non-trivial, multi-step task and Hermes can save the approach it worked out as a reusable skill, patch that skill later when it turns out to be wrong, and keep a growing library of capabilities it authored itself. That is a real learning loop, not a slogan.

But skill creation is only half of what makes an agent get better over time. A skill is a file on disk. For it to actually improve the agent, Hermes has to remember the skill exists, know when it applies, and carry the project and user context that made it matter in the first place. That second half is memory, and it is what turns a pile of self-authored skills into an agent that compounds.

<!-- truncate -->

## TL;DR

- Hermes's self-improvement is genuine and specific: it [creates its own skills](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) from experience, patches them when they are outdated or wrong, and can even evolve them.
- A skill only helps if the agent remembers it exists, when to use it, and the context that motivated it. That is memory, not skills.
- Hindsight is Hermes's [native memory provider](/blog/2026/04/06/hermes-native-memory-provider): retain after each turn, consolidate into durable observations, recall the relevant ones before the next.
- Skills give Hermes new capabilities. Memory makes them compound across sessions and machines instead of resetting.

## The engine: Hermes writes its own skills

This is the part that makes Hermes genuinely self-improving. When it works out a non-trivial workflow, it can save that approach as a reusable [skill](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills), so next time the same kind of task comes up it reaches for what it already figured out instead of solving it from scratch. Skills are not frozen either. When one turns out to be incomplete or wrong in practice, Hermes patches it. There is even a `/learn` command that captures a workflow you just did and writes the skill document for you, and an optional [self-evolution](https://github.com/NousResearch/hermes-agent-self-evolution) layer that uses evolutionary search (DSPy and GEPA) to optimize skills, tool descriptions, and prompts.

The result is an agent whose set of capabilities grows with use. That is the headline, and it is why Hermes calls itself the agent that grows with you.

## The missing half: a skill the agent forgets is a skill it never wrote

Now the catch. Skill creation makes Hermes more capable, but capability is not the same as continuity. Skills live as files, and a file does nothing on its own. For a self-authored skill to actually make the agent better, three things have to happen every time you come back:

- **It has to remember the skill is relevant here.** A library of skills is only useful if the right one surfaces at the right moment, against what you are actually asking.
- **It has to carry the context that made the skill matter.** The workflow you saved last week assumed things about your project, your stack, and your conventions. Apply it blind to that context and you get a generic run, not a better one.
- **It has to keep its model of you.** Part of what Hermes improves is its understanding of who you are and how you work. That understanding has to persist across sessions, or it resets to zero each morning.

Hermes even nudges itself to persist knowledge, which tells you persistence is the point. But persistence is exactly what a local pile of files does not give you: no structure, no retrieval intelligence, no cross-machine sync, and no sense of which context a given skill or fact belongs to. Without a memory layer, the agent can author a hundred skills and still start every session cold, re-deriving a workflow it already saved or applying one without knowing your conventions.

## The other half: the memory loop

This is where Hindsight comes in, as Hermes's native memory provider. It closes the loop around the skills, hooking into two points in the Hermes lifecycle so it runs on its own.

**After each response, Hermes retains the exchange.** Asynchronously, Hindsight extracts the facts, entities, and relationships from what just happened, including the decisions and context around the work the agent did. It captures what was learned, not only what got written into a skill file.

**Between turns, those facts are consolidated.** Hindsight merges related facts about the same entity, resolves that "Alice" and "my coworker Alice" are the same person, and distills history into durable [observations](/blog/2026/05/21/agent-memory-consolidation) rather than an ever-growing pile of notes. That consolidated view is exactly the context a self-authored skill needs to be applied well: who you are, what the project is, and how you work.

**Before each turn, Hermes recalls what is relevant.** A prefetch pulls the memories that matter for your current message and injects them into the system prompt before the model runs, so the agent starts already knowing your project and preferences, and can bring the right learned behavior to bear. One honest detail: the prefetch is queued for speed, so a fact from this turn is available on the next call, which keeps every response fast.

## Why the two halves compound together

Skills and memory improve different things, and that is the point. Skills expand what Hermes can do. Memory decides when and how to do it, in your context, and makes both persist.

Put them together and the improvement compounds instead of plateauing. Hermes writes a skill from a hard task; memory remembers the skill and the situation it fit; next time, it recalls that context and applies the right capability in your project's terms. Two things keep the memory half pointing the right way as it grows: consolidation keeps the store distilled rather than noisy, and retrieval holds up at scale. Of the memory providers Hermes supports, Hindsight is the one with published results on [BEAM](/blog/2026/04/02/beam-sota), the benchmark that tests memory at 10 million tokens where context stuffing is impossible. Hindsight scores 64.1% there; the next-best published result is 40.6%.

## What it looks like in practice

- **Self-authored skills get reused with context, not blind.** Hermes reaches for the workflow it saved last week and applies it in your project's terms.
- **Corrections and conventions stick.** Tell it once how this repo does things and it stops re-suggesting the other way, next session and the one after.
- **It stops re-deriving what it already knows.** Fewer from-scratch reruns of a task it already turned into a skill.
- **Context survives time and machines.** Plan on Monday, resume Friday on another laptop, and the project is already there.

## Turning on memory

Skill creation is built into Hermes. Adding the memory half is one wizard command:

```bash
hermes memory setup    # select "hindsight"
hermes memory status   # confirm it is active
```

Use [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) for zero infrastructure, or point it at a self-hosted server. For the full walkthrough, see [Hindsight Is Now a Native Memory Provider in Hermes Agent](/blog/2026/04/06/hermes-native-memory-provider).

## Frequently asked questions

**Does memory replace Hermes's skills?**
No. They are two halves. Skills are the capabilities Hermes authors for itself; memory is the persistent context and user model that decides when and how to use them.

**What actually makes Hermes self-improving?**
Autonomous skill creation: it saves workflows as reusable skills, patches them when they are wrong, and can evolve them. Memory is what makes that improvement compound across sessions rather than reset.

**Will a big memory slow the agent down?**
No. Retain runs asynchronously after the response, recall is a single prefetch before the turn, and consolidation keeps the store distilled rather than ever-growing.

**Why not just use a bigger context window?**
Context is rented per session and capped. Memory is durable, selective, and cross-session. BEAM tests exactly the regime where context stuffing fails, and that is where a memory system earns its keep.

## Further reading

- [Hermes skills system](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills): how Hermes authors and patches its own skills.
- [Hindsight is now a native memory provider in Hermes](/blog/2026/04/06/hermes-native-memory-provider): the setup guide.
- [Agent memory consolidation](/blog/2026/05/21/agent-memory-consolidation): how facts become durable observations.
