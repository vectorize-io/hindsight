---
title: "What We Got Wrong About Agent Memory"
authors: [benfrank241]
slug: "2026/09/04/what-we-got-wrong-agent-memory"
date: 2026-09-04T12:00
tags: [agent-memory, engineering, postmortem, retain, recall, hindsight]
description: "Six decisions we made building Hindsight and later reversed: a misdiagnosed memory profile, a published speedup that wasn't real, a workaround we deleted, and three places we made failure too quiet."
image: /img/blog/what-we-got-wrong-agent-memory.png
hide_table_of_contents: true
---

![What we got wrong about agent memory: six reversals from building Hindsight](/img/blog/what-we-got-wrong-agent-memory.png)

Most engineering writing describes the version that worked. That's the least interesting part, because by the time something works it looks obvious, and the reasoning that made the wrong version attractive has been quietly deleted.

So here are six things we got wrong building Hindsight, each of which shipped, and each of which we later undid. Every one is in the public commit history, and every one seemed reasonable when we did it. That's the part worth keeping.

<!-- truncate -->

## 1. We optimized the thing that looked expensive

Retaining a large document used to consume memory proportional to the document rather than to the work being done. Feed it a 90 MB body and peak allocation climbed with it.

The obvious culprit was embeddings. Extracted facts carry vectors, vectors are `list[float]`, and Python boxes every float. It's a genuinely expensive representation and it was right there.

It wasn't the problem. When we measured instead of guessing, **two whole-document operations dominated** — and both ran *before* a single fact or embedding existed. The balloon was inflating before the suspect entered the room.

The fix was windowed token sizing, streamed chunking, and a streamed sub-batch split, none of which touch embeddings at all. Peak memory is now flat from a 4 MB body to a 90 MB one.

The lesson isn't "profile your code," which everyone already agrees with and then skips. It's that a *plausible* explanation is more dangerous than no explanation. Nobody profiles when they think they already know.

## 2. We published a speedup we hadn't earned

We found that every remote embedding provider walked its batches in a plain loop, holding exactly one embedding request open at a time regardless of how much text a retain had. That's a real ceiling.

We attributed an observed retain throughput of ~674 texts/s to that serialization and predicted a **3.1x end-to-end improvement**.

That attribution was wrong. The retain bottleneck was somewhere else, and removing the ceiling didn't move the end-to-end number the way we'd said it would.

The follow-up claims no end-to-end figure at all. It reports what we could actually measure in isolation — the same server sustains 903 texts/s at one in-flight request and 2,080 at eight — and describes the change as removing a client-side ceiling that will bind *once whatever is currently in front of it moves*.

Finding a bottleneck and removing a bottleneck are two separate claims, and only one of them was supported. We'd merged them into a single number because the story was clean.

## 3. We engineered a careful workaround for a problem we could have deleted

Counting tokens on a large retain body used to allocate memory proportional to the body. So we wrote `count_tokens_windowed`: a windowed counter that bounded the memory cost, with helper functions, and six call sites across the pipeline. It worked.

Then we replaced the tokenizer. The new one allocates essentially nothing at any input size, **and** returns an exact count rather than a windowed approximation.

So the windowing went away. The helpers went away. All six call sites now get an exact count for less memory than the approximation used to cost.

The workaround was good engineering aimed one level too low. We'd accepted the tokenizer as fixed and built around it, and the question "should this component be here at all?" never came up because the workaround was working.

## 4. We made failure quiet, in three different places

These landed separately and are obviously one mistake in hindsight:

- **Config defaults resolved silently.** Missing configuration fell back through `getattr` defaults, so a *wrong* config ran with plausible values instead of complaining. Now a bad config fails loudly.
- **Memory Defense skipped screening on a malformed policy.** A content-inspection control that quietly does nothing when its policy won't parse is not a control. Now a malformed policy fails the retain.
- **Reflect stored a placeholder when a run produced no answer.** Rather than surfacing the failure, it wrote something.

That third one is the one that should have been obvious immediately, because of what this system *is*. In a stateless service, a placeholder response is one bad request. In a memory system, **a placeholder is a wrong fact that persists and gets recalled** — the failure doesn't end when the request does. It gets retrieved next week and treated as knowledge.

Every one of these was written defensively. Keep going, don't break the caller, degrade gracefully. That reflex is correct in a request-response service and actively harmful in something that remembers, because the artifacts of graceful degradation are permanent.

## 5. We treated every source as the same kind of claim

On a repository worked by a coding agent, two very different things flow into memory: commit diffs and session transcripts.

A diff records what the code *does*. A transcript records what someone *intended*, argued for, or explicitly rejected. We consolidated both into one undifferentiated belief set.

The result is exactly what you'd expect once it's said out loud. An idea floated in chat and never implemented becomes indistinguishable from a belief derived from the committed code. And the question "what does this codebase actually do" can no longer be answered from commit-derived knowledge alone, because chat speculation is sitting in the same pile.

The uncomfortable part was that this couldn't be fixed by configuration. The consolidator treated an explicit scope list as unconditional, so configuring three scopes wrote every document into all three. The design had to change, not a setting.

Provenance isn't decoration on a memory. It's part of what the memory *claims*.

## 6. We inferred a mode instead of being told it

Recall applies recency and other signals as multiplicative boosts on top of a reranker's score. Some deployments run a passthrough instead of a real reranker, and in that mode every candidate gets an identical score — which would make those boosts the only ranking signal and silently turn recall into a date sort.

We needed to detect that case. The first approach was to notice that all the scores were identical, which is elegant and requires no plumbing.

It's also wrong. A real cross-encoder can legitimately tie scores, especially on small or synthetic result sets, and inferring "passthrough" from a tie would corrupt a genuine rerank.

Now the caller passes an explicit flag. It's less elegant and it's correct. A cheap signal that's *usually* right is a bad trade when being wrong is silent.

## What these have in common

Reading them together, they're mostly two mistakes wearing different clothes.

**Three were believing a plausible story instead of measuring one.** The embedding suspect, the 3.1x prediction, the identical-scores heuristic. In each case there was a clean explanation available, and its cleanliness was doing the persuading.

**Three were making failure invisible.** Silent config defaults, skipped screening, stored placeholders. All written defensively, all of them the right instinct in a stateless service and the wrong one in a system whose entire job is to keep things.

That second pattern is the one we'd flag for anyone building in this space. Memory systems invert a lot of ordinary engineering intuition, because your mistakes don't get garbage collected at the end of the request. They get retrieved.

## Learn more

- [How We Made Retain's Peak Memory Flat](/blog/2026/08/27/retain-memory-budget) — the misdiagnosis in full, with the measurements
- [Cross-Encoder Reranking: The Last Stage of Agent Memory Recall](/blog/2026/08/28/cross-encoder-reranking-agent-memory) — where the passthrough problem lives
- [The Consolidation Problem in Agent Memory](/blog/2026/05/21/agent-memory-consolidation) — why sources and beliefs need separating
