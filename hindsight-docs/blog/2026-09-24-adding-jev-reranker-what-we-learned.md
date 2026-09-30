---
title: "We Added Jev as a Reranker. Here's What We Learned."
authors: [benfrank241]
slug: "2026/09/24/adding-jev-reranker-what-we-learned"
date: 2026-09-24T16:00
tags: [hindsight, agent-memory, recall, reranking, jev, typesafe, retrieval, deep-dive]
description: "Six things we learned adding Jev, a model that returns typed decisions instead of text, as a reranker in Hindsight 0.10.1. Including two designs that failed and the numbers that killed them."
image: /img/blog/jev-reranking.png
hide_table_of_contents: true
---

![What we learned adding Jev as a reranker in Hindsight 0.10.1](/img/blog/jev-reranking.png)

Hindsight 0.10.1 adds [Jev](https://typesafe.ai) as a reranker provider. Jev is an unusual model: it produces no text at all. You send it a passage and a typed question, and you get back a typed answer with a calibrated probability. TypeSafe calls this class of thing a System One model, after Kahneman's fast, automatic kind of thinking.

That shape is a good fit for reranking, which is not a writing task. There is a pile of candidate memories and a question about which ones matter, and the only thing you want back is an ordering.

Getting there took longer than we expected, and two of the designs we tried failed outright. Here is what we learned.

<!-- truncate -->

## TL;DR

- **The obvious way to use it was the wrong one.** We ask one question about the whole pool rather than one question per candidate.
- **Two designs failed with numbers attached**, both about giving the model a way to say "nothing here."
- **Ranking and filtering are different product decisions**, so we ship one on and one off.
- **The scores are positions, not confidences**, which has consequences downstream.
- **It measured well**: recall@1 0.800 to 0.950 against our default at 30 candidates, and 0.12 to 0.027 seconds per query.
- **It is a hosted third-party API**, which disqualifies it for some deployments no matter how good the numbers are.

## 1. The obvious mapping was the wrong one

Jev gives you a small vocabulary of question types. Choice picks among unordered options and returns a probability for each. Score rates against ordered levels. Noul returns the probability that a yes/no proposition holds.

For reranking, Noul is the natural fit: ask "is this candidate relevant to this query?" for each one and sort by the probability. It is clean, it is what the primitive looks designed for, and at Jev's prices it is affordable in a way per-candidate LLM scoring never was. It is also, as far as we can tell, the pattern TypeSafe's own reranking material suggests.

We built that first, and then we stopped, because it is still pairwise. Three hundred candidates means three hundred round trips and three hundred judgements that never see each other. A cheap pairwise reranker is still a pairwise reranker.

So we used Choice instead, with the entire pool as options. One request, however many candidates, and the probabilities that come back are the ranking. The design note we left in the code:

> A Choice answers with a probability for every option, summing to 1, so handing it the whole pool returns the ranking in a single call. That beats scoring each candidate on its own: judged together the model only has to say which candidate beats which, instead of pinning every candidate to an absolute scale it must re-derive each time.

That is the lesson, and it generalises past this model. Relative judgement is an easier task than absolute judgement. Asking "which of these is best" avoids a question that scoring forces on you every single time: what does 0.6 actually mean?

We measured the two shapes against each other on the same model. On a 200-question LoCoMo set, listwise scored recall@1 0.94 against 0.87 for one call per candidate, at a thirtieth of the calls.

## 2. "None of these" does not work as an option

Ranking is only half of what you want. The other half is knowing where relevance stops, so you are not handing an agent three hundred memories when four were relevant.

The obvious way to express that with a Choice is to add a "none of these" option. If the model picks it, nothing is relevant. It is the first thing anyone would try, and we tried it.

It fails badly. Choice options are unrivalled alternatives, not points on a scale, so "none of these" is not competing with the candidates on relevance. It simply wins outright whenever the query is hard. **Thirty-five of two hundred questions came back completely empty.**

The fix was to stop using the wrong primitive. The cut is a Score, whose levels are *ordered*, which is what a cut point actually needs. It sees the ranked shortlist and answers how far down relevance extends, with levels in plain language: only the first, the first two, the first three, the first five, the first ten, all of them. The model picks one, and there is no threshold for us to tune.

## 3. Do not give it an escape hatch

Having built the Score, we made the same mistake one level down: we added a "nothing is relevant" level to it.

Same failure, smaller. It cost 7% of queries returning nothing, and dropped gold retention from 0.81 to 0.65.

So there is deliberately no such level. At least one candidate always survives, and no query comes back empty. The reasoning we settled on is that recall runs on a pool retrieval has already judged plausible, so one weak memory the caller can dismiss beats silence.

Twice in one feature, giving the model a clean way to answer "nothing" made it answer "nothing" far more often than the data justified. If we build another typed-decision integration, that is the thing we will look for first.

## 4. Ranking and filtering are different decisions

The cut works, and the precision numbers are dramatic. On the 30-candidate run it keeps 1.6 candidates out of 30 and lifts the precision of what survives from 0.051 to 0.850, a factor of seventeen.

It also cuts 19% of the gold evidence. On a real bank it took 300 candidates down to 3.

We went back and forth on the default and landed on off. Better ordering is something everyone wants. Dropping most of the pool is a product decision about what your agent is for. If the consumer is an LLM prompt and every irrelevant memory is wasted context, that trade is excellent. If a human is going to read the list, or your agent needs to find a needle that ranked fourteenth, it is not.

One limit we should have documented earlier: **the cut only ever sees the top twelve candidates, so with pruning on, recall returns at most twelve results regardless of pool size.** The shortlist is capped at twelve and the depth is clamped to it. That is consistent with "300 down to 3," but if you expected a large pool to yield a large relevant set, it will not.

## 5. Positions are not confidences

The scores Jev hands back through this path are ranks, normalised into a 0 to 1 range. The top candidate is 1.0 and each next one is 1/n lower.

A 0.7 therefore means "the best of these," not "relevant." Two different pools are not comparable, because each call's probabilities are normalised within that call. This is also why a pool larger than 250 is ranked in rounds and the round winners ranked against each other, rather than the rounds simply being concatenated.

The practical consequence is for anyone setting an absolute score floor on recall. For this provider, that floor filters on rank position rather than on relevance, which is almost certainly not what you meant. Worth knowing before you copy a threshold across from a different reranker.

## 6. What we would tell you before you switch

Four things that have nothing to do with quality.

**It is a hosted third-party API.** Jev runs at TypeSafe's endpoint and needs an API key. If your reason for self-hosting Hindsight is that memory contents must not leave your infrastructure, this sends candidate text to someone else's, and no benchmark makes that acceptable.

**It fails closed.** A reranker that keeps erroring propagates the failure after its retry budget. Configure a fallback chain with `rrf` last so a bad day degrades to fusion order rather than failing the recall outright. That matters more once there is a network hop in the path.

**It is server-level.** A bank cannot pick its own reranker, only turn reranking off. A multi-tenant deployment runs one for everybody.

**Jev's context is 32,000 tokens** for the state plus the questions, and this provider does not truncate candidate text before sending. A pool of unusually long memories is something to size up rather than assume.

## A thing we got right by accident

Worth recording the one decision that made this cheap, because we made it for an unrelated reason.

The retrieval arms move ids and scores, not payloads. Full memory text is fetched only for the candidates that survive fusion and the 300-candidate cap, and that hydration step sits immediately before reranking because the reranker is the first stage that actually reads text.

We ordered it that way to keep the wide parallel arms light. The payoff arrived later: swapping in a provider that sends candidate text over the network cost us nothing extra, because by that point the set had already been narrowed and fetched exactly once.

## The numbers

Measured on LoCoMo, gold defined as the dataset's own evidence turns, against the reranker Hindsight ships by default.

**30 candidates per query:**

| | recall@1 | recall@5 | NDCG@10 | s/query |
|---|---|---|---|---|
| `local` MiniLM (current default) | 0.800 | 0.876 | 0.850 | 0.12 |
| **Jev, ranking only (the default)** | **0.950** | **0.966** | **0.957** | **0.027** |

**240 candidates per query, the pool size a production recall actually reranks, 60 questions:**

| | recall@1 | recall@5 | NDCG@10 | s/query |
|---|---|---|---|---|
| `local` MiniLM | 0.583 | 0.719 | 0.682 | 0.41 |
| **Jev, ranking only** | **0.783** | **0.903** | **0.856** | **0.063** |

The gap widens with the pool, which is what you would expect when one side is making 240 sequential judgements and the other is making a single comparison.

## If you cannot use it

The default has not changed, and that is deliberate. `local` runs a MiniLM cross-encoder in-process, costs nothing, sends nothing anywhere, and is the 0.800 row above.

Beyond it: `tei` for a self-hosted Hugging Face inference endpoint, `flashrank` for something small and local, `jina-mlx` on Apple Silicon though its licence is non-commercial, and `rrf` to skip neural reranking and keep the fusion order.

None of them prune, because none of them produce a decision. They produce an ordering, and the lowest score in an ordering still means "least bad of these" rather than "not relevant." That distinction is the whole reason the cut exists here and nowhere else.

## FAQ

**Do I have to change anything?**
No. Jev is opt-in through the reranker provider setting, and the default is unchanged.

**Is Jev a Vectorize model?**
No. TypeSafe is a separate company and Jev is their model. Hindsight supports it the way it supports Cohere, TEI and the rest.

**Does it cost more than the local reranker?**
The local one is free, so yes in absolute terms. But the listwise shape uses roughly a thirtieth of the calls of per-candidate scoring, and Jev charges for input only. Check TypeSafe's pricing against your recall volume.

**Should I turn on pruning?**
Only when the consumer is an LLM prompt and every irrelevant memory is wasted context, and only after trying it on your own data. The two costs are 19% of gold evidence and a ceiling of twelve results.

**What happens if TypeSafe is down?**
Without a fallback chain, the recall fails after its retries. Configure indexed fallback members with `rrf` last and it degrades to fusion order.

## Learn more

- [Cross-Encoder Reranking: The Last Stage of Agent Memory Recall](https://hindsight.vectorize.io/blog/2026/08/28/cross-encoder-reranking-agent-memory) on the stage itself, normalization and failing open
- [Knowledge Graphs vs Vector Search](https://hindsight.vectorize.io/blog/2026/08/24/knowledge-graphs-vs-vector-search) on the retrieval arms that feed it
- [Bring the Facts, Not the Beliefs](https://hindsight.vectorize.io/blog/2026/09/23/bring-facts-not-beliefs) on the deduplication reranking does not do
- [What's new in Hindsight 0.10.1](https://hindsight.vectorize.io/blog/2026/09/21/version-0-10-1) for the rest of the release
