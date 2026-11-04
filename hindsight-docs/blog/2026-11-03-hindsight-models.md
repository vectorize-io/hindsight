---
title: "Introducing Hindsight Models: memory 2.5x cheaper and 2x faster, with better answers"
description: "The LLM writes, the decision model judges. hs-decide-1-flash takes over the small judgments inside Hindsight: better answers from six of seven LLMs, cheaper for five. With a reranker and an embedding model built for agent memory, open weights and hosted."
authors: [nicoloboschi, andrewneeser, cdbartholomew, benfrank241]
date: 2026-11-03
slug: "2026/11/03/hindsight-models"
tags: [release, inference, models]
image: /img/blog/hindsight-models/cover.png
hide_table_of_contents: false
---

import ThemedImage from '@theme/ThemedImage';
import YouTubeVideo from '@site/src/components/YouTubeVideo';

**The LLM writes, the decision model judges.** Most of what an AI application asks an LLM isn't writing. It's small judgments: is this new, which of these matters, is this enough. Today we're releasing three models built for agent memory, starting with `hs-decide-1-flash`, a decision model that answers those judgments directly, so the LLM is left with the writing.

<div className="post-wide">

<a href="https://benchmarks.hindsight.vectorize.io/leaderboards/all-operations"><ThemedImage alt={"Quality against cost for every LLM on all of Hindsight; an arrow joins each LLM to the same LLM with hs-decide-1-flash, which sits to the left (cheaper) and mostly higher (better)"} sources={{light: "/img/blog/hindsight-models/chart-all-operations-light.png", dark: "/img/blog/hindsight-models/chart-all-operations-dark.png"}} /></a>

</div>

## TL;DR

- **Better and cheaper.** With `hs-decide-1-flash` making the small judgments, six of seven LLMs give better answers across Hindsight, and five get cheaper. On `gemini-3.8-flash`, the operations the decision model takes part in cost 2.5x less, spend 1.9x less time in model calls and score 2.2 points higher.
- **New: the Hindsight Inference API, and open weights.** All three models are hosted behind one API, from $0.005 per million input tokens, that works with Cohere and OpenAI clients; new accounts get $5 of free credit. **[Sign up for Hindsight Inference →](https://inference.hindsight.vectorize.io)** ([docs](https://docs.hindsight.vectorize.io/cloud/inference/overview/)) The weights are on [Hugging Face](https://huggingface.co/hindsight-ai), the decision model under Apache 2.0 and the other two under MIT.
- **A cheap LLM near a frontier one.** `glm-5.3-flash` with the decision model scores within 2 points of `gemini-3.8-flash` alone, at 19x less cost.
- **167 ms per decision-model request** at the median; an LLM call takes 1 to 7 seconds.
- **Leaner recall.** Recall can drop irrelevant and repeated memories before they reach your agent, so your agent's own LLM reads fewer tokens.
- **A reranker and an embedding model built for agent memory.** `hs-rerank-1` and `hs-embed-1` score highest overall against the small open models people run today, in eight languages, and run on a CPU.

<!-- truncate -->

<YouTubeVideo id="NqipFrDSZgU" title="Introducing Hindsight Models: Better Agent Memory, for Less" poster="/img/blog/hindsight-models/video-poster.jpg" duration="1:17" />

## Judgments are most of the work

A memory system is full of judgments. Between storing a memory and recalling it, Hindsight asks: is this fact new, or already known? Does it change what we believe? Which of these memories matter for this question? Is this evidence enough to answer? Could this new memory change that knowledge page?

Hindsight used to ask an LLM every time. Each answer was a full generation call: seconds of latency, a bill for every token of context, and free text to parse back into a yes or no. On our evals, with a decision model answering the judgments, the LLM makes about a third fewer calls: 3.1 per case instead of 4.5.

A decision model is built for exactly this shape of question. You give it the state and the question, and it returns a typed answer: a choice with a probability for each option, a point on a scale, or how likely something is true. It doesn't generate text, so it answers in a fraction of the time and price.

## Where it started

We first tried this with [Jev](https://typesafe.ai), TypeSafe's decision model, and saw how much of Hindsight's work had never needed an LLM. So we trained our own on the judgments Hindsight makes: 0.8B parameters, open weights, and the same System One requests, so an app written for one runs on the other by changing the model name.

## How Hindsight uses it

Hindsight asks the decision model at four points. Without one, each runs exactly as before.

**Recall: a smaller bill for your agent.** Every memory recall returns ends up in your agent's prompt, and for most teams the agent's own LLM is the biggest cost they have. Recall used to fill its token budget whether the memories helped or not. With a decision model, it can hand your agent only what answers the question:
- **Prune** drops the memories that are no help to the query. It reuses the ranking answers, so it costs no extra request, and the best memory always stays.
- **Dedup** drops a memory that repeats a better-ranked one, so a fact retained from several documents reaches the agent once. It costs one more decision-model request, asked only about memories whose embeddings are alike; memories dated differently, or of two kinds, are never called repeats.

Fewer, more relevant memories means fewer tokens on every agent turn, and less noise for the agent to reason past. Both are off by default; turn them on per request (`"rerank": {"prune": true, "dedup": true}`) or per bank. Reflect and page refreshes search through recall, so they get it too. ([Recall docs](/developer/api/recall#rerank))

**Reflect: is this evidence enough?** Reflect answers a question by searching memory in rounds. Before, an LLM agent ran each round: it picked a search tool, wrote a query, read every result and decided whether to search again. With a decision model:
1. One LLM call turns the request into a search query, or finds there's nothing to search and returns at once.
2. The searches run in parallel with that query.
3. The decision model **trims the evidence**: it ranks what came back against the request and cuts what doesn't bear on it, so the LLM reads less.
4. It **judges whether the evidence already answers the request**. If yes, the next LLM call writes the answer. If not, the LLM writes the next query and another round runs.
5. When the bank has knowledge pages, it first asks **whether the pages alone answer it**, and skips the raw search when they do.

If the decision model fails, reflect falls back to agent turns.

**Consolidation: is this fact new?** Consolidation turns new facts into observations, the bank's current beliefs. Before, every new fact went to the LLM with the observations it might touch. With a decision model, it settles the easy facts first, one question per fact: one that **restates** an observation is attached as evidence and **chatter** is skipped, neither with an LLM call. Only the facts that change what an observation says, add something new, or that it isn't sure about, go to the LLM, which writes every observation's text.

**Knowledge pages: could this change the page?** Rebuilding a page runs reflect, so it gets everything above. Updating a page with what's new is where the saving is. Before, every update ran a reflect over the new memories and rewrote the page. With a decision model, it asks of each new memory *could this change the page?* and edits the page from those alone; when none can, the update makes no LLM call at all.

## Results

Every number here is on [benchmarks.hindsight.vectorize.io](https://benchmarks.hindsight.vectorize.io): Hindsight's own system evals, run on Hindsight 1.2.0 against a fresh server through the public API. Each LLM runs alone and with `hs-decide-1-flash` on the same cases, all graded by the same judge, `gemini-3.1-flash-lite` (also one of the seven LLMs tested, so its own rows are self-graded).

- **Quality** is the share of cases passed, each graded by the judge; for all of Hindsight, the mean over operations.
- **Cost** is the tokens really used at list prices, decision-model tokens included at $0.02 per million, per thousand cases. "All of Hindsight" is an equal mix of the eval's operations (retain, consolidation, reflect, knowledge pages); a real workload weighs them differently.
- **Time** is the time spent in model calls (LLM and decision model) per operation, from the same runs' traces, not end-to-end latency.

Each chart plots quality against cost, one point per LLM; an arrow joins an LLM to the same LLM with the decision model. Up and to the left is better.

### All of Hindsight

<div className="post-wide">

<a href="https://benchmarks.hindsight.vectorize.io/leaderboards/all-operations"><ThemedImage alt={"All of Hindsight: quality against cost for each LLM, with arrows to the same LLM with hs-decide-1-flash"} sources={{light: "/img/blog/hindsight-models/chart-all-operations-light.png", dark: "/img/blog/hindsight-models/chart-all-operations-dark.png"}} /></a>

</div>

Quality goes up for six of the seven LLMs, most for the ones that need it most (`glm-5.3-flash` 73.8 → 87.1, `gemma-4-26b-a4b` 76.6 → 81.1, `deepseek-v4.1-flash` 81.9 → 85.1), and holds level for `gemini-3.1-flash-lite` (87.1 → 86.6). The LLM makes about a third fewer calls and reads half the input tokens or less, so five of them get cheaper: `gemini-3.8-flash` $147 → $61 (2.4x less), `gemini-3.1-flash-lite` $42 → $22, `claude-haiku-5.5` $31 → $18, `gpt-6-luna` $16 → $9.88, `gemma-4-26b-a4b` $16 → $12. Time in model calls drops for all seven, from 1.2x (`gemini-3.1-flash-lite`) to 1.9x (`gemini-3.8-flash`).

Retain doesn't use the decision model, so it costs the same either way and waters down the totals above. Leave it out and count only consolidation, reflect and knowledge pages, the operations the decision model takes part in: on `gemini-3.8-flash` they cost 2.5x less, spend 1.9x less time in model calls, and score 2.2 points higher.

**A cheap LLM gets close to a frontier setup.** `glm-5.3-flash` with the decision model scores 87.1 at $7.72 per thousand cases, and `gpt-6-luna` with it scores 87.0 at $9.88. `gemini-3.8-flash` alone scores 89.0 at $147: about 2 points more, at 15–19x the cost. Switching to a cheaper LLM and adding the decision model is the biggest saving in these results.

**Faster calls.** A decision-model request takes 167 ms at the median (350 ms at p90) and answers every question about a fact or a memory at once; the LLM call it replaces takes 1 to 7 seconds at the median, depending on the model.

**The decision model is a small part of the bill for most LLMs.** Its own tokens cost $4 to $7 per thousand cases: $3.93 of the $61 on `gemini-3.8-flash`. On the cheapest LLMs that's most of the bill, which is why they cost more with it (below).

### The trade-offs

The decision model makes Hindsight cheaper and faster almost everywhere, and better in most places, but not all:

- **Reflect** gives the biggest saving and is where quality dips. On `gemini-3.8-flash`, reflect costs 4.6x less, and quality goes from 94.1 to 91.2. Four of the seven LLMs dip by about 3 points, and `gemini-3.1-flash-lite` drops from 94.1 to 85.3.
- **The two cheapest LLMs get much better answers, for a little more money.** `glm-5.3-flash` goes from 73.8 to 87.1 across Hindsight ($4.58 → $7.72 per thousand cases), and from 36 to 84 on consolidation. `deepseek-v4.1-flash` goes from 81.9 to 85.1 ($10.57 → $12.21), and from 68 to 86 on consolidation. They cost more because the LLM is so cheap that the decision model's own tokens outweigh what it saves.

### Reflect

<div className="post-wide">

<a href="https://benchmarks.hindsight.vectorize.io/leaderboards/reflect"><ThemedImage alt={"Reflect: quality against cost for each LLM, with arrows to the same LLM with hs-decide-1-flash"} sources={{light: "/img/blog/hindsight-models/chart-reflect-light.png", dark: "/img/blog/hindsight-models/chart-reflect-dark.png"}} /></a>

</div>

Five of the seven arrows point left. The LLM takes about half the turns (4.4 → 2.5 per question) and reads a fraction of the evidence, because the decision model trims it first: `gemini-3.8-flash` reads 7.5K input tokens per question instead of 45K. Cost drops 4.6x on `gemini-3.8-flash` ($36 → $7.91 per thousand questions), 2.9x on `gemini-3.1-flash-lite`, 2.2x on `claude-haiku-5.5` and 2x on `gpt-6-luna`; time in model calls drops 1.1x to 2.1x. Quality rises for `glm-5.3-flash` (91.2 → 94.1), holds for `gpt-6-luna` (94.1), dips about 3 points for four LLMs, and drops from 94.1 to 85.3 for `gemini-3.1-flash-lite`.

### Knowledge pages

<div className="post-wide">

<a href="https://benchmarks.hindsight.vectorize.io/leaderboards/refresh-pages"><ThemedImage alt={"Refresh pages: quality against cost for each LLM, with arrows to the same LLM with hs-decide-1-flash"} sources={{light: "/img/blog/hindsight-models/chart-refresh-pages-light.png", dark: "/img/blog/hindsight-models/chart-refresh-pages-dark.png"}} /></a>

</div>

Here the arrows point up and left at once. `gpt-6-luna` goes from 82.5 to 90.0 at less than half the cost ($7.33 → $3.16), `gemini-3.1-flash-lite` from 85.0 to 90.0 at a third ($17 → $5.78), `gemini-3.8-flash` from 82.5 to 90.0 at 3.6x less ($65 → $18). Five of the seven LLMs keep pages more accurately, and the LLM makes half the calls (about 8.5 → 4 per case), since a memory that can't change the page never reaches it.

### Consolidation

<div className="post-wide">

<a href="https://benchmarks.hindsight.vectorize.io/leaderboards/consolidation"><ThemedImage alt={"Consolidation: quality against cost for each LLM, with arrows to the same LLM with hs-decide-1-flash"} sources={{light: "/img/blog/hindsight-models/chart-consolidation-light.png", dark: "/img/blog/hindsight-models/chart-consolidation-dark.png"}} /></a>

</div>

Every arrow points up: all seven LLMs keep observations more accurately with the decision model. The biggest jump is `glm-5.3-flash`, from 36 to 84. Alone, its calls all succeed, but it rarely updates the observations: most of its failed checks are a new fact that no observation cites. With the decision model settling the restatements and chatter first, the LLM only gets the few facts that need rewriting, and it handles those. `deepseek-v4.1-flash` goes from 68 to 86 and `gemma-4-26b-a4b` from 66 to 82. Here the gain is quality and time rather than money: the LLM makes about 20% fewer calls, consolidation spends up to 2.3x less time in model calls, and cost stays about level.

"Is this new, or already known?" is a question your own app may ask too. Outside Hindsight, it's one call to the [Decisions API](#hindsight-inference).

## Use them with Hindsight

**Every result above, in your own Hindsight: one login and one environment variable, no GPU.** In open-source Hindsight 1.2 the decision model is off until you set one up. With the `hindsight` provider it runs consolidation, reflect and knowledge pages, and recall keeps its reranker unless you turn it on for recall too. New accounts start with $5 of free credit, enough for the decision model's part of about 4,000 reflects on our evals.

```bash
hindsight-admin cloud login     # signs in to Hindsight Cloud and saves a key
export HINDSIGHT_API_DECISION_MODEL_PROVIDER=hindsight
export HINDSIGHT_API_RECALL_DECISION_MODEL_PROVIDER=hindsight   # optional: rank recall with it too
export HINDSIGHT_API_RECALL_DECISION_MODEL_PRUNE=true          # optional: drop results that don't help
export HINDSIGHT_API_RECALL_DECISION_MODEL_DEDUP=true          # optional: drop repeats
export HINDSIGHT_API_RERANKER_PROVIDER=hindsight
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=hindsight
```

The first two lines are all the results above need; the rest are optional. If you'd rather run everything yourself, the reranker and the embedding model run on your own machine, inside the Hindsight API process:

```bash
export HINDSIGHT_API_RERANKER_PROVIDER=local
export HINDSIGHT_API_RERANKER_LOCAL_MODEL=hindsight-ai/hs-rerank-1
export HINDSIGHT_API_EMBEDDINGS_PROVIDER=local
export HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL=hindsight-ai/hs-embed-1
```

A bank's stored vectors must be re-embedded to switch embedding models. To serve the decision model yourself on vLLM, see [Run It Yourself](/developer/decision-models#run-it-yourself); for every setting, see [Decision Models](/developer/decision-models) and [Models](/developer/models).

## Open weights on Hugging Face

All three models are in the [`hindsight-ai`](https://huggingface.co/hindsight-ai) organization on Hugging Face, and all three work in English, Spanish, French, German, Portuguese, Japanese, Korean and Chinese.

| Model | What it does | Size | Built on | License | Context |
|---|---|---|---|---|---|
| [`hs-decide-1-flash`](https://huggingface.co/hindsight-ai/hs-decide-1-flash) | Typed judgments: choice, score, how likely | 0.8B | `Qwen/Qwen3.5-0.8B` | Apache 2.0 | 65,536 tokens |
| [`hs-rerank-1`](https://huggingface.co/hindsight-ai/hs-rerank-1) | Orders documents by relevance to a query | 114M | `jhu-clsp/mmBERT-small` | MIT | 8,192 tokens |
| [`hs-embed-1`](https://huggingface.co/hindsight-ai/hs-embed-1) | Turns text into 384-dimension vectors | 112M | `hotchpotch/bekko-embedding-v1-a25m` | MIT | 8,192 tokens |

The decision model answers every question about a request in one pass and serves on vLLM with the plugin in its repository's `serving/` folder. The reranker and the embedding model run 15M- and 13M-parameter transformers per pass (most of their weights are the vocabulary table), so they're fast on CPU; both ship an int8 ONNX export and serve on Text Embeddings Inference as is.

### How they compare

Full tables: [Hindsight Models benchmarks](https://benchmarks.hindsight.vectorize.io/models).

**Decision model.** We compared `hs-decide-1-flash` with the other open decision models of its size: `StartLux-Decision 0.8B Instant`, `d3-lite` (vLLM Semantic Router) and `decider-0.8b` (Mapika). It has the best mean over eight benchmarks (75.6, against 74.5, 73.4 and 66.6), and its lead is in the memory judgments Hindsight makes: telling whether a new memory changes a page (79.5, against 56.0–72.0) and skipping refreshes that wouldn't change anything (89.4, against 45.0–78.3). On general decision benchmarks it trails: the public Decision Index (36.1, against 53.8 for StartLux) and dialogue inference (75.8, against 83.0 for StartLux and 88.0 for `d3-lite`).

<div className="post-wide">

<a href="https://benchmarks.hindsight.vectorize.io/models/decision"><ThemedImage alt={"Decision model benchmarks: hs-decide-1-flash against other open decision models"} sources={{light: "/img/blog/hindsight-models/decision-light.png", dark: "/img/blog/hindsight-models/decision-dark.png"}} /></a>

</div>

**Reranker.** For a Hindsight user, it means better recall in eight languages out of the box, and the hosted version takes reranking off your API server's CPU. We compared it against open rerankers up to 278M parameters, on seven benchmarks: English (NanoBEIR, BRIGHT), multilingual (MIRACL, and a bank of 26 tasks in 8 languages) and memory (KnowMeBench, MemBench, ConvoMem). Scores are nDCG@10; speed is the median time to rerank 50 memories on two CPU threads.

<div className="post-wide">

<a href="https://benchmarks.hindsight.vectorize.io/models/reranker"><ThemedImage alt={"Reranker benchmarks: hs-rerank-1 against open rerankers up to 278M parameters"} sources={{light: "/img/blog/hindsight-models/reranker-light.png", dark: "/img/blog/hindsight-models/reranker-dark.png"}} /></a>

</div>

**Embeddings.** For a Hindsight user, it means memories found by meaning in eight languages, and over long conversations and documents where small English models fall off. We compared it against open embedding models up to 118M parameters, on twelve benchmarks: English and multilingual retrieval, long conversations (BEAM), long documents (LongEmbed) and memory retrieval (LMEB). Speed is the median time per query on two CPU threads. The image shows one benchmark per group; the mean covers all twelve, and the full table is one click away. The fair comparison is with the multilingual models; the English-only ones are listed because they’re common defaults.

<div className="post-wide">

<a href="https://benchmarks.hindsight.vectorize.io/models/embeddings"><ThemedImage alt={"Embedding benchmarks: hs-embed-1 against open embedding models up to 118M parameters, six of twelve benchmarks shown"} sources={{light: "/img/blog/hindsight-models/embeddings-light.png", dark: "/img/blog/hindsight-models/embeddings-dark.png"}} /></a>

</div>

## Hindsight Inference

Hindsight Inference serves the same three models to any application. It's part of Hindsight Cloud: sign in at [inference.hindsight.vectorize.io](https://inference.hindsight.vectorize.io) with your Cloud account (same organization, teams and credit balance), and new accounts start with $5 of free credit.

**Try every model in the Playground before writing any code.** Pick reranking, embeddings or a decision, and it comes prefilled with an example you can edit. Each run shows the answer, how long it took, the tokens and what it cost. Here the reranker, asked which company Mia joined as principal ML engineer, puts the right memory first, and with **Drop irrelevant documents** on it hands back two of the four instead of all of them, the same pruning recall uses:

<div className="post-wide">

<a href="https://inference.hindsight.vectorize.io/playground"><img alt="The Hindsight Inference Playground: a rerank of four memories with Drop irrelevant documents on returns the right memory first and drops two" src="/img/blog/hindsight-models/playground-rerank.png" /></a>

</div>

When it does what you want, the Playground shows the same request as code (curl, Python, the Cohere or OpenAI SDK, or a Hindsight config) to copy into your app; create a key under **API Keys** and you're done. Every API is a drop-in for one you may already use:

| API | Endpoint | Models | Drop-in for | Price per million input tokens |
|---|---|---|---|---|
| Decisions | `POST /v1/systemone` | `hs-decide-1-flash` | System One | $0.02 |
| Reranker | `POST /v1/rerank` | `hs-rerank-1`, or `hs-decide-1-flash` to also drop irrelevant documents | Cohere Rerank | $0.015 / $0.02 |
| Embeddings | `POST /v1/embeddings` | `hs-embed-1` | OpenAI Embeddings | $0.005 |

**Already on Cohere Rerank?** Point your client here and set the model to `hs-decide-1-flash`: same code, and it hands back only what's relevant.

None of the three generates text, so you pay for input tokens only. Point your Cohere client at `https://inference.hindsight.vectorize.io` or your OpenAI client at `https://inference.hindsight.vectorize.io/v1`, change the model name, and keep the rest of your code. Full reference: [Inference docs](https://docs.hindsight.vectorize.io/cloud/inference/overview/).

## Try it

- **Self-hosted Hindsight:** run `hindsight-admin cloud login` and [turn on the decision model](/developer/decision-models). $5 of free credit covers the decision model for about 4,000 reflects.
- **Your own app:** [open the Playground](https://inference.hindsight.vectorize.io/playground), try the three models, then create an API key ([Inference docs](https://docs.hindsight.vectorize.io/cloud/inference/overview/)).
- **Open weights:** get them on [Hugging Face](https://huggingface.co/hindsight-ai), and see every result on [benchmarks.hindsight.vectorize.io](https://benchmarks.hindsight.vectorize.io).
- Star us on [GitHub](https://github.com/vectorize-io/hindsight).
