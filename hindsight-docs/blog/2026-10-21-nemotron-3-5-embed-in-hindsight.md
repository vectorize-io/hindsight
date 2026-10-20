---
title: "We Got Early Access to NVIDIA Nemotron 3.5 Embed. It's Hindsight's New Top Embedding Model."
authors: [andrewneeser]
slug: "2026/10/21/nemotron-3-5-embed-in-hindsight"
date: 2026-10-21T11:30
tags: [hindsight, agent-memory, embeddings, recall, nemotron, nvidia, benchmarks]
description: "When NVIDIA gave us early access to Nemotron 3.5 Embed, we expected a solid embedding model. What we didn’t expect was for it to beat a model four times its size on memory retrieval and take the top spot on the Hindsight embedding leaderboard."
image: /img/blog/nemotron-3-5-embed-locomo.png
hide_table_of_contents: true
---

Embedding models are one of the core models inside Hindsight. Every memory Hindsight retains is turned into a vector embedding, and every recall searches those vectors to find memories by meaning, alongside keyword, graph and time search. A better embedding model gives everything downstream a better set of memories to work with.

We ran NVIDIA Nemotron 3.5 Embed, part of the Nemotron Retriever collection, and seven other embedding models, from small open models to OpenAI's API, on the benchmarks we use to evaluate Hindsight: LoCoMo and BEAM for long-conversation memory, 27 retrieval tasks across eight languages, our public embedding leaderboard, and query latency on a single GPU. Nemotron finished first on memory retrieval and first on the leaderboard, and it answered queries faster than a model just over half its size.

<!-- truncate -->

## TL;DR

- **#1 on memory retrieval** on LoCoMo (93.8%) and BEAM 100K (65.9%), against seven other embedding models including OpenAI text-embedding-3-small, BGE-M3 and Qwen3-Embedding-0.6B.
- **#1 on the Hindsight embedding leaderboard**, with the highest MRR and recall@5.
- **It reads the whole passage.** With a 32,768-token context, Nemotron finds 21.7 points more evidence in stored conversation passages than BGE-small-en v1.5 on LoCoMo.
- **Best on million-token histories**: 44.2% recall@10 on BEAM 1M and 38.0% on BEAM 10M.
- **Best across eight languages**: 55.1 mean nDCG@10 on 27 public retrieval tasks.
- **30 ms per query on one NVIDIA L4.** Faster than Qwen3-Embedding-0.6B, a model just over half its size, in half the time of Qwen3-Embedding-4B, a model four times its size, and more accurate than both.

![Results for Nemotron 3.5 Embed 1B and seven other embedding models across memory retrieval, long passages, long conversations, latency and eight languages](/img/blog/nemotron-3-5-embed-results-table.png)

## 1. It finds the right memory more often

LoCoMo and BEAM are the long-conversation memory benchmarks we track Hindsight on. We retained their conversations into Hindsight once, so every model ranked the exact same memories, and the embedding model was the only thing that changed. The score is the share of questions with an evidence memory in the top 10.

![Memory retrieval on LoCoMo](/img/blog/nemotron-3-5-embed-locomo.png)

![Memory retrieval on BEAM 100K](/img/blog/nemotron-3-5-embed-beam-100k.png)

Nemotron is first on both, ahead of OpenAI, BGE-M3, Qwen3 and four strong small models. BEAM 100K is the harder test: every model loses ground, and Nemotron holds the top spot.

## 2. It reads the whole passage

Alongside each memory, Hindsight keeps the conversation passage it came from, 600 to 800 tokens on average. Recall can search those passages directly, and a model can only match what it reads. Nemotron reads up to 32,768 tokens, so every passage is embedded whole.

![Recall@10 over Hindsight's stored conversation passages on LoCoMo](/img/blog/nemotron-3-5-embed-passages-locomo.png)

On LoCoMo, Nemotron finds 81.9% of the evidence in the top 10 passages, against 79.8% for BGE-M3 and 60.3% for BGE-small-en v1.5 (+21.7 points, 95% interval +18.8 to +25.0).

## 3. It holds up on million-token histories

BEAM's longest tiers hold about 1 million and 10 million tokens of conversation. Each model embeds every turn and ranks them for each question.

![Recall@10 on BEAM 1M](/img/blog/nemotron-3-5-embed-beam-1m.png)

Nemotron has the highest recall@10 on both tiers: 44.2% at 1M and 38.0% at 10M.

## 4. It works across languages

Hindsight supports Spanish, French, German, Portuguese, Japanese, Korean and Chinese. We scored every model on 27 public retrieval tasks in those languages, from MIRACL, MLDR, WebFAQ, mMARCO and T2Ranking.

![Mean nDCG@10 across eight languages](/img/blog/nemotron-3-5-embed-multilingual.png)

![nDCG@10 by language for every model](/img/blog/nemotron-3-5-embed-languages-table.png)

Nemotron scores 55.1 mean nDCG@10, the highest of any model we scored on these tasks, and leads in seven of the eight languages. The gaps over text-embedding-3-small are largest in Japanese, Korean and Traditional Chinese, nine to eleven points each.

## 5. It ranks first on the Hindsight embedding leaderboard

The [Hindsight embedding leaderboard](https://benchmarks.hindsight.vectorize.io/leaderboard/embeddings) runs each model end to end as Hindsight's embedder, through retain and recall, and ranks it on retrieval quality, speed and cost. Nemotron ranks first, with the highest MRR (0.852) and recall@5 (94.8%) on the board.

![The Hindsight embedding leaderboard](/img/blog/nemotron-3-5-embed-leaderboard.png)

## 6. It is faster than models a fraction of its size

A 1B embedding model has to earn its place in the recall path. On one NVIDIA L4, Nemotron embeds a query in 30 ms. Qwen3-Embedding-0.6B, just over half its size, takes 48 ms. Qwen3-Embedding-4B, four times its size, takes 59 ms. Nemotron scores higher than both on our memory benchmarks.

![Retrieval quality against query latency on one NVIDIA L4](/img/blog/nemotron-3-5-embed-latency.png)

Small models like BGE-small-en v1.5 and all-MiniLM-L6-v2 are faster still, at 6 to 11 ms, and remain a reasonable choice for small, English-only memory banks. For long conversations, many languages or large banks, Nemotron finds more.

## Using Nemotron 3.5 Embed with Hindsight

Serve the model behind an OpenAI-compatible embeddings endpoint and point Hindsight at it:

```bash
HINDSIGHT_API_EMBEDDINGS_PROVIDER=openai
HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL=<your endpoint>/v1
HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL=nvidia/Nemotron-3.5-Embed-1B-BF16
HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY=<key, or any value for a local server>
HINDSIGHT_API_EMBEDDINGS_OPENAI_DIMENSIONS=1536
HINDSIGHT_API_EMBEDDINGS_QUERY_PREFIX="query: "
HINDSIGHT_API_EMBEDDINGS_PASSAGE_PREFIX="passage: "
```

Two settings matter:

- **Dimensions.** Hindsight's pgvector index accepts up to 2,000 dimensions, so set 1,536. Nemotron supports Matryoshka truncation, and in our tests its full 2,048 dimensions scored within one point of 1,536 on every benchmark.
- **Prefixes.** Nemotron expects `query:` on searches and `passage:` on stored text. The two prefix settings add them for you.

## How we measured

- **Same memories for every model.** LoCoMo (10 conversations, 1,536 questions) and BEAM 100K (20 conversations, 355 questions) were retained into Hindsight once. A memory counts as evidence when it was extracted from a turn the dataset labels as evidence.
- **Semantic retrieval only.** These numbers isolate Hindsight's semantic search. Full recall also uses keyword search, graph traversal, time filtering and a reranker.
- **Comparison models.** OpenAI text-embedding-3-small through its API; BGE-M3, Qwen3-Embedding-0.6B, e5-small-v2, Arctic Embed S, all-MiniLM-L6-v2 and BGE-small-en v1.5 as open models; multilingual-e5-small and multilingual MiniLM on the multilingual tasks; Qwen3-Embedding-4B on the latency comparison. Each ran with its own prompts, native width and context limit.
- **Hardware.** One NVIDIA L4 for Nemotron and the larger open models. Latency is the median of 300 single queries after warm-up, every model in SentenceTransformers on the same L4.

Thanks to the NVIDIA Nemotron team for the early access.
