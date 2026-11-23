---
title: "Introducing Knowledge Banks: extending agent memory with SOTA multimodal search, at a fraction of the cost"
description: "Knowledge banks give Hindsight agents your PDFs, images, audio and video, kept exactly as written. Search ranks first on LegalBench-RAG and ATM-Bench, costs no LLM to fill, and linked to a memory bank lifts a support agent from 75% to 85%."
authors: [nicoloboschi]
date: 2026-11-24
slug: "2026/11/24/knowledge-banks"
image: /img/blog/knowledge-banks/cover.png
tags: [release, knowledge-banks]
draft: true
hide_table_of_contents: false
---

import ThemedImage from '@theme/ThemedImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

**Hindsight's memory banks learn what your agent hears. Today they can also read your documents.** A **knowledge bank** keeps your PDFs, images, audio and video exactly as written, finds the right page, paragraph or second with state-of-the-art multimodal search, and costs no LLM to fill. Link one to a memory bank and your agent answers from both: a support agent gets **85%** of questions right, against **75%** with memory alone.

## TL;DR

- **Memory, extended.** Link a knowledge bank to a memory bank and reflect reads both, with the documents as the reference. On a support-agent benchmark: 85% right, against 75% with everything in memory and 71% with everything in documents.
- **State-of-the-art multimodal search.** First on LegalBench-RAG (contracts) and on ATM-Bench (a person's photos, videos and emails). Second on ViDoRe V3 (enterprise PDFs), 1.9 points behind Mixedbread with its reranker.
- **One search over every kind of file.** PDFs, images, audio, video, text and Office files share one embedding space: ask in text and land on a chart, a photo or the 2:40 mark of a video.
- **A fraction of the cost.** No LLM reads your documents at ingest: on the support benchmark, filling the banks cost $0.34 in LLM calls instead of $0.95. On Hindsight Cloud a search costs $4.50 per 1,000, against $7.50 for Mixedbread with its reranker.
- **Both models on Hindsight Inference.** The embedder, `embeddinggemma-2`, and a new reranker for text and images, `hs-decide-1-flash-mm`, at $0.05 per million input tokens: about $0.11 to rerank 1,000 PDF pages.
- **Open source and in Hindsight Cloud.** In Hindsight 1.3, and in [Hindsight Cloud](https://hindsight.vectorize.io) for every account.

<!-- truncate -->

## Why memory

A memory bank is built to change its mind. Retain sends what your agent hears to an LLM, which pulls out facts, people and dates. Consolidation folds them into observations, and updates them when something new comes in. Ask about a customer and you get what's true about them now, not a pile of old chat logs.

That's exactly right for a person's history. It's wrong for a manual. A manual sent through retain comes out as loose facts, cut from the page they were on. A newer chat message can quietly outweigh the warranty terms. And the LLM reads every page on the way in, which costs money and loses whatever it gets wrong.

## Why knowledge

A knowledge bank does the opposite: it never changes what you give it. A PDF stays a PDF, one passage per page, as image and text. A video is split into short clips. A search returns the paragraph with the text around it, and points to its page or its second in the original file. Nothing is summarised or rewritten, and no LLM reads it at ingest.

That's exactly right for a manual. It's wrong for a person. A knowledge bank doesn't know that Lukas's server runs an old version, or that Amara has an approved exception, unless someone wrote it down as a document. It can't connect two chats, or tell that last week's answer was wrong.

## Why together

Most real questions need both. "Does my server still get updates?" needs the customer's version, which is in memory, and the support policy, which is in the manual. "How long do you keep my activity log?" has one answer in the manual, and a different one for a customer with an exception.

So link them. Add a knowledge bank to a memory bank's config, and reflect searches both at once. The agent is told the documents are the reference: when a memory says the return window is 60 days and the policy says 30, it answers 30 and says the memory is out of date. Consolidation never reads the documents, so a manual is never folded into the bank's memories.

## The proof: a support agent

Search scores don't show what happens when an agent has to mix a manual with what it remembers. So we built a benchmark for it: a support agent for a Nextcloud hosting service. It has the real Nextcloud 32 user and admin manuals (837 pages of PDF), plus 260 past support chats with 20 customers. The chats are noisy on purpose: about half of them are product chatter, and much of that chatter is wrong. Some customers also have an approved exception, a note that changes one setting for their account only. There are 114 questions, and none of them says where to look.

We ran it three ways, with the same model (Gemma 4) answering through reflect:

- **All in memory:** the manuals and the chats go into one memory bank.
- **All in a knowledge bank:** everything is kept as documents.
- **Together:** the chats go into the memory bank, and the manuals and exception notes go into a linked knowledge bank.

| Setup | All | Manual only | Manual + chat history | Chat contradicts manual | Approved exception |
|---|---|---|---|---|---|
| All in memory | 74.6% | 81% | 69% | 73% | 75% |
| All in a knowledge bank | 71.1% | 96% | 62% | 69% | 61% |
| **Together** | **85.1%** | **100%** | **85%** | 73% | **83%** |

Each setup fails where you'd expect. All in memory loses the manual's exact wording (81% on questions the manual alone answers). All in a knowledge bank keeps the manual but can't put a customer's history together (62% when an answer needs both, 61% on exceptions). Together gets both. The one place it doesn't help is a chat that contradicts the manual: 73%, the same as memory alone.

Here's one question of each kind, with what each setup answered.

<div className="post-wide">

<Tabs>
<TabItem value="docs" label="Manual only">

<ThemedImage alt={"A federated share's only permission option, answered by each setup"} sources={{light: "/img/blog/knowledge-banks/supportbench-docs-light.png", dark: "/img/blog/knowledge-banks/supportbench-docs-dark.png"}} />

The manual says there's only one option: **Can edit**. With everything in memory, the manual had been broken into loose facts, and the agent mixed in the options for calendars and public links.

</TabItem>
<TabItem value="both" label="Manual + chat history">

<ThemedImage alt={"Whether an old Nextcloud 30 server still gets updates, answered by each setup"} sources={{light: "/img/blog/knowledge-banks/supportbench-both-light.png", dark: "/img/blog/knowledge-banks/supportbench-both-dark.png"}} />

Only Lukas's chats say the server runs Nextcloud 30, released in September 2024. Only the manual says each version is supported for one year. Each single-bank setup missed one half. Together, it put them side by side: **no, it reached end of life around September 2025**.

</TabItem>
<TabItem value="conflict" label="Chat contradicts the manual">

<ThemedImage alt={"Whether adding a file to a project shares it, answered by each setup"} sources={{light: "/img/blog/knowledge-banks/supportbench-conflict-light.png", dark: "/img/blog/knowledge-banks/supportbench-conflict-dark.png"}} />

In an old chat a colleague says, with confidence, that adding a file to a project shares it. Both single-bank setups repeated that. Together, the agent went with the manual: **share the file with Daisuke first**.

</TabItem>
<TabItem value="exception" label="Approved exception">

<ThemedImage alt={"Activity retention for two customers, one with an approved exception, answered by each setup"} sources={{light: "/img/blog/knowledge-banks/supportbench-exception-light.png", dark: "/img/blog/knowledge-banks/supportbench-exception-dark.png"}} />

Two customers ask the exact same question, and only Amara has an exception. Both single-bank setups told both customers 365 days. Together, each got their own answer, because the exception note is tagged to its customer and the other customer's question never sees it.

</TabItem>
</Tabs>

</div>

## How it works

**Ingest: every file split its own way, then embedded as it is.** A PDF becomes one passage per page: the page as an image, and its text layer next to it. Audio and video become short overlapping clips. Text and Office files become passages that keep a heading with the content under it. Every passage goes through the same embedding model, so a question in text can land on a page image or the 2:40 mark of a video. Nothing is OCR'd, summarised or rewritten.

**Search: meaning, the rare words, then a token budget.** A search runs two searches side by side: one by meaning, and one by the query's rare words ("X200", a policy number, a name), in any language. Every candidate is scored from all its vectors, keyword matches are mixed in, a reranker can reorder the top ones, and the best hits come back with the text around them until your `max_tokens` is full. Your agent gets the paragraph, not a cut sentence, and every result points to its page or its second in the original file.

The full picture is in the [Knowledge Banks docs](/next/developer/knowledge-banks).

## State-of-the-art search: first on contracts and personal media, second on enterprise PDFs

A knowledge bank is only useful if it finds the right page. On contracts and on a person's photos, videos and emails it ranks first. On enterprise PDFs it ranks second, behind Mixedbread with its reranker. Every number was run with the open [agent-memory-benchmark](https://github.com/vectorize-io/agent-memory-benchmark) against a fresh Hindsight server through the public API, and is on [benchmarks.hindsight.vectorize.io](https://benchmarks.hindsight.vectorize.io).

- **Search only.** Each benchmark scores the ranking a knowledge bank returns against the dataset's own human labels, with the leaderboard's own metric (nDCG@10, or Recall@10 on ATM-Bench). No LLM writes an answer and no LLM judges one.
- **The whole corpus, every query.** We never shrink a corpus to make it easier.
- **One setup for all of them.** EmbeddingGemma 2 to embed, `hs-decide-1-flash-mm` to rerank, default settings. Other systems' scores are from their own published tables.

### Enterprise PDFs: ViDoRe V3

Eight sets of real enterprise documents (HR, finance in English and French, pharmaceuticals, energy, physics, industrial manuals, computer science), each searched with questions in its own language, as in Mixedbread's table.

We ran it two ways. From the **raw PDFs**, the server reads and splits every file itself. From the **page text** the dataset ships, every page is already plain text.

| Raw PDFs | nDCG@10 |
|---|---|
| Mixedbread + reranker | 69.5 |
| **Hindsight knowledge bank** | **67.6** |
| Cohere Embed 4 + Rerank 4 Pro | 67.4 |
| Mixedbread | 64.2 |
| Voyage Multimodal 3.5 | 57.5 |
| Gemini Embedding 2 | 48.5 |

| Page text | nDCG@10 |
|---|---|
| Mixedbread + reranker | 69.9 |
| **Hindsight knowledge bank** | **66.4** |
| Mixedbread | 62.3 |
| Voyage 4 Large | 60.6 |
| Gemini Embedding 2 | 58.7 |
| Cohere Embed 4 | 56.6 |
| OpenAI text-embedding-3-large | 51.2 |

A knowledge bank does better on the raw PDFs than on the clean text, because it reads each page as an image as well as its text. Mixedbread with its reranker stays ahead on both.

### Contracts: LegalBench-RAG

776 questions over 714 contracts (NDAs, M&A agreements, commercial contracts, privacy policies), scored on whether the top 5 results contain the exact text lawyers marked. This is the benchmark's mini version.

| System | Recall@5 |
|---|---|
| **Hindsight knowledge bank** | **85.7** |
| Conare | 85.4 |
| Voyage context-3 | 63.7 |
| Cohere Embed v5 + Rerank v5 Pro | 49.4 |
| OpenAI file search | 47.8 |
| NVIDIA Nemotron 3.5 Embed 8B | 44.5 |
| Qwen3 Embedding 8B | 42.5 |

The other systems ran without the contracts' file names, so this is our score without them too. With the file names, which a real contract store would have, a knowledge bank reaches **94.3**.

### One person's life: ATM-Bench

Four years of one person's data: 3,759 photos, 533 videos and 6,742 emails, and 1,013 questions about them. A knowledge bank finds **93.1%** of the evidence in its top 10 results, the top of the leaderboard.

| System | Recall@10 |
|---|---|
| **Hindsight knowledge bank** | **93.1** |
| Memexa v2 | 92.3 |
| RE-call | 90.1 |
| Graphon V1 | 90.0 |
| Memexa | 79.1 |
| ATM-RAG (paper baseline) | 68.7 |

Memexa v2 and Graphon V1, the closest behind, first run an LLM over every photo and email to build their memory. A knowledge bank embeds each item once, and that's all.

## A fraction of the cost

**Ingest: no LLM.** Filling a knowledge bank costs only the embedding. On the support benchmark, putting the manuals into a memory bank cost **$0.95** in LLM calls, because retain reads every page. With the manuals in a knowledge bank and only the chats in memory, it cost **$0.34**, and scored higher. A knowledge bank alone cost nothing in LLM calls.

**Search.** On Hindsight Cloud, a search costs **$4.50 per 1,000**, reranking included. Mixedbread's list price, with its reranker, is $7.50 per 1,000. On your own server you pay nothing per search.

### New on Hindsight Inference: the embedder and a reranker

We're adding both models a knowledge bank needs to [Hindsight Inference](https://inference.hindsight.vectorize.io), next to the models we [released this month](/blog/2026/11/03/hindsight-models). Each costs **$0.05 per million input tokens**.

- **`embeddinggemma-2`:** Google's multimodal EmbeddingGemma, with text, images, audio and video in one 768-dimension space. An image counts as about 270 tokens, 30 seconds of audio as at most 754, and a video as at most 32 frames (about 4,000 tokens; the soundtrack isn't read). Per 1,000 at this rate: images about $0.014, 30-second audio clips about $0.038, long videos about $0.20.
- **`hs-decide-1-flash-mm`:** our decision reranker, for text and images. An image counts as width × height ÷ 384 tokens, so a 792 × 1,024 page is 2,112. Each side can be 280 to 1,568 pixels, with up to 64 images per request. Reranking 1,000 such pages costs about $0.11.

Point a self-hosted server at them and you get a knowledge bank without running a model yourself. See [Configuration](/next/developer/configuration#knowledge-banks) and [Searching](/next/developer/knowledge-banks#searching).

### The trade-offs

- **Not first on every search benchmark.** Mixedbread with its reranker is ahead on ViDoRe V3, by 1.9 points on the raw PDFs and 3.5 on page text. Our leads are small: 0.3 points over Conare on contracts, 0.8 over Memexa v2 on ATM-Bench, and 0.2 over Cohere with its reranker on the raw PDFs.
- **Contradictions are still the hardest case.** When an old chat confidently contradicts the manual, the linked setup gets 73% right, no better than memory alone. Documents win when the agent finds both; it doesn't always find both.
- **Our scores use a reranker.** Every search score above is reranked by `hs-decide-1-flash-mm`. You can [plug in your own](/next/developer/knowledge-banks#searching) (any service that speaks Cohere's rerank API, pictures included). We haven't published scores without one.
- **Pick the embedder before you fill the bank.** Every passage is stored as a vector of the model's size. Switching to a model of another size means ingesting again.
- **Video and audio take time to ingest.** Decoding clips is the slow part: a one-hour video takes about a minute on a GPU. You can turn either off per bank.

## Try it

**Self-hosted Hindsight:** knowledge banks are on by default.

```bash
# Create a bank, upload a PDF, search it
hindsight kb create manuals
hindsight kb ingest manuals x200-manual.pdf --wait
hindsight kb search manuals "the pump keeps losing pressure"

# Let a memory bank's reflect read it
hindsight bank set-config support-agent --knowledge-banks manuals
```

EmbeddingGemma 2 runs inside the Hindsight API process by default. To call it on [Hindsight Inference](https://inference.hindsight.vectorize.io) with `hs-decide-1-flash-mm` as the reranker, or to use Gemini, Vertex, Voyage, Cohere or any OpenAI-compatible server, see [Configuration](/next/developer/configuration#knowledge-banks).

- **Hindsight Cloud:** knowledge banks are on for every account. Open **Knowledge Banks** in the console, drop in your files and search.
- **Every endpoint:** [Knowledge Banks API](/next/developer/api/knowledge-banks), with examples for Python, Node.js, Go, the CLI and HTTP.
- **Every result:** [benchmarks.hindsight.vectorize.io](https://benchmarks.hindsight.vectorize.io), and the [benchmark code](https://github.com/vectorize-io/agent-memory-benchmark) to run them yourself.
- Star us on [GitHub](https://github.com/vectorize-io/hindsight).
