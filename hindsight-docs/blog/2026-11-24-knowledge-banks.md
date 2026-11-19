---
title: "Introducing Knowledge Banks: top search quality on PDFs, images and video, at a fraction of the cost"
description: "Knowledge banks keep your files as they are and find the right page, clip or paragraph. No LLM at ingest or search: they match or beat Mixedbread on five public benchmarks, at 30x less to ingest."
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

:::danger Draft with mock numbers
Every Hindsight score and every cost in this post is **made up** to show the layout. The competitor scores are the published ones collected in [agent-memory-benchmark #78](https://github.com/vectorize-io/agent-memory-benchmark/pull/78). Replace the mock numbers with real runs before publishing, and remove this box. The support-agent section is real, but it comes from a single local run.
:::

**Your agent's memory learns. Its documents shouldn't change.** A memory bank turns what your agent hears into facts. A product manual, a contract or a training video should stay exactly as written: found, quoted, and pointed to by page or by second. Today we're releasing **knowledge banks**: upload your files as they are, and search them with text, an image or a clip. There's no LLM anywhere, at ingest or at search, and on five public benchmarks they match or beat the best search APIs at a fraction of the cost.

<div className="post-wide">

<a href="https://benchmarks.hindsight.vectorize.io"><ThemedImage alt={"Quality against cost on ViDoRe V3: Hindsight knowledge banks at the top left, more accurate than Mixedbread, Voyage, Gemini, Cohere and OpenAI, at a fraction of the cost to ingest"} sources={{light: "/img/blog/knowledge-banks/chart-vidore-cost-light.png", dark: "/img/blog/knowledge-banks/chart-vidore-cost-dark.png"}} /></a>

</div>

## TL;DR

- **Top quality on every kind of content.** On ViDoRe V3 (enterprise PDFs), MIRACL-Vision (page images in 18 languages), text-to-video, BrowseComp-Plus (100K web pages) and ATM-Bench (four years of one person's photos, videos and emails), a knowledge bank scores at or above the best published result. On ViDoRe V3 it reaches 63.4 nDCG@10, against 62.2 for Mixedbread.
- **30x cheaper to fill.** Ingesting 1,000 PDF pages costs $0.02 on Hindsight Cloud, or $0.004 on your own GPU, against $0.60 on Mixedbread. Nothing is sent to an LLM: a page is embedded as it is, image and text together.
- **No LLM in the loop.** On ATM-Bench, a knowledge bank beats the systems that run an LLM over every photo and email first. It reads each item once with an open embedding model, and it's done.
- **One open model for everything.** Text, PDF pages, images, audio and video go into one space with [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), which runs on your hardware. No reranker, no LLM, no per-query fee.
- **Memory and documents together.** Link a knowledge bank to a memory bank and reflect reads both. When a document and a memory disagree, the document wins.
- **Open source and in Hindsight Cloud.** In Hindsight 1.3, and in [Hindsight Cloud](https://hindsight.vectorize.io) for every account.

<!-- truncate -->

## Memories are learned. Documents are kept.

A memory bank is built to change its mind. Retain sends what your agent hears to an LLM, which pulls out facts, people and dates; consolidation folds them into observations that update as the world does. That's what you want for a user's preferences or a project's history.

It's the wrong tool for a product manual. You don't want the manual summarised into "facts", and you don't want a newer chat message to quietly override the warranty terms. You want the paragraph, the page it's on, and nothing reworded. And you want it for every kind of file your team has: scanned PDFs, slides, screenshots, recorded calls, how-to videos.

Most document search does this in one of two ways. It runs OCR or an LLM over every file first, which costs money on every page and loses whatever the model gets wrong. Or it sends every page to an embedding API priced per page or per token, for as long as you keep re-indexing. A knowledge bank does neither.

## How it works

**Ingest: every file split its own way, then embedded as it is.** A PDF becomes one passage per page: the page as an image, and its text layer next to it. Audio and video become short overlapping clips. Text and Office files become passages that keep a heading with the content under it. Every passage goes through the same embedding model, so a question in text can land on a page image or the 2:40 mark of a video. Nothing is OCR'd, summarised or rewritten.

**Search: meaning, the rare words, then a token budget.** A search runs two searches side by side: one by meaning, and one by the query's rare words ("X200", a policy number, a name), in any language. Every candidate is scored from all its vectors, keyword matches are mixed in, and the best hits come back with the text around them until your `max_tokens` is full. Your agent gets the paragraph, not a cut sentence, and every result points to its page or its second in the original file.

**Linked to a memory bank: documents win.** Add a knowledge bank to a memory bank's config and reflect searches both at once. The agent is told the documents are the reference: when a memory says the return window is 60 days and the policy says 30, it answers 30 and says the memory is out of date. Consolidation never reads the documents, so a manual is never folded into the bank's memories.

The full picture is in the [Knowledge Banks docs](/developer/knowledge-banks).

## Results

Every number here is on [benchmarks.hindsight.vectorize.io](https://benchmarks.hindsight.vectorize.io), run with the open [agent-memory-benchmark](https://github.com/vectorize-io/agent-memory-benchmark) against a fresh Hindsight server through the public API.

- **Search only.** Each benchmark scores the ranking a knowledge bank returns against the dataset's own human labels, with the leaderboard's own metric (nDCG@10, or Recall@10 on ATM-Bench). No LLM writes an answer and no LLM judges one.
- **The whole corpus, every query.** We never shrink a corpus to make it easier.
- **One setup for all five.** EmbeddingGemma 2, no reranker, no LLM, default settings. Other systems' scores are from their own published tables.

<div className="post-wide">

<a href="https://benchmarks.hindsight.vectorize.io"><ThemedImage alt={"Five search benchmarks: Hindsight knowledge banks against the best published results on ViDoRe V3, MIRACL-Vision, video, BrowseComp-Plus and ATM-Bench"} sources={{light: "/img/blog/knowledge-banks/chart-benchmarks-light.png", dark: "/img/blog/knowledge-banks/chart-benchmarks-dark.png"}} /></a>

</div>

### Enterprise PDFs: ViDoRe V3

Eight sets of real enterprise documents (HR, finance in English and French, pharmaceuticals, energy, physics, industrial manuals, computer science), searched in six languages. A knowledge bank scores **63.4** nDCG@10, against 62.2 for Mixedbread, 60.6 for Voyage 4 Large and 58.7 for Gemini Embedding 2.

We also ran it the hard way: from the **raw PDFs**, with the server reading and splitting every file itself, where the published numbers start from text the dataset already extracted. It scores **61.9**, still above every published system but Mixedbread.

### Page images in 18 languages: MIRACL-Vision

338,734 screenshots of Wikipedia pages, each language searched on its own corpus. **86.0**, against 85.2 for Mixedbread and 76.1 for Gemini Embedding 2. The page is never OCR'd: the model reads the image.

### Video: MSR-VTT, DiDeMo, YouCook2

A sentence finds the clip. **68.1** on average, against 67.0 for Mixedbread and 62.7 for Gemini Embedding 2. A video is split into 30-second windows with their sound, so the hit points to the moment, not just the file.

### A hundred thousand web pages: BrowseComp-Plus

830 hard research questions over 100,195 web pages, where most retrievers find almost nothing. **23.7** nDCG@10 on the pages needed to answer, against 20.3 for Qwen3-Embedding-8B, a far larger model, and 1.6 for BM25. Long pages are where keeping a hit's surrounding text, and matching the rare words, pays off.

### One person's life: ATM-Bench

Four years of one person's data: 3,759 photos, 533 videos and 6,742 emails, and 1,013 questions about them. **92.9** Recall@10, the top of the leaderboard. The two systems closest to it, Memexa v2 (92.3) and Graphon V1 (90.0), first run an LLM over every item to build their memory. A knowledge bank embeds each photo, video and email once, and that's all.

### Memory and documents together: a support agent

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

## What it costs

A knowledge bank pays for embedding and nothing else. No LLM reads your files when they come in, and none reads the results when they go out.

| Ingest 1,000 PDF pages | Cost | vs. Hindsight Cloud |
|---|---|---|
| Hindsight, your own GPU | $0.004 | 5x less |
| **Hindsight Cloud** | **$0.02** | |
| OpenAI text-embedding-3-large (text only) | $0.04 | 2x more |
| Cohere Embed 4 | $0.30 | 15x more |
| Voyage 4 Large | $0.36 | 18x more |
| Gemini Embedding 2 | $0.45 | 22x more |
| Mixedbread | $0.60 | 30x more |

On ATM-Bench, building the memory with an LLM first (DeepSeek, as Memexa v2 does) costs about **$14** for the 11,034 items, before the first question. Filling a knowledge bank with the same items costs **$0.31**, and scores higher.

Search costs one query embedding: about 40 ms and a fraction of a cent per thousand searches on Hindsight Cloud, and nothing per query on your own server.

**A month, end to end.** For a team with 100,000 PDF pages and 100,000 searches a month, storage included, a knowledge bank on Hindsight Cloud runs about **$55–155 a month**. The same workload on Mixedbread is about **$750**.

### The trade-offs

- **Our numbers are without a reranker.** Mixedbread with its own reranker reaches 69.9 on ViDoRe V3, above our 63.4. You can [plug in a reranker](/developer/knowledge-banks#searching) (any service that speaks Cohere's rerank API, pictures included), at a cost per search; we haven't published scores with one yet.
- **Pick the embedder before you fill the bank.** Every passage is stored as a vector of the model's size. Switching to a model of another size means ingesting again.
- **Video and audio take time to ingest.** Decoding clips is the slow part: a one-hour video takes about a minute on a GPU. You can turn either off per bank.

## Try it

**Self-hosted Hindsight:** knowledge banks are on by default.

```bash
# Create a bank, upload a PDF, search it
curl -X POST localhost:8888/v1/default/knowledge-banks -H "Content-Type: application/json" \
  -d '{"bank_id": "manuals"}'
curl -X POST localhost:8888/v1/default/knowledge-banks/manuals/files/ingest -F "files=@x200-manual.pdf"
curl -X POST localhost:8888/v1/default/knowledge-banks/manuals/search -H "Content-Type: application/json" \
  -d '{"query": "the pump keeps losing pressure"}'
```

EmbeddingGemma 2 runs inside the Hindsight API process by default. To use Gemini, Vertex, Voyage, Cohere or any OpenAI-compatible server instead, see [Choosing an Embedder](/developer/knowledge-banks#choosing-an-embedder).

- **Hindsight Cloud:** knowledge banks are on for every account. Open **Knowledge Banks** in the console, drop in your files and search.
- **Every endpoint:** [Knowledge Banks API](/developer/api/knowledge-banks).
- **Every result:** [benchmarks.hindsight.vectorize.io](https://benchmarks.hindsight.vectorize.io), and the [benchmark code](https://github.com/vectorize-io/agent-memory-benchmark) to run them yourself.
- Star us on [GitHub](https://github.com/vectorize-io/hindsight).
