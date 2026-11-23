---
title: "Introducing Knowledge Banks: extending agent memory with SOTA multimodal search, at a fraction of the cost"
description: "Knowledge banks give Hindsight agents your PDFs, images, audio and video, kept exactly as written. Search ranks first on LegalBench-RAG and ATM-Bench, a bank costs no LLM calls to fill, and linked to a memory bank it lifts a support agent from 75% to 85%."
authors: [nicoloboschi]
date: 2026-11-24
slug: "2026/11/24/knowledge-banks"
image: /img/blog/knowledge-banks/cover.png
tags: [release, knowledge-banks]
hide_table_of_contents: false
---

import ThemedImage from '@theme/ThemedImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

**Hindsight's memory banks learn what your agent hears. Today they can also read your documents.** A **knowledge bank** keeps your PDFs, images, audio and video exactly as written, finds the right page, paragraph or second with state-of-the-art multimodal search, and needs no LLM to fill. Link one to a memory bank and your agent answers from both: a support agent gets **85%** of questions right, against **75%** with memory alone.

## TL;DR

- **Memory, extended.** Link a knowledge bank to a memory bank and one reflect call reads both, with the documents as the reference. On a support-agent benchmark: 85% right, against 75% with everything in memory and 71% with everything in documents.
- **State-of-the-art multimodal search.** First on LegalBench-RAG (contracts, 87.2) and ATM-Bench (a person's photos, videos and emails, 93.1). Second on ViDoRe V3 (enterprise PDFs), 1.5 points behind Mixedbread with its reranker.
- **A fraction of the cost.** No LLM reads your documents: on the support benchmark, the manuals cost about $0.61 in LLM calls to put in a memory bank, and none in a knowledge bank, only the embedding. A hosted search costs $4.50 per 1,000, against $7.50 for Mixedbread with its reranker, and both models run on Hindsight Inference at $0.05 per million tokens.
- **Open source, in Hindsight Cloud, and at scale on Enterprise.** In Hindsight 1.3, in [Hindsight Cloud](https://hindsight.vectorize.io) for every account, and on Hindsight Enterprise, where knowledge banks can live on Memlake, in object storage.

<!-- truncate -->

## Why memory

<div className="post-wide">

<ThemedImage alt={"A memory bank turns a chat into facts and observations that update"} sources={{light: "/img/blog/knowledge-banks/why-memory-light.png", dark: "/img/blog/knowledge-banks/why-memory-dark.png"}} />

</div>

A memory bank is built to change its mind. Retain sends what your agent hears to an LLM, which pulls out facts, people and dates. Consolidation folds them into observations, and updates them when something new comes in. Ask about a customer and you get what's true about them now, not a pile of old chat logs.

That's exactly right for a person's history. It's wrong for a manual. A manual sent through retain comes out as loose facts, cut from the page they were on. A newer chat message can quietly outweigh the warranty terms. And the LLM reads every page on the way in, which costs money and loses whatever it gets wrong.

## Why knowledge

<div className="post-wide">

<ThemedImage alt={"A knowledge bank splits files into passages and returns the page or the second"} sources={{light: "/img/blog/knowledge-banks/why-knowledge-light.png", dark: "/img/blog/knowledge-banks/why-knowledge-dark.png"}} />

</div>

A knowledge bank does the opposite: it never changes what you give it. A PDF stays a PDF, one passage per page, as image and text. A video is split into short clips. A search returns the paragraph with the text around it, and points to its page or its second in the original file. Nothing is summarised or rewritten, and no LLM reads it at ingest.

That's exactly right for a manual. It's wrong for a person. A knowledge bank doesn't know that Lukas's server runs an old version, or that Amara has an approved exception, unless someone wrote it down as a document. It can't connect two chats, or tell that last week's answer was wrong.

## Why together

<div className="post-wide">

<ThemedImage alt={"One reflect call combines Lukas's version from memory with the support policy from the manual"} sources={{light: "/img/blog/knowledge-banks/why-together-light.png", dark: "/img/blog/knowledge-banks/why-together-dark.png"}} />

</div>

Most real questions need both. "Does my server still get updates?" needs the customer's version, which is in memory, and the support policy, which is in the manual. "How long do you keep my activity log?" has one answer in the manual, and a different one for Amara, whose account has an approved exception.

So link them. Add a knowledge bank to a memory bank's config, and reflect searches both at once. The exception note is tagged to Amara, so her question finds it and every other customer's question doesn't. On the benchmark below, that's the difference between 61% and 83% on exceptions.

### Isn't this just RAG next to memory?

<div className="post-wide">

<ThemedImage alt={"A vector store next to memory, against a knowledge bank linked to a memory bank"} sources={{light: "/img/blog/knowledge-banks/rag-vs-link-light.png", dark: "/img/blog/knowledge-banks/rag-vs-link-dark.png"}} />

</div>

You could put a vector store beside your memory and search both. Three things make a link different:

- **One call, one budget.** Reflect searches the memories and every linked knowledge bank in the same step, and the passages share the reflect's token budget and its tag filters. Your agent doesn't decide which store to ask.
- **Documents are the reference.** The agent sees which evidence is a document and which is a memory. When they disagree, it answers from the document and says the memory is out of date.
- **Memory never absorbs the manual.** Consolidation and mental models never read the documents, so a manual is never turned into loose facts that drift from the page.

## The proof: a support agent

Search scores don't show what happens when an agent has to mix a manual with what it remembers. So we built a benchmark for it: a support agent for a Nextcloud hosting service. It has the real Nextcloud 32 user and admin manuals (837 pages of PDF), plus 260 past support chats with 20 customers. The chats are noisy on purpose: about half of them are product chatter, and much of that chatter is wrong. Some customers also have an approved exception, a note that changes one setting for their account only. There are 114 questions, and none of them says where to look.

We ran it three ways, with the same model (Gemma 4 26B) answering through reflect:

- **All in memory:** the manuals and the chats go into one memory bank.
- **All in a knowledge bank:** everything is kept as documents.
- **Together:** the chats go into the memory bank, and the manuals and exception notes go into a linked knowledge bank.

<div className="post-wide">

<ThemedImage alt={"SupportBench scores by kind of question for the three setups"} sources={{light: "/img/blog/knowledge-banks/supportbench-categories-light.png", dark: "/img/blog/knowledge-banks/supportbench-categories-dark.png"}} />

</div>

Each setup fails where you'd expect. All in memory loses the manual's exact wording (81% on questions the manual alone answers). All in a knowledge bank keeps the manual but can't put a customer's history together (62% when an answer needs both, 61% on exceptions). Together gets both. The one place it doesn't help is a chat that contradicts the manual: 73%, the same as memory alone (19 of 26).

The judge is the same Gemma 4 26B: it grades each answer against the question's reference answer, and fails it if it makes a wrong claim the question was built to tempt. One run per setup.

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

<div className="post-wide">

<ThemedImage alt={"Ingest splits each kind of file; search runs by meaning and rare words, reranks and fills the token budget"} sources={{light: "/img/blog/knowledge-banks/how-it-works-light.png", dark: "/img/blog/knowledge-banks/how-it-works-dark.png"}} />

</div>

**Ingest: every file split its own way, then embedded as it is.** A PDF becomes one passage per page: the page as an image, and its text layer next to it (a scanned page has no text layer and is found by its picture). Audio and video become short overlapping clips. Text and Office files become passages that keep a heading with the content under it. Every passage goes through the same embedding model, so a question in text can land on a page image or the 2:40 mark of a video. Nothing is OCR'd, summarised or rewritten.

**Search: meaning, the rare words, then a token budget.** A search runs two searches side by side: one by meaning, and one by the query's rare words ("X200", a policy number, a name), in any language. Every candidate is scored from all its vectors, keyword matches are mixed in, a reranker can reorder the top ones, and the best hits come back with the text around them until your `max_tokens` is full. Your agent gets the paragraph, not a cut sentence, and every result points to its page or its second in the original file.

The full picture is in the [Knowledge Banks docs](/next/developer/knowledge-banks).

## State-of-the-art search: first on contracts and personal media, second on enterprise PDFs

A knowledge bank is only useful if it finds the right page. On contracts and on a person's photos, videos and emails it ranks first. On enterprise PDFs it ranks second, behind Mixedbread with its reranker.

We ran every benchmark with [Agent Memory Benchmark](https://agentmemorybenchmark.ai), the open harness we maintain at Vectorize, against a fresh Hindsight server through the public API. The results are on [benchmarks.hindsight.vectorize.io](https://benchmarks.hindsight.vectorize.io/benchmarks), and every query and hit is in the raw runs on Agent Memory Benchmark.

- **Search only.** Each benchmark scores the ranking a knowledge bank returns against the dataset's own human labels, with the leaderboard's own metric. No LLM writes an answer and no LLM judges one.
- **Every query, on the full corpus each benchmark defines.**
- **One setup for all of them.** EmbeddingGemma 2 to embed, `hs-decide-1-flash-mm` to rerank, default settings. Other systems' scores are the ones they or the leaderboard published, as of November 2026.

### Enterprise PDFs: ViDoRe V3

Eight sets of real enterprise documents (HR, finance in English and French, pharmaceuticals, energy, physics, industrial manuals, computer science), each searched with questions in its own language, as in Mixedbread's table.

We ran it two ways. From the **raw PDFs**, the server reads and splits every file itself. From the **page text** the dataset ships, every page is already plain text.

<ThemedImage alt={"ViDoRe V3 raw PDFs, nDCG@10: Mixedbread with reranker 69.5, Hindsight 68.0, Cohere Embed 4 with Rerank 4 Pro 67.4"} sources={{light: "/img/blog/knowledge-banks/bench-vidore-pdf-light.png", dark: "/img/blog/knowledge-banks/bench-vidore-pdf-dark.png"}} />

<ThemedImage alt={"ViDoRe V3 page text, nDCG@10: Mixedbread with reranker 69.9, Hindsight 66.2, Mixedbread 62.3"} sources={{light: "/img/blog/knowledge-banks/bench-vidore-text-light.png", dark: "/img/blog/knowledge-banks/bench-vidore-text-dark.png"}} />

A knowledge bank does better on the raw PDFs than on the clean text, because it reads each page as an image as well as its text. Mixedbread with its reranker stays ahead on both.

### Contracts: LegalBench-RAG

776 questions over 714 contracts (NDAs, M&A agreements, commercial contracts, privacy policies): the benchmark's official mini set. The score is the share of the exact text lawyers marked that the top 5 results contain.

<ThemedImage alt={"LegalBench-RAG mini, recall@5: Hindsight 95.0 with file names and 87.2 without, Conare 85.4"} sources={{light: "/img/blog/knowledge-banks/bench-legalbench-light.png", dark: "/img/blog/knowledge-banks/bench-legalbench-dark.png"}} />

This metric rewards returning more text, so here is how much we return: the search runs at the API's default 4,096-token budget, and the 5 results average about 3,800 tokens, about what 5 chunks of a typical RAG setup hold. The other systems ran without the contracts' file names, so this is our score without them too. With the file names, which a real contract store would have, a knowledge bank reaches **95.0**. Competitor numbers are from [Conare's table](https://conare.ai/benchmarks), as of 9 November 2026.

### One person's life: ATM-Bench

Four years of one person's data: 3,759 photos, 533 videos and 6,742 emails, and 1,013 questions about them. A knowledge bank finds **93.1%** of the evidence in its top 10 results, above every entry on the [ATM-Bench leaderboard](https://atmbench.github.io/leaderboard.html) (we haven't submitted there yet).

<ThemedImage alt={"ATM-Bench, recall@10: Hindsight 93.1, Memexa v2 92.3, RE-call 90.1"} sources={{light: "/img/blog/knowledge-banks/bench-atm-light.png", dark: "/img/blog/knowledge-banks/bench-atm-dark.png"}} />

Each photo and video goes in twice: as the raw file, and as the text record the benchmark ships with it (time, place, caption, OCR, tags). A hit on either counts. Memexa v2 and Graphon V1, the closest behind, run an LLM over every item to build their memory. A knowledge bank runs no LLM of its own: it embeds the files and the records as they come.

Memexa v2's recall counts the items it passes to its answer model, which may not be the same cut as a top 10.

## A fraction of the cost

<div className="post-wide">

<ThemedImage alt={"LLM cost to fill, hosted search price, and Hindsight Inference prices"} sources={{light: "/img/blog/knowledge-banks/cost-light.png", dark: "/img/blog/knowledge-banks/cost-dark.png"}} />

</div>

**Filling it: no LLM.** A knowledge bank costs only the embedding. On the support benchmark, filling one memory bank with the manuals and the chats cost **$0.95** in LLM calls. With the manuals in a knowledge bank and only the chats in memory, it cost **$0.34**, and it scored higher. The difference, about **$0.61**, is what retain spent reading the 837 pages of manuals; in a knowledge bank they cost only the embedding. The linked setup sends the agent more context, so a question costs more: $0.00059 against $0.00031 with memory alone. On LLM spend, memory alone only catches up after about 2,200 questions.

**Searching.** On Hindsight Cloud, a search costs **$4.50 per 1,000**, reranking included. Mixedbread's list price, with its reranker, is $7.50 per 1,000. On your own server you pay only for the models you call.

### New on Hindsight Inference: the embedder and a reranker

Both models a knowledge bank needs are now on [Hindsight Inference](https://inference.hindsight.vectorize.io), next to the models we [released this month](/blog/2026/11/03/hindsight-models), each at **$0.05 per million input tokens**:

- **`embeddinggemma-2`:** Google's multimodal EmbeddingGemma, text, images, audio and video in one 768-dimension space. 1,000 images cost about $0.014. On Inference a video is read from its frames; the in-process model also hears its soundtrack.
- **`hs-decide-1-flash-mm`:** our decision reranker, for text and images. Reranking 1,000 PDF pages costs about $0.11.

Point a self-hosted server at them and you get a knowledge bank without running a model yourself. See [Configuration](/next/developer/configuration#knowledge-banks) for the settings.

## At scale: Memlake on Hindsight Enterprise

Open-source Hindsight keeps knowledge banks in Postgres, with a pgvector index. On Hindsight Enterprise they can live on [Hindsight Memlake](/blog/2026/10/13/memlake) instead, the storage engine that already serves agent memory from object storage: a bank's documents and passages go to a bucket you control, and the API, SDKs and CLI don't change. See [Hindsight Enterprise](/cloud-enterprise/enterprise).

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
- **Every result:** [benchmarks.hindsight.vectorize.io](https://benchmarks.hindsight.vectorize.io/benchmarks), and [Agent Memory Benchmark](https://agentmemorybenchmark.ai) to run them yourself.
- Star us on [GitHub](https://github.com/vectorize-io/hindsight).
