---
title: "What's New in Hindsight Cloud: Memory That Reads Screenshots"
authors: [benfrank241]
slug: "2026/09/21/hindsight-cloud-0-10-0"
date: 2026-09-21T15:00
tags: [hindsight-cloud, release, multimodal, retain, prompts, recall]
description: "Hindsight Cloud now runs 0.10.0. Images and files are first-class memory, you can preview the exact prompt an operation sends, and recall is harder to miss with."
image: /img/blog/hindsight-cloud-0-10-0.png
hide_table_of_contents: true
---

![What's new in Hindsight Cloud: images and files as memory, prompt preview, and typo-tolerant tags](/img/blog/hindsight-cloud-0-10-0.png)

A support thread lands in your memory bank. Two paragraphs of prose, a screenshot of what the CLI printed, one more sentence. Until this week, Hindsight Cloud read the prose and ignored the screenshot, which is a problem when the error code exists nowhere else.

Cloud now runs 0.10.0, and that changes. Here's what you get, and what we confirmed by running it against Cloud ourselves.

<!-- truncate -->

## TL;DR

- **Images and files are first-class memory.** Retain them inline with your text, and a recalled fact can hand back the exact image it was read from.
- **Preview the prompt before you spend a token.** See the exact messages retain, consolidation or reflect would send. No model call, no writes.
- **Recall forgives typos in tags**, and entity labels no longer need a fixed vocabulary.
- **Your knowledge base travels.** Whole-bank exports now carry knowledge pages and mental models.
- **Two behaviour changes worth knowing** before you upgrade your client: bursts of synchronous retain can now be refused with a 503, and listing mental models returns metadata by default.

## Memory that reads screenshots

The `content` field on a retain call now takes an ordered list of text, image and file blocks instead of only a string. The attachment is read in the position it occupies, so a chart between two paragraphs is read as the chart between those two paragraphs, not as an appendix.

We tested this against Cloud with a deliberately unfair example: a short support thread whose prose mentions no error code at all, and a PNG of a terminal containing one. The facts that came back included this:

> The CLI command 'hindsight retain --file report.pdf' triggered error 'VISIONCHECK_7781' with request_id 'req_c4f09ab2e7d15'

Neither string appears anywhere in the text we sent. The extractor read them out of the image.

**Provenance is per fact, not per document.** That same fact came back carrying the attachment it was extracted from. The other facts in the thread, the ones stated in the prose, came back with nothing attached. A fact cites an image only when it could not have been stated without looking at it.

A recall result carries a reference rather than bytes: an id, the media type, the size, the original filename, and a URL. Fetch that URL with the same credentials your recall used and you get the image back under the content type you sent it as, cacheable forever because the id is derived from the content hash. That's enough to render a thumbnail as a citation next to an answer, or to hand the original back to a vision model on the next turn.

For the design decisions underneath, including why an unresolvable citation is dropped rather than rounded to the nearest image, see [What It Takes to Put a Screenshot in an Agent's Memory](https://hindsight.vectorize.io/blog/2026/09/16/screenshot-agent-memory).

## See the prompt before you send it

Your bank's mission, disposition and strategies only mean something once you can see the prompt they land in. Now you can: one call returns the exact messages retain, consolidation or reflect would send, split into blocks that each name the setting behind them.

It costs nothing. There's no model call and nothing is written, which we also confirmed against Cloud.

Two things surprise people. The first is that a retain or consolidation mission lands in the **user** message rather than the system prompt, so a system-prompt-only view would show a configured mission as missing. The second is that settings you have switched off still come back, marked inactive, sitting in the position they would occupy if you turned them on.

[Full details here](https://hindsight.vectorize.io/blog/2026/09/17/prompt-preview), including the settings that silently do nothing.

## Recall that forgives typos

A tag filter used to be exact array containment, so a caller filtering on `typsecript` lost every memory tagged `typescript` before ranking even started. Better ranking can't fix that, because the match itself has to tolerate the misspelling.

Tag groups now take an optional `resolve` field. Leave it alone and nothing changes. Set it to `fuzzy` and that leaf's tags match by trigram similarity instead of literally. It resolves `typescropt` to `typescript` and `user:alcie` to `user:alice`, and deliberately does not resolve `mango` to `mongo`.

One limit worth knowing: similarity is length-sensitive, so a short tag has few trigrams and a single typo destroys most of them. `kakfa` does not resolve to `kafka`. Fuzzy matching earns its keep on descriptive tags and is close to inert on very short ones, which is the same property that keeps unrelated short words apart.

Alongside it, entity labels no longer need a vocabulary you define up front. A `multi-text` label takes as many values as the content warrants, each becoming its own entity, so a bank can derive a classification from its own content and then filter on it at recall.

## Your knowledge base travels

A bank transfer used to move documents and observations, the raw layer. Everything synthesized on top of it, the knowledge pages and mental models your bank spent real budget building, stayed behind and had to be rebuilt on the other side.

Pass `include_knowledge_base` on a whole-bank export and the archive carries them too. It's opt-in, off by default, and only available for whole-bank exports rather than a document list, because a page synthesized from an entire bank means nothing next to an arbitrary subset of it. The manifest reports what came along, so an archive tells you whether it holds a knowledge base before you import it.

## Two behaviour changes to know about

Most of 0.10.0 is additive. Two changes can affect code you already have.

**Bursts of synchronous retain can be refused.** Recall, reflect and retain now pass through admission control. When capacity is full a request waits for a slot up to a deadline and is then refused with a `503` and a `Retry-After` header, rather than queueing without limit while latency quietly climbs. The Python and TypeScript clients retry recall and reflect for you, honouring `Retry-After`. They deliberately do not retry retain, because a retried synchronous retain could write twice. If you push synchronous retain in bursts, handle the 503 or switch to async retain.

**Listing mental models returns metadata by default.** `GET .../mental-models` used to return every model's full synthesized content, which bloats a caller's context. It now defaults to metadata. Pass `detail=content` or `detail=full` when you actually want the text.

## What this means if you self-host

Nothing here is Cloud-only. All of it shipped in v0.10.0, which has been available for self-hosted deployments since mid-September; Cloud has now caught up. The difference is that on Cloud you did not have to upgrade anything, and the vision-capable model that image retains require is already configured for you.

If you run your own deployment, two upgrade notes matter more for you than for Cloud users: `curl` is no longer in the Docker images, so an exec-based health probe needs switching to an HTTP one, and the bank profile and background endpoints now return `410 Gone`. The [0.10.0 release notes](https://hindsight.vectorize.io/blog/2026/09/14/version-0-10-0) have the full list.

## Learn more

- [What It Takes to Put a Screenshot in an Agent's Memory](https://hindsight.vectorize.io/blog/2026/09/16/screenshot-agent-memory) on how per-fact provenance works
- [See the Exact Prompt Before You Spend a Token](https://hindsight.vectorize.io/blog/2026/09/17/prompt-preview) on the preview endpoint
- [What Hindsight Learned This Summer](https://hindsight.vectorize.io/blog/2026/09/18/what-hindsight-learned-this-summer) for the wider arc of the last few releases
- [What's New in Hindsight Cloud: June to August](https://hindsight.vectorize.io/blog/2026/09/01/hindsight-cloud-june-august-updates) for the previous roundup, including the knowledge base and enterprise controls
