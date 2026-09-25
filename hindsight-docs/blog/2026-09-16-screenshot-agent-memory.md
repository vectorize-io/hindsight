---
title: "What It Takes to Put a Screenshot in an Agent's Memory"
authors: [benfrank241]
slug: "2026/09/16/screenshot-agent-memory"
date: 2026-09-16T16:00
tags: [hindsight, agent-memory, multimodal, retain, provenance, vision, release]
description: "Storing an image is easy. Making a fact cite the screenshot it was read from, without inventing evidence, is the hard part. How attachments work in Hindsight 0.10.0."
image: /img/blog/screenshot-agent-memory.png
hide_table_of_contents: true
---

![A fact recalled from agent memory, shown with the screenshot it was extracted from](/img/blog/screenshot-agent-memory.png)

A support thread comes in. Two paragraphs, a screenshot of what the CLI printed, one more sentence. The request ID, the bank ID and the error code exist in exactly one place: inside the image.

Your agent reads the prose and remembers the prose. The screenshot becomes a file in a bucket somewhere.

<!-- truncate -->

## TL;DR

- `content` now takes an ordered list of text, image and file blocks, so an attachment is read in the position it occupies.
- A recalled fact comes back with the attachments it was extracted from, per fact rather than per chunk.
- Attribution isn't inferred from position. The extractor is asked, and answers that don't resolve are dropped rather than rounded to the nearest image.
- If the configured model can't read images, the retain fails before a byte is written.
- Shipped in v0.10.0, available on Hindsight Cloud and for self-hosted deployments.

## This isn't file upload

Hindsight Cloud has [accepted file uploads since March](https://hindsight.vectorize.io/blog/2026/03/09/hindsight-document-upload): drop in a PDF, extract text, retain it. That's still the right tool when the file is the unit.

A screenshot pasted mid-conversation isn't a document. Treating it as one loses the thing worth keeping: it was attached *there*, after that sentence, as evidence for that claim.

Two things are new, and both are about position rather than storage. An attachment is read where it sits, so a chart between two paragraphs is read as the chart between those two paragraphs. And provenance is per fact, so when recall returns "the error was a dimension mismatch on a PNG declared as image/png," it can hand back the screenshot that sentence came out of. Not the document. Not every image in the chunk.

## One string, or everything breaks

The obvious implementation threads a new content shape through the pipeline. Hindsight does the opposite: blocks flatten at the API boundary into a single canonical body, each attachment becoming an atomic placeholder on its own paragraph.

The reason is that too much already depends on content being one string: idempotency by content hash, `update_mode=append`, chunk-delta re-extraction, `reprocess_document`, export and import. So the flattening is exactly equivalent for text. This:

```json
{"content": "The importer rounds half-up."}
```

and this:

```json
{"content": [{"type": "text", "text": "The importer rounds half-up."}]}
```

produce an identical stored body and an identical content hash. Placeholders expand back into real multimodal parts at one point only, prompt assembly, with each image sitting exactly where its placeholder stood. The bytes never travel in the request payload; they're written to storage first, content-addressed by SHA-256, so retrying a queued operation doesn't drag a base64 blob along.

## Asking, rather than guessing

The obvious approaches are all wrong. Attribute by proximity and a fact gets the nearest image. Attribute by chunk and every fact gets every attachment. Both produce an answer for every fact, and both are frequently lying.

Instead the prompt numbers the attachments and asks the extractor which ones each fact needed, with a narrow instruction: list an attachment only when the fact could not be stated without looking at it. Facts come back carrying integers, which resolve against that chunk's placeholder order.

What happens to a number that doesn't resolve is the decision that matters. It's dropped. A fact claiming attachment 3 in a chunk holding two ends up citing nothing, and the code comment says why: attaching the nearest one would be inventing provenance. Zero and negatives go the same way, and a text-only chunk resolves to nothing even if the model insists it used an image.

That leaves a system that will return a fact with no evidence attached, which is the correct failure. "I can't show you where this came from" is usable. "Here's a screenshot that doesn't support this claim" is not.

The tests pin both directions, including one that runs a real model against a diagram whose facts appear nowhere in the prose and asserts a partition: some fact must cite the image, or the evidence is invisible, and not every fact may, or it's being cited for prose it doesn't support.

Worth being precise: this is a model judgement with deterministic, range-checked resolution around it. Not a guarantee.

## An image costs 22 characters

Chunking has a character budget. Count an image against it by its real size and a long article with three screenshots gets split away from its own screenshots. An earlier iteration did exactly that.

Now the budget counts text, and an attachment costs the length of its placeholder token, roughly 22 characters. The real limit is a separate per-chunk cap matched to what the provider accepts in one request, defaulting to 8. One sentence followed by ten images stays in a single chunk, because splitting it would destroy the only thing making those images interpretable. Exceeding the cap doesn't reject anything, it just produces more chunks.

## Two kinds of attachments in one response

A recall response carries attachment data in two places, and they deliberately differ.

| | `chunks{}.attachments` | `results[].attachments` |
|---|---|---|
| **Scope** | everything the chunk's text references | only what the extractor attributed to that fact |
| **Granularity** | per chunk | per fact |
| **Answers** | "what was in the source here?" | "what is this claim based on?" |
| **When empty** | chunk has no attachments | fact was stated in the text |

Rendering the source document? Use the chunk set. Showing a user why the agent believes something? Use the fact set, because the chunk set would overstate your evidence. When there's nothing to report, the field is omitted rather than returned empty.

## Getting the image back

A recall result doesn't carry bytes. It carries a reference:

```json
{
  "id": "8f41c2ad9e6b",
  "hash": "8f41c2ad9e6b47f0...",
  "kind": "image",
  "media_type": "image/png",
  "byte_size": 68344,
  "filename": "error-screenshot.png",
  "url": "/v1/default/banks/acme-support/attachments/8f41c2ad9e6b"
}
```

`GET` that path and you get the bytes under the Content-Type the caller declared at retain. It's authorized against the bank, so the same credentials your recall used. A missing attachment and a bank you can't see both return 404, so the endpoint can't be used to probe what a bank holds. Because the id is derived from the content hash, the bytes at a URL can never change, and they're served cacheable indefinitely.

That reference is enough for either of the two things you'd want to do with it.

**Show it as a citation.** You have `media_type`, `byte_size` and the original `filename` before fetching anything, so you can render a thumbnail next to the answer, or a download link for a PDF, and only pull the bytes when someone asks to see it.

**Feed it back to a model.** Fetch the bytes, base64 them, and send them as an image block in your next prompt: the same shape you retained it with. The agent that recalls "the error was a dimension mismatch" can put the original screenshot back in front of a vision model and ask it something new.

## Where it refuses

Most of the interesting engineering here is in the failure paths.

**No vision model, no retain.** An image-bearing retain fails at ingress, before anything is written. The alternative is extracting from the prose and quietly skipping the images, which leaves a document that looks retained while the information you cared about is gone. A hard error is recoverable; a silent omission isn't, because nothing will ever tell you it happened.

**"Probably fine" counts as no.** Capability is three-valued: yes, no, and can't tell. Can't tell refuses. This one catches self-hosters, because an OpenAI-compatible endpoint that isn't OpenAI reports can't tell, which covers most local and gateway setups. Setting the vision flag explicitly is an operator decision, not a default.

**Batch mode is incompatible.** With the batch retain API enabled, attachment-bearing retains fail outright, because that path builds provider request bodies directly and never sees the interleaved parts.

One thing degrades instead of refusing: if an attachment's bytes go missing from storage, the prompt gets an "attachment unavailable" marker rather than failing the retain.

## Storage, briefly

Attachments are deduplicated by SHA-256 across the bank, so retaining the same screenshot in forty documents gives you one blob and one row. The filename can't live with the bytes, since it isn't a property of them: the same PDF can be `policy-v1.pdf` in one document and `escalation-runbook.pdf` in another. Cleanup runs off references, so re-ingesting a document without an image drops that edge, and the blob goes when the last reference does.

On upgrading: the migration adds two empty tables plus an array column on the memory tables, which may not be small in a mature deployment. Plan it like any column addition at scale.

## Limits worth knowing

| Limit | Value | Notes |
|---|---|---|
| Source type | base64 inline only | no URL fetch or pre-uploaded handle yet |
| Max size | 20 MB per attachment | decoded, server-level setting |
| Max count | 50 per retain item | split across items to go higher |
| Per chunk | 8 by default | bank-configurable, not a rejection |
| Media types | any well-formed type | no allowlist; the provider's error surfaces if it can't read it |
| Per-fact provenance | `facts` extraction mode | `chunks` and `verbatim` take everything in the chunk |
| MCP | not yet | attachments come back over HTTP, not the MCP tools |
| Hindsight Cloud | available | running 0.10.0, with a vision-capable model already configured |

The extraction-mode caveat is the one most likely to surprise you. If you've configured chunk or verbatim mode, the fact is the chunk, so it carries everything the chunk holds and no model is consulted.

## FAQ

**Do I have to change existing code?**
No. `content` still takes a plain string, and that path is unchanged down to the content hash. Blocks are opt-in per item.

**What if I send an image to a model that can't read it?**
The retain fails with a clear error rather than dropping the image. If your provider is OpenAI-compatible but not OpenAI, it reports an unknown capability and is refused until you set the vision flag.

**Is the fact-to-image link guaranteed correct?**
No. The extractor decides which attachments a fact required. What's guaranteed is that an answer which doesn't resolve to a real attachment in that chunk is discarded rather than approximated, so a wrong citation is less likely than no citation.

**Can I get attachments through MCP?**
Not in 0.10.0. The fields are on the HTTP responses; the MCP recall tool returns facts without them.

## Learn more

- [Inside retain()](https://hindsight.vectorize.io/blog/2026/07/13/inside-retain-agent-memory) is the write path this extends
- [What's new in Hindsight 0.10.0](https://hindsight.vectorize.io/blog/2026/09/14/version-0-10-0) covers the rest of the release
- [Cross-Encoder Reranking](https://hindsight.vectorize.io/blog/2026/08/28/cross-encoder-reranking-agent-memory) explains what happens to a fact after it's retrieved
- [Stop Growing Your Always-On Context](https://hindsight.vectorize.io/blog/2026/09/16/stop-growing-your-system-prompt) argues for retrieving evidence per turn
