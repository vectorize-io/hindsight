---
title: "From Documents to Decisions: architxt and Hindsight"
authors: [garethjcooper]
slug: "2026/07/03/architxt-hindsight-temporal-mosaic"
date: 2026-07-03T12:00
tags: [hindsight, architxt, integration, memory, temporal, enterprise-architecture, community, tutorial]
description: "How architxt uses Hindsight as a time-aware agent memory layer to turn fragmented enterprise architecture documents into a queryable, current-state Temporal Mosaic."
image: /img/blog/architxt-hindsight.png
hide_table_of_contents: true
---

![From documents to decisions: architxt and Hindsight](/img/blog/architxt-hindsight.png)

Every enterprise system is described somewhere. The problem is *where*.

Architecture decisions live in impact assessments buried in SharePoint. Integration details hide in Confluence pages written three years ago. Current-state diagrams sit in slide decks from a programme that was "phase 2'd" into oblivion. When someone asks, "How does billing actually work now?", the answer is never in one place. It is spread across dozens of documents, each written at a different time, for a different audience, with different assumptions about what was "current."

This is the problem architxt was built to solve. It integrates with [Hindsight](https://github.com/vectorize-io/hindsight), an agent memory system used as a durable, time-aware memory layer for enterprise semantic search and cross-document reasoning.

<!-- truncate -->

## architxt: turn documents into structured knowledge

architxt is a document processing pipeline and research framework, accessed via a web UI. You upload documents (PDFs, Word files, PowerPoint decks) and it extracts clean, structured content using Docling, LLM-based denoising, vision analysis for diagrams, and entity detection. The output is combined with key metadata to prepare it for Hindsight. The main components of this are:

- **Documents** with metadata, tags, and extracted blocks.
- **Entities** (systems, services, capabilities) detected and normalised across documents.
- **Mental models**: reusable LLM prompts that analyse an entity from a specific angle (capabilities, interfaces, summaries). These allow for quick retrieval of common dimensions per entity.
- **A temporal mosaic**: research and query extraction of the best-known state of each entity, regardless of when the source document was written.

![architxt detects and normalises entities across a document, mapping aliases back to a single entity.](/img/blog/architxt-entity-detection.png)

The key insight is that **knowledge freshness is not document freshness**. A 2020 component design is still valid if that component has not changed, even if the rest of the system was rewritten twice since. architxt tracks which document touched which entity, when, and lets you reason across the whole corpus without pretending there is a single "as-is" document.

## The architxt process and minimum viable input

The tool works best when documents arrive with a small amount of consistent data. architxt does not need everything to be perfect; it needs enough structure to know what each document is, when it was produced, and what it is about.

### Document hygiene

Each document should carry:

- **Document ID**: a stable identifier. The same document re-uploaded keeps the same ID.
- **Document date**: the publish date, in ISO 8601. This is the anchor for temporal reasoning.
- **Context**: a curated value describing the layer or purpose of the document, such as impact assessment, component design, or business capability definition.
- **Tags**: consistent filters such as project, domain, system, or data area.
- **Source metadata**: where the document came from (Confluence, SharePoint, a file path) and who produced it.

architxt includes tagging alignment tools that help assign context, tags, source metadata, and entity identifiers. The time saving comes from the bulk change and consistency checking architxt enables across the whole corpus. It aligns information across dozens or hundreds of documents without requiring each one to be manually updated directly through the UI. A document is only useful in the mosaic if you can locate it again, filter by it, and trust its date; architxt uses that externally curated structure to build a reliable, consistent current-state view.

### Entities are the anchor

Entities are the central unit. A component, a service, a capability: each becomes a stable reference point that observations from different documents can attach to. Without entities, a document is just a bag of text. With entities, a sentence in a 2020 design and a paragraph in a 2024 migration document can both refer to the same thing, and architxt can keep the latest view of that thing intact.

In practice, the same entity is rarely called the same thing in every document. One document might say "Billing Engine", another "Billing Service", another "BE". Aliases let architxt map these varied names back to a single entity. The more aliases are known, the more complete the entity timeline becomes.

To make this work across time, entities are embedded into documents with a stable ID, using a lightweight tagging convention such as `Billing Engine (SYS-001)`. The human name can change ("Billing Engine" might become "Billing Platform" in a later design) but the stable ID survives. architxt then resolves the current name against the stable ID, so references from older documents remain usable even after naming conventions shift.

![The entity namespace in architxt: stable IDs, aliases, and the documents each entity appears in.](/img/blog/architxt-entities.png)

This is why the minimum viable input matters. Good metadata and tags make recall precise. A stable, aliased entity namespace makes composition across documents and across years possible. The rest, mental models, reflections, and the temporal mosaic, builds on top of that foundation.

## What Hindsight adds

So, we have the data and have a basic set of metadata. We know there's a goldmine of information, that probably cost thousands or millions of dollars to get written. How do we process and store it in a way that accommodates the variance in source text?

Hindsight stores the extracted knowledge as a durable, time-aware memory bank. Rather than replacing documents with a single summary, it keeps observations as discrete entries that include when they were captured and where they came from. This matters because the temporal mosaic is only possible when you can ask "what do we most recently know about X?" instead of "what does the latest document say?"

Two Hindsight primitives make this work:

- **Retain / Observation**: when architxt extracts facts from a document, they are retained as timestamped, source-referenced facts, and then consolidated into observations. A new document about the same entity does not overwrite the old one; it adds newer facts, and the observation is re-consolidated. The mosaic can then prefer the latest observation per facet while still keeping older ones visible where nothing newer exists.

- **Reflect / Mental models**: mental models are reusable prompts that run over the current set of memories. They can be refreshed as new documentation is added, so summaries, capability lists, and interface descriptions stay current; auto-refresh after consolidation is opt-in. The output is tied to the observations it was based on, so the generated view remains grounded and traceable.

Together, retain and reflect mean the mosaic updates incrementally. New documents are ingested, facts are retained, and mental models are re-run. The "current state" is not rebuilt from scratch; it is the latest layer of a continuously updated knowledge stack.

![Comparing architxt's mental models against a remote Hindsight bank: only on architxt, on both, or only on the bank.](/img/blog/architxt-hindsight-compare.png)

## The Temporal Mosaic: current state without a single source of truth

This combination matters because most architecture tools force one of two models:

1. **Static models**: draw a diagram once, watch it rot.
2. **Designed-vs-delivered reconciliation**: try to maintain two parallel realities and merge them.

architxt takes a third path, using Hindsight as its durable memory layer. The "current state" is a mosaic: the latest reliable knowledge for each entity, sourced from whichever document last touched it. A component from a 2020 design sits next to a service from a 2024 migration document. Seams (contradictions, outdated interfaces, orphaned dependencies) surface only when a query crosses them.

This is the **Temporal Mosaic**. It accepts that organisations do not produce one consistent architecture document. They produce a stream of partial, dated, overlapping documents. Rather than flattening them into a single model, architxt uses Hindsight to make them queryable as a composite, with answers grounded in source documents and tagged with entities so they carry provenance rather than relying on model hallucination.

## What this looks like in practice

Once the documents are tagged and ingested, the question changes. Instead of "which document might have the answer?", you can ask direct questions against the corpus.

For example:

- "What are the integration points between the billing service and the customer platform?"
- "Which capabilities does the order processing system support, and which documents describe them?"
- "Has the data model for customer records changed since the 2022 platform migration?"

![A grounded capability summary for one entity, synthesised from the source documents and organised into facets such as capabilities and interfaces.](/img/blog/architxt-grounded-answer.png)

Hindsight returns grounded answers. Each claim is tied back to the document and observation it came from, so you can verify it rather than trust a generated summary. If two documents disagree, that disagreement is surfaced rather than smoothed over. This turns document search from a guessing game into a structured query.

## In short

The real cost of fragmented architecture knowledge is not the documents themselves. It is the time people spend trying to reconstruct what those documents mean when taken together.

architxt reduces that cost by turning documents into structured, entity-tagged observations. Hindsight keeps those observations alive over time. The Temporal Mosaic is the result: a current-state view that does not pretend the organisation ever produced a single authoritative description, but still makes the combined knowledge searchable, verifiable, and current.

That is the shift. Not more documents, or better diagrams, but a way to extract value from the documents already in place and provide a pathway for keeping that state current as new documents get written. The format and scope of future design documents is unknown, but they will contain information worth leveraging. architxt, coupled with Hindsight, is built to make that possible.
