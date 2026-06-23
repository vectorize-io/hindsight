---
title: "Structuring Chat Logs for Agent Memory"
authors: [hindsight]
slug: "2026/06/23/structuring-chat-logs"
date: 2026-06-23T12:00
tags: [hindsight, memory, retain, conversations, chat, best-practices]
description: "The shape of your chat logs decides your agent's memory quality. Retain whole conversations, label who is speaking, set context, and anchor in time."
image: /img/blog/structuring-chat-logs.png
hide_table_of_contents: true
---

![Structuring Chat Logs for Agent Memory with Hindsight](/img/blog/structuring-chat-logs.png)

Hindsight doesn't store your chat logs. When you [`retain`](/developer/api/retain) a conversation, it chunks the text, sends each chunk to an LLM for fact extraction, and stores the *facts*, not the original transcript. That single design choice is why the structure of what you send matters: the model can only extract what the text makes clear. A well-shaped transcript yields clean, correctly-attributed memories. A wall of unlabeled text yields guesses.

The good news is there's no schema to learn and no required format. Plain text, JSON, Markdown: anything works, as long as it conveys **who said what, and when**. This post is about how to make that true, and the few high-leverage choices that separate a mediocre ingestion from a great one.

## TL;DR

<!-- truncate -->

- **Retain the whole conversation as one item, not one item per message.** Facts get extracted and cross-referenced with full context, and a stable `document_id` makes re-ingestion idempotent.
- **Document length isn't the constraint.** Hindsight decomposes the entire document into facts, so the tail of a long transcript isn't dropped or down-weighted the way it is in systems that stuff raw text into a context window. Segment by *how soon you need to recall it*, not by size.
- **Label every line with a speaker.** The simplest reliable format is `Name (timestamp): text`. Hindsight uses *who is speaking* to decide whether a statement is a fact about the world or the agent's own experience.
- **Tell Hindsight who the speaker is via `context`.** A context like `"Customer Maria is speaking"` keeps her "I bought a Tesla" stored as a fact about Maria, not mistaken for something the agent did.
- **Anchor it in time.** Pass a real `timestamp` so the model can resolve "last Monday" and so temporal recall works later.
- **Drop the noise.** Strip system prompts and any memories Hindsight injected back into the conversation. They're instructions and echoes, not facts worth remembering.

## The One Rule: Who Said What, and When

Everything else follows from this. Because extraction is an LLM reading your text, the transcript has to make speaker and time legible *to a reader*. The format we recommend in the docs is deliberately boring:

```
Alice (2024-03-15T09:00:00Z): Did you end up going to the doctor last week?
Bob (2024-03-15T09:01:00Z): Yes, finally. Turns out I have a mild peanut allergy.
Alice (2024-03-15T09:02:00Z): Oh no — are you okay?
Bob (2024-03-15T09:03:00Z): Yeah, nothing serious. Just carrying an antihistamine now.
```

`Name (timestamp): text`, one line per turn. That's it. JSON works too if that's what you already have, and the model parses it fine. But plain labeled text is the lowest-friction option, and the easiest to eyeball when you're debugging what got extracted.

## Retain the Whole Conversation, Not Each Message

A common instinct is to call `retain` after every message. Don't. Send the **full conversation as a single item**.

Two reasons. First, fact extraction is better with context: "Yeah, nothing serious" is meaningless on its own and meaningful right after "I have a mild peanut allergy." Splitting the conversation into per-message retains strips exactly the context the model needs. Second, a conversation is a living thing, and it grows. Hindsight is built for that:

- Give each conversation a stable **`document_id`**. Retaining again with the same ID **upserts**: the old version and its memories are deleted, and the updated content is reprocessed from scratch. Memories always reflect the latest state, with no duplicates.
- For transcripts that grow one message at a time, use **`update_mode: "append"`**. Instead of re-sending the entire history, send just the new turns; Hindsight concatenates them onto the existing document and skips re-extracting the unchanged chunks.

So the pattern is: one `document_id` per conversation, re-`retain` (replace) when you have the whole thing, or `append` as messages stream in.

## How Long Is Too Long? Segment by Recall Latency, Not Size

A frequent worry, and a real failure mode in other memory systems, is that the **tail of a long transcript gets missed**. Stuff thousands of messages into a context window and the model quietly under-weights whatever's at the end.

Hindsight doesn't work that way, and this is the heart of what it does for you. It doesn't keep the raw transcript and hope the right part is in view at recall time. It chunks the whole document, extracts the facts from *every* chunk, then categorizes, links, and de-duplicates them into consolidated memory. No part of the conversation is privileged or dropped because of where it sits. **Document length, on its own, is not the thing to optimize.**

So the real question isn't "how big should a document be." It's **how soon do you need to recall what's in it?** That's the axis to segment on:

- If an agent needs to act on something said earlier *today*, don't buffer a day (or a week) of logs into one giant document before retaining, or you'd be unable to recall this morning's detail until tonight's flush. Retain in smaller, timelier units.
- If the material is reference-grade and you won't query it until much later, batching more aggressively is fine.

Segment for **freshness of recall**, not to stay under some length ceiling. The expensive, valuable work is splitting a conversation into facts and consolidating them, so a recall returns the best answer instead of a pile of raw lines. That's exactly what Hindsight is for, and it happens regardless of how long the document is.

### Streaming a live conversation

If you're ingesting close to real time, two practical notes:

- **Buffer a few messages together rather than firing one retain per line.** A batch of turns gives the extractor the context to produce good facts; a lone `"lol"` or `"um"` gives it nothing useful to remember. A small rolling buffer (a handful of turns, or a short time window) is the sweet spot.
- **There's a modest per-user ingest rate limit on Hindsight Cloud**, which buffering naturally keeps you under. If you need higher sustained throughput, batch related items into a single retain call.

## Label the Speaker, and Tell Hindsight Who That Is

This is the highest-leverage part, and it's where naive ingestions go wrong.

Hindsight classifies every extracted fact by *whose perspective it captures*. A fact is an **experience** when the bank's own agent is the one acting or observing ("I patched the auth bug"). It's a **world** fact when it's about someone or something else ("Maria drives a Tesla"). The split is decided by **who is speaking**, not by grammar. A first-person "I bought a Tesla" is an *experience* only if the speaker is the agent. From a customer, it's a *world* fact about the customer.

Speaker labels in the transcript do half the job. The **`context`** parameter does the other half, and it's injected straight into the extraction prompt, so it actively steers attribution:

```python
client.retain(
    bank_id="support-agent",
    content=(
        "Maria (2024-03-15T09:00:00Z): I switched to the Pro plan last month "
        "but I'm still being billed for Basic.\n"
        "Agent (2024-03-15T09:01:00Z): Thanks Maria — I'll fix the billing today."
    ),
    context="Customer Maria is speaking with the support agent",
    document_id="ticket-4471",
)
```

That `context` ensures Maria's first-person statements land as `world` facts about Maria, while the agent's "I'll fix the billing" is recorded as the agent's own experience. The docs put it plainly: providing context consistently is one of the highest-leverage things you can do to improve memory quality. The same sentence means different things in a `"performance review"` versus a `"product roadmap"`: context is what disambiguates.

## Anchor It in Time

Conversations are full of relative time: "last Monday," "the meeting yesterday," "since the launch." The model can only resolve those against an anchor, so pass a real **`timestamp`** (ISO 8601) for when the conversation happened. It gets injected into the extraction prompt as the reference point, and it's what makes later temporal recall (like "what did the customer report last spring?") actually work.

Three forms are accepted:

| `timestamp` value | Behavior |
| --- | --- |
| Omitted / `null` | Defaults to the current time at ingestion. |
| ISO 8601 (e.g. `"2024-03-15T09:00:00Z"`) | Used as the anchor; resolves relative references. |
| `"unset"` | Stored with no time. Use for timeless reference material like docs, books, or fiction, where there's no real event time. |

If your transcript already has per-line timestamps (like the examples above), keep them, since they help the model order events within the conversation. The top-level `timestamp` anchors the conversation as a whole.

## Drop the Noise

Not everything in a chat session is worth remembering. The official integrations converge on two things to strip before retaining, and it's worth copying them:

- **System prompts.** They're instructions to the model, not facts about the user or the world. Retaining them pollutes the bank with memories like "the assistant should be concise."
- **Memories Hindsight injected.** If you recall memories and splice them into the conversation as context, don't then retain that same text, or you'd be re-memorizing the bank's own echoes, compounding them every turn.

What's left, the actual user and assistant turns, tool calls, and tool results, is the signal. The LiteLLM integration, for example, renders exactly that into lines like `USER: ...`, `ASSISTANT: ...`, `TOOL_RESULT: ...`, joined by blank lines, and nothing else.

## Use Metadata and Tags for Everything Else

Two more parameters carry the structured context that doesn't belong in the prose:

- **`metadata`**: arbitrary string key-values like `{"source": "slack", "channel": "engineering", "thread_id": "T123"}`. It's fed into the extraction prompt *and* stored on every resulting memory, so you can filter or link memories back to their source later without a second lookup.
- **`tags`**: visibility scoping. A memory is only returned at recall time if its tags intersect the recall filter, which is what keeps one bank safely serving many users or sessions. Use consistent patterns: `user:<id>`, `session:<id>`, `room:<id>`, `topic:<name>`.

## Links and Attachments

Conversations carry more than text: URLs, files, images. A couple of expectations to set:

- **Links go in as text.** Drop a URL into the content and it's retained as part of the conversation, so the facts around it ("Maria shared the new pricing page") are remembered and the link comes back with the memory. Hindsight does **not** fetch or parse the linked page. The memory is about the link in context, not its contents.
- **Attachments aren't ingested directly.** There's no file-upload interface today. The common pattern is to store the file (e.g. in an S3 bucket) and include the link as reference text alongside the message, plus any caption or description you have. That keeps the attachment discoverable from memory even though its bytes live elsewhere.

If you need the *contents* of files and pages to become memory, that's retrieval-over-documents territory, a different pipeline from conversational memory.

## Putting It Together

A single, well-formed retain for a support conversation:

```python
client.retain(
    bank_id="support-agent",
    content=(
        "Maria (2024-03-15T09:00:00Z): I switched to Pro last month but I'm "
        "still billed for Basic.\n"
        "Agent (2024-03-15T09:01:00Z): Sorry about that, Maria. I've corrected "
        "the billing and credited the difference.\n"
        "Maria (2024-03-15T09:03:00Z): Thank you! Also, can you make my account "
        "email maria@example.com going forward?"
    ),
    context="Customer Maria is speaking with the support agent",
    timestamp="2024-03-15T09:00:00Z",
    document_id="ticket-4471",
    metadata={"source": "zendesk", "ticket_id": "4471"},
    tags=["user:maria", "topic:billing"],
)
```

From that one call, Hindsight extracts world facts about Maria (she was on Basic, switched to Pro, wants her email updated), records the agent's experience (corrected the billing, applied a credit), links the entities, and scopes it all to Maria, ready to recall the next time she gets in touch.

## Recap

| Decision | Do this | Why |
| --- | --- | --- |
| Granularity | One item per **conversation**, not per message | Extraction needs surrounding context |
| Length | Don't optimize for it; segment by **recall latency** | The whole doc is decomposed; the tail isn't dropped |
| Streaming | Buffer a few turns per retain; mind the per-user rate limit | Context beats single noise-only lines |
| Re-ingestion | Stable `document_id`; `replace` for rewrites, `append` for streams | Idempotent, no duplicates |
| Speaker | Label every line: `Name (timestamp): text` | Drives the world vs. experience split |
| Attribution | Set `context` to name the speaker | Keeps a user's "I…" a fact about the user |
| Time | Pass a real `timestamp` | Resolves relative dates; enables temporal recall |
| Noise | Strip system prompts and injected memories | They're instructions and echoes, not facts |
| Structured context | `metadata` for provenance, `tags` for scoping | Filter and link without re-extraction |

You don't need to pre-summarize, pre-chunk, or hand-extract facts. Hindsight does all of that, then consolidates patterns in the background. Your job is just to hand it a transcript that's honest about who spoke, when, and in what situation. Get those four things right and the memory quality takes care of itself.

## Next Steps

- **Retain API:** [Ingesting conversations and the full parameter list](/developer/api/retain)
- **How retain works:** [Fact extraction, entity resolution, and the world vs. experience split](/developer/retain)
- **Recall:** [Retrieving memories, including tag filtering](/developer/api/recall)
- **Try it:** [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup) or [self-host with one Docker command](https://hindsight.vectorize.io/developer/installation)
