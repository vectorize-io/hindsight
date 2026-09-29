---
title: "See the Exact Prompt Before You Spend a Token"
authors: [benfrank241]
slug: "2026/09/17/prompt-preview"
date: 2026-09-17T14:00
tags: [hindsight, agent-memory, prompts, observability, retain, reflect, release]
description: "You configured a mission, a disposition and three strategies. What did the model actually receive? Hindsight 0.10.0 will show you, without spending a token."
image: /img/blog/prompt-preview.png
hide_table_of_contents: true
---

![See the exact prompt an operation would send, split into the settings that produced it](/img/blog/prompt-preview.png)

You write a mission for your memory bank. Something like "focus on billing decisions and who approved them." You save it, and the extraction gets better, or it doesn't.

Now: where did that sentence go?

If you guessed the system prompt, you're wrong for two of the three operations that use it.

<!-- truncate -->

## TL;DR

- A retain or consolidation mission lands in the **user** message, not the system prompt. Reflect's lands in the system prompt, where it replaces the agent's role line rather than adding to it.
- `POST /banks/{id}/prompts/preview` returns the exact messages retain, consolidation or reflect would send. No LLM call, no writes.
- Active blocks concatenate byte-for-byte into the real prompt, and there's a test pinning that against the actual builder.
- A setting you switched off comes back as an inactive block, positioned where it would have landed.
- The endpoint refuses to accept your sample text or a candidate mission, on purpose.
- Shipped in v0.10.0, available on Hindsight Cloud and for self-hosted deployments.

## Why the mission moved

The system prefix for retain and consolidation is deliberately bank-agnostic. Every bank on your server sends the same one, which means one provider-side prompt cache serves all of them. Bake a per-bank mission into that prefix and you get a separate cache per bank, which on high-volume ingestion is a real bill.

So the mission goes in the user message instead. For retain it's a fenced `FOCUS` preamble; for consolidation, a `## MISSION` heading. Reflect is the exception, and it's a different kind of surprise: `reflect_mission` doesn't append anything, it **replaces the role line**. Writing "focus on billing questions" quietly deletes "You are a reflection agent that answers questions by reasoning over retrieved memories."

None of this is visible from the config page. You set a string, and where it lands is a property of the code.

That's the gap the preview closes. The commit that added it says so directly: a bank's missions only mean something once you can see the prompt they land in, and until now that meant tracing Python constants and format calls.

## What it returns

`POST /v1/default/banks/{bank_id}/prompts/preview` takes an operation and gives you back the messages in send order, each split into blocks:

```json
{
  "text": "FOCUS — What to retain for this bank...",
  "source": "config",
  "field": "retain_mission",
  "active": true,
  "kind": "text",
  "editable": true
}
```

Every block says where it came from: `source` is `config` or `builtin`, `field` names the setting behind it, and `editable` tells you whether it's a bank-level setting you can change or a server-level one where a UI offering to edit it "would only collect a 400."

The important guarantee is the one in the code comment: *the active blocks partition it exactly*. Concatenate the active blocks and you have the bytes the model would receive. There's a test that renders a preview and builds a real extraction prompt side by side and asserts they're identical, so the two can't drift.

Both messages always come back, even when one is empty, because a system-prompt-only view would show your configured mission as absent. Which, as the module docstring puts it, is exactly backwards.

## Absence you can see

The part I find genuinely well designed is what happens to a setting that's switched off.

It still comes back. `active: false`, empty text, positioned at the point in the message where it *would* land if you turned it on. And since empty text can't be located by matching against the prompt, the renderer anchors it to the last fragment that did match, because a switched-off setting floated to the end of the response says nothing.

The reasoning, from the source:

> the mission you have not written yet is the one you came here to write, and it has no text to sit beside

Two refinements stop that from becoming noise. A setting gets no off-block where the slot doesn't exist at all: `retain_custom_instructions` isn't rendered outside `custom` mode, because the builder never reads it there. And a consolidation mission is never inactive, because leaving it unset just means the built-in default fills the slot.

## It won't take your sample text

Here's the design decision most people will push back on. The request body has two fields: which operation, and optionally which retain strategy. You cannot pass a candidate mission. You cannot pass your own sample text.

The docstring argues the case rather than just documenting it:

> The operation is the whole request: everything that shapes the prompt comes from the bank. There is deliberately nothing to override. A preview answers "what does this bank send"; letting a caller pass its own mission or sample text only moved that question somewhere the bank cannot answer it. To try a candidate value, save it and look again.

Which is why every block carries `editable`. The workflow isn't "simulate a change," it's "make the change, look again." The control plane builds exactly that loop: edit the setting behind a block inline, save, and the preview refetches.

If you do want to see what comes *out*, that's a different endpoint. The retain tester calls a dry-run extraction, which spends a real LLM call, stores nothing, and unlike the preview does accept a small allowlist of overrides. One shows what goes in for free; the other shows what comes back, for the price of a call.

## Things you can only learn by looking

The preview is useful because prompt assembly has sharp edges. Every one of these is real, and none is visible from a config page.

| What you set | What actually happens |
|---|---|
| A strategy name with a typo | Logged as a warning, not an error. Unknown fields inside a strategy are silently ignored. |
| `retain_custom_instructions` outside `custom` mode | Never read. The setting does nothing. |
| `custom` mode with empty instructions | Silently falls back to `concise`. |
| `entities_allow_free_form` without `entity_labels` | Does nothing. It's one sentence inside the labels section, which only exists when labels are configured. |
| `llm_output_language` | Replaces the default language rule rather than adding to it, because the two contradict each other. |
| `chunks` extraction mode | No prompt is sent at all. Mission, labels and custom instructions all go inert. |

The one that convinced me this endpoint had to exist is in the feature's own commit. Both the dry run and the preview were resolving config directly instead of through the retain resolver, so a bank with a default strategy was being previewed under settings a real retain would never use. The tool built to show you the truth was, briefly, lying in the same way everything else was. There are regression tests for it now.

## What it costs and what it touches

Nothing, and nothing.

No LLM client, queue or lock is touched anywhere in the renderer. No bank is created if one is missing, no operation row is written, and the route carries no audit decorator. A test asserts the memory-unit count is unchanged after a preview.

One access note worth knowing: the preview sits behind the same permission as reading the bank's config, not a general read. Rendering settings as prompt text is the same disclosure by another route, so if an extension denies config reads, it denies previews too. There's also a dedicated test that no provider credential can leak into the response, since the renderer works from the fully resolved internal config.

## What it doesn't show

Worth knowing before you treat a preview as the whole truth.

- **Some runtime sections are omitted.** The retain preview renders the chunk, event date and context, but not the metadata, narrator or attachment instructions a real retain includes. For an image-heavy bank that's a meaningful chunk of text you won't see.
- **Reflect is rendered at maximum.** Every tool available, untagged directives, no per-request budget guidance. A real reflect with different options sends something different.
- **Consolidation's capacity note never appears.**
- **No CLI, no MCP.** It's an HTTP endpoint with generated client methods; the CLI waiver says the multi-block payload is a UI shape, not useful CLI output.
- **Block attribution is best effort.** If a builder reworded a fragment, that block is skipped and its text stays inside the surrounding built-in block. The concatenation guarantee holds either way, so the prompt is always right even when the labelling degrades.

## FAQ

**Does previewing cost tokens?**
No. It never calls a model. The dry-run extraction in the control plane's tester does, and it's labeled as the paid half.

**Can I preview an unsaved change?**
No. Save it and look again. The response marks which blocks are editable so you know what you can change.

**Why are there two messages when the system one looks empty?**
Because for retain and consolidation your mission is in the user message. Showing only the system prompt would make a configured mission look like it wasn't set.

**What's the difference between this and the Extraction Tester?**
Same dialog, different halves. Preview shows what would be sent, for free, for all three operations. The tester runs a real retain extraction against sample text, for retain only, and stores nothing.

## Learn more

- [What's new in Hindsight 0.10.0](https://hindsight.vectorize.io/blog/2026/09/14/version-0-10-0) covers the rest of the release
- [How We Built Disposition-Aware Agents](https://hindsight.vectorize.io/blog/2026/03/13/disposition-aware-agents) explains the traits that show up near the bottom of a reflect prompt
- [Inside retain()](https://hindsight.vectorize.io/blog/2026/07/13/inside-retain-agent-memory) is the write path these prompts drive
- [What It Takes to Put a Screenshot in an Agent's Memory](https://hindsight.vectorize.io/blog/2026/09/16/screenshot-agent-memory) covers the attachment instructions the preview leaves out
- [Stop Growing Your Always-On Context](https://hindsight.vectorize.io/blog/2026/09/16/stop-growing-your-system-prompt) on deciding what deserves to be sent every turn
