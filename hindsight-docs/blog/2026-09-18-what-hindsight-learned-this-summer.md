---
title: "What Hindsight Learned This Summer"
authors: [benfrank241]
slug: "2026/09/18/what-hindsight-learned-this-summer"
date: 2026-09-18T14:00
tags: [hindsight, agent-memory, release, knowledge-pages, coding-agents, roundup]
description: "Six releases in eight weeks, from 0.8.5 to 0.10.0. The biggest things Hindsight learned to do this summer, grouped by what they let you build."
image: /img/blog/what-hindsight-learned-this-summer.png
hide_table_of_contents: true
---

![What Hindsight learned this summer: six releases, from 0.8.5 to 0.10.0](/img/blog/what-hindsight-learned-this-summer.png)

Between late July and mid-September, Hindsight shipped six releases: 0.8.5, 0.8.6, 0.9.0, 0.9.1, 0.9.2 and 0.10.0. Each one got its own release notes, which is great if you read every one and less great if you didn't.

So this is the other view. Not what changed in which version, but what Hindsight can do now that it couldn't at the start of the summer.

<!-- truncate -->

## TL;DR

- **Memory writes its own documentation.** Knowledge pages turn a bank into a living wiki, reachable over MCP and portable between instances.
- **One memory for every coding agent.** A single install wires 18 coding agents into one shared bank per repository.
- **Memory can see.** Images and files are first-class retain content, and each fact can cite the screenshot it came from.
- **Memory you can inspect.** Preview the exact prompt, dry-run a mental model refresh, and browse everything the bank knows about one entity.
- **Memory that understands time.** Pass recall a time window directly, and reflect now knows what day it is.
- **Memory that costs less to run.** Batch discounts, prompt caching, per-operation reasoning effort and a much cheaper request path.

## Memory that writes its own documentation

The biggest idea of the summer arrived in [0.9.0](https://hindsight.vectorize.io/blog/2026/08/07/version-0-9-0): **knowledge pages**.

A knowledge page is a living document a bank writes about itself, answering one question like "What's our error-handling convention?" and rewriting itself as the bank learns more. Pages sit in a folder tree like a wiki, and `hindsight fs mount` projects that tree onto disk as real markdown files, so `ls`, `grep`, your editor and an agent's file tools all just work.

The part that makes it more than a wiki: a page is a projection, not storage. Delete one and nothing is lost, because it's rebuilt from memory that's already been extracted, deduplicated and reconciled. A page reflects what holds *now*, not every contradictory thing anyone ever said.

Two releases later, [0.9.2](https://hindsight.vectorize.io/blog/2026/08/24/version-0-9-2) made the knowledge base a native MCP surface: seven tools for browsing, searching and maintaining pages, so an agent can keep its own wiki over the same connection it already uses to retain and recall. And [0.10.0](https://hindsight.vectorize.io/blog/2026/09/14/version-0-10-0) made the whole synthesized layer portable: knowledge pages and mental models now travel with a whole-bank export instead of being rebuilt from scratch on the other side.

## One memory for every coding agent

The coding-agents plugin now wires memory into **18 coding agents** with one command:

```bash
npx @vectorize-io/hindsight-coding-agents install all
```

Claude Code, Codex CLI, Cursor CLI, GitHub Copilot CLI, opencode, Qwen Code, Cline CLI, Devin CLI and the rest all share **one bank per repository** by default. What you explained to one agent this morning is there when you open a different one this afternoon, because the memory belongs to the project, not the tool.

The premise behind it is worth repeating: most of a real fix is derivable from the code, but the last mile often hinges on a decision that isn't in the code at all, like a rounding rule or a retry policy. Those live in git history and past conversations, and the plugin puts them in front of the agent when it starts working, alongside knowledge pages it keeps current about your architecture and conventions.

[Four more agents joined in early September](https://hindsight.vectorize.io/blog/2026/09/02/coding-agents-050-four-new-harnesses), and [Knowledge Pages for Coding Agents](https://hindsight.vectorize.io/blog/2026/08/13/knowledge-pages-coding-agents) covers the curated layer in depth.

## Memory that can see

Until 0.10.0, a screenshot in a support thread was invisible to memory. The request ID printed in the image didn't exist as far as your agent was concerned.

Now `content` can be an ordered list of text, image and file blocks. An image is read in the position it occupies, so a chart between two paragraphs is read as the chart between those two paragraphs. And provenance is per fact: recall can return a fact alongside the exact screenshot it was extracted from, not just every image in the chunk.

The design decision worth knowing is what happens when attribution is uncertain. The extractor is asked which image each fact needed, and an answer that doesn't resolve is dropped rather than rounded to the nearest image. A fact with no citation beats a fact with the wrong one. [The full story is here](https://hindsight.vectorize.io/blog/2026/09/16/screenshot-agent-memory).

## Memory you can inspect

A recurring theme this summer: less guessing about what your memory is actually doing.

- **See the prompt before you spend a token.** The preview endpoint in 0.10.0 returns the exact messages retain, consolidation or reflect would send, split into blocks that name the setting behind each one, with no LLM call and no writes. It's how you find out, for example, that your retain mission lands in the user message, not the system prompt. [More on that here](https://hindsight.vectorize.io/blog/2026/09/17/prompt-preview).
- **Debug a mental model without touching it.** 0.9.0 added a dry-run refresh that computes the next version without persisting it, and keeps the trace of what it read and decided. 0.9.2 added a typed outcome on every refresh, so a refresh that produced identical content says so instead of claiming a write.
- **Everything about one entity, in order.** [0.8.6](https://hindsight.vectorize.io/blog/2026/07/29/version-0-8-6) added an entity timeline: pick a person, company or project and see everything the bank knows about it from the first mention onward. It's backed by an API filter, so your own tools get the same view.

## Memory that understands time

Two changes made time a first-class input rather than something inferred from prose.

[0.9.1](https://hindsight.vectorize.io/blog/2026/08/14/version-0-9-1) gave reflect the current date and time, so "last week" and "recently" resolve against the actual clock instead of staying ambiguous.

0.9.2 let you hand recall a `temporal_window` with an explicit start and end. Before, the only way to constrain time was to phrase it in the query and hope the date parser agreed. Now a date picker, or an agent that has already resolved "last quarter" against a real calendar, can just say the range. Setting it also skips the date extraction it replaces, which saves up to around 1.3 seconds on document-sized queries.

## Memory that costs less to run

The least glamorous work of the summer, and possibly the most valuable.

| Change | Release | What it saves |
|---|---|---|
| Anthropic Message Batches | 0.8.5 | 50% token discount on eligible retain and consolidation workloads |
| Anthropic prompt caching | 0.8.5 | Repeated prompt prefixes stop paying full price on every call |
| Per-operation reasoning effort | 0.8.6 | Pay for reasoning where it helps, like extraction, and skip it where it doesn't |
| Retain memory budget | 0.9.2 | Sizing a 45 MB document used to allocate 385 MB before a single fact existed; that's now flat across document sizes |
| Pure ASGI request path | 0.10.0 | Health-check throughput went from 2,476 to 7,917 requests per second on two CPUs |
| Token counting without materializing IDs | 0.10.0 | One recall's counting stages dropped from 34.2 ms to 5.0 ms |

None of these change what your agent remembers. They change what it costs to remember it, which matters more the more you run it. The [retain memory budget post](https://hindsight.vectorize.io/blog/2026/08/27/retain-memory-budget) goes deep on the memory ceiling.

## Where to go from here

Everything above is in v0.10.0, which runs on Hindsight Cloud and on self-hosted deployments alike. The individual release notes have the full detail, including a few breaking changes in 0.10.0 worth reading before you upgrade.

- [0.10.0](https://hindsight.vectorize.io/blog/2026/09/14/version-0-10-0): attachments, prompt preview, faster request path
- [0.9.2](https://hindsight.vectorize.io/blog/2026/08/24/version-0-9-2): knowledge base over MCP, time windows, retain memory budget
- [0.9.1](https://hindsight.vectorize.io/blog/2026/08/14/version-0-9-1): current-time reflect, portable transfers
- [0.9.0](https://hindsight.vectorize.io/blog/2026/08/07/version-0-9-0): knowledge pages, redesigned control plane
- [0.8.6](https://hindsight.vectorize.io/blog/2026/07/29/version-0-8-6): entity timeline, per-operation model control
- [0.8.5](https://hindsight.vectorize.io/blog/2026/07/21/version-0-8-5): batch API, prompt caching, resilient model output

## FAQ

**Do I have to upgrade through every version?**
No. The release notes flag the breaking changes, and 0.10.0 has several worth reading first, including removed bank profile endpoints and `curl` no longer being in the Docker images.

**Are these features on Hindsight Cloud?**
Yes. Cloud runs 0.10.0, so everything listed here is available there, and the knowledge base has been since the [Cloud June to August roundup](https://hindsight.vectorize.io/blog/2026/09/01/hindsight-cloud-june-august-updates). Everything listed here is available today if you self-host v0.10.0.

**Is the coding-agents plugin part of the core release?**
It ships separately as `@vectorize-io/hindsight-coding-agents`, and works with Hindsight Cloud, a server you run, or a local daemon.
