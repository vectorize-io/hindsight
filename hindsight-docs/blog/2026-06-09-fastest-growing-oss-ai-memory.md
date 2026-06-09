---
title: "Hindsight Is the Fastest-Growing Open-Source AI Memory Project Ever"
description: "Hindsight — the agent memory that learns — is the fastest-growing open-source agent-memory project ever, and it isn't close. The data: per-star timestamps, equal-age comparisons, a screen for bought stars, and independent confirmation from OSSCAR and a security firm's scan of real enterprise traffic."
slug: "2026/06/09/fastest-growing-oss-ai-memory"
date: 2026-06-09T12:00
authors: [cdbartholomew]
image: "/img/blog/fastest-growing-oss-ai-memory.png"
tags: [release]
---

![Hindsight is the fastest-growing open-source AI memory project ever](/img/blog/fastest-growing-oss-ai-memory.png)

# Hindsight Is the Fastest-Growing Open-Source AI Memory Project Ever

Hindsight — the agent memory that learns — is the fastest-growing open-source agent-memory project ever. This post lays out the evidence: our own analysis of GitHub's per-star data, and independent confirmation from two third parties who have no stake in the claim.

First, a word about why "fastest-growing" is harder to prove than it sounds — because the most obvious way to measure it is also the most misleading.

<!-- truncate -->

Go looking for AI memory projects on GitHub and you'll find one that *appears* to tower over everything else: north of 50,000 stars, accumulated in roughly two months. Case closed? Not quite. GitHub records a timestamp on every star, and when you pull them, that project's story falls apart. The bulk of its stars — tens of thousands of them — arrived within a few days of the repository being created, including **more than 16,000 in a single day**, before it had any audience at all. Real projects don't grow like that. Stars, it turns out, can be purchased, and a vertical wall of them on a brand-new repo is exactly what a purchase looks like.

So raw counts alone prove nothing. To make this claim honestly, we need measures that reflect real developers making a real choice — and that's exactly what holds up under every one we apply.

---

## Start simple: who gains stars fastest?

The simplest honest measure is total stars divided by how long a project has existed. Here's every major open-source agent-memory project — the legitimately-grown ones — by that number.

| Project | Stars | Age | Stars/day (lifetime avg) |
|---|---:|---:|---:|
| **Hindsight** | **16,035** | **7.3 mo** | **72.2** |
| Mem0 | 58,168 | 35.6 mo | 53.6 |
| Graphiti | 27,213 | 22.0 mo | 40.7 |
| Supermemory | 26,303 | 27.3 mo | 31.6 |
| Letta / MemGPT | 23,226 | 31.9 mo | 23.9 |
| Cognee | 17,740 | 33.7 mo | 17.3 |
| Memobase | 2,747 | 21.2 mo | 4.3 |
| Zep | 4,654 | 37.4 mo | 4.1 |

Hindsight has the highest star velocity in the category — ahead of Mem0, which went on to become the largest agent-memory project on GitHub, and more than double Supermemory's pace. And it's doing it while being **three to five times younger** than everything else on the list. That's a promising first answer. But it's not yet a careful one.

---

## The harder question: are we comparing fairly?

A lifetime average can flatter a young project. Viral repositories tend to earn most of their stars in an early burst and then coast, so a project that's been around three years carries a long, slow tail that drags its average down. Measure Hindsight's seven hot months against Mem0's three-year average and you might be comparing a sprint to a marathon's pace — not a fair fight in either direction.

The honest way to settle it is to compare every project at the *same* point in its life. The same per-star timestamps that exposed the bought-star project let us reconstruct exactly how many stars each project had when it was the age Hindsight is today — 222 days.

| Project | Stars at day 222 | Stars today |
|---|---:|---:|
| **Hindsight** | **16,035** | 16,035 |
| Letta / MemGPT | 9,696 | 23,226 |
| Mem0 | 6,942 | 58,168 |
| Supermemory | 6,389 | 26,303 |
| Graphiti | 2,383 | 27,213 |
| Zep | 1,505 | 4,654 |
| Cognee | 344 | 17,740 |

This is where the answer sharpens. At the same age, Hindsight has **1.6× the stars of the next-fastest project** — Letta, which launched as the viral MemGPT — and **2.3× Mem0's**. Mem0 eventually reached 57,000 stars, but at the seven-month mark where Hindsight stands today, it had fewer than 7,000. Every project in the category had a smaller audience at this age than Hindsight does now. There's no early spike hiding underneath the average. This *is* the spike — and it's the biggest one the category has produced through real adoption.

---

## A project still accelerating, not coasting

It would be easy to assume Hindsight only squeaked ahead in the final weeks. The month-by-month numbers say the opposite. Here is the cumulative star count for each project at 30-day intervals from the day its repository was created.

| Day | 30 | 60 | 90 | 120 | 150 | 180 | 210 | 222 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Hindsight** | 2 | 842 | 1,191 | 1,921 | 6,426 | 11,175 | 15,042 | 16,035 |
| Letta / MemGPT | 5,618 | 6,344 | 6,814 | 7,350 | 7,652 | 8,051 | 9,277 | 9,696 |
| Mem0 | 3,382 | 3,662 | 4,761 | 5,293 | 5,591 | 5,779 | 6,723 | 6,942 |
| Supermemory | 0 | 2,296 | 2,738 | 2,949 | 3,853 | 5,218 | 6,181 | 6,389 |

Hindsight didn't lead from day one — and that's the most encouraging part of the picture. For the first four months it was a sleeper, under 2,000 stars while Letta and Mem0 sat comfortably ahead. Then, between day 120 and day 222, it added more than **14,000 stars in about 100 days** — a sustained run of roughly **150 stars a day at its peak**. That's the steepest stretch of organic growth the category has recorded, and the curve hasn't bent back down. While its competitors had already flattened at this age, Hindsight is still climbing.

And note the contrast in *shape*. The bought-star project's growth was a single vertical wall in week one. Hindsight's is a months-long ramp that gets steeper as more people use it — the signature of real word-of-mouth, not a one-time transaction. That inflection tracks with the moment production deployments started working, the integration ecosystem filled out, and the [BEAM benchmark results](https://hindsight.vectorize.io/blog/2026/04/02/beam-sota) gave teams a reason to trust agent memory enough to build on it. People didn't star Hindsight on launch-day hype. They starred it after they used it.

The picture is starkest in absolute time. Hindsight is the red line in the bottom-right corner — a near-vertical climb to 16,000 stars, the steepest slope on the chart, while every older and larger project rises gradually:

[![GitHub star-history comparison showing Hindsight with the steepest slope of any agent-memory project, despite being the youngest line by years](/img/blog/star-history-comparison.png)](https://star-history.com/#vectorize-io/hindsight&mem0ai/mem0&supermemoryai/supermemory&letta-ai/letta&getzep/graphiti&topoteretes/cognee&Date)

---

## How we measured this

So you can reproduce it:

**Data source.** GitHub's stargazers API returns a timestamp for every star (`Accept: application/vnd.github.star+json`). We paged each repository's stargazers oldest-to-newest and counted how many fell within the first 222 days after the repo was created. Pulled June 9, 2026. None of the legitimately-grown repositories hit the API's pagination cap, so their early-window counts are complete. (For the bought-star project, the cap is itself the tell: all 40,000 retrievable stars fall inside week one.)

**Why 222 days.** That's Hindsight's exact age at the time of writing (repo created October 30, 2025). Every other project is measured against the same 222-day window from its own creation date.

**Projects compared.** The most prominent open-source agent-memory projects we could find: Mem0, Letta/MemGPT, Supermemory, Graphiti, Zep, Cognee, and Memobase. We excluded any project whose star history shows non-organic bulk acquisition — tens of thousands of stars on a single date is trivial to spot once you have the timestamps, and we're measuring real developer interest, not stars you can buy. If there's a legitimately-grown project we missed that reached ~16,000 stars in under seven and a half months, we want to know — open a discussion and we'll update this post.

A few details worth naming. Mem0's repository began life as *embedchain* before it was renamed, so its first 222 days partly reflect a different project's early audience — which only flatters Mem0's numbers, and Hindsight still leads by 2.3×. Letta's repository was originally *MemGPT*, which went viral on release, making it the strongest honest comparison on the list; Hindsight still beats its day-222 count by 1.6×. And Supermemory isn't actually open source in the way Hindsight is — its core product is closed, and the public repository is a companion to a proprietary service. We include it because its stars are still a meaningful proxy for developer interest in that product, but it's worth being clear that it's not a like-for-like open-source comparison: Hindsight's stars measure adoption of the actual thing you run, not interest in a closed system behind it.

One last note on what stars do and don't mean: they measure mindshare and momentum, not revenue or production usage. This is a growth claim. For the adoption story, see the [10,000-stars retrospective](https://hindsight.vectorize.io/blog/2026/04/22/hindsight-10k-stars).

---

## It isn't just us saying it

We have an obvious stake in this claim, so the part that matters most is that two independent third parties — neither of which we control — landed in the same place.

**OSSCAR ranks Vectorize a top-10 fastest-growing open-source organization.** [OSSCAR](https://osscar.dev/) is a quarterly ranking of the fastest-growing open-source orgs, produced by **Supabase and >commit**, scored on composite growth across GitHub stars, contributors, and package downloads. In the Q1 2026 "Scaling" tier, **vectorize-io ranks #10** — out of the tens of thousands of organizations they evaluate, across every category of software, not just memory. And among dedicated AI-memory projects, it's the highest-ranked one on the board: **Mem0 sits at #22**, with Supermemory, Letta, Zep, and Cognee absent from the tier's top 25 entirely. The >commit team [called it out directly](https://x.com/commitvc/status/2056707157840588950): *"#10 Vectorize's Hindsight is agent memory that learns: adaptive memory infrastructure that grows smarter with every agent interaction."*

**A security firm found Vectorize is the #1 MCP server in the wild.** This is the one we like best, because it measures real deployment, not stars. In its report [*"MCP Servers Are the New Shadow IT: 56 Common Domains We Found Hiding in Plain Sight"*](https://dope.security/post/mcp-servers-new-shadow-it-56-domains-hiding-in-plain-sight), the security company **dope.security** scanned roughly 10,000 devices' worth of enterprise network traffic to see which Model Context Protocol servers people are actually connecting to. Their top 10, in order: **1. Vectorize**, 2. Anthropic, 3. Atlassian, 4. Zapier, 5. Context7, 6. Granola, 7. Microsoft 365, 8. Slack, 9. Google Workspace, 10. VS Code ([summary on LinkedIn](https://www.linkedin.com/posts/kunala_mcp-servers-are-the-new-shadow-it-56-common-share-7454629357601525760-3AEf)). Hindsight's MCP server shows up ahead of Anthropic, Slack, and Google Workspace in genuine production environments. Stars measure interest; this measures use — and it points the same direction.

---

## The verdict

By every honest measure, Hindsight is the fastest-growing open-source agent-memory project ever. It gains stars faster than any other by lifetime average; it had more stars at seven months old than any of them did at the same age; it's the only one still accelerating at this point in its life rather than flattening; an independent cross-industry ranking places it ahead of every other memory project; and a security firm's traffic scan finds its MCP server more widely deployed than anyone else's. The one project that *appears* to beat it on raw count bought the number, and the timestamps prove it.

None of that is the same as being the biggest — Mem0, Graphiti, Supermemory, and Letta are all larger today, and they're good projects. This is a statement about trajectory. But trajectory is what tells you where a category is heading, and right now it's heading toward the agent memory that learns.

---

## Why it happened

Fast growth is a symptom, not a cause. A few things drove it:

- **Memory that actually learns.** This is the whole idea behind Hindsight — agent memory that learns. It doesn't just store and retrieve text; it extracts facts, resolves entities, and consolidates them into higher-level mental models over time, so agents get sharper the more they're used instead of just accumulating a bigger pile of history to search.
- **Accuracy you can verify.** Hindsight ranks #1 on [BEAM](https://hindsight.vectorize.io/blog/2026/04/02/beam-sota) ("Beyond a Million Tokens") — 64.1% at the punishing 10M-token tier, versus 40.6% for the next-best system — and, unusually, its accuracy *holds up* as scale grows instead of collapsing. That's a concrete reason to switch: a memory system you can trust at the scales real agents actually hit.
- **MIT-licensed and self-hostable.** No vendor lock-in, no closed black box. Fintech, healthcare, and enterprise teams can run it on their own infrastructure.
- **A real integration ecosystem.** Claude Code, LangGraph, CrewAI, Pydantic AI, Agno, Strands, and more — adopting Hindsight rarely means rearchitecting your stack.
- **Built in the open.** Bugs filed, discussions held, fixes shipped on a steady cadence. The curve bent upward when the community decided the project was worth betting on.

---

## Try it yourself

The fastest way to see what 16,000 developers are excited about is to run it:

```bash
export OPENAI_API_KEY=sk-xxx
docker run --rm -it --pull always -p 8888:8888 -p 9999:9999 \
  -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY \
  -v $HOME/.hindsight-docker:/home/hindsight/.pg0 \
  ghcr.io/vectorize-io/hindsight:latest
```

API at `http://localhost:8888`, web UI at `http://localhost:9999`. Connect it to your agent framework and start extracting facts from conversations.

- **Star [the repo](https://github.com/vectorize-io/hindsight)** — the real kind — and help us find out how fast the *next* milestone arrives
- **Try [Hindsight Cloud](https://ui.hindsight.vectorize.io/signup)** — managed, scaled, usage-based pricing
- **Join the conversation** — [GitHub Discussions](https://github.com/vectorize-io/hindsight/discussions) or [Slack](https://join.slack.com/t/hindsight-space/shared_invite/zt-3nhbm4w29-LeSJ5Ixi6j8PdiYOCPlOgg)

Seven months, two stars to sixteen thousand — every one of them earned, for the agent memory that learns. Thanks for building with us — let's see where the curve goes next.
