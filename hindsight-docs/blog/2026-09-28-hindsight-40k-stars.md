---
title: "40,000 Stars, and the Number We Actually Watch"
authors: [benfrank241]
slug: "2026/09/28/hindsight-40k-stars"
date: 2026-09-28T16:00
tags: [hindsight, agent-memory, open-source, milestone, community]
description: "Hindsight just crossed 40,000 GitHub stars in eleven months. The stats behind it, the 45 days it took to double, and the one number that tells us more than stars do."
image: /img/blog/hindsight-40k-stars.png
hide_table_of_contents: true
---

![Hindsight crosses 40,000 GitHub stars](/img/blog/hindsight-40k-stars.png)

Hindsight just crossed **40,000 stars on GitHub**.

We wrote a post at [20,000](https://hindsight.vectorize.io/blog/2026/08/14/hindsight-20k-stars) that walked the whole timeline version by version. That first 20,000 took about ten months.

The second 20,000 took 45 days.

We are not launching anything today, so this is just the numbers, what landed while they moved, and a thank you.

<!-- truncate -->

We also made a silly little video for the occasion, in which a coding agent and its memory go for a walk in the park and work out what a knowledge page is for.

<div style={{position: "relative", paddingBottom: "56.25%", height: 0, margin: "1rem 0 2rem", borderRadius: "8px", overflow: "hidden"}}>
  <iframe
    style={{position: "absolute", top: 0, left: 0, width: "100%", height: "100%"}}
    src="https://www.youtube.com/embed/mTpSHe41WSU"
    title="One Memory. Every Agent. — Hindsight hits 40,000 stars"
    frameBorder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowFullScreen
  ></iframe>
</div>

## Where things stand

| | |
|---|---|
| Stars | **40,000** |
| Forks | 5,280 |
| Contributors | 263 |
| Merged pull requests | 2,691 |
| Releases | 70 |
| Age | 333 days |
| License | MIT |

That works out to a release every 4.8 days, and roughly eight merged pull requests a day, sustained for eleven months.

## The one we actually watch

Stars are a bookmark. They cost nothing and they tell you someone saw you once.

The number we pay attention to is **forks: 5,280, or about 13% of stars.** For an infrastructure project that ratio is the interesting one, because forking is what people do when they intend to run something, read it properly, or change it. A repo can trend without that number moving. Ours moved with it.

263 people have had a commit merged. That is the number behind the release cadence, and it is not a number you can buy attention for.

![The 263 people who have had a commit merged into Hindsight](/img/blog/hindsight-40k-contributors.jpg)

## What landed in those 45 days

The doubling happened to overlap with the busiest stretch of shipping we have had. Since the 20,000 post: **24 releases**, 678 merged pull requests, and four core versions.

- **[0.9.1](https://hindsight.vectorize.io/blog/2026/08/14/version-0-9-1)** gave reflect the current date and made transfers portable.
- **[0.9.2](https://hindsight.vectorize.io/blog/2026/08/24/version-0-9-2)** put the knowledge base on MCP, added explicit time windows to recall, and flattened the retain memory ceiling.
- **[0.10.0](https://hindsight.vectorize.io/blog/2026/09/14/version-0-10-0)** made images and files first-class memory, added a prompt preview so you can see exactly what an operation would send, and rebuilt the request path.
- **[0.10.1](https://hindsight.vectorize.io/blog/2026/09/21/version-0-10-1)** added the [Jev reranker](https://hindsight.vectorize.io/blog/2026/09/24/adding-jev-reranker-what-we-learned) and made a [bank something you can pick up and move](https://hindsight.vectorize.io/blog/2026/09/22/move-a-memory-bank).

The integrations moved faster than the core did, which is usually the sign that people are actually wiring this into things.

## Thank you

To everyone who filed an issue with a real reproduction, sent a pull request, argued with a design decision in a thread, or wrote up their own migration so the next person had something to read: thank you. Several of the sharper things in the last two releases exist because someone outside the team pointed at something and said *this is wrong*.

If you want to be part of the next stretch, the [repository](https://github.com/vectorize-io/hindsight) is the place, and good first issues are labelled as such.

## What's next

No roadmap promises. The honest version is that the work we are most interested in right now is the unglamorous half: making recall cheaper, making what the system believes easier to inspect, and making it harder to end up with a bank full of things you did not mean to keep.

If you are running Hindsight and something about it annoys you, that is the most useful thing you can tell us.

## Learn more

- [20,000 Stars: How Hindsight Got Here, Version by Version](https://hindsight.vectorize.io/blog/2026/08/14/hindsight-20k-stars) for the full timeline
- [What Hindsight Learned This Summer](https://hindsight.vectorize.io/blog/2026/09/18/what-hindsight-learned-this-summer) for the capability arc rather than the version list
- [The repository](https://github.com/vectorize-io/hindsight)
