---
title: "50,000 Stars, and What We Stand On"
authors: [benfrank241]
slug: "2026/10/15/hindsight-50k-stars"
date: 2026-10-15T12:00
tags: [hindsight, agent-memory, open-source, milestone, community]
description: "Hindsight crossed 50,000 GitHub stars in under a year, and shipped 1.0 in the same week. The numbers, the projects we are built on, and the people who sent the pull requests."
image: /img/blog/hindsight-50k-stars.png
hide_table_of_contents: true
---

![Hindsight crosses 50,000 GitHub stars](/img/blog/hindsight-50k-stars.png)

We hit **50,000 GitHub stars**. The repository is 350 days old.

Seventeen days ago we wrote a post at 40,000 that ended with "no roadmap promises" and a note that the work we were most interested in was the unglamorous half. That still holds. It just turned out the unglamorous half was bigger than we thought, and this week it shipped as 1.0.

<!-- truncate -->

## Where things stand

| | |
|---|---|
| Stars | **50,000** |
| Forks | 6,077 |
| Contributors | 288 |
| Merged pull requests | 3,042 |
| Releases | 73 |
| Age | 350 days |
| License | MIT |

A release every 4.8 days and about nine merged pull requests a day, held for eleven and a half months.

## The one we actually watch

We said at 40,000 that the number we pay attention to is forks, because forking is what people do when they intend to run something rather than bookmark it.

It is **6,077 now, about 12% of stars.** That ratio has barely moved in two months, through a stretch where stars grew faster than at any point in the project's life. We would rather it stayed flat at twelve than spiked while the fork count stood still.

**288 people** have had a commit merged, up 25 since the last post.

## What we are built on

Hindsight is a memory system, which is a polite way of saying it is a thin layer of opinions on top of other people's hard work. The ones that matter most:

- **[Postgres](https://www.postgresql.org/)** stores every memory, every link between them, and every index over both. Open-source Hindsight runs on it today and will keep running on it.
- **[pgvector](https://github.com/pgvector/pgvector)** does the similarity search that recall is built on.
- **[Model Context Protocol](https://modelcontextprotocol.io/)** is why one memory bank can be read by Claude Code, Cursor, a Grok Bot and a dozen other things without us writing a connector for each.
- **[Oracle 23ai](https://www.oracle.com/database/)** carries the deployments that cannot use Postgres, which is more of them than we expected.
- **The open-weight model ecosystem**, which is why a serious memory system can run entirely on your own hardware if you want it to.

None of that is ours. All of it is why the project could be useful eleven months after it started instead of five years after.

## Thank you

To everyone who filed an issue with a reproduction instead of a screenshot. To everyone who sent a pull request to an integration we had never heard of because you were already using it. To the people who argued with a design decision in a thread and were right.

Several of the sharper things in 1.0 exist because someone outside the team pointed at something and said *this is wrong*. That is worth more than the star count, and it is the part you cannot manufacture.

## What's next

Two things landed alongside this number.

**[Hindsight 1.0](https://github.com/vectorize-io/hindsight/releases/tag/v1.0.0).** A year in production, more than 30,000 users on Hindsight Cloud, and first place on every dataset of the Agent Memory Benchmark. The architecture and the API are stable, and from here the API is a promise. It stays open source, it stays free, it keeps running on Postgres.

**[Memlake](https://hindsight.vectorize.io/blog/2026/10/13/memlake).** The other thing 50,000 stars buys you is people running this at sizes we did not design for. Memlake is a store for that: 100 million memories at under 100ms p99, with object storage as the only source of truth. It is an enterprise option for deployments that outgrew a single database, and it does not change anything about the open-source project. If your banks are under about 50,000 memories each, Postgres is still the right answer and we will keep saying so.

The unglamorous half is still the interesting half. Recall that costs less. Memory you can inspect and argue with. Fewer things in a bank that you did not mean to keep.

Thanks for all the stars.

The [repository](https://github.com/vectorize-io/hindsight) is where the next stretch happens, and the good first issues are labelled.
