---
title: "50,000 GitHub Stars"
authors: [benfrank241]
slug: "2026/10/15/hindsight-50k-stars"
date: 2026-10-15T12:00
tags: [hindsight, agent-memory, open-source, milestone, community]
description: "Hindsight crossed 50,000 GitHub stars in 350 days, and shipped 1.0 and Hindsight Memlake in the same week."
image: /img/blog/hindsight-50k-stars.png
hide_table_of_contents: true
---

![Hindsight crosses 50,000 GitHub stars](/img/blog/hindsight-50k-stars.png)

We hit **50,000 GitHub stars**. The repository is 350 days old.

Two other things landed the same week.

**[Hindsight 1.0](https://github.com/vectorize-io/hindsight/releases/tag/v1.0.0).** A year in production, more than 30,000 users on Hindsight Cloud, first place on every dataset of the Agent Memory Benchmark. The architecture and the API are stable, and from here the API is a promise. It stays open source, stays free, and keeps running on Postgres.

**[Hindsight Memlake](https://hindsight.vectorize.io/blog/2026/10/13/memlake).** A storage engine that keeps agent memory in object storage, in a bucket you control, searchable in milliseconds and built for a billion memories and beyond. 100 million memories at under 100ms p99. It is part of Hindsight Enterprise, and most of Hindsight Cloud already runs on it. If your banks are under about 50,000 memories each, Postgres is still the right answer and we will keep saying so.

<!-- truncate -->

Where things stand today:

| | |
|---|---|
| Stars | **50,000** |
| Forks | 6,077 |
| Contributors | 288 |
| Merged pull requests | 3,042 |
| Releases | 73 |
| Age | 350 days |
| License | MIT |

That is a release every 4.8 days, sustained for eleven and a half months.

None of this would exist without other people's work. [pgvector](https://github.com/pgvector/pgvector) does the similarity search recall is built on. Open-weight models are why you can run all of it on your own hardware.

Thank you to everyone who filed an issue with a reproduction instead of a screenshot, sent a pull request to an integration we had never heard of, or argued with a design decision and turned out to be right. Several of the sharper things in 1.0 exist because someone outside the team said *this is wrong*.

Thanks for all the stars. If you have not tried it yet, one command wires every coding agent on your machine into one shared memory bank:

```bash
npx @vectorize-io/hindsight-coding-agents install all
```
