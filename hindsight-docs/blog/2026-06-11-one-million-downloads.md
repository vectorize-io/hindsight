---
title: "Hindsight Passes 1,000,000 Downloads"
description: "The hindsight-client package just crossed a million downloads on PyPI — in under six months. A quick note on the milestone and what the install curve says."
slug: "2026/06/11/one-million-downloads"
date: 2026-06-11T12:00
image: "/img/blog/hindsight-one-million-downloads.png"
tags: [milestone]
hide_table_of_contents: true
---

![Hindsight Passes 1,000,000 Downloads](/img/blog/hindsight-one-million-downloads.png)

Sometime in the last day, the [`hindsight-client`](https://pypi.org/project/hindsight-client/) package crossed **1,000,000 downloads on PyPI** — on its own, not counting the rest of the family.

The first version went up on December 12, 2025. So that's a million installs in **just under six months**.

<!-- truncate -->

---

## The numbers

A few weeks ago we wrote that Hindsight is the [fastest-growing open-source AI memory project ever](/blog/2026/06/09/fastest-growing-oss-ai-memory), measured by star velocity. Downloads tell the same story from a different angle — these are people actually installing and running it, not just bookmarking the repo.

| Package | Registry | Downloads |
|---|---|---:|
| `hindsight-client` | PyPI | **~1,000,000** |
| `hindsight-api` | PyPI | ~175,000 |
| `@vectorize-io/hindsight-client` | npm | ~90,000 |

The client alone is now pulling **~140,000 downloads a week** and accelerating. Add the API package and the npm client and Hindsight is well past **1.2 million** total installs across registries.

## Why the client number matters most

`hindsight-client` is the thin SDK — it's what you `pip install` to talk to a Hindsight server from your agent. It's the closest proxy we have for "someone is building on Hindsight," because you don't install it to kick the tires; you install it because something downstream imports it.

That it crossed a million faster than it took us to go from 10k to 15k stars says the adoption isn't just attention. It's integration.

## Thank you

To everyone who ran `pip install hindsight-client`, wired it into an agent, filed an issue when it broke, or shipped it to production — thank you. You're the reason the curve looks the way it does.

If you haven't yet:

```bash
pip install hindsight-client -U
```

[Star Hindsight on GitHub](https://github.com/vectorize-io/hindsight) or [self-host with a single Docker command](https://hindsight.vectorize.io/developer/installation).
