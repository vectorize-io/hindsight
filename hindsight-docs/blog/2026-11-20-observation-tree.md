---
title: "Your Agent's Memory as a Folder of Markdown Files"
description: "Observation Tree turns a Hindsight bank into one markdown file per topic, grouped with no LLM call. How it decides what goes where, and why paths hold still."
authors: [benfrank241]
date: 2026-11-20
slug: "2026/11/20/observation-tree"
tags: [memory, observations, open-knowledge-format, developer]
image: /img/blog/version-1-2-2-observation-tree.jpg
hide_table_of_contents: false
---

**Recall answers a question you already know how to ask. The tree answers the one before it: what does this bank know?** Observation Tree lays a bank's observations out as a folder of markdown files, one file per topic, that an agent can `ls` and `grep` and a person can skim. No LLM runs to build it, and the same bank gives the same paths every time. This post is how that works.

## TL;DR

- **One file per topic, `index.md` per folder.** Every observation lands in exactly one file. The bundle follows Google's [Open Knowledge Format](https://okf.md/spec/) (OKF v0.2), so anything that reads OKF reads it as-is.
- **Grouped by entities, no LLM call.** Topics come from the entities the observations mention, seeded by weight. Deterministic: sorted input, no k-means rounds, paths named after the seed entity. It builds in about 0.06 s.
- **The paths hold still.** On a real 2,569-observation bank, reordering the same observations keeps 100% of them in their file; removing 1% keeps 93 to 97%. The tree is rebuilt per request, so stability has to come from the build itself, and it does.
- **Three ways to get it.** `GET /v1/default/banks/{bank_id}/observations/tree`, a `hindsight memory observation-tree` CLI command, and a control-plane **Pages &rarr; Observations** tab with search, a constellation view, and `.zip` export.
- **Experimental.** The endpoint and the file layout may still change. The grouping rules below are what ships today.

<!-- truncate -->

## What does this bank know?

[Recall](/developer/retrieval) is good at a question. You ask "what did we decide about retries?" and it returns the memories that bear on it. But it cannot answer the question you have before you know what to ask: what is in here at all? A vector index is the wrong shape for that. You cannot `ls` an embedding, and scrolling a flat list of a few thousand observations is not browsing, it is paging.

Agents hit the same wall. An agent that has to decide whether it already knows something has no cheap way to look around first. It can only query, read, query again. Code does not work that way. An agent explores a repository by listing directories and grepping files, forming a map before it asks a precise question. Observation Tree gives a bank that same surface.

## One file per topic

The tree is a folder. At the root is an `index.md` that says what the top-level topics are. Each topic is a markdown file: the observations on that topic, with their dates, sources, and tags, under a heading that names the topic. A folder that grows too large gets its own `index.md` and splits into sub-files. That is the whole structure. You can open it in an editor, point an agent at it, or `grep -ri` across it.

Because every observation is in exactly one file, the tree is a partition, not a tag cloud. A fact shows up once, where it fits best, and the file it is in is the answer to "what topic is this part of."

## How it groups, with no LLM call

The grouping is deterministic and runs no model. [Recall's reranking](/blog/2026/11/03/hindsight-models) leans on a decision model; the tree deliberately does not. It works from the entities already attached to each observation's [source facts](/developer/retain).

1. **Each observation inherits the entities of its source facts.** Identifiers do not count as topics: ids, hashes, IPs, and speaker roles are ignored, and so is any entity that appears on more than a third of the bank, because an entity that is everywhere tells you nothing about where a given observation goes.
2. **Each folder seeds its topics from its strongest entities,** by total weight. An entity that mostly co-occurs with one already picked is skipped, so you do not get two files that are really the same topic. Ties break by name, so the choice is stable.
3. **An observation joins the seed entity it carries the most of.** A memory about three things goes to the one it is most about, not all three.
4. **An observation with no seed entity** joins the topic it shares at least a quarter of its entity weight with, or the folder's `misc` if there is no such topic. It is never forced into an unrelated file to avoid a `misc`.
5. **Sizes have hard rules.** A bank of 120 observations or fewer is a single `all.md`. A topic with more than 120 observations becomes a folder and splits by the same rules one level down. A `misc` that gets large splits by calendar: by month, then by ISO week. Calendar buckets never reshuffle as the bank grows, so last March's observations stay in last March's file.

No step asks a model to name a cluster or decide a boundary. The topics are the entities the bank already talks about most, and the paths are named after them.

## Why the paths hold still

A stored tree would need one copy per tag scope, so Observation Tree is rebuilt on every request instead. That makes stability a property of the build, not of a cache: run it twice and you must get the same files, and add a few observations and the old ones must not move.

They mostly do not. On a copy of a real 2,569-observation bank:

| Change | Keeping their file before | After |
| --- | --- | --- |
| Same observations, different order | 3% | **100%** |
| 1% of observations removed | 3% | **93 to 97%** |
| 5% of observations removed | 3 to 4% | **84 to 85%** |

The "before" column is what a naive clustering pass gives you: reorder the input and almost everything lands somewhere new. Entity-count seeding with sorted input and no k-means rounds is what moves the first column to the second. The whole build takes about 0.06 s, and the determinism is held in place by unit tests that assert the file assignments across reorder, removal, and addition.

## It follows the Open Knowledge Format

The bundle is not an ad-hoc layout. It is [OKF v0.2](https://okf.md/spec/), so a tool that reads Open Knowledge Format reads a Hindsight export without knowing it came from Hindsight.

Each topic file opens with frontmatter: `type: Memory Topic`, a `title` and `description`, its `tags`, a `timestamp`, and a `generated` block recording `hindsight/<version>` and when. The timestamps are the newest observation's, not the moment you ran the export, so the same bank produces a byte-stable bundle you can commit and diff. Folder overviews use the reserved `index.md` name, and only the root's carries the `okf_version: "0.2"` marker. Links between topics are bundle-relative, like `/gemma/vllm.md`, so the folder is self-contained once it is on disk. A conformance test checks the layout and that every internal link resolves, and a real 3,005-observation export was validated with PyYAML.

This is the portability point. Your agent's memory is not locked behind an API shape. It is files, in a published format, that you own and can read without us.

## Getting the tree

Over the API, one call returns the folders and files, each with its markdown in `content`:

```bash
curl "$HINDSIGHT_API_URL/v1/default/banks/$BANK_ID/observations/tree"
```

The same `tags`, `tags_match`, and `tag_groups` filters that [recall](/developer/api/recall) uses apply here, so you can build the tree from one scope instead of the whole bank. The response is gzipped (2.3 MB down to 0.55 MB for 3,000 observations).

The CLI prints the files, or writes the bundle to disk:

```bash
# print the tree
hindsight memory observation-tree "$BANK_ID"

# one scope only
hindsight memory observation-tree "$BANK_ID" --tags team:platform --tags-match any_strict

# write the bundle to a directory
hindsight memory observation-tree "$BANK_ID" --out ./bank-export
```

From Python, the tree walks like any folder structure:

```python
tree = await client.memory.get_observation_tree(bank_id=BANK_ID)
print(f"{tree.observation_count} observations")

def walk(folder):
    for f in folder.files:
        print(f"/{f.path} ({f.observation_count})")
    for sub in folder.folders:
        walk(sub)

walk(tree)
```

And in the control plane, open a bank and go to **Pages &rarr; Observations**. The **Files** view is the tree, with a **Find** box that filters like `grep -i` and highlights matches, a **Raw .md** toggle, a tag filter that rebuilds the tree for one scope, and **Export .zip**. The **Constellation** view puts every observation on a map, clustered by the same topics, with related topics bridged.

![The Constellation view: every observation in a bank as a star, clustered by topic, with bridges drawn between related clusters](/img/blog/constellation-view.png)

## What it does not do yet

Two honest limits. First, between a third and a half of a coding-agent bank's observations share no common entity with anything else and land in `misc` files. That is the cost of not forcing unrelated memories into a topic to make the tree look tidier. Second, the whole tree comes back in one response, around 2 MB for 3,000 observations, so a very large bank will want its files loaded on demand rather than all at once.

And the standing caveat: this is experimental. The endpoint and the file layout can still change. The grouping rules in this post are what ships today, and the determinism tests are what keep them honest while they settle.

If you want to see what one of your banks actually knows, the fastest path is `hindsight memory observation-tree <bank> --out ./export` and open the folder. The [Observation Tree docs](/developer/observations#observation-tree) have the full API, CLI, and language examples.
