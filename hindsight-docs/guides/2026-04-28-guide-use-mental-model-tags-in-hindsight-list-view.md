---
title: "Use Mental Model Tags in Hindsight's New List View"
authors: [benfrank241]
date: 2026-04-28T14:00:00Z
tags: [mental-models, tags, control-plane, guide]
description: "Use Hindsight mental model tags with the new List view, server-side tag suggestions, and safer filtering for refresh and reflect workflows today."
image: /img/guides/guide-use-mental-model-tags-in-hindsight-list-view.png
hide_table_of_contents: true
---

![Use Mental Model Tags in Hindsight's New List View](/img/guides/guide-use-mental-model-tags-in-hindsight-list-view.png)

If you want to **use mental model tags in Hindsight's new List view**, the important change is that the control plane now treats mental models more like real working documents. The page defaults to a split pane list, and tag suggestions now come from the actual mental model tag source instead of leaving you to remember tags by hand. Keep [the mental models API docs](https://hindsight.vectorize.io/sdks/developer/api/mental-models), [the reflect docs](https://hindsight.vectorize.io/sdks/developer/api/reflect), [the observations docs](https://hindsight.vectorize.io/sdks/developer/observations), and [the docs home](https://hindsight.vectorize.io) nearby while you tune the workflow.

<!-- truncate -->

## The quick answer

- Mental Models now open in a default List view, with the old card dashboard still available as a secondary view.
- Tag suggestions are fetched from the new mental model tag source, which means filtering is now grounded in the tags your models actually use.
- Mental model tags affect both refresh input and reflect visibility, so filtering is not just cosmetic.

## What the new List view changes

The List view is a better default because mental models are long lived operational documents, not just dashboard cards. You can scan names, source queries, and refresh state in one pane, then inspect content in the other.

That sounds small, but it fixes a real workflow problem. When you are maintaining a bank with multiple mental models, the card grid is nice for browsing, while the list is better for actual maintenance. The update keeps both, but it makes the maintenance view the first thing you land on.

## Filter by tags with real suggestions

The new tag suggestion path matters because it queries the mental model tag source directly. In practice, that means the UI can suggest tags from mental models instead of from general memories.

If you want to inspect the underlying endpoint shape, the idea is:

```bash
curl "$BASE_URL/v1/default/banks/$BANK_ID/tags?source=mental_models"
```

That is exactly the distinction you want when a bank has many ordinary memory tags, but only a smaller set of tags attached to mental models.

## Remember what mental model tags actually do

This is the part that trips people up. A mental model tag is not just a label for browsing. As the [mental models API docs](https://hindsight.vectorize.io/sdks/developer/api/mental-models) explain, tags narrow which memories the model can read during refresh, and they also affect which mental models are visible during [reflect](https://hindsight.vectorize.io/sdks/developer/api/reflect).

So if you tag a mental model with `user:alice`, refresh will only read memories that also carry that required tag. That is good when you want a scoped model, but it also means over tagging can make the model look empty or stale if the underlying memories were never backfilled.

## Use any vs all deliberately

The control plane now supports tag filtering with match modes. In practice, that gives you two useful habits:

- use **any** when you are browsing and trying to find related mental models quickly
- use **all** when you are debugging a precise scope and want to see only models that match the whole tag set

If the filtered list suddenly looks too small, the first thing to try is switching from **all** back to **any**. The second thing is checking whether the mental model itself has tags that are stricter than the memories you expected it to read.

## Troubleshoot empty or misleading results

A few patterns show up often:

- **The model exists, but content is empty.** Usually the model tags are stricter than the available memories.
- **The tag chip you want never appears.** Usually no mental model carries that tag yet, even if ordinary memories do.
- **Reflect seems to miss the model.** Usually the reflect call is using tags that do not overlap with the model's own tags.

When in doubt, step back through [the mental models API docs](https://hindsight.vectorize.io/sdks/developer/api/mental-models) and [the observations guide](https://hindsight.vectorize.io/sdks/developer/observations). Most mental model confusion is really tag scope confusion.

## FAQ

### Is the Dashboard view gone?

No. Dashboard is still there. The update makes List view the default because it is better for day to day maintenance.

### Why do mental model tag suggestions need a separate source?

Because memory tags and mental model tags solve different browsing problems. Suggestions are much more useful when they come from the mental model table you are actually filtering.

### Can a tag make a mental model disappear from reflect?

Yes. Reflect visibility is filtered by tags too, so a mismatch between request tags and model tags can hide the model.

## Next Steps

- [Hindsight Cloud](https://hindsight.vectorize.io)
- [the mental models API docs](https://hindsight.vectorize.io/sdks/developer/api/mental-models)
- [the reflect docs](https://hindsight.vectorize.io/sdks/developer/api/reflect)
- [the observations guide](https://hindsight.vectorize.io/sdks/developer/observations)
- [the docs home](https://hindsight.vectorize.io)
