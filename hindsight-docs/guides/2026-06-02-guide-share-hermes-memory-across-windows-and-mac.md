---
title: "Guide: Share Hermes Memory Across Windows and Mac"
authors: [benfrank241]
date: 2026-06-02T15:00:00Z
tags: [how-to, hermes, windows, mac, memory]
description: "Use Hindsight Cloud to share one Hermes memory workflow across Windows and Mac without re-explaining the same context on each machine."
image: /img/guides/guide-share-hermes-memory-across-windows-and-mac.svg
hide_table_of_contents: true
---

![Guide: Share Hermes Memory Across Windows and Mac](/img/guides/guide-share-hermes-memory-across-windows-and-mac.svg)

A lot of real workflows are mixed-platform now. You might use Hermes on a Windows workstation during the day, then pick up the same thread on a Mac laptop later. Without a shared memory backend, those feel like two separate assistants.

Hindsight lets you turn that into one continuous workflow. The trick is simple: keep the same bank strategy on both machines, point both Hermes installs at the same backend, and verify recall across the platform boundary.

<!-- truncate -->

> **Quick answer**
>
> 1. Set up Hermes with Hindsight on both Windows and Mac.
> 2. Use Hindsight Cloud or another shared backend.
> 3. Keep the same bank IDs on both platforms.
> 4. Test recall on Mac after retaining on Windows, and vice versa.
> 5. Separate banks when personal, project, or team context should stay isolated.

## Why this matters

Mixed-platform usage is one of the fastest ways to expose whether your agent really has memory.

If Hermes remembers on Windows but forgets on Mac, the problem is not the model. It is the memory boundary.

A shared backend fixes that by making the bank the unit of continuity instead of the machine.

## Step 1: Put both installs on the same backend

The easiest pattern is Hindsight Cloud.

On both machines:

~~~bash
hermes memory setup
~~~

Choose **Hindsight**, then **Cloud**.

Now both Hermes installs can point at the same memory service.

## Step 2: Keep bank naming consistent

This is the rule that matters most.

If you want the same project context on both machines, use the same project bank on both machines.

Examples:

- project-docs
- acct-acme
- personal-ben

Cross-platform memory is mostly a naming discipline problem once the shared backend exists.

## Step 3: Test the platform handoff

A quick real-world test:

1. On Windows, tell Hermes a new fact about the project.
2. End the session.
3. Open Hermes on Mac.
4. Ask for that context without repeating it.

If Mac recalls what Windows retained, you have true cross-platform continuity.

## Good use cases

### Workstation plus laptop

Research or coding on a desktop, planning or review on a laptop.

### Team workflows across mixed operating systems

One engineer uses Windows, another uses Mac, but both need the same shared project memory.

### Personal assistant continuity

Your preferences and ongoing tasks should not change just because the machine did.

## What should stay separate

Cross-platform does not mean cross-everything.

Use separate banks for:

- unrelated projects
- different customers
- personal vs team workflows
- anything privacy-sensitive that should not be shared

## Common mistakes

- using slightly different bank names on each platform
- assuming a shared backend is enough without testing recall
- mixing personal context into a shared team bank
- trying to share local-only memory across machines

## FAQ

### Does this require the same filesystem layout on both machines?

No. Bank continuity does not depend on matching local paths.

### Is Cloud required?

A shared backend is required. Cloud is just the easiest way to get one.

### Can this work for team-shared repo memory too?

Yes. Mixed-platform engineering teams are a strong use case.

## Next Steps

- Read [give Hermes cross-device memory with Hindsight Cloud](/guides/2026/06/02/guide-hermes-cross-device-memory-with-hindsight-cloud)
- Read [Hermes Agent on Windows: Set Up Persistent Memory with Hindsight](/blog/2026/06/01/hermes-hindsight-windows-setup)
- Start with [the Hermes integration docs](https://hindsight.vectorize.io/sdks/integrations/hermes)
