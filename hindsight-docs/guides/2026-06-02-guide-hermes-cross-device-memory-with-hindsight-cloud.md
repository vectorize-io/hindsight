---
title: "Guide: Give Hermes Cross-Device Memory with Hindsight Cloud"
authors: [benfrank241]
date: 2026-06-02T15:00:00Z
tags: [how-to, hermes, cloud, memory]
description: "Set up Hermes with Hindsight Cloud so the same memory follows you across laptop, desktop, and server sessions."
image: /img/guides/guide-hermes-cross-device-memory-with-hindsight-cloud.svg
hide_table_of_contents: true
---

![Guide: Give Hermes Cross-Device Memory with Hindsight Cloud](/img/guides/guide-hermes-cross-device-memory-with-hindsight-cloud.svg)

If you run Hermes on more than one machine, **cross-device memory** is the difference between one assistant and three disconnected ones. Without a shared memory backend, your laptop Hermes, desktop Hermes, and server Hermes each build their own partial history. You end up re-explaining the same context everywhere.

Hindsight Cloud fixes that. Point multiple Hermes installs at the same memory backend, keep a stable bank strategy, and the assistant carries context across devices instead of starting over.

<!-- truncate -->

> **Quick answer**
>
> 1. Set up Hermes with Hindsight Cloud.
> 2. Use the same bank IDs on every device that should share memory.
> 3. Keep project or user bank naming stable.
> 4. Verify recall from one device after retaining facts on another.
> 5. Use separate banks when you want isolation.

## Why local memory is not enough across devices

Local memory works until your workflow spans machines.

Maybe you research on a laptop, code on a desktop, and run automation on a server. If each environment keeps memory locally, then your context fragments immediately. The information exists, but it is scattered.

Cross-device memory solves that by moving the bank behind the agent instead of inside one machine.

## Step 1: Choose Hindsight Cloud

The easiest path is the Hermes memory setup wizard:

~~~bash
hermes memory setup
~~~

Choose **Hindsight**, then choose **Cloud**.

That gives every Hermes install a shared backend instead of a machine-local one.

## Step 2: Make your bank strategy explicit

A shared backend is only half the job. The other half is making sure every device points at the **same bank when it should**.

Good patterns:

- one bank per user assistant
- one bank per project
- one bank per customer or account

Examples:

- ben-personal
- project-hindsight-docs
- acct-acme

The key is stability. If your laptop uses project-hindsight-docs and your desktop uses docs-project, you did not create cross-device memory. You created two separate histories with similar names.

## Step 3: Verify the flow end to end

A quick verification loop:

1. On device A, tell Hermes one fact worth remembering.
2. End the session.
3. On device B, start a fresh Hermes session using the same bank.
4. Ask for the fact.

If the second device can recall it cleanly, the setup is working.

## Best use cases for cross-device memory

### Personal assistant workflows

You might message Hermes from your phone, continue from your laptop, and later ask a follow-up from your desktop. Shared memory makes those all feel like one ongoing relationship with the same assistant.

### Coding workflows

Research on one machine, implementation on another, deployment or monitoring on a server. A shared bank keeps the repo context continuous.

### Team operations

A server-side Hermes worker can build memory that a human-operated Hermes chat can later use, and vice versa.

## When not to share a bank

Not everything should be in one pool.

Use separate banks when:

- different projects should stay isolated
- personal context should not mix with team memory
- customer-specific memory should not leak across accounts

Cross-device should not mean cross-everything.

## Common mistakes

- using different bank IDs on different machines
- reusing one giant bank for unrelated work
- assuming Cloud alone guarantees good retrieval without a clear naming strategy
- forgetting to test recall from a second device

## FAQ

### Can I share one bank across laptop and server?

Yes. That is one of the main reasons to use Cloud.

### Do I need the same local Hermes install path on every machine?

No. What matters is the shared backend and stable bank IDs.

### Should teams share one bank too?

Only when the workflow itself should share memory. Otherwise, keep banks separate.

## Next Steps

- Read [move Hermes memory from local files to Hindsight Cloud](/guides/2026/04/20/guide-move-hermes-memory-from-local-files-to-hindsight-cloud)
- Review [single bank vs multi-bank with Hindsight](/guides/2026/04/16/comparison-single-bank-vs-multi-bank-hindsight)
- Start with [Hindsight Cloud](https://hindsight.vectorize.io)
