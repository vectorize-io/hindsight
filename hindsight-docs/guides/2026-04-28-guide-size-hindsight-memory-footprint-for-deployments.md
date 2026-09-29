---
title: "Size Hindsight Memory Footprint for Real Deployments"
authors: [benfrank241]
date: 2026-04-28T14:00:00Z
tags: [deployment, hardware, memory, guide]
description: "Size Hindsight deployments with the new memory footprint guidance for full and slim images, workers, the control plane, and PostgreSQL capacity planning."
image: /img/guides/guide-size-hindsight-memory-footprint-for-deployments.png
hide_table_of_contents: true
---

![Size Hindsight Memory Footprint for Real Deployments](/img/guides/guide-size-hindsight-memory-footprint-for-deployments.png)

If you are trying to **size Hindsight's memory footprint for deployments**, the docs are much better now because the installation guide finally spells out realistic RAM ranges by component. That is useful because the right box size depends less on hype and more on one simple question: are you running the full image with local models, or the slim image with external providers? Keep [the installation guide](https://hindsight.vectorize.io/sdks/developer/installation), [the configuration guide](https://hindsight.vectorize.io/sdks/developer/configuration), [the services guide](https://hindsight.vectorize.io/sdks/developer/services), and [the quickstart guide](https://hindsight.vectorize.io/sdks/developer/quickstart) open while you plan.

<!-- truncate -->

## The quick answer

- The full API image needs about 1.5 GB minimum and 2 GB recommended, because it loads local embedding and reranker models.
- The slim API image can run in roughly 512 MB minimum and 1 GB recommended, but only if embeddings and reranking are offloaded.
- The control plane is light, workers mirror the API image footprint, and PostgreSQL still needs its own headroom.

## Start with the current baseline numbers

The new installation guidance gives a practical baseline:

| Component | Minimum RAM | Recommended RAM |
|---|---:|---:|
| API, full image | 1.5 GB | 2 GB |
| API, slim image | 512 MB | 1 GB |
| Control plane | 128 MB | 256 MB |
| Worker | same as API variant | same as API variant |
| PostgreSQL | 512 MB | 1 GB+ |

These are not theoretical floor values. They are planning numbers that reflect the cost of local models, runtime overhead, and the fact that PostgreSQL and workers still need room to breathe.

## Pick full vs slim before you pick a machine

The biggest fork in the road is whether you want local embedding and reranking bundled into the API process.

- Choose **full** when you want a simpler all in one deployment and can afford the extra RAM.
- Choose **slim** when you want smaller hosts and are comfortable wiring external providers from [the configuration guide](https://hindsight.vectorize.io/sdks/developer/configuration).

That one decision usually matters more than arguing about small VM families. The full image buys convenience. The slim image buys a much smaller footprint.

## Use practical deployment recipes

A few starting points work well:

- **Laptop or personal dev box**: full image, embedded database, 2 vCPUs, 2 GB to 4 GB RAM.
- **Small cloud VM**: slim image, external embeddings and reranker, 1 GB to 2 GB RAM plus a separate database.
- **Heavier production setup**: API and worker separated, PostgreSQL sized independently, reranker moved off box if recall latency matters.

This is why the docs now separate API, worker, UI, and database guidance. Hindsight is one product, but not one memory footprint.

## Know where the pressure really comes from

For production traffic, the reranker is usually the first thing that makes the host feel expensive. On CPU only boxes, it can become the main latency and memory pressure point. That is why the installation docs now say CPU is fine for development and basic workloads, but production traffic often benefits from GPU backed reranking or an external reranker service.

In other words, if a deployment feels larger than expected, the culprit is often model locality, not the control plane or a simple docs page count.

## Troubleshoot memory pressure without guessing

If a node is running hot, work through the stack in order:

1. Confirm whether you are on the full or slim image.
2. Check whether workers are sharing the same host and doubling the expected model footprint.
3. Verify PostgreSQL is not starved on the same box.
4. Review external provider settings in [the configuration guide](https://hindsight.vectorize.io/sdks/developer/configuration).
5. Compare the deployment shape with [the services guide](https://hindsight.vectorize.io/sdks/developer/services) if you are splitting API and worker roles.

The new docs do not remove tuning work, but they do make the first estimate much less hand wavy.

## FAQ

### Can I run Hindsight in 1 GB of RAM?

Yes, but usually only with the slim image and external providers. The full image is not the right choice for that envelope.

### Do workers need separate sizing?

Yes. Workers load the same model stack as the API image variant, so they should be budgeted like another API process.

### Is the control plane the expensive part?

No. The control plane is relatively light. Local embedding and reranking are what dominate the footprint.

## Next Steps

- [Hindsight Cloud](https://hindsight.vectorize.io)
- [the installation guide](https://hindsight.vectorize.io/sdks/developer/installation)
- [the configuration guide](https://hindsight.vectorize.io/sdks/developer/configuration)
- [the services guide](https://hindsight.vectorize.io/sdks/developer/services)
- [the quickstart guide](https://hindsight.vectorize.io/sdks/developer/quickstart)
