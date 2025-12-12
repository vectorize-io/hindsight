---
sidebar_position: 9
---

# Operations

Background tasks that Hindsight executes asynchronously.

:::tip Prerequisites
Make sure you've completed the [Quick Start](./quickstart) and understand [how retain works](./retain).
:::

## How Operations Work

Hindsight processes several types of tasks in the background to maintain memory quality and consistency. These operations run automatically—you don't need to trigger them manually.

By default, all background operations are executed in-process within the API service.

:::note Kafka Integration
Support for external streaming platforms like Kafka for scale-out processing is planned but **not available out of the box** in the current release.
:::

## Operation Types

| Operation | Trigger | Description |
|-----------|---------|-------------|
| **batch_retain** | `retain_batch` with `async=True` | Processes large content batches in the background |
| **form_opinion** | After each `reflect` call | Extracts and stores new opinions formed during reflection |
| **reinforce_opinion** | After `retain` | Updates opinion confidence based on new supporting evidence |
| **access_count_update** | After `recall` | Tracks which memories are accessed for relevance scoring |
| **regenerate_observations** | Bank profile update | Regenerates entity observations when disposition changes |

## Next Steps

- [**Documents**](./documents) — Track document sources
- [**Entities**](./entities) — Monitor entity tracking
- [**Memory Banks**](./memory-banks) — Configure bank settings
