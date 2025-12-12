---
sidebar_position: 6
---

# Memory Banks

Memory banks are isolated containers that store all memory-related data for a specific context or use case.

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

## What is a Memory Bank?

A memory bank is a complete, isolated storage unit containing:

- **Memories** — Facts and information retained from conversations
- **Documents** — Files and content indexed for retrieval
- **Entities** — People, places, concepts extracted from memories
- **Relationships** — Connections between entities in the knowledge graph

Banks are completely isolated from each other — memories stored in one bank are not visible to another.

You don't need to pre-create a bank. Hindsight will automatically create it with default settings when you first use it.

:::tip Prerequisites
Make sure you've completed the [Quick Start](./quickstart) to install the client and start the server.
:::

## Creating a Memory Bank

<Tabs>
<TabItem value="python" label="Python">

```python
from hindsight_client import Hindsight

client = Hindsight(base_url="http://localhost:8888")

client.create_bank(
    bank_id="my-bank",
    name="Research Assistant",
    background="I am a research assistant specializing in machine learning",
    disposition={
        "skepticism": 4,
        "literalism": 3,
        "empathy": 3
    }
)
```

</TabItem>
<TabItem value="node" label="Node.js">

```typescript
import { HindsightClient } from '@vectorize-io/hindsight-client';

const client = new HindsightClient({ baseUrl: 'http://localhost:8888' });

await client.createBank('my-bank', {
    name: 'Research Assistant',
    background: 'I am a research assistant specializing in machine learning',
    disposition: {
        skepticism: 4,
        literalism: 3,
        empathy: 3
    }
});
```

</TabItem>
<TabItem value="cli" label="CLI">

```bash
# Set background
hindsight bank background my-bank "I am a research assistant specializing in ML"

# Set disposition
hindsight bank disposition my-bank \
    --skepticism 4 \
    --literalism 3 \
    --empathy 3
```

</TabItem>
</Tabs>

## Background and Disposition

Background and disposition are optional settings that influence how the bank forms opinions during [reflect](./reflect) operations.

:::info
Background and disposition only affect the `reflect` operation (opinion formation). They do not impact `retain`, `recall`, or other memory operations.
:::

### Background

The background is a first-person narrative providing context for opinion formation:

<Tabs>
<TabItem value="python" label="Python">

```python
client.create_bank(
    bank_id="financial-advisor",
    background="""I am a conservative financial advisor with 20 years of experience.
    I prioritize capital preservation over aggressive growth.
    I have seen multiple market crashes and believe in diversification."""
)
```

</TabItem>
<TabItem value="node" label="Node.js">

```typescript
await client.createBank('financial-advisor', {
    background: `I am a conservative financial advisor with 20 years of experience.
    I prioritize capital preservation over aggressive growth.
    I have seen multiple market crashes and believe in diversification.`
});
```

</TabItem>
</Tabs>

### Disposition Traits

Disposition traits influence how opinions are formed during reflection. Each trait is scored 1 to 5:

| Trait | Low (1) | High (5) |
|-------|---------|----------|
| **Skepticism** | Trusting, accepts information at face value | Skeptical, questions and doubts claims |
| **Literalism** | Flexible interpretation, reads between the lines | Literal interpretation, takes things exactly as stated |
| **Empathy** | Detached, focuses on facts and logic | Empathetic, considers emotional context |
