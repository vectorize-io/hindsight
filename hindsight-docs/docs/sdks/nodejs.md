---
sidebar_position: 2
---

# Node.js Client

Official TypeScript/JavaScript client for the Hindsight API.

## Installation

```bash
npm install @vectorize-io/hindsight-client
```

## Quick Start

```typescript
const { HindsightClient } = require('@vectorize-io/hindsight-client');

const client = new HindsightClient({ baseUrl: 'http://localhost:8888' });

// Retain a memory
await client.retain('my-bank', 'Alice works at Google');

// Recall memories
const response = await client.recall('my-bank', 'What does Alice do?');
for (const r of response.results) {
    console.log(r.text);
}

// Reflect - generate response with disposition
const answer = await client.reflect('my-bank', 'Tell me about Alice');
console.log(answer.text);
```

## Client Initialization

```typescript
import { HindsightClient } from '@vectorize-io/hindsight-client';

const client = new HindsightClient({
    baseUrl: 'http://localhost:8888',
});
```

## Core Operations

### Retain (Store Memory)

```typescript
// Simple
await client.retain('my-bank', 'Alice works at Google');

// With options
await client.retain('my-bank', 'Alice got promoted', {
    timestamp: new Date('2024-01-15'),
    context: 'career update',
    metadata: { source: 'slack' },
    async: false,  // Set true for background processing
});
```

### Retain Batch

```typescript
await client.retainBatch('my-bank', [
    { content: 'Alice works at Google', context: 'career' },
    { content: 'Bob is a data scientist', context: 'career' },
], {
    async: false,
});
```

### Recall (Search)

```typescript
// Simple - returns RecallResponse
const response = await client.recall('my-bank', 'What does Alice do?');

for (const r of response.results) {
    console.log(`${r.text} (type: ${r.type})`);
}

// With options
const response = await client.recall('my-bank', 'What does Alice do?', {
    types: ['world', 'opinion'],  // Filter by fact type
    maxTokens: 4096,
    budget: 'high',  // 'low', 'mid', or 'high'
});
```

### Reflect (Generate Response)

```typescript
const answer = await client.reflect('my-bank', 'What should I know about Alice?', {
    budget: 'low',  // 'low', 'mid', or 'high'
    context: 'preparing for a meeting',
});

console.log(answer.text);       // Generated response
```

## Bank Management

### Create Bank

```typescript
await client.createBank('my-bank', {
    name: 'Assistant',
    background: 'I am a helpful AI assistant',
    disposition: {
        skepticism: 3,   // 1-5: trusting to skeptical
        literalism: 3,   // 1-5: flexible to literal
        empathy: 3,      // 1-5: detached to empathetic
    },
});
```

### List Memories

```typescript
const response = await client.listMemories('my-bank', {
    type: 'world',  // Optional filter
    q: 'Alice',     // Optional text search
    limit: 100,
    offset: 0,
});
console.log(response)
```
