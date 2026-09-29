# Hindsight TypeScript Client

TypeScript client library for the Hindsight API.

## Installation

```bash
npm install @vectorize-io/hindsight-client
# or
yarn add @vectorize-io/hindsight-client
```

## Usage

```typescript
import { HindsightClient } from "@vectorize-io/hindsight-client";

const client = new HindsightClient({ baseUrl: "http://localhost:8888" });

// Retain information
await client.retain("my-bank", "Alice works at Google in Mountain View.");

// Recall memories
const results = await client.recall("my-bank", "Where does Alice work?");

// Reflect and get an opinion
const response = await client.reflect("my-bank", "What do you think about Alice's career?");
```

## API Reference

### `retain(bankId, content, options?)`

Store a single memory.

```typescript
await client.retain("my-bank", "User prefers dark mode", {
  timestamp: new Date(),
  context: "Settings conversation",
  metadata: { source: "chat" },
});
```

### `retainBatch(bankId, items, options?)`

Store multiple memories in batch.

```typescript
await client.retainBatch(
  "my-bank",
  [{ content: "Alice loves hiking" }, { content: "Alice visited Paris last summer" }],
  { async: true }
);
```

### `recall(bankId, query, options?)`

Recall memories matching a query.

```typescript
const results = await client.recall("my-bank", "What are Alice's hobbies?", {
  budget: "mid",
});
```

### `reflect(bankId, query, options?)`

Generate a contextual answer using the bank's identity and memories.

```typescript
const response = await client.reflect("my-bank", "What should I do this weekend?", {
  budget: "low",
});
console.log(response.text);
```

### `getVersion(options?)`

Read the connected Hindsight API version and feature flags. This is useful for
integrations that need to enforce a minimum server version before enabling a
workflow.

```typescript
const version = await client.getVersion();

const isAtLeast = (actual: string, minimum: string) => {
  const actualParts = actual.split(".").map(Number);
  const minimumParts = minimum.split(".").map(Number);
  for (let i = 0; i < Math.max(actualParts.length, minimumParts.length); i++) {
    const actualPart = actualParts[i] ?? 0;
    const minimumPart = minimumParts[i] ?? 0;
    if (actualPart !== minimumPart) return actualPart > minimumPart;
  }
  return true;
};

if (!isAtLeast(version.api_version, "0.8.2")) {
  throw new Error(`Hindsight ${version.api_version} is too old for this integration`);
}

if (version.features.observations) {
  console.log("Observation consolidation is enabled.");
}
```

### `createBank(bankId, options)`

Create or update a memory bank with personality.

```typescript
await client.createBank("my-bank", {
  name: "My Assistant",
  background: "A helpful assistant that remembers everything.",
});
```

## Documentation

For full documentation, visit [hindsight](https://github.com/vectorize-io/hindsight).
