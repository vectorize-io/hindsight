
# Vercel AI SDK

The `@vectorize-io/hindsight-ai-sdk` package integrates [Hindsight](https://hindsight.vectorize.io) memory with the [Vercel AI SDK](https://ai-sdk.dev). It provides five ready-to-use tools for retaining, recalling, and reflecting on long-term memories.

[View Changelog →](../../changelog/integrations/ai-sdk.md)

## Installation

```bash
npm install @vectorize-io/hindsight-ai-sdk @vectorize-io/hindsight-client ai
```

## Setup

Create a Hindsight client and pass it to `createHindsightTools` along with a `bankId`. The `bankId` identifies the memory store for this session—typically a user ID.

```typescript
import { HindsightClient } from '@vectorize-io/hindsight-client';
import { createHindsightTools } from '@vectorize-io/hindsight-ai-sdk';

const client = new HindsightClient({ baseUrl: 'http://localhost:8888' });

const tools = createHindsightTools({
  client,
  bankId: 'user-123',
});
```

> **💡 Per-request bank IDs**
>
In multi-user applications, create `tools` inside your request handler so each request closes over the correct `bankId`. See the [Next.js example](#in-a-nextjs-route-handler) below.
## Usage

### With `generateText`

```typescript
import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

const { text } = await generateText({
  model: openai('gpt-4o'),
  tools,
  maxSteps: 5,
  system: 'You are a helpful assistant with long-term memory.',
  prompt: 'Remember that I prefer dark mode and large fonts.',
});
```

### With `streamText`

```typescript
import { streamText } from 'ai';

const result = streamText({
  model: openai('gpt-4o'),
  tools,
  maxSteps: 5,
  system: 'You are a helpful assistant with long-term memory.',
  prompt: 'What are my display preferences?',
});

for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}
```

### With `ToolLoopAgent`

```typescript
import { generateText, ToolLoopAgent, stepCountIs } from 'ai';
import { openai } from '@ai-sdk/openai';
import { HindsightClient } from '@vectorize-io/hindsight-client';
import { createHindsightTools } from '@vectorize-io/hindsight-ai-sdk';

const client = new HindsightClient({ baseUrl: process.env.HINDSIGHT_API_URL! });

const agent = new ToolLoopAgent({
  model: openai('gpt-4o'),
  tools: createHindsightTools({ client, bankId: 'user-123' }),
  stopWhen: stepCountIs(10),
  system: 'You are a helpful assistant with long-term memory.',
});

const result = await agent.generate({
  prompt: 'Remember that my favorite editor is Neovim',
});
```

### In a Next.js Route Handler

```typescript
// app/api/chat/route.ts
import { streamText } from 'ai';
import { openai } from '@ai-sdk/openai';
import { HindsightClient } from '@vectorize-io/hindsight-client';
import { createHindsightTools } from '@vectorize-io/hindsight-ai-sdk';

const hindsightClient = new HindsightClient({
  baseUrl: process.env.HINDSIGHT_API_URL!,
});

export async function POST(req: Request) {
  const { messages, userId } = await req.json();

  // Tools are created per-request, closing over the current user's bankId
  const tools = createHindsightTools({
    client: hindsightClient,
    bankId: userId,
  });

  return streamText({
    model: openai('gpt-4o'),
    tools,
    maxSteps: 5,
    system: 'You are a helpful assistant with long-term memory.',
    messages,
  }).toDataStreamResponse();
}
```

---

## Tools Reference

Five tools are registered. The `bankId` is fixed at creation time—the agent cannot change it.

| Tool | What the agent provides | What the constructor controls |
|------|------------------------|-------------------------------|
| `retain` | `content`, `documentId`, `timestamp`, `context` | `async`, `tags`, `metadata` |
| `recall` | `query`, `queryTimestamp` | `budget`, `types`, `maxTokens`, `includeEntities`, `includeChunks` |
| `reflect` | `query`, `context` | `budget` |
| `getMentalModel` | `mentalModelId` | — |
| `getDocument` | `documentId` | — |

**Why this split?** Semantic inputs (what to remember, what to search for) belong to the agent. Infrastructure concerns (cost budget, tagging strategy, async mode) belong to the application.

---

## Constructor Options

All options except `client` and `bankId` are optional. Each tool's options are grouped under the tool name.

```typescript
const tools = createHindsightTools({
  client,
  bankId: userId,

  retain: {
    async: true,                        // fire-and-forget (default: false)
    tags: ['env:prod', 'app:support'],  // always attached to every retained memory
    metadata: { version: '2.0' },       // always attached to every retained memory
  },

  recall: {
    budget: 'high',                     // processing depth: low | mid | high (default: 'mid')
    types: ['experience', 'world'],     // restrict to these fact types (default: all)
    maxTokens: 2048,                    // cap token budget (default: API default)
    includeEntities: true,              // include entity observations (default: false)
    includeChunks: true,                // include raw source chunks (default: false)
  },

  reflect: {
    budget: 'mid',                      // processing depth (default: 'mid')
  },

});
```

### `retain`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `async` | `boolean` | `false` | Fire-and-forget — do not wait for ingestion to complete |
| `tags` | `string[]` | — | Tags attached to every retained memory |
| `metadata` | `Record<string, string>` | — | Metadata attached to every retained memory |
| `description` | `string` | built-in | Override the tool description shown to the model |

### `recall`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `budget` | `'low' \| 'mid' \| 'high'` | `'mid'` | Controls retrieval depth and latency |
| `types` | `('world' \| 'experience' \| 'observation')[]` | all | Restrict results to these fact types |
| `maxTokens` | `number` | API default | Cap the total tokens returned |
| `includeEntities` | `boolean` | `false` | Include entity observations in results |
| `includeChunks` | `boolean` | `false` | Include raw source chunks in results |
| `description` | `string` | built-in | Override the tool description shown to the model |

### `reflect`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `budget` | `'low' \| 'mid' \| 'high'` | `'mid'` | Controls synthesis depth and latency |
| `maxTokens` | `number` | API default | Maximum tokens for the response |
| `description` | `string` | built-in | Override the tool description shown to the model |
