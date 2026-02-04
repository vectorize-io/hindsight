# Hindsight Memory Integration for Vercel AI SDK

Give your AI agents persistent, human-like memory using [Hindsight](https://vectorize.io/hindsight) with the [Vercel AI SDK](https://ai-sdk.dev).

## Features

- **Three Memory Operations**: `retain` (store), `recall` (retrieve), and `reflect` (reason over memories)
- **Multi-User Support**: Dynamic bank IDs per call for multi-user/multi-tenant scenarios
- **Full API Coverage**: Complete parameter support for all Hindsight operations
- **Type-Safe**: Full TypeScript support with Zod schemas for validation
- **AI SDK 6 Native**: Works seamlessly with `generateText`, `streamText`, and `ToolLoopAgent`

## Installation

```bash
npm install @vectorize-io/hindsight-ai-sdk ai zod
```

You'll also need a Hindsight client. Choose one:

**Option A: TypeScript/JavaScript Client**
```bash
npm install @vectorize-io/hindsight-client
```

**Option B: Direct HTTP Client** (no additional dependencies)
```typescript
// See "HTTP Client Example" below
```

## Quick Start

### 1. Set up your Hindsight client

```typescript
import { HindsightClient } from '@vectorize-io/hindsight-client';

const hindsightClient = new HindsightClient({
  apiUrl: process.env.HINDSIGHT_API_URL || 'http://localhost:8000',
});
```

### 2. Create Hindsight tools

```typescript
import { createHindsightTools } from '@vectorize-io/hindsight-ai-sdk';

const tools = createHindsightTools({
  client: hindsightClient,
});
```

### 3. Use with AI SDK

```typescript
import { generateText } from 'ai';
import { anthropic } from '@ai-sdk/anthropic';

const result = await generateText({
  model: anthropic('claude-sonnet-4-20250514'),
  tools,
  prompt: 'Remember that Alice loves hiking and prefers spicy food',
});

console.log(result.text);
```

## Full Example: Memory-Enabled Chatbot

```typescript
import { HindsightClient } from '@vectorize-io/hindsight-client';
import { createHindsightTools } from '@vectorize-io/hindsight-ai-sdk';
import { streamText } from 'ai';
import { anthropic } from '@ai-sdk/anthropic';

// Initialize Hindsight
const hindsightClient = new HindsightClient({
  apiUrl: 'http://localhost:8000',
});

const tools = createHindsightTools({ client: hindsightClient });

// Chat with memory
const result = await streamText({
  model: anthropic('claude-sonnet-4-20250514'),
  tools,
  system: `You are a helpful assistant with long-term memory.

IMPORTANT:
- Before answering questions, use the 'recall' tool to check for relevant memories
- When users share important information, use the 'retain' tool to remember it
- For complex questions requiring synthesis, use the 'reflect' tool
- Always pass the user's ID as the bankId parameter

Your memory persists across sessions!`,
  prompt: 'Remember that I am Alice and I love hiking',
});

for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}
```

## API Reference

### `createHindsightTools(options)`

Creates AI SDK tool definitions for Hindsight memory operations.

**Parameters:**

- `options.client`: `HindsightClient` - Hindsight client instance
- `options.retainDescription`: `string` (optional) - Custom description for the retain tool
- `options.recallDescription`: `string` (optional) - Custom description for the recall tool
- `options.reflectDescription`: `string` (optional) - Custom description for the reflect tool

**Returns:** Object with three tools: `retain`, `recall`, and `reflect`

### Tool: `retain`

Store information in long-term memory.

**Parameters:**
- `bankId`: `string` - Memory bank ID (usually the user ID)
- `content`: `string` - Content to store
- `documentId`: `string` (optional) - Document ID for grouping/upserting
- `timestamp`: `string` (optional) - ISO timestamp for when the memory occurred
- `context`: `string` (optional) - Additional context about the memory

**Returns:**
```typescript
{
  success: boolean;
  itemsCount: number;
}
```

### Tool: `recall`

Search memory for relevant information.

**Parameters:**
- `bankId`: `string` - Memory bank ID
- `query`: `string` - What to search for
- `types`: `string[]` (optional) - Filter by fact types
- `maxTokens`: `number` (optional) - Maximum tokens to return
- `budget`: `'low' | 'mid' | 'high'` (optional) - Processing budget
- `queryTimestamp`: `string` (optional) - Query from a specific time (ISO format)
- `includeEntities`: `boolean` (optional) - Include entity observations
- `includeChunks`: `boolean` (optional) - Include raw chunks

**Returns:**
```typescript
{
  results: Array<{
    id: string;
    text: string;
    type?: string;
    entities?: string[];
    context?: string;
    occurred_start?: string;
    occurred_end?: string;
    mentioned_at?: string;
    document_id?: string;
    metadata?: Record<string, string>;
    chunk_id?: string;
  }>;
  entities?: Record<string, EntityState>;
}
```

### Tool: `reflect`

Analyze memories to form insights and generate contextual answers.

**Parameters:**
- `bankId`: `string` - Memory bank ID
- `query`: `string` - Question to reflect on
- `context`: `string` (optional) - Additional context for reflection
- `budget`: `'low' | 'mid' | 'high'` (optional) - Processing budget

**Returns:**
```typescript
{
  text: string;
  basedOn?: Array<{
    id?: string;
    text: string;
    type?: string;
    context?: string;
    occurred_start?: string;
    occurred_end?: string;
  }>;
}
```

## Advanced Usage

### Custom Tool Descriptions

Customize tool descriptions to guide model behavior:

```typescript
const tools = createHindsightTools({
  client: hindsightClient,
  retainDescription: 'Store user preferences and important facts. Always include context.',
  recallDescription: 'Search past conversations. Use specific queries for best results.',
  reflectDescription: 'Synthesize insights from memories. Use for complex questions.',
});
```

### Multi-User Scenarios

Each tool call accepts a `bankId` parameter, making it easy to support multiple users:

```typescript
const result = await generateText({
  model: anthropic('claude-sonnet-4-20250514'),
  tools,
  prompt: `User ID: ${userId}\n\nRemember that I prefer dark mode`,
});
```

The model will automatically pass the user ID to the tools.

### Using with ToolLoopAgent

```typescript
import { ToolLoopAgent, stopWhen, stepCountIs } from 'ai';

const agent = new ToolLoopAgent({
  model: anthropic('claude-sonnet-4-20250514'),
  tools,
  instructions: `You are a personal assistant with long-term memory.

    Always check memory before responding using the recall tool.
    Store important user preferences with the retain tool.
    Use the reflect tool to analyze patterns in the user's behavior.`,
  stopWhen: stepCountIs(10),
});

const result = await agent.generate({
  prompt: 'What did I say I wanted to work on this week?',
});
```

## HTTP Client Example

If you prefer not to install the full Hindsight client, you can use a simple HTTP client:

```typescript
import type { HindsightClient } from '@vectorize-io/hindsight-ai-sdk';

const httpClient: HindsightClient = {
  async retain(bankId, content, options = {}) {
    const response = await fetch(`${HINDSIGHT_URL}/v1/default/banks/${bankId}/memories/retain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content,
        timestamp: options.timestamp,
        context: options.context,
        metadata: options.metadata,
        document_id: options.documentId,
        async: options.async,
      }),
    });
    return response.json();
  },

  async recall(bankId, query, options = {}) {
    const response = await fetch(`${HINDSIGHT_URL}/v1/default/banks/${bankId}/memories/recall`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        types: options.types,
        max_tokens: options.maxTokens,
        budget: options.budget,
        trace: options.trace,
        query_timestamp: options.queryTimestamp,
        include_entities: options.includeEntities,
        max_entity_tokens: options.maxEntityTokens,
        include_chunks: options.includeChunks,
        max_chunk_tokens: options.maxChunkTokens,
      }),
    });
    return response.json();
  },

  async reflect(bankId, query, options = {}) {
    const response = await fetch(`${HINDSIGHT_URL}/v1/default/banks/${bankId}/reflect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        context: options.context,
        budget: options.budget,
      }),
    });
    return response.json();
  },
};

const tools = createHindsightTools({ client: httpClient });
```

## Running Hindsight Locally

The easiest way to run Hindsight for development:

```bash
# Install and run with embedded mode (no setup required)
uvx hindsight-embed@latest -p myapp daemon start

# The API will be available at http://localhost:8000
```

For production deployments, see the [Hindsight Documentation](https://vectorize.io/hindsight).

## TypeScript Types

All types are exported for your convenience:

```typescript
import type {
  Budget,
  HindsightClient,
  HindsightTools,
  HindsightToolsOptions,
  RecallResult,
  RecallResponse,
  ReflectFact,
  ReflectResponse,
  RetainResponse,
  EntityState,
  ChunkData,
} from '@vectorize-io/hindsight-ai-sdk';
```

## Documentation & Resources

- [Hindsight Documentation](https://vectorize.io/hindsight)
- [Vercel AI SDK Documentation](https://ai-sdk.dev)
- [GitHub Repository](https://github.com/vectorize-io/hindsight)
- [Examples](https://github.com/vectorize-io/hindsight/tree/main/examples)

## License

MIT

## Support

For issues and questions:
- [GitHub Issues](https://github.com/vectorize-io/hindsight/issues)
- Email: support@vectorize.io
