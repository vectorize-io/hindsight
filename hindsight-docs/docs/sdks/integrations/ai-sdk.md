---
sidebar_position: 4
---

# Vercel AI SDK

Official Hindsight integration for the [Vercel AI SDK](https://ai-sdk.dev). Add persistent, biomimetic memory to your AI agents with native tool support.

## Features

- **Three Memory Tools**: `retain` (store), `recall` (retrieve), and `reflect` (reason over memories)
- **AI SDK 6 Native**: Works seamlessly with `generateText`, `streamText`, and `ToolLoopAgent`
- **Multi-User Support**: Dynamic bank IDs per tool call for multi-user/multi-tenant scenarios
- **Full Parameter Support**: Complete access to all Hindsight API parameters
- **Type-Safe**: Full TypeScript support with Zod schemas for validation
- **Flexible Client**: Works with the official TypeScript client or custom HTTP clients

## Installation

```bash
npm install @vectorize-io/hindsight-ai-sdk ai zod
```

You'll also need a Hindsight client. Choose one:

**Option A: TypeScript/JavaScript Client** (Recommended)
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

## Memory Tools

The integration provides three tools that the AI model can use to manage memory:

### `retain` - Store Information

The model calls this tool to store information for future recall.

**Parameters:**
- `bankId` (required): Memory bank ID (usually the user ID)
- `content` (required): Content to store
- `documentId` (optional): Document ID for grouping/upserting related memories
- `timestamp` (optional): ISO timestamp for when the memory occurred
- `context` (optional): Additional context about the memory
- `metadata` (optional): Key-value metadata for filtering

**Example tool call:**
```typescript
{
  bankId: "user-123",
  content: "Alice loves hiking and goes to Yosemite every summer",
  context: "User preferences",
  timestamp: "2024-01-15T10:30:00Z"
}
```

**Returns:**
```typescript
{
  success: true,
  itemsCount: 1
}
```

### `recall` - Search Memories

The model calls this tool to search for relevant information in memory.

**Parameters:**
- `bankId` (required): Memory bank ID
- `query` (required): What to search for
- `types` (optional): Filter by fact types (`['world', 'experience', 'opinion']`)
- `maxTokens` (optional): Maximum tokens to return (default: 4096)
- `budget` (optional): Processing budget - `'low'`, `'mid'`, or `'high'`
- `queryTimestamp` (optional): Query from a specific time (ISO format)
- `includeEntities` (optional): Include entity observations
- `includeChunks` (optional): Include raw document chunks

**Example tool call:**
```typescript
{
  bankId: "user-123",
  query: "What does Alice like to do outdoors?",
  types: ["world", "experience"],
  maxTokens: 2048,
  budget: "mid"
}
```

**Returns:**
```typescript
{
  results: [
    {
      id: "mem-123",
      text: "Alice loves hiking",
      type: "world",
      entities: ["Alice"],
      context: "User preferences",
      occurred_start: "2024-01-15T10:30:00Z",
      document_id: "doc-456",
      metadata: { source: "chat" }
    }
  ],
  entities: {
    "Alice": {
      canonical_name: "Alice",
      mention_count: 15,
      observations: [...]
    }
  }
}
```

### `reflect` - Synthesize Insights

The model calls this tool to analyze memories and generate contextual insights.

**Parameters:**
- `bankId` (required): Memory bank ID
- `query` (required): Question to reflect on
- `context` (optional): Additional context for reflection
- `budget` (optional): Processing budget - `'low'`, `'mid'`, or `'high'`

**Example tool call:**
```typescript
{
  bankId: "user-123",
  query: "What outdoor activities does Alice enjoy?",
  context: "Planning a weekend trip",
  budget: "mid"
}
```

**Returns:**
```typescript
{
  text: "Alice is an avid hiker who particularly enjoys visiting Yosemite National Park during summer months. She has expressed strong preferences for mountain trails over beach activities.",
  basedOn: [
    {
      id: "mem-123",
      text: "Alice loves hiking",
      type: "world",
      context: "User preferences",
      occurred_start: "2024-01-15T10:30:00Z"
    }
  ]
}
```

## Complete Example: Memory-Enabled Chatbot

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
  system: `You are a helpful assistant. The user's ID is: ${userId}`,
  prompt: 'Remember that I prefer dark mode',
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

### Memory-Aware Streaming

```typescript
import { streamText } from 'ai';

async function chat(userMessage: string, userId: string) {
  const result = streamText({
    model: anthropic('claude-sonnet-4-20250514'),
    tools,
    system: `User ID: ${userId}. Check memory before responding.`,
    prompt: userMessage,
  });

  // Stream the response
  for await (const chunk of result.textStream) {
    process.stdout.write(chunk);
  }

  // Access tool calls after streaming
  const { toolCalls } = await result;
  console.log('Memory operations:', toolCalls);
}
```

## HTTP Client Example

If you prefer not to install the full Hindsight client, you can use a simple HTTP client:

```typescript
import type { HindsightClient } from '@vectorize-io/hindsight-ai-sdk';

const HINDSIGHT_URL = 'http://localhost:8000';

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
        query_timestamp: options.queryTimestamp,
        include_entities: options.includeEntities,
        include_chunks: options.includeChunks,
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

## API Reference

### `createHindsightTools(options)`

Creates AI SDK tool definitions for Hindsight memory operations.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `options.client` | `HindsightClient` | Yes | Hindsight client instance |
| `options.retainDescription` | `string` | No | Custom description for the retain tool |
| `options.recallDescription` | `string` | No | Custom description for the recall tool |
| `options.reflectDescription` | `string` | No | Custom description for the reflect tool |

**Returns:** `HindsightTools` object with three tools: `retain`, `recall`, and `reflect`

### HindsightClient Interface

```typescript
interface HindsightClient {
  retain(
    bankId: string,
    content: string,
    options?: RetainOptions
  ): Promise<RetainResponse>;

  recall(
    bankId: string,
    query: string,
    options?: RecallOptions
  ): Promise<RecallResponse>;

  reflect(
    bankId: string,
    query: string,
    options?: ReflectOptions
  ): Promise<ReflectResponse>;
}
```

### Type Definitions

```typescript
type Budget = 'low' | 'mid' | 'high';

interface RetainOptions {
  documentId?: string;
  timestamp?: string;
  context?: string;
  metadata?: Record<string, string>;
  async?: boolean;
}

interface RecallOptions {
  types?: string[];
  maxTokens?: number;
  budget?: Budget;
  queryTimestamp?: string;
  includeEntities?: boolean;
  includeChunks?: boolean;
}

interface ReflectOptions {
  context?: string;
  budget?: Budget;
}

interface RetainResponse {
  success: boolean;
  itemsCount: number;
}

interface RecallResponse {
  results: RecallResult[];
  entities?: Record<string, EntityState>;
}

interface RecallResult {
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
}

interface ReflectResponse {
  text: string;
  basedOn?: ReflectFact[];
}

interface ReflectFact {
  id?: string;
  text: string;
  type?: string;
  context?: string;
  occurred_start?: string;
  occurred_end?: string;
}
```

## Best Practices

### System Prompts

Guide the model to use memory effectively:

```typescript
const system = `You are a helpful assistant with persistent memory.

Memory Guidelines:
- Use 'recall' to check for relevant information before answering questions
- Use 'retain' to store important user preferences, facts, and context
- Use 'reflect' for complex questions that require synthesizing multiple memories
- Always pass the user's ID as bankId in tool calls
- Include context when storing memories to improve future retrieval

Your memory persists across all conversations with this user.`;
```

### Bank ID Strategy

Choose a bank ID strategy based on your use case:

- **Per-User**: `bankId: user-${userId}` - Each user gets their own memory
- **Per-Conversation**: `bankId: conv-${conversationId}` - Each conversation is isolated
- **Per-Agent**: `bankId: agent-${agentId}` - Shared memory across all users of an agent
- **Hybrid**: `bankId: ${agentId}-${userId}` - Agent-specific memory per user

### Memory Organization

Use context and metadata to organize memories:

```typescript
await hindsightClient.retain(
  'user-123',
  'User prefers TypeScript over JavaScript',
  {
    context: 'Programming preferences',
    metadata: { category: 'preferences', topic: 'programming' },
    timestamp: new Date().toISOString(),
  }
);
```

## Running Hindsight Locally

The easiest way to run Hindsight for development:

```bash
# Install and run with embedded mode (no setup required)
uvx hindsight-embed@latest -p myapp daemon start

# The API will be available at http://localhost:8000
```

For production deployments, see the [Installation Guide](/developer/installation).

## Requirements

- Node.js >= 18
- AI SDK >= 6.0
- A running Hindsight API server

## Examples

Full examples are available in the [GitHub repository](https://github.com/vectorize-io/hindsight/tree/main/examples/ai-sdk):

- [Basic chatbot with memory](https://github.com/vectorize-io/hindsight/tree/main/examples/ai-sdk/basic-chatbot)
- [Multi-user support](https://github.com/vectorize-io/hindsight/tree/main/examples/ai-sdk/multi-user)
- [ToolLoopAgent integration](https://github.com/vectorize-io/hindsight/tree/main/examples/ai-sdk/agent-loop)

## Support

For issues and questions:
- [GitHub Issues](https://github.com/vectorize-io/hindsight/issues)
- [Documentation](https://vectorize.io/hindsight)
- Email: support@vectorize.io
