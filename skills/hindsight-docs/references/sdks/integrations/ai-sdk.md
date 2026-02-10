---
sidebar_position: 4
---

# Vercel AI SDK

Official Hindsight integration for the [Vercel AI SDK](https://ai-sdk.dev).

## Features

- **7 Memory Tools**: Core memory operations (retain, recall, reflect), mental models (create, query), documents (get), and directives (create)
- **AI SDK 6 Native**: Works seamlessly with `generateText`, `streamText`, and `ToolLoopAgent`
- **Multi-User Support**: Dynamic bank IDs per tool call for multi-user/multi-tenant scenarios
- **Full Parameter Support**: Complete access to all Hindsight API parameters
- **Type-Safe**: Full TypeScript support with Zod schemas for validation

## Installation

```bash
npm install @vectorize-io/hindsight-ai-sdk @vectorize-io/hindsight-client ai zod
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

The integration provides seven tools that the AI model can use to manage memory:

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

### `createMentalModel` - Create Knowledge Consolidation

The model calls this tool to create a mental model that automatically consolidates memories into structured knowledge.

**Parameters:**
- `bankId` (required): Memory bank ID
- `mentalModelId` (optional): Custom ID for the mental model (auto-generated if not provided)
- `name` (optional): Name for the mental model
- `sourceQuery` (optional): Query defining which memories to consolidate
- `tags` (optional): Tags for organizing mental models
- `maxTokens` (optional): Maximum tokens for the content
- `autoRefresh` (optional): Auto-refresh after new consolidations (default: false)

**Example tool call:**
```typescript
{
  bankId: "user-123",
  name: "User Preferences",
  sourceQuery: "What are the user's preferences?",
  tags: ["preferences"],
  autoRefresh: true
}
```

**Returns:**
```typescript
{
  mentalModelId: "mm-456",
  createdAt: "2024-01-15T10:30:00Z"
}
```

### `queryMentalModel` - Retrieve Consolidated Knowledge

The model calls this tool to retrieve synthesized insights from an existing mental model.

**Parameters:**
- `bankId` (required): Memory bank ID
- `mentalModelId` (required): ID of the mental model to query

**Example tool call:**
```typescript
{
  bankId: "user-123",
  mentalModelId: "mm-456"
}
```

**Returns:**
```typescript
{
  content: "The user prefers outdoor activities, particularly hiking. They enjoy mountain trails and visit Yosemite regularly during summer.",
  name: "User Preferences",
  updatedAt: "2024-01-20T15:45:00Z"
}
```

### `getDocument` - Retrieve Stored Document

The model calls this tool to retrieve a stored document by its ID.

**Parameters:**
- `bankId` (required): Memory bank ID
- `documentId` (required): ID of the document to retrieve

**Example tool call:**
```typescript
{
  bankId: "user-123",
  documentId: "doc-789"
}
```

**Returns:**
```typescript
{
  originalText: "User profile: Alice, Software Engineer, loves hiking...",
  id: "doc-789",
  createdAt: "2024-01-10T09:00:00Z",
  updatedAt: "2024-01-15T14:30:00Z"
}
```

### `createDirective` - Create Behavioral Rule

The model calls this tool to create a directive—a hard rule injected into prompts during reflect operations.

**Parameters:**
- `bankId` (required): Memory bank ID
- `name` (required): Human-readable name for the directive
- `content` (required): The directive text to inject
- `priority` (optional): Higher priority directives are injected first (default: 0)
- `isActive` (optional): Whether this directive is active (default: true)
- `tags` (optional): Tags for filtering (e.g., user-specific directives)

**Example tool call:**
```typescript
{
  bankId: "user-123",
  name: "Response Format",
  content: "Always provide responses in bullet-point format",
  priority: 10,
  tags: ["formatting"]
}
```

**Returns:**
```typescript
{
  id: "dir-321",
  name: "Response Format",
  content: "Always provide responses in bullet-point format",
  tags: ["formatting"],
  createdAt: "2024-01-15T10:30:00Z"
}
```

## Usage Examples

### Using with `generateText`

```typescript
import { HindsightClient } from '@vectorize-io/hindsight-client';
import { createHindsightTools } from '@vectorize-io/hindsight-ai-sdk';
import { generateText } from 'ai';
import { anthropic } from '@ai-sdk/anthropic';

const hindsightClient = new HindsightClient({
  apiUrl: 'http://localhost:8000',
});

const tools = createHindsightTools({ client: hindsightClient });

const result = await generateText({
  model: anthropic('claude-sonnet-4-20250514'),
  tools,
  system: `You are a helpful assistant with long-term memory. Use the recall tool to check for relevant memories before responding.`,
  prompt: 'Remember that Alice loves hiking and prefers spicy food',
});

console.log(result.text);
```

### Using with `streamText`

```typescript
import { streamText } from 'ai';

const result = streamText({
  model: anthropic('claude-sonnet-4-20250514'),
  tools,
  system: `You have persistent memory. Use retain to store important information and recall to retrieve it.`,
  prompt: 'What do you know about Alice?',
});

for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}
```

### Using with `ToolLoopAgent`

```typescript
import { ToolLoopAgent, stopWhen, stepCountIs } from 'ai';

const agent = new ToolLoopAgent({
  model: anthropic('claude-sonnet-4-20250514'),
  tools,
  instructions: `You are a personal assistant with long-term memory. Always check recall before responding and use retain to store important information.`,
  stopWhen: stepCountIs(10),
});

const result = await agent.generate({
  prompt: 'What did I say I wanted to work on this week?',
});
```

### Multi-User Support

```typescript
const result = await generateText({
  model: anthropic('claude-sonnet-4-20250514'),
  tools,
  system: `You are a helpful assistant. The user's ID is: ${userId}. Always pass this as the bankId parameter to memory tools.`,
  prompt: 'Remember that I prefer dark mode',
});
```
