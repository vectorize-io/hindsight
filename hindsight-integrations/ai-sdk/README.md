# Hindsight Memory Integration for Vercel AI SDK

Give your AI agents persistent, human-like memory using [Hindsight](https://vectorize.io/hindsight) with the [Vercel AI SDK](https://ai-sdk.dev).

## Quick Start

```bash
npm install @vectorize-io/hindsight-ai-sdk @vectorize-io/hindsight-client ai zod
```

```typescript
import { HindsightClient } from '@vectorize-io/hindsight-client';
import { createHindsightTools } from '@vectorize-io/hindsight-ai-sdk';
import { generateText } from 'ai';
import { anthropic } from '@ai-sdk/anthropic';

// 1. Initialize Hindsight client
const hindsightClient = new HindsightClient({
  apiUrl: 'http://localhost:8000',
});

// 2. Create memory tools
const tools = createHindsightTools({ client: hindsightClient });

// 3. Use with AI SDK
const result = await generateText({
  model: anthropic('claude-sonnet-4-20250514'),
  tools,
  system: `You have long-term memory. Use:
  - 'recall' to search past conversations
  - 'retain' to remember important information
  - 'reflect' to synthesize insights from memories`,
  prompt: 'Remember that Alice loves hiking and prefers spicy food',
});

console.log(result.text);
```

## Features

✅ **Three Memory Tools**: `retain` (store), `recall` (retrieve), and `reflect` (reason over memories)
✅ **AI SDK 6 Native**: Works with `generateText`, `streamText`, and `ToolLoopAgent`
✅ **Multi-User Support**: Dynamic bank IDs per call for multi-user scenarios
✅ **Type-Safe**: Full TypeScript support with Zod schemas
✅ **Flexible Client**: Works with the official TypeScript client or custom HTTP clients

## Documentation

📖 **[Full Documentation](https://vectorize.io/hindsight/sdks/integrations/ai-sdk)**

The complete documentation includes:
- Detailed tool descriptions and parameters
- Advanced usage patterns (streaming, multi-user, ToolLoopAgent)
- HTTP client example (no dependencies)
- TypeScript types and API reference
- Best practices and system prompt examples

## Running Hindsight Locally

```bash
# Install and run with embedded mode (no setup required)
uvx hindsight-embed@latest -p myapp daemon start

# The API will be available at http://localhost:8000
```

## Examples

Full examples are available in the [GitHub repository](https://github.com/vectorize-io/hindsight/tree/main/examples/ai-sdk).

## Support

- [Documentation](https://vectorize.io/hindsight)
- [GitHub Issues](https://github.com/vectorize-io/hindsight/issues)
- Email: support@vectorize.io

## License

MIT
