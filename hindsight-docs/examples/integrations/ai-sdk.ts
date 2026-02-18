/**
 * Hindsight AI SDK integration examples
 * These snippets are embedded in the documentation via CodeSnippet.
 */

// [docs:setup]
import { HindsightClient } from '@vectorize-io/hindsight-client';
import { createHindsightTools } from '@vectorize-io/hindsight-ai-sdk';

const client = new HindsightClient({ baseUrl: 'http://localhost:8888' });

const tools = createHindsightTools({
  client,
  bankId: 'user-123',
});
// [/docs:setup]

// [docs:generate-text]
import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

const { text } = await generateText({
  model: openai('gpt-4o'),
  tools,
  maxSteps: 5,
  system: 'You are a helpful assistant with long-term memory.',
  prompt: 'Remember that I prefer dark mode and large fonts.',
});
// [/docs:generate-text]

// [docs:stream-text]
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
// [/docs:stream-text]

// [docs:tool-loop-agent]
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
// [/docs:tool-loop-agent]

// [docs:next-api-route]
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
// [/docs:next-api-route]

// [docs:constructor-options]
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

// [/docs:constructor-options]
