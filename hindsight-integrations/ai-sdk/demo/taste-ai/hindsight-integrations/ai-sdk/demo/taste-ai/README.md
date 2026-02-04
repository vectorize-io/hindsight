# TasteAI - AI SDK Integration Demo

A demo application showcasing the Hindsight AI SDK integration with Vercel AI SDK 6. This is a food/recipe assistant that uses Hindsight for long-term memory to remember user preferences, dietary restrictions, and meal history.

## Features

- **Persistent Memory**: Uses Hindsight to remember user preferences, dietary restrictions, and meal history
- **AI SDK 6 Tools**: Demonstrates `retain`, `recall`, and `reflect` tools
- **Simple HTTP Client**: Shows how to implement the `HindsightClient` interface without external dependencies
- **Next.js App Router**: Built with Next.js 16 and Tailwind CSS

## Prerequisites

1. **Hindsight API**: You need a running Hindsight API server
   ```bash
   # Start Hindsight API (from project root)
   ./scripts/dev/start-api.sh

   # Or use embedded mode
   uvx hindsight-embed@latest -p demo daemon start
   ```

2. **LLM API Key**: You need an API key for an LLM provider (Groq, OpenAI, Anthropic, etc.)

## Setup

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Configure environment variables**:

   Create a `.env.local` file:
   ```bash
   # Hindsight API URL (default: http://localhost:8888)
   HINDSIGHT_URL=http://localhost:8888

   # LLM provider (Groq example)
   GROQ_MODEL=llama-3.3-70b-versatile
   GROQ_API_KEY=your-groq-api-key

   # Or use OpenAI
   # OPENAI_MODEL=gpt-4o-mini
   # OPENAI_API_KEY=your-openai-key
   ```

## Running the Demo

```bash
# Development mode
npm run dev

# Production build
npm run build
npm start
```

Visit [http://localhost:3000](http://localhost:3000)

## How It Works

### 1. Simple HTTP Client

The demo implements a simple HTTP client that conforms to the `HindsightClient` interface from `@vectorize-io/hindsight-ai-sdk`:

```typescript
import { createHindsightTools, type HindsightClient } from '@vectorize-io/hindsight-ai-sdk';

class SimpleHindsightClient implements HindsightClient {
  async retain(bankId, content, options) { /* ... */ }
  async recall(bankId, query, options) { /* ... */ }
  async reflect(bankId, query, options) { /* ... */ }
}

const hindsightClient = new SimpleHindsightClient(HINDSIGHT_URL);
```

### 2. AI SDK Tools

Create tools for use with AI SDK 6:

```typescript
export const hindsightTools = createHindsightTools({
  client: hindsightClient,
});

// Use with generateText, streamText, or ToolLoopAgent
const result = await generateText({
  model: groq('llama-3.3-70b-versatile'),
  tools: hindsightTools,
  prompt: 'Remember that I love spicy Thai food',
});
```

### 3. Memory Operations

The demo stores:
- **User preferences**: Dietary restrictions, cuisines, goals
- **Meal history**: What you've cooked or eaten
- **Health assessments**: Generated insights about eating patterns

All stored using Hindsight's document storage for structured data.

## Code Structure

```
src/
├── app/
│   ├── page.tsx              # Main page with onboarding
│   └── api/
│       ├── suggestions/      # Get meal suggestions
│       ├── log-meal/         # Log a meal
│       ├── preferences/      # Update preferences
│       └── dashboard/        # Get user stats
├── components/
│   ├── Onboarding.tsx        # Initial preference collection
│   ├── Dashboard.tsx         # Show meal history
│   ├── EatNow.tsx            # Meal suggestions
│   └── RecipeView.tsx        # Recipe details
└── lib/
    └── hindsight.ts          # Hindsight client & tools
```

## Key Learnings

1. **No Client Dependency Required**: The demo shows you can implement the `HindsightClient` interface with just `fetch` - no need for the full client library

2. **Type Safety**: TypeScript ensures your HTTP client implementation matches the interface exactly

3. **Document Storage**: Structured data (meals, preferences) stored as JSON documents for easy retrieval

4. **AI SDK Integration**: Seamless integration with AI SDK 6's tool calling system

## Troubleshooting

**"Failed to fetch"**: Make sure Hindsight API is running at the URL specified in `HINDSIGHT_URL`

**"No LLM configured"**: Set your LLM API key in `.env.local`

**TypeScript errors**: Rebuild the parent package:
```bash
cd ../..  # Go to hindsight-integrations/ai-sdk
npm run build
```

## License

MIT
