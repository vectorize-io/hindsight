# TasteAI Demo

Personal food assistant showcasing `@anthropic/hindsight-ai-sdk` with AI SDK v6.

## Flow

1. **Onboarding** - User sets cuisine preferences, dietary restrictions, goals
2. **Suggestions** - AI generates personalized recipes using memory context
3. **Logging** - User logs meals (ate today/yesterday, cook, never eat)
4. **Health tracking** - AI reflects on eating patterns to compute health score

## Hindsight Integration

Uses `@anthropic/hindsight-ai-sdk` tools with AI SDK v6's `generateText`:

```ts
import { createHindsightTools } from '@anthropic/hindsight-ai-sdk';

const tools = createHindsightTools({ client: hindsightClient });

const result = await generateText({
  model: llmModel,
  tools: {
    recall: tools.recall,   // Search user memories
    reflect: tools.reflect, // Analyze patterns
  },
  toolChoice: 'auto',
  prompt: 'Get user food preferences...',
});
```

**Tools used:**
- `retain` - Store preferences, meals, dislikes
- `recall` - Search memories for personalization
- `reflect` - Generate health insights from eating patterns

**Document storage:**
- Meals stored as JSON document with `document_id` for reliable retrieval
- Health score cached in document, recalculated after meal changes

## AI SDK v6 Features

- **Tool calling with `generateText`** - Agent loop for context gathering
- **Dynamic tool parameters** - `bankId` passed per call for multi-user support

## Run

```bash
npm run dev
```

Requires Hindsight server at `http://localhost:8888` (or set `HINDSIGHT_URL`).
