import { formatMemoriesAsSystemPrompt } from "./format.js";
import type {
  HindsightChatOptions,
  HindsightChatContext,
  HindsightChatHandler,
  ChatThread,
  ChatMessage,
  RecallResult,
  EntityState,
  MemoryPromptOptions,
} from "./types.js";

/**
 * Wraps a Chat SDK handler to automatically provide Hindsight memory context.
 *
 * Before the handler runs:
 * 1. Resolves the bank ID from the message
 * 2. Optionally auto-retains the inbound message (off by default)
 * 3. Auto-recalls relevant memories (on by default)
 * 4. Builds a HindsightChatContext and passes it to the handler
 *
 * Works with `onNewMention`, `onSubscribedMessage`, and `onNewMessage`.
 *
 * @example
 * ```ts
 * chat.onNewMention(
 *   withHindsightChat(
 *     { client, bankId: (msg) => msg.author.userId },
 *     async (thread, message, ctx) => {
 *       const result = await streamText({
 *         system: ctx.memoriesAsSystemPrompt(),
 *         messages: [{ role: 'user', content: message.text }],
 *       });
 *       await thread.post(result.textStream);
 *       await ctx.retain(`User: ${message.text}\nAssistant: ${fullResponse}`);
 *     }
 *   )
 * );
 * ```
 */
export function withHindsightChat<TState = unknown>(
  options: HindsightChatOptions,
  handler: HindsightChatHandler<TState>
): (thread: ChatThread<TState>, message: ChatMessage) => Promise<void> {
  const {
    client,
    bankId: bankIdResolver,
    recall: recallOpts = {},
    retain: retainOpts = {},
  } = options;

  const recallEnabled = recallOpts.enabled !== false;
  const retainEnabled = retainOpts.enabled === true;
  const retainAsync = retainOpts.async !== false; // default true when retain is enabled

  return async (thread: ChatThread<TState>, message: ChatMessage): Promise<void> => {
    // 1. Resolve bank ID
    const resolvedBankId =
      typeof bankIdResolver === "function" ? bankIdResolver(message) : bankIdResolver;

    // 2. Auto-retain inbound message (fire-and-forget if async)
    if (retainEnabled && message.text && !message.author.isMe) {
      const retainPromise = client
        .retain(resolvedBankId, message.text, {
          tags: retainOpts.tags,
          metadata: retainOpts.metadata,
          async: retainAsync,
        })
        .catch((err) => {
          console.warn("[hindsight-chat] Auto-retain failed:", err);
        });

      // If not async, wait for retain to complete before proceeding
      if (!retainAsync) {
        await retainPromise;
      }
    }

    // 3. Auto-recall memories
    let memories: RecallResult[] = [];
    let entities: Record<string, EntityState> | null = null;

    if (recallEnabled && message.text) {
      try {
        const recallResponse = await client.recall(resolvedBankId, message.text, {
          budget: recallOpts.budget ?? "mid",
          maxTokens: recallOpts.maxTokens,
          types: recallOpts.types,
          includeEntities: recallOpts.includeEntities !== false,
        });
        memories = recallResponse.results ?? [];
        entities = recallResponse.entities ?? null;
      } catch (err) {
        console.warn("[hindsight-chat] Auto-recall failed:", err);
      }
    }

    // 4. Build context
    const ctx: HindsightChatContext = {
      bankId: resolvedBankId,
      memories,
      entities,

      memoriesAsSystemPrompt(promptOptions?: MemoryPromptOptions): string {
        return formatMemoriesAsSystemPrompt(memories, entities, promptOptions);
      },

      retain(content, retainCallOpts) {
        return client.retain(resolvedBankId, content, retainCallOpts);
      },

      recall(query, recallCallOpts) {
        return client.recall(resolvedBankId, query, recallCallOpts);
      },

      reflect(query, reflectCallOpts) {
        return client.reflect(resolvedBankId, query, reflectCallOpts);
      },
    };

    // 5. Call the user's handler
    await handler(thread, message, ctx);
  };
}
