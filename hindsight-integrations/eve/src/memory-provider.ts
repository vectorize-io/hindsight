/**
 * Hindsight as a first-class eve memory provider. One authored file gives an
 * agent long-term memory with no model tool-calling on the critical path:
 *
 * ```ts
 * // agent/memory/hindsight.ts
 * import { hindsightMemory } from "@vectorize-io/hindsight-eve";
 * import { defineMemory } from "eve/memory";
 * import { byPrincipal } from "eve/memory/scope";
 *
 * export default defineMemory({
 *   description: "Long-term memory about the current user.",
 *   provider: hindsightMemory(),
 *   scope: byPrincipal,
 * });
 * ```
 *
 * eve calls `recall` before the model runs (with the live user message as the
 * query), `capture` after the turn settles, and exposes `reflect` as a tool.
 * Every read and write is partitioned by the locked scope: one Hindsight bank
 * per scope. This is the only module that imports `eve`; the mechanics live in
 * `./provider-core` and `./client`, which unit-test without the framework.
 */
import {
  defineMemoryProvider,
  type MemoryCompactionCompletedContext,
  type MemoryProvider,
  type MemoryRecallResult,
  type MemoryToolSet,
  type MemoryToolsContext,
  type MemoryTurnCompletedContext,
  type MemoryTurnStartedContext,
} from "eve/memory";
import { defineTool } from "eve/tools";

import { HindsightRestClient } from "./client.js";
import { buildRetainContent } from "./config.js";
import {
  RECALL_MESSAGE_ID,
  buildRecallContent,
  buildRecallQuery,
  completedTurn,
  resolveMemoryProvider,
  type MemoryProviderOptions,
} from "./provider-core.js";

export type { MemoryProviderOptions } from "./provider-core.js";

const REFLECT_INPUT_SCHEMA = {
  type: "object",
  properties: {
    question: {
      type: "string",
      description: "The question to answer from long-term memory.",
    },
  },
  required: ["question"],
  additionalProperties: false,
} as const;

/**
 * Build the Hindsight memory provider for `defineMemory({ provider })`.
 * Configure via env (`HINDSIGHT_API_KEY`, `HINDSIGHT_API_URL`) or options.
 */
export function hindsightMemory(options: MemoryProviderOptions = {}): MemoryProvider {
  const cfg = resolveMemoryProvider(options);
  const client = new HindsightRestClient(cfg.apiUrl, cfg.apiKey, cfg.timeoutMs);

  // A failed recall must never fail the turn (eve fails the turn on a throwing
  // recall), so errors are reported and the turn proceeds without memory.
  const recall = async (
    ctx: MemoryTurnStartedContext | MemoryCompactionCompletedContext
  ): Promise<MemoryRecallResult> => {
    const bank = cfg.bank(ctx.memory.scope);
    const query = buildRecallQuery(ctx.turn?.input ?? [], cfg.recallQuery);
    try {
      const { results } = await client.recall(bank, query, {
        budget: cfg.budget,
        maxTokens: cfg.maxTokens,
        signal: ctx.abortSignal,
      });
      return { messages: [{ id: RECALL_MESSAGE_ID, content: buildRecallContent(results) }] };
    } catch (error) {
      if (!ctx.abortSignal.aborted) cfg.onError(error, "recall");
      return null;
    }
  };

  const capture = async (ctx: MemoryTurnCompletedContext): Promise<void> => {
    const content = buildRetainContent(
      completedTurn(ctx.turn.input, ctx.messages),
      cfg.includeAssistantReply
    );
    if (content === null) return;
    try {
      await client.retain(
        cfg.bank(ctx.memory.scope),
        [
          {
            content,
            context: cfg.context,
            metadata: {
              sessionId: ctx.session.id,
              turnId: ctx.turn.id,
              slot: ctx.memory.slot,
            },
            timestamp: new Date().toISOString(),
            // eve may replay the handler with the same operation id; a repeat
            // replaces the earlier document instead of storing it twice.
            document_id: ctx.operationId,
          },
        ],
        { async: true, signal: ctx.abortSignal }
      );
    } catch (error) {
      if (!ctx.abortSignal.aborted) cfg.onError(error, "capture");
    }
  };

  const tools = async (ctx: MemoryToolsContext): Promise<MemoryToolSet | null> => {
    if (!cfg.tools) return null;
    const bank = cfg.bank(ctx.memory.scope);
    return {
      reflect: defineTool({
        description:
          "Ask long-term memory a question and get a reasoned answer grounded in what " +
          "was remembered about this user across sessions. Use it when the answer " +
          "depends on past conversations that are not in the current context.",
        inputSchema: REFLECT_INPUT_SCHEMA,
        async execute(input, toolCtx) {
          const question = String(input.question ?? "");
          const { text } = await client.reflect(bank, question, { signal: toolCtx.abortSignal });
          return { answer: text };
        },
      }),
    };
  };

  return defineMemoryProvider({
    recall: {
      "turn.started": recall,
      "compaction.completed": recall,
    },
    ...(cfg.capture ? { capture: { "turn.completed": capture } } : {}),
    tools,
  });
}
