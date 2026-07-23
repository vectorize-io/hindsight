/**
 * Automatic, no-tool long-term memory for Vercel Eve agents, backed by
 * Hindsight's REST API. Two authored files give an agent memory that works
 * without the model ever choosing to call a tool:
 *
 * ```ts
 * // agent/instructions/hindsight.ts  — recall: inject memory before each turn
 * import { hindsightMemory } from "@vectorize-io/hindsight-eve";
 * export default hindsightMemory();
 *
 * // agent/hooks/hindsight.ts          — retain: save each exchange after the turn
 * import { hindsightRetainHook } from "@vectorize-io/hindsight-eve";
 * export default hindsightRetainHook();
 * ```
 *
 * This module is the only one that imports `eve`. The HTTP client and config
 * resolution are kept pure (in `./client` and `./config`) so they unit-test
 * without the framework.
 */
import { defineHook, type HookDefinition } from "eve/hooks";
import { defineDynamic, defineInstructions, type DynamicSentinel } from "eve/instructions";

import { HindsightRestClient, buildRecallMarkdown } from "./client.js";
import {
  buildRetainContent,
  recordAssistantMessage,
  recordUserMessage,
  resolveAutoMemory,
  takeTurn,
  type AutoMemoryOptions,
  type TurnBuffer,
} from "./config.js";

export type { AutoMemoryOptions } from "./config.js";

/**
 * Inject the user's stored memory as a system message before each turn.
 * Drop the returned value as the default export of `agent/instructions/hindsight.ts`.
 *
 * Recall uses a fixed broad query (not the live message — eve's instruction
 * resolver can't see it), which surfaces the user's ambient profile/context.
 * Tune it with `recallQuery`.
 */
export function hindsightAutoRecall(options: AutoMemoryOptions = {}): DynamicSentinel {
  const cfg = resolveAutoMemory(options);
  const client = new HindsightRestClient(cfg.apiUrl, cfg.apiKey, cfg.timeoutMs);

  return defineDynamic({
    events: {
      "turn.started": async (): Promise<unknown> => {
        try {
          const { results } = await client.recall(cfg.bankId, cfg.recallQuery, {
            budget: cfg.budget,
            maxTokens: cfg.maxTokens,
          });
          if (results.length === 0) return undefined;
          return defineInstructions({ markdown: buildRecallMarkdown(results) });
        } catch (error) {
          cfg.onError(error, "recall");
          return undefined;
        }
      },
    },
  });
}

/** Primary name for {@link hindsightAutoRecall} — the memory-injection half. */
export const hindsightMemory = hindsightAutoRecall;

/**
 * Retain each completed exchange to Hindsight. Drop the returned value as the
 * default export of `agent/hooks/hindsight.ts`.
 *
 * Pairs the user message (`message.received`) with the final assistant answer
 * (`message.completed` where `finishReason === "stop"`) by `turnId`, then
 * retains on `turn.completed`. All side effects are guarded — a failure warns
 * via `onError` and never breaks the turn.
 */
export function hindsightRetainHook(options: AutoMemoryOptions = {}): HookDefinition {
  const cfg = resolveAutoMemory(options);
  const client = new HindsightRestClient(cfg.apiUrl, cfg.apiKey, cfg.timeoutMs);
  const buffer: TurnBuffer = new Map();

  return defineHook({
    events: {
      "message.received": (event) => {
        recordUserMessage(buffer, event.data.turnId, event.data.message);
      },
      "message.completed": (event) => {
        // Only the terminal assistant text; intermediate steps end in "tool-calls".
        if (event.data.finishReason === "stop" && event.data.message) {
          recordAssistantMessage(buffer, event.data.turnId, event.data.message);
        }
      },
      "turn.completed": async (event, ctx) => {
        try {
          const content = buildRetainContent(
            takeTurn(buffer, event.data.turnId),
            cfg.includeAssistantReply
          );
          if (content === null) return;
          const metadata: Record<string, string> = {
            sessionId: ctx.session.id,
            turnId: event.data.turnId,
          };
          if (ctx.channel.kind) metadata.channel = ctx.channel.kind;
          await client.retain(
            cfg.bankId,
            [{ content, context: cfg.context, metadata, timestamp: new Date().toISOString() }],
            { async: true }
          );
        } catch (error) {
          cfg.onError(error, "retain");
        }
      },
    },
  });
}
