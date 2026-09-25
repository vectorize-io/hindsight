/**
 * Hindsight long-term memory for Vercel Eve agents.
 *
 * One authored file gives an Eve agent memory that just works: relevant memory
 * is recalled on the live user message before each turn, the exchange is
 * retained after, and every scope gets its own Hindsight bank.
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
 * Configure via env: `HINDSIGHT_API_KEY`, `HINDSIGHT_API_URL` (defaults to
 * Hindsight Cloud).
 */
export { hindsightMemory, type MemoryProviderOptions } from "./memory-provider.js";

export {
  bankFromScope,
  buildRecallContent,
  buildRecallQuery,
  completedTurn,
  resolveMemoryProvider,
  EMPTY_RECALL_CONTENT,
  RECALL_MESSAGE_ID,
  type BankResolver,
  type MemoryPhase,
  type MemoryScopeLike,
  type ResolvedMemoryProvider,
} from "./provider-core.js";

// v0.2 hooks/instructions entrypoints — deprecated, removed in the next release.
export { hindsightAutoRecall, hindsightRetainHook, type AutoMemoryOptions } from "./auto-memory.js";

export {
  resolveAutoMemory,
  resolveConnection,
  isHindsightCloudUrl,
  HINDSIGHT_CLOUD_API_URL,
  DEFAULT_RECALL_QUERY,
  DEFAULT_BANK_ID,
  type ConnectionOptions,
  type ResolvedConnection,
  type ResolvedAutoMemory,
} from "./config.js";

export {
  HindsightRestClient,
  HindsightHttpError,
  buildRecallMarkdown,
  stripSentinelBlocks,
  type RecallResult,
  type RecallResponse,
  type ReflectResponse,
  type RetainItem,
  type RecallBudget,
  type RecallOptions,
  type RetainOptions,
  type ReflectOptions,
} from "./client.js";
