/**
 * Hindsight long-term memory for Vercel Eve agents — automatic, no-tool memory.
 *
 * Two authored files give an Eve agent memory that just works, without the model
 * ever deciding to call a tool: relevant memory is injected before each turn, and
 * the exchange is retained after.
 *
 * ```ts
 * // agent/instructions/hindsight.ts
 * import { hindsightMemory } from "@vectorize-io/hindsight-eve";
 * export default hindsightMemory();
 *
 * // agent/hooks/hindsight.ts
 * import { hindsightRetainHook } from "@vectorize-io/hindsight-eve";
 * export default hindsightRetainHook();
 * ```
 *
 * Configure via env: `HINDSIGHT_API_KEY`, `HINDSIGHT_API_URL` (defaults to
 * Hindsight Cloud), `HINDSIGHT_BANK_ID`.
 */
export {
  hindsightMemory,
  hindsightAutoRecall,
  hindsightRetainHook,
  type AutoMemoryOptions,
} from "./auto-memory.js";

export {
  resolveAutoMemory,
  isHindsightCloudUrl,
  HINDSIGHT_CLOUD_API_URL,
  DEFAULT_RECALL_QUERY,
  DEFAULT_BANK_ID,
  type ResolvedAutoMemory,
} from "./config.js";

export {
  HindsightRestClient,
  buildRecallMarkdown,
  stripSentinelBlocks,
  type RecallResult,
  type RecallResponse,
  type RetainItem,
  type RecallBudget,
  type RecallOptions,
} from "./client.js";
