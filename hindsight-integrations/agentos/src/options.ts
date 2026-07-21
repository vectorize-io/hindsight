import type { Budget, FactType, HindsightClient } from "./client.js";

export interface RecallOptions {
  /** Disable memory recall / context injection (default: enabled). */
  enabled?: boolean;
  /** Restrict recall to these fact types (default: all). */
  types?: FactType[];
  /**
   * Maximum tokens to return. When omitted, the `tokenBudget` AgentOS passes to
   * `getContext` is used (defaults to AgentOS's own budget).
   */
  maxTokens?: number;
  /** Processing budget controlling latency vs. depth (default: 'mid'). */
  budget?: Budget;
  /** Include entity observations in recall (default: false). */
  includeEntities?: boolean;
  /** Heading rendered above the recalled memories in the injected context. */
  heading?: string;
  /**
   * Prefix each recalled memory with its fact kind, e.g. `[world]` / `[experience]`
   * (default: false). Keeps provenance visible when the model reasons over the
   * injected block.
   */
  labelTypes?: boolean;
}

export interface RetainOptions {
  /** Disable retaining turns (default: enabled). */
  enabled?: boolean;
  /** Fire-and-forget retain without awaiting completion (default: true). */
  async?: boolean;
  /** Tags attached to every retained memory. */
  tags?: string[];
  /** Metadata attached to every retained memory. */
  metadata?: Record<string, string>;
  /**
   * Also retain the agent's own replies, not just the user's turns
   * (default: false).
   */
  includeAgentMessages?: boolean;
}

export interface HindsightMemoryOptions {
  /** A Hindsight client instance (e.g. from `@vectorize-io/hindsight-client`). */
  client: HindsightClient;
  /**
   * The memory bank this agent reads and writes. AgentOS memory-provider hooks
   * receive only turn text (no per-message routing context), so a provider
   * instance maps to exactly one bank — give each agent/user its own bank for
   * isolation. Defaults to `"default"`.
   */
  bank?: string;
  /** Recall (read) behaviour. */
  recall?: RecallOptions;
  /** Retain (write) behaviour. */
  retain?: RetainOptions;
}

/** Default bank used when none is configured. */
export const DEFAULT_BANK = "default";
