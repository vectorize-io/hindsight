import type { Memory } from "@elizaos/core";
import type { Budget, FactType, HindsightClient } from "./client.js";

/**
 * Resolves which Hindsight memory bank a message belongs to.
 *
 * A string is used as a fixed bank for every message; a function lets callers
 * derive the bank per message (e.g. one bank per user or per room).
 */
export type BankResolver = string | ((message: Memory) => string);

export interface RecallOptions {
  /** Disable memory recall (default: enabled). */
  enabled?: boolean;
  /** Restrict recall to these fact types (default: all). */
  types?: FactType[];
  /** Maximum tokens to return (default: API default). */
  maxTokens?: number;
  /** Processing budget controlling latency vs. depth (default: 'mid'). */
  budget?: Budget;
  /** Include entity observations in recall (default: false). */
  includeEntities?: boolean;
  /** Heading rendered above the recalled memories in the prompt. */
  heading?: string;
}

export interface RetainOptions {
  /** Disable retaining messages (default: enabled). */
  enabled?: boolean;
  /** Fire-and-forget retain without awaiting completion (default: true). */
  async?: boolean;
  /** Tags attached to every retained memory. */
  tags?: string[];
  /** Metadata attached to every retained memory. */
  metadata?: Record<string, string>;
  /** Also retain the agent's own replies, not just incoming messages (default: false). */
  includeAgentMessages?: boolean;
}

export interface HindsightPluginOptions {
  /** A Hindsight client instance (e.g. from `@vectorize-io/hindsight-client`). */
  client: HindsightClient;
  /**
   * The memory bank to read and write. Defaults to the message's `entityId`,
   * giving each user/agent an isolated memory store.
   */
  bank?: BankResolver;
  /** Recall (read) behaviour. */
  recall?: RecallOptions;
  /** Retain (write) behaviour. */
  retain?: RetainOptions;
}

/** Resolve the bank id for a message, defaulting to its `entityId`. */
export function resolveBank(bank: BankResolver | undefined, message: Memory): string {
  if (typeof bank === "function") return bank(message);
  if (typeof bank === "string" && bank.length > 0) return bank;
  return message.entityId;
}
