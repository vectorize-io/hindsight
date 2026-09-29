/**
 * Minimal Hindsight client surface used by this integration.
 *
 * It is a structural subset of `@vectorize-io/hindsight-client` so that
 * consumers can pass a real client instance without this package taking a hard
 * dependency on it. Only `recall` and `retain` are required.
 */

/** Processing budget controlling latency vs. depth. */
export type Budget = "low" | "mid" | "high";

/** Fact types for filtering recall results. */
export type FactType = "world" | "experience" | "observation";

/** A single recalled memory. */
export interface RecallResult {
  id: string;
  text: string;
  type?: string | null;
  entities?: string[] | null;
  context?: string | null;
  occurred_start?: string | null;
  occurred_end?: string | null;
  mentioned_at?: string | null;
  document_id?: string | null;
  metadata?: Record<string, string> | null;
  chunk_id?: string | null;
}

export interface RecallResponse {
  results: RecallResult[];
  trace?: Record<string, unknown> | null;
  entities?: Record<string, unknown> | null;
  chunks?: Record<string, unknown> | null;
}

export interface RetainResponse {
  success: boolean;
  bank_id: string;
  items_count: number;
  async: boolean;
}

/**
 * Hindsight client interface - matches `@vectorize-io/hindsight-client`.
 */
export interface HindsightClient {
  retain(
    bankId: string,
    content: string,
    options?: {
      timestamp?: Date | string;
      context?: string;
      metadata?: Record<string, string>;
      documentId?: string;
      tags?: string[];
      async?: boolean;
    }
  ): Promise<RetainResponse>;

  recall(
    bankId: string,
    query: string,
    options?: {
      types?: FactType[];
      maxTokens?: number;
      budget?: Budget;
      includeEntities?: boolean;
      includeChunks?: boolean;
    }
  ): Promise<RecallResponse>;
}
