/**
 * Budget levels for recall/reflect operations.
 */
export type Budget = "low" | "mid" | "high";

/**
 * Fact types for filtering recall results.
 */
export type FactType = "world" | "experience" | "observation";

/**
 * Recall result item from Hindsight.
 */
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

/**
 * Entity state with observations.
 */
export interface EntityState {
  entity_id: string;
  canonical_name: string;
  observations: Array<{ text: string; mentioned_at?: string | null }>;
}

/**
 * Recall response from Hindsight.
 */
export interface RecallResponse {
  results: RecallResult[];
  trace?: Record<string, unknown> | null;
  entities?: Record<string, EntityState> | null;
}

/**
 * Reflect fact.
 */
export interface ReflectFact {
  id?: string | null;
  text: string;
  type?: string | null;
  context?: string | null;
  occurred_start?: string | null;
  occurred_end?: string | null;
}

/**
 * Reflect response from Hindsight.
 */
export interface ReflectResponse {
  text: string;
  based_on?: ReflectFact[];
}

/**
 * Retain response from Hindsight.
 */
export interface RetainResponse {
  success: boolean;
  bank_id: string;
  items_count: number;
  async: boolean;
}

/**
 * Hindsight client interface — matches @vectorize-io/hindsight-client.
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
      trace?: boolean;
      queryTimestamp?: string;
      includeEntities?: boolean;
      maxEntityTokens?: number;
      includeChunks?: boolean;
      maxChunkTokens?: number;
    }
  ): Promise<RecallResponse>;

  reflect(
    bankId: string,
    query: string,
    options?: {
      context?: string;
      budget?: Budget;
      maxTokens?: number;
    }
  ): Promise<ReflectResponse>;
}

/**
 * Resolves a bank ID from a message. Can be a static string or a function
 * that derives the bank ID from the message (e.g. per-user memory).
 */
export type BankIdResolver = string | ((message: ChatMessage) => string);

/**
 * Minimal message shape expected from the Chat SDK.
 * We declare our own interface so `chat` stays a peer dep only.
 */
export interface ChatMessage {
  author: {
    userId: string;
    name?: string;
    isMe?: boolean;
  };
  text: string;
  threadId: string;
  isMention?: boolean;
  metadata?: {
    timestamp?: Date;
  };
}

/**
 * Minimal thread shape expected from the Chat SDK.
 */
export interface ChatThread<TState = unknown> {
  post(message: unknown): Promise<unknown>;
  subscribe(): Promise<void>;
  unsubscribe(): Promise<void>;
  isSubscribed(): Promise<boolean>;
  startTyping(status?: string): Promise<void>;
  state: Promise<TState | null>;
  setState(state: TState): Promise<void>;
}

/**
 * Options for formatting memories as a system prompt.
 */
export interface MemoryPromptOptions {
  /** Custom preamble text before the memories section. */
  preamble?: string;
  /** Maximum number of memories to include. */
  maxMemories?: number;
  /** Filter to specific fact types. */
  includeTypes?: FactType[];
  /** Include entity observations section. */
  includeEntities?: boolean;
}

/**
 * Options for the auto-recall step.
 */
export interface RecallOptions {
  /** Enable auto-recall before the handler runs (default: true). */
  enabled?: boolean;
  /** Processing budget (default: 'mid'). */
  budget?: Budget;
  /** Maximum tokens for recall results. */
  maxTokens?: number;
  /** Filter to specific fact types. */
  types?: FactType[];
  /** Include entity observations (default: true). */
  includeEntities?: boolean;
  /** Tags to filter recall results. */
  tags?: string[];
}

/**
 * Options for the auto-retain step.
 */
export interface RetainOptions {
  /** Enable auto-retain of inbound messages (default: false). */
  enabled?: boolean;
  /** Fire-and-forget retain (default: true when enabled). */
  async?: boolean;
  /** Tags to attach to retained memories. */
  tags?: string[];
  /** Metadata to attach to retained memories. */
  metadata?: Record<string, string>;
}

/**
 * Configuration for withHindsightChat.
 */
export interface HindsightChatOptions {
  /** Hindsight client instance. */
  client: HindsightClient;
  /** Bank ID — static string or function deriving it from the message. */
  bankId: BankIdResolver;
  /** Auto-recall options (default: enabled). */
  recall?: RecallOptions;
  /** Auto-retain options for inbound messages (default: disabled). */
  retain?: RetainOptions;
}

/**
 * Context object passed to the wrapped handler, providing memory operations.
 */
export interface HindsightChatContext {
  /** Resolved bank ID for this message. */
  bankId: string;
  /** Recalled memories (empty array if recall disabled or failed). */
  memories: RecallResult[];
  /** Recalled entity observations (null if entities not included). */
  entities: Record<string, EntityState> | null;
  /** Format memories as a system prompt string. */
  memoriesAsSystemPrompt(options?: MemoryPromptOptions): string;
  /** Store content in memory. */
  retain(
    content: string,
    options?: {
      timestamp?: Date | string;
      context?: string;
      metadata?: Record<string, string>;
      tags?: string[];
      async?: boolean;
    }
  ): Promise<RetainResponse>;
  /** Search memories. */
  recall(
    query: string,
    options?: {
      types?: FactType[];
      maxTokens?: number;
      budget?: Budget;
      includeEntities?: boolean;
    }
  ): Promise<RecallResponse>;
  /** Reflect on memories to form insights. */
  reflect(
    query: string,
    options?: {
      context?: string;
      budget?: Budget;
      maxTokens?: number;
    }
  ): Promise<ReflectResponse>;
}

/**
 * The handler signature that withHindsightChat wraps.
 */
export type HindsightChatHandler<TState = unknown> = (
  thread: ChatThread<TState>,
  message: ChatMessage,
  ctx: HindsightChatContext
) => void | Promise<void>;
