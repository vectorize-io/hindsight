import { tool } from 'ai';
import { z } from 'zod';

/**
 * Budget levels for recall/reflect operations.
 */
export const BudgetSchema = z.enum(['low', 'mid', 'high']);
export type Budget = z.infer<typeof BudgetSchema>;

/**
 * Fact types for filtering recall results.
 */
export const FactTypeSchema = z.enum(['world', 'experience', 'observation']);
export type FactType = z.infer<typeof FactTypeSchema>;

/**
 * Recall result item from Hindsight
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
 * Entity state with observations
 */
export interface EntityState {
  entity_id: string;
  canonical_name: string;
  observations: Array<{ text: string; mentioned_at?: string | null }>;
}

/**
 * Chunk data
 */
export interface ChunkData {
  id: string;
  text: string;
  chunk_index: number;
  truncated?: boolean;
}

/**
 * Recall response from Hindsight
 */
export interface RecallResponse {
  results: RecallResult[];
  trace?: Record<string, unknown> | null;
  entities?: Record<string, EntityState> | null;
  chunks?: Record<string, ChunkData> | null;
}

/**
 * Reflect fact
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
 * Reflect response from Hindsight
 */
export interface ReflectResponse {
  text: string;
  based_on?: ReflectFact[];
}

/**
 * Retain response from Hindsight
 */
export interface RetainResponse {
  success: boolean;
  bank_id: string;
  items_count: number;
  async: boolean;
}

/**
 * Mental model trigger configuration
 */
export interface MentalModelTrigger {
  refresh_after_consolidation?: boolean;
}

/**
 * Mental model response from Hindsight
 */
export interface MentalModelResponse {
  mental_model_id: string;
  bank_id: string;
  name?: string;
  content?: string;
  source_query?: string;
  tags?: string[];
  created_at: string;
  updated_at: string;
  trigger?: MentalModelTrigger;
}

/**
 * Document response from Hindsight
 */
export interface DocumentResponse {
  id: string;
  bank_id: string;
  original_text: string;
  content_hash: string | null;
  created_at: string;
  updated_at: string;
  memory_unit_count: number;
  tags?: string[];
}

/**
 * Hindsight client interface - matches @vectorize-io/hindsight-client
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

  getMentalModel(
    bankId: string,
    mentalModelId: string
  ): Promise<MentalModelResponse>;

  getDocument(
    bankId: string,
    documentId: string
  ): Promise<DocumentResponse | null>;
}

export interface HindsightToolsOptions {
  /** Hindsight client instance */
  client: HindsightClient;
  /** Memory bank ID to use for all tool calls (e.g. the user ID) */
  bankId: string;

  /** Options for the retain tool */
  retain?: {
    /** Fire-and-forget retain without waiting for completion (default: false) */
    async?: boolean;
    /** Tags always attached to every retained memory (default: undefined) */
    tags?: string[];
    /** Metadata always attached to every retained memory (default: undefined) */
    metadata?: Record<string, string>;
    /** Custom tool description */
    description?: string;
  };

  /** Options for the recall tool */
  recall?: {
    /** Restrict results to these fact types: 'world', 'experience', 'observation' (default: undefined = all types) */
    types?: FactType[];
    /** Maximum tokens to return (default: undefined = API default) */
    maxTokens?: number;
    /** Processing budget controlling latency vs. depth (default: 'mid') */
    budget?: Budget;
    /** Include entity observations in results (default: false) */
    includeEntities?: boolean;
    /** Include raw source chunks in results (default: false) */
    includeChunks?: boolean;
    /** Custom tool description */
    description?: string;
  };

  /** Options for the reflect tool */
  reflect?: {
    /** Processing budget controlling latency vs. depth (default: 'mid') */
    budget?: Budget;
    /** Maximum tokens for the response (default: undefined = API default) */
    maxTokens?: number;
    /** Custom tool description */
    description?: string;
  };

  /** Options for the getMentalModel tool */
  getMentalModel?: {
    /** Custom tool description */
    description?: string;
  };

  /** Options for the getDocument tool */
  getDocument?: {
    /** Custom tool description */
    description?: string;
  };
}

/**
 * Creates AI SDK tools for Hindsight memory operations.
 *
 * The bank ID and all infrastructure concerns (budget, tags, async mode, etc.)
 * are fixed at creation time. The agent only controls semantic inputs:
 * content, queries, names, and timestamps.
 *
 * @example
 * ```ts
 * const tools = createHindsightTools({
 *   client: hindsightClient,
 *   bankId: userId,
 *   recall: { budget: 'high', includeEntities: true },
 *   retain: { async: true, tags: ['env:prod'] },
 * });
 *
 * const result = await generateText({
 *   model: openai('gpt-4o'),
 *   tools,
 *   messages,
 * });
 * ```
 */
export function createHindsightTools({
  client,
  bankId,
  retain: retainOpts = {},
  recall: recallOpts = {},
  reflect: reflectOpts = {},
  getMentalModel: getMentalModelOpts = {},
  getDocument: getDocumentOpts = {},
}: HindsightToolsOptions) {
  // Agent-controlled params only: content, timestamp, documentId, context
  const retainParams = z.object({
    content: z.string().describe('Content to store in memory'),
    documentId: z.string().optional().describe('Optional document ID for grouping/upserting content'),
    timestamp: z.string().optional().describe('Optional ISO timestamp for when the memory occurred'),
    context: z.string().optional().describe('Optional context about the memory'),
  });

  // Agent-controlled params only: query, queryTimestamp
  const recallParams = z.object({
    query: z.string().describe('What to search for in memory'),
    queryTimestamp: z.string().optional().describe('Query from a specific point in time (ISO format)'),
  });

  // Agent-controlled params only: query, context
  const reflectParams = z.object({
    query: z.string().describe('Question to reflect on based on memories'),
    context: z.string().optional().describe('Additional context for the reflection'),
  });

  const getMentalModelParams = z.object({
    mentalModelId: z.string().describe('ID of the mental model to retrieve'),
  });

  const getDocumentParams = z.object({
    documentId: z.string().describe('ID of the document to retrieve'),
  });

  type RetainInput = z.infer<typeof retainParams>;
  type RetainOutput = { success: boolean; itemsCount: number };

  type RecallInput = z.infer<typeof recallParams>;
  type RecallOutput = { results: RecallResult[]; entities?: Record<string, EntityState> | null };

  type ReflectInput = z.infer<typeof reflectParams>;
  type ReflectOutput = { text: string; basedOn?: ReflectFact[] };

  type GetMentalModelInput = z.infer<typeof getMentalModelParams>;
  type GetMentalModelOutput = { content: string; name?: string; updatedAt: string };

  type GetDocumentInput = z.infer<typeof getDocumentParams>;
  type GetDocumentOutput = { originalText: string; id: string; createdAt: string; updatedAt: string } | null;

  return {
    retain: tool<RetainInput, RetainOutput>({
      description:
        retainOpts.description ??
        `Store information in long-term memory. Use this when information should be remembered for future interactions, such as user preferences, facts, experiences, or important context.`,
      inputSchema: retainParams,
      execute: async (input) => {
        console.log('[AI SDK Tool] Retain input:', {
          bankId,
          documentId: input.documentId,
          hasContent: !!input.content,
        });
        const result = await client.retain(bankId, input.content, {
          documentId: input.documentId,
          timestamp: input.timestamp,
          context: input.context,
          tags: retainOpts.tags,
          metadata: retainOpts.metadata,
          async: retainOpts.async ?? false,
        });
        return { success: result.success, itemsCount: result.items_count };
      },
    }),

    recall: tool<RecallInput, RecallOutput>({
      description:
        recallOpts.description ??
        `Search memory for relevant information. Use this to find previously stored information that can help personalize responses or provide context.`,
      inputSchema: recallParams,
      execute: async (input) => {
        const result = await client.recall(bankId, input.query, {
          types: recallOpts.types,
          maxTokens: recallOpts.maxTokens,
          budget: recallOpts.budget ?? 'mid',
          queryTimestamp: input.queryTimestamp,
          includeEntities: recallOpts.includeEntities ?? false,
          includeChunks: recallOpts.includeChunks ?? false,
        });
        return {
          results: result.results ?? [],
          entities: result.entities,
        };
      },
    }),

    reflect: tool<ReflectInput, ReflectOutput>({
      description:
        reflectOpts.description ??
        `Analyze memories to form insights and generate contextual answers. Use this to understand patterns, synthesize information, or answer questions that require reasoning over stored memories.`,
      inputSchema: reflectParams,
      execute: async (input) => {
        const result = await client.reflect(bankId, input.query, {
          context: input.context,
          budget: reflectOpts.budget ?? 'mid',
          maxTokens: reflectOpts.maxTokens,
        });
        return {
          text: result.text ?? 'No insights available yet.',
          basedOn: result.based_on,
        };
      },
    }),

    getMentalModel: tool<GetMentalModelInput, GetMentalModelOutput>({
      description:
        getMentalModelOpts.description ??
        `Retrieve a mental model to get consolidated knowledge synthesized from memories. Mental models provide synthesized insights that are faster and more efficient to retrieve than searching through raw memories.`,
      inputSchema: getMentalModelParams,
      execute: async (input) => {
        const result = await client.getMentalModel(bankId, input.mentalModelId);
        return {
          content: result.content ?? 'No content available yet.',
          name: result.name,
          updatedAt: result.updated_at,
        };
      },
    }),

    getDocument: tool<GetDocumentInput, GetDocumentOutput>({
      description:
        getDocumentOpts.description ??
        `Retrieve a stored document by its ID. Documents are used to store structured data like application state, user profiles, or any data that needs exact retrieval.`,
      inputSchema: getDocumentParams,
      execute: async (input) => {
        const result = await client.getDocument(bankId, input.documentId);
        if (!result) {
          return null;
        }
        return {
          originalText: result.original_text,
          id: result.id,
          createdAt: result.created_at,
          updatedAt: result.updated_at,
        };
      },
    }),

  };
}

export type HindsightTools = ReturnType<typeof createHindsightTools>;
