import { tool } from 'ai';
import { z } from 'zod';

/**
 * Budget levels for recall/reflect operations.
 */
export const BudgetSchema = z.enum(['low', 'mid', 'high']);
export type Budget = z.infer<typeof BudgetSchema>;

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
 * Create mental model response from Hindsight
 */
export interface CreateMentalModelResponse {
  mental_model_id: string;
  bank_id: string;
  created_at: string;
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
 * Directive response from Hindsight
 */
export interface DirectiveResponse {
  id: string;
  bank_id: string;
  name: string;
  content: string;
  priority: number;
  is_active: boolean;
  tags: string[];
  created_at: string;
  updated_at: string;
}

/**
 * Create directive response from Hindsight
 */
export interface CreateDirectiveResponse {
  id: string;
  bank_id: string;
  name: string;
  content: string;
  priority: number;
  is_active: boolean;
  tags: string[];
  created_at: string;
  updated_at: string;
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
      types?: string[];
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
    }
  ): Promise<ReflectResponse>;

  createMentalModel(
    bankId: string,
    options?: {
      id?: string;
      name?: string;
      sourceQuery?: string;
      tags?: string[];
      maxTokens?: number;
      trigger?: MentalModelTrigger;
    }
  ): Promise<CreateMentalModelResponse>;

  getMentalModel(
    bankId: string,
    mentalModelId: string
  ): Promise<MentalModelResponse>;

  getDocument(
    bankId: string,
    documentId: string
  ): Promise<DocumentResponse | null>;

  createDirective(
    bankId: string,
    options: {
      name: string;
      content: string;
      priority?: number;
      isActive?: boolean;
      tags?: string[];
    }
  ): Promise<CreateDirectiveResponse>;

  getDirective(
    bankId: string,
    directiveId: string
  ): Promise<DirectiveResponse | null>;

  listDirectives(
    bankId: string,
    options?: {
      tags?: string[];
      tagsMatch?: 'any' | 'all' | 'exact';
      activeOnly?: boolean;
      limit?: number;
      offset?: number;
    }
  ): Promise<{ directives: DirectiveResponse[]; total: number }>;
}

export interface HindsightToolsOptions {
  /** Hindsight client instance */
  client: HindsightClient;
  /**
   * Custom description for the retain tool.
   */
  retainDescription?: string;
  /**
   * Custom description for the recall tool.
   */
  recallDescription?: string;
  /**
   * Custom description for the reflect tool.
   */
  reflectDescription?: string;
  /**
   * Custom description for the createMentalModel tool.
   */
  createMentalModelDescription?: string;
  /**
   * Custom description for the queryMentalModel tool.
   */
  queryMentalModelDescription?: string;
  /**
   * Custom description for the getDocument tool.
   */
  getDocumentDescription?: string;
}

/**
 * Creates AI SDK tools for Hindsight memory operations.
 *
 * Features:
 * - Dynamic bank ID per call (supports multi-user/multi-bank scenarios)
 * - Full API parameter support for retain, recall, and reflect
 * - Ready to use with streamText, generateText, or ToolLoopAgent
 *
 * @example
 * ```ts
 * const tools = createHindsightTools({
 *   client: hindsightClient,
 * });
 *
 * // Use with AI SDK
 * const result = await generateText({
 *   model: openai('gpt-4'),
 *   tools,
 *   prompt: 'Remember that Alice loves hiking',
 * });
 * ```
 */
export function createHindsightTools({
  client,
  retainDescription,
  recallDescription,
  reflectDescription,
  createMentalModelDescription,
  queryMentalModelDescription,
  getDocumentDescription,
}: HindsightToolsOptions) {
  const retainParams = z.object({
    bankId: z.string().describe('Memory bank ID (usually the user ID)'),
    content: z.string().describe('Content to store in memory'),
    documentId: z.string().optional().describe('Optional document ID for grouping/upserting content'),
    timestamp: z.string().optional().describe('Optional ISO timestamp for when the memory occurred'),
    context: z.string().optional().describe('Optional context about the memory'),
    tags: z.array(z.string()).optional().describe('Optional tags for visibility scoping'),
    metadata: z.record(z.string(), z.string()).optional().describe('Optional user-defined metadata'),
  });

  const recallParams = z.object({
    bankId: z.string().describe('Memory bank ID (usually the user ID)'),
    query: z.string().describe('What to search for in memory'),
    types: z.array(z.string()).optional().describe('Filter by fact types'),
    maxTokens: z.number().optional().describe('Maximum tokens to return'),
    budget: BudgetSchema.optional().describe('Processing budget: low, mid, or high'),
    queryTimestamp: z.string().optional().describe('Query from a specific point in time (ISO format)'),
    includeEntities: z.boolean().optional().describe('Include entity observations in results'),
    includeChunks: z.boolean().optional().describe('Include raw chunks in results'),
  });

  const reflectParams = z.object({
    bankId: z.string().describe('Memory bank ID (usually the user ID)'),
    query: z.string().describe('Question to reflect on based on memories'),
    context: z.string().optional().describe('Additional context for the reflection'),
    budget: BudgetSchema.optional().describe('Processing budget: low, mid, or high'),
  });

  const createMentalModelParams = z.object({
    bankId: z.string().describe('Memory bank ID (usually the user ID)'),
    mentalModelId: z.string().optional().describe('Optional custom ID for the mental model (auto-generated if not provided)'),
    name: z.string().optional().describe('Optional name for the mental model'),
    sourceQuery: z.string().optional().describe('Query to define what memories to consolidate'),
    tags: z.array(z.string()).optional().describe('Optional tags for organizing mental models'),
    maxTokens: z.number().optional().describe('Maximum tokens for the mental model content'),
    autoRefresh: z.boolean().optional().describe('Auto-refresh mental model after new consolidations (default: false)'),
  });

  const queryMentalModelParams = z.object({
    bankId: z.string().describe('Memory bank ID (usually the user ID)'),
    mentalModelId: z.string().describe('ID of the mental model to query'),
  });

  const getDocumentParams = z.object({
    bankId: z.string().describe('Memory bank ID (usually the user ID)'),
    documentId: z.string().describe('ID of the document to retrieve'),
  });

  const createDirectiveParams = z.object({
    bankId: z.string().describe('Memory bank ID (usually the user ID)'),
    name: z.string().describe('Human-readable name for the directive'),
    content: z.string().describe('The directive text to inject into prompts'),
    priority: z.number().optional().describe('Higher priority directives are injected first (default 0)'),
    isActive: z.boolean().optional().describe('Whether this directive is active (default true)'),
    tags: z.array(z.string()).optional().describe('Tags for filtering'),
  });

  const getDirectiveParams = z.object({
    bankId: z.string().describe('Memory bank ID (usually the user ID)'),
    directiveId: z.string().describe('ID of the directive to retrieve'),
  });

  type RetainInput = z.infer<typeof retainParams>;
  type RetainOutput = { success: boolean; itemsCount: number };

  type RecallInput = z.infer<typeof recallParams>;
  type RecallOutput = { results: RecallResult[]; entities?: Record<string, EntityState> | null };

  type ReflectInput = z.infer<typeof reflectParams>;
  type ReflectOutput = { text: string; basedOn?: ReflectFact[] };

  type CreateMentalModelInput = z.infer<typeof createMentalModelParams>;
  type CreateMentalModelOutput = { mentalModelId: string; createdAt: string };

  type QueryMentalModelInput = z.infer<typeof queryMentalModelParams>;
  type QueryMentalModelOutput = { content: string; name?: string; updatedAt: string };

  type GetDocumentInput = z.infer<typeof getDocumentParams>;
  type GetDocumentOutput = { originalText: string; id: string; createdAt: string; updatedAt: string } | null;

  type CreateDirectiveInput = z.infer<typeof createDirectiveParams>;
  type CreateDirectiveOutput = { id: string; name: string; content: string; tags: string[]; createdAt: string };

  type GetDirectiveInput = z.infer<typeof getDirectiveParams>;
  type GetDirectiveOutput = { id: string; name: string; content: string; tags: string[]; isActive: boolean } | null;

  return {
    retain: tool<RetainInput, RetainOutput>({
      description:
        retainDescription ??
        `Store information in long-term memory. Use this when information should be remembered for future interactions, such as user preferences, facts, experiences, or important context.`,
      inputSchema: retainParams,
      execute: async (input) => {
        console.log('[AI SDK Tool] Retain input:', {
          bankId: input.bankId,
          documentId: input.documentId,
          tags: input.tags,
          hasContent: !!input.content,
        });
        const result = await client.retain(input.bankId, input.content, {
          documentId: input.documentId,
          timestamp: input.timestamp,
          context: input.context,
          tags: input.tags,
          metadata: input.metadata as Record<string, string> | undefined,
        });
        return { success: result.success, itemsCount: result.items_count };
      },
    }),

    recall: tool<RecallInput, RecallOutput>({
      description:
        recallDescription ??
        `Search memory for relevant information. Use this to find previously stored information that can help personalize responses or provide context.`,
      inputSchema: recallParams,
      execute: async (input) => {
        const result = await client.recall(input.bankId, input.query, {
          types: input.types,
          maxTokens: input.maxTokens,
          budget: input.budget,
          queryTimestamp: input.queryTimestamp,
          includeEntities: input.includeEntities,
          includeChunks: input.includeChunks,
        });
        return {
          results: result.results ?? [],
          entities: result.entities,
        };
      },
    }),

    reflect: tool<ReflectInput, ReflectOutput>({
      description:
        reflectDescription ??
        `Analyze memories to form insights and generate contextual answers. Use this to understand patterns, synthesize information, or answer questions that require reasoning over stored memories.`,
      inputSchema: reflectParams,
      execute: async (input) => {
        const result = await client.reflect(input.bankId, input.query, {
          context: input.context,
          budget: input.budget,
        });
        return {
          text: result.text ?? 'No insights available yet.',
          basedOn: result.based_on,
        };
      },
    }),

    createMentalModel: tool<CreateMentalModelInput, CreateMentalModelOutput>({
      description:
        createMentalModelDescription ??
        `Create a mental model that automatically consolidates memories into structured knowledge. Mental models are continuously updated as new memories are added, making them ideal for maintaining up-to-date user preferences, behavioral patterns, and accumulated wisdom.`,
      inputSchema: createMentalModelParams,
      execute: async (input) => {
        const result = await client.createMentalModel(input.bankId, {
          id: input.mentalModelId,
          name: input.name,
          sourceQuery: input.sourceQuery,
          tags: input.tags,
          maxTokens: input.maxTokens,
          trigger: input.autoRefresh !== undefined ? { refresh_after_consolidation: input.autoRefresh } : undefined,
        });
        return {
          mentalModelId: result.mental_model_id,
          createdAt: result.created_at,
        };
      },
    }),

    queryMentalModel: tool<QueryMentalModelInput, QueryMentalModelOutput>({
      description:
        queryMentalModelDescription ??
        `Query an existing mental model to retrieve consolidated knowledge. Mental models provide synthesized insights from memories, making them faster and more efficient than searching through raw memories.`,
      inputSchema: queryMentalModelParams,
      execute: async (input) => {
        const result = await client.getMentalModel(input.bankId, input.mentalModelId);
        return {
          content: result.content ?? 'No content available yet.',
          name: result.name,
          updatedAt: result.updated_at,
        };
      },
    }),

    getDocument: tool<GetDocumentInput, GetDocumentOutput>({
      description:
        getDocumentDescription ??
        `Retrieve a stored document by its ID. Documents are used to store structured data like application state, user profiles, or any data that needs exact retrieval.`,
      inputSchema: getDocumentParams,
      execute: async (input) => {
        const result = await client.getDocument(input.bankId, input.documentId);
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

    createDirective: tool<CreateDirectiveInput, CreateDirectiveOutput>({
      description:
        `Create a directive - a hard rule that is injected into prompts during reflect operations. Directives are explicit instructions that guide agent behavior. Use tags to control when directives are applied (e.g., user-specific directives with 'user:username' tags).`,
      inputSchema: createDirectiveParams,
      execute: async (input) => {
        const result = await client.createDirective(input.bankId, {
          name: input.name,
          content: input.content,
          priority: input.priority,
          isActive: input.isActive,
          tags: input.tags,
        });
        return {
          id: result.id,
          name: result.name,
          content: result.content,
          tags: result.tags,
          createdAt: result.created_at,
        };
      },
    }),

    getDirective: tool<GetDirectiveInput, GetDirectiveOutput>({
      description:
        `Retrieve a directive by its ID. Returns the directive's content, tags, and active status.`,
      inputSchema: getDirectiveParams,
      execute: async (input) => {
        const result = await client.getDirective(input.bankId, input.directiveId);
        if (!result) {
          return null;
        }
        return {
          id: result.id,
          name: result.name,
          content: result.content,
          tags: result.tags,
          isActive: result.is_active,
        };
      },
    }),
  };
}

export type HindsightTools = ReturnType<typeof createHindsightTools>;
