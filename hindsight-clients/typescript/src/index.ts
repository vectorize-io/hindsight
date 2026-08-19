/**
 * Hindsight Client - Clean, TypeScript SDK for the Hindsight API.
 *
 * Example:
 * ```typescript
 * import { HindsightClient } from '@vectorize-io/hindsight-client';
 *
 * // Without authentication
 * const client = new HindsightClient({ baseUrl: 'http://localhost:8888' });
 *
 * // With API key authentication
 * const client = new HindsightClient({
 *   baseUrl: 'http://localhost:8888',
 *   apiKey: 'your-api-key'
 * });
 *
 * // Retain a memory
 * await client.retain('alice', 'Alice loves AI');
 *
 * // Recall memories
 * const results = await client.recall('alice', 'What does Alice like?');
 *
 * // Generate contextual answer
 * const answer = await client.reflect('alice', 'What are my interests?');
 * ```
 */

import { createClient, createConfig } from "../generated/client";
import type { Client } from "../generated/client";
import * as sdk from "../generated/sdk.gen";
import type {
  RetainRequest,
  RetainResponse,
  RecallRequest,
  RecallResponse,
  RecallResult,
  ReflectRequest,
  ReflectResponse,
  FileRetainResponse,
  ListMemoryUnitsResponse,
  BankProfileResponse,
  BankConfigResponse,
  CreateBankRequest,
  Budget,
  BankTemplateManifest,
  BankTemplateConfig,
  BankTemplateMentalModel,
  BankTemplateDirective,
  BankTemplateImportResponse,
  TagGroupLeaf,
  TagGroupAndInput,
  TagGroupOrInput,
  TagGroupNotInput,
  MinScores,
  AsyncOperationSubmitResponse,
  CreateKnowledgePageResponse,
  CreateMentalModelResponse,
  DirectiveListResponse,
  DirectiveResponse,
  DocumentResponse,
  KnowledgeNode,
  KnowledgePageBundleResponse,
  KnowledgePageResponse,
  KnowledgePageSearchResponse,
  KnowledgeTreeResponse,
  ListDocumentsResponse,
  MentalModelListResponse,
  MentalModelResponse,
  MentalModelTriggerInput,
  MentalModelDryRunRefreshResult,
  UpdateDocumentResponse,
  VersionResponse,
} from "../generated/types.gen";

// __CLIENT_VERSION__ is replaced by tsup's `define` with package.json's version
// at build time. The typeof guard keeps raw-source loads (jest, deno test:deno)
// from throwing ReferenceError; they get a sentinel that makes drift obvious.
declare const __CLIENT_VERSION__: string | undefined;
export const CLIENT_VERSION: string =
  typeof __CLIENT_VERSION__ !== "undefined" ? __CLIENT_VERSION__ : "0.0.0-dev";
export const DEFAULT_USER_AGENT = `hindsight-client-typescript/${CLIENT_VERSION}`;

export interface HindsightClientOptions {
  baseUrl: string;
  /**
   * Optional API key for authentication (sent as Bearer token in Authorization header)
   */
  apiKey?: string;
  /**
   * Override the default `User-Agent` header. Integrations should set this to
   * identify themselves (e.g. `"hindsight-ai-sdk/1.2.0"`). Browsers ignore
   * attempts to set `User-Agent`; this only takes effect in Node.js / Bun /
   * Deno runtimes. Defaults to `hindsight-client-typescript/<version>`.
   */
  userAgent?: string;
  /** Optional headers sent with every request. */
  headers?: Record<string, string>;
}

/**
 * Error thrown by the Hindsight client when an API request fails.
 * Includes the HTTP status code and error details from the API.
 */
export class HindsightError extends Error {
  public statusCode?: number;
  public details?: unknown;

  constructor(message: string, statusCode?: number, details?: unknown) {
    super(message);
    this.name = "HindsightError";
    this.statusCode = statusCode;
    this.details = details;
  }
}

export interface EntityInput {
  text: string;
  type?: string;
}

export interface MemoryItemInput {
  content: string;
  timestamp?: string | Date;
  context?: string;
  metadata?: Record<string, string>;
  document_id?: string;
  entities?: EntityInput[];
  /** Resolve the supplied `entities` against existing ones (default true); false stores them as written */
  resolve_entities?: boolean;
  tags?: string[];
  observation_scopes?: "per_tag" | "combined" | "all_combinations" | "shared" | string[][];
  strategy?: string;
  update_mode?: "replace" | "append";
}

/**
 * Warn when a caller-supplied operationId will be silently ignored.
 *
 * operationId only enables idempotent retries for asynchronous retain; on a
 * synchronous request it is dropped before reaching the API, so surface the
 * likely mistake instead of failing silently.
 */
function warnIfOperationIdDropped(
  async: boolean | undefined,
  operationId: string | null | undefined
): void {
  if (operationId != null && async !== true) {
    console.warn(
      "operationId is ignored for synchronous retain; pass async: true to enable idempotent retries."
    );
  }
}

export class HindsightClient {
  private client: Client;

  constructor(options: HindsightClientOptions) {
    const headers: Record<string, string> = {
      ...options.headers,
      "User-Agent": options.userAgent ?? DEFAULT_USER_AGENT,
    };
    if (options.apiKey) {
      headers.Authorization = `Bearer ${options.apiKey}`;
    }
    this.client = createClient(
      createConfig({
        baseUrl: options.baseUrl,
        headers,
      })
    );
  }

  /**
   * Get API version and feature flags for the connected Hindsight deployment.
   */
  async getVersion(options?: { signal?: AbortSignal }): Promise<VersionResponse> {
    const response = await sdk.getVersion({
      client: this.client,
      signal: options?.signal,
    });

    return this.validateResponse(response, "getVersion");
  }

  /**
   * Validates the API response and throws an error if the request failed.
   */
  private validateResponse<T>(
    response: { data?: T; error?: unknown; response?: Response },
    operation: string
  ): T {
    if (!response.data) {
      // The generated client returns { error, response, request }
      // Status code is in response.status, not in the error object
      const error = response.error as any;
      const httpResponse = (response as any).response as Response | undefined;

      // Extract status code from the HTTP response object
      const statusCode = httpResponse?.status;
      const details = error?.detail || error?.message || error;

      throw new HindsightError(
        `${operation} failed: ${JSON.stringify(details)}`,
        statusCode,
        details
      );
    }
    return response.data;
  }

  /**
   * Retain a single memory for a bank.
   */
  async retain(
    bankId: string,
    content: string,
    options?: {
      timestamp?: Date | string;
      context?: string;
      metadata?: Record<string, string>;
      documentId?: string;
      async?: boolean;
      /** Optional caller-supplied UUID for idempotent async retries */
      operationId?: string;
      entities?: EntityInput[];
      /** Resolve the supplied `entities` against existing ones (default true); false stores them as written */
      resolveEntities?: boolean;
      /** Optional list of tags for this memory */
      tags?: string[];
      /** How to handle existing documents: 'replace' (default) or 'append' */
      updateMode?: "replace" | "append";
      /** Observation scoping strategy: 'per_tag', 'combined', 'all_combinations', 'shared', or explicit scope groups */
      observationScopes?: "per_tag" | "combined" | "all_combinations" | "shared" | string[][];
      /** Extraction strategy override */
      strategy?: string;
      signal?: AbortSignal;
    }
  ): Promise<RetainResponse> {
    warnIfOperationIdDropped(options?.async, options?.operationId);
    return this.retainBatch(
      bankId,
      [
        {
          content,
          timestamp: options?.timestamp,
          context: options?.context,
          metadata: options?.metadata,
          document_id: options?.documentId,
          entities: options?.entities,
          resolve_entities: options?.resolveEntities,
          tags: options?.tags,
          update_mode: options?.updateMode,
          observation_scopes: options?.observationScopes,
          strategy: options?.strategy,
        },
      ],
      {
        async: options?.async,
        signal: options?.signal,
        ...(options?.async === true && options.operationId != null
          ? { operationId: options.operationId }
          : {}),
      }
    );
  }

  /**
   * Retain multiple memories in batch.
   */
  async retainBatch(
    bankId: string,
    items: MemoryItemInput[],
    options?: {
      documentId?: string;
      documentTags?: string[];
      async?: boolean;
      /** Optional caller-supplied UUID for idempotent async retries */
      operationId?: string;
      signal?: AbortSignal;
    }
  ): Promise<RetainResponse> {
    warnIfOperationIdDropped(options?.async, options?.operationId);
    const processedItems = items.map((item) => ({
      content: item.content,
      context: item.context,
      metadata: item.metadata,
      document_id: item.document_id,
      entities: item.entities,
      resolve_entities: item.resolve_entities,
      tags: item.tags,
      observation_scopes: item.observation_scopes,
      strategy: item.strategy,
      update_mode: item.update_mode,
      timestamp: item.timestamp instanceof Date ? item.timestamp.toISOString() : item.timestamp,
    }));

    // If documentId is provided at the batch level, add it to all items that don't have one
    const itemsWithDocId = processedItems.map((item) => ({
      ...item,
      document_id: item.document_id || options?.documentId,
    }));

    const response = await sdk.retainMemories({
      client: this.client,
      path: { bank_id: bankId },
      body: {
        items: itemsWithDocId,
        document_tags: options?.documentTags,
        async: options?.async,
        ...(options?.async === true && options.operationId != null
          ? { operation_id: options.operationId }
          : {}),
      },
      signal: options?.signal,
    });

    return this.validateResponse(response, "retainBatch");
  }

  /**
   * Upload files and retain their contents as memories.
   *
   * Files are automatically converted to text (PDF, DOCX, images via OCR, audio via
   * transcription, and more) and ingested as memories. Processing is always asynchronous —
   * use the returned operation IDs to track progress via the operations endpoint.
   *
   * @param bankId - The memory bank ID
   * @param files - Array of File or Blob objects to upload
   * @param options - Optional settings: context, documentTags, filesMetadata
   */
  async retainFiles(
    bankId: string,
    files: Array<File | Blob>,
    options?: {
      context?: string;
      filesMetadata?: Array<{
        context?: string;
        document_id?: string;
        tags?: string[];
        metadata?: Record<string, string>;
      }>;
      signal?: AbortSignal;
    }
  ): Promise<FileRetainResponse> {
    const meta =
      options?.filesMetadata ??
      files.map(() => (options?.context ? { context: options.context } : {}));

    const requestBody = JSON.stringify({
      files_metadata: meta,
    });

    const response = await sdk.fileRetain({
      client: this.client,
      path: { bank_id: bankId },
      body: { files, request: requestBody },
      signal: options?.signal,
    });

    return this.validateResponse(response, "retainFiles");
  }

  /**
   * Recall memories with a natural language query.
   */
  async recall(
    bankId: string,
    query: string,
    options?: {
      types?: string[];
      /** When recalling raw facts ('world'/'experience') together with 'observation', drop any raw fact a returned observation was consolidated from, so the observation supersedes it (no duplicate content). Disabled by default; no effect unless 'observation' and at least one raw type are both in types. */
      preferObservations?: boolean;
      maxTokens?: number;
      budget?: Budget;
      trace?: boolean;
      queryTimestamp?: string;
      includeEntities?: boolean;
      maxEntityTokens?: number;
      includeChunks?: boolean;
      maxChunkTokens?: number;
      /** Include source facts for observation-type results */
      includeSourceFacts?: boolean;
      /** Maximum tokens for source facts (default: 4096) */
      maxSourceFactsTokens?: number;
      /** Optional list of tags to filter memories by */
      tags?: string[];
      /** How to match tags: 'any' (OR, includes untagged), 'all' (AND, includes untagged), 'any_strict' (OR, excludes untagged), 'all_strict' (AND, excludes untagged), 'exact' (set equality, excludes untagged). Default: 'any' */
      tagsMatch?: "any" | "all" | "any_strict" | "all_strict" | "exact";
      /** Compound tag filter using boolean groups. Groups are AND-ed. Each group is a leaf {tags, match} or compound {and: [...]}, {or: [...]}, {not: ...}. Mutually exclusive with tags/tagsMatch. */
      tagGroups?: Array<TagGroupLeaf | TagGroupAndInput | TagGroupOrInput | TagGroupNotInput>;
      /** Optional per-stage score floors, e.g. {semantic: 0.2, final: 0.5}. 'semantic' and 'keyword' are retrieval-level cutoffs; 'reranker' and 'final' are applied to the scored results after reranking. Any omitted stage imposes no floor. */
      minScores?: MinScores;
      signal?: AbortSignal;
    }
  ): Promise<RecallResponse> {
    const response = await sdk.recallMemories({
      client: this.client,
      path: { bank_id: bankId },
      body: {
        query,
        types: options?.types,
        prefer_observations: options?.preferObservations,
        max_tokens: options?.maxTokens,
        budget: options?.budget || "mid",
        trace: options?.trace,
        query_timestamp: options?.queryTimestamp,
        include: {
          entities:
            options?.includeEntities === false
              ? null
              : options?.includeEntities
                ? { max_tokens: options?.maxEntityTokens ?? 500 }
                : undefined,
          chunks: options?.includeChunks
            ? { max_tokens: options?.maxChunkTokens ?? 8192 }
            : undefined,
          source_facts: options?.includeSourceFacts
            ? { max_tokens: options?.maxSourceFactsTokens ?? 4096 }
            : undefined,
        },
        tags: options?.tags,
        tags_match: options?.tagsMatch,
        tag_groups: options?.tagGroups,
        min_scores: options?.minScores,
      },
      signal: options?.signal,
    });

    return this.validateResponse(response, "recall");
  }

  /**
   * Reflect and generate a contextual answer using the bank's identity and memories.
   */
  async reflect(
    bankId: string,
    query: string,
    options?: {
      context?: string;
      budget?: Budget;
      /** Optional list of tags to filter memories by */
      tags?: string[];
      /** How to match tags: 'any' (OR, includes untagged), 'all' (AND, includes untagged), 'any_strict' (OR, excludes untagged), 'all_strict' (AND, excludes untagged), 'exact' (set equality, excludes untagged). Default: 'any' */
      tagsMatch?: "any" | "all" | "any_strict" | "all_strict" | "exact";
      /** Compound tag filter using boolean groups. Groups are AND-ed. Mutually exclusive with tags/tagsMatch. */
      tagGroups?: Array<TagGroupLeaf | TagGroupAndInput | TagGroupOrInput | TagGroupNotInput>;
      /** Optional JSON Schema for structured output. When provided, the response includes a 'structured_output' field. */
      responseSchema?: Record<string, unknown>;
      /** Filter which fact types are retrieved: 'world', 'experience', 'observation'. None means all. */
      factTypes?: Array<"world" | "experience" | "observation">;
      /** If true, exclude all mental models from reflection. */
      excludeMentalModels?: boolean;
      /** Exclude specific mental models by ID from reflection. */
      excludeMentalModelIds?: string[];
      /** If true, the response includes a 'based_on' field listing the memories, mental models, and directives used. */
      includeFacts?: boolean;
      /** If true, the response includes a 'trace' field with the tool calls and LLM calls made during reflection (trace.tool_calls / trace.llm_calls). */
      includeToolCalls?: boolean;
      /** When includeToolCalls is true, set to false for an inputs-only trace (smaller payload). Ignored otherwise. Default: true. */
      includeToolCallOutput?: boolean;
      signal?: AbortSignal;
    }
  ): Promise<ReflectResponse> {
    const include =
      options?.includeFacts || options?.includeToolCalls
        ? {
            facts: options?.includeFacts ? {} : undefined,
            tool_calls: options?.includeToolCalls
              ? { output: options?.includeToolCallOutput ?? true }
              : undefined,
          }
        : undefined;
    const response = await sdk.reflect({
      client: this.client,
      path: { bank_id: bankId },
      body: {
        query,
        context: options?.context,
        budget: options?.budget || "low",
        tags: options?.tags,
        tags_match: options?.tagsMatch,
        tag_groups: options?.tagGroups,
        response_schema: options?.responseSchema,
        fact_types: options?.factTypes,
        exclude_mental_models: options?.excludeMentalModels,
        exclude_mental_model_ids: options?.excludeMentalModelIds,
        include,
      },
      signal: options?.signal,
    });

    return this.validateResponse(response, "reflect");
  }

  /**
   * List memories with pagination.
   */
  async listMemories(
    bankId: string,
    options?: {
      limit?: number;
      offset?: number;
      type?: string;
      q?: string;
      consolidationState?: "failed" | "pending" | "done";
      state?: "valid" | "invalidated";
      documentId?: string;
      entityId?: string;
      signal?: AbortSignal;
    }
  ): Promise<ListMemoryUnitsResponse> {
    const response = await sdk.listMemories({
      client: this.client,
      path: { bank_id: bankId },
      query: {
        limit: options?.limit,
        offset: options?.offset,
        type: options?.type,
        q: options?.q,
        consolidation_state: options?.consolidationState,
        state: options?.state,
        document_id: options?.documentId,
        entity_id: options?.entityId,
      },
      signal: options?.signal,
    });

    return this.validateResponse(response, "listMemories");
  }

  /**
   * Create or update a bank with disposition, missions, and operational configuration.
   */
  async createBank(
    bankId: string,
    options: {
      /** @deprecated Display label only. */
      name?: string;
      /** @deprecated Use reflectMission instead. */
      mission?: string;
      /** Mission/context for Reflect operations. */
      reflectMission?: string;
      /** @deprecated Alias for mission. */
      background?: string;
      /** @deprecated Use dispositionSkepticism, dispositionLiteralism, dispositionEmpathy instead. */
      disposition?: { skepticism: number; literalism: number; empathy: number };
      /** @deprecated Use updateBankConfig({ dispositionSkepticism }) instead. */
      dispositionSkepticism?: number;
      /** @deprecated Use updateBankConfig({ dispositionLiteralism }) instead. */
      dispositionLiteralism?: number;
      /** @deprecated Use updateBankConfig({ dispositionEmpathy }) instead. */
      dispositionEmpathy?: number;
      /** Steers what gets extracted during retain(). Injected alongside built-in rules. */
      retainMission?: string;
      /** Fact extraction mode: 'concise' (default), 'verbose', 'custom', 'verbatim', or 'chunks'. */
      retainExtractionMode?: string;
      /** Custom extraction prompt (only active when retainExtractionMode is 'custom'). */
      retainCustomInstructions?: string;
      /** Target maximum characters for each content chunk during retain. */
      retainChunkSize?: number;
      /** Maximum characters for a single JSONL line or conversation turn to keep whole during retain. */
      retainStructuredChunkSize?: number;
      /** Toggle automatic observation consolidation after retain(). */
      enableObservations?: boolean;
      /** Controls what gets synthesised into observations. Replaces built-in rules. */
      observationsMission?: string;
      /** Run the temporal retrieval arm during recall, and the date-aware query analysis feeding it. */
      enableTemporalRetrieval?: boolean;
      /** Run the entity/link graph traversal arm during recall. */
      enableGraphRetrieval?: boolean;
      /** Rerank fused candidates with the cross-encoder. False returns the RRF order. */
      enableReranking?: boolean;
      signal?: AbortSignal;
    } = {}
  ): Promise<BankProfileResponse> {
    const response = await sdk.createOrUpdateBank({
      client: this.client,
      path: { bank_id: bankId },
      body: {
        name: options.name,
        mission: options.mission,
        reflect_mission: options.reflectMission,
        background: options.background,
        disposition: options.disposition,
        disposition_skepticism: options.dispositionSkepticism,
        disposition_literalism: options.dispositionLiteralism,
        disposition_empathy: options.dispositionEmpathy,
        retain_mission: options.retainMission,
        retain_extraction_mode: options.retainExtractionMode,
        retain_custom_instructions: options.retainCustomInstructions,
        retain_chunk_size: options.retainChunkSize,
        retain_structured_chunk_size: options.retainStructuredChunkSize,
        enable_observations: options.enableObservations,
        observations_mission: options.observationsMission,
        enable_temporal_retrieval: options.enableTemporalRetrieval,
        enable_graph_retrieval: options.enableGraphRetrieval,
        enable_reranking: options.enableReranking,
      },
      signal: options.signal,
    });

    return this.validateResponse(response, "createBank");
  }

  /**
   * Set or update the reflect mission for a memory bank.
   * @deprecated Use createBank({ reflectMission: '...' }) instead.
   */
  async setMission(
    bankId: string,
    mission: string,
    options?: { signal?: AbortSignal }
  ): Promise<BankProfileResponse> {
    return this.createBank(bankId, { reflectMission: mission, signal: options?.signal });
  }

  /**
   * Get a bank's profile.
   */
  async getBankProfile(
    bankId: string,
    options?: { signal?: AbortSignal }
  ): Promise<BankProfileResponse> {
    const response = await sdk.getBankProfile({
      client: this.client,
      path: { bank_id: bankId },
      signal: options?.signal,
    });

    return this.validateResponse(response, "getBankProfile");
  }

  /**
   * Get the resolved configuration for a bank, including any bank-level overrides.
   *
   * Can be disabled on the server by setting `HINDSIGHT_API_ENABLE_BANK_CONFIG_API=false`.
   */
  async getBankConfig(
    bankId: string,
    options?: { signal?: AbortSignal }
  ): Promise<BankConfigResponse> {
    const response = await sdk.getBankConfig({
      client: this.client,
      path: { bank_id: bankId },
      signal: options?.signal,
    });

    return this.validateResponse(response, "getBankConfig");
  }

  /**
   * Update configuration overrides for a bank.
   *
   * Can be disabled on the server by setting `HINDSIGHT_API_ENABLE_BANK_CONFIG_API=false`.
   *
   * @param bankId - The memory bank ID
   * @param options - Fields to override
   */
  async updateBankConfig(
    bankId: string,
    options: {
      reflectMission?: string;
      retainMission?: string;
      retainExtractionMode?: string;
      retainCustomInstructions?: string;
      retainChunkSize?: number;
      retainStructuredChunkSize?: number;
      enableObservations?: boolean;
      observationsMission?: string;
      /** Run the temporal retrieval arm during recall, and the date-aware query analysis feeding it. */
      enableTemporalRetrieval?: boolean;
      /** Run the entity/link graph traversal arm during recall. */
      enableGraphRetrieval?: boolean;
      /** Rerank fused candidates with the cross-encoder. False returns the RRF order. */
      enableReranking?: boolean;
      /** How skeptical vs trusting (1=trusting, 5=skeptical). */
      dispositionSkepticism?: number;
      /** How literally to interpret information (1=flexible, 5=literal). */
      dispositionLiteralism?: number;
      /** How much to consider emotional context (1=detached, 5=empathetic). */
      dispositionEmpathy?: number;
      signal?: AbortSignal;
    }
  ): Promise<BankConfigResponse> {
    const updates: Record<string, unknown> = {};
    if (options.reflectMission !== undefined) updates.reflect_mission = options.reflectMission;
    if (options.retainMission !== undefined) updates.retain_mission = options.retainMission;
    if (options.retainExtractionMode !== undefined)
      updates.retain_extraction_mode = options.retainExtractionMode;
    if (options.retainCustomInstructions !== undefined)
      updates.retain_custom_instructions = options.retainCustomInstructions;
    if (options.retainChunkSize !== undefined) updates.retain_chunk_size = options.retainChunkSize;
    if (options.retainStructuredChunkSize !== undefined)
      updates.retain_structured_chunk_size = options.retainStructuredChunkSize;
    if (options.enableObservations !== undefined)
      updates.enable_observations = options.enableObservations;
    if (options.observationsMission !== undefined)
      updates.observations_mission = options.observationsMission;
    if (options.enableTemporalRetrieval !== undefined)
      updates.enable_temporal_retrieval = options.enableTemporalRetrieval;
    if (options.enableGraphRetrieval !== undefined)
      updates.enable_graph_retrieval = options.enableGraphRetrieval;
    if (options.enableReranking !== undefined) updates.enable_reranking = options.enableReranking;
    if (options.dispositionSkepticism !== undefined)
      updates.disposition_skepticism = options.dispositionSkepticism;
    if (options.dispositionLiteralism !== undefined)
      updates.disposition_literalism = options.dispositionLiteralism;
    if (options.dispositionEmpathy !== undefined)
      updates.disposition_empathy = options.dispositionEmpathy;

    const response = await sdk.updateBankConfig({
      client: this.client,
      path: { bank_id: bankId },
      body: { updates },
      signal: options.signal,
    });

    return this.validateResponse(response, "updateBankConfig");
  }

  /**
   * Reset all bank-level configuration overrides, reverting to server defaults.
   *
   * Can be disabled on the server by setting `HINDSIGHT_API_ENABLE_BANK_CONFIG_API=false`.
   */
  async resetBankConfig(
    bankId: string,
    options?: { signal?: AbortSignal }
  ): Promise<BankConfigResponse> {
    const response = await sdk.resetBankConfig({
      client: this.client,
      path: { bank_id: bankId },
      signal: options?.signal,
    });

    return this.validateResponse(response, "resetBankConfig");
  }

  /**
   * Delete a bank.
   */
  async deleteBank(bankId: string, options?: { signal?: AbortSignal }): Promise<void> {
    const response = await sdk.deleteBank({
      client: this.client,
      path: { bank_id: bankId },
      signal: options?.signal,
    });
    if (response.error) {
      throw new Error(`deleteBank failed: ${JSON.stringify(response.error)}`);
    }
  }

  // Directive methods

  /**
   * Create a directive (hard rule for reflect).
   */
  async createDirective(
    bankId: string,
    name: string,
    content: string,
    options?: {
      priority?: number;
      isActive?: boolean;
      tags?: string[];
      signal?: AbortSignal;
    }
  ): Promise<DirectiveResponse> {
    const response = await sdk.createDirective({
      client: this.client,
      path: { bank_id: bankId },
      body: {
        name,
        content,
        priority: options?.priority ?? 0,
        is_active: options?.isActive ?? true,
        tags: options?.tags,
      },
      signal: options?.signal,
    });

    return this.validateResponse(response, "createDirective");
  }

  /**
   * List all directives in a bank.
   */
  async listDirectives(
    bankId: string,
    options?: { tags?: string[]; limit?: number; offset?: number; signal?: AbortSignal }
  ): Promise<DirectiveListResponse> {
    const response = await sdk.listDirectives({
      client: this.client,
      path: { bank_id: bankId },
      query: {
        tags: options?.tags,
        ...(options?.limit !== undefined ? { limit: options.limit } : {}),
        ...(options?.offset !== undefined ? { offset: options.offset } : {}),
      },
      signal: options?.signal,
    });

    return this.validateResponse(response, "listDirectives");
  }

  /**
   * Get a specific directive.
   */
  async getDirective(
    bankId: string,
    directiveId: string,
    options?: { signal?: AbortSignal }
  ): Promise<DirectiveResponse> {
    const response = await sdk.getDirective({
      client: this.client,
      path: { bank_id: bankId, directive_id: directiveId },
      signal: options?.signal,
    });

    return this.validateResponse(response, "getDirective");
  }

  /**
   * Update a directive.
   */
  async updateDirective(
    bankId: string,
    directiveId: string,
    options: {
      name?: string;
      content?: string;
      priority?: number;
      isActive?: boolean;
      tags?: string[];
      signal?: AbortSignal;
    }
  ): Promise<DirectiveResponse> {
    const response = await sdk.updateDirective({
      client: this.client,
      path: { bank_id: bankId, directive_id: directiveId },
      body: {
        name: options.name,
        content: options.content,
        priority: options.priority,
        is_active: options.isActive,
        tags: options.tags,
      },
      signal: options.signal,
    });

    return this.validateResponse(response, "updateDirective");
  }

  /**
   * Delete a directive.
   */
  async deleteDirective(
    bankId: string,
    directiveId: string,
    options?: { signal?: AbortSignal }
  ): Promise<void> {
    const response = await sdk.deleteDirective({
      client: this.client,
      path: { bank_id: bankId, directive_id: directiveId },
      signal: options?.signal,
    });
    if (response.error) {
      throw new Error(`deleteDirective failed: ${JSON.stringify(response.error)}`);
    }
  }

  // Mental Model methods

  /**
   * Create a mental model (runs reflect in background).
   */
  async createMentalModel(
    bankId: string,
    name: string,
    sourceQuery: string,
    options?: {
      id?: string;
      tags?: string[];
      maxTokens?: number;
      trigger?: {
        refreshAfterConsolidation?: boolean;
        /** How this model's tags filter source memories on refresh. If omitted, a tagged model defaults to 'all_strict' (a memory must carry every one of the model's tags), which silently drops memories that only carry a subset. Set 'any' to match memories carrying any of the tags — the same default recall/reflect use. */
        tagsMatch?: "any" | "all" | "any_strict" | "all_strict" | "exact";
        /** Compound tag filter using boolean groups; overrides the model's flat tags/tagsMatch during refresh. */
        tagGroups?: Array<TagGroupLeaf | TagGroupAndInput | TagGroupOrInput | TagGroupNotInput>;
      };
      signal?: AbortSignal;
    }
  ): Promise<CreateMentalModelResponse> {
    const response = await sdk.createMentalModel({
      client: this.client,
      path: { bank_id: bankId },
      body: {
        id: options?.id,
        name,
        source_query: sourceQuery,
        tags: options?.tags,
        max_tokens: options?.maxTokens,
        trigger: options?.trigger
          ? {
              refresh_after_consolidation: options.trigger.refreshAfterConsolidation,
              tags_match: options.trigger.tagsMatch,
              tag_groups: options.trigger.tagGroups,
            }
          : undefined,
      },
      signal: options?.signal,
    });

    return this.validateResponse(response, "createMentalModel");
  }

  /**
   * List all mental models in a bank.
   */
  async listMentalModels(
    bankId: string,
    options?: {
      tags?: string[];
      tagsMatch?: "any" | "all" | "exact";
      /** Exclude large provenance chains with "metadata" or "content" when they are not needed. */
      detail?: "metadata" | "content" | "full";
      limit?: number;
      offset?: number;
      signal?: AbortSignal;
    }
  ): Promise<MentalModelListResponse> {
    const response = await sdk.listMentalModels({
      client: this.client,
      path: { bank_id: bankId },
      query: {
        tags: options?.tags,
        ...(options?.tagsMatch !== undefined ? { tags_match: options.tagsMatch } : {}),
        ...(options?.detail !== undefined ? { detail: options.detail } : {}),
        ...(options?.limit !== undefined ? { limit: options.limit } : {}),
        ...(options?.offset !== undefined ? { offset: options.offset } : {}),
      },
      signal: options?.signal,
    });

    return this.validateResponse(response, "listMentalModels");
  }

  /**
   * Get a specific mental model.
   */
  async getMentalModel(
    bankId: string,
    mentalModelId: string,
    options?: {
      /** Exclude large provenance chains with "metadata" or "content" when they are not needed. */
      detail?: "metadata" | "content" | "full";
      signal?: AbortSignal;
    }
  ): Promise<MentalModelResponse> {
    const response = await sdk.getMentalModel({
      client: this.client,
      path: { bank_id: bankId, mental_model_id: mentalModelId },
      ...(options?.detail ? { query: { detail: options.detail } } : {}),
      signal: options?.signal,
    });

    return this.validateResponse(response, "getMentalModel");
  }

  /**
   * Refresh a mental model to update with current knowledge.
   */
  async refreshMentalModel(
    bankId: string,
    mentalModelId: string,
    options?: { signal?: AbortSignal }
  ): Promise<AsyncOperationSubmitResponse> {
    const response = await sdk.refreshMentalModel({
      client: this.client,
      path: { bank_id: bankId, mental_model_id: mentalModelId },
      signal: options?.signal,
    });

    return this.validateResponse(response, "refreshMentalModel");
  }

  /**
   * Preview what a refresh would do to a mental model without changing it.
   *
   * The production refresh pipeline with two writes skipped — the content and the
   * watermark — so what it reports is what the next refresh will do. Reports the
   * mode it ran in and why, the scope and window it read, the evidence it would
   * ground on, and a diff from the stored content to the content it would write.
   *
   * Not configurable, and costs the same LLM tokens as a real refresh.
   */
  async dryRunRefreshMentalModel(
    bankId: string,
    mentalModelId: string,
    options?: { signal?: AbortSignal }
  ): Promise<MentalModelDryRunRefreshResult> {
    const response = await sdk.dryRunRefreshMentalModel({
      client: this.client,
      path: { bank_id: bankId, mental_model_id: mentalModelId },
      signal: options?.signal,
    });

    return this.validateResponse(response, "dryRunRefreshMentalModel");
  }

  /**
   * Clear a mental model's content so the next refresh performs a full re-synthesis.
   */
  async clearMentalModel(
    bankId: string,
    mentalModelId: string,
    options?: { signal?: AbortSignal }
  ): Promise<MentalModelResponse> {
    const response = await sdk.clearMentalModel({
      client: this.client,
      path: { bank_id: bankId, mental_model_id: mentalModelId },
      signal: options?.signal,
    });

    return this.validateResponse(response, "clearMentalModel");
  }

  /**
   * Update a mental model's metadata.
   */
  async updateMentalModel(
    bankId: string,
    mentalModelId: string,
    options: {
      name?: string;
      sourceQuery?: string;
      tags?: string[];
      maxTokens?: number;
      trigger?: { refreshAfterConsolidation?: boolean };
      signal?: AbortSignal;
    }
  ): Promise<MentalModelResponse> {
    const response = await sdk.updateMentalModel({
      client: this.client,
      path: { bank_id: bankId, mental_model_id: mentalModelId },
      body: {
        name: options.name,
        source_query: options.sourceQuery,
        tags: options.tags,
        max_tokens: options.maxTokens,
        trigger: options.trigger
          ? { refresh_after_consolidation: options.trigger.refreshAfterConsolidation }
          : undefined,
      },
      signal: options.signal,
    });

    return this.validateResponse(response, "updateMentalModel");
  }

  /**
   * Delete a mental model.
   */
  async deleteMentalModel(
    bankId: string,
    mentalModelId: string,
    options?: { signal?: AbortSignal }
  ): Promise<void> {
    const response = await sdk.deleteMentalModel({
      client: this.client,
      path: { bank_id: bankId, mental_model_id: mentalModelId },
      signal: options?.signal,
    });
    if (response.error) {
      throw new Error(`deleteMentalModel failed: ${JSON.stringify(response.error)}`);
    }
  }

  /**
   * Get the change history of a mental model.
   */
  async getMentalModelHistory(
    bankId: string,
    mentalModelId: string,
    options?: { signal?: AbortSignal }
  ): Promise<unknown> {
    const response = await sdk.getMentalModelHistory({
      client: this.client,
      path: { bank_id: bankId, mental_model_id: mentalModelId },
      signal: options?.signal,
    });

    return this.validateResponse(response, "getMentalModelHistory");
  }

  /**
   * Get the knowledge base as a nested folder/page tree.
   *
   * Page bodies are not included — fetch one with `getKnowledgePage`.
   */
  async getKnowledgeBaseTree(
    bankId: string,
    options?: { signal?: AbortSignal }
  ): Promise<KnowledgeTreeResponse> {
    const response = await sdk.getKnowledgeBaseTree({
      client: this.client,
      path: { bank_id: bankId },
      signal: options?.signal,
    });

    return this.validateResponse(response, "getKnowledgeBaseTree");
  }

  /**
   * Create a knowledge-base folder.
   */
  async createKnowledgeFolder(
    bankId: string,
    name: string,
    options?: { parentId?: string | null; signal?: AbortSignal }
  ): Promise<KnowledgeNode> {
    const response = await sdk.createKnowledgeFolder({
      client: this.client,
      path: { bank_id: bankId },
      body: { name, parent_id: options?.parentId },
      signal: options?.signal,
    });

    return this.validateResponse(response, "createKnowledgeFolder");
  }

  /**
   * Create a knowledge-base page. Content is generated asynchronously — poll the
   * returned `operation_id` to know when the first build has finished.
   *
   * Omit `trigger` to use the page defaults (observation-only, delta mode,
   * refresh after consolidation); a supplied trigger replaces those defaults
   * rather than merging with them.
   */
  async createKnowledgePage(
    bankId: string,
    name: string,
    sourceQuery: string,
    options?: {
      parentId?: string | null;
      /** Scopes which memories the page is built from. A `type:<x>` tag also sets the page's rendered type. */
      tags?: string[];
      maxTokens?: number;
      trigger?: {
        mode?: "full" | "delta";
        refreshAfterConsolidation?: boolean;
        refreshCron?: string | null;
        factTypes?: Array<"world" | "experience" | "observation">;
        excludeMentalModels?: boolean;
        excludeMentalModelIds?: string[];
        tagsMatch?: "any" | "all" | "any_strict" | "all_strict" | "exact";
        tagGroups?: Array<TagGroupLeaf | TagGroupAndInput | TagGroupOrInput | TagGroupNotInput>;
        includeChunks?: boolean;
        recallMaxTokens?: number;
        recallChunksMaxTokens?: number;
      };
      signal?: AbortSignal;
    }
  ): Promise<CreateKnowledgePageResponse> {
    const response = await sdk.createKnowledgePage({
      client: this.client,
      path: { bank_id: bankId },
      body: {
        name,
        source_query: sourceQuery,
        parent_id: options?.parentId,
        tags: options?.tags,
        max_tokens: options?.maxTokens,
        trigger: options?.trigger
          ? {
              mode: options.trigger.mode,
              refresh_after_consolidation: options.trigger.refreshAfterConsolidation,
              refresh_cron: options.trigger.refreshCron,
              fact_types: options.trigger.factTypes,
              exclude_mental_models: options.trigger.excludeMentalModels,
              exclude_mental_model_ids: options.trigger.excludeMentalModelIds,
              tags_match: options.trigger.tagsMatch,
              tag_groups: options.trigger.tagGroups,
              include_chunks: options.trigger.includeChunks,
              recall_max_tokens: options.trigger.recallMaxTokens,
              recall_chunks_max_tokens: options.trigger.recallChunksMaxTokens,
            }
          : undefined,
      },
      signal: options?.signal,
    });

    return this.validateResponse(response, "createKnowledgePage");
  }

  /**
   * Get a knowledge page rendered as a markdown document (frontmatter + body).
   */
  async getKnowledgePage(
    bankId: string,
    pageId: string,
    options?: { signal?: AbortSignal }
  ): Promise<KnowledgePageResponse> {
    const response = await sdk.getKnowledgePage({
      client: this.client,
      path: { bank_id: bankId, page_id: pageId },
      signal: options?.signal,
    });

    return this.validateResponse(response, "getKnowledgePage");
  }

  /**
   * Hybrid search (full-text + vector) over the bank's knowledge pages.
   */
  async searchKnowledgeBase(
    bankId: string,
    query: string,
    options?: { limit?: number; signal?: AbortSignal }
  ): Promise<KnowledgePageSearchResponse> {
    const response = await sdk.searchKnowledgeBase({
      client: this.client,
      path: { bank_id: bankId },
      query: {
        q: query,
        ...(options?.limit !== undefined ? { limit: options.limit } : {}),
      },
      signal: options?.signal,
    });

    return this.validateResponse(response, "searchKnowledgeBase");
  }

  /**
   * Rename/move a knowledge node and/or update a page's options.
   *
   * Only the fields present in `options` are sent, so passing `parentId: null`
   * explicitly moves the node to the root.
   */
  async updateKnowledgeNode(
    bankId: string,
    nodeId: string,
    options: {
      name?: string;
      parentId?: string | null;
      /** Pages only — changing it rebuilds the page against the new question. */
      sourceQuery?: string;
      /** Pages only — replaces the page's tags (pass [] to clear). */
      tags?: string[];
      maxTokens?: number;
      /** Pages only — refresh settings to change. Applied as a patch: the fields you send are
       *  updated and the rest keep the page's current values. */
      trigger?: MentalModelTriggerInput;
      signal?: AbortSignal;
    }
  ): Promise<KnowledgeNode> {
    const response = await sdk.updateKnowledgeNode({
      client: this.client,
      path: { bank_id: bankId, node_id: nodeId },
      body: {
        ...(options.name !== undefined ? { name: options.name } : {}),
        ...("parentId" in options ? { parent_id: options.parentId } : {}),
        ...(options.sourceQuery !== undefined ? { source_query: options.sourceQuery } : {}),
        ...(options.tags !== undefined ? { tags: options.tags } : {}),
        ...(options.maxTokens !== undefined ? { max_tokens: options.maxTokens } : {}),
        ...(options.trigger !== undefined ? { trigger: options.trigger } : {}),
      },
      signal: options.signal,
    });

    return this.validateResponse(response, "updateKnowledgeNode");
  }

  /**
   * Delete a knowledge folder or page and its whole subtree.
   */
  async deleteKnowledgeNode(
    bankId: string,
    nodeId: string,
    options?: { signal?: AbortSignal }
  ): Promise<unknown> {
    const response = await sdk.deleteKnowledgeNode({
      client: this.client,
      path: { bank_id: bankId, node_id: nodeId },
      signal: options?.signal,
    });

    return this.validateResponse(response, "deleteKnowledgeNode");
  }

  /**
   * Export the knowledge base as a portable markdown bundle.
   */
  async exportKnowledgeBase(
    bankId: string,
    options?: { signal?: AbortSignal }
  ): Promise<KnowledgePageBundleResponse> {
    const response = await sdk.exportKnowledgeBase({
      client: this.client,
      path: { bank_id: bankId },
      signal: options?.signal,
    });

    return this.validateResponse(response, "exportKnowledgeBase");
  }

  /**
   * Get a document by ID. Returns null if not found.
   */
  async getDocument(
    bankId: string,
    documentId: string,
    options?: { signal?: AbortSignal }
  ): Promise<DocumentResponse | null> {
    const response = await sdk.getDocument({
      client: this.client,
      path: { bank_id: bankId, document_id: documentId },
      signal: options?.signal,
    });

    if ((response as any).response?.status === 404) {
      return null;
    }

    return this.validateResponse(response, "getDocument");
  }

  /**
   * List documents in a bank.
   */
  async listDocuments(
    bankId: string,
    options?: { limit?: number; offset?: number; signal?: AbortSignal }
  ): Promise<ListDocumentsResponse> {
    const response = await sdk.listDocuments({
      client: this.client,
      path: { bank_id: bankId },
      query: { limit: options?.limit, offset: options?.offset },
      signal: options?.signal,
    });

    return this.validateResponse(response, "listDocuments");
  }

  /**
   * Delete a document.
   */
  async deleteDocument(
    bankId: string,
    documentId: string,
    options?: { signal?: AbortSignal }
  ): Promise<void> {
    const response = await sdk.deleteDocument({
      client: this.client,
      path: { bank_id: bankId, document_id: documentId },
      signal: options?.signal,
    });
    if (response.error) {
      throw new Error(`deleteDocument failed: ${JSON.stringify(response.error)}`);
    }
  }

  /**
   * Update a document's mutable fields.
   */
  async updateDocument(
    bankId: string,
    documentId: string,
    options: { tags?: string[]; signal?: AbortSignal }
  ): Promise<UpdateDocumentResponse> {
    const response = await sdk.updateDocument({
      client: this.client,
      path: { bank_id: bankId, document_id: documentId },
      body: { tags: options.tags },
      signal: options.signal,
    });

    return this.validateResponse(response, "updateDocument");
  }

  /**
   * Export a bank's documents as a transfer ZIP archive (blocking convenience).
   *
   * The export runs as a background operation server-side (a whole-bank export can
   * be large). This helper submits it, polls the operation to completion, downloads
   * the archive, and resolves with its bytes. For the raw flow use the low-level
   * `sdk.exportDocuments` / `sdk.getOperationStatus` / `sdk.downloadFile`.
   *
   * @throws {HindsightError} if the export fails, times out, or completes without an archive.
   */
  async exportDocuments(
    bankId: string,
    options?: {
      documentIds?: string[];
      includeObservations?: boolean;
      /** Milliseconds between operation-status polls (default 2000). */
      pollIntervalMs?: number;
      /** Maximum milliseconds to wait for the export to finish (default 300000). */
      timeoutMs?: number;
      signal?: AbortSignal;
    }
  ): Promise<Uint8Array> {
    const submitResponse = await sdk.exportDocuments({
      client: this.client,
      path: { bank_id: bankId },
      query: {
        ...(options?.documentIds !== undefined ? { document_id: options.documentIds } : {}),
        ...(options?.includeObservations !== undefined
          ? { include_observations: options.includeObservations }
          : {}),
      },
      signal: options?.signal,
    });
    const submission = this.validateResponse(submitResponse, "exportDocuments");
    const operationId = submission.operation_id;

    const pollInterval = options?.pollIntervalMs ?? 2000;
    const timeout = options?.timeoutMs ?? 300000;
    const deadline = Date.now() + timeout;
    let resultMetadata: Record<string, unknown> | null | undefined;
    for (;;) {
      const statusResponse = await sdk.getOperationStatus({
        client: this.client,
        path: { bank_id: bankId, operation_id: operationId },
        signal: options?.signal,
      });
      const status = this.validateResponse(statusResponse, "getOperationStatus");
      if (status.status === "completed") {
        resultMetadata = status.result_metadata;
        break;
      }
      if (status.status === "failed" || status.status === "cancelled") {
        throw new HindsightError(
          `Export operation ${operationId} ${status.status}: ${status.error_message ?? ""}`
        );
      }
      if (Date.now() >= deadline) {
        throw new HindsightError(
          `Export operation ${operationId} did not complete within ${timeout}ms`
        );
      }
      await new Promise((resolve) => setTimeout(resolve, pollInterval));
    }

    const downloadUrl = (resultMetadata as { download_url?: string } | null | undefined)
      ?.download_url;
    if (!downloadUrl) {
      throw new HindsightError(`Export operation ${operationId} completed without a download_url`);
    }
    // Fetch the server-provided download_url directly (it carries the raw,
    // slash-bearing storage key). Going through the templated `downloadFile`
    // would percent-encode the slashes, which fronting proxies often reject.
    const downloadResponse = await this.client.get({
      url: downloadUrl,
      parseAs: "arrayBuffer",
      signal: options?.signal,
    });
    const data = this.validateResponse(downloadResponse as { data?: ArrayBuffer }, "downloadFile");
    return new Uint8Array(data);
  }
}

/**
 * Serialize a RecallResponse to a string suitable for LLM prompts.
 *
 * Builds a prompt containing:
 * - Facts: each result as a JSON object with text, context, temporal fields,
 *   and source_chunk (if the result's chunk_id matches a chunk in the response).
 * - Entities: entity summaries from observations, formatted as sections.
 *
 * Mirrors the format used internally by Hindsight's reflect operation.
 */
export function recallResponseToPromptString(response: RecallResponse): string {
  const chunksMap = response.chunks ?? {};
  const sections: string[] = [];

  // Facts
  const formattedFacts = (response.results ?? []).map((result) => {
    const obj: Record<string, string> = { text: result.text };
    if (result.context) obj.context = result.context;
    if (result.occurred_start) obj.occurred_start = result.occurred_start;
    if (result.occurred_end) obj.occurred_end = result.occurred_end;
    if (result.mentioned_at) obj.mentioned_at = result.mentioned_at;
    if (result.chunk_id && chunksMap[result.chunk_id]) {
      obj.source_chunk = chunksMap[result.chunk_id].text;
    }
    return obj;
  });
  sections.push("FACTS:\n" + JSON.stringify(formattedFacts, null, 2));

  // Entities
  const entities = response.entities;
  if (entities) {
    const entityParts: string[] = [];
    for (const [name, state] of Object.entries(entities)) {
      if (state.observations?.length) {
        entityParts.push(`## ${name}\n${state.observations[0].text}`);
      }
    }
    if (entityParts.length) {
      sections.push("ENTITIES:\n" + entityParts.join("\n\n"));
    }
  }

  return sections.join("\n\n");
}

// Re-export types for convenience
export type {
  RetainRequest,
  RetainResponse,
  RecallRequest,
  RecallResponse,
  RecallResult,
  ReflectRequest,
  ReflectResponse,
  FileRetainResponse,
  ListMemoryUnitsResponse,
  BankProfileResponse,
  BankConfigResponse,
  CreateBankRequest,
  Budget,
  BankTemplateManifest,
  BankTemplateConfig,
  BankTemplateMentalModel,
  BankTemplateDirective,
  BankTemplateImportResponse,
  TagGroupLeaf,
  TagGroupAndInput,
  TagGroupOrInput,
  TagGroupNotInput,
  MinScores,
  AsyncOperationSubmitResponse,
  CreateKnowledgePageResponse,
  CreateMentalModelResponse,
  DirectiveListResponse,
  DirectiveResponse,
  DocumentResponse,
  KnowledgeNode,
  KnowledgePageBundleResponse,
  KnowledgePageResponse,
  KnowledgePageSearchResponse,
  KnowledgeTreeResponse,
  ListDocumentsResponse,
  MentalModelListResponse,
  MentalModelResponse,
  MentalModelTriggerInput,
  MentalModelDryRunRefreshResult,
  UpdateDocumentResponse,
  VersionResponse,
};

// Also export low-level SDK functions for advanced usage
export * as sdk from "../generated/sdk.gen";
export { createClient, createConfig } from "../generated/client";
export type { Client } from "../generated/client";
