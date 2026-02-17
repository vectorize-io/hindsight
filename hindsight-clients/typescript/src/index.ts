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

import { createClient, createConfig } from '../generated/client';
import type { Client } from '../generated/client';
import * as sdk from '../generated/sdk.gen';
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
    CreateBankRequest,
    Budget,
} from '../generated/types.gen';

export interface HindsightClientOptions {
    baseUrl: string;
    /**
     * Optional API key for authentication (sent as Bearer token in Authorization header)
     */
    apiKey?: string;
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
        this.name = 'HindsightError';
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
    tags?: string[];
}

export class HindsightClient {
    private client: Client;

    constructor(options: HindsightClientOptions) {
        this.client = createClient(
            createConfig({
                baseUrl: options.baseUrl,
                headers: options.apiKey
                    ? { Authorization: `Bearer ${options.apiKey}` }
                    : undefined,
            })
        );
    }

    /**
     * Validates the API response and throws an error if the request failed.
     */
    private validateResponse<T>(response: { data?: T; error?: unknown; response?: Response }, operation: string): T {
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
            entities?: EntityInput[];
            /** Optional list of tags for this memory */
            tags?: string[];
        }
    ): Promise<RetainResponse> {
        const item: {
            content: string;
            timestamp?: string;
            context?: string;
            metadata?: Record<string, string>;
            document_id?: string;
            entities?: EntityInput[];
            tags?: string[];
        } = { content };
        if (options?.timestamp) {
            item.timestamp =
                options.timestamp instanceof Date
                    ? options.timestamp.toISOString()
                    : options.timestamp;
        }
        if (options?.context) {
            item.context = options.context;
        }
        if (options?.metadata) {
            item.metadata = options.metadata;
        }
        if (options?.documentId) {
            item.document_id = options.documentId;
        }
        if (options?.entities) {
            item.entities = options.entities;
        }
        if (options?.tags) {
            item.tags = options.tags;
        }

        const response = await sdk.retainMemories({
            client: this.client,
            path: { bank_id: bankId },
            body: { items: [item], async: options?.async },
        });

        return this.validateResponse(response, 'retain');
    }

    /**
     * Retain multiple memories in batch.
     */
    async retainBatch(bankId: string, items: MemoryItemInput[], options?: { documentId?: string; documentTags?: string[]; async?: boolean }): Promise<RetainResponse> {
        const processedItems = items.map((item) => ({
            content: item.content,
            context: item.context,
            metadata: item.metadata,
            document_id: item.document_id,
            entities: item.entities,
            tags: item.tags,
            timestamp:
                item.timestamp instanceof Date
                    ? item.timestamp.toISOString()
                    : item.timestamp,
        }));

        // If documentId is provided at the batch level, add it to all items that don't have one
        const itemsWithDocId = processedItems.map(item => ({
            ...item,
            document_id: item.document_id || options?.documentId
        }));

        const response = await sdk.retainMemories({
            client: this.client,
            path: { bank_id: bankId },
            body: {
                items: itemsWithDocId,
                document_tags: options?.documentTags,
                async: options?.async,
            },
        });

        return this.validateResponse(response, 'retainBatch');
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
            filesMetadata?: Array<{ context?: string; document_id?: string; tags?: string[]; metadata?: Record<string, string> }>;
        }
    ): Promise<FileRetainResponse> {
        const meta = options?.filesMetadata ?? files.map(() => options?.context ? { context: options.context } : {});

        const requestBody = JSON.stringify({
            files_metadata: meta,
        });

        const response = await sdk.fileRetain({
            client: this.client,
            path: { bank_id: bankId },
            body: { files, request: requestBody },
        });

        return this.validateResponse(response, 'retainFiles');
    }

    /**
     * Recall memories with a natural language query.
     */
    async recall(
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
            /** Optional list of tags to filter memories by */
            tags?: string[];
            /** How to match tags: 'any' (OR, includes untagged), 'all' (AND, includes untagged), 'any_strict' (OR, excludes untagged), 'all_strict' (AND, excludes untagged). Default: 'any' */
            tagsMatch?: 'any' | 'all' | 'any_strict' | 'all_strict';
        }
    ): Promise<RecallResponse> {
        const response = await sdk.recallMemories({
            client: this.client,
            path: { bank_id: bankId },
            body: {
                query,
                types: options?.types,
                max_tokens: options?.maxTokens,
                budget: options?.budget || 'mid',
                trace: options?.trace,
                query_timestamp: options?.queryTimestamp,
                include: {
                    entities: options?.includeEntities ? { max_tokens: options?.maxEntityTokens ?? 500 } : undefined,
                    chunks: options?.includeChunks ? { max_tokens: options?.maxChunkTokens ?? 8192 } : undefined,
                },
                tags: options?.tags,
                tags_match: options?.tagsMatch,
            },
        });

        return this.validateResponse(response, 'recall');
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
            /** How to match tags: 'any' (OR, includes untagged), 'all' (AND, includes untagged), 'any_strict' (OR, excludes untagged), 'all_strict' (AND, excludes untagged). Default: 'any' */
            tagsMatch?: 'any' | 'all' | 'any_strict' | 'all_strict';
        }
    ): Promise<ReflectResponse> {
        const response = await sdk.reflect({
            client: this.client,
            path: { bank_id: bankId },
            body: {
                query,
                context: options?.context,
                budget: options?.budget || 'low',
                tags: options?.tags,
                tags_match: options?.tagsMatch,
            },
        });

        return this.validateResponse(response, 'reflect');
    }

    /**
     * List memories with pagination.
     */
    async listMemories(
        bankId: string,
        options?: { limit?: number; offset?: number; type?: string; q?: string }
    ): Promise<ListMemoryUnitsResponse> {
        const response = await sdk.listMemories({
            client: this.client,
            path: { bank_id: bankId },
            query: {
                limit: options?.limit,
                offset: options?.offset,
                type: options?.type,
                q: options?.q,
            },
        });

        return this.validateResponse(response, 'listMemories');
    }

    /**
     * Create or update a bank with disposition and background.
     */
    async createBank(
        bankId: string,
        options: { name?: string; background?: string; disposition?: any }
    ): Promise<BankProfileResponse> {
        const response = await sdk.createOrUpdateBank({
            client: this.client,
            path: { bank_id: bankId },
            body: {
                name: options.name,
                background: options.background,
                disposition: options.disposition,
            },
        });

        return this.validateResponse(response, 'createBank');
    }

    /**
     * Get a bank's profile.
     */
    async getBankProfile(bankId: string): Promise<BankProfileResponse> {
        const response = await sdk.getBankProfile({
            client: this.client,
            path: { bank_id: bankId },
        });

        return this.validateResponse(response, 'getBankProfile');
    }

    /**
     * Set or update the mission for a memory bank.
     */
    async setMission(bankId: string, mission: string): Promise<BankProfileResponse> {
        const response = await sdk.createOrUpdateBank({
            client: this.client,
            path: { bank_id: bankId },
            body: { mission },
        });

        return this.validateResponse(response, 'setMission');
    }

    /**
     * Delete a bank.
     */
    async deleteBank(bankId: string): Promise<void> {
        const response = await sdk.deleteBank({
            client: this.client,
            path: { bank_id: bankId },
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
        }
    ): Promise<any> {
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
        });

        return this.validateResponse(response, 'createDirective');
    }

    /**
     * List all directives in a bank.
     */
    async listDirectives(bankId: string, options?: { tags?: string[] }): Promise<any> {
        const response = await sdk.listDirectives({
            client: this.client,
            path: { bank_id: bankId },
            query: { tags: options?.tags },
        });

        return this.validateResponse(response, 'listDirectives');
    }

    /**
     * Get a specific directive.
     */
    async getDirective(bankId: string, directiveId: string): Promise<any> {
        const response = await sdk.getDirective({
            client: this.client,
            path: { bank_id: bankId, directive_id: directiveId },
        });

        return this.validateResponse(response, 'getDirective');
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
        }
    ): Promise<any> {
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
        });

        return this.validateResponse(response, 'updateDirective');
    }

    /**
     * Delete a directive.
     */
    async deleteDirective(bankId: string, directiveId: string): Promise<void> {
        const response = await sdk.deleteDirective({
            client: this.client,
            path: { bank_id: bankId, directive_id: directiveId },
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
            tags?: string[];
            maxTokens?: number;
            trigger?: { refreshAfterConsolidation?: boolean };
        }
    ): Promise<any> {
        const response = await sdk.createMentalModel({
            client: this.client,
            path: { bank_id: bankId },
            body: {
                name,
                source_query: sourceQuery,
                tags: options?.tags,
                max_tokens: options?.maxTokens,
                trigger: options?.trigger ? { refresh_after_consolidation: options.trigger.refreshAfterConsolidation } : undefined,
            },
        });

        return this.validateResponse(response, 'createMentalModel');
    }

    /**
     * List all mental models in a bank.
     */
    async listMentalModels(bankId: string, options?: { tags?: string[] }): Promise<any> {
        const response = await sdk.listMentalModels({
            client: this.client,
            path: { bank_id: bankId },
            query: { tags: options?.tags },
        });

        return this.validateResponse(response, 'listMentalModels');
    }

    /**
     * Get a specific mental model.
     */
    async getMentalModel(bankId: string, mentalModelId: string): Promise<any> {
        const response = await sdk.getMentalModel({
            client: this.client,
            path: { bank_id: bankId, mental_model_id: mentalModelId },
        });

        return this.validateResponse(response, 'getMentalModel');
    }

    /**
     * Refresh a mental model to update with current knowledge.
     */
    async refreshMentalModel(bankId: string, mentalModelId: string): Promise<any> {
        const response = await sdk.refreshMentalModel({
            client: this.client,
            path: { bank_id: bankId, mental_model_id: mentalModelId },
        });

        return this.validateResponse(response, 'refreshMentalModel');
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
        }
    ): Promise<any> {
        const response = await sdk.updateMentalModel({
            client: this.client,
            path: { bank_id: bankId, mental_model_id: mentalModelId },
            body: {
                name: options.name,
                source_query: options.sourceQuery,
                tags: options.tags,
                max_tokens: options.maxTokens,
                trigger: options.trigger ? { refresh_after_consolidation: options.trigger.refreshAfterConsolidation } : undefined,
            },
        });

        return this.validateResponse(response, 'updateMentalModel');
    }

    /**
     * Delete a mental model.
     */
    async deleteMentalModel(bankId: string, mentalModelId: string): Promise<void> {
        const response = await sdk.deleteMentalModel({
            client: this.client,
            path: { bank_id: bankId, mental_model_id: mentalModelId },
        });
        if (response.error) {
            throw new Error(`deleteMentalModel failed: ${JSON.stringify(response.error)}`);
        }
    }
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
    CreateBankRequest,
    Budget,
};

// Also export low-level SDK functions for advanced usage
export * as sdk from '../generated/sdk.gen';
export { createClient, createConfig } from '../generated/client';
export type { Client } from '../generated/client';
