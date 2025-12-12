/**
 * Hindsight Client - Clean, TypeScript SDK for the Hindsight API.
 *
 * Example:
 * ```typescript
 * import { HindsightClient } from '@vectorize-io/hindsight-client';
 *
 * const client = new HindsightClient({ baseUrl: 'http://localhost:8888' });
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
    ListMemoryUnitsResponse,
    BankProfileResponse,
    CreateBankRequest,
    Budget,
} from '../generated/types.gen';

export interface HindsightClientOptions {
    baseUrl: string;
}

export interface MemoryItemInput {
    content: string;
    timestamp?: string | Date;
    context?: string;
    metadata?: Record<string, string>;
    document_id?: string;
}

export class HindsightClient {
    private client: Client;

    constructor(options: HindsightClientOptions) {
        this.client = createClient(
            createConfig({
                baseUrl: options.baseUrl,
            })
        );
    }

    /**
     * Retain a single memory for a bank.
     */
    async retain(
        bankId: string,
        content: string,
        options?: { timestamp?: Date | string; context?: string; metadata?: Record<string, string>; async?: boolean }
    ): Promise<RetainResponse> {
        const item: { content: string; timestamp?: string; context?: string; metadata?: Record<string, string> } = { content };
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

        const response = await sdk.retainMemories({
            client: this.client,
            path: { bank_id: bankId },
            body: { items: [item], async: options?.async },
        });

        return response.data!;
    }

    /**
     * Retain multiple memories in batch.
     */
    async retainBatch(bankId: string, items: MemoryItemInput[], options?: { documentId?: string; async?: boolean }): Promise<RetainResponse> {
        const processedItems = items.map((item) => ({
            content: item.content,
            context: item.context,
            metadata: item.metadata,
            document_id: item.document_id,
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
                async: options?.async,
            },
        });

        return response.data!;
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
            },
        });

        if (!response.data) {
            throw new Error(`API returned no data: ${JSON.stringify(response.error || 'Unknown error')}`);
        }

        return response.data;
    }

    /**
     * Reflect and generate a contextual answer using the bank's identity and memories.
     */
    async reflect(
        bankId: string,
        query: string,
        options?: { context?: string; budget?: Budget }
    ): Promise<ReflectResponse> {
        const response = await sdk.reflect({
            client: this.client,
            path: { bank_id: bankId },
            body: {
                query,
                context: options?.context,
                budget: options?.budget || 'low',
            },
        });

        return response.data!;
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

        return response.data!;
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

        return response.data!;
    }

    /**
     * Get a bank's profile.
     */
    async getBankProfile(bankId: string): Promise<BankProfileResponse> {
        const response = await sdk.getBankProfile({
            client: this.client,
            path: { bank_id: bankId },
        });

        return response.data!;
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
    ListMemoryUnitsResponse,
    BankProfileResponse,
    CreateBankRequest,
    Budget,
};

// Also export low-level SDK functions for advanced usage
export * as sdk from '../generated/sdk.gen';
export { createClient, createConfig } from '../generated/client';
export type { Client } from '../generated/client';
