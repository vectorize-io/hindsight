/**
 * Client for calling Control Plane API routes (which proxy to the dataplane via SDK)
 * This should be used in client components, not the SDK directly
 */

import { toast } from "sonner";
import { bankApi, bankStatsApi, documentApi, memoryApi } from "./bank-url";

export interface WebhookHttpConfig {
  method: string;
  timeout_seconds: number;
  headers: Record<string, string>;
  params: Record<string, string>;
}

export interface Webhook {
  id: string;
  bank_id: string | null;
  url: string;
  event_types: string[];
  enabled: boolean;
  http_config: WebhookHttpConfig;
  created_at: string | null;
  updated_at: string | null;
}

export interface WebhookDelivery {
  id: string;
  webhook_id: string | null;
  url: string;
  event_type: string;
  status: string;
  attempts: number;
  next_retry_at: string | null;
  last_error: string | null;
  last_response_status: number | null;
  last_response_body: string | null;
  last_attempt_at: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface AuditLogEntry {
  id: string;
  action: string;
  transport: string;
  bank_id: string | null;
  started_at: string | null;
  ended_at: string | null;
  request: Record<string, unknown> | null;
  response: Record<string, unknown> | null;
  metadata: Record<string, unknown>;
}

export interface AuditLogsResponse {
  bank_id: string;
  total: number;
  limit: number;
  offset: number;
  items: AuditLogEntry[];
}

export interface AuditStatsBucket {
  time: string;
  actions: Record<string, number>;
  total: number;
}

export interface AuditStatsResponse {
  bank_id: string;
  period: string;
  trunc: string;
  start: string;
  buckets: AuditStatsBucket[];
}

export type TagsMatch = "any" | "all" | "any_strict" | "all_strict";

export type TagGroup =
  | { tags: string[]; match?: TagsMatch }
  | { and: TagGroup[] }
  | { or: TagGroup[] }
  | { not: TagGroup };

export interface MentalModel {
  id: string;
  bank_id: string;
  name: string;
  source_query: string;
  content: string;
  tags: string[];
  max_tokens: number;
  trigger: {
    mode?: "full" | "delta";
    refresh_after_consolidation: boolean;
    fact_types?: Array<"world" | "experience" | "observation">;
    exclude_mental_models?: boolean;
    exclude_mental_model_ids?: string[];
    tags_match?: TagsMatch;
    tag_groups?: TagGroup[];
    include_chunks?: boolean;
    recall_max_tokens?: number;
    recall_chunks_max_tokens?: number;
  };
  last_refreshed_at: string;
  created_at: string;
  reflect_response?: any;
  is_stale?: boolean | null;
}

export interface BankTemplateImportResponse {
  bank_id: string;
  config_applied: boolean;
  mental_models_created: string[];
  mental_models_updated: string[];
  operation_ids: string[];
  dry_run: boolean;
}

export class ControlPlaneClient {
  private async fetchApi<T>(path: string, options?: RequestInit): Promise<T> {
    try {
      const response = await fetch(path, {
        ...options,
        headers: {
          "Content-Type": "application/json",
          ...options?.headers,
        },
      });

      if (!response.ok) {
        // Try to parse error response
        let errorMessage = `HTTP ${response.status}`;
        let errorDetails: string | undefined;

        try {
          const errorData = await response.json();
          errorMessage = errorData.error || errorMessage;
          errorDetails = errorData.details;
        } catch {
          // If JSON parse fails, try to get text
          try {
            const errorText = await response.text();
            if (errorText) {
              errorDetails = errorText;
            }
          } catch {
            // Ignore text parse errors
          }
        }

        // Show toast with different styles based on status code
        const description = errorDetails || errorMessage;
        const status = response.status;

        if (status >= 400 && status < 500) {
          // Client errors (4xx) - validation, bad request, etc. - show as warning
          toast.warning("Client Error", {
            description,
            duration: 5000,
          });
        } else if (status >= 500) {
          // Server errors (5xx) - show as error
          toast.error("Server Error", {
            description,
            duration: 5000,
          });
        } else {
          // Other HTTP errors - show as error
          toast.error("API Error", {
            description,
            duration: 5000,
          });
        }

        // Still throw error for callers that want to handle it
        const error = new Error(errorMessage);
        (error as any).status = response.status;
        (error as any).details = errorDetails;
        throw error;
      }

      return response.json();
    } catch (error) {
      // If it's not a response error (network error, etc.), show toast
      if (!(error as any).status) {
        toast.error("Network Error", {
          description: error instanceof Error ? error.message : "Failed to connect to server",
          duration: 5000,
        });
      }
      throw error;
    }
  }

  /**
   * List all banks
   */
  async listBanks() {
    return this.fetchApi<{ banks: any[] }>("/api/banks", { cache: "no-store" as RequestCache });
  }

  /**
   * Create a new bank
   */
  async createBank(bankId: string) {
    return this.fetchApi<{ bank_id: string }>("/api/banks", {
      method: "POST",
      body: JSON.stringify({ bank_id: bankId }),
    });
  }

  /**
   * Import a bank template manifest
   */
  async importBankTemplate(bankId: string, manifest: Record<string, unknown>, dryRun = false) {
    const params = dryRun ? "?dry_run=true" : "";
    return this.fetchApi<BankTemplateImportResponse>(bankApi(bankId, `/import${params}`), {
      method: "POST",
      body: JSON.stringify(manifest),
    });
  }

  /**
   * Export a bank as a template manifest
   */
  async exportBankTemplate(bankId: string) {
    return this.fetchApi<Record<string, unknown>>(bankApi(bankId, "/export"));
  }

  /**
   * Recall memories
   */
  async recall(params: {
    query: string;
    types?: string[];
    bank_id: string;
    budget?: string;
    max_tokens?: number;
    trace?: boolean;
    include?: {
      entities?: { max_tokens: number } | null;
      chunks?: { max_tokens: number } | null;
      source_facts?: { max_tokens?: number } | null;
    };
    query_timestamp?: string;
    tags?: string[];
    tags_match?: "any" | "all" | "any_strict" | "all_strict";
  }) {
    return this.fetchApi("/api/recall", {
      method: "POST",
      body: JSON.stringify(params),
    });
  }

  /**
   * Reflect and generate answer
   */
  async reflect(params: {
    query: string;
    bank_id: string;
    budget?: string;
    max_tokens?: number;
    include_facts?: boolean;
    include_tool_calls?: boolean;
    tags?: string[];
    tags_match?: "any" | "all" | "any_strict" | "all_strict";
    fact_types?: Array<"world" | "experience" | "observation">;
    exclude_mental_models?: boolean;
    exclude_mental_model_ids?: string[];
  }) {
    return this.fetchApi("/api/reflect", {
      method: "POST",
      body: JSON.stringify(params),
    });
  }

  /**
   * Retain memories (batch)
   */
  async retain(params: {
    bank_id: string;
    items: Array<{
      content: string;
      timestamp?: string;
      context?: string;
      document_id?: string;
      metadata?: Record<string, string>;
      entities?: Array<{ text: string; type?: string }>;
      tags?: string[];
      observation_scopes?: "per_tag" | "combined" | "all_combinations" | string[][];
      strategy?: string;
    }>;
    document_id?: string;
    async?: boolean;
  }) {
    const endpoint = params.async ? "/api/memories/retain_async" : "/api/memories/retain";
    return this.fetchApi<{ message?: string }>(endpoint, {
      method: "POST",
      body: JSON.stringify(params),
    });
  }

  /**
   * Get bank statistics
   */
  async getBankStats(bankId: string) {
    return this.fetchApi(bankStatsApi(bankId));
  }

  async getMemoriesTimeseries(bankId: string, period: string) {
    return this.fetchApi<{
      bank_id: string;
      period: string;
      trunc: string;
      buckets: Array<{
        time: string;
        world: number;
        experience: number;
        observation: number;
      }>;
    }>(bankStatsApi(bankId, `/memories-timeseries?period=${encodeURIComponent(period)}`));
  }

  /**
   * Get graph data
   */
  async getGraph(params: {
    bank_id: string;
    type?: string;
    limit?: number;
    q?: string;
    tags?: string[];
  }) {
    const queryParams = new URLSearchParams();
    queryParams.append("bank_id", params.bank_id);
    if (params.type) queryParams.append("type", params.type);
    if (params.limit) queryParams.append("limit", params.limit.toString());
    if (params.q) queryParams.append("q", params.q);
    if (params.tags && params.tags.length > 0) {
      params.tags.forEach((tag) => queryParams.append("tags", tag));
    }
    return this.fetchApi(`/api/graph?${queryParams}`);
  }

  /**
   * List operations with optional filtering and pagination
   */
  async listOperations(
    bankId: string,
    options?: { status?: string; type?: string; limit?: number; offset?: number }
  ) {
    const params = new URLSearchParams();
    if (options?.status) params.append("status", options.status);
    if (options?.type) params.append("type", options.type);
    if (options?.limit) params.append("limit", options.limit.toString());
    if (options?.offset) params.append("offset", options.offset.toString());
    const query = params.toString();
    return this.fetchApi<{
      bank_id: string;
      total: number;
      limit: number;
      offset: number;
      operations: Array<{
        id: string;
        task_type: string;
        items_count: number;
        document_id: string | null;
        created_at: string;
        status: string;
        error_message: string | null;
      }>;
    }>(`/api/operations/${encodeURIComponent(bankId)}${query ? `?${query}` : ""}`);
  }

  /**
   * Cancel a pending operation
   */
  async cancelOperation(bankId: string, operationId: string) {
    return this.fetchApi<{
      success: boolean;
      message: string;
      operation_id: string;
    }>(
      `/api/operations/${encodeURIComponent(bankId)}?operation_id=${encodeURIComponent(operationId)}`,
      {
        method: "DELETE",
      }
    );
  }

  /**
   * Retry a failed operation
   */
  async retryOperation(bankId: string, operationId: string) {
    return this.fetchApi<{
      success: boolean;
      message: string;
      operation_id: string;
    }>(bankApi(bankId, `/operations/${encodeURIComponent(operationId)}`), {
      method: "POST",
    });
  }

  /**
   * List entities
   */
  async listEntities(params: { bank_id: string; limit?: number; offset?: number }) {
    const queryParams = new URLSearchParams();
    queryParams.append("bank_id", params.bank_id);
    if (params.limit) queryParams.append("limit", params.limit.toString());
    if (params.offset) queryParams.append("offset", params.offset.toString());
    return this.fetchApi<{
      items: any[];
      total: number;
      limit: number;
      offset: number;
    }>(`/api/entities?${queryParams}`);
  }

  /**
   * Get entity co-occurrence graph
   */
  async getEntityGraph(params: { bank_id: string; limit?: number; min_count?: number }) {
    const queryParams = new URLSearchParams();
    queryParams.append("bank_id", params.bank_id);
    if (params.limit) queryParams.append("limit", params.limit.toString());
    if (params.min_count !== undefined)
      queryParams.append("min_count", params.min_count.toString());
    return this.fetchApi<{
      nodes: Array<{ data: { id: string; label: string; mentionCount: number; color: string } }>;
      edges: Array<{
        data: {
          id: string;
          source: string;
          target: string;
          linkType: string;
          weight: number;
          color: string;
          lineStyle: string;
          lastCooccurred: string | null;
        };
      }>;
      total_entities: number;
      total_edges: number;
      limit: number;
    }>(`/api/entities/graph?${queryParams}`);
  }

  /**
   * Get entity details
   */
  async getEntity(entityId: string, bankId: string) {
    return this.fetchApi(
      `/api/entities/${encodeURIComponent(entityId)}?bank_id=${encodeURIComponent(bankId)}`
    );
  }

  /**
   * Regenerate entity observations
   */
  async regenerateEntityObservations(entityId: string, bankId: string) {
    return this.fetchApi(
      `/api/entities/${encodeURIComponent(entityId)}/regenerate?bank_id=${encodeURIComponent(bankId)}`,
      {
        method: "POST",
      }
    );
  }

  /**
   * List documents
   */
  async listDocuments(params: { bank_id: string; q?: string; limit?: number; offset?: number }) {
    const queryParams = new URLSearchParams();
    queryParams.append("bank_id", params.bank_id);
    if (params.q) queryParams.append("q", params.q);
    if (params.limit) queryParams.append("limit", params.limit.toString());
    if (params.offset) queryParams.append("offset", params.offset.toString());
    return this.fetchApi(`/api/documents?${queryParams}`);
  }

  /**
   * Get document
   */
  async getDocument(documentId: string, bankId: string) {
    return this.fetchApi(documentApi(documentId, bankId));
  }

  /**
   * Update tags on a document and its associated memory units
   */
  async updateDocument(documentId: string, bankId: string, tags: string[]) {
    return this.fetchApi<{ success: boolean }>(documentApi(documentId, bankId), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tags }),
    });
  }

  /**
   * Delete document and all its associated memory units
   */
  async deleteDocument(documentId: string, bankId: string) {
    return this.fetchApi<{
      success: boolean;
      message: string;
      document_id: string;
      memory_units_deleted: number;
    }>(documentApi(documentId, bankId), {
      method: "DELETE",
    });
  }

  /**
   * Delete an entire memory bank and all its data
   */
  async deleteBank(bankId: string) {
    return this.fetchApi<{
      success: boolean;
      message: string;
      deleted_count: number;
    }>(bankApi(bankId), {
      method: "DELETE",
    });
  }

  /**
   * Clear all observations for a bank
   */
  async clearObservations(bankId: string) {
    return this.fetchApi<{
      success: boolean;
      message: string;
      deleted_count: number;
    }>(bankApi(bankId, "/observations"), {
      method: "DELETE",
    });
  }

  /**
   * Trigger consolidation for a bank
   */
  async triggerConsolidation(bankId: string) {
    return this.fetchApi<{
      operation_id: string;
      deduplicated: boolean;
    }>(bankApi(bankId, "/consolidate"), {
      method: "POST",
    });
  }

  /**
   * Recover failed consolidation for a bank (reset memories marked consolidation_failed_at)
   */
  async recoverConsolidation(bankId: string) {
    return this.fetchApi<{
      retried_count: number;
    }>(bankApi(bankId, "/consolidation-recover"), {
      method: "POST",
    });
  }

  /**
   * List memory units for a bank with optional filters.
   */
  async listMemories(
    bankId: string,
    options?: {
      type?: string;
      q?: string;
      consolidationState?: "failed" | "pending" | "done";
      limit?: number;
      offset?: number;
    }
  ) {
    const params = new URLSearchParams({ bank_id: bankId });
    if (options?.type) params.set("type", options.type);
    if (options?.q) params.set("q", options.q);
    if (options?.consolidationState) params.set("consolidation_state", options.consolidationState);
    if (options?.limit !== undefined) params.set("limit", String(options.limit));
    if (options?.offset !== undefined) params.set("offset", String(options.offset));
    return this.fetchApi<{
      items: Array<{
        id: string;
        text: string;
        context: string;
        date: string;
        fact_type: string;
        mentioned_at: string | null;
        occurred_start: string | null;
        occurred_end: string | null;
        entities: string;
        chunk_id: string | null;
        proof_count: number;
        tags: string[];
        consolidated_at: string | null;
        consolidation_failed_at: string | null;
      }>;
      total: number;
      limit: number;
      offset: number;
    }>(`/api/list?${params.toString()}`);
  }

  /**
   * Get chunk
   */
  async getChunk(chunkId: string) {
    return this.fetchApi(`/api/chunks/${chunkId}`);
  }

  /**
   * Get a single memory by ID
   */
  async getMemory(memoryId: string, bankId: string) {
    return this.fetchApi<{
      id: string;
      text: string;
      context: string;
      date: string;
      type: string;
      mentioned_at: string | null;
      occurred_start: string | null;
      occurred_end: string | null;
      entities: string[];
      document_id: string | null;
      chunk_id: string | null;
      tags: string[];
      observation_scopes: string | string[][] | null;
      history?: {
        previous_text: string;
        previous_tags: string[];
        previous_occurred_start: string | null;
        previous_occurred_end: string | null;
        previous_mentioned_at: string | null;
        changed_at: string;
        new_source_memory_ids: string[];
      }[];
    }>(memoryApi(memoryId, bankId));
  }

  /**
   * Get the history of an observation with resolved source facts
   */
  async getObservationHistory(memoryId: string, bankId: string) {
    return this.fetchApi<
      {
        previous_text: string;
        previous_tags: string[];
        previous_occurred_start: string | null;
        previous_occurred_end: string | null;
        previous_mentioned_at: string | null;
        changed_at: string;
        new_source_memory_ids: string[];
        source_facts: {
          id: string;
          text: string | null;
          type: string | null;
          context: string | null;
          is_new: boolean;
        }[];
      }[]
    >(memoryApi(memoryId, bankId, "/history"));
  }

  /**
   * Get bank profile
   */
  async getBankProfile(bankId: string) {
    return this.fetchApi<{
      bank_id: string;
      name: string;
      disposition: {
        skepticism: number;
        literalism: number;
        empathy: number;
      };
      mission: string;
      background?: string; // Deprecated, kept for backwards compatibility
    }>(`/api/profile/${encodeURIComponent(bankId)}`);
  }

  /**
   * Set bank mission
   */
  async setBankMission(bankId: string, mission: string) {
    return this.fetchApi(bankApi(bankId), {
      method: "PATCH",
      body: JSON.stringify({ mission }),
    });
  }

  /**
   * List directives for a bank
   */
  async listDirectives(bankId: string, tags?: string[], tagsMatch?: string) {
    const params = new URLSearchParams();
    if (tags && tags.length > 0) {
      tags.forEach((t) => params.append("tags", t));
    }
    if (tagsMatch) {
      params.append("tags_match", tagsMatch);
    }
    const query = params.toString();
    return this.fetchApi<{
      items: Array<{
        id: string;
        bank_id: string;
        name: string;
        content: string;
        priority: number;
        is_active: boolean;
        tags: string[];
        created_at: string;
        updated_at: string;
      }>;
    }>(bankApi(bankId, `/directives${query ? `?${query}` : ""}`));
  }

  /**
   * Create a directive
   */
  async createDirective(
    bankId: string,
    params: {
      name: string;
      content: string;
      priority?: number;
      is_active?: boolean;
      tags?: string[];
    }
  ) {
    return this.fetchApi<{
      id: string;
      bank_id: string;
      name: string;
      content: string;
      priority: number;
      is_active: boolean;
      tags: string[];
      created_at: string;
      updated_at: string;
    }>(bankApi(bankId, "/directives"), {
      method: "POST",
      body: JSON.stringify(params),
    });
  }

  /**
   * Get a directive
   */
  async getDirective(bankId: string, directiveId: string) {
    return this.fetchApi<{
      id: string;
      bank_id: string;
      name: string;
      content: string;
      priority: number;
      is_active: boolean;
      tags: string[];
      created_at: string;
      updated_at: string;
    }>(bankApi(bankId, `/directives/${encodeURIComponent(directiveId)}`));
  }

  /**
   * Delete a directive
   */
  async deleteDirective(bankId: string, directiveId: string) {
    return this.fetchApi(bankApi(bankId, `/directives/${encodeURIComponent(directiveId)}`), {
      method: "DELETE",
    });
  }

  /**
   * Update a directive
   */
  async updateDirective(
    bankId: string,
    directiveId: string,
    params: {
      name?: string;
      content?: string;
      priority?: number;
      is_active?: boolean;
      tags?: string[];
    }
  ) {
    return this.fetchApi<{
      id: string;
      bank_id: string;
      name: string;
      content: string;
      priority: number;
      is_active: boolean;
      tags: string[];
      created_at: string;
      updated_at: string;
    }>(bankApi(bankId, `/directives/${encodeURIComponent(directiveId)}`), {
      method: "PATCH",
      body: JSON.stringify(params),
    });
  }

  /**
   * Get operation status
   */
  async getOperationStatus(
    bankId: string,
    operationId: string,
    opts?: { includePayload?: boolean }
  ) {
    const qs = opts?.includePayload ? "?include_payload=true" : "";
    return this.fetchApi<{
      operation_id: string;
      status: "pending" | "completed" | "failed" | "not_found";
      operation_type: string | null;
      created_at: string | null;
      updated_at: string | null;
      completed_at: string | null;
      error_message: string | null;
      result_metadata?: {
        items_count?: number;
        total_tokens?: number;
        num_sub_batches?: number;
        is_parent?: boolean;
        [key: string]: any;
      } | null;
      child_operations?: Array<{
        operation_id: string;
        status: string;
        sub_batch_index: number | null;
        items_count: number | null;
        error_message: string | null;
      }> | null;
      task_payload?: Record<string, unknown> | null;
    }>(bankApi(bankId, `/operations/${encodeURIComponent(operationId)}${qs}`));
  }

  /**
   * Update bank profile
   */
  async updateBankProfile(
    bankId: string,
    profile: {
      name?: string;
      disposition?: {
        skepticism: number;
        literalism: number;
        empathy: number;
      };
      mission?: string;
    }
  ) {
    return this.fetchApi(`/api/profile/${encodeURIComponent(bankId)}`, {
      method: "PUT",
      body: JSON.stringify(profile),
    });
  }

  // ============= OBSERVATIONS (auto-consolidated, read-only) =============

  /**
   * List observations for a bank (auto-consolidated knowledge)
   */
  async listObservations(bankId: string, tags?: string[], tagsMatch?: string) {
    const params = new URLSearchParams();
    if (tags && tags.length > 0) {
      tags.forEach((t) => params.append("tags", t));
    }
    if (tagsMatch) {
      params.append("tags_match", tagsMatch);
    }
    const query = params.toString();
    return this.fetchApi<{
      items: Array<{
        id: string;
        bank_id: string;
        text: string;
        proof_count: number;
        history: Array<{
          previous_text: string;
          changed_at: string;
          reason: string;
        }>;
        tags: string[];
        source_memory_ids: string[];
        source_memories: Array<{
          id: string;
          text: string;
          type: string;
          context?: string;
          occurred_start?: string;
          mentioned_at?: string;
        }>;
        created_at: string;
        updated_at: string;
      }>;
    }>(bankApi(bankId, `/observations${query ? `?${query}` : ""}`));
  }

  /**
   * Get an observation with source memories
   */
  async getObservation(bankId: string, observationId: string) {
    return this.fetchApi<{
      id: string;
      bank_id: string;
      text: string;
      proof_count: number;
      history: Array<{
        previous_text: string;
        changed_at: string;
        reason: string;
      }>;
      tags: string[];
      source_memory_ids: string[];
      source_memories: Array<{
        id: string;
        text: string;
        type: string;
        context?: string;
        occurred_start?: string;
        mentioned_at?: string;
      }>;
      created_at: string;
      updated_at: string;
    }>(bankApi(bankId, `/observations/${encodeURIComponent(observationId)}`));
  }

  // ============= MENTAL MODELS (stored reflect responses) =============

  /**
   * List mental models for a bank
   */
  async listMentalModels(bankId: string, tags?: string[], tagsMatch?: string) {
    const params = new URLSearchParams();
    if (tags && tags.length > 0) {
      tags.forEach((t) => params.append("tags", t));
    }
    if (tagsMatch) {
      params.append("tags_match", tagsMatch);
    }
    const query = params.toString();
    return this.fetchApi<{
      items: Array<{
        id: string;
        bank_id: string;
        name: string;
        source_query: string;
        content: string;
        tags: string[];
        max_tokens: number;
        trigger: {
          mode?: "full" | "delta";
          refresh_after_consolidation: boolean;
          fact_types?: Array<"world" | "experience" | "observation">;
          exclude_mental_models?: boolean;
          exclude_mental_model_ids?: string[];
          tags_match?: TagsMatch;
          tag_groups?: TagGroup[];
          include_chunks?: boolean;
          recall_max_tokens?: number;
          recall_chunks_max_tokens?: number;
        };
        last_refreshed_at: string;
        created_at: string;
        reflect_response?: {
          text: string;
          based_on: Record<string, Array<{ id: string; text: string; type: string }>>;
        };
      }>;
    }>(bankApi(bankId, `/mental-models${query ? `?${query}` : ""}`));
  }

  /**
   * Create a mental model (async - content auto-generated in background)
   * Returns operation_id to track progress
   */
  async createMentalModel(
    bankId: string,
    params: {
      id?: string;
      name: string;
      source_query: string;
      tags?: string[];
      max_tokens?: number;
      trigger?: {
        mode?: "full" | "delta";
        refresh_after_consolidation: boolean;
        fact_types?: Array<"world" | "experience" | "observation">;
        exclude_mental_models?: boolean;
        exclude_mental_model_ids?: string[];
        tags_match?: TagsMatch;
        tag_groups?: TagGroup[];
        include_chunks?: boolean;
        recall_max_tokens?: number;
        recall_chunks_max_tokens?: number;
      };
    }
  ) {
    return this.fetchApi<{
      operation_id: string;
    }>(bankApi(bankId, "/mental-models"), {
      method: "POST",
      body: JSON.stringify(params),
    });
  }

  /**
   * Get a mental model
   */
  async getMentalModel(bankId: string, mentalModelId: string): Promise<MentalModel> {
    return this.fetchApi<MentalModel>(
      bankApi(bankId, `/mental-models/${encodeURIComponent(mentalModelId)}`)
    );
  }

  /**
   * Update a mental model
   */
  async updateMentalModel(
    bankId: string,
    mentalModelId: string,
    params: {
      name?: string;
      source_query?: string;
      max_tokens?: number;
      tags?: string[];
      trigger?: {
        mode?: "full" | "delta";
        refresh_after_consolidation: boolean;
        fact_types?: Array<"world" | "experience" | "observation">;
        exclude_mental_models?: boolean;
        exclude_mental_model_ids?: string[];
        tags_match?: TagsMatch;
        tag_groups?: TagGroup[];
        include_chunks?: boolean;
        recall_max_tokens?: number;
        recall_chunks_max_tokens?: number;
      };
    }
  ) {
    return this.fetchApi<{
      id: string;
      bank_id: string;
      name: string;
      source_query: string;
      content: string;
      tags: string[];
      max_tokens: number;
      trigger: {
        refresh_after_consolidation: boolean;
        fact_types?: Array<"world" | "experience" | "observation">;
        exclude_mental_models?: boolean;
        exclude_mental_model_ids?: string[];
        tags_match?: TagsMatch;
        tag_groups?: TagGroup[];
        include_chunks?: boolean;
        recall_max_tokens?: number;
        recall_chunks_max_tokens?: number;
      };
      last_refreshed_at: string;
      created_at: string;
      reflect_response?: {
        text: string;
        based_on: Record<string, Array<{ id: string; text: string; type: string }>>;
      };
    }>(bankApi(bankId, `/mental-models/${encodeURIComponent(mentalModelId)}`), {
      method: "PATCH",
      body: JSON.stringify(params),
    });
  }

  /**
   * Delete a mental model
   */
  async deleteMentalModel(bankId: string, mentalModelId: string) {
    return this.fetchApi(bankApi(bankId, `/mental-models/${encodeURIComponent(mentalModelId)}`), {
      method: "DELETE",
    });
  }

  /**
   * Refresh a mental model (re-run source query) - async operation
   */
  async refreshMentalModel(bankId: string, mentalModelId: string) {
    return this.fetchApi<{
      operation_id: string;
    }>(bankApi(bankId, `/mental-models/${encodeURIComponent(mentalModelId)}/refresh`), {
      method: "POST",
    });
  }

  /**
   * Get the refresh history of a mental model
   */
  async getMentalModelHistory(bankId: string, mentalModelId: string) {
    return this.fetchApi<
      {
        previous_content: string | null;
        previous_reflect_response: {
          text?: string;
          based_on?: Record<
            string,
            { id: string; text: string; type: string; context?: string | null }[]
          >;
          mental_models?: unknown[];
        } | null;
        changed_at: string;
      }[]
    >(bankApi(bankId, `/mental-models/${encodeURIComponent(mentalModelId)}/history`));
  }

  /**
   * Get API version and feature flags
   * Use this to check which capabilities are available in the dataplane
   */
  async getVersion() {
    return this.fetchApi<{
      api_version: string;
      features: {
        observations: boolean;
        mcp: boolean;
        worker: boolean;
        bank_config_api: boolean;
        file_upload_api: boolean;
      };
    }>("/api/version");
  }

  /**
   * Upload files for retain (uses file conversion API)
   * Requires file_upload_api feature flag to be enabled
   * Converter is configured server-side via HINDSIGHT_API_FILE_CONVERTER
   */
  async uploadFiles(params: {
    bank_id: string;
    files: File[];
    document_tags?: string[];
    async?: boolean;
    files_metadata?: Array<{
      document_id?: string;
      context?: string;
      metadata?: Record<string, any>;
      tags?: string[];
      timestamp?: string;
      strategy?: string;
    }>;
  }) {
    const formData = new FormData();

    // Add files
    params.files.forEach((file) => {
      formData.append("files", file);
    });

    // Add request JSON (including bank_id)
    const requestData: any = {
      bank_id: params.bank_id,
      async: params.async ?? true,
    };
    if (params.document_tags) requestData.document_tags = params.document_tags;
    if (params.files_metadata) requestData.files_metadata = params.files_metadata;

    formData.append("request", JSON.stringify(requestData));

    // Use fetch directly for multipart/form-data
    const response = await fetch(`/api/files/retain`, {
      method: "POST",
      body: formData,
      // Don't set Content-Type - browser will set it with boundary
    });

    if (!response.ok) {
      let errorMessage = `HTTP ${response.status}`;
      try {
        const errorData = await response.json();
        errorMessage = errorData.error || errorMessage;
      } catch {
        // Ignore parse errors
      }
      const error = new Error(errorMessage);
      (error as any).status = response.status;
      throw error;
    }

    return response.json();
  }

  /**
   * Get bank configuration (resolved with hierarchy)
   */
  async getBankConfig(bankId: string) {
    return this.fetchApi<{
      bank_id: string;
      config: Record<string, any>;
      overrides: Record<string, any>;
    }>(bankApi(bankId, "/config"));
  }

  /**
   * Update bank configuration overrides
   */
  async updateBankConfig(bankId: string, updates: Record<string, any>) {
    return this.fetchApi<{
      bank_id: string;
      config: Record<string, any>;
      overrides: Record<string, any>;
    }>(bankApi(bankId, "/config"), {
      method: "PATCH",
      body: JSON.stringify({ updates }),
    });
  }

  /**
   * Reset bank configuration to defaults
   */
  async resetBankConfig(bankId: string) {
    return this.fetchApi<{
      bank_id: string;
      config: Record<string, any>;
      overrides: Record<string, any>;
    }>(bankApi(bankId, "/config"), {
      method: "DELETE",
    });
  }

  /**
   * List webhooks for a bank
   */
  async listWebhooks(bankId: string): Promise<{ items: Webhook[] }> {
    return this.fetchApi<{ items: Webhook[] }>(bankApi(bankId, "/webhooks"));
  }

  /**
   * Create a webhook
   */
  async createWebhook(
    bankId: string,
    params: {
      url: string;
      secret?: string;
      event_types?: string[];
      enabled?: boolean;
      http_config?: WebhookHttpConfig;
    }
  ): Promise<Webhook> {
    return this.fetchApi<Webhook>(bankApi(bankId, "/webhooks"), {
      method: "POST",
      body: JSON.stringify(params),
    });
  }

  /**
   * Update a webhook (PATCH — only provided fields are changed)
   */
  async updateWebhook(
    bankId: string,
    webhookId: string,
    params: {
      url?: string;
      secret?: string | null;
      event_types?: string[];
      enabled?: boolean;
      http_config?: WebhookHttpConfig;
    }
  ): Promise<Webhook> {
    return this.fetchApi<Webhook>(bankApi(bankId, `/webhooks/${encodeURIComponent(webhookId)}`), {
      method: "PATCH",
      body: JSON.stringify(params),
    });
  }

  /**
   * Delete a webhook
   */
  async deleteWebhook(bankId: string, webhookId: string): Promise<{ success: boolean }> {
    return this.fetchApi<{ success: boolean }>(
      bankApi(bankId, `/webhooks/${encodeURIComponent(webhookId)}`),
      {
        method: "DELETE",
      }
    );
  }

  /**
   * List webhook deliveries
   */
  async listWebhookDeliveries(
    bankId: string,
    webhookId: string,
    limit?: number,
    cursor?: string
  ): Promise<{ items: WebhookDelivery[]; next_cursor: string | null }> {
    const params = new URLSearchParams();
    if (limit) params.append("limit", limit.toString());
    if (cursor) params.append("cursor", cursor);
    const query = params.toString();
    return this.fetchApi<{ items: WebhookDelivery[]; next_cursor: string | null }>(
      bankApi(
        bankId,
        `/webhooks/${encodeURIComponent(webhookId)}/deliveries${query ? `?${query}` : ""}`
      )
    );
  }

  /**
   * List audit logs for a bank
   */
  async listAuditLogs(
    bankId: string,
    options?: {
      action?: string;
      transport?: string;
      start_date?: string;
      end_date?: string;
      limit?: number;
      offset?: number;
    }
  ): Promise<AuditLogsResponse> {
    const params = new URLSearchParams();
    if (options?.action) params.append("action", options.action);
    if (options?.transport) params.append("transport", options.transport);
    if (options?.start_date) params.append("start_date", options.start_date);
    if (options?.end_date) params.append("end_date", options.end_date);
    if (options?.limit) params.append("limit", options.limit.toString());
    if (options?.offset) params.append("offset", options.offset.toString());
    const query = params.toString();
    return this.fetchApi<AuditLogsResponse>(
      bankApi(bankId, `/audit-logs${query ? `?${query}` : ""}`)
    );
  }

  async getAuditLogStats(
    bankId: string,
    options?: { action?: string; period?: string }
  ): Promise<AuditStatsResponse> {
    const params = new URLSearchParams();
    if (options?.action) params.append("action", options.action);
    if (options?.period) params.append("period", options.period);
    const query = params.toString();
    return this.fetchApi<AuditStatsResponse>(
      bankApi(bankId, `/audit-logs/stats${query ? `?${query}` : ""}`)
    );
  }
}

// Export singleton instance
export const client = new ControlPlaneClient();
