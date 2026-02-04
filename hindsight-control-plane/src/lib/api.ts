/**
 * Client for calling Control Plane API routes (which proxy to the dataplane via SDK)
 * This should be used in client components, not the SDK directly
 */

export interface MentalModel {
  id: string;
  bank_id: string;
  name: string;
  source_query: string;
  content: string;
  tags: string[];
  max_tokens: number;
  trigger: { refresh_after_consolidation: boolean };
  last_refreshed_at: string;
  created_at: string;
  reflect_response?: any;
}

export class ControlPlaneClient {
  private async fetchApi<T>(path: string, options?: RequestInit): Promise<T> {
    const response = await fetch(path, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API Error: ${response.status} - ${error}`);
    }

    return response.json();
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
      observations?: { max_results?: number } | null;
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
      metadata?: Record<string, string>;
      document_id?: string;
      entities?: Array<{ text: string; type?: string }>;
    }>;
    document_id?: string;
    async?: boolean;
  }) {
    const endpoint = params.async ? "/api/memories/retain_async" : "/api/memories/retain";
    return this.fetchApi(endpoint, {
      method: "POST",
      body: JSON.stringify(params),
    });
  }

  /**
   * Get bank statistics
   */
  async getBankStats(bankId: string) {
    return this.fetchApi(`/api/stats/${bankId}`);
  }

  /**
   * Get graph data
   */
  async getGraph(params: { bank_id: string; type?: string; limit?: number }) {
    const queryParams = new URLSearchParams();
    queryParams.append("bank_id", params.bank_id);
    if (params.type) queryParams.append("type", params.type);
    if (params.limit) queryParams.append("limit", params.limit.toString());
    return this.fetchApi(`/api/graph?${queryParams}`);
  }

  /**
   * List operations with optional filtering and pagination
   */
  async listOperations(
    bankId: string,
    options?: { status?: string; limit?: number; offset?: number }
  ) {
    const params = new URLSearchParams();
    if (options?.status) params.append("status", options.status);
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
    }>(`/api/operations/${bankId}${query ? `?${query}` : ""}`);
  }

  /**
   * Cancel a pending operation
   */
  async cancelOperation(bankId: string, operationId: string) {
    return this.fetchApi<{
      success: boolean;
      message: string;
      operation_id: string;
    }>(`/api/operations/${bankId}?operation_id=${operationId}`, {
      method: "DELETE",
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
   * Get entity details
   */
  async getEntity(entityId: string, bankId: string) {
    return this.fetchApi(`/api/entities/${entityId}?bank_id=${bankId}`);
  }

  /**
   * Regenerate entity observations
   */
  async regenerateEntityObservations(entityId: string, bankId: string) {
    return this.fetchApi(`/api/entities/${entityId}/regenerate?bank_id=${bankId}`, {
      method: "POST",
    });
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
    return this.fetchApi(`/api/documents/${documentId}?bank_id=${bankId}`);
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
    }>(`/api/documents/${documentId}?bank_id=${bankId}`, {
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
    }>(`/api/banks/${bankId}`, {
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
    }>(`/api/banks/${bankId}/observations`, {
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
    }>(`/api/banks/${bankId}/consolidate`, {
      method: "POST",
    });
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
    }>(`/api/memories/${memoryId}?bank_id=${bankId}`);
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
    }>(`/api/profile/${bankId}`);
  }

  /**
   * Set bank mission
   */
  async setBankMission(bankId: string, mission: string) {
    return this.fetchApi(`/api/banks/${bankId}`, {
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
    }>(`/api/banks/${bankId}/directives${query ? `?${query}` : ""}`);
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
    }>(`/api/banks/${bankId}/directives`, {
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
    }>(`/api/banks/${bankId}/directives/${directiveId}`);
  }

  /**
   * Delete a directive
   */
  async deleteDirective(bankId: string, directiveId: string) {
    return this.fetchApi(`/api/banks/${bankId}/directives/${directiveId}`, {
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
    }>(`/api/banks/${bankId}/directives/${directiveId}`, {
      method: "PATCH",
      body: JSON.stringify(params),
    });
  }

  /**
   * Get operation status
   */
  async getOperationStatus(bankId: string, operationId: string) {
    return this.fetchApi<{
      operation_id: string;
      status: "pending" | "completed" | "failed" | "not_found";
      operation_type: string | null;
      created_at: string | null;
      updated_at: string | null;
      completed_at: string | null;
      error_message: string | null;
    }>(`/api/banks/${bankId}/operations/${operationId}`);
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
    return this.fetchApi(`/api/profile/${bankId}`, {
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
    }>(`/api/banks/${bankId}/observations${query ? `?${query}` : ""}`);
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
    }>(`/api/banks/${bankId}/observations/${observationId}`);
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
        trigger: { refresh_after_consolidation: boolean };
        last_refreshed_at: string;
        created_at: string;
        reflect_response?: {
          text: string;
          based_on: Record<string, Array<{ id: string; text: string; type: string }>>;
        };
      }>;
    }>(`/api/banks/${bankId}/mental-models${query ? `?${query}` : ""}`);
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
      trigger?: { refresh_after_consolidation: boolean };
    }
  ) {
    return this.fetchApi<{
      operation_id: string;
    }>(`/api/banks/${bankId}/mental-models`, {
      method: "POST",
      body: JSON.stringify(params),
    });
  }

  /**
   * Get a mental model
   */
  async getMentalModel(bankId: string, mentalModelId: string): Promise<MentalModel> {
    return this.fetchApi<MentalModel>(`/api/banks/${bankId}/mental-models/${mentalModelId}`);
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
      trigger?: { refresh_after_consolidation: boolean };
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
      trigger: { refresh_after_consolidation: boolean };
      last_refreshed_at: string;
      created_at: string;
      reflect_response?: {
        text: string;
        based_on: Record<string, Array<{ id: string; text: string; type: string }>>;
      };
    }>(`/api/banks/${bankId}/mental-models/${mentalModelId}`, {
      method: "PATCH",
      body: JSON.stringify(params),
    });
  }

  /**
   * Delete a mental model
   */
  async deleteMentalModel(bankId: string, mentalModelId: string) {
    return this.fetchApi(`/api/banks/${bankId}/mental-models/${mentalModelId}`, {
      method: "DELETE",
    });
  }

  /**
   * Refresh a mental model (re-run source query) - async operation
   */
  async refreshMentalModel(bankId: string, mentalModelId: string) {
    return this.fetchApi<{
      operation_id: string;
    }>(`/api/banks/${bankId}/mental-models/${mentalModelId}/refresh`, {
      method: "POST",
    });
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
      };
    }>("/api/version");
  }
}

// Export singleton instance
export const client = new ControlPlaneClient();
