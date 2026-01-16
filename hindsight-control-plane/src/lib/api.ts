/**
 * Client for calling Control Plane API routes (which proxy to the dataplane via SDK)
 * This should be used in client components, not the SDK directly
 */

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
   * List operations
   */
  async listOperations(bankId: string) {
    return this.fetchApi(`/api/operations/${bankId}`);
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
   * List mental models for a bank
   */
  async listMentalModels(bankId: string) {
    return this.fetchApi<{
      items: Array<{
        id: string;
        bank_id: string;
        subtype: string;
        name: string;
        description: string;
        observations?: Array<{ title: string; text: string; based_on: string[] }>;
        entity_id: string | null;
        links: string[];
        tags?: string[];
        last_updated: string | null;
        created_at: string;
      }>;
    }>(`/api/banks/${bankId}/mental-models`);
  }

  /**
   * Refresh mental models for a bank (async)
   * @param subtype - Optional subtype to refresh. If not specified, refreshes all.
   */
  async refreshMentalModels(
    bankId: string,
    subtype?: "structural" | "emergent" | "pinned" | "learned"
  ) {
    return this.fetchApi<{ operation_id: string; message: string }>(
      `/api/banks/${bankId}/mental-models/refresh`,
      {
        method: "POST",
        body: JSON.stringify(subtype ? { subtype } : {}),
      }
    );
  }

  /**
   * Delete a mental model
   */
  async deleteMentalModel(bankId: string, modelId: string) {
    return this.fetchApi(`/api/banks/${bankId}/mental-models/${modelId}`, {
      method: "DELETE",
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
   * Generate/refresh content for a specific mental model (async)
   */
  async generateMentalModel(bankId: string, modelId: string) {
    return this.fetchApi<{ operation_id: string; message: string }>(
      `/api/banks/${bankId}/mental-models/${modelId}/generate`,
      {
        method: "POST",
      }
    );
  }

  /**
   * Create a pinned mental model
   */
  async createMentalModel(
    bankId: string,
    params: {
      name: string;
      description: string;
      tags?: string[];
    }
  ) {
    return this.fetchApi<{
      id: string;
      bank_id: string;
      subtype: string;
      name: string;
      description: string;
      observations?: Array<{ title: string; text: string; based_on: string[] }>;
      entity_id: string | null;
      links: string[];
      tags?: string[];
      last_updated: string | null;
      created_at: string;
    }>(`/api/banks/${bankId}/mental-models`, {
      method: "POST",
      body: JSON.stringify(params),
    });
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
}

// Export singleton instance
export const client = new ControlPlaneClient();
