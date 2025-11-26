/**
 * API client for the control plane API (which proxies to the dataplane)
 */

export class DataplaneClient {
  private async fetchApi<T>(
    path: string,
    options?: RequestInit
  ): Promise<T> {
    // Call the control plane API routes, not the dataplane directly
    const response = await fetch(path, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
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
   * Search memory using semantic similarity
   */
  async search(params: {
    query: string;
    fact_type: ('world' | 'agent' | 'opinion')[];
    agent_id: string;
    thinking_budget?: number;
    max_tokens?: number;
    trace?: boolean;
  }) {
    return this.fetchApi(`/api/search`, {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  /**
   * Think and generate answer
   */
  async think(params: {
    query: string;
    agent_id: string;
    thinking_budget?: number;
    context?: string;
  }) {
    return this.fetchApi(`/api/think`, {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  /**
   * Store multiple memories in batch
   */
  async batchPut(params: {
    agent_id: string;
    items: Array<{
      content: string;
      event_date?: string;
      context?: string;
    }>;
    document_id?: string;
  }) {
    return this.fetchApi(`/api/memories/batch`, {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  /**
   * Store multiple memories asynchronously
   * Note: If document_id is provided and already exists, the document will be automatically replaced (upsert behavior).
   */
  async batchPutAsync(params: {
    agent_id: string;
    items: Array<{
      content: string;
      event_date?: string;
      context?: string;
    }>;
    document_id?: string;
  }) {
    return this.fetchApi(`/api/memories/batch_async`, {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  /**
   * List all agents
   */
  async listAgents() {
    return this.fetchApi<{ agents: any[] }>('/api/agents', { cache: 'no-store' });
  }

  /**
   * Get agent statistics
   */
  async getAgentStats(agentId: string) {
    return this.fetchApi(`/api/stats/${agentId}`);
  }

  /**
   * Get graph data for visualization
   */
  async getGraphData(params: {
    agent_id: string;
    fact_type?: string;
  }) {
    const queryParams = new URLSearchParams();
    queryParams.append('agent_id', params.agent_id);
    if (params.fact_type) queryParams.append('fact_type', params.fact_type);

    return this.fetchApi(`/api/graph?${queryParams}`);
  }

  /**
   * List memory units
   */
  async listMemoryUnits(params: {
    agent_id: string;
    fact_type?: string;
    q?: string;
    limit?: number;
    offset?: number;
  }) {
    const queryParams = new URLSearchParams();
    queryParams.append('agent_id', params.agent_id);
    if (params.fact_type) queryParams.append('fact_type', params.fact_type);
    if (params.q) queryParams.append('q', params.q);
    if (params.limit) queryParams.append('limit', params.limit.toString());
    if (params.offset) queryParams.append('offset', params.offset.toString());

    return this.fetchApi(`/api/list?${queryParams}`);
  }

  /**
   * List documents
   */
  async listDocuments(params: {
    agent_id: string;
    q?: string;
    limit?: number;
    offset?: number;
  }) {
    const queryParams = new URLSearchParams();
    queryParams.append('agent_id', params.agent_id);
    if (params.q) queryParams.append('q', params.q);
    if (params.limit) queryParams.append('limit', params.limit.toString());
    if (params.offset) queryParams.append('offset', params.offset.toString());

    return this.fetchApi(`/api/documents?${queryParams}`);
  }

  /**
   * Get document by ID
   */
  async getDocument(documentId: string, agentId: string) {
    return this.fetchApi(`/api/documents/${documentId}?agent_id=${agentId}`);
  }

  /**
   * List async operations for an agent
   */
  async listOperations(agentId: string) {
    return this.fetchApi(`/api/operations/${agentId}`);
  }

  /**
   * Cancel a pending async operation
   */
  async cancelOperation(agentId: string, operationId: string) {
    return this.fetchApi(`/api/operations/${agentId}?operation_id=${operationId}`, {
      method: 'DELETE',
    });
  }

  /**
   * Delete a memory unit
   */
  async deleteMemoryUnit(agentId: string, unitId: string) {
    return this.fetchApi(`/api/list?agent_id=${agentId}&unit_id=${unitId}`, {
      method: 'DELETE',
    });
  }

  /**
   * List entities for an agent
   */
  async listEntities(params: {
    agent_id: string;
    limit?: number;
  }) {
    const queryParams = new URLSearchParams();
    queryParams.append('agent_id', params.agent_id);
    if (params.limit) queryParams.append('limit', params.limit.toString());

    return this.fetchApi<{
      entities: Array<{
        id: string;
        canonical_name: string;
        mention_count: number;
        first_seen?: string;
        last_seen?: string;
        metadata?: Record<string, any>;
      }>;
    }>(`/api/entities?${queryParams}`);
  }

  /**
   * Get entity details with observations
   */
  async getEntity(entityId: string, agentId: string) {
    return this.fetchApi<{
      id: string;
      canonical_name: string;
      mention_count: number;
      first_seen?: string;
      last_seen?: string;
      metadata?: Record<string, any>;
      observations: Array<{
        text: string;
        mentioned_at?: string;
      }>;
    }>(`/api/entities/${entityId}?agent_id=${agentId}`);
  }

  /**
   * Regenerate observations for an entity
   */
  async regenerateEntityObservations(entityId: string, agentId: string) {
    return this.fetchApi(`/api/entities/${entityId}/regenerate?agent_id=${agentId}`, {
      method: 'POST',
    });
  }
}

// Export a singleton instance
export const dataplaneClient = new DataplaneClient();

/**
 * Server-side dataplane client that calls the dataplane directly
 * Only use this on the server side (in API routes)
 */
export class ServerDataplaneClient {
  private baseUrl: string;

  constructor() {
    this.baseUrl = process.env.HINDSIGHT_CP_DATAPLANE_API_URL || 'http://localhost:8888';
  }

  async fetchDataplane<T>(
    path: string,
    options?: RequestInit
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;

    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Dataplane Error: ${response.status} - ${error}`);
    }

    return response.json();
  }
}

export const serverDataplaneClient = new ServerDataplaneClient();
