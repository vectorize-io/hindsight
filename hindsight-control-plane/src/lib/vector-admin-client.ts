const VECTOR_ADMIN_URL = process.env.NEXT_PUBLIC_VECTOR_ADMIN_URL || "http://localhost:3001";

export type ConnectionStatus = "connected" | "disconnected" | "error";

export interface VectorDBInfo {
  id: string;
  name: string;
  type: string;
  status: ConnectionStatus;
  stats?: {
    documents: number;
    namespaces: number;
    totalVectors: number;
  };
}

export interface CollectionInfo {
  name: string;
  vectorCount: number;
  dimension: number;
  metric: string;
  status: string;
}

export interface DocumentInfo {
  id: string;
  metadata: Record<string, unknown>;
  vectorCount: number;
  createdAt: string;
}

export class VectorAdminClient {
  private baseUrl: string;

  constructor(baseUrl: string = VECTOR_ADMIN_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetch(`${this.baseUrl}/api${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...init?.headers,
      },
    });
    if (!res.ok) {
      throw new Error(`VectorAdmin API error: ${res.status} ${res.statusText}`);
    }
    return res.json();
  }

  async getConnections(): Promise<VectorDBInfo[]> {
    return this.request<VectorDBInfo[]>("/connections");
  }

  async testConnection(id: string): Promise<{ status: ConnectionStatus; message: string }> {
    return this.request(`/connections/${id}/test`, { method: "POST" });
  }

  async getCollections(dbId: string): Promise<CollectionInfo[]> {
    return this.request<CollectionInfo[]>(`/connections/${dbId}/collections`);
  }

  async getDocuments(
    dbId: string,
    collection: string,
    limit = 50
  ): Promise<DocumentInfo[]> {
    return this.request<DocumentInfo[]>(
      `/connections/${dbId}/collections/${collection}/documents?limit=${limit}`
    );
  }

  async health(): Promise<{ status: string; version: string }> {
    return this.request("/health");
  }

  async getStats(): Promise<{
    totalConnections: number;
    connectedCount: number;
    totalDocuments: number;
    totalVectors: number;
  }> {
    return this.request("/stats");
  }
}

export const vectorAdminClient = new VectorAdminClient();
