/**
 * Lightweight fetch-based HTTP client for the Hindsight API.
 *
 * Zero npm dependencies. Covers retain, recall, bank management, and health
 * checks. Uses the same REST endpoints as the official Python/Node SDKs.
 */

export interface RetainOptions {
  bankId: string;
  content: string;
  documentId?: string;
  context?: string;
  timestamp?: string;
  metadata?: Record<string, string>;
  tags?: string[];
  async?: boolean;
}

export interface RecallOptions {
  bankId: string;
  query: string;
  budget?: "low" | "mid" | "high";
  maxTokens?: number;
  types?: string[];
  tags?: string[];
}

export interface RecallResult {
  id: string;
  text: string;
  type: string;
  context?: string;
  metadata?: Record<string, string>;
  tags?: string[];
  entities?: string[];
  mentioned_at?: string;
}

export interface RecallResponse {
  results: RecallResult[];
}

export interface BankConfig {
  bankId: string;
  name?: string;
  mission?: string;
  retainMission?: string;
  disposition?: {
    skepticism?: number;
    literalism?: number;
    empathy?: number;
  };
}

export class HindsightClient {
  private baseUrl: string;
  private token: string | null;
  private namespace: string;

  constructor(baseUrl: string, token: string | null = null, namespace = "default") {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.token = token;
    this.namespace = namespace;
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.token) h["Authorization"] = `Bearer ${this.token}`;
    return h;
  }

  private url(path: string): string {
    return `${this.baseUrl}/v1/${this.namespace}${path}`;
  }

  async healthy(timeoutMs = 3000): Promise<boolean> {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      const resp = await fetch(`${this.baseUrl}/health`, {
        signal: controller.signal,
      });
      clearTimeout(timer);
      return resp.ok;
    } catch {
      return false;
    }
  }

  async retain(options: RetainOptions, timeoutMs = 15000): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = {
      items: [
        {
          content: options.content,
          ...(options.context && { context: options.context }),
          ...(options.timestamp && { timestamp: options.timestamp }),
          ...(options.metadata && { metadata: options.metadata }),
          ...(options.tags && { tags: options.tags }),
        },
      ],
      ...(options.documentId && { document_id: options.documentId }),
      ...(options.async && { async: true }),
    };

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const resp = await fetch(this.url(`/banks/${options.bankId}/retain`), {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    clearTimeout(timer);

    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`Retain failed (${resp.status}): ${text}`);
    }
    return resp.json() as Promise<Record<string, unknown>>;
  }

  async recall(options: RecallOptions, timeoutMs = 12000): Promise<RecallResponse> {
    const body: Record<string, unknown> = {
      query: options.query,
      ...(options.budget && { budget: options.budget }),
      ...(options.maxTokens && { max_tokens: options.maxTokens }),
      ...(options.types && { types: options.types }),
      ...(options.tags && { tags: options.tags }),
    };

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const resp = await fetch(this.url(`/banks/${options.bankId}/recall`), {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    clearTimeout(timer);

    if (!resp.ok) {
      const text = await resp.text().catch(() => "");
      throw new Error(`Recall failed (${resp.status}): ${text}`);
    }
    return resp.json() as Promise<RecallResponse>;
  }

  async createOrUpdateBank(config: BankConfig): Promise<void> {
    const body: Record<string, unknown> = {
      ...(config.name && { name: config.name }),
      ...(config.mission && { mission: config.mission }),
      ...(config.retainMission && { config: { retain_mission: config.retainMission } }),
      ...(config.disposition && { disposition: config.disposition }),
    };

    const resp = await fetch(this.url(`/banks/${config.bankId}`), {
      method: "PUT",
      headers: this.headers(),
      body: JSON.stringify(body),
    });

    if (!resp.ok && resp.status !== 409) {
      const text = await resp.text().catch(() => "");
      throw new Error(`Bank create/update failed (${resp.status}): ${text}`);
    }
  }

  async bankExists(bankId: string): Promise<boolean> {
    try {
      const resp = await fetch(this.url(`/banks/${bankId}`), {
        method: "GET",
        headers: this.headers(),
      });
      return resp.ok;
    } catch {
      return false;
    }
  }
}
