/**
 * HTTP client for the Hindsight REST API.
 *
 * Uses native fetch (Node 20+). No external dependencies.
 */

import type { PaperclipMemoryConfig } from './config.js';

export interface Memory {
  text: string;
  type?: string;
  mentionedAt?: string;
}

export interface RecallResponse {
  results: Memory[];
}

export interface RetainResponse {
  success: boolean;
  bankId?: string;
}

export class HindsightClient {
  private readonly baseUrl: string;
  private readonly token: string | undefined;
  private readonly timeoutMs: number;

  constructor(config: PaperclipMemoryConfig) {
    const url = config.hindsightApiUrl.trim();
    if (!url) throw new Error('hindsightApiUrl is required');
    this.baseUrl = url.replace(/\/$/, '');
    this.token = config.hindsightApiToken;
    this.timeoutMs = config.timeoutMs ?? 15_000;
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { 'Content-Type': 'application/json' };
    if (this.token) h['Authorization'] = `Bearer ${this.token}`;
    return h;
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
    timeoutMs?: number,
  ): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs ?? this.timeoutMs);

    try {
      const resp = await fetch(`${this.baseUrl}${path}`, {
        method,
        headers: this.headers(),
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      if (!resp.ok) {
        const text = await resp.text().catch(() => '');
        throw new Error(`HTTP ${resp.status} from ${path}: ${text}`);
      }

      return (await resp.json()) as T;
    } finally {
      clearTimeout(timer);
    }
  }

  async recall(
    bankId: string,
    query: string,
    options?: { budget?: string; maxTokens?: number },
  ): Promise<RecallResponse> {
    const path = `/v1/default/banks/${encodeURIComponent(bankId)}/memories/recall`;
    return this.request<RecallResponse>('POST', path, {
      query,
      budget: options?.budget ?? 'mid',
      max_tokens: options?.maxTokens ?? 1024,
    });
  }

  async retain(
    bankId: string,
    content: string,
    options?: {
      documentId?: string;
      context?: string;
      metadata?: Record<string, string>;
      tags?: string[];
    },
  ): Promise<RetainResponse> {
    const path = `/v1/default/banks/${encodeURIComponent(bankId)}/memories`;
    const item: Record<string, unknown> = { content };
    if (options?.documentId) item['document_id'] = options.documentId;
    if (options?.context) item['context'] = options.context;
    if (options?.metadata) item['metadata'] = options.metadata;
    if (options?.tags) item['tags'] = options.tags;
    return this.request<RetainResponse>('POST', path, { items: [item], async: true });
  }

  async setBankMission(bankId: string, mission: string, retainMission?: string): Promise<void> {
    const path = `/v1/default/banks/${encodeURIComponent(bankId)}/config`;
    const updates: Record<string, string> = { reflect_mission: mission };
    if (retainMission) updates['retain_mission'] = retainMission;
    await this.request('PATCH', path, { updates });
  }

  async health(): Promise<boolean> {
    try {
      const resp = await fetch(`${this.baseUrl}/health`, {
        headers: this.headers(),
        signal: AbortSignal.timeout(5_000),
      });
      return resp.ok;
    } catch {
      return false;
    }
  }
}
