/**
 * Minimal Hindsight HTTP client for the Obsidian plugin.
 *
 * Uses Obsidian's `requestUrl` (NOT `fetch`) so requests run outside the
 * renderer's CORS sandbox. No external dependencies.
 */

import { requestUrl, type RequestUrlParam, type RequestUrlResponse } from "obsidian";
import type { ReflectOptions, ReflectResponse, RetainOptions } from "./types";

/**
 * Encode a vault-relative document id for use in a URL `:path` segment:
 * encode each path segment but preserve the `/` separators the server's
 * `{document_id:path}` route expects.
 */
function encodeDocPath(documentId: string): string {
  return documentId.split("/").map(encodeURIComponent).join("/");
}

export class HindsightClient {
  private readonly baseUrl: string;
  private readonly token: string | undefined;

  constructor(baseUrl: string, token?: string) {
    const url = (baseUrl ?? "").trim();
    if (!url) throw new Error("Hindsight API URL is required");
    this.baseUrl = url.replace(/\/+$/, "");
    this.token = token?.trim() || undefined;
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.token) h["Authorization"] = `Bearer ${this.token}`;
    return h;
  }

  private bankUrl(bankId: string, suffix: string): string {
    return `${this.baseUrl}/v1/default/banks/${encodeURIComponent(bankId)}${suffix}`;
  }

  private async send(method: string, url: string, body?: unknown): Promise<RequestUrlResponse> {
    const params: RequestUrlParam = {
      url,
      method,
      headers: this.headers(),
      // Handle non-2xx ourselves so we can surface a useful message.
      throw: false,
    };
    if (body !== undefined) params.body = JSON.stringify(body);

    const resp = await requestUrl(params);
    if (resp.status < 200 || resp.status >= 300) {
      const detail = (resp.text ?? "").slice(0, 500);
      throw new Error(`Hindsight ${method} ${url} → HTTP ${resp.status}: ${detail}`);
    }
    return resp;
  }

  /** Upsert a note into the bank. Re-retaining the same `documentId` replaces it. */
  async retain(
    bankId: string,
    documentId: string,
    content: string,
    options: RetainOptions = {}
  ): Promise<void> {
    const item: Record<string, unknown> = {
      content,
      document_id: documentId,
      context: options.context ?? "obsidian",
      update_mode: options.updateMode ?? "replace",
    };
    if (options.tags?.length) item.tags = options.tags;
    if (options.metadata && Object.keys(options.metadata).length) item.metadata = options.metadata;
    if (options.timestamp) item.timestamp = options.timestamp;

    await this.send("POST", this.bankUrl(bankId, "/memories"), { items: [item], async: true });
  }

  /** Delete a document and cascade to its memory units. */
  async deleteDocument(bankId: string, documentId: string): Promise<void> {
    await this.send("DELETE", this.bankUrl(bankId, `/documents/${encodeDocPath(documentId)}`));
  }

  async reflect(
    bankId: string,
    query: string,
    options: ReflectOptions = {}
  ): Promise<ReflectResponse> {
    const body: Record<string, unknown> = {
      query,
      budget: options.budget ?? "low",
    };
    if (options.includeCitations) {
      // `facts: {}` makes the server return `based_on`; `tool_calls: {}` returns the trace.
      body.include = { facts: {}, tool_calls: {} };
    }
    // tag_groups and tags are mutually exclusive server-side; prefer tag_groups.
    if (options.tagGroups?.length) body.tag_groups = options.tagGroups;
    else if (options.tags?.length) body.tags = options.tags;
    const resp = await this.send("POST", this.bankUrl(bankId, "/reflect"), body);
    return resp.json as ReflectResponse;
  }

  /** Lightweight reachability check for the settings "Test connection" button. */
  async health(): Promise<boolean> {
    try {
      const resp = await requestUrl({
        url: `${this.baseUrl}/health`,
        method: "GET",
        headers: this.headers(),
        throw: false,
      });
      return resp.status >= 200 && resp.status < 300;
    } catch {
      return false;
    }
  }
}
