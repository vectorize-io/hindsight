/**
 * Thin wrapper over the Hindsight API.
 *
 * The generated TS SDK (@vectorize-io/hindsight-client) covers createBank / retain /
 * updateBankConfig. Consolidation control, observation clearing and bank stats are not in the
 * SDK, so we call those endpoints directly with fetch.
 */

import { HindsightClient } from "@vectorize-io/hindsight-client";

import type { ProgressFn } from "./types.js";

export interface BankStats {
  pendingConsolidation: number;
  totalObservations: number;
}

/** A fact (world/experience memory) as returned by the list endpoint. */
export interface FactRow {
  id: string;
  text: string;
  factType: string;
  docId: string | null;
}

/**
 * Resolve the Hindsight deployment API key: an explicit value (from the project or a flag)
 * wins, otherwise fall back to the HINDSIGHT_API_KEY env var. Returns undefined when unset
 * (i.e. the deployment is unauthenticated).
 */
export function resolveApiKey(explicit?: string | null): string | undefined {
  return explicit || process.env.HINDSIGHT_API_KEY || undefined;
}

export class SandboxApi {
  private readonly sdk: HindsightClient;
  private readonly baseUrl: string;
  private readonly apiKey?: string;

  constructor(apiUrl: string, apiKey?: string) {
    this.baseUrl = apiUrl.replace(/\/+$/, "");
    this.apiKey = apiKey;
    this.sdk = new HindsightClient({ baseUrl: this.baseUrl, apiKey });
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) h.Authorization = `Bearer ${this.apiKey}`;
    return h;
  }

  private async raw(method: string, pathSuffix: string, body?: unknown): Promise<unknown> {
    const res = await fetch(`${this.baseUrl}${pathSuffix}`, {
      method,
      headers: this.headers(),
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`${method} ${pathSuffix} failed: ${res.status} ${text}`);
    }
    return res.status === 204 ? null : res.json();
  }

  /** Create the bank with the retain + observation missions for this version. */
  async createBank(
    bankId: string,
    opts: { retainMission?: string | null; observationsMission?: string | null } = {}
  ): Promise<void> {
    await this.sdk.createBank(bankId, {
      enableObservations: true,
      retainMission: opts.retainMission || undefined,
      observationsMission: opts.observationsMission || undefined,
    });
  }

  async retain(bankId: string, content: string, documentId: string): Promise<void> {
    await this.sdk.retain(bankId, content, { documentId, updateMode: "replace" });
  }

  /** List facts (world/experience memories), optionally filtered by document or search query. */
  async listFacts(bankId: string, opts: { docId?: string; q?: string } = {}): Promise<FactRow[]> {
    const out: FactRow[] = [];
    const qs = new URLSearchParams({ limit: "200" });
    if (opts.docId) qs.set("document_id", opts.docId);
    if (opts.q) qs.set("q", opts.q);
    let offset = 0;
    for (;;) {
      qs.set("offset", String(offset));
      const page = (await this.raw("GET", `/v1/default/banks/${bankId}/memories/list?${qs}`)) as {
        items?: Array<Record<string, unknown>>;
      };
      const items = page.items ?? [];
      for (const m of items) {
        const factType = String(m.type ?? m.fact_type ?? "");
        if (factType === "observation") continue;
        out.push({
          id: String(m.id ?? ""),
          text: String(m.text ?? ""),
          factType,
          docId: m.document_id != null ? String(m.document_id) : null,
        });
      }
      if (items.length < 200) break;
      offset += items.length;
    }
    return out;
  }

  /** Recall facts for a query (what the bank would retrieve to answer it). */
  async recall(bankId: string, query: string, limit = 10): Promise<FactRow[]> {
    const res = (await this.raw("POST", `/v1/default/banks/${bankId}/memories/recall`, {
      query,
      budget: "mid",
    })) as { results?: Array<Record<string, unknown>> };
    return (res.results ?? []).slice(0, limit).map((m) => ({
      id: String(m.id ?? ""),
      text: String(m.text ?? ""),
      factType: String(m.type ?? m.fact_type ?? ""),
      docId: m.document_id != null ? String(m.document_id) : null,
    }));
  }

  /**
   * Dry-run fact extraction: extract facts from `content` with the given retain mission, WITHOUT
   * persisting (no resolution/links/embeddings). The API chunks internally, so this faithfully
   * reproduces what ingestion would extract — used by Phase 2 to score mission→golden coverage
   * without re-ingesting. The bank only supplies LLM/extraction config; it is not modified.
   */
  async dryRunExtract(
    bankId: string,
    content: string,
    opts: { retainMission?: string | null; extractionMode?: string; chunkSize?: number } = {}
  ): Promise<FactRow[]> {
    const facts = await this.dryRunExtractItems(bankId, content, opts);
    return facts.map((m) => ({
      id: "",
      text: String(m.text ?? ""),
      factType: String(m.fact_type ?? ""),
      docId: null,
    }));
  }

  /** Dry-run extraction returning the extracted-fact items (text, fact_type, dates, entities). */
  async dryRunExtractItems(
    bankId: string,
    content: string,
    opts: { retainMission?: string | null; extractionMode?: string; chunkSize?: number } = {}
  ): Promise<Array<Record<string, unknown>>> {
    const body: Record<string, unknown> = { content };
    if (opts.retainMission != null) body.retain_mission = opts.retainMission;
    if (opts.extractionMode) body.retain_extraction_mode = opts.extractionMode;
    if (opts.chunkSize) body.retain_chunk_size = opts.chunkSize;
    const res = (await this.raw(
      "POST",
      `/v1/default/banks/${bankId}/memories/dry-run-extract`,
      body
    )) as {
      facts?: Array<Record<string, unknown>>;
    };
    return res.facts ?? [];
  }

  /** Fetch a single memory's text (for recording the before-state of a curation). */
  async getMemoryText(bankId: string, memoryId: string): Promise<string> {
    const m = (await this.raw("GET", `/v1/default/banks/${bankId}/memories/${memoryId}`)) as {
      text?: string;
    };
    return m?.text ?? "";
  }

  /** Curate a single memory: edit text / invalidate / revert (PATCH, in place — no re-ingest). */
  async updateMemory(
    bankId: string,
    memoryId: string,
    body: { text?: string; state?: "valid" | "invalidated"; reason?: string }
  ): Promise<void> {
    await this.raw("PATCH", `/v1/default/banks/${bankId}/memories/${memoryId}`, body);
  }

  async updateObservationsMission(bankId: string, mission: string | null): Promise<void> {
    await this.sdk.updateBankConfig(bankId, { observationsMission: mission || undefined });
  }

  async triggerConsolidation(bankId: string): Promise<void> {
    await this.raw("POST", `/v1/default/banks/${bankId}/consolidate`);
  }

  async clearObservations(bankId: string): Promise<number> {
    const res = (await this.raw("DELETE", `/v1/default/banks/${bankId}/observations`)) as {
      deleted_count?: number;
    } | null;
    return res?.deleted_count ?? 0;
  }

  async getStats(bankId: string): Promise<BankStats> {
    const res = (await this.raw("GET", `/v1/default/banks/${bankId}/stats`)) as {
      pending_consolidation?: number;
      total_observations?: number;
    };
    return {
      pendingConsolidation: res.pending_consolidation ?? 0,
      totalObservations: res.total_observations ?? 0,
    };
  }

  /** Poll bank stats until no consolidation remains pending. */
  async waitForConsolidation(
    bankId: string,
    opts: { timeoutMs?: number; pollMs?: number; onProgress?: ProgressFn } = {}
  ): Promise<void> {
    const timeoutMs = opts.timeoutMs ?? 600_000;
    const pollMs = opts.pollMs ?? 3_000;
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      const stats = await this.getStats(bankId);
      if (stats.pendingConsolidation === 0) return;
      opts.onProgress?.(`Waiting for consolidation… (${stats.pendingConsolidation} pending)`);
      await new Promise((r) => setTimeout(r, pollMs));
    }
    throw new Error(`Consolidation did not complete within ${Math.round(timeoutMs / 1000)}s`);
  }
}
