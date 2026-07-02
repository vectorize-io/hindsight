/**
 * Harness-agnostic Hindsight HTTP client (raw fetch, no SDK dep).
 *
 * Every harness adapter and the backfill CLI go through this one client: it configures the bank
 * (missions + git/chat retain strategies), retains memories, reflects, drains async operations, and
 * creates knowledge pages. Nothing here knows about opencode/claude-code/etc.
 */
import {
  GIT_MISSION, REFLECT_MISSION, OBSERVATIONS_MISSION, RETAIN_STRATEGIES, PAGES,
} from "./missions";
import { sleep } from "./util";

export interface ClientOpts {
  apiUrl: string;
  apiToken?: string;
  bank: string;
  log?: (msg: string) => void;
}

export interface RetainOpts {
  timestamp?: string;                        // when the content occurred (temporal ranking)
  metadata?: Record<string, string>;         // source provenance (returned with recalls)
  async?: boolean;                           // enqueue server-side (default) vs block on extraction
}

const TERMINAL = new Set(["completed", "failed", "cancelled", "error"]);

export class HindsightClient {
  readonly apiUrl: string;
  readonly apiToken?: string;
  readonly bank: string;
  readonly opIds: string[] = [];             // async operation ids collected by retain(), for drain()
  private readonly log: (msg: string) => void;

  constructor(o: ClientOpts) {
    this.apiUrl = o.apiUrl.replace(/\/$/, "");
    this.apiToken = o.apiToken;
    this.bank = o.bank;
    this.log = o.log ?? (() => {});
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiToken) h["Authorization"] = `Bearer ${this.apiToken}`;
    return h;
  }

  bankUrl(suffix = ""): string {
    return `${this.apiUrl}/v1/default/banks/${encodeURIComponent(this.bank)}${suffix}`;
  }

  async req(method: string, url: string, body?: unknown): Promise<Response> {
    const r = await fetch(url, {
      method, headers: this.headers(), body: body ? JSON.stringify(body) : undefined,
    });
    if (!r.ok && r.status !== 404) throw new Error(`${method} ${url} -> ${r.status} ${await r.text()}`);
    return r;
  }

  /** Reflect: synthesized, root-cause answer over the bank. Bounded so a slow server never hangs a caller. */
  async reflect(query: string, opts: { budget?: string; timeoutMs?: number } = {}): Promise<string> {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), opts.timeoutMs ?? 120000);
    try {
      const resp = await fetch(this.bankUrl("/reflect"), {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify({ query, budget: opts.budget ?? "high" }),
        signal: ctrl.signal,
      });
      if (!resp.ok) throw new Error(`reflect ${resp.status}`);
      const data = (await resp.json()) as { text?: string };
      return (data.text || "").trim();
    } finally {
      clearTimeout(timer);
    }
  }

  /** Retain one memory. Async by default: enqueue extraction server-side and collect its op-id for drain(). */
  async retain(
    content: string, context: string, documentId: string, tags: string[], strategy: string,
    opts: RetainOpts = {},
  ): Promise<void> {
    const item: Record<string, unknown> = { content, context, document_id: documentId, tags, strategy };
    if (opts.timestamp) item.timestamp = opts.timestamp;
    if (opts.metadata) item.metadata = opts.metadata;
    const isAsync = opts.async !== false;
    const r = await this.req("POST", this.bankUrl("/memories"), { items: [item], async: isAsync });
    if (isAsync) {
      try {
        const j = (await r.json()) as { operation_id?: string };
        if (j.operation_id) this.opIds.push(j.operation_id);
      } catch { /* ignore */ }
    }
  }

  /** Configure the bank: reflect mission, observations ON, and the named git/chat retain strategies. */
  async configureBank(opts: { reset?: boolean } = {}): Promise<void> {
    if (opts.reset) {
      await this.req("DELETE", this.bankUrl());
      this.log(`[bank] reset ${this.bank}`);
    }
    await this.req("PUT", this.bankUrl(), {
      name: this.bank,
      reflect_mission: REFLECT_MISSION,
      enable_observations: true,
      observations_mission: OBSERVATIONS_MISSION,
      retain_mission: GIT_MISSION,
      retain_extraction_mode: "verbose",
    });
    await this.req("PATCH", this.bankUrl("/config"), {
      updates: { retain_strategies: RETAIN_STRATEGIES, retain_default_strategy: "git" },
    });
    this.log(`[bank] configured ${this.bank}: reflect mission, observations ON, strategies {git, chat}`);
  }

  /** Poll each enqueued operation by id until terminal. LIST only shows active ops, so per-id GET is reliable. */
  async drain(ids: string[], label: string, maxMs = 60 * 60 * 1000): Promise<void> {
    if (!ids.length) return;
    this.log(`[wait] draining ${ids.length} ${label} operations …`);
    const start = Date.now();
    const pending = new Set(ids);
    let failed = 0;
    while (pending.size && Date.now() - start < maxMs) {
      await Promise.all([...pending].map(async (id) => {
        try {
          const r = await fetch(this.bankUrl(`/operations/${id}`), { headers: this.headers() });
          if (!r.ok) return;
          const st = (((await r.json()) as { status?: string }).status || "").toLowerCase();
          if (TERMINAL.has(st)) { pending.delete(id); if (st !== "completed") failed++; }
        } catch { /* transient — retry next cycle */ }
      }));
      if (pending.size) { this.log(`  … ${pending.size}/${ids.length} ${label} ops pending`); await sleep(5000); }
    }
    this.log(`[wait] ${label} drained — ${ids.length - pending.size} done, ${failed} failed`
      + (pending.size ? `, ${pending.size} still pending at timeout` : ""));
  }

  /** Knowledge pages — synthesized from the EXTRACTED facts, so call AFTER the retain drain. */
  async createPages(): Promise<void> {
    this.log(`[pages] creating ${PAGES.length} knowledge pages …`);
    const pageOps: string[] = [];
    for (const p of PAGES) {
      try {
        // fact_types = ALL (world+experience+observation) so a page draws from raw facts AND
        // consolidated observations; refresh after consolidation keeps it a living document.
        const body = { ...p, trigger: { fact_types: ["world", "experience", "observation"],
                                        refresh_after_consolidation: true } };
        const r = await this.req("POST", this.bankUrl("/knowledge-base/pages"), body);
        const j = (await r.json()) as { operation_id?: string; page_id?: string };
        if (j.operation_id) pageOps.push(j.operation_id);
        this.log(`  created page '${p.name}' -> ${j.page_id || "?"}`);
      } catch (e) {
        this.log(`  ! page '${p.name}' failed: ${(e as Error).message?.slice(0, 140)}`);
      }
    }
    await this.drain(pageOps, "page-generation");
  }
}
