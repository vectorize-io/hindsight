/**
 * Harness-agnostic Hindsight HTTP client (raw fetch, no SDK dep).
 *
 * Every harness adapter and the backfill CLI go through this one client: it configures the bank
 * (missions + git/chat retain strategies), retains memories, reflects, drains async operations, and
 * creates knowledge pages. Nothing here knows about opencode/claude-code/etc.
 */
import {
  buildPageTrigger,
  CODING_BANK_STRUCTURE,
  CODING_BANK_TEMPLATE,
  PAGE_MAX_TOKENS,
  pagesFor,
  type PageTrigger,
} from "./missions";
import { pool, semverGte, sleep } from "./util";
import type { RetainStamp } from "./retain-stamp";

/** One node of GET /knowledge-base/tree. Only the fields this client reads. */
export interface KnowledgeNode {
  id: string;
  kind: "folder" | "page";
  name: string;
  /** The page's source query (OKF `description`) — what a re-sync compares against. */
  description?: string;
  children?: KnowledgeNode[];
}

/**
 * How consolidation scopes the observations a retained memory feeds (`observation_scopes` on the
 * retain API). The scalar modes are the server's; a `string[][]` declares the scopes explicitly.
 */
export type ObservationScopes = "shared" | "combined" | "per_tag" | "all_combinations" | string[][];

/**
 * One global scope for everything this plugin writes.
 *
 * The server default (`combined`) scopes an observation to the memory's WHOLE tag set, and every
 * document we write carries provenance tags — `source:chat`, `harness:<id>`, `knowledge:<kind>`,
 * anything from `retainTags`. That splits one repository's knowledge into a separate observation
 * set per tag combination: work the same repo with two agents and the harness tag alone gives two
 * parallel sets of beliefs that never merge, each blind to the other, at double the consolidation
 * cost (#3564). Those tags are provenance — they say who wrote a memory, not which project the
 * belief is about — so they belong on the facts (where they still filter recall) and not on the
 * consolidation boundary. `shared` keeps them on the facts and consolidates into ONE untagged
 * scope per bank, which is what a bank already is: one project's memory.
 */
export const DEFAULT_OBSERVATION_SCOPES: ObservationScopes = "shared";

export interface ClientOpts {
  apiUrl: string;
  apiToken?: string;
  bank: string;
  /** Repository this bank is about, named in every seeded page's query (`pageScopeRule`). Only
   *  `seedPages()` reads it; it falls back to the bank id, which carries the repo name in the
   *  default `coding-agent::{gitProject}` template. */
  project?: string;
  log?: (msg: string) => void;
  /** Cap on concurrent retain-related requests (drain op polls, deepen pools). Default 10. */
  maxParallelRetains?: number;
  /** Observation scoping for every retain this client sends. Default `DEFAULT_OBSERVATION_SCOPES`. */
  observationScopes?: ObservationScopes;
}

export interface RetainOpts {
  timestamp?: string; // when the content occurred (temporal ranking)
  metadata?: Record<string, string>; // source provenance (returned with recalls)
  /** "append" concatenates `content` onto the stored document instead of replacing it — the whole
   *  point of the live write-back cursor (core/retain-cursor.ts). Requires a document_id. */
  updateMode?: "append";
  /** Deterministic operation id: re-submitting the same payload returns the original operation
   *  instead of admitting a second one. Ignored by servers older than 0.8.6. */
  operationId?: string;
}

/** First release whose retain endpoint honours a caller-supplied `operation_id` (#2937, v0.8.6).
 *  Below it the field is silently ignored — unknown request fields are not rejected — so an append
 *  could be applied twice without us ever knowing. Hence: no idempotency, no append. */
export const MIN_IDEMPOTENT_RETAIN_VERSION = "0.8.6";

/**
 * Raised when the API rate-limits a request (HTTP 429), carrying how long it asked us to wait.
 *
 * A plain Error would be indistinguishable from a 500 or a network fault, and those want opposite
 * treatment: a rate limit is a "not now, ask again" that the caller can honour, while a server
 * error is not worth hammering. Callers that have somewhere safe to wait retry; the rest fail as
 * before.
 */
export class RateLimitedError extends Error {
  readonly code = "rate_limited";
  constructor(readonly retryAfterMs: number) {
    super(`rate limited; retry after ${Math.round(retryAfterMs / 1000)}s`);
    this.name = "RateLimitedError";
  }
}

/** Parse a `Retry-After` header (delta-seconds or HTTP-date) into ms; 0 when absent or unusable. */
export function retryAfterMs(header: string | null | undefined): number {
  if (!header) return 0;
  const secs = Number(header.trim());
  if (Number.isFinite(secs) && secs >= 0) return secs * 1000;
  const at = Date.parse(header);
  return Number.isNaN(at) ? 0 : Math.max(0, at - Date.now());
}

/** Raised when the target server predates the knowledge-pages API surface. */
export class KnowledgePagesUnavailableError extends Error {
  readonly code = "knowledge_pages_unavailable";
  constructor() {
    super("Hindsight server does not support knowledge pages");
    this.name = "KnowledgePagesUnavailableError";
  }
}

const TERMINAL = new Set(["completed", "failed", "cancelled", "error"]);

/** Default cap on concurrent retain-related requests; configurable via `maxParallelRetains`. */
export const DEFAULT_MAX_PARALLEL_RETAINS = 10;

/** How long drain() pauses between poll cycles when the API did not rate-limit (429). */
const POLL_CYCLE_MS = 5000;

/** Minimum backoff after a 429 that carried no (or a shorter) Retry-After. */
const RETRY_AFTER_FLOOR_MS = 10 * 1000;

/**
 * Ceiling on a single backoff, however long `Retry-After` asks for.
 *
 * The header is a server's hint, not a budget we owe it: a large value (an incident, a
 * misconfigured limiter, a proxy inventing one) would otherwise park a drain for as long as it
 * says — up to the whole `maxMs`, with the background seed frozen behind it. Capping keeps the
 * signal without handing over the schedule; if the limit still applies, the next poll simply gets
 * another 429 and backs off again.
 */
const RETRY_AFTER_CEILING_MS = 60 * 1000;

/** Bank-level missions the template seeds once and then leaves alone (#2492). */
const MISSION_FIELDS = ["reflect_mission", "retain_mission", "observations_mission"] as const;

export class HindsightClient {
  readonly apiUrl: string;
  readonly apiToken?: string;
  readonly bank: string;
  readonly project?: string;
  readonly opIds: string[] = []; // async operation ids collected by retain(), for drain()
  /** Tri-state capability probe: unknown until the first page request, then cached. */
  knowledgePagesSupported: boolean | undefined;
  /** Tri-state capability probe: unknown until the first append-mode retain, then cached. */
  private idempotentRetain: boolean | undefined;
  private readonly log: (msg: string) => void;
  readonly maxParallelRetains: number;
  readonly observationScopes: ObservationScopes;

  constructor(o: ClientOpts) {
    this.apiUrl = o.apiUrl.replace(/\/$/, "");
    this.apiToken = o.apiToken;
    this.bank = o.bank;
    this.project = o.project;
    this.log = o.log ?? (() => {});
    this.maxParallelRetains = o.maxParallelRetains || DEFAULT_MAX_PARALLEL_RETAINS;
    this.observationScopes = o.observationScopes ?? DEFAULT_OBSERVATION_SCOPES;
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiToken) h["Authorization"] = `Bearer ${this.apiToken}`;
    return h;
  }

  bankUrl(suffix = ""): string {
    return `${this.apiUrl}/v1/default/banks/${encodeURIComponent(this.bank)}${suffix}`;
  }

  /** `tolerate` adds statuses that are returned to the caller instead of thrown (404 always is). */
  async req(
    method: string,
    url: string,
    body?: unknown,
    tolerate: number[] = []
  ): Promise<Response> {
    // Hard cap on EVERY request: a stalled server (pool deadlock, network) must degrade to a
    // memoryless turn — never hang a host that awaits us (opencode blocks its BOOT on plugin init).
    const r = await fetch(url, {
      method,
      headers: this.headers(),
      body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(15_000),
    });
    if (r.status === 429 && !tolerate.includes(429))
      throw new RateLimitedError(retryAfterMs(r.headers.get("retry-after")));
    if (!r.ok && r.status !== 404 && !tolerate.includes(r.status))
      throw new Error(`${method} ${url} -> ${r.status} ${await r.text()}`);
    return r;
  }

  /** Retain one memory. ALWAYS async: enqueue extraction server-side and collect its op-id for
   *  drain(). Nothing in this plugin can afford to block a coding agent's hook on extraction. */
  async retain(
    content: string,
    context: string,
    documentId: string,
    tags: string[],
    strategy: string,
    opts: RetainOpts = {}
  ): Promise<void> {
    const item: Record<string, unknown> = {
      content,
      context,
      document_id: documentId,
      tags,
      strategy,
      // Sent on EVERY retain, including the server default `combined`, so the scoping a bank's
      // observations were built under is a property of the write rather than of whichever server
      // version happened to process it. Servers older than 0.4.15 ignore the field.
      observation_scopes: this.observationScopes,
    };
    if (opts.timestamp) item.timestamp = opts.timestamp;
    if (opts.metadata) item.metadata = opts.metadata;
    if (opts.updateMode) item.update_mode = opts.updateMode;
    const body: Record<string, unknown> = { items: [item], async: true };
    if (opts.operationId) body.operation_id = opts.operationId;
    const r = await this.req("POST", this.bankUrl("/memories"), body);
    try {
      const j = (await r.json()) as { operation_id?: string };
      if (j.operation_id) this.opIds.push(j.operation_id);
    } catch {
      /* ignore */
    }
  }

  /**
   * Whether this server honours `operation_id` on an async retain, and can therefore be appended to
   * safely (see MIN_IDEMPOTENT_RETAIN_VERSION). Probed once per client via GET /version; anything
   * unreachable, unparseable or older answers "no", which costs efficiency, never correctness.
   */
  async supportsIdempotentRetain(): Promise<boolean> {
    if (this.idempotentRetain === undefined) {
      this.idempotentRetain = await this.probeIdempotentRetain();
      this.log(`server retain idempotency: ${this.idempotentRetain ? "supported" : "unavailable"}`);
    }
    return this.idempotentRetain;
  }

  private async probeIdempotentRetain(): Promise<boolean> {
    try {
      const r = await this.req("GET", `${this.apiUrl}/version`);
      if (!r.ok) return false;
      const j = (await r.json()) as { api_version?: string };
      return semverGte(j.api_version, MIN_IDEMPOTENT_RETAIN_VERSION);
    } catch {
      return false;
    }
  }

  /**
   * Every document_id currently in the bank under a strategy tag (e.g. `source:git`), paginated into a
   * Set. Powers the incremental git-sync's "what's already ingested?" check — since git commits are stored
   * with document_id `git:<sha>`, the returned Set lets a caller diff a ref's commits against memory.
   */
  async listDocumentIds(tag: string): Promise<Set<string>> {
    const ids = new Set<string>();
    const limit = 500;
    for (let offset = 0; ; offset += limit) {
      const q = `?tags=${encodeURIComponent(tag)}&tags_match=all&limit=${limit}&offset=${offset}`;
      const r = await this.req("GET", this.bankUrl(`/documents${q}`));
      let items: { id?: string }[] = [];
      let total = 0;
      try {
        const j = (await r.json()) as { items?: { id?: string }[]; total?: number };
        items = j.items || [];
        total = j.total ?? 0;
      } catch {
        break;
      }
      for (const it of items) if (it.id) ids.add(it.id);
      if (items.length < limit || ids.size >= total) break; // last page reached
    }
    return ids;
  }

  /** Configure the bank: POST the coding bank template manifest to /import (missions, retain
   *  strategies, entity labels), then seed knowledge pages when the server supports them. Both
   *  halves are idempotent, so the deepen engine can re-run this every pass. Creates the bank if
   *  missing; legacy servers continue with the template-only path. */
  async configureBank(opts: { reset?: boolean; pageTrigger?: PageTrigger } = {}): Promise<void> {
    if (opts.reset) {
      await this.req("DELETE", this.bankUrl());
      this.log(`[bank] reset ${this.bank}`);
    }
    // Seed the missions ONCE. After that they belong to whoever set them — see CODING_BANK_STRUCTURE.
    const seeded = opts.reset ? false : await this.hasBankMissions();
    await this.req(
      "POST",
      this.bankUrl("/import"),
      seeded ? CODING_BANK_STRUCTURE : CODING_BANK_TEMPLATE
    );
    this.log(
      seeded
        ? `[bank] structure applied to ${this.bank}: entity_labels {knowledge}, ` +
            `strategies {git, gitlog, conversation, document} — missions left as configured`
        : `[bank] template applied to ${this.bank}: missions, entity_labels {knowledge}, ` +
            `strategies {git, gitlog, conversation, document}`
    );
    await this.seedPages(opts.pageTrigger);
  }

  /**
   * Whether this bank already carries bank-level mission overrides — ours from an earlier seed, or
   * the user's own edit. Either way they are not ours to overwrite.
   *
   * Reads the bank-scoped OVERRIDES, not the resolved config, so inherited global defaults don't
   * read as "already set". A missing bank, or a deployment with the bank-config API switched off,
   * answers false: there is nothing to preserve, and without that API a user cannot set per-bank
   * missions in the first place.
   */
  private async hasBankMissions(): Promise<boolean> {
    try {
      const r = await this.req("GET", this.bankUrl("/config"));
      if (!r.ok) return false;
      const j = (await r.json()) as { overrides?: Record<string, unknown> };
      return MISSION_FIELDS.some((f) => {
        const v = j.overrides?.[f];
        return typeof v === "string" && v.trim() !== "";
      });
    } catch {
      return false;
    }
  }

  /** Delete one document (cascades its memory units/links). Used by deepen's self-cleanup. */
  async deleteDocument(documentId: string): Promise<void> {
    await this.req("DELETE", this.bankUrl(`/documents/${encodeURIComponent(documentId)}`));
  }

  /** Count of operations still ACTIVE on this bank — the list includes terminal ops (completed/
   *  failed/cancelled), so filter by status. Powers syncStatus's "extractions drained" check. */
  async activeOperations(): Promise<number> {
    const r = await this.req("GET", this.bankUrl("/operations"));
    try {
      const j = (await r.json()) as {
        operations?: { status?: string }[];
        items?: { status?: string }[];
      };
      const ops = j.operations ?? j.items ?? [];
      return ops.filter((o) => !TERMINAL.has((o?.status || "").toLowerCase())).length;
    } catch {
      return 0;
    }
  }

  /**
   * Poll each enqueued operation by id until terminal. LIST only shows active ops, so per-id GET is reliable.
   *
   * Concurrency is capped at `maxParallelRetains` (the API rate-limits bursts, not single
   * requests — a 200 to a lone GET with 429s under `Promise.all` over every pending op). A 429
   * leaves the op pending and backs the next cycle off by its `Retry-After` (10s floor) instead
   * of hammering the next cycle 5s later.
   */
  async drain(ids: string[], label: string, maxMs = 60 * 60 * 1000): Promise<void> {
    if (!ids.length) return;
    this.log(`[wait] draining ${ids.length} ${label} operations …`);
    const start = Date.now();
    const pending = new Set(ids);
    let failed = 0;
    while (pending.size && Date.now() - start < maxMs) {
      // Cycle backoff: default 5s; any 429 in the cycle raises it to the longest Retry-After seen
      // (floor 10s) so a rate-limited API gets room to recover before the next poll round.
      let backoffMs = POLL_CYCLE_MS;
      await pool([...pending], this.maxParallelRetains, async (id) => {
        try {
          const r = await fetch(this.bankUrl(`/operations/${id}`), { headers: this.headers() });
          if (r.status === 429) {
            backoffMs = Math.min(
              RETRY_AFTER_CEILING_MS,
              Math.max(backoffMs, RETRY_AFTER_FLOOR_MS, retryAfterMs(r.headers.get("retry-after")))
            );
            return; // op stays pending — retried after the backoff
          }
          if (!r.ok) return;
          const st = (((await r.json()) as { status?: string }).status || "").toLowerCase();
          if (TERMINAL.has(st)) {
            pending.delete(id);
            if (st !== "completed") failed++;
          }
        } catch {
          /* transient — retry next cycle */
        }
      });
      if (pending.size) {
        this.log(`  … ${pending.size}/${ids.length} ${label} ops pending`);
        await sleep(backoffMs);
      }
    }
    this.log(
      `[wait] ${label} drained — ${ids.length - pending.size} done, ${failed} failed` +
        (pending.size ? `, ${pending.size} still pending at timeout` : "")
    );
  }

  /** Reflect: synthesized, root-cause answer over the bank. Bounded so a slow server never hangs a caller. */
  async reflect(
    query: string,
    opts: { budget?: string; timeoutMs?: number } = {}
  ): Promise<string> {
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

  /**
   * The bank's knowledge-base tree (folders + pages, nested). The tree carries names, source
   * queries and staleness but NOT synthesized content, so it is cheap enough to poll.
   */
  async tree(): Promise<KnowledgeNode[]> {
    if (this.knowledgePagesSupported === false) throw new KnowledgePagesUnavailableError();
    const r = await this.req("GET", this.bankUrl("/knowledge-base/tree"));
    if ([404, 405, 501].includes(r.status)) {
      this.knowledgePagesSupported = false;
      throw new KnowledgePagesUnavailableError();
    }
    this.knowledgePagesSupported = true;
    try {
      return ((await r.json()) as { roots?: KnowledgeNode[] }).roots ?? [];
    } catch {
      return [];
    }
  }

  /**
   * List knowledge pages (ids + names only — no synthesized content), flattened out of the tree
   * and shaped as `{items:[…]}` for `parsePageList`. Folders are dropped: they carry no content
   * to read, and a caller enumerating pages wants leaves, not structure.
   *
   * NOTE the ids are knowledge-base node ids (`kp-…`), NOT the backing mental-model ids. Every
   * page read/update in this client goes through the same knowledge-base ids, so a page id from
   * here, from `searchKnowledgePages`, or from a `[[page:<id>]]` link all resolve identically.
   */
  async listPages(): Promise<unknown> {
    const items: { id: string; name: string; description?: string; folder?: string }[] = [];
    const walk = (nodes: KnowledgeNode[], folder?: string): void => {
      for (const n of nodes) {
        if (!n?.id || !n?.name) continue;
        if (n.kind === "page") {
          items.push({
            id: n.id,
            name: n.name,
            ...(n.description ? { description: n.description } : {}),
            ...(folder ? { folder } : {}),
          });
        }
        if (n.children?.length) walk(n.children, n.kind === "folder" ? n.name : folder);
      }
    };
    walk(await this.tree());
    return { items };
  }

  /**
   * Read one knowledge page's synthesized content by knowledge-base id, as an OKF document
   * (YAML frontmatter + markdown body). The endpoint omits the internal reflect trace that built
   * the page — that is 70-95% of the raw bytes and can blow past an MCP host's per-tool-result
   * token cap.
   */
  async getPage(pageId: string): Promise<unknown> {
    if (this.knowledgePagesSupported === false) throw new KnowledgePagesUnavailableError();
    const r = await this.req(
      "GET",
      this.bankUrl(`/knowledge-base/pages/${encodeURIComponent(pageId)}`)
    );
    if (r.status === 404) throw new Error(`knowledge page not found: ${pageId}`);
    return await r.json();
  }

  /** Hybrid (BM25 + vector, RRF-fused) server-side search over the bank's knowledge pages.
   *  Returns page-level hits with a relevance snippet — the real search behind
   *  hindsight_search_knowledge_pages. */
  async searchKnowledgePages(
    query: string,
    limit = 3
  ): Promise<{ id: string; name: string; snippet: string; score: number }[]> {
    if (this.knowledgePagesSupported === false) throw new KnowledgePagesUnavailableError();
    const q = `?q=${encodeURIComponent(query)}&limit=${limit}`;
    const r = await this.req("GET", this.bankUrl(`/knowledge-base/search${q}`));
    const j = (await r.json()) as {
      results?: { id: string; name: string; snippet?: string; score?: number }[];
    };
    return (j.results ?? []).map((x) => ({
      id: x.id,
      name: x.name,
      snippet: x.snippet ?? "",
      score: x.score ?? 0,
    }));
  }

  /**
   * Seed the fixed page taxonomy as knowledge-base pages at the tree root, idempotently.
   *
   * Matched by NAME, not id: `/knowledge-base/pages` mints its own `kp-…` id, so a stable
   * client-chosen id isn't available to match on (unlike the old mental-model path, which keyed
   * off a slug). Names are unique per folder server-side, which makes them a sound key.
   *
   * An existing page is PATCHed rather than recreated so a plugin upgrade that rewords a
   * `source_query` re-syncs onto the live page instead of orphaning its synthesized content —
   * which is how `pageScopeRule`'s repo name reaches banks seeded by an earlier version.
   */
  async seedPages(pageTrigger: PageTrigger = buildPageTrigger()): Promise<void> {
    const pages = pagesFor(this.project ?? this.bank);
    const existing = new Map<string, KnowledgeNode>();
    let roots: KnowledgeNode[];
    try {
      roots = await this.tree();
    } catch (e) {
      if (e instanceof KnowledgePagesUnavailableError) {
        this.log(`[bank] knowledge pages unavailable on ${this.apiUrl}; continuing without pages`);
        return;
      }
      throw e;
    }
    for (const n of roots) {
      if (n.kind === "page" && n.name) existing.set(n.name.toLowerCase(), n);
    }
    let created = 0;
    let updated = 0;
    for (const page of pages) {
      const hit = existing.get(page.name.toLowerCase());
      const body = {
        name: page.name,
        source_query: page.source_query,
        tags: page.tags,
        max_tokens: PAGE_MAX_TOKENS,
        trigger: pageTrigger,
      };
      if (!hit) {
        // 409 = another deepen run seeded this name between our tree read and this POST. That is
        // the outcome we wanted anyway, so tolerate it rather than failing the whole run.
        const r = await this.req("POST", this.bankUrl("/knowledge-base/pages"), body, [409]);
        if ([404, 405, 501].includes(r.status)) {
          this.knowledgePagesSupported = false;
          this.log(
            `[bank] knowledge pages unavailable on ${this.apiUrl}; continuing without pages`
          );
          return;
        }
        if (r.status !== 409) created++;
      } else if (hit.description !== page.source_query) {
        // Only the source query can drift — the name IS the match key, so it can't.
        const r = await this.req(
          "PATCH",
          this.bankUrl(`/knowledge-base/nodes/${encodeURIComponent(hit.id)}`),
          { source_query: page.source_query, tags: page.tags }
        );
        if ([404, 405, 501].includes(r.status)) {
          this.knowledgePagesSupported = false;
          this.log(
            `[bank] knowledge pages unavailable on ${this.apiUrl}; continuing without pages`
          );
          return;
        }
        updated++;
      }
    }
    this.log(
      `[bank] knowledge pages seeded on ${this.bank}: ${created} created, ${updated} re-synced, ` +
        `${pages.length - created - updated} unchanged`
    );
  }

  /** URL/id-safe slug: lowercase, non-alphanumerics → "-", trim dashes, cap length; fallback "initiative". */
  private slugify(s: string): string {
    return (
      s
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "")
        .slice(0, 60) || "initiative"
    );
  }

  /**
   * Active-path capture: register a major feature as a per-initiative page + a tagged marker memory.
   * New initiative → creates a page under the Initiatives folder tagged `relatedPageId:<pageId>`.
   * Enhancement (relatesToPageId) → no new page; only a marker accruing to the existing page.
   */
  async captureInitiative(args: {
    title: string;
    summary: string;
    relatesToPageId?: string;
    stamp?: RetainStamp;
    /** Same refresh policy as the seeded pages — an initiative page is one of them, and used to
     *  carry its own hardcoded copy of this trigger. */
    pageTrigger?: PageTrigger;
  }): Promise<{ page_id: string }> {
    // `/knowledge-base/pages` mints its OWN page id (kp-…); we can't set it. So for a new initiative
    // we create the page first and adopt the server-assigned id — that id is what the return value
    // and the marker's `relatedPageId` tag must use, or `read_knowledge_page(id)` 404s and the
    // `[[page:<id>]]` link points at nothing.
    let pageId = args.relatesToPageId;
    if (!pageId) {
      const folderId = await this.ensureFolder("Initiatives");
      const r = await this.req("POST", this.bankUrl("/knowledge-base/pages"), {
        name: args.title,
        source_query: `Summarize the "${args.title}" initiative: what is being built or changed and why, and its current state — drawn from the project's memory.`,
        parent_id: folderId,
        tags: ["knowledge:feature-work"],
        trigger: args.pageTrigger ?? buildPageTrigger(),
      });
      try {
        const j = (await r.json()) as { page_id?: string; id?: string };
        pageId = j.page_id ?? j.id;
      } catch {
        /* fall through to the slug fallback below */
      }
      pageId ||= `initiative-${this.slugify(args.title)}`; // last resort if the response lacked an id
    }
    const verb = args.relatesToPageId ? "Enhancement to an existing initiative" : "New initiative";
    const content = `${verb}: ${args.title}. ${args.summary}`;
    // Unique marker document id (NOT pageId) so repeated captures accrue instead of replacing.
    const markerId = `initiative-marker-${this.slugify(args.title)}-${Date.now()}`;
    const tags = [
      ...new Set([
        ...(args.stamp?.tags ?? []),
        "knowledge:feature-work",
        `relatedPageId:${pageId}`,
      ]),
    ];
    if (Object.keys(args.stamp?.metadata ?? {}).length) {
      await this.retain(content, "initiative marker", markerId, tags, "document", {
        metadata: args.stamp?.metadata,
      });
    } else {
      await this.retain(content, "initiative marker", markerId, tags, "document");
    }
    return { page_id: pageId };
  }

  /** Find a root folder by name (case-insensitive) or create it; returns its id. Fail-open to undefined. */
  async ensureFolder(name: string): Promise<string | undefined> {
    try {
      const tree = (await (await this.req("GET", this.bankUrl("/knowledge-base/tree"))).json()) as {
        roots?: { id?: string; kind?: string; name?: string }[];
      };
      const hit = (tree.roots || []).find(
        (n) => n.kind === "folder" && (n.name || "").toLowerCase() === name.toLowerCase()
      );
      if (hit?.id) return hit.id;
    } catch {
      /* fall through to create */
    }
    try {
      const r = await this.req("POST", this.bankUrl("/knowledge-base/folders"), { name });
      return ((await r.json()) as { id?: string }).id;
    } catch {
      return undefined;
    }
  }
}
