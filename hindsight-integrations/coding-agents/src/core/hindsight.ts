/**
 * Harness-agnostic Hindsight HTTP client (raw fetch, no SDK dep).
 *
 * Every harness adapter and the backfill CLI go through this one client: it configures the bank
 * (missions + git/chat retain strategies), retains memories, reflects, drains async operations, and
 * creates knowledge pages. Nothing here knows about opencode/claude-code/etc.
 */
import { CODING_BANK_TEMPLATE, PAGE_MAX_TOKENS, PAGE_TRIGGER, PAGES } from "./missions";
import { createHash } from "node:crypto";
import { sleep } from "./util";

/** One node of GET /knowledge-base/tree. Only the fields this client reads. */
export interface KnowledgeNode {
  id: string;
  kind: "folder" | "page";
  name: string;
  /** The page's source query (OKF `description`) — what a re-sync compares against. */
  description?: string;
  children?: KnowledgeNode[];
}

export interface ClientOpts {
  apiUrl: string;
  apiToken?: string;
  bank: string;
  log?: (msg: string) => void;
}

export interface RetainOpts {
  timestamp?: string; // when the content occurred (temporal ranking)
  metadata?: Record<string, string>; // source provenance (returned with recalls)
  async?: boolean; // enqueue server-side (default) vs block on extraction
  /** Reuse the server operation when this exact async retain is retried. */
  idempotent?: boolean;
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

// UUIDv5(URL namespace, "https://github.com/vectorize-io/hindsight/tree/main/" +
// "hindsight-integrations/coding-agents#retain-operation/v1"). The operation name is the
// canonical JSON string produced by JSON.stringify({ bank, item }).
const RETAIN_OPERATION_NAMESPACE = "61cecd19-ec9a-56d6-aa7e-7e891b8b2bcb";

function uuidV5(namespace: string, name: string): string {
  const namespaceBytes = Buffer.from(namespace.replaceAll("-", ""), "hex");
  const bytes = createHash("sha1")
    .update(namespaceBytes)
    .update(name, "utf8")
    .digest()
    .subarray(0, 16);

  bytes[6] = (bytes[6] & 0x0f) | 0x50;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = bytes.toString("hex");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function retainOperationId(bank: string, item: Record<string, unknown>): string {
  return uuidV5(RETAIN_OPERATION_NAMESPACE, JSON.stringify({ bank, item }));
}

export class HindsightClient {
  readonly apiUrl: string;
  readonly apiToken?: string;
  readonly bank: string;
  readonly opIds: string[] = []; // async operation ids collected by retain(), for drain()
  /** Tri-state capability probe: unknown until the first page request, then cached. */
  knowledgePagesSupported: boolean | undefined;
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
    if (!r.ok && r.status !== 404 && !tolerate.includes(r.status))
      throw new Error(`${method} ${url} -> ${r.status} ${await r.text()}`);
    return r;
  }

  /** Retain one memory. Async by default: enqueue extraction server-side and collect its op-id for drain(). */
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
    };
    if (opts.timestamp) item.timestamp = opts.timestamp;
    if (opts.metadata) item.metadata = opts.metadata;
    const isAsync = opts.async !== false;
    const r = await this.req("POST", this.bankUrl("/memories"), {
      items: [item],
      async: isAsync,
      ...(isAsync && opts.idempotent ? { operation_id: retainOperationId(this.bank, item) } : {}),
    });
    if (isAsync) {
      try {
        const j = (await r.json()) as { operation_id?: string };
        if (j.operation_id) this.opIds.push(j.operation_id);
      } catch {
        /* ignore */
      }
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
  async configureBank(opts: { reset?: boolean } = {}): Promise<void> {
    if (opts.reset) {
      await this.req("DELETE", this.bankUrl());
      this.log(`[bank] reset ${this.bank}`);
    }
    await this.req("POST", this.bankUrl("/import"), CODING_BANK_TEMPLATE);
    this.log(
      `[bank] template applied to ${this.bank}: missions, entity_labels {knowledge}, ` +
        `strategies {git, gitlog, conversation, document}`
    );
    await this.seedPages();
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

  /** Poll each enqueued operation by id until terminal. LIST only shows active ops, so per-id GET is reliable. */
  async drain(ids: string[], label: string, maxMs = 60 * 60 * 1000): Promise<void> {
    if (!ids.length) return;
    this.log(`[wait] draining ${ids.length} ${label} operations …`);
    const start = Date.now();
    const pending = new Set(ids);
    let failed = 0;
    while (pending.size && Date.now() - start < maxMs) {
      await Promise.all(
        [...pending].map(async (id) => {
          try {
            const r = await fetch(this.bankUrl(`/operations/${id}`), { headers: this.headers() });
            if (!r.ok) return;
            const st = (((await r.json()) as { status?: string }).status || "").toLowerCase();
            if (TERMINAL.has(st)) {
              pending.delete(id);
              if (st !== "completed") failed++;
            }
          } catch {
            /* transient — retry next cycle */
          }
        })
      );
      if (pending.size) {
        this.log(`  … ${pending.size}/${ids.length} ${label} ops pending`);
        await sleep(5000);
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
   * Seed the fixed `PAGES` taxonomy as knowledge-base pages at the tree root, idempotently.
   *
   * Matched by NAME, not id: `/knowledge-base/pages` mints its own `kp-…` id, so a stable
   * client-chosen id isn't available to match on (unlike the old mental-model path, which keyed
   * off a slug). Names are unique per folder server-side, which makes them a sound key.
   *
   * An existing page is PATCHed rather than recreated so a plugin upgrade that rewords a
   * `source_query` re-syncs onto the live page instead of orphaning its synthesized content.
   */
  async seedPages(): Promise<void> {
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
    for (const page of PAGES) {
      const hit = existing.get(page.name.toLowerCase());
      const body = {
        name: page.name,
        source_query: page.source_query,
        tags: page.tags,
        max_tokens: PAGE_MAX_TOKENS,
        trigger: PAGE_TRIGGER,
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
        `${PAGES.length - created - updated} unchanged`
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
        trigger: {
          fact_types: ["world", "experience", "observation"],
          refresh_after_consolidation: true,
        },
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
    await this.retain(
      content,
      "initiative marker",
      markerId,
      ["knowledge:feature-work", `relatedPageId:${pageId}`],
      "document",
      { async: true }
    );
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
