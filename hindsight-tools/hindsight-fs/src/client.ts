/**
 * Minimal Hindsight API client — only the knowledge-base read endpoints needed
 * to mirror a bank's knowledge base into a folder. Uses the global `fetch`
 * (Node 18+). Intentionally dependency-free so the CLI stays npx-installable.
 *
 * Two endpoints, fetched once per sync:
 *  - GET /knowledge-base/tree   → the folder/page hierarchy (no page bodies)
 *  - GET /knowledge-base/export → every page's OKF markdown in one bundle
 * We join them by page id, so a bank of any size is two HTTP calls.
 */

export interface KnowledgeNode {
  id: string;
  kind: "folder" | "page";
  name: string;
  parent_id: string | null;
  mental_model_id?: string | null;
  mission?: string | null;
  managed?: boolean;
  description?: string | null;
  tags?: string[];
  timestamp?: string | null;
  children: KnowledgeNode[];
}

export interface KnowledgeSnapshot {
  /** Top-level folder/page nodes (each with nested `children`). */
  roots: KnowledgeNode[];
  /** page id → its full OKF markdown document (frontmatter + body). */
  content: Map<string, string>;
}

export interface ClientOptions {
  apiUrl: string;
  apiToken?: string;
}

export class HindsightFsClient {
  private readonly apiUrl: string;
  private readonly apiToken?: string;

  constructor(opts: ClientOptions) {
    this.apiUrl = opts.apiUrl.replace(/\/+$/, "");
    this.apiToken = opts.apiToken;
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = {
      Accept: "application/json",
      "User-Agent": "hindsight-fs/0.1.0",
    };
    if (this.apiToken) h.Authorization = `Bearer ${this.apiToken}`;
    return h;
  }

  private base(bankId: string): string {
    return `${this.apiUrl}/v1/default/banks/${encodeURIComponent(bankId)}/knowledge-base`;
  }

  private async getJson<T>(url: string, what: string, bankId: string): Promise<T> {
    const resp = await fetch(url, { headers: this.headers() });
    if (!resp.ok) {
      const body = await resp.text().catch(() => "");
      throw new ApiError(
        `Failed to ${what} for bank "${bankId}" (HTTP ${resp.status})`,
        resp.status,
        body
      );
    }
    return (await resp.json()) as T;
  }

  /** Fetch the knowledge-base tree + page contents and join them. */
  async loadKnowledge(bankId: string): Promise<KnowledgeSnapshot> {
    const tree = await this.getJson<{ roots?: KnowledgeNode[] }>(
      `${this.base(bankId)}/tree`,
      "fetch knowledge-base tree",
      bankId
    );
    const bundle = await this.getJson<{ files?: { path: string; content: string }[] }>(
      `${this.base(bankId)}/export`,
      "export knowledge base",
      bankId
    );

    // The bundle holds `<page-id>.md` (the page doc), `index.md`, and
    // `<page-id>.log.md` (history). We only want the page docs.
    const content = new Map<string, string>();
    for (const file of bundle.files ?? []) {
      if (file.path === "index.md" || file.path.endsWith(".log.md")) continue;
      if (file.path.endsWith(".md")) {
        content.set(file.path.slice(0, -".md".length), file.content);
      }
    }

    return { roots: tree.roots ?? [], content };
  }
}

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly body: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}
