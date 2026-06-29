/**
 * Minimal Hindsight API client — only the mental-model read endpoints needed to
 * mirror a bank into a folder. Uses the global `fetch` (Node 18+). Intentionally
 * dependency-free so the CLI stays npx-installable with no transitive risk.
 */

export interface MentalModel {
  id: string;
  bank_id: string;
  name: string;
  source_query?: string | null;
  content?: string | null;
  tags?: string[];
  max_tokens?: number | null;
  trigger?: Record<string, unknown> | null;
  last_refreshed_at?: string | null;
  created_at?: string | null;
  is_stale?: boolean | null;
}

export interface ClientOptions {
  apiUrl: string;
  apiToken?: string;
}

const PAGE_SIZE = 1000;

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

  /**
   * List every mental model in a bank, paginating until exhausted.
   *
   * @param detail "content" includes the markdown body + tags; "full" also
   *   carries is_stale and the (large) reflect payload, which we ignore.
   */
  async listMentalModels(
    bankId: string,
    detail: "metadata" | "content" | "full" = "content"
  ): Promise<MentalModel[]> {
    const all: MentalModel[] = [];
    let offset = 0;

    for (;;) {
      const url =
        `${this.apiUrl}/v1/default/banks/${encodeURIComponent(bankId)}/mental-models` +
        `?detail=${detail}&limit=${PAGE_SIZE}&offset=${offset}`;

      const resp = await fetch(url, { headers: this.headers() });
      if (!resp.ok) {
        const body = await resp.text().catch(() => "");
        throw new ApiError(
          `Failed to list mental models for bank "${bankId}" (HTTP ${resp.status})`,
          resp.status,
          body
        );
      }

      const data = (await resp.json()) as { items?: MentalModel[] };
      const items = data.items ?? [];
      all.push(...items);
      if (items.length < PAGE_SIZE) break;
      offset += PAGE_SIZE;
    }

    return all;
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
