/**
 * Minimal Hindsight REST client + memory-formatting helpers. Native `fetch`,
 * zero dependencies. Pure (no `eve` import) so it can be unit-tested with a
 * mocked `fetch`.
 *
 * Endpoints (tenant is the literal `default`, bank is in the path):
 *   recall:  POST /v1/default/banks/{bank}/memories/recall
 *   retain:  POST /v1/default/banks/{bank}/memories
 *   reflect: POST /v1/default/banks/{bank}/reflect
 */

export type RecallBudget = "low" | "mid" | "high";

/** One memory returned by recall. The content lives in `text`. */
export interface RecallResult {
  id: string;
  text: string;
  type?: string;
  context?: string;
  tags?: string[];
}

export interface RecallResponse {
  results: RecallResult[];
}

/** One item to retain. `content` is the only required field. */
export interface RetainItem {
  content: string;
  context?: string;
  metadata?: Record<string, string>;
  /** ISO-8601, `"unset"`, or null (= now). */
  timestamp?: string | null;
  /**
   * Stable document id. Retaining the same id again replaces the earlier
   * version, so callers can use an idempotency key (eve's `operationId`).
   */
  document_id?: string;
}

/** A caller-supplied cancellation signal, combined with the client timeout. */
export interface RequestOptions {
  signal?: AbortSignal;
}

export interface RecallOptions extends RequestOptions {
  budget?: RecallBudget;
  maxTokens?: number;
  types?: Array<"world" | "experience" | "observation">;
}

export interface RetainOptions extends RequestOptions {
  async?: boolean;
}

export interface ReflectOptions extends RequestOptions {
  budget?: RecallBudget;
}

/** Reflect answer. `text` is the reasoned answer grounded in the bank's memories. */
export interface ReflectResponse {
  text: string;
}

/**
 * Markers that fence the recalled-context block injected as a system message.
 * Used by {@link buildRecallMarkdown} (to wrap) and {@link stripSentinelBlocks}
 * (to ensure recalled facts are never re-retained).
 */
export const SENTINEL_OPEN = "<!-- hindsight:recalled-context -->";
export const SENTINEL_CLOSE = "<!-- /hindsight:recalled-context -->";

const SENTINEL_RE = new RegExp(`${SENTINEL_OPEN}[\\s\\S]*?${SENTINEL_CLOSE}`, "g");

/** Remove any injected recalled-context block from text (defensive de-dup guard). */
export function stripSentinelBlocks(text: string): string {
  return text.replace(SENTINEL_RE, "").trim();
}

/**
 * Render recalled memories as a system-message markdown block, fenced with the
 * sentinel markers so the retain side can recognize and exclude it.
 */
export function buildRecallMarkdown(results: readonly RecallResult[]): string {
  if (results.length === 0) return "";
  const lines = results.map((r) => `- ${r.text}`).join("\n");
  return [
    SENTINEL_OPEN,
    "## What you already know about this user (from long-term memory)",
    "Use this context to tailor your response. Do not repeat it back verbatim.",
    "",
    lines,
    SENTINEL_CLOSE,
  ].join("\n");
}

/** A non-2xx response, with the status so callers can special-case it. */
export class HindsightHttpError extends Error {
  readonly status: number;

  constructor(status: number, path: string, detail: string) {
    super(`Hindsight HTTP ${status} from ${path}: ${detail}`);
    this.name = "HindsightHttpError";
    this.status = status;
  }
}

/** Thin HTTP client for Hindsight's memory REST API. */
export class HindsightRestClient {
  private readonly baseUrl: string;
  private readonly token: string | null;
  private readonly timeoutMs: number;

  constructor(baseUrl: string, token?: string | null, timeoutMs = 15_000) {
    const url = (baseUrl ?? "").trim();
    if (!url) throw new Error("Hindsight API URL is required");
    this.baseUrl = url.replace(/\/$/, "");
    this.token = token ?? null;
    this.timeoutMs = timeoutMs;
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.token) h["Authorization"] = `Bearer ${this.token}`;
    return h;
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
    signal?: AbortSignal
  ): Promise<T> {
    const timeout = AbortSignal.timeout(this.timeoutMs);
    try {
      const resp = await fetch(`${this.baseUrl}${path}`, {
        method,
        headers: this.headers(),
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
      });
      if (!resp.ok) {
        const text = await resp.text().catch(() => "");
        throw new HindsightHttpError(resp.status, path, text);
      }
      return (await resp.json()) as T;
    } catch (error) {
      // Surface the timeout as an ordinary error, not a bare DOMException.
      if (timeout.aborted && !signal?.aborted) {
        throw new Error(`Hindsight request to ${path} timed out after ${this.timeoutMs}ms`);
      }
      throw error;
    }
  }

  private bankPath(bankId: string): string {
    return `/v1/default/banks/${encodeURIComponent(bankId)}`;
  }

  /**
   * Recall memories for a bank. `query` is required by the API. A bank that
   * does not exist yet (nothing has been retained into it) has no memories,
   * so its 404 is returned as an empty result rather than an error.
   */
  async recall(bankId: string, query: string, opts: RecallOptions = {}): Promise<RecallResponse> {
    const body: Record<string, unknown> = {
      query,
      budget: opts.budget ?? "mid",
      max_tokens: opts.maxTokens ?? 1024,
    };
    if (opts.types) body["types"] = opts.types;
    try {
      return await this.request<RecallResponse>(
        "POST",
        `${this.bankPath(bankId)}/memories/recall`,
        body,
        opts.signal
      );
    } catch (error) {
      if (error instanceof HindsightHttpError && error.status === 404) return { results: [] };
      throw error;
    }
  }

  /** Retain items into a bank. The bank is auto-created on first retain. */
  async retain(
    bankId: string,
    items: readonly RetainItem[],
    opts: RetainOptions = {}
  ): Promise<void> {
    await this.request(
      "POST",
      `${this.bankPath(bankId)}/memories`,
      { items, async: opts.async ?? true },
      opts.signal
    );
  }

  /** Ask the bank a question and get a reasoned answer grounded in its memories. */
  async reflect(
    bankId: string,
    query: string,
    opts: ReflectOptions = {}
  ): Promise<ReflectResponse> {
    return this.request<ReflectResponse>(
      "POST",
      `${this.bankPath(bankId)}/reflect`,
      { query, budget: opts.budget ?? "low" },
      opts.signal
    );
  }
}
