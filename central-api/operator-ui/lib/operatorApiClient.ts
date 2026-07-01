export type ApiMode = "local" | "remote";

export type ConnectionConfig = {
  apiBaseUrl: string;
  apiKey?: string;
  jwtToken?: string;
  mode: ApiMode;
  connected: boolean;
  lastCheckedAt?: string;
  lastError?: string;
};

export type ApiError = {
  code: string;
  message: string;
  details?: unknown;
  path?: string;
  attemptedPaths?: string[];
};

export type ApiResult<T> =
  | { ok: true; data: T; status: number }
  | { ok: false; error: ApiError; status: number };

export type HealthResponse = {
  status: string;
  service?: string;
  version?: string;
  engines?: EngineHealth[];
};

export type DependencyHealthResponse = {
  status: string;
  timestamp_ms: number;
  services: Record<string, {
    name: string;
    port: number;
    status: string;
    latency_ms: number;
    last_checked_at: number;
    error_summary?: string | null;
    service_type?: string | null;
    is_critical?: boolean;
  }>;
  providers: Array<{
    name: string;
    type: string;
    status: string;
    latency_ms: number;
    model_count: number;
    error_summary?: string | null;
  }>;
  governance_healthy: boolean;
  audit_events_count: number;
  quarantine_items_count: number;
  is_quarantined: boolean;
};

export type AuthValidationResponse = {
  user?: {
    id?: string;
    email?: string;
    is_operator?: boolean;
  };
  auth_method?: string;
  scopes?: string[];
};

export type MemorySearchRequest = {
  query: string;
  mode?: "semantic" | "keyword" | "hybrid";
  limit?: number;
};

export type MemorySearchHit = {
  memory_id?: string;
  id?: string;
  backend?: string;
  content?: string;
  memory_type?: string;
  score?: number | null;
  citation?: Record<string, unknown> | null;
  metadata?: Record<string, unknown>;
};

export type MemorySearchResponse = {
  query: string;
  mode: string;
  count: number;
  results: MemorySearchHit[];
};

export type AuditEvent = {
  trace_id?: string;
  tenant_id?: string;
  actor_id?: string | null;
  operation?: string;
  resource_type?: string | null;
  resource_id?: string | null;
  outcome?: string;
  metadata?: Record<string, unknown>;
  timestamp?: number;
  [key: string]: unknown;
};

export type AuditEventsResponse = {
  tenant_id?: string;
  count: number;
  events: AuditEvent[];
};

export type QuarantineQueueResponse = {
  total: number;
  items: Array<{
    id: string;
    content_hash: string;
    classification: string;
    status: string;
    created_at: string;
    created_by?: string | null;
    reason?: string | null;
  }>;
};

export type PolicyCheckRequest = {
  content: string;
  memory_type?: string;
  workspace_id?: string;
  metadata?: Record<string, unknown>;
};

export type PolicyCheckResponse = {
  decision: string;
  reason?: string;
  classification?: string;
  quarantine_id?: string | null;
};

export type ExecutionLedgerResponse = {
  executions: unknown[];
  limit?: number;
  offset?: number;
};

export type EngineHealth = {
  backend: string;
  status: string;
  detail?: Record<string, unknown>;
};

export type ServiceHealthResponse = {
  overall_status?: string;
  service_count?: number;
  healthy_services?: number;
  degraded_services?: number;
  down_services?: number;
  provider_count?: number;
  governance_healthy?: boolean;
  [key: string]: unknown;
};

type FetchLike = typeof fetch;

type ClientOptions = {
  fetchImpl?: FetchLike;
  timeoutMs?: number;
  logger?: Pick<Console, "debug" | "warn" | "error">;
};

const DEFAULT_TIMEOUT_MS = 10_000;
const HEALTH_PATHS = ["/health", "/api/health", "/api/health/engines"] as const;

export function normalizeApiBaseUrl(rawBaseUrl: string): string {
  const trimmed = rawBaseUrl.trim();
  if (!trimmed) {
    throw new Error("API base URL is required");
  }

  const withScheme = /^[a-z][a-z0-9+.-]*:\/\//i.test(trimmed) ? trimmed : `http://${trimmed}`;
  const url = new URL(withScheme);
  url.pathname = url.pathname.replace(/\/+$/, "");
  url.search = "";
  url.hash = "";
  return url.toString().replace(/\/$/, "");
}

export function maskSecret(value: string, secrets: Array<string | undefined> = []): string {
  let masked = value;
  for (const secret of secrets) {
    if (secret && secret.length > 0) {
      masked = masked.split(secret).join("[REDACTED]");
    }
  }

  return masked
    .replace(/Bearer\s+[A-Za-z0-9._~+/=-]+/gi, "Bearer [REDACTED]")
    .replace(/(X-Api-Key["':\s]+)[A-Za-z0-9._~+/=-]+/gi, "$1[REDACTED]")
    .replace(/(api[_-]?key["':\s]+)[A-Za-z0-9._~+/=-]+/gi, "$1[REDACTED]")
    .replace(/(jwt[_-]?token["':\s]+)[A-Za-z0-9._~+/=-]+/gi, "$1[REDACTED]");
}

export function buildAuthHeaders(config: ConnectionConfig): Record<string, string> {
  const headers: Record<string, string> = {};

  if (config.jwtToken) {
    headers.Authorization = `Bearer ${config.jwtToken}`;
    return headers;
  }

  if (config.apiKey) {
    headers.Authorization = `Bearer ${config.apiKey}`;
    headers["X-Api-Key"] = config.apiKey;
  }

  return headers;
}

export function createDisconnectedConfig(
  apiBaseUrl: string,
  mode: ApiMode = "local",
): ConnectionConfig {
  return {
    apiBaseUrl: normalizeApiBaseUrl(apiBaseUrl),
    mode,
    connected: false,
  };
}

export class OperatorApiClient {
  private readonly baseUrl: string;
  private readonly fetchImpl: FetchLike;
  private readonly timeoutMs: number;
  private readonly logger?: Pick<Console, "debug" | "warn" | "error">;

  constructor(
    private readonly config: ConnectionConfig,
    options: ClientOptions = {},
  ) {
    this.baseUrl = normalizeApiBaseUrl(config.apiBaseUrl);
    this.fetchImpl = options.fetchImpl ?? fetch;
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.logger = options.logger;
  }

  get connection(): ConnectionConfig {
    return {
      ...this.config,
      apiBaseUrl: this.baseUrl,
      apiKey: this.config.apiKey ? "[REDACTED]" : undefined,
      jwtToken: this.config.jwtToken ? "[REDACTED]" : undefined,
    };
  }

  async health(): Promise<ApiResult<HealthResponse>> {
    const result = await this.requestWithFallback<HealthResponse | EngineHealth[]>(
      HEALTH_PATHS,
      { method: "GET" },
    );
    if (!result.ok) return result;

    return {
      ok: true,
      status: result.status,
      data: Array.isArray(result.data)
        ? { status: "ok", service: "central-api", engines: result.data }
        : result.data,
    };
  }

  healthDependencies(): Promise<ApiResult<DependencyHealthResponse>> {
    return this.request<DependencyHealthResponse>("/health/dependencies", { method: "GET" });
  }

  validateAuth(): Promise<ApiResult<AuthValidationResponse>> {
    return this.request<AuthValidationResponse>("/me", { method: "GET" });
  }

  async memorySearch(request: MemorySearchRequest): Promise<ApiResult<MemorySearchResponse>> {
    const result = await this.request<Partial<MemorySearchResponse & { degraded?: string[] }>>(
      "/api/memory/search/unified",
      {
        method: "POST",
        body: JSON.stringify({
          query: request.query,
          k: request.limit ?? 10,
        }),
      },
    );

    if (!result.ok) return result;

    const results = Array.isArray(result.data.results) ? result.data.results : [];
    return {
      ok: true,
      status: result.status,
      data: {
        query: result.data.query ?? request.query,
        mode: result.data.mode ?? request.mode ?? "hybrid",
        count: typeof result.data.count === "number" ? result.data.count : results.length,
        results,
      },
    };
  }

  async auditEvents(limit = 100): Promise<ApiResult<AuditEventsResponse>> {
    const result = await this.request<Partial<AuditEventsResponse>>(
      `/api/audit/events?${new URLSearchParams({ limit: String(limit) }).toString()}`,
      { method: "GET" },
    );

    if (!result.ok) return result;

    const events = Array.isArray(result.data.events) ? result.data.events : [];
    return {
      ok: true,
      status: result.status,
      data: {
        ...result.data,
        count: typeof result.data.count === "number" ? result.data.count : events.length,
        events,
      },
    };
  }

  governanceQuarantine(status?: string, limit = 50): Promise<ApiResult<QuarantineQueueResponse>> {
    const params = new URLSearchParams({ limit: String(limit) });
    if (status) params.set("status", status);
    return this.request<QuarantineQueueResponse>(`/api/gov/approval-queue?${params.toString()}`, {
      method: "GET",
    });
  }

  governancePolicyCheck(request: PolicyCheckRequest): Promise<ApiResult<PolicyCheckResponse>> {
    return this.request<PolicyCheckResponse>("/api/gov/policy-check", {
      method: "POST",
      body: JSON.stringify(request),
    });
  }

  executionLedger(status?: string, limit = 50, offset = 0): Promise<ApiResult<ExecutionLedgerResponse>> {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (status) params.set("status", status);
    return this.request<ExecutionLedgerResponse>(`/api/executions/history?${params.toString()}`, {
      method: "GET",
    });
  }

  adapterStatus(): Promise<ApiResult<EngineHealth[]>> {
    return this.request<EngineHealth[]>("/api/health/engines", { method: "GET" });
  }

  serviceHealth(): Promise<ApiResult<ServiceHealthResponse>> {
    return this.request<ServiceHealthResponse>("/api/observability/service-status", {
      method: "GET",
    });
  }

  async request<T>(path: string, init: RequestInit = {}): Promise<ApiResult<T>> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);
    const headers = mergeHeaders(
      {
        "Content-Type": "application/json",
        ...buildAuthHeaders(this.config),
      },
      init.headers,
    );

    try {
      const res = await this.fetchImpl(this.url(path), {
        ...init,
        headers,
        cache: "no-store",
        signal: init.signal ?? controller.signal,
      });
      const data = await readResponseBody(res);

      if (!res.ok) {
        return {
          ok: false,
          status: res.status,
          error: this.normalizeError(data, res.status, path),
        };
      }

      return { ok: true, status: res.status, data: data as T };
    } catch (error) {
      const normalized = this.normalizeError(error, 0, path);
      this.logSafe("warn", `Operator API request failed: ${normalized.message}`);
      return { ok: false, status: 0, error: normalized };
    } finally {
      clearTimeout(timeout);
    }
  }

  private async requestWithFallback<T>(
    paths: readonly string[],
    init: RequestInit,
  ): Promise<ApiResult<T>> {
    let last: ApiResult<T> | undefined;
    const attemptedPaths: string[] = [];

    for (const path of paths) {
      attemptedPaths.push(path);
      const result = await this.request<T>(path, init);
      if (result.ok) return result;
      last = result;

      if (result.status !== 404 && result.status !== 405) {
        break;
      }
    }

    return {
      ok: false,
      status: last?.status ?? 0,
      error: {
        ...(last?.ok === false ? last.error : { code: "request_failed", message: "Request failed" }),
        attemptedPaths,
      },
    };
  }

  private url(path: string): string {
    const normalizedPath = path.startsWith("/") ? path : `/${path}`;
    return `${this.baseUrl}${normalizedPath}`;
  }

  private normalizeError(error: unknown, status: number, path: string): ApiError {
    const secrets = [this.config.apiKey, this.config.jwtToken];
    let message = "Request failed";
    let details: unknown;

    if (error instanceof Error) {
      message = error.name === "AbortError" ? "Request timed out" : error.message;
      details = { name: error.name };
    } else if (typeof error === "string") {
      message = error;
    } else if (isRecord(error)) {
      details = error;
      const candidate = error.detail ?? error.message ?? error.error;
      message = typeof candidate === "string" ? candidate : JSON.stringify(candidate ?? error);
    }

    const safeMessage = maskSecret(message, secrets);
    const code =
      status === 0
        ? "network_error"
        : status === 401 || status === 403
          ? "auth_error"
          : status === 404
            ? "not_found"
            : status >= 500
              ? "server_error"
              : "request_error";

    return {
      code,
      message: safeMessage,
      details: sanitizeDetails(details, secrets),
      path,
    };
  }

  private logSafe(level: "debug" | "warn" | "error", message: string): void {
    this.logger?.[level]?.(maskSecret(message, [this.config.apiKey, this.config.jwtToken]));
  }
}

export function createOperatorApiClient(
  config: ConnectionConfig,
  options?: ClientOptions,
): OperatorApiClient {
  return new OperatorApiClient(config, options);
}

async function readResponseBody(res: Response): Promise<unknown> {
  const text = await res.text().catch(() => "");
  if (!text) return null;

  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function sanitizeDetails(details: unknown, secrets: Array<string | undefined>): unknown {
  if (typeof details === "string") return maskSecret(details, secrets);
  if (Array.isArray(details)) return details.map((item) => sanitizeDetails(item, secrets));
  if (!isRecord(details)) return details;

  return Object.fromEntries(
    Object.entries(details).map(([key, value]) => [
      key,
      /token|secret|key|authorization/i.test(key)
        ? "[REDACTED]"
        : sanitizeDetails(value, secrets),
    ]),
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function mergeHeaders(
  baseHeaders: Record<string, string>,
  overrideHeaders?: HeadersInit,
): Record<string, string> {
  const headers = new Headers(baseHeaders);
  if (overrideHeaders) {
    new Headers(overrideHeaders).forEach((value, key) => headers.set(key, value));
  }

  return Object.fromEntries(headers.entries());
}
