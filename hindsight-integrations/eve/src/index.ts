/**
 * Hindsight long-term memory for Vercel Eve agents.
 *
 * Wraps eve's `defineMcpClientConnection` so an agent gains persistent memory by
 * dropping a single file under `agent/connections/`. The helper fills in the
 * Hindsight MCP endpoint, a model-facing description, and bearer auth, reading
 * sensible defaults from the environment:
 *
 * ```ts
 * // agent/connections/hindsight.ts
 * import { defineHindsightConnection } from "@vectorize-io/hindsight-eve";
 * export default defineHindsightConnection(); // HINDSIGHT_MCP_URL + HINDSIGHT_API_KEY
 * ```
 */
import { defineMcpClientConnection } from "eve/connections";

/** The argument eve's connection factory accepts; options pass straight through. */
type McpConnectionInput = Parameters<typeof defineMcpClientConnection>[0];

/** Hindsight Cloud MCP endpoint, used when no URL is configured. */
export const HINDSIGHT_CLOUD_MCP_URL = "https://api.hindsight.vectorize.io/mcp";

/**
 * Default model-facing description written into the generated connection. Eve
 * surfaces it when the agent discovers this connection's tools
 * (`connection__hindsight__retain` / `recall` / `reflect`).
 */
export const DEFAULT_DESCRIPTION =
  "Hindsight long-term memory: retain facts from this session, recall relevant history " +
  "from past sessions, and reflect over consolidated mental models.";

export interface HindsightConnectionOptions {
  /** Hindsight MCP endpoint. Defaults to `HINDSIGHT_MCP_URL`, then Hindsight Cloud. */
  url?: string;
  /**
   * API key sent as `Authorization: Bearer <key>`. Defaults to `HINDSIGHT_API_KEY`.
   * Pass `null` to emit a no-auth connection (local/self-hosted dev only).
   */
  apiKey?: string | null;
  /** Bank to scope memory to; sent as the `X-Bank-Id` header. Defaults to `HINDSIGHT_MCP_BANK_ID`. */
  bankId?: string;
  /** Override the model-facing description. */
  description?: string;
  /** Restrict which Hindsight tools the model can see. */
  tools?: McpConnectionInput["tools"];
  /** Human-in-the-loop approval policy (e.g. `once()` from `eve/tools/approval`). */
  approval?: McpConnectionInput["approval"];
}

/** Fully-resolved connection settings, after applying options and environment defaults. */
export interface ResolvedHindsightConnection {
  url: string;
  description: string;
  apiKey: string | null;
  bankId: string | null;
  tools?: McpConnectionInput["tools"];
  approval?: McpConnectionInput["approval"];
}

/** First non-empty string among the candidates, or `null`. */
function firstNonEmpty(...values: Array<string | null | undefined>): string | null {
  for (const value of values) {
    if (typeof value === "string" && value.length > 0) return value;
  }
  return null;
}

/**
 * Whether a URL points at Hindsight Cloud. Matched on host (not exact string)
 * so a trailing slash, `http`/`https`, or a regional subdomain still triggers
 * the missing-key guard below instead of letting the request fail with a raw
 * 401. The dot boundary keeps it from matching look-alike hosts like
 * `nothindsight.vectorize.io`.
 */
function isHindsightCloudUrl(url: string): boolean {
  try {
    const host = new URL(url).hostname;
    return host === "hindsight.vectorize.io" || host.endsWith(".hindsight.vectorize.io");
  } catch {
    return false;
  }
}

/**
 * Resolve options against environment defaults. Pure and side-effect free so the
 * precedence rules can be unit-tested without constructing a live connection.
 */
export function resolveHindsightConnection(
  options: HindsightConnectionOptions = {},
  env: NodeJS.ProcessEnv = process.env
): ResolvedHindsightConnection {
  const url = options.url ?? firstNonEmpty(env.HINDSIGHT_MCP_URL) ?? HINDSIGHT_CLOUD_MCP_URL;

  // `apiKey: null` is an explicit no-auth opt-out; `undefined` falls back to the env var.
  const apiKey =
    options.apiKey === undefined ? firstNonEmpty(env.HINDSIGHT_API_KEY) : options.apiKey;

  const bankId = options.bankId ?? firstNonEmpty(env.HINDSIGHT_MCP_BANK_ID);

  if (isHindsightCloudUrl(url) && !apiKey) {
    throw new Error(
      "Hindsight Cloud requires an API key. Set HINDSIGHT_API_KEY, pass `apiKey`, or point " +
        "`url`/HINDSIGHT_MCP_URL at a self-hosted server (use `apiKey: null` for a no-auth server)."
    );
  }

  return {
    url,
    description: options.description ?? DEFAULT_DESCRIPTION,
    apiKey,
    bankId,
    tools: options.tools,
    approval: options.approval,
  };
}

/**
 * Build the plain definition object handed to eve. Kept separate from
 * {@link defineHindsightConnection} so the auth/header wiring is testable without
 * depending on the shape of eve's returned connection.
 */
export function buildHindsightConnectionDefinition(
  resolved: ResolvedHindsightConnection
): McpConnectionInput {
  return {
    url: resolved.url,
    description: resolved.description,
    // `{ token }` is eve's TokenResult shape (sent as `Authorization: Bearer`).
    // It rides on eve 0.11's auth contract, which the pinned peer/dev dep covers.
    ...(resolved.apiKey
      ? { auth: { getToken: async () => ({ token: resolved.apiKey as string }) } }
      : {}),
    ...(resolved.bankId ? { headers: { "X-Bank-Id": resolved.bankId } } : {}),
    ...(resolved.tools ? { tools: resolved.tools } : {}),
    ...(resolved.approval ? { approval: resolved.approval } : {}),
  };
}

/**
 * Define an eve MCP connection to a Hindsight memory server. Export the result as
 * the default from `agent/connections/hindsight.ts`.
 */
export function defineHindsightConnection(options: HindsightConnectionOptions = {}) {
  return defineMcpClientConnection(
    buildHindsightConnectionDefinition(resolveHindsightConnection(options))
  );
}

export default defineHindsightConnection;
