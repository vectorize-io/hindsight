/**
 * Tenant-aware Hindsight API client factory for the control plane.
 *
 * Supports two modes:
 * 1. Multi-tenant: HINDSIGHT_API_TENANT_KEY_MAP=key1:name1;key2:name2
 * 2. Single-tenant (backwards-compat): HINDSIGHT_CP_DATAPLANE_API_KEY=key
 *
 * In multi-tenant mode, getClientForTenant(name) returns a client scoped to
 * that tenant's schema. In single-tenant mode, all calls use the single key.
 */

import {
  HindsightClient,
  HindsightError,
  createClient,
  createConfig,
  sdk,
} from "@vectorize-io/hindsight-client";

export const DATAPLANE_URL =
  process.env.HINDSIGHT_CP_DATAPLANE_API_URL || "http://localhost:8888";

// --- Tenant key map parsing ---

interface TenantEntry {
  name: string;
  apiKey: string;
}

function parseTenantKeyMap(raw: string): TenantEntry[] {
  if (!raw.trim()) return [];
  return raw.split(";").filter(Boolean).map((entry) => {
    const colonIdx = entry.indexOf(":");
    if (colonIdx === -1) {
      throw new Error(
        `Invalid HINDSIGHT_API_TENANT_KEY_MAP entry "${entry}". Expected "key:name".`
      );
    }
    return {
      apiKey: entry.slice(0, colonIdx).trim(),
      name: entry.slice(colonIdx + 1).trim(),
    };
  });
}

// Same env var as the API server — one key map for both.
const TENANT_KEY_MAP_RAW = process.env.HINDSIGHT_API_TENANT_KEY_MAP || "";
const SINGLE_KEY = process.env.HINDSIGHT_CP_DATAPLANE_API_KEY || "";

const tenantEntries: TenantEntry[] = TENANT_KEY_MAP_RAW
  ? parseTenantKeyMap(TENANT_KEY_MAP_RAW)
  : SINGLE_KEY
    ? [{ name: "default", apiKey: SINGLE_KEY }]
    : [];

const tenantsByName = new Map(tenantEntries.map((e) => [e.name, e]));

// --- Client cache (one per tenant, created lazily) ---

interface TenantClients {
  hindsightClient: HindsightClient;
  lowLevelClient: ReturnType<typeof createClient>;
  apiKey: string;
}

const clientCache = new Map<string, TenantClients>();

function buildClients(apiKey: string): TenantClients {
  return {
    hindsightClient: new HindsightClient({
      baseUrl: DATAPLANE_URL,
      apiKey: apiKey || undefined,
    }),
    lowLevelClient: createClient(
      createConfig({
        baseUrl: DATAPLANE_URL,
        headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : undefined,
      })
    ),
    apiKey,
  };
}

/**
 * Get SDK clients for a specific tenant.
 *
 * In multi-tenant mode, providing an unknown tenant name throws an error
 * to prevent silent cross-tenant data leakage.  In single-tenant mode
 * (or when no name is provided), falls back to the first configured tenant.
 */
export function getClientForTenant(tenantName?: string | null): TenantClients {
  let name: string;
  if (tenantName) {
    if (!tenantsByName.has(tenantName)) {
      if (isMultiTenant()) {
        throw new Error(`Unknown tenant: ${tenantName}`);
      }
      // Single-tenant fallback
      name = tenantEntries[0]?.name ?? "default";
    } else {
      name = tenantName;
    }
  } else {
    name = tenantEntries[0]?.name ?? "default";
  }

  let clients = clientCache.get(name);
  if (!clients) {
    const entry = tenantsByName.get(name);
    clients = buildClients(entry?.apiKey ?? "");
    clientCache.set(name, clients);
  }
  return clients;
}

/**
 * Return the list of configured tenant names.
 * Used by the /api/tenants route and TenantContext.
 */
export function getTenantNames(): string[] {
  return tenantEntries.map((e) => e.name);
}

/**
 * Whether multi-tenant mode is active (more than one tenant configured).
 */
export function isMultiTenant(): boolean {
  return tenantEntries.length > 1;
}

/**
 * Auth headers for direct fetch calls to the dataplane API.
 */
export function getDataplaneHeaders(
  tenantName?: string | null,
  extra?: Record<string, string>
): Record<string, string> {
  const { apiKey } = getClientForTenant(tenantName);
  const headers: Record<string, string> = { ...extra };
  if (apiKey) {
    headers["Authorization"] = `Bearer ${apiKey}`;
  }
  return headers;
}

/**
 * Build a dataplane URL for a bank-scoped endpoint with the bank id properly encoded.
 * Bank ids may contain `:`, `/`, `%`, etc. (e.g. openclaw `agent::channel::user`),
 * which must be percent-encoded before being interpolated into a URL path.
 */
export function dataplaneBankUrl(bankId: string, suffix = ""): string {
  return `${DATAPLANE_URL}/v1/default/banks/${encodeURIComponent(bankId)}${suffix}`;
}

export { sdk, HindsightError };
