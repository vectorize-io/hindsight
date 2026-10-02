import type { MinScores } from "@vectorize-io/hindsight-client";
import type { RecallResponse } from "./types.js";
import type { BankScopedClient } from "./index.js";
import type { PluginConfig, PluginHookAgentContext } from "./types.js";
import { info, verbose, warn } from "./logger.js";

/**
 * Native OpenClaw memory capability.
 *
 * OpenClaw resolves the active memory runtime from the capability registration of
 * the plugin that owns the exclusive `memory` slot (see its `registerMemoryCapability`
 * / `MemoryPluginRuntime` API). When hindsight holds the slot without registering a
 * capability, every native consumer — the Control UI Memory page (`doctor.memory.status`),
 * the `active-memory` plugin, and the `plugin-sdk/memory-host-search` helpers — reports
 * "memory plugin unavailable".
 *
 * This module provides that capability, backed by Hindsight recall on the same
 * per-context bank the hooks and knowledge tools use. It never injects into prompts
 * (no promptBuilder): prompt-side memory remains the autoRecall hook's job, so
 * enabling this cannot double-inject.
 *
 * Types below intentionally mirror OpenClaw 2026.7–2026.9's shipped plugin types
 * (they are not importable from the plugin SDK at build time). Structurally
 * compatible with memory-core's implementation.
 */

// --- Mirrored OpenClaw types ------------------------------------------------

export interface NativeMemorySearchResult {
  path: string;
  startLine: number;
  endLine: number;
  score: number;
  vectorScore?: number;
  textScore?: number;
  snippet: string;
  source: "memory" | "sessions";
  importance?: number;
  citation?: string;
  provenance?: {
    originClass: "owner" | "agent" | "untrusted" | "system";
    sessionKind: "interactive" | "cron" | "heartbeat" | "subagent" | "unknown";
    observedAt: number;
  };
}

export interface NativeMemoryReadResult {
  status?: "not_found";
  text: string;
  path: string;
  truncated?: boolean;
  from?: number;
  lines?: number;
  nextFrom?: number;
}

export interface NativeMemoryProviderStatus {
  backend: "builtin";
  provider: string;
  model?: string;
  requestedProvider?: string;
  dirty?: boolean;
  custom?: Record<string, unknown>;
}

export interface NativeEmbeddingProbeResult {
  ok: boolean;
  error?: string;
  checked?: boolean;
  cached?: boolean;
  checkedAtMs?: number;
}

export interface NativeMemorySearchManager {
  search(
    query: string,
    opts?: {
      maxResults?: number;
      minScore?: number;
      signal?: AbortSignal;
    } & Record<string, unknown>
  ): Promise<NativeMemorySearchResult[]>;
  readFile(params: { relPath: string; from?: number; lines?: number }): Promise<NativeMemoryReadResult>;
  status(): NativeMemoryProviderStatus;
  probeEmbeddingAvailability(): Promise<NativeEmbeddingProbeResult>;
  getCachedEmbeddingAvailability?(): NativeEmbeddingProbeResult | null;
  probeVectorAvailability(): Promise<boolean>;
  close?(): Promise<void>;
}

export interface NativeMemoryPluginRuntime {
  getMemorySearchManager(params: {
    cfg: unknown;
    agentId: string;
    purpose?: "default" | "status" | "cli";
  }): Promise<{
    manager: NativeMemorySearchManager | null;
    error?: string;
    debug?: { backend?: "builtin"; purpose?: string; managerMs?: number };
  }>;
  resolveMemoryBackendConfig(params: { cfg: unknown; agentId: string }): { backend: "builtin" };
  closeMemorySearchManager?(params: { cfg: unknown; agentId: string }): Promise<void>;
  closeAllMemorySearchManagers?(): Promise<void>;
}

export interface NativeMemoryPluginCapability {
  runtime?: NativeMemoryPluginRuntime;
}

// --- Wiring -----------------------------------------------------------------

export interface NativeCapabilityDeps {
  /** Bank-scoped client factory — the same path hooks and knowledge tools use. */
  getClientForContext(ctx?: PluginHookAgentContext): Promise<BankScopedClient | null>;
  /** Current plugin config (post-normalization). */
  getPluginConfig(): PluginConfig;
  /** Server endpoint for availability probes (external API URL/token, or daemon). */
  getServerEndpoint(): { apiUrl: string | null; apiToken: string | null };
}

const PROBE_CACHE_TTL_MS = 60_000;
const DEFAULT_RECALL_TIMEOUT_MS = 10_000;

interface ManagerEntry {
  manager: NativeMemorySearchManager;
  probeCache: { result: NativeEmbeddingProbeResult; expiresAtMs: number } | null;
}

function mapRecallResults(
  bankId: string,
  response: RecallResponse,
  maxResults?: number
): NativeMemorySearchResult[] {
  const results = response.results ?? [];
  const sliced = typeof maxResults === "number" && maxResults > 0 ? results.slice(0, maxResults) : results;
  return sliced.map((r) => {
    const mentionedAtMs = r.mentioned_at ? Date.parse(r.mentioned_at) : NaN;
    return {
      path: `hindsight://${bankId}/${r.id}`,
      startLine: 1,
      endLine: 1,
      score: r.scores?.final ?? 0,
      vectorScore: r.scores?.semantic ?? undefined,
      textScore: r.scores?.keyword ?? undefined,
      snippet: r.text,
      source: "memory" as const,
      citation: r.context ?? undefined,
      provenance:
        Number.isFinite(mentionedAtMs) && mentionedAtMs > 0
          ? {
              originClass: "agent" as const,
              sessionKind: "unknown" as const,
              observedAt: mentionedAtMs,
            }
          : undefined,
    };
  });
}

async function probeServer(
  endpoint: { apiUrl: string | null; apiToken: string | null }
): Promise<NativeEmbeddingProbeResult> {
  if (!endpoint.apiUrl) {
    return { ok: false, error: "hindsight API URL not configured", checked: true };
  }
  try {
    const headers: Record<string, string> = {};
    if (endpoint.apiToken) headers.Authorization = `Bearer ${endpoint.apiToken}`;
    const res = await fetch(`${endpoint.apiUrl.replace(/\/+$/, "")}/v1/default/banks`, {
      method: "GET",
      headers,
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) {
      return { ok: false, error: `hindsight API responded ${res.status}`, checked: true };
    }
    return { ok: true, checked: true };
  } catch (err) {
    return {
      ok: false,
      error: `hindsight API unreachable: ${err instanceof Error ? err.message : String(err)}`,
      checked: true,
    };
  }
}

function createManager(agentId: string, deps: NativeCapabilityDeps): ManagerEntry {
  const entry: ManagerEntry = {
    manager: {
      async search(query, opts) {
        const config = deps.getPluginConfig();
        const bankClient = await deps.getClientForContext({ agentId });
        if (!bankClient) {
          throw new Error(`hindsight client unavailable for agent "${agentId}"`);
        }
        const minScores: MinScores | undefined =
          typeof opts?.minScore === "number" ? { final: opts.minScore } : config.recallMinScores;
        const response = await bankClient.recall(
          {
            query,
            maxTokens: config.recallMaxTokens ?? 1024,
            budget: config.recallBudget ?? "mid",
            types: config.recallTypes ?? ["observation"],
            preferObservations: config.preferObservations ?? false,
            minScores,
          },
          config.recallTimeoutMs ?? DEFAULT_RECALL_TIMEOUT_MS,
          opts?.signal
        );
        const maxResults =
          typeof opts?.maxResults === "number" && opts.maxResults > 0
            ? opts.maxResults
            : config.recallTopK;
        return mapRecallResults(bankClient.bankId, response, maxResults);
      },
      async readFile(params) {
        // Hindsight memories are not file-backed; report not-found rather than
        // pretending. The native memory_get tool remains served by memory-core.
        return { status: "not_found" as const, text: "", path: params.relPath };
      },
      status() {
        const config = deps.getPluginConfig();
        const endpoint = deps.getServerEndpoint();
        return {
          backend: "builtin" as const,
          provider: "hindsight",
          model: typeof config.llmModel === "string" ? config.llmModel : undefined,
          dirty: false,
          custom: {
            bankIdSource: "hindsight-openclaw",
            apiUrl: endpoint.apiUrl ?? `local-daemon:${config.apiPort ?? 9077}`,
          },
        };
      },
      async probeEmbeddingAvailability() {
        const now = Date.now();
        if (entry.probeCache && entry.probeCache.expiresAtMs > now) {
          return { ...entry.probeCache.result, cached: true };
        }
        const result = await probeServer(deps.getServerEndpoint());
        entry.probeCache = { result, expiresAtMs: now + PROBE_CACHE_TTL_MS };
        return result;
      },
      getCachedEmbeddingAvailability() {
        return entry.probeCache ? { ...entry.probeCache.result, cached: true } : null;
      },
      async probeVectorAvailability() {
        // Semantic retrieval runs server-side (embeddings + reranker in Hindsight).
        return true;
      },
      async close() {
        entry.probeCache = null;
      },
    },
    probeCache: null,
  };
  return entry;
}

export function createNativeMemoryCapability(
  deps: NativeCapabilityDeps
): NativeMemoryPluginCapability {
  const managers = new Map<string, ManagerEntry>();

  return {
    runtime: {
      async getMemorySearchManager({ agentId, purpose }) {
        const started = Date.now();
        const key = agentId || "default";
        let entry = managers.get(key);
        if (!entry) {
          entry = createManager(key, deps);
          managers.set(key, entry);
        }
        // Fail fast when hindsight itself is not initialized, so the native host
        // can surface a precise error instead of a manager whose calls throw.
        try {
          const bankClient = await deps.getClientForContext({ agentId: key });
          if (!bankClient) {
            return {
              manager: null,
              error: "hindsight client not initialized (service not started or init failed)",
            };
          }
        } catch (err) {
          return {
            manager: null,
            error: `hindsight client error: ${err instanceof Error ? err.message : String(err)}`,
          };
        }
        verbose(
          `[Hindsight] native memory manager ready for agent "${key}" (purpose: ${purpose ?? "default"})`
        );
        return {
          manager: entry.manager,
          debug: {
            backend: "builtin",
            purpose,
            managerMs: Date.now() - started,
          },
        };
      },
      resolveMemoryBackendConfig() {
        return { backend: "builtin" as const };
      },
      async closeMemorySearchManager({ agentId }) {
        const entry = managers.get(agentId);
        if (entry) {
          await entry.manager.close?.();
          managers.delete(agentId);
        }
      },
      async closeAllMemorySearchManagers() {
        for (const [key, entry] of managers) {
          await entry.manager.close?.();
          managers.delete(key);
        }
      },
    },
  };
}

export function registerNativeMemoryCapabilityIfEnabled(
  api: unknown,
  deps: NativeCapabilityDeps,
  config: PluginConfig
): boolean {
  if (config.nativeCapability === false) {
    verbose("[Hindsight] native memory capability disabled via config");
    return false;
  }
  // Feature-detected: api.registerMemoryCapability exists on OpenClaw builds that
  // ship the native memory runtime (memory-core registers through the same method).
  const register = (
    api as { registerMemoryCapability?: (capability: NativeMemoryPluginCapability) => void }
  )?.registerMemoryCapability;
  if (typeof register !== "function") {
    warn(
      "[Hindsight] api.registerMemoryCapability unavailable on this OpenClaw version — native memory page/active-memory integration skipped"
    );
    return false;
  }
  try {
    register.call(api, createNativeMemoryCapability(deps));
    info("[Hindsight] native memory capability registered (slot runtime + search manager)");
    return true;
  } catch (err) {
    warn(
      `[Hindsight] native memory capability registration failed: ${
        err instanceof Error ? err.message : String(err)
      }`
    );
    return false;
  }
}
