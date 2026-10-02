import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  createNativeMemoryCapability,
  registerNativeMemoryCapabilityIfEnabled,
  type NativeCapabilityDeps,
} from "./native-capability.js";
import type { BankScopedClient } from "./index.js";
import type { PluginConfig, PluginHookAgentContext } from "./types.js";

function makeBankClient(overrides?: Partial<BankScopedClient>): BankScopedClient {
  return {
    bankId: "seo-ops",
    retain: vi.fn(),
    recall: vi.fn().mockResolvedValue({
      results: [
        {
          id: "fact-1",
          text: "Owner ruled pgedeon-20 for all marketplaces.",
          scores: { final: 0.91, semantic: 0.8, keyword: 0.42, reranker: 0.95 },
          context: "owner ruling",
          mentioned_at: "2026-09-29T00:00:00.000Z",
        },
        {
          id: "fact-2",
          text: "Train deploys nightly.",
          scores: { final: 0.71 },
        },
      ],
    }),
    setMissions: vi.fn(),
    ...overrides,
  } as BankScopedClient;
}

function makeDeps(bankClient: BankScopedClient | null, config: PluginConfig = {}) {
  const deps: NativeCapabilityDeps = {
    getClientForContext: vi.fn(async (_ctx?: PluginHookAgentContext) => bankClient),
    getPluginConfig: () => config,
    getServerEndpoint: () => ({ apiUrl: "http://192.168.0.81:8888", apiToken: null }),
  };
  return deps;
}

describe("createNativeMemoryCapability", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, status: 200 }));
  });

  it("exposes a runtime that resolves a search manager per agent", async () => {
    const capability = createNativeMemoryCapability(makeDeps(makeBankClient()));
    expect(capability.runtime).toBeDefined();

    const { manager, error } = await capability.runtime!.getMemorySearchManager({
      cfg: {},
      agentId: "seo-ops",
    });
    expect(error).toBeUndefined();
    expect(manager).toBeDefined();
  });

  it("maps recall results into native search results", async () => {
    const bankClient = makeBankClient();
    const capability = createNativeMemoryCapability(makeDeps(bankClient, { recallTopK: 10 }));
    const { manager } = (await capability.runtime!.getMemorySearchManager({
      cfg: {},
      agentId: "seo-ops",
    }))!;

    const results = await manager!.search("pgedeon tag ruling");
    expect(bankClient.recall).toHaveBeenCalledWith(
      expect.objectContaining({
        query: "pgedeon tag ruling",
        maxTokens: 1024,
        budget: "mid",
        types: ["observation"],
      }),
      10000,
      undefined
    );
    expect(results).toHaveLength(2);
    expect(results[0]).toMatchObject({
      path: "hindsight://seo-ops/fact-1",
      startLine: 1,
      endLine: 1,
      score: 0.91,
      vectorScore: 0.8,
      textScore: 0.42,
      snippet: "Owner ruled pgedeon-20 for all marketplaces.",
      source: "memory",
    });
    expect(results[0].provenance?.observedAt).toBe(Date.parse("2026-09-29T00:00:00.000Z"));
    // result without mentioned_at carries no provenance
    expect(results[1].provenance).toBeUndefined();
  });

  it("forwards maxResults as a hard slice and minScore as a final-score floor", async () => {
    const bankClient = makeBankClient();
    const capability = createNativeMemoryCapability(makeDeps(bankClient));
    const { manager } = (await capability.runtime!.getMemorySearchManager({
      cfg: {},
      agentId: "seo-ops",
    }))!;

    const results = await manager!.search("q", { maxResults: 1, minScore: 0.5 });
    expect(bankClient.recall).toHaveBeenCalledWith(
      expect.objectContaining({ minScores: { final: 0.5 } }),
      expect.anything(),
      undefined
    );
    expect(results).toHaveLength(1);
  });

  it("returns manager: null with an error when the hindsight client is unavailable", async () => {
    const capability = createNativeMemoryCapability(makeDeps(null));
    const { manager, error } = await capability.runtime!.getMemorySearchManager({
      cfg: {},
      agentId: "seo-ops",
    });
    expect(manager).toBeNull();
    expect(error).toMatch(/not initialized/);
  });

  it("reports a hindsight provider status and probeVectorAvailability true", async () => {
    const capability = createNativeMemoryCapability(
      makeDeps(makeBankClient(), { llmModel: "compact" })
    );
    const { manager } = (await capability.runtime!.getMemorySearchManager({
      cfg: {},
      agentId: "seo-ops",
    }))!;

    const status = manager!.status();
    expect(status).toMatchObject({ backend: "builtin", provider: "hindsight", model: "compact" });
    expect(await manager!.probeVectorAvailability()).toBe(true);
  });

  it("readFile reports not_found (memories are not file-backed)", async () => {
    const capability = createNativeMemoryCapability(makeDeps(makeBankClient()));
    const { manager } = (await capability.runtime!.getMemorySearchManager({
      cfg: {},
      agentId: "seo-ops",
    }))!;
    const read = await manager!.readFile({ relPath: "MEMORY.md" });
    expect(read).toEqual({ status: "not_found", text: "", path: "MEMORY.md" });
  });

  it("caches embedding probes within the TTL", async () => {
    const capability = createNativeMemoryCapability(makeDeps(makeBankClient()));
    const { manager } = (await capability.runtime!.getMemorySearchManager({
      cfg: {},
      agentId: "seo-ops",
    }))!;

    const first = await manager!.probeEmbeddingAvailability();
    expect(first.ok).toBe(true);
    expect(first.cached).toBeUndefined();

    const cached = manager!.getCachedEmbeddingAvailability?.();
    expect(cached?.ok).toBe(true);
    expect(cached?.cached).toBe(true);

    const second = await manager!.probeEmbeddingAvailability();
    expect(second.cached).toBe(true);
    expect(vi.mocked(fetch).mock.calls).toHaveLength(1);
  });

  it("closeAllMemorySearchManagers drops cached managers and probe state", async () => {
    const capability = createNativeMemoryCapability(makeDeps(makeBankClient()));
    await capability.runtime!.getMemorySearchManager({ cfg: {}, agentId: "seo-ops" });
    await capability.runtime!.closeAllMemorySearchManagers();
    // recreating a manager after close must work
    const { manager } = (await capability.runtime!.getMemorySearchManager({
      cfg: {},
      agentId: "seo-ops",
    }))!;
    expect(manager).toBeDefined();
  });

  it("resolveMemoryBackendConfig reports the builtin backend", () => {
    const capability = createNativeMemoryCapability(makeDeps(makeBankClient()));
    expect(capability.runtime!.resolveMemoryBackendConfig({ cfg: {}, agentId: "x" })).toEqual({
      backend: "builtin",
    });
  });
});

describe("registerNativeMemoryCapabilityIfEnabled", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, status: 200 }));
  });

  it("registers the capability when the api supports it", () => {
    const registerMemoryCapability = vi.fn();
    const ok = registerNativeMemoryCapabilityIfEnabled(
      { registerMemoryCapability },
      makeDeps(makeBankClient()),
      {}
    );
    expect(ok).toBe(true);
    expect(registerMemoryCapability).toHaveBeenCalledWith(
      expect.objectContaining({ runtime: expect.objectContaining({ getMemorySearchManager: expect.any(Function) }) })
    );
  });

  it("is a no-op when nativeCapability is false", () => {
    const registerMemoryCapability = vi.fn();
    const ok = registerNativeMemoryCapabilityIfEnabled(
      { registerMemoryCapability },
      makeDeps(makeBankClient()),
      { nativeCapability: false }
    );
    expect(ok).toBe(false);
    expect(registerMemoryCapability).not.toHaveBeenCalled();
  });

  it("degrades gracefully when the OpenClaw version lacks the API", () => {
    const ok = registerNativeMemoryCapabilityIfEnabled({}, makeDeps(makeBankClient()), {});
    expect(ok).toBe(false);
  });
});
