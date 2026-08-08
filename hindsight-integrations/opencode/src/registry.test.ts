import { describe, it, expect, vi } from "vitest";
import { ClientRegistry, toResolver } from "./registry.js";
import { makeConfig } from "./test-helpers.js";

function makeMockClient() {
  return {
    retain: vi.fn().mockResolvedValue({}),
    recall: vi.fn().mockResolvedValue({ results: [] }),
    reflect: vi.fn().mockResolvedValue({ text: "" }),
    createBank: vi.fn().mockResolvedValue({}),
  } as any;
}

describe("toResolver", () => {
  it("wraps a bare client so forAgent() always returns it", () => {
    const client = makeMockClient();
    const resolver = toResolver(client);
    expect(resolver.forAgent("build")).toBe(client);
    expect(resolver.forAgent(undefined)).toBe(client);
    expect(resolver.forAgent(null)).toBe(client);
  });

  it("passes through an existing ClientResolver unchanged", () => {
    const client = makeMockClient();
    const resolver = { forAgent: () => client };
    expect(toResolver(resolver)).toBe(resolver);
  });
});

describe("ClientRegistry", () => {
  it("returns the default client for the fallback token", () => {
    const defaultClient = makeMockClient();
    const factory = vi.fn();
    const registry = new ClientRegistry({
      baseUrl: "https://api.test",
      config: makeConfig({ hindsightApiToken: "static" }),
      defaultClient,
      clientFactory: factory,
    });

    expect(registry.forAgent(undefined)).toBe(defaultClient);
    expect(registry.forAgent("anything")).toBe(defaultClient); // no per-agent map → fallback token
    expect(factory).not.toHaveBeenCalled();
  });

  it("builds and caches a per-agent client only when the agent has a distinct token", () => {
    const defaultClient = makeMockClient();
    const buildClient = makeMockClient();
    const reviewClient = makeMockClient();
    const factory = vi.fn();
    factory.mockReturnValueOnce(buildClient).mockReturnValueOnce(reviewClient);

    const config = makeConfig({
      hindsightApiToken: "static",
      hindsightApiTokens: { build: "build-key", "code-reviewer": "review-key" },
    });
    const registry = new ClientRegistry({
      baseUrl: "https://api.test",
      config,
      defaultClient,
      clientFactory: factory,
    });

    expect(registry.forAgent("build")).toBe(buildClient);
    expect(factory).toHaveBeenCalledWith({ baseUrl: "https://api.test", apiKey: "build-key" });

    // Second call for the same agent reuses the cached client (no new construction).
    expect(registry.forAgent("build")).toBe(buildClient);
    expect(factory).toHaveBeenCalledTimes(1);

    // A different agent gets its own client.
    expect(registry.forAgent("code-reviewer")).toBe(reviewClient);
    expect(factory).toHaveBeenCalledWith({
      baseUrl: "https://api.test",
      apiKey: "review-key",
    });
    expect(factory).toHaveBeenCalledTimes(2);

    // An agent with no entry falls back to the default client (no construction).
    expect(registry.forAgent("security-reviewer")).toBe(defaultClient);
    expect(factory).toHaveBeenCalledTimes(2);
  });

  it("passes the resolved token as apiKey to the factory", () => {
    const defaultClient = makeMockClient();
    const built = makeMockClient();
    const factory = vi.fn().mockReturnValue(built);
    const registry = new ClientRegistry({
      baseUrl: "https://api.test",
      config: makeConfig({
        hindsightApiToken: "static",
        hindsightApiTokens: { build: "build-key" },
      }),
      defaultClient,
      clientFactory: factory,
    });

    registry.forAgent("build");
    expect(factory).toHaveBeenCalledWith({ baseUrl: "https://api.test", apiKey: "build-key" });
  });

  it("shares one client across agents that map to the same token", () => {
    const defaultClient = makeMockClient();
    const shared = makeMockClient();
    const factory = vi.fn().mockReturnValue(shared);
    const registry = new ClientRegistry({
      baseUrl: "https://api.test",
      config: makeConfig({
        hindsightApiToken: "static",
        hindsightApiTokens: { "agent-a": "shared-key", "agent-b": "shared-key" },
      }),
      defaultClient,
      clientFactory: factory,
    });

    expect(registry.forAgent("agent-a")).toBe(shared);
    expect(registry.forAgent("agent-b")).toBe(shared); // same token → same client
    expect(factory).toHaveBeenCalledTimes(1);
  });

  it("uses the default client when dynamicApiKey is disabled", () => {
    const defaultClient = makeMockClient();
    const factory = vi.fn();
    const registry = new ClientRegistry({
      baseUrl: "https://api.test",
      config: makeConfig({
        dynamicApiKey: false,
        hindsightApiToken: "static",
        hindsightApiTokens: { build: "build-key" },
      }),
      defaultClient,
      clientFactory: factory,
    });

    expect(registry.forAgent("build")).toBe(defaultClient);
    expect(factory).not.toHaveBeenCalled();
  });
});
