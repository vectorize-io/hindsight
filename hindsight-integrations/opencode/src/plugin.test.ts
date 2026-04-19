import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock the HindsightClient before importing the plugin
vi.mock("@vectorize-io/hindsight-client", () => {
  const MockHindsightClient = vi.fn(function (this: any) {
    this.retain = vi.fn().mockResolvedValue({});
    this.recall = vi.fn().mockResolvedValue({ results: [] });
    this.reflect = vi.fn().mockResolvedValue({ text: "" });
    this.createBank = vi.fn().mockResolvedValue({});
  });
  return { HindsightClient: MockHindsightClient };
});

import { HindsightPlugin } from "./index.js";
import { HindsightClient } from "@vectorize-io/hindsight-client";

const mockPluginInput = {
  client: {
    session: {
      messages: vi.fn().mockResolvedValue({ data: [] }),
    },
  },
  project: { id: "test-project", worktree: "/tmp/test", vcs: "git" },
  directory: "/tmp/test-project",
  worktree: "/tmp/test-project",
  serverUrl: new URL("http://localhost:3000"),
  $: {} as any,
};

describe("HindsightPlugin", () => {
  const originalEnv = { ...process.env };

  beforeEach(() => {
    for (const key of Object.keys(process.env)) {
      if (key.startsWith("HINDSIGHT_")) delete process.env[key];
    }
    vi.clearAllMocks();
  });

  afterEach(() => {
    process.env = { ...originalEnv };
  });

  it("returns empty hooks when no API URL configured", async () => {
    const result = await HindsightPlugin(mockPluginInput as any);
    expect(result).toEqual({});
    expect(HindsightClient).not.toHaveBeenCalled();
  });

  it("returns tools and hooks when configured", async () => {
    process.env.HINDSIGHT_API_URL = "http://localhost:8888";

    const result = await HindsightPlugin(mockPluginInput as any);

    expect(HindsightClient).toHaveBeenCalledWith({
      baseUrl: "http://localhost:8888",
      apiKey: undefined,
    });

    expect(result.tool).toBeDefined();
    expect(result.tool!.hindsight_retain).toBeDefined();
    expect(result.tool!.hindsight_recall).toBeDefined();
    expect(result.tool!.hindsight_reflect).toBeDefined();
    expect(result.event).toBeDefined();
    expect(result["experimental.session.compacting"]).toBeDefined();
    expect(result["experimental.chat.system.transform"]).toBeDefined();
  });

  it("passes API key when configured", async () => {
    process.env.HINDSIGHT_API_URL = "http://localhost:8888";
    process.env.HINDSIGHT_API_TOKEN = "my-token";

    await HindsightPlugin(mockPluginInput as any);

    expect(HindsightClient).toHaveBeenCalledWith({
      baseUrl: "http://localhost:8888",
      apiKey: "my-token",
    });
  });

  it("accepts plugin options", async () => {
    const result = await HindsightPlugin(mockPluginInput as any, {
      hindsightApiUrl: "http://example.com",
      bankId: "custom-bank",
    });

    expect(result.tool).toBeDefined();
    expect(HindsightClient).toHaveBeenCalledWith({
      baseUrl: "http://example.com",
      apiKey: undefined,
    });
  });
});

describe("HindsightPlugin state sharing", () => {
  beforeEach(() => {
    for (const key of Object.keys(process.env)) {
      if (key.startsWith("HINDSIGHT_")) delete process.env[key];
    }
    vi.clearAllMocks();
  });

  it("shares state across multiple plugin instantiations (sessions)", async () => {
    process.env.HINDSIGHT_API_URL = "http://localhost:8888";

    // Simulate two sessions calling the plugin (OpenCode instantiates per session)
    const result1 = await HindsightPlugin(mockPluginInput as any);
    const result2 = await HindsightPlugin(mockPluginInput as any);

    // Trigger session.created on session 1 — should track 'sess-A'
    await result1.event!({
      event: { type: "session.created", properties: { info: { id: "sess-A" } } },
    });

    // Session 2's system transform should see 'sess-A' because state is shared
    const output = { system: [] as string[] };
    await result2["experimental.chat.system.transform"]!(
      { sessionID: "sess-A", model: {} },
      output
    );

    // The recall was attempted (state was shared — sess-A was found in recalledSessions).
    // If state were per-instance, result2 would have an empty recalledSessions and skip recall.
    // result2 uses the second HindsightClient instance (index 1).
    const clientInstance = (HindsightClient as any).mock.instances[1];
    expect(clientInstance.recall).toHaveBeenCalled();
  });
});

describe("plugin default export", () => {
  it("default-exports the Plugin function itself", async () => {
    const mod = await import("./index.js");
    expect(typeof mod.default).toBe("function");
    // OpenCode iterates Object.entries(mod) and calls every export as a
    // Plugin factory, deduping by reference. The default export must be
    // the same reference as the named HindsightPlugin export to avoid
    // running the factory twice.
    expect(mod.default).toBe(mod.HindsightPlugin);
  });
});
