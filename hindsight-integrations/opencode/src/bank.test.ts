import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { deriveBankId, ensureBankMission } from "./bank.js";
import { makeConfig } from "./test-helpers.js";

describe("deriveBankId", () => {
  const originalEnv = { ...process.env };

  afterEach(() => {
    process.env = { ...originalEnv };
  });

  it("returns default bank name in static mode", () => {
    expect(deriveBankId(makeConfig(), "/home/user/project")).toBe("opencode");
  });

  it("returns configured bankId in static mode", () => {
    const config = makeConfig({ bankId: "my-bank" });
    expect(deriveBankId(config, "/home/user/project")).toBe("my-bank");
  });

  it("adds prefix in static mode", () => {
    const config = makeConfig({ bankIdPrefix: "dev", bankId: "my-bank" });
    expect(deriveBankId(config, "/home/user/project")).toBe("dev-my-bank");
  });

  it("composes from granularity fields in dynamic mode", () => {
    const config = makeConfig({
      dynamicBankId: true,
      dynamicBankGranularity: ["agent", "project"],
      agentName: "opencode",
    });
    expect(deriveBankId(config, "/home/user/my-project")).toBe("opencode::my-project");
  });

  it("uses default granularity when not specified", () => {
    const config = makeConfig({
      dynamicBankId: true,
      dynamicBankGranularity: [],
    });
    expect(deriveBankId(config, "/home/user/proj")).toBe("opencode::proj");
  });

  it("URL-encodes special characters", () => {
    const config = makeConfig({
      dynamicBankId: true,
      dynamicBankGranularity: ["project"],
    });
    expect(deriveBankId(config, "/home/user/my project")).toBe("my%20project");
  });

  it("uses channel/user from env vars", () => {
    process.env.HINDSIGHT_CHANNEL_ID = "slack-general";
    process.env.HINDSIGHT_USER_ID = "user123";
    const config = makeConfig({
      dynamicBankId: true,
      dynamicBankGranularity: ["agent", "channel", "user"],
    });
    expect(deriveBankId(config, "/home/user/proj")).toBe("opencode::slack-general::user123");
  });

  it("uses defaults for missing env vars", () => {
    delete process.env.HINDSIGHT_CHANNEL_ID;
    delete process.env.HINDSIGHT_USER_ID;
    const config = makeConfig({
      dynamicBankId: true,
      dynamicBankGranularity: ["channel", "user"],
    });
    expect(deriveBankId(config, "/home/user/proj")).toBe("default::anonymous");
  });

  it("adds prefix in dynamic mode", () => {
    const config = makeConfig({
      dynamicBankId: true,
      bankIdPrefix: "dev",
      dynamicBankGranularity: ["agent"],
    });
    expect(deriveBankId(config, "/home/user/proj")).toBe("dev-opencode");
  });
});

describe("ensureBankMission", () => {
  it("calls createBank on first use", async () => {
    const client = { createBank: vi.fn().mockResolvedValue({}) } as any;
    const missionsSet = new Set<string>();
    const config = makeConfig({ bankMission: "Test mission" });

    await ensureBankMission(client, "test-bank", config, missionsSet);

    expect(client.createBank).toHaveBeenCalledWith("test-bank", {
      reflectMission: "Test mission",
      retainMission: undefined,
    });
    expect(missionsSet.has("test-bank")).toBe(true);
  });

  it("skips if already set", async () => {
    const client = { createBank: vi.fn() } as any;
    const missionsSet = new Set(["test-bank"]);
    const config = makeConfig({ bankMission: "Test mission" });

    await ensureBankMission(client, "test-bank", config, missionsSet);

    expect(client.createBank).not.toHaveBeenCalled();
  });

  it("skips if no mission configured", async () => {
    const client = { createBank: vi.fn() } as any;
    const missionsSet = new Set<string>();
    const config = makeConfig({ bankMission: "" });

    await ensureBankMission(client, "test-bank", config, missionsSet);

    expect(client.createBank).not.toHaveBeenCalled();
  });

  it("does not throw on client error", async () => {
    const client = { createBank: vi.fn().mockRejectedValue(new Error("Network error")) } as any;
    const missionsSet = new Set<string>();
    const config = makeConfig({ bankMission: "Mission" });

    await expect(
      ensureBankMission(client, "test-bank", config, missionsSet)
    ).resolves.not.toThrow();
    expect(missionsSet.has("test-bank")).toBe(false);
  });

  it("passes retainMission when configured", async () => {
    const client = { createBank: vi.fn().mockResolvedValue({}) } as any;
    const missionsSet = new Set<string>();
    const config = makeConfig({ bankMission: "Reflect", retainMission: "Extract carefully" });

    await ensureBankMission(client, "test-bank", config, missionsSet);

    expect(client.createBank).toHaveBeenCalledWith("test-bank", {
      reflectMission: "Reflect",
      retainMission: "Extract carefully",
    });
  });
});
