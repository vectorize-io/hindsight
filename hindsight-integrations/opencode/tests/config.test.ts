import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { loadConfig } from "../src/config";

describe("loadConfig", () => {
  const originalEnv = { ...process.env };

  beforeEach(() => {
    // Clear Hindsight env vars
    for (const key of Object.keys(process.env)) {
      if (key.startsWith("HINDSIGHT_")) {
        delete process.env[key];
      }
    }
  });

  afterEach(() => {
    process.env = { ...originalEnv };
  });

  test("returns defaults when no config file or env vars", () => {
    const config = loadConfig();
    expect(config.autoRetain).toBe(true);
    expect(config.autoRecall).toBe(true);
    expect(config.bankId).toBe("opencode");
    expect(config.retainContext).toBe("opencode");
    expect(config.apiPort).toBe(9077);
    expect(config.debug).toBe(false);
    expect(config.retainSkipSubagent).toBe(true);
    expect(config.recallBudget).toBe("mid");
    expect(config.dynamicBankId).toBe(false);
  });

  test("env vars override defaults", () => {
    process.env.HINDSIGHT_API_URL = "http://custom:9999";
    process.env.HINDSIGHT_AUTO_RETAIN = "false";
    process.env.HINDSIGHT_RECALL_MAX_TOKENS = "2048";
    process.env.HINDSIGHT_DEBUG = "true";
    process.env.HINDSIGHT_BANK_ID = "my-bank";

    const config = loadConfig();
    expect(config.hindsightApiUrl).toBe("http://custom:9999");
    expect(config.autoRetain).toBe(false);
    expect(config.recallMaxTokens).toBe(2048);
    expect(config.debug).toBe(true);
    expect(config.bankId).toBe("my-bank");
  });

  test("boolean env vars handle various truthy values", () => {
    process.env.HINDSIGHT_DEBUG = "1";
    expect(loadConfig().debug).toBe(true);

    process.env.HINDSIGHT_DEBUG = "TRUE";
    expect(loadConfig().debug).toBe(true);

    process.env.HINDSIGHT_DEBUG = "false";
    expect(loadConfig().debug).toBe(false);

    process.env.HINDSIGHT_DEBUG = "0";
    expect(loadConfig().debug).toBe(false);
  });

  test("integer env vars parse correctly", () => {
    process.env.HINDSIGHT_API_PORT = "8888";
    expect(loadConfig().apiPort).toBe(8888);

    process.env.HINDSIGHT_RECALL_MAX_TOKENS = "512";
    expect(loadConfig().recallMaxTokens).toBe(512);
  });

  test("invalid integer env vars are ignored", () => {
    process.env.HINDSIGHT_API_PORT = "not-a-number";
    expect(loadConfig().apiPort).toBe(9077); // default
  });

  test("bank mission has sensible default", () => {
    const config = loadConfig();
    expect(config.bankMission).toContain("coding sessions");
    expect(config.bankMission).toContain("architectural decisions");
  });

  test("recall prompt preamble is set", () => {
    const config = loadConfig();
    expect(config.recallPromptPreamble).toContain("memories");
  });
});
