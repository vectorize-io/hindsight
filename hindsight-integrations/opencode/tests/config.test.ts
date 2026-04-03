import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { loadConfig } from "../src/config";

describe("loadConfig", () => {
  // Track env vars we add so we can clean them up without replacing process.env
  const addedKeys: string[] = [];

  function setEnv(key: string, value: string) {
    addedKeys.push(key);
    process.env[key] = value;
  }

  beforeEach(() => {
    // Clear any Hindsight env vars that may exist
    for (const key of Object.keys(process.env)) {
      if (key.startsWith("HINDSIGHT_")) {
        delete process.env[key];
      }
    }
  });

  afterEach(() => {
    // Remove only the keys we added
    for (const key of addedKeys) {
      delete process.env[key];
    }
    addedKeys.length = 0;
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
    setEnv("HINDSIGHT_API_URL", "http://custom:9999");
    setEnv("HINDSIGHT_AUTO_RETAIN", "false");
    setEnv("HINDSIGHT_RECALL_MAX_TOKENS", "2048");
    setEnv("HINDSIGHT_DEBUG", "true");
    setEnv("HINDSIGHT_BANK_ID", "my-bank");

    const config = loadConfig();
    expect(config.hindsightApiUrl).toBe("http://custom:9999");
    expect(config.autoRetain).toBe(false);
    expect(config.recallMaxTokens).toBe(2048);
    expect(config.debug).toBe(true);
    expect(config.bankId).toBe("my-bank");
  });

  test("boolean env vars handle various truthy values", () => {
    setEnv("HINDSIGHT_DEBUG", "1");
    expect(loadConfig().debug).toBe(true);

    process.env.HINDSIGHT_DEBUG = "TRUE";
    expect(loadConfig().debug).toBe(true);

    process.env.HINDSIGHT_DEBUG = "false";
    expect(loadConfig().debug).toBe(false);

    process.env.HINDSIGHT_DEBUG = "0";
    expect(loadConfig().debug).toBe(false);
  });

  test("integer env vars parse correctly", () => {
    setEnv("HINDSIGHT_API_PORT", "8888");
    expect(loadConfig().apiPort).toBe(8888);

    setEnv("HINDSIGHT_RECALL_MAX_TOKENS", "512");
    expect(loadConfig().recallMaxTokens).toBe(512);
  });

  test("invalid integer env vars are ignored", () => {
    setEnv("HINDSIGHT_API_PORT", "not-a-number");
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
