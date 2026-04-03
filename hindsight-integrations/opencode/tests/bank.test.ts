import { describe, test, expect } from "bun:test";
import { deriveBankId } from "../src/bank";
import type { HindsightConfig } from "../src/config";
import { loadConfig } from "../src/config";

function makeConfig(overrides: Partial<HindsightConfig> = {}): HindsightConfig {
  return { ...loadConfig(), ...overrides };
}

describe("deriveBankId", () => {
  test("returns static bank ID when dynamicBankId is false", () => {
    const config = makeConfig({ dynamicBankId: false, bankId: "my-bank" });
    expect(deriveBankId(config, "/Users/dev/project")).toBe("my-bank");
  });

  test("applies prefix to static bank ID", () => {
    const config = makeConfig({ dynamicBankId: false, bankId: "opencode", bankIdPrefix: "prod" });
    expect(deriveBankId(config, "/Users/dev/project")).toBe("prod-opencode");
  });

  test("derives dynamic bank ID from project directory", () => {
    const config = makeConfig({
      dynamicBankId: true,
      dynamicBankGranularity: ["project"],
    });
    expect(deriveBankId(config, "/Users/dev/my-project")).toBe("my-project");
  });

  test("derives dynamic bank ID from agent + project", () => {
    const config = makeConfig({
      dynamicBankId: true,
      dynamicBankGranularity: ["agent", "project"],
    });
    expect(deriveBankId(config, "/Users/dev/my-project")).toBe("opencode-my-project");
  });

  test("applies prefix to dynamic bank ID", () => {
    const config = makeConfig({
      dynamicBankId: true,
      dynamicBankGranularity: ["project"],
      bankIdPrefix: "staging",
    });
    expect(deriveBankId(config, "/Users/dev/project")).toBe("staging-project");
  });

  test("falls back to 'opencode' when no granularity fields match", () => {
    const config = makeConfig({
      dynamicBankId: true,
      dynamicBankGranularity: [],
    });
    expect(deriveBankId(config, "/some/dir")).toBe("opencode");
  });

  test("sanitizes special characters in directory names", () => {
    const config = makeConfig({
      dynamicBankId: true,
      dynamicBankGranularity: ["project"],
    });
    expect(deriveBankId(config, "/Users/dev/My Project (v2)")).toBe("my-project--v2-");
  });

  test("truncates and hashes very long bank IDs", () => {
    const config = makeConfig({
      dynamicBankId: true,
      dynamicBankGranularity: ["agent", "project"],
      bankIdPrefix: "very-long-prefix-that-takes-up-space",
    });
    const longDir = "/Users/dev/" + "a".repeat(100);
    const bankId = deriveBankId(config, longDir);
    expect(bankId.length).toBeLessThanOrEqual(64);
  });
});
