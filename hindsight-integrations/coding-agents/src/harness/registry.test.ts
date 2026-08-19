import { describe, expect, it } from "vitest";
import { getHarness, HARNESS_NAMES } from "./registry";

describe("HARNESS_NAMES", () => {
  it("lists all registered harnesses", () => {
    expect(HARNESS_NAMES).toEqual(
      expect.arrayContaining([
        "opencode",
        "kilo",
        "cline-cli",
        "prime-agent",
        "dsh",
        "claude-code",
        "cursor-cli",
        "codex",
        "antigravity-cli",
        "devin-cli",
        "copilot-cli",
        "grok-build",
      ])
    );
    expect(HARNESS_NAMES).toHaveLength(12);
  });
});

describe("getHarness", () => {
  it("resolves hook harnesses without touching the opencode adapter", async () => {
    for (const name of ["claude-code", "cursor-cli", "codex", "antigravity-cli", "devin-cli"]) {
      const adapter = await getHarness(name);
      expect(adapter.name).toBe(name);
      // Lightweight hook adapters have no persistent runtime — createRuntime always throws before
      // touching its argument, so a stand-in value is fine here.
      expect(() => adapter.createRuntime({} as never)).toThrow();
    }
  });

  it("resolves the opencode adapter by name", async () => {
    const adapter = await getHarness("opencode");
    expect(adapter.name).toBe("opencode");
    // Deliberate invariant: opencode via the registry is a no-runtime adapter (same shape as the
    // hook harnesses) — the real opencode runtime is built by src/index.ts importing opencodeAdapter
    // directly, bypassing this registry. Lock it so a future change can't silently make this look
    // functional.
    expect(() => adapter.createRuntime({} as never)).toThrow();
  });

  it("resolves Cline as a native-plugin harness rather than a hook binary", async () => {
    const adapter = await getHarness("cline-cli");
    expect(adapter.name).toBe("cline-cli");
    expect(() => adapter.createRuntime({} as never)).toThrow(/src\/cline\.ts/);
  });

  it("resolves DeepSeek Harness as a native Cordis-plugin harness", async () => {
    const adapter = await getHarness("dsh");
    expect(adapter.name).toBe("dsh");
    expect(() => adapter.createRuntime({} as never)).toThrow(/src\/dsh\.ts/);
  });

  it("rejects unknown harness names", async () => {
    await expect(getHarness("nope")).rejects.toThrow(/unknown harness/);
  });
});

/**
 * The registry and the installer are separate lists that must agree: `deepen` resolves a harness
 * through the registry, so an installer the registry doesn't know throws "unknown harness" and the
 * background git-diff enrichment dies. That is exactly how grok-build and copilot-cli shipped
 * broken in 0.0.1 — installable, but unusable by deepen.
 */
describe("registry covers every installable harness", () => {
  it("has no harness the installer wires but the registry rejects", async () => {
    const { INSTALLERS } = await import("../installer");
    const missing = INSTALLERS.map((i) => i.name).filter((n) => !HARNESS_NAMES.includes(n));
    expect(missing).toEqual([]);
  });
});
