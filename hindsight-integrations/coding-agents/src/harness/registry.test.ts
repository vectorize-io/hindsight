import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { getHarness, HARNESS_NAMES, PLUGIN_ENTRYPOINTS } from "./registry";

describe("HARNESS_NAMES", () => {
  it("lists all registered harnesses", () => {
    expect(HARNESS_NAMES).toEqual(
      expect.arrayContaining([
        "opencode",
        "kilo",
        "cline-cli",
        "pi",
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
    expect(HARNESS_NAMES).toHaveLength(13);
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

  it("resolves pi and its Prime Agent fork as separate extension harnesses", async () => {
    for (const [name, entry] of [
      ["pi", /src\/pi\.ts/],
      ["prime-agent", /src\/prime-agent\.ts/],
    ] as const) {
      const adapter = await getHarness(name);
      expect(adapter.name).toBe(name);
      // Both drive the same shared adapter (harness/pi-extension.ts) but are wired by their own
      // entrypoint, so each must resolve to its own dist bundle — swapping them would silently
      // retain one host's sessions under the other's bank.
      expect(() => adapter.createRuntime({} as never)).toThrow(entry);
    }
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

/**
 * The harness name an entrypoint reports is a bare string literal that nothing else checks: it
 * selects the `harnesses.<name>` config section, feeds `{harness}` bank templating, and is stamped
 * on every document that host retains. A typo there ships green — the registry, the installer and
 * the control plane's logo map are three SEPARATE hand-maintained lists, so none of them notices
 * that the running plugin calls itself something else.
 *
 * So assert it over the whole family, enumerated from the registry rather than a fourth list: every
 * entrypoint the registry names must report the harness the registry maps it to.
 */
describe("every plugin entrypoint reports the harness the registry maps it to", () => {
  const PKG = fileURLToPath(new URL("../..", import.meta.url));

  /** The two idioms an entrypoint uses to name itself: the argument it hands the shared factory
   *  (opencode/Kilo via createPluginEntry, pi/Prime Agent via createPiExtension), or its own module
   *  constant (Cline and dsh, which build their runtime themselves). */
  const DECLARES_HARNESS =
    /(?:createPluginEntry|createPiExtension)\("([^"]+)"\)|const HARNESS = "([^"]+)"/;

  it.each(Object.entries(PLUGIN_ENTRYPOINTS))("%s (%s)", (harness, entry) => {
    const declaration = readFileSync(join(PKG, entry), "utf8").match(DECLARES_HARNESS);
    expect(declaration, `${entry} names no harness in a form this guard recognises`).not.toBeNull();
    expect(declaration![1] ?? declaration![2]).toBe(harness);
  });
});
