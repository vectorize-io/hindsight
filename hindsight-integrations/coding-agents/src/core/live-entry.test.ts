import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// CONFIG_PATH is read when config.ts loads, so each test points HINDSIGHT_CONFIG at its own file
// and imports a fresh copy.
let root: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-live-entry-"));
  vi.resetModules();
});

afterEach(() => {
  delete process.env.HINDSIGHT_CONFIG;
  rmSync(root, { recursive: true, force: true });
});

async function liveConfigWith(config: object) {
  process.env.HINDSIGHT_CONFIG = join(root, "coding-agent.json");
  writeFileSync(process.env.HINDSIGHT_CONFIG, JSON.stringify(config));
  const { liveConfig } = await import("./live-entry");
  return liveConfig("claude-code", root);
}

describe("liveConfig: the per-repo override decides whether this session is live", () => {
  it("is live by default on Claude Code", async () => {
    expect((await liveConfigWith({ bankId: "b1" }))?.bankId).toBe("b1");
  });

  it("a repo set to reflect is not live, though the global default does", async () => {
    expect(
      await liveConfigWith({ bankId: "b1", banks: { b1: { autoInject: "reflect" } } })
    ).toBeNull();
  });

  it("a repo set to live is live, though the global setting is reflect", async () => {
    const got = await liveConfigWith({
      bankId: "b1",
      autoInject: "reflect",
      banks: { b1: { autoInject: "live" } },
    });
    expect(got?.cfg.autoInject).toBe("live");
  });

  it("a disabled repo is not live", async () => {
    expect(await liveConfigWith({ bankId: "b1", banks: { b1: { disabled: true } } })).toBeNull();
  });
});

describe("every live harness is wired to its session", () => {
  it("has a PostToolUse live hook and a Stop hook that ends the session's turn", async () => {
    const { LIVE_HARNESSES } = await import("./config");
    const { SESSION_HARNESSES } = await import("./session-events");
    const { HOOK_HARNESSES } = await import("../harness/hook-lifecycle");
    for (const harness of LIVE_HARNESSES) {
      const spec = HOOK_HARNESSES[harness as keyof typeof HOOK_HARNESSES];
      expect(spec, harness).toBeDefined();
      expect(
        spec.additionalHooks?.map((h) => h.entry),
        harness
      ).toContain(SESSION_HARNESSES[harness].entry);
      // Without it the Stop hook writes the transcript itself next to the session's own save.
      expect(spec.retain?.harness, harness).toBe(harness);
    }
  });
});
