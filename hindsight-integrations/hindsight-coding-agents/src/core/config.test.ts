import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { loadConfig } from "./config";

let root: string;
let globalCfg: string;

function writeJson(path: string, value: unknown): void {
  mkdirSync(join(path, ".."), { recursive: true });
  writeFileSync(path, JSON.stringify(value));
}

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-cfg-"));
  globalCfg = join(root, "global.json");
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

describe("loadConfig layering", () => {
  it("missing files yield defaults", () => {
    const cfg = loadConfig({ path: join(root, "nope.json") });
    expect(cfg.apiUrl).toBe("http://localhost:8888");
    expect(cfg.bankId).toBeUndefined();
    expect(cfg.disabled).toBe(false);
  });

  it("malformed global file falls back to defaults with a warning", () => {
    writeFileSync(globalCfg, "{not json");
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    const cfg = loadConfig({ path: globalCfg });
    expect(cfg.apiUrl).toBe("http://localhost:8888");
    expect(err).toHaveBeenCalledOnce();
    err.mockRestore();
  });

  it("applies the requesting harness's section over the base", () => {
    writeJson(globalCfg, {
      apiUrl: "http://x:1",
      bankId: "shared",
      harnesses: {
        "claude-code": { bankId: "claude-bank" },
        opencode: { disabled: true },
      },
    });
    expect(loadConfig({ path: globalCfg, harness: "claude-code" }).bankId).toBe("claude-bank");
    expect(loadConfig({ path: globalCfg, harness: "claude-code" }).apiUrl).toBe("http://x:1");
    expect(loadConfig({ path: globalCfg, harness: "opencode" }).disabled).toBe(true);
    expect(loadConfig({ path: globalCfg, harness: "opencode" }).bankId).toBe("shared");
    expect(loadConfig({ path: globalCfg }).bankId).toBe("shared"); // no harness: base only
  });

  it("project config layers over global, its harness section over both", () => {
    writeJson(globalCfg, { apiUrl: "http://global:1", bankId: "global-bank" });
    const proj = join(root, "repo");
    writeJson(join(proj, ".hindsight", "coding-agent.json"), {
      bankId: "proj-bank",
      harnesses: { "cursor-cli": { bankId: "proj-cursor-bank" } },
    });
    const base = loadConfig({ path: globalCfg, projectDir: proj });
    expect(base.apiUrl).toBe("http://global:1"); // inherited from global
    expect(base.bankId).toBe("proj-bank"); // project overrides
    expect(loadConfig({ path: globalCfg, projectDir: proj, harness: "cursor-cli" }).bankId).toBe(
      "proj-cursor-bank"
    );
  });

  it("finds the project config by walking UP from a nested subdirectory", () => {
    const proj = join(root, "repo");
    writeJson(join(proj, ".hindsight", "coding-agent.json"), { bankId: "walked-up" });
    const deep = join(proj, "a", "b", "c");
    mkdirSync(deep, { recursive: true });
    expect(loadConfig({ path: globalCfg, projectDir: deep }).bankId).toBe("walked-up");
  });

  it("the NEAREST project config wins over an ancestor's", () => {
    const outer = join(root, "repo");
    const inner = join(outer, "packages", "pkg");
    writeJson(join(outer, ".hindsight", "coding-agent.json"), { bankId: "outer" });
    writeJson(join(inner, ".hindsight", "coding-agent.json"), { bankId: "inner" });
    expect(loadConfig({ path: globalCfg, projectDir: join(inner, "src") }).bankId).toBe("inner");
    expect(loadConfig({ path: globalCfg, projectDir: outer }).bankId).toBe("outer");
  });

  it("gitSync merges field-wise across layers", () => {
    writeJson(globalCfg, { gitSync: { enabled: true, ref: "origin/main" } });
    const proj = join(root, "repo");
    writeJson(join(proj, ".hindsight", "coding-agent.json"), { gitSync: { ref: "origin/dev" } });
    const cfg = loadConfig({ path: globalCfg, projectDir: proj });
    expect(cfg.gitSync.enabled).toBe(true); // kept from global
    expect(cfg.gitSync.ref).toBe("origin/dev"); // overridden by project
  });

  it("legacy string signature still works as the global path", () => {
    writeJson(globalCfg, { bankId: "legacy" });
    expect(loadConfig(globalCfg).bankId).toBe("legacy");
  });
});
