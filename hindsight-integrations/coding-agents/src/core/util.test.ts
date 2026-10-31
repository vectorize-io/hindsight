import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { MIN_IDEMPOTENT_RETAIN_VERSION } from "./hindsight";
import { binOnPath, semverGte } from "./util";

describe("binOnPath", () => {
  let dir: string;
  const file = (name: string, mode = 0o755) => {
    writeFileSync(join(dir, name), "");
    chmodSync(join(dir, name), mode);
  };
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "binonpath-"));
  });
  afterEach(() => rmSync(dir, { recursive: true, force: true }));

  // #5710: npm puts an extensionless `#!/bin/sh` shim, a `.cmd` and a `.ps1` on PATH; only the
  // native `.exe` can be spawned without a shell.
  it("Windows: resolves a bare name through PATHEXT to the full path, never the bare shim", () => {
    const win = { platform: "win32" as const, env: { PATH: dir, PATHEXT: ".COM;.EXE;.BAT;.CMD" } };
    file("codex");
    file("codex.cmd");
    expect(binOnPath("codex", { ...win, exts: [".exe"] })).toBeUndefined();
    expect(binOnPath("codex", win)).toBe(join(dir, "codex.cmd")); // npx/npm run through a shell
    file("codex.exe");
    expect(binOnPath("codex", { ...win, exts: [".exe"] })).toBe(join(dir, "codex.exe"));
  });

  it("a directory is never a match (X_OK accepts one)", () => {
    mkdirSync(join(dir, "agy.exe"));
    expect(binOnPath("agy", { platform: "win32", env: { PATH: dir } })).toBeUndefined();
    expect(binOnPath(join(dir, "agy.exe"))).toBeUndefined();
  });

  it("POSIX: a bare name needs the exec bit, and no extension is tried", () => {
    const posix = { platform: "linux" as const, env: { PATH: dir } };
    file("plain", 0o644);
    file("codex.exe");
    expect(binOnPath("plain", posix)).toBeUndefined();
    expect(binOnPath("codex", posix)).toBeUndefined();
    file("codex");
    expect(binOnPath("codex", posix)).toBe(join(dir, "codex"));
    expect(binOnPath(join(dir, "codex"), posix)).toBe(join(dir, "codex"));
  });
});

describe("semverGte", () => {
  it("compares numerically, not lexically", () => {
    expect(semverGte("0.10.0", "0.9.0")).toBe(true);
    expect(semverGte("0.9.0", "0.10.0")).toBe(false);
  });

  it("decides the retain-idempotency floor correctly", () => {
    // Pinned to the real constant: operation_id first shipped in v0.8.6 (#2937/#2947).
    const floor = MIN_IDEMPOTENT_RETAIN_VERSION;
    expect(floor).toBe("0.8.6");
    expect(semverGte("0.8.6", floor)).toBe(true);
    expect(semverGte("0.8.5", floor)).toBe(false); // one patch short
    expect(semverGte("0.9.0", floor)).toBe(true);
    expect(semverGte("1.0.0", floor)).toBe(true);
  });

  it("treats an unknown version as too old — a capability probe must fail closed", () => {
    expect(semverGte(undefined, "0.8.6")).toBe(false);
    expect(semverGte("", "0.8.6")).toBe(false);
    expect(semverGte("dev", "0.8.6")).toBe(false);
  });

  it("compares by the numeric part of a pre-release or build-tagged version", () => {
    expect(semverGte("0.9.0rc1", "0.8.6")).toBe(true);
    expect(semverGte("0.8.5+dev", "0.8.6")).toBe(false);
  });

  it("treats a missing patch component as zero", () => {
    expect(semverGte("0.9", "0.8.6")).toBe(true);
    expect(semverGte("0.8", "0.8.6")).toBe(false);
  });
});
