import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { hasGitHistory } from "./git";

let dir: string;

afterEach(() => {
  if (dir) rmSync(dir, { recursive: true, force: true });
});

describe("hasGitHistory", () => {
  it("true for a git repo with commits (this repo)", () => {
    expect(hasGitHistory(process.cwd())).toBe(true);
  });

  it("false for a fresh non-git directory", () => {
    dir = mkdtempSync(join(tmpdir(), "hs-nogit-"));
    expect(hasGitHistory(dir)).toBe(false);
  });
});
