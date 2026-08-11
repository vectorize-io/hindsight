import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { HindsightClient } from "./hindsight";
import { gitLogText, ingestGitLog, repoNameOf, retainCommit } from "./git";

let dir: string;

function initRepo(d: string): void {
  execFileSync("git", ["-C", d, "init", "-q"]);
  execFileSync("git", ["-C", d, "config", "user.email", "test@example.com"]);
  execFileSync("git", ["-C", d, "config", "user.name", "Test User"]);
}

afterEach(() => {
  if (dir) rmSync(dir, { recursive: true, force: true });
});

describe("gitLogText", () => {
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "hs-gitlog-"));
    initRepo(dir);
  });

  it("contains both commit subjects, no diff hunks, as a single string", () => {
    execFileSync("git", ["-C", dir, "commit", "--allow-empty", "-m", "feat: thing one"]);
    execFileSync("git", ["-C", dir, "commit", "--allow-empty", "-m", "fix: thing two"]);

    const text = gitLogText(dir, 10);

    expect(typeof text).toBe("string");
    expect(text).toContain("feat: thing one");
    expect(text).toContain("fix: thing two");
    expect(text).not.toContain("diff --git");
  });

  it("returns empty string for a repo with no commits", () => {
    expect(gitLogText(dir, 10)).toBe("");
  });
});

describe("ingestGitLog", () => {
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "hs-gitlog-ingest-"));
    initRepo(dir);
  });

  it("retains exactly one document with the aggregated commit-message history", async () => {
    execFileSync("git", ["-C", dir, "commit", "--allow-empty", "-m", "feat: thing one"]);
    execFileSync("git", ["-C", dir, "commit", "--allow-empty", "-m", "fix: thing two"]);

    const retainSpy = vi.fn().mockResolvedValue(undefined);
    const client = { retain: retainSpy, opIds: [] } as unknown as HindsightClient;

    const failures = await ingestGitLog(client, dir, { limit: 10 });

    expect(failures).toBe(0);
    expect(retainSpy).toHaveBeenCalledTimes(1);
    const [content, , documentId, tags, strategy] = retainSpy.mock.calls[0];
    expect(documentId).toBe(`gitlog:${repoNameOf(dir)}`);
    expect(tags).toContain("source:git");
    expect(tags).toContain("source:git-log");
    expect(strategy).toBe("gitlog");
    expect(content).toContain("feat: thing one");
  });

  it("does not call retain for a repo with no commits, and returns 0", async () => {
    const retainSpy = vi.fn().mockResolvedValue(undefined);
    const client = { retain: retainSpy, opIds: [] } as unknown as HindsightClient;

    const failures = await ingestGitLog(client, dir, { limit: 10 });

    expect(retainSpy).not.toHaveBeenCalled();
    expect(failures).toBe(0);
  });

  it("applies retain attribution to aggregated git history", async () => {
    execFileSync("git", ["-C", dir, "commit", "--allow-empty", "-m", "feat: attributed"]);
    const retain = vi.fn().mockResolvedValue(undefined);
    const client = { retain, opIds: [] } as unknown as HindsightClient;

    await ingestGitLog(client, dir, {
      limit: 10,
      stampFor: () => ({ tags: ["project:repo-a"], metadata: { project: "repo-a" } }),
    });

    expect(retain.mock.calls[0][3]).toEqual(
      expect.arrayContaining(["project:repo-a", "source:git", "source:git-log"])
    );
    expect(retain.mock.calls[0][5]).toEqual({ metadata: { project: "repo-a" } });
  });

  it("applies retain attribution to full commit documents with built-ins authoritative", async () => {
    execFileSync("git", ["-C", dir, "commit", "--allow-empty", "-m", "feat: full diff"]);
    const sha = execFileSync("git", ["-C", dir, "rev-parse", "HEAD"], { encoding: "utf8" }).trim();
    const retain = vi.fn().mockResolvedValue(undefined);
    const client = { retain, opIds: [] } as unknown as HindsightClient;

    await retainCommit(client, dir, sha, repoNameOf(dir), {
      tags: ["project:repo-a"],
      metadata: { project: "repo-a", source: "configured" },
    });

    expect(retain.mock.calls[0][3]).toEqual(["project:repo-a", "source:git"]);
    expect(retain.mock.calls[0][5].metadata).toMatchObject({
      project: "repo-a",
      source: "git",
      commit: sha,
    });
  });
});
