import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { HindsightClient } from "./hindsight";
import {
  gitLogNewestAuthorDate,
  gitLogText,
  ingestGitLog,
  repoNameOf,
  retainCommit,
  syncGitLog,
} from "./git";

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

  it("gitLogNewestAuthorDate returns null for a repo with no commits", () => {
    expect(gitLogNewestAuthorDate(dir)).toBeNull();
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
    expect(retain.mock.calls[0][5]).toMatchObject({ metadata: { project: "repo-a" } });
  });

  it("timestamps the aggregated document with the newest commit's author date", async () => {
    execFileSync("git", ["-C", dir, "commit", "--allow-empty", "-m", "feat: older"], {
      env: { ...process.env, GIT_AUTHOR_DATE: "2024-01-02T03:04:05+00:00" },
    });
    execFileSync("git", ["-C", dir, "commit", "--allow-empty", "-m", "feat: newest"], {
      env: { ...process.env, GIT_AUTHOR_DATE: "2024-03-04T05:06:07+00:00" },
    });
    const retain = vi.fn().mockResolvedValue(undefined);
    const client = { retain, opIds: [] } as unknown as HindsightClient;

    await ingestGitLog(client, dir, { limit: 10 });

    expect(new Date(retain.mock.calls[0][5].timestamp as string).toISOString()).toBe(
      "2024-03-04T05:06:07.000Z"
    );
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

describe("syncGitLog", () => {
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "hs-gitlog-sync-"));
    initRepo(dir);
  });

  it("never enumerates or deletes foreign git-log documents from a shared bank", async () => {
    execFileSync("git", ["-C", dir, "commit", "--allow-empty", "-m", "feat: current repo"]);
    const listDocumentIds = vi.fn(
      async (_tag: string, _tagsMatch?: "all" | "all_strict") => new Set(["gitlog:foreign-repo"])
    );
    const retain = vi.fn().mockResolvedValue(undefined);
    const deleteDocument = vi.fn().mockResolvedValue(undefined);
    const client = {
      listDocumentIds,
      retain,
      deleteDocument,
      opIds: [],
    } as unknown as HindsightClient;

    const failures = await syncGitLog(client, dir, { limit: 10 });

    expect(failures).toBe(0);
    expect(retain).toHaveBeenCalledTimes(1);
    expect(listDocumentIds).toHaveBeenCalledTimes(1);
    expect(listDocumentIds.mock.calls[0][0]).toMatch(/^gitlog-head:/);
    expect(listDocumentIds.mock.calls[0][1]).toBe("all_strict");
    expect(listDocumentIds).not.toHaveBeenCalledWith("source:git-log");
    expect(deleteDocument).not.toHaveBeenCalled();
  });

  it("skips the upsert only when this repository's canonical document has the current HEAD tag", async () => {
    execFileSync("git", ["-C", dir, "commit", "--allow-empty", "-m", "feat: current repo"]);
    const listDocumentIds = vi.fn(
      async (_tag: string, _tagsMatch?: "all" | "all_strict") =>
        new Set([`gitlog:${repoNameOf(dir)}`])
    );
    const retain = vi.fn().mockResolvedValue(undefined);
    const client = { listDocumentIds, retain, opIds: [] } as unknown as HindsightClient;

    const failures = await syncGitLog(client, dir, { limit: 10 });

    expect(failures).toBe(0);
    expect(listDocumentIds.mock.calls[0][0]).toMatch(/^gitlog-head:/);
    expect(listDocumentIds.mock.calls[0][1]).toBe("all_strict");
    expect(retain).not.toHaveBeenCalled();
  });
});
