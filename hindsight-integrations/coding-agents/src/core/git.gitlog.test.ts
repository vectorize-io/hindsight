import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { HindsightClient } from "./hindsight";
import {
  commitsSince,
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

function commit(d: string, message: string): void {
  execFileSync("git", ["-C", d, "commit", "-q", "--allow-empty", "-m", message]);
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

  /** A bank that keeps what the git-log retain writes (document id -> tags) and answers the two
   *  reads the freshness check makes. */
  function bank() {
    const docs = new Map<string, string[]>();
    const retain = vi.fn(async (_content: string, _context: string, id: string, tags: string[]) => {
      docs.set(id, tags);
    });
    const client = {
      retain,
      opIds: [],
      listDocumentIds: async (tag: string) =>
        new Set([...docs].filter(([, tags]) => tags.includes(tag)).map(([id]) => id)),
      documentTags: async (id: string) => docs.get(id),
    } as unknown as HindsightClient;
    return { client, retain, docs };
  }

  it("does not re-send the log when switching back to a worktree behind the last writer (#4661)", async () => {
    commit(dir, "feat: shared history");
    const ahead = join(dir, "wt");
    execFileSync("git", ["-C", dir, "worktree", "add", "-q", "-b", "feature", ahead]);
    commit(ahead, "feat: only on the feature branch");
    const { client, retain } = bank();

    for (const worktree of [dir, ahead, dir, ahead]) {
      await syncGitLog(client, worktree, { limit: 10 });
    }

    // Both worktrees write the one canonical document. The feature worktree's copy holds every
    // commit the behind one would send, so only the first visit to each writes it — main re-sent
    // it on every switch.
    expect(retain.mock.calls.map((call) => call[2])).toEqual([
      `gitlog:${repoNameOf(dir)}`,
      `gitlog:${repoNameOf(dir)}`,
    ]);
    expect(retain.mock.calls[1][0]).toContain("feat: only on the feature branch");
  });

  it("still re-sends the log from a worktree with a commit the document lacks", async () => {
    commit(dir, "feat: shared history");
    const other = join(dir, "wt");
    execFileSync("git", ["-C", dir, "worktree", "add", "-q", "-b", "feature", other]);
    commit(other, "feat: only on the feature branch");
    commit(dir, "fix: only on the main branch");
    const { client, retain } = bank();

    for (const worktree of [dir, other, dir]) {
      await syncGitLog(client, worktree, { limit: 10 });
    }

    // The worktrees diverged: each log has a commit the other lacks, so every switch changes what
    // the one document must hold.
    expect(retain).toHaveBeenCalledTimes(3);
    expect(retain.mock.calls[2][0]).toContain("fix: only on the main branch");
  });

  it("re-sends the log when the recorded commit is unknown to this clone", async () => {
    commit(dir, "feat: current repo");
    const { client, retain, docs } = bank();
    // A same-named repository elsewhere, or a rebase that dropped the commit.
    docs.set(`gitlog:${repoNameOf(dir)}`, ["source:git-log", `gitlog-head:${"f".repeat(40)}`]);

    await syncGitLog(client, dir, { limit: 10 });

    expect(retain).toHaveBeenCalledTimes(1);
  });
});

describe("commitsSince", () => {
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "hs-commits-since-"));
    initRepo(dir);
  });

  it("counts the commits on HEAD that a full sha lacks", () => {
    commit(dir, "feat: one");
    const first = execFileSync("git", ["-C", dir, "rev-parse", "HEAD"], { encoding: "utf8" });
    commit(dir, "feat: two");

    expect(commitsSince(dir, first.trim())).toBe(1);
  });

  it("never hands git a value from the bank that is not a full sha", () => {
    commit(dir, "feat: one");
    const planted = join(dir, "planted");

    // A tag or document id starting with "-" would reach `git rev-list` as an option, and
    // `--output=<path>..HEAD` creates that file before git rejects the command.
    expect(commitsSince(dir, `--output=${planted}`)).toBeNull();
    expect(existsSync(`${planted}..HEAD`)).toBe(false);
  });
});
