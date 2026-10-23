import { execFileSync } from "node:child_process";
import { chmodSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { HindsightClient, RetainOpts } from "./hindsight";
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

  // #5003: the server keeps unchanged chunks only when the text before them is unchanged, so a new
  // commit must append to the document, not shift it — including once the window is full.
  it("appends a new commit to the previous text, stepping the window by half the limit", () => {
    const subjects = (text: string) => text.match(/feat: c\d+/g);
    const texts = [""];
    for (let i = 1; i <= 9; i++) {
      commit(dir, `feat: c${i}`);
      texts.push(gitLogText(dir, 4));
    }
    expect(subjects(texts[4])).toEqual(["feat: c1", "feat: c2", "feat: c3", "feat: c4"]);
    expect(subjects(texts[5])).toEqual(["feat: c3", "feat: c4", "feat: c5"]);
    expect(subjects(texts[9])).toEqual(["feat: c7", "feat: c8", "feat: c9"]);
    for (const n of [1, 2, 3, 5, 7]) {
      expect(texts[n + 1].startsWith(`${texts[n]}\n\n---\n\n`)).toBe(true);
    }
  }, 30_000); // 27 git processes: slow under the parallel suite

  // #5269: `log.showSignature=true` printed signature checks into the parsed output.
  it("ignores the user's log.showSignature setting", () => {
    const key = join(dir, "key");
    execFileSync("ssh-keygen", ["-q", "-t", "ed25519", "-N", "", "-f", key]);
    for (const [k, v] of [
      ["gpg.format", "ssh"],
      ["user.signingkey", `${key}.pub`],
      ["commit.gpgsign", "true"],
      ["log.showSignature", "true"],
    ])
      execFileSync("git", ["-C", dir, "config", k, v]);
    commit(dir, "feat: signed");

    expect(gitLogNewestAuthorDate(dir)).toMatch(/^\d{4}-\d\d-\d\dT[\d:]+[+-Z][\d:]*$/);
    expect(gitLogText(dir, 10)).toMatch(/^[0-9a-f]+ \d{4}-\d\d-\d\d Test User\nfeat: signed$/);
  }, 30_000);

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

  it("says how many old commits a full write leaves out (#5708)", async () => {
    for (let i = 0; i < 5; i++) commit(dir, `feat: ${i}`);
    const client = { retain: vi.fn().mockResolvedValue(undefined), opIds: [] };
    const lines: string[] = [];
    await ingestGitLog(client as unknown as HindsightClient, dir, {
      limit: 4,
      log: (l) => lines.push(l),
    });
    // limit 4 moves the window in steps of 2: five commits keep the newest three.
    expect(lines).toContain(
      "[gitlog] dropping the 2 oldest commit(s): the document holds at most seedLimit (4)"
    );
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

describe("syncGitLog appends", () => {
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "hs-gitlog-append-"));
    initRepo(dir);
  });

  const head = (d: string) =>
    execFileSync("git", ["-C", d, "rev-parse", "HEAD"], { encoding: "utf8" }).trim();

  /** A bank that stores the git-log document the way the server does: a replace overwrites body
   *  and tags; an append joins the new content onto the stored body with "\n" and takes the new
   *  tags. `append: false` models a server that cannot append safely. */
  function server({ append = true } = {}) {
    const docs = new Map<string, { body: string; tags: string[] }>();
    const retain = vi.fn(
      async (
        content: string,
        _context: string,
        id: string,
        tags: string[],
        _strategy: string,
        opts: RetainOpts = {}
      ) => {
        const prior = docs.get(id);
        const body = opts.updateMode === "append" && prior ? `${prior.body}\n${content}` : content;
        docs.set(id, { body, tags });
      }
    );
    const client = {
      bank: "test-bank",
      retain,
      listDocumentIds: async (tag: string) =>
        new Set([...docs].filter(([, d]) => d.tags.includes(tag)).map(([id]) => id)),
      documentTags: async (id: string) => docs.get(id)?.tags,
      supportsAppendRetain: async () => append,
    } as unknown as HindsightClient;
    /** What the `i`th retain asked for beyond its positional arguments. */
    const optsOf = (i: number): RetainOpts => retain.mock.calls[i][5] ?? {};
    return { client, retain, docs, optsOf };
  }

  it("seeds a new document with a full write, oldest commit first", async () => {
    commit(dir, "feat: c1");
    commit(dir, "feat: c2");
    const { client, retain, optsOf } = server();

    await syncGitLog(client, dir, { limit: 10 });

    expect(retain).toHaveBeenCalledTimes(1);
    const [content] = retain.mock.calls[0];
    expect(optsOf(0).updateMode).toBeUndefined();
    expect(content.indexOf("feat: c1")).toBeLessThan(content.indexOf("feat: c2"));
  });

  it("appends only the commits since the recorded one, oldest first, then skips", async () => {
    commit(dir, "feat: c1");
    commit(dir, "feat: c2");
    const { client, retain, docs, optsOf } = server();
    await syncGitLog(client, dir, { limit: 10 });

    commit(dir, "feat: c3");
    commit(dir, "feat: c4");
    await syncGitLog(client, dir, { limit: 10 });

    expect(retain).toHaveBeenCalledTimes(2);
    const [content, , id, tags, strategy] = retain.mock.calls[1];
    expect(optsOf(1).updateMode).toBe("append");
    expect(optsOf(1).operationId).toMatch(/^[0-9a-f-]{36}$/);
    expect(strategy).toBe("gitlog");
    expect(content).not.toContain("feat: c1");
    expect(content).not.toContain("feat: c2");
    expect(content.indexOf("feat: c3")).toBeLessThan(content.indexOf("feat: c4"));
    expect(tags).toContain(`gitlog-head:${head(dir)}`);
    // The appended document reads exactly like a full write of the same history.
    expect(docs.get(id)?.body).toBe(gitLogText(dir, 10));

    await syncGitLog(client, dir, { limit: 10 });
    expect(retain).toHaveBeenCalledTimes(2);
  });

  it("gives a re-submitted append the same operation id, so the server applies it once", async () => {
    commit(dir, "feat: c1");
    const { client, retain, docs, optsOf } = server();
    await syncGitLog(client, dir, { limit: 10 });
    const seeded = structuredClone(docs.get(`gitlog:${repoNameOf(dir)}`)!);
    commit(dir, "feat: c2");

    await syncGitLog(client, dir, { limit: 10 });
    docs.set(`gitlog:${repoNameOf(dir)}`, seeded); // the append's outcome was never learned
    await syncGitLog(client, dir, { limit: 10 });

    expect(retain).toHaveBeenCalledTimes(3);
    expect(optsOf(2).operationId).toBe(optsOf(1).operationId);
  });

  it("writes the whole log again when the recorded commit is no longer an ancestor of HEAD", async () => {
    commit(dir, "feat: c1");
    commit(dir, "feat: c2");
    const { client, retain, optsOf } = server();
    await syncGitLog(client, dir, { limit: 10 });

    // Rewrite the recorded commit (an amend, as a rebase or force-push would).
    execFileSync("git", [
      "-C",
      dir,
      "commit",
      "-q",
      "--amend",
      "--allow-empty",
      "-m",
      "feat: c2 v2",
    ]);
    await syncGitLog(client, dir, { limit: 10 });

    expect(retain).toHaveBeenCalledTimes(2);
    const [content] = retain.mock.calls[1];
    expect(optsOf(1).updateMode).toBeUndefined();
    expect(content).toContain("feat: c1");
    expect(content).toContain("feat: c2 v2");
  });

  it("writes the whole log when the server cannot append safely", async () => {
    commit(dir, "feat: c1");
    const { client, retain, optsOf } = server({ append: false });
    await syncGitLog(client, dir, { limit: 10 });
    commit(dir, "feat: c2");

    await syncGitLog(client, dir, { limit: 10 });

    expect(retain).toHaveBeenCalledTimes(2);
    expect(optsOf(1).updateMode).toBeUndefined();
    expect(retain.mock.calls[1][0]).toContain("feat: c1");
  });

  it("writes the bounded log instead of appending more new commits than the limit", async () => {
    commit(dir, "feat: c1");
    const { client, retain, optsOf } = server();
    await syncGitLog(client, dir, { limit: 2 });
    for (const c of ["c2", "c3", "c4"]) commit(dir, `feat: ${c}`);

    await syncGitLog(client, dir, { limit: 2 });

    expect(retain).toHaveBeenCalledTimes(2);
    expect(optsOf(1).updateMode).toBeUndefined();
  });

  it("appends a merged branch's commits after the stored ones instead of rewriting the log", async () => {
    commit(dir, "feat: c1");
    const { client, docs, optsOf } = server();
    await syncGitLog(client, dir, { limit: 10 });
    const stored = docs.get(`gitlog:${repoNameOf(dir)}`)!.body;
    execFileSync("git", ["-C", dir, "checkout", "-q", "-b", "side"]);
    commit(dir, "feat: on the side branch");
    execFileSync("git", ["-C", dir, "checkout", "-q", "-"]);
    commit(dir, "feat: c2");
    execFileSync("git", ["-C", dir, "merge", "-q", "--no-ff", "--no-edit", "side"]);

    await syncGitLog(client, dir, { limit: 10 });

    expect(optsOf(1).updateMode).toBe("append");
    const body = docs.get(`gitlog:${repoNameOf(dir)}`)!.body;
    expect(body.startsWith(stored)).toBe(true);
    expect(body).toContain("feat: on the side branch");
    expect(body).toContain("feat: c2");
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

  // A partial clone would fetch an unknown sha from its remote, seconds per run to fail (#5606).
  it.skipIf(process.platform === "win32")("never lets git fetch a missing sha", () => {
    const bin = join(dir, "bin");
    execFileSync("mkdir", [bin]);
    writeFileSync(
      join(bin, "git"),
      `#!/bin/sh\necho "$GIT_NO_LAZY_FETCH" > "${bin}/env"\necho 0\n`
    );
    chmodSync(join(bin, "git"), 0o755);
    vi.stubEnv("PATH", `${bin}:${process.env.PATH}`);
    try {
      expect(commitsSince(dir, "a".repeat(40))).toBe(0);
    } finally {
      vi.unstubAllEnvs();
    }
    expect(readFileSync(join(bin, "env"), "utf8").trim()).toBe("1");
  });
});
