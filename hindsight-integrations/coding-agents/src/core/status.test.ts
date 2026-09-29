import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { syncStatus, DEEPEN_DIFF_TARGET, type StatusClient } from "./status";

/** Build a StatusClient stub; per-tag doc sets, overridable listPages/activeOperations. */
function stubClient(opts: {
  gitIds?: string[];
  chatIds?: string[];
  listDocumentIds?: StatusClient["listDocumentIds"];
  listPages?: StatusClient["listPages"];
  knowledgePagesSupported?: boolean;
  activeOperations?: StatusClient["activeOperations"];
}): StatusClient {
  return {
    listDocumentIds:
      opts.listDocumentIds ??
      (async (tag: string) =>
        new Set(tag === "source:git" ? (opts.gitIds ?? []) : (opts.chatIds ?? []))),
    listPages: opts.listPages ?? (async () => ({ items: [{ id: "p1", name: "Component map" }] })),
    knowledgePagesSupported: opts.knowledgePagesSupported,
    activeOperations: opts.activeOperations ?? (async () => 0),
  };
}

let dir: string | undefined;

afterEach(() => {
  if (dir) rmSync(dir, { recursive: true, force: true });
  dir = undefined;
});

describe("syncStatus", () => {
  it("synced: gitlog doc present + pages exist + zero active ops (no repoDir: any gitlog: doc counts)", async () => {
    const client = stubClient({ gitIds: ["gitlog:whatever-repo"] });
    const s = await syncStatus(client, "bank-1");
    expect(s).toEqual({
      bank: "bank-1",
      gitlogPresent: true,
      gitDiffDocs: 0,
      gitDiffTarget: null, // no repoDir passed
      surveyBaseline: null,
      surveyDocs: 0,
      surveyCommitsBehind: null,
      chatDocs: 0,
      pagesCount: 1,
      activeOps: 0,
      synced: true,
    });
  });

  it("not synced when no knowledge pages exist yet", async () => {
    const client = stubClient({
      gitIds: ["gitlog:repo"],
      listPages: async () => ({ items: [] }),
    });
    const s = await syncStatus(client, "bank-1");
    expect(s.pagesCount).toBe(0);
    expect(s.synced).toBe(false);
  });

  it("stays synced when an older server explicitly lacks knowledge pages", async () => {
    const client = stubClient({
      gitIds: ["gitlog:repo"],
      listPages: async () => {
        throw new Error("knowledge pages unavailable");
      },
      knowledgePagesSupported: false,
    });
    const s = await syncStatus(client, "bank-1");
    expect(s.pagesCount).toBe(0);
    expect(s.synced).toBe(true);
  });

  it("not synced while extraction operations are still active", async () => {
    const client = stubClient({ gitIds: ["gitlog:repo"], activeOperations: async () => 3 });
    const s = await syncStatus(client, "bank-1");
    expect(s.activeOps).toBe(3);
    expect(s.synced).toBe(false);
  });

  it("not synced when the gitlog seed document is missing (git: diffs alone don't count)", async () => {
    const client = stubClient({ gitIds: ["git:sha1", "git:sha2"] });
    const s = await syncStatus(client, "bank-1");
    expect(s.gitlogPresent).toBe(false);
    expect(s.synced).toBe(false);
  });

  it("fail-open: a listPages rejection yields pagesCount 0 (and thus not synced), no throw", async () => {
    const client = stubClient({
      gitIds: ["gitlog:repo"],
      listPages: async () => {
        throw new Error("pages endpoint down");
      },
    });
    const s = await syncStatus(client, "bank-1");
    expect(s.pagesCount).toBe(0);
    expect(s.synced).toBe(false);
  });

  it("fail-open: an activeOperations rejection yields activeOps null and does NOT block synced", async () => {
    const client = stubClient({
      gitIds: ["gitlog:repo"],
      activeOperations: async () => {
        throw new Error("ops endpoint down");
      },
    });
    const s = await syncStatus(client, "bank-1");
    expect(s.activeOps).toBeNull();
    // (activeOps ?? 0) === 0 — an unreportable count doesn't gate completion.
    expect(s.synced).toBe(true);
  });

  it("fail-open: a source:chat listDocumentIds rejection yields chatDocs 0, no throw", async () => {
    const client = stubClient({
      listDocumentIds: async (tag: string) => {
        if (tag === "source:chat") throw new Error("chat tag broken");
        return new Set(["gitlog:repo"]);
      },
    });
    const s = await syncStatus(client, "bank-1");
    expect(s.chatDocs).toBe(0);
    expect(s.synced).toBe(true);
  });

  it("counts git: diff docs and chat docs, ignoring other ids", async () => {
    const client = stubClient({
      gitIds: ["gitlog:repo", "git:aaa", "git:bbb", "git:ccc", "other:x"],
      chatIds: ["chat:s1", "chat:s2"],
    });
    const s = await syncStatus(client, "bank-1");
    expect(s.gitDiffDocs).toBe(3);
    expect(s.chatDocs).toBe(2);
  });

  it("gitDiffTarget is null for a non-git repoDir; gitlog match is by repo basename", async () => {
    dir = mkdtempSync(join(tmpdir(), "hs-status-nogit-"));
    const client = stubClient({ gitIds: [`gitlog:${basename(dir)}`] });
    const s = await syncStatus(client, "bank-1", dir);
    expect(s.gitDiffTarget).toBeNull();
    expect(s.gitlogPresent).toBe(true);
  });

  it("with a repoDir, only gitlog:<repoBasename> counts as the seed doc", async () => {
    dir = mkdtempSync(join(tmpdir(), "hs-status-nogit-"));
    const client = stubClient({ gitIds: ["gitlog:some-other-repo"] });
    const s = await syncStatus(client, "bank-1", dir);
    expect(s.gitlogPresent).toBe(false);
    expect(s.synced).toBe(false);
  });

  it("gitDiffTarget is min(DEEPEN_DIFF_TARGET, commit count) for a real git repo", async () => {
    const repo = mkdtempSync(join(tmpdir(), "hs-status-git-"));
    dir = repo; // module-scoped for afterEach cleanup; `repo` is the narrowed string
    execFileSync("git", ["-C", repo, "init", "-q"]);
    // The identity is passed per-command: a CI runner has no global user.name/user.email, so
    // committing without it fails with "empty ident name not allowed". A developer machine hides
    // this behind its own global git config, which is why it only ever broke in CI.
    const commit = (message: string) =>
      execFileSync("git", [
        "-C",
        repo,
        "-c",
        "user.email=status-test@example.test",
        "-c",
        "user.name=Status Test",
        "commit",
        "--allow-empty",
        "-m",
        message,
      ]);
    commit("one");
    commit("two");
    const client = stubClient({ gitIds: [`gitlog:${basename(repo)}`] });
    const s = await syncStatus(client, "bank-1", repo);
    expect(s.gitDiffTarget).toBe(Math.min(DEEPEN_DIFF_TARGET, 2));
    expect(s.gitDiffTarget).toBe(2);
    expect(s.gitlogPresent).toBe(true);
  });
});
