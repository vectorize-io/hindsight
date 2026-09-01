import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { deriveBankId, resolveProjectRoot } from "./bank";
import { probeGitLayout } from "./git-layout";

/** Real git, real worktrees — the layout reader has to agree with the tool that wrote the layout. */
function git(cwd: string, args: string[]): string {
  return execFileSync("git", ["-C", cwd, ...args], { encoding: "utf8" }).trim();
}

describe("git layout resolution", () => {
  let root: string;
  let repo: string;
  let worktree: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), "hs-layout-"));
    repo = join(root, "myrepo");
    git(root, ["init", "-q", "-b", "main", repo]);
    git(repo, ["config", "user.email", "test@example.invalid"]);
    git(repo, ["config", "user.name", "Test"]);
    git(repo, ["commit", "--allow-empty", "-qm", "seed"]);
    worktree = join(root, "myrepo-wt1"); // a SIBLING directory, as `git worktree add` makes them
    git(repo, ["worktree", "add", "-q", "-b", "wt1", worktree]);
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it("reports the repository's own git directory for a checkout", () => {
    expect(probeGitLayout(repo)).toEqual({
      status: "resolved",
      commonDir: join(realpathSync(repo), ".git"),
      bare: false,
    });
  });

  it("follows a linked worktree to the MAIN worktree's git directory", () => {
    expect(probeGitLayout(worktree)).toEqual({
      status: "resolved",
      commonDir: join(realpathSync(repo), ".git"),
      bare: false,
    });
  });

  it("answers the same from a nested subdirectory", () => {
    const nested = join(worktree, "a", "b");
    mkdirSync(nested, { recursive: true });
    expect(resolveProjectRoot(nested)).toEqual({ status: "resolved", root: realpathSync(repo) });
  });

  it("reports absence — not failure — outside any repository", () => {
    const plain = mkdtempSync(join(tmpdir(), "hs-plain-"));
    try {
      expect(probeGitLayout(plain)).toEqual({ status: "absent" });
    } finally {
      rmSync(plain, { recursive: true, force: true });
    }
  });

  it("resolves a worktree with no `git` binary reachable at all", () => {
    // The point of reading the layout: #3950 was a SPAWN failing (timeout, EAGAIN under load) and
    // being read as "not a repository". With no subprocess there is no such failure to have — an
    // empty PATH is the strongest available proof that none is attempted.
    const path = process.env.PATH;
    process.env.PATH = "";
    try {
      expect(deriveBankId({}, worktree)).toBe(`coding-agent::${basename(realpathSync(repo))}`);
    } finally {
      process.env.PATH = path;
    }
  });

  it("does not mistake a `.git` file with no gitdir pointer for a repository", () => {
    const junk = mkdtempSync(join(tmpdir(), "hs-junk-"));
    try {
      writeFileSync(join(junk, ".git"), "not a pointer\n");
      expect(probeGitLayout(junk)).toEqual({ status: "absent" });
    } finally {
      rmSync(junk, { recursive: true, force: true });
    }
  });
});

describe("a worktree whose administrative directory was pruned", () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), "hs-pruned-"));
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it("does not name the dangling gitdir (that basename IS the worktree)", () => {
    // `git worktree prune` after the checkout survived: the `.git` pointer still names
    // `<repo>/.git/worktrees/wt1`, whose basename is exactly the wrong bank id.
    const stale = join(root, "myrepo-wt1");
    mkdirSync(stale);
    writeFileSync(join(stale, ".git"), `gitdir: ${join(root, "myrepo/.git/worktrees/wt1")}\n`);
    expect(probeGitLayout(stale)).toEqual({ status: "absent" });
  });
});
