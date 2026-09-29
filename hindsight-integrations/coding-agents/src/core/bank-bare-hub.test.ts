import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { projectNameOf, resolveProjectRoot } from "./bank";

const fixtures: string[] = [];

/** The resolved root, or "" — these fixtures are all real repositories, so anything else fails. */
function rootOf(directory: string): string {
  const projectRoot = resolveProjectRoot(directory);
  return projectRoot.status === "resolved" ? projectRoot.root : "";
}

function git(cwd: string, args: string[]): string {
  return execFileSync("git", ["-C", cwd, ...args], { encoding: "utf8" }).trim();
}

function createBareHub(root: string, bareName: string): { hub: string; worktree: string } {
  const hub = join(root, bareName === ".bare" ? "hub" : "git-bare-hub");
  const bare = join(hub, bareName);
  const seed = join(root, `${bareName}-seed`);
  mkdirSync(hub, { recursive: true });
  git(root, ["init", "--bare", "-q", bare]);
  git(root, ["init", "-q", "-b", "main", seed]);
  git(seed, ["config", "user.email", "test@example.invalid"]);
  git(seed, ["config", "user.name", "Test"]);
  git(seed, ["commit", "--allow-empty", "-qm", "seed"]);
  git(seed, ["push", "-q", bare, "HEAD:refs/heads/main"]);
  writeFileSync(join(hub, ".git"), `gitdir: ./${bareName}\n`);
  const worktree = join(hub, "main");
  git(bare, ["worktree", "add", "-q", worktree, "main"]);
  return { hub, worktree };
}

describe("bare-hub project resolution", () => {
  afterEach(() => {
    for (const fixture of fixtures.splice(0)) rmSync(fixture, { recursive: true, force: true });
  });

  it.each([".bare", ".git-bare"])("uses the hub root for a %s layout", (bareName) => {
    const root = mkdtempSync(join(tmpdir(), "hs-bare-hub-"));
    fixtures.push(root);
    const { hub, worktree } = createBareHub(root, bareName);
    const canonicalHub = realpathSync(hub);

    expect(rootOf(hub)).toBe(canonicalHub);
    expect(rootOf(worktree)).toBe(canonicalHub);
    expect(projectNameOf(join(worktree, "nested"))).toBe(basename(canonicalHub));
  });

  it("preserves the directory name for a standalone bare repository", () => {
    const root = mkdtempSync(join(tmpdir(), "hs-standalone-bare-"));
    fixtures.push(root);
    const bare = join(root, "myproject.git");
    git(root, ["init", "--bare", "-q", bare]);

    expect(rootOf(bare)).toBe(realpathSync(bare));
    expect(projectNameOf(bare)).toBe("myproject.git");
  });

  it("keeps ordinary repositories unchanged", () => {
    const root = mkdtempSync(join(tmpdir(), "hs-ordinary-repo-"));
    fixtures.push(root);
    const repo = join(root, "myproject");
    git(root, ["init", "-q", "-b", "main", repo]);
    git(repo, ["config", "user.email", "test@example.invalid"]);
    git(repo, ["config", "user.name", "Test"]);
    git(repo, ["commit", "--allow-empty", "-qm", "seed"]);
    const worktree = join(root, "feature");
    git(repo, ["worktree", "add", "-q", "-b", "feature", worktree]);

    expect(rootOf(repo)).toBe(realpathSync(repo));
    expect(rootOf(worktree)).toBe(realpathSync(repo));
  });
});
