import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { deriveBankId } from "./bank";

/**
 * Real git, real filesystem — deliberately NOT the mocked child_process of bank.test.ts, because
 * what is under test is how resolution behaves against paths that do and do not exist.
 *
 * A hook runs after the fact, so the directory it reports can already be gone: an ephemeral
 * worktree removed once the task finished, or a checkout moved mid-session (#3110).
 */
describe("project resolution when the working directory is gone", () => {
  let repo: string;

  beforeEach(() => {
    delete process.env.CLAUDE_PROJECT_DIR;
    repo = mkdtempSync(join(tmpdir(), "hs-gone-"));
    execFileSync("git", ["init", "-q"], { cwd: repo });
  });

  afterEach(() => {
    rmSync(repo, { recursive: true, force: true });
    delete process.env.CLAUDE_PROJECT_DIR;
  });

  it("names the repository rather than the deleted leaf directory", () => {
    // Never created: this is the state the hook actually observes once the worktree is removed.
    const gone = join(repo, ".agent", "worktrees", "agent-a33c4d63");
    expect(deriveBankId({}, gone, "claude-code")).toBe(`coding-agent::${basename(repo)}`);
  });

  it("keeps the historical answer while the directory still exists", () => {
    // The walk is a no-op for a live directory, so no existing bank moves.
    expect(deriveBankId({}, repo, "claude-code")).toBe(`coding-agent::${basename(repo)}`);
  });

  it("uses an exported project root when the walk lands outside the repository", () => {
    // A LINKED worktree is a sibling of the repo, not a child, so walking up leaves it entirely;
    // only the harness's own variable still names the project.
    process.env.CLAUDE_PROJECT_DIR = repo;
    const goneSibling = join(tmpdir(), "hs-gone-linked-worktree-never-created");
    expect(deriveBankId({}, goneSibling, "claude-code")).toBe(`coding-agent::${basename(repo)}`);
  });

  it("prefers a resolvable directory over the exported root", () => {
    const other = mkdtempSync(join(tmpdir(), "hs-other-"));
    execFileSync("git", ["init", "-q"], { cwd: other });
    process.env.CLAUDE_PROJECT_DIR = other;
    try {
      expect(deriveBankId({}, repo, "claude-code")).toBe(`coding-agent::${basename(repo)}`);
    } finally {
      rmSync(other, { recursive: true, force: true });
    }
  });

  it("never yields an empty project name", () => {
    // basename("/") is "", which produced bank ids like `coding-agent::` that name nothing.
    expect(deriveBankId({}, "/", "claude-code")).toBe("coding-agent::unknown");
    expect(deriveBankId({ bankIdTemplate: "{project}" }, "/", "claude-code")).toBe("unknown");
  });

  it("still yields a name for a directory that is not a repository", () => {
    const plain = mkdtempSync(join(tmpdir(), "hs-plain-"));
    try {
      expect(deriveBankId({}, plain, "claude-code")).toBe(`coding-agent::${basename(plain)}`);
    } finally {
      rmSync(plain, { recursive: true, force: true });
    }
  });

  it("keeps ONE bank when the agent navigates below a non-repository session root", () => {
    // #3563: no repo to resolve to, so the bank id used to follow the agent — `analysis` and
    // `evidence` each became a bank holding part of the SAME conversation.
    const plain = mkdtempSync(join(tmpdir(), "hs-plain-"));
    const sub = join(plain, "analysis", "evidence");
    mkdirSync(sub, { recursive: true });
    try {
      for (const cwd of [plain, join(plain, "analysis"), sub]) {
        expect(deriveBankId({}, cwd, "claude-code", plain)).toBe(
          `coding-agent::${basename(plain)}`
        );
      }
    } finally {
      rmSync(plain, { recursive: true, force: true });
    }
  });

  it("still names the repository the agent is IN, not the session root", () => {
    // The session root is a rescue for what git cannot name, never an override: a session started
    // in a plain directory that then works inside a repo still gets that repo's bank.
    const plain = mkdtempSync(join(tmpdir(), "hs-plain-"));
    try {
      expect(deriveBankId({}, repo, "claude-code", plain)).toBe(`coding-agent::${basename(repo)}`);
    } finally {
      rmSync(plain, { recursive: true, force: true });
    }
  });

  it("names the session's repository when the agent steps outside it", () => {
    // Harness-neutral counterpart of the CLAUDE_PROJECT_DIR rescue above: a `cd /tmp` to run
    // something must not rename the bank after the directory it landed in.
    const outside = mkdtempSync(join(tmpdir(), "hs-outside-"));
    try {
      expect(deriveBankId({}, outside, "codex", repo)).toBe(`coding-agent::${basename(repo)}`);
    } finally {
      rmSync(outside, { recursive: true, force: true });
    }
  });

  it("leaves {project} on the live working directory", () => {
    // {project} is documented as the working-directory basename — it is the escape hatch for
    // anyone who WANTS a bank per directory, so the session root must not reach it.
    const plain = mkdtempSync(join(tmpdir(), "hs-plain-"));
    const sub = join(plain, "analysis");
    mkdirSync(sub);
    try {
      expect(deriveBankId({ bankIdTemplate: "{project}" }, sub, "claude-code", plain)).toBe(
        "analysis"
      );
    } finally {
      rmSync(plain, { recursive: true, force: true });
    }
  });

  it("lets mapPathToBank keep overriding the session root", () => {
    const plain = mkdtempSync(join(tmpdir(), "hs-plain-"));
    const sub = join(plain, "analysis");
    mkdirSync(sub);
    try {
      expect(deriveBankId({ mapPathToBank: { [sub]: "pinned" } }, sub, "claude-code", plain)).toBe(
        "pinned"
      );
    } finally {
      rmSync(plain, { recursive: true, force: true });
    }
  });
});
