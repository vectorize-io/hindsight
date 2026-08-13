import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { deriveBankId, isOptedIn } from "./bank";
import { applyBankConfig, resolveConfig } from "./config";

/**
 * Opt-in-only: memory runs in declared projects and nowhere else. Real directories, because the
 * question is entirely about how a working directory relates to what was configured.
 */
describe("optInOnly", () => {
  let root: string;
  let approved: string;
  let other: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), "hs-optin-"));
    approved = join(root, "work", "client-x");
    other = join(root, "scratch", "throwaway");
    for (const d of [approved, other]) {
      mkdirSync(d, { recursive: true });
      execFileSync("git", ["init", "-q"], { cwd: d });
    }
  });

  afterEach(() => rmSync(root, { recursive: true, force: true }));

  it("allows everything when it is off — the zero-setup default is unchanged", () => {
    const cfg = resolveConfig({});
    expect(isOptedIn(cfg, approved)).toBe(true);
    expect(isOptedIn(cfg, other)).toBe(true);
  });

  it("approves a listed directory and the repos beneath it", () => {
    const cfg = resolveConfig({ optInOnly: true, optInPaths: [join(root, "work")] });
    expect(isOptedIn(cfg, approved)).toBe(true);
    expect(isOptedIn(cfg, join(approved, "src", "deep"))).toBe(true);
    expect(isOptedIn(cfg, other)).toBe(false);
  });

  it("leaves an approved project its own dynamic bank — approving names nothing", () => {
    // The whole point of not reusing mapPathToBank: you approve a tree, each repo still gets its
    // own bank rather than being merged into one.
    const cfg = resolveConfig({ optInOnly: true, optInPaths: [join(root, "work")] });
    expect(deriveBankId(cfg, approved, "codex")).toBe("coding-agent::client-x");
  });

  it("treats a mapPathToBank entry as opted in", () => {
    // Routing a path to a named bank is already a deliberate declaration of that project.
    const cfg = resolveConfig({ optInOnly: true, mapPathToBank: { [approved]: "client-x" } });
    expect(isOptedIn(cfg, approved)).toBe(true);
    expect(isOptedIn(cfg, other)).toBe(false);
  });

  it("carries path approval and bank mapping from a checkout to its linked worktree", () => {
    const worktree = join(root, "external-worktrees", "client-x");
    execFileSync(
      "git",
      [
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-qm",
        "seed",
      ],
      { cwd: approved }
    );
    mkdirSync(join(root, "external-worktrees"), { recursive: true });
    execFileSync("git", ["worktree", "add", "-q", "-b", "linked", worktree], { cwd: approved });

    const checkoutParent = join(realpathSync(root), "work");
    const byPath = resolveConfig({ optInOnly: true, optInPaths: [checkoutParent] });
    expect(isOptedIn(byPath, worktree)).toBe(true);

    const byMap = resolveConfig({
      optInOnly: true,
      mapPathToBank: { [checkoutParent]: "client-x" },
    });
    expect(isOptedIn(byMap, worktree)).toBe(true);
    expect(deriveBankId(byMap, worktree, "codex")).toBe("client-x");

    const nested = join(worktree, "src", "deep");
    mkdirSync(nested, { recursive: true });
    expect(isOptedIn(byMap, nested)).toBe(true);
    expect(deriveBankId(byMap, nested, "codex")).toBe("client-x");

    rmSync(nested, { recursive: true });
    expect(isOptedIn(byMap, nested)).toBe(true);
    expect(deriveBankId(byMap, nested, "codex")).toBe("client-x");

    const worktreeOverride = resolveConfig({
      optInOnly: true,
      mapPathToBank: {
        [checkoutParent]: "client-x",
        [worktree]: "client-x-experiment",
      },
    });
    expect(deriveBankId(worktreeOverride, worktree, "codex")).toBe("client-x-experiment");
  });

  it("keeps a linked worktree denied when its checkout is outside every approved path", () => {
    const worktree = join(root, "external-worktrees", "throwaway");
    execFileSync(
      "git",
      [
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-qm",
        "seed",
      ],
      { cwd: other }
    );
    mkdirSync(join(root, "external-worktrees"), { recursive: true });
    execFileSync("git", ["worktree", "add", "-q", "-b", "linked", worktree], { cwd: other });

    const cfg = resolveConfig({
      optInOnly: true,
      mapPathToBank: { [join(realpathSync(root), "work")]: "client-x" },
    });
    expect(isOptedIn(cfg, worktree)).toBe(false);
  });

  it("does not carry checkout approval to linked worktrees when resolution is disabled", () => {
    const worktree = join(root, "external-worktrees", "client-x");
    execFileSync(
      "git",
      [
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-qm",
        "seed",
      ],
      { cwd: approved }
    );
    mkdirSync(join(root, "external-worktrees"), { recursive: true });
    execFileSync("git", ["worktree", "add", "-q", "-b", "linked", worktree], { cwd: approved });

    const cfg = resolveConfig({
      optInOnly: true,
      optInPaths: [join(realpathSync(root), "work")],
      mapPathToBank: { [join(realpathSync(root), "work")]: "client-x" },
      resolveWorktrees: false,
    });
    expect(isOptedIn(cfg, worktree)).toBe(false);
    expect(deriveBankId(cfg, worktree, "codex")).toBe("coding-agent::client-x");
  });

  it("does not let a bare bankId approve anything", () => {
    // It names a bank, not a project, so it cannot say which work may be remembered. A privacy
    // switch fails closed.
    const cfg = resolveConfig({ optInOnly: true, bankId: "shared" });
    expect(isOptedIn(cfg, approved)).toBe(false);
  });

  it("renders an unlisted project inert through the gate every entry point already checks", () => {
    const cfg = resolveConfig({ optInOnly: true, optInPaths: [approved] });
    const inert = applyBankConfig(cfg, deriveBankId(cfg, other, "codex"), other);
    expect(inert.cfg.disabled).toBe(true);

    const live = applyBankConfig(cfg, deriveBankId(cfg, approved, "codex"), approved);
    expect(live.cfg.disabled).toBe(false);
    expect(live.bankId).toBe("coding-agent::client-x");
  });

  it("stays inert when no directory is known and the switch is on", () => {
    const cfg = resolveConfig({ optInOnly: true, optInPaths: [approved] });
    expect(isOptedIn(cfg, "")).toBe(false);
  });

  it("ignores a banks.<id> section trying to set it — approval is decided before that runs", () => {
    const cfg = resolveConfig({
      optInOnly: true,
      optInPaths: [approved],
      banks: { "coding-agent::throwaway": { optInOnly: false } as never },
    });
    expect(applyBankConfig(cfg, deriveBankId(cfg, other, "codex"), other).cfg.disabled).toBe(true);
  });
});
