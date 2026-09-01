import { mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./git-layout", () => ({ probeGitLayout: vi.fn() }));

import { BankResolutionError, deriveBankId, deriveBankIdOrSkip, isOptedIn } from "./bank";
import { probeGitLayout } from "./git-layout";

const mockProbe = vi.mocked(probeGitLayout);

/**
 * #3950: a probe that FAILS is not a probe that says "no repository here". Reading the two the
 * same way is what silently retained a linked worktree's session into `coding-agent::<repo>-wt1`,
 * a bank nobody ever reads back, permanently and with no log line.
 */
describe("a git probe that could not complete", () => {
  const WORKTREE = "/home/me/dev/myrepo-wt1";
  let diagDir: string;

  beforeEach(() => {
    mockProbe.mockReturnValue({ status: "failed", reason: "EAGAIN" });
    diagDir = mkdtempSync(join(tmpdir(), "hs-diag-"));
    process.env.HINDSIGHT_DIAG_FILE = join(diagDir, "diag.jsonl");
  });

  afterEach(() => {
    rmSync(diagDir, { recursive: true, force: true });
    delete process.env.HINDSIGHT_DIAG_FILE;
    vi.clearAllMocks();
  });

  it("never falls back to the worktree's own basename", () => {
    expect(() => deriveBankId({}, WORKTREE)).toThrow(BankResolutionError);
  });

  it("skips the session instead, and says so in the diagnostics", () => {
    expect(deriveBankIdOrSkip({}, WORKTREE, "claude-code")).toBeNull();
    const events = readFileSync(process.env.HINDSIGHT_DIAG_FILE as string, "utf8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line) as Record<string, unknown>);
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({ event: "bank_unresolved", directory: WORKTREE });
  });

  it("still honours a static bank id — nothing was guessed there", () => {
    expect(deriveBankIdOrSkip({ bankId: "pinned" }, WORKTREE)).toBe("pinned");
  });

  it("still honours an explicit mapPathToBank entry for the directory itself", () => {
    expect(deriveBankIdOrSkip({ mapPathToBank: { [WORKTREE]: "mapped" } }, WORKTREE)).toBe(
      "mapped"
    );
  });

  it("leaves opt-in approval fail-closed rather than throwing", () => {
    expect(isOptedIn({ optInOnly: true, optInPaths: ["/home/me/dev"] }, WORKTREE)).toBe(true);
    expect(isOptedIn({ optInOnly: true, optInPaths: ["/elsewhere"] }, WORKTREE)).toBe(false);
  });
});

/**
 * The family guard. Bank resolution happens once per entrypoint, and #3950 was one entrypoint's
 * worth of a wrong answer becoming permanent — so the rule has to hold for ALL of them, including
 * the next harness added. A per-entrypoint test cannot catch this: the entrypoint that forgets is
 * by definition the one whose test nobody wrote.
 */
describe("every entrypoint resolves banks through the skip path", () => {
  const SRC = fileURLToPath(new URL("..", import.meta.url));

  /** The only module allowed to call the throwing form: it IS the throwing form's home. */
  const OWNER = "core/bank.ts";

  function sourceFiles(dir: string, prefix = ""): string[] {
    return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
      const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
      if (entry.isDirectory())
        return entry.name === "e2e" ? [] : sourceFiles(join(dir, entry.name), rel);
      return entry.name.endsWith(".ts") && !entry.name.includes(".test.") ? [rel] : [];
    });
  }

  /** Comment lines are excluded: prose naming the function is documentation, not a call. */
  const callsDirectly = (src: string): boolean =>
    src
      .split("\n")
      .filter((line) => !/^\s*(\/\/|\*|\/\*)/.test(line))
      .some((line) => /\bderiveBankId\s*\(/.test(line));

  it("has no module calling deriveBankId( directly", () => {
    const direct = sourceFiles(SRC).filter(
      (rel) => rel !== OWNER && callsDirectly(readFileSync(join(SRC, rel), "utf8"))
    );
    expect(direct).toEqual([]);
  });
});
