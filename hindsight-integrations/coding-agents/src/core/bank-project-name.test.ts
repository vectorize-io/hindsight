import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Same seam as bank.test.ts: state what the repository layout probe says, assert what is derived.
vi.mock("./git-layout", () => ({ probeGitLayout: vi.fn() }));

import { bankProjectName, deriveBankId } from "./bank";
import { probeGitLayout } from "./git-layout";
import { pagesFor } from "./missions";

const mockProbe = vi.mocked(probeGitLayout);
const inRepo = (commonDir: string, bare = false) =>
  ({ status: "resolved", commonDir, bare }) as const;

/**
 * #4146: the seeded knowledge pages carry a scope sentence naming the repository they are about,
 * and that sentence is PATCHed onto pages that outlive the session. Deriving it from the session's
 * cwd made it a property of whoever ran last: on a bank several repositories share, every session
 * start rewrote all five pages to its own repo and the previous repo's synthesis was thrown away.
 *
 * The name may therefore only be answered when the BANK and the REPOSITORY are the same unit.
 */
describe("bankProjectName", () => {
  beforeEach(() => {
    mockProbe.mockReturnValue({ status: "absent" }); // default: not in a git repo
  });
  afterEach(() => {
    vi.clearAllMocks();
  });

  describe("a bank the repository owns", () => {
    it("names the repo under the default template, matching the bank id's own {gitProject}", () => {
      mockProbe.mockReturnValue(inRepo("/home/me/dev/myrepo/.git"));
      expect(bankProjectName({}, "/home/me/dev/myrepo")).toBe("myrepo");
      expect(deriveBankId({}, "/home/me/dev/myrepo")).toBe("coding-agent::myrepo");
    });

    it("gives every linked worktree the same name, as the shared bank id does", () => {
      mockProbe.mockReturnValue(inRepo("/home/me/dev/myrepo/.git"));
      const main = bankProjectName({}, "/home/me/dev/myrepo");
      const worktree = bankProjectName({}, "/home/me/dev/.worktrees/myrepo-233");
      expect(worktree).toBe(main);
      expect(worktree).toBe("myrepo");
    });

    it("answers for a custom template that still keys on {gitProject}", () => {
      mockProbe.mockReturnValue(inRepo("/home/me/dev/myrepo/.git"));
      const cfg = { bankIdTemplate: "{harness}-{gitProject}" };
      expect(bankProjectName(cfg, "/home/me/dev/myrepo")).toBe("myrepo");
    });

    it("honours resolveWorktrees:false — the SAME call the bank id makes, not a second one", () => {
      mockProbe.mockReturnValue(inRepo("/home/me/dev/myrepo/.git"));
      const cfg = { resolveWorktrees: false };
      // The bank is per-WORKTREE here, so the page scope must be too, or a bank named after the
      // worktree would carry a sentence naming the main repo.
      expect(bankProjectName(cfg, "/home/me/dev/myrepo-wt")).toBe("myrepo-wt");
      expect(deriveBankId(cfg, "/home/me/dev/myrepo-wt")).toBe("coding-agent::myrepo-wt");
      expect(mockProbe).not.toHaveBeenCalled();
    });

    it("names the session's starting directory outside a repository, as the bank id does", () => {
      expect(bankProjectName({}, "/home/me/scratch")).toBe("scratch");
    });
  });

  describe("a bank no single repository owns", () => {
    it("declines for a static bankId — the reported configuration", () => {
      mockProbe.mockReturnValue(inRepo("/work/one-repo/.git"));
      expect(bankProjectName({ bankId: "shared_bank_id" }, "/work/one-repo")).toBeUndefined();
    });

    it("declines for dynamicBankId:false", () => {
      mockProbe.mockReturnValue(inRepo("/work/one-repo/.git"));
      const cfg = { bankId: "shared_bank_id", dynamicBankId: false };
      expect(bankProjectName(cfg, "/work/one-repo")).toBeUndefined();
    });

    it("declines for a mapPathToBank destination, which many paths may share", () => {
      mockProbe.mockReturnValue(inRepo("/work/one-repo/.git"));
      const cfg = { mapPathToBank: { "/work": "workspace" } };
      expect(bankProjectName(cfg, "/work/one-repo")).toBeUndefined();
      expect(deriveBankId(cfg, "/work/one-repo")).toBe("workspace");
    });

    it("declines for a template that does not key on the repository", () => {
      mockProbe.mockReturnValue(inRepo("/work/one-repo/.git"));
      // {project} is the live directory basename — it names a subdirectory as readily as a repo.
      expect(bankProjectName({ bankIdTemplate: "{project}" }, "/work/one-repo")).toBeUndefined();
      expect(bankProjectName({ bankIdTemplate: "{user}-{channel}" }, "/work/x")).toBeUndefined();
    });
  });

  it("returns undefined rather than throwing when the probe cannot name the repository", () => {
    mockProbe.mockReturnValue({ status: "failed", reason: "EAGAIN" });
    // deriveBankId refuses to guess here (#3950) because a wrong bank id is permanent; a page
    // scope name is not identity, so the caller falls back to the bank id instead of crashing.
    expect(bankProjectName({}, "/home/me/dev/myrepo-wt1")).toBeUndefined();
    expect(() => deriveBankId({}, "/home/me/dev/myrepo-wt1")).toThrow();
  });

  it("keeps the seeded page queries identical across repos on a shared bank (#4146 repro)", () => {
    const cfg = { bankId: "shared_bank_id" };
    const BANK = "shared_bank_id";

    mockProbe.mockReturnValue(inRepo("/work/one-repo/.git"));
    const fromRepoA = pagesFor(bankProjectName(cfg, "/work/.worktrees/one-repo-233") ?? BANK);

    mockProbe.mockReturnValue({ status: "absent" });
    const fromWorkspaceRoot = pagesFor(bankProjectName(cfg, "/work") ?? BANK);

    // Same text => `seedPages()` sees no source drift => no PATCH, no page regeneration.
    expect(fromWorkspaceRoot).toEqual(fromRepoA);
    for (const page of fromRepoA) expect(page.source_query).toContain("shared_bank_id");
  });
});
