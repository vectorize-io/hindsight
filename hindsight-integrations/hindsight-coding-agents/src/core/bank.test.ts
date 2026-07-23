import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("node:child_process", () => ({ execFileSync: vi.fn() }));

import { execFileSync } from "node:child_process";
import { deriveBankId } from "./bank";

const mockExec = vi.mocked(execFileSync);
const NOT_A_REPO = () => {
  throw new Error("fatal: not a git repository");
};

describe("deriveBankId", () => {
  beforeEach(() => {
    mockExec.mockImplementation(NOT_A_REPO); // default: not in a git repo
  });
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("defaults to the directory basename outside git", () => {
    expect(deriveBankId({}, "/home/me/scratch")).toBe("scratch");
  });

  it("resolves the MAIN worktree root inside git (worktrees share one bank)", () => {
    mockExec.mockReturnValue("/home/me/dev/myrepo/.git\n");
    expect(deriveBankId({}, "/home/me/dev/myrepo-feature-wt")).toBe("myrepo");
  });

  it("uses the bare-repo directory name when common-dir is not .git", () => {
    mockExec.mockReturnValue("/srv/git/myrepo.git\n");
    expect(deriveBankId({}, "/srv/git/myrepo.git")).toBe("myrepo.git");
  });

  it("resolveWorktrees=false skips git and uses the directory basename", () => {
    mockExec.mockReturnValue("/home/me/dev/myrepo/.git\n");
    expect(deriveBankId({ resolveWorktrees: false }, "/home/me/dev/myrepo-wt")).toBe("myrepo-wt");
    expect(mockExec).not.toHaveBeenCalled();
  });

  it("explicit bankId means static", () => {
    expect(deriveBankId({ bankId: "pinned" }, "/any/where")).toBe("pinned");
  });

  it("dynamicBankId=true overrides an explicit bankId", () => {
    expect(deriveBankId({ bankId: "pinned", dynamicBankId: true }, "/home/me/proj")).toBe("proj");
  });

  it("dynamicBankId=false without bankId falls back to the default name", () => {
    expect(deriveBankId({ dynamicBankId: false }, "/home/me/proj")).toBe("coding");
  });

  describe("bankIdTemplate", () => {
    it("supports literal text around placeholders", () => {
      expect(deriveBankId({ bankIdTemplate: "hindsight-{gitProject}" }, "/home/me/proj")).toBe(
        "hindsight-proj"
      );
    });

    it("fills {harness} from the caller", () => {
      expect(
        deriveBankId({ bankIdTemplate: "{harness}-{gitProject}" }, "/home/me/proj", "claude-code")
      ).toBe("claude-code-proj");
    });

    it("{project} is the plain directory basename even inside git", () => {
      mockExec.mockReturnValue("/home/me/dev/myrepo/.git\n");
      expect(deriveBankId({ bankIdTemplate: "{project}" }, "/home/me/dev/myrepo-wt")).toBe(
        "myrepo-wt"
      );
    });

    it("warns on unknown placeholders and substitutes 'unknown'", () => {
      const err = vi.spyOn(console, "error").mockImplementation(() => {});
      expect(deriveBankId({ bankIdTemplate: "x-{gitProjectId}" }, "/home/me/proj")).toBe(
        "x-unknown"
      );
      expect(err).toHaveBeenCalledOnce();
      expect(err.mock.calls[0][0]).toContain("{gitProjectId}");
      err.mockRestore();
    });
  });

  describe("directoryBankMap", () => {
    const WT = "/home/me/dev/myrepo";

    it("matches the exact directory", () => {
      expect(deriveBankId({ directoryBankMap: { [WT]: "mapped" } }, WT)).toBe("mapped");
    });

    it("matches any subdirectory by prefix", () => {
      expect(deriveBankId({ directoryBankMap: { [WT]: "mapped" } }, `${WT}/deep/sub`)).toBe(
        "mapped"
      );
    });

    it("does not match sibling directories sharing a name prefix", () => {
      expect(deriveBankId({ directoryBankMap: { [WT]: "mapped" } }, `${WT}-other`)).toBe(
        "myrepo-other"
      );
    });

    it("longest prefix wins", () => {
      const map = { [WT]: "outer", [`${WT}/pkg`]: "inner" };
      expect(deriveBankId({ directoryBankMap: map }, `${WT}/pkg/x`)).toBe("inner");
      expect(deriveBankId({ directoryBankMap: map }, `${WT}/other`)).toBe("outer");
    });

    it("overrides an explicit bankId", () => {
      expect(deriveBankId({ bankId: "static", directoryBankMap: { [WT]: "mapped" } }, WT)).toBe(
        "mapped"
      );
    });

    it("tolerates trailing slashes in map keys", () => {
      expect(deriveBankId({ directoryBankMap: { [`${WT}/`]: "mapped" } }, `${WT}/sub`)).toBe(
        "mapped"
      );
    });
  });
});
