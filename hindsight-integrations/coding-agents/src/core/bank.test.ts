import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// The git LAYOUT probe is the seam, not `child_process`: resolution reads the repository layout
// off disk (see git-layout.ts), so these unit tests state what the layout says and assert the bank
// id derived from it. Real repositories are covered by bank-bare-hub / bank-missing-dir.
vi.mock("./git-layout", () => ({ probeGitLayout: vi.fn() }));

import { deriveBankId } from "./bank";
import { probeGitLayout } from "./git-layout";

const mockProbe = vi.mocked(probeGitLayout);
const inRepo = (commonDir: string, bare = false) =>
  ({ status: "resolved", commonDir, bare }) as const;

describe("deriveBankId", () => {
  beforeEach(() => {
    mockProbe.mockReturnValue({ status: "absent" }); // default: not in a git repo
  });
  afterEach(() => {
    vi.clearAllMocks();
  });

  // The default template is harness-neutral `coding-agent::{gitProject}` (every coding agent shares
  // one bank per repo), so the default cases below carry that prefix.
  it("defaults to coding-agent::<directory basename> outside git", () => {
    expect(deriveBankId({}, "/home/me/scratch")).toBe("coding-agent::scratch");
  });

  it("resolves the MAIN worktree root inside git (worktrees share one bank)", () => {
    mockProbe.mockReturnValue(inRepo("/home/me/dev/myrepo/.git"));
    expect(deriveBankId({}, "/home/me/dev/myrepo-feature-wt")).toBe("coding-agent::myrepo");
  });

  it("uses the bare-repo directory name when common-dir is not .git", () => {
    mockProbe.mockReturnValue(inRepo("/srv/git/myrepo.git", true));
    expect(deriveBankId({}, "/srv/git/myrepo.git")).toBe("coding-agent::myrepo.git");
  });

  it("uses the hub name for a hidden bare repository shared by worktrees", () => {
    mockProbe.mockReturnValue(inRepo("/home/me/myrepo/.bare", true));
    expect(deriveBankId({}, "/home/me/myrepo/main")).toBe("coding-agent::myrepo");
  });

  it("keeps a hidden non-bare common directory unchanged", () => {
    mockProbe.mockReturnValue(inRepo("/home/me/project/.git-bare", false));
    expect(deriveBankId({}, "/home/me/project")).toBe("coding-agent::.git-bare");
  });

  it("resolveWorktrees=false skips git and uses the directory basename", () => {
    mockProbe.mockReturnValue(inRepo("/home/me/dev/myrepo/.git"));
    expect(deriveBankId({ resolveWorktrees: false }, "/home/me/dev/myrepo-wt")).toBe(
      "coding-agent::myrepo-wt"
    );
    expect(mockProbe).not.toHaveBeenCalled();
  });

  it("the default bank is harness-neutral — same id regardless of the harness arg", () => {
    expect(deriveBankId({}, "/home/me/scratch", "claude-code")).toBe("coding-agent::scratch");
    expect(deriveBankId({}, "/home/me/scratch", "codex")).toBe("coding-agent::scratch");
  });

  it("explicit bankId means static", () => {
    expect(deriveBankId({ bankId: "pinned" }, "/any/where")).toBe("pinned");
  });

  it("dynamicBankId=true overrides an explicit bankId", () => {
    expect(deriveBankId({ bankId: "pinned", dynamicBankId: true }, "/home/me/proj")).toBe(
      "coding-agent::proj"
    );
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
      mockProbe.mockReturnValue(inRepo("/home/me/dev/myrepo/.git"));
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

  describe("mapPathToBank", () => {
    const WT = "/home/me/dev/myrepo";

    it("matches the exact directory", () => {
      expect(deriveBankId({ mapPathToBank: { [WT]: "mapped" } }, WT)).toBe("mapped");
    });

    it("matches any subdirectory by prefix", () => {
      expect(deriveBankId({ mapPathToBank: { [WT]: "mapped" } }, `${WT}/deep/sub`)).toBe("mapped");
    });

    it("does not match sibling directories sharing a name prefix", () => {
      expect(deriveBankId({ mapPathToBank: { [WT]: "mapped" } }, `${WT}-other`)).toBe(
        "coding-agent::myrepo-other"
      );
    });

    it("longest prefix wins", () => {
      const map = { [WT]: "outer", [`${WT}/pkg`]: "inner" };
      expect(deriveBankId({ mapPathToBank: map }, `${WT}/pkg/x`)).toBe("inner");
      expect(deriveBankId({ mapPathToBank: map }, `${WT}/other`)).toBe("outer");
    });

    it("overrides an explicit bankId", () => {
      expect(deriveBankId({ bankId: "static", mapPathToBank: { [WT]: "mapped" } }, WT)).toBe(
        "mapped"
      );
    });

    it("tolerates trailing slashes in map keys", () => {
      expect(deriveBankId({ mapPathToBank: { [`${WT}/`]: "mapped" } }, `${WT}/sub`)).toBe("mapped");
    });
  });
});

describe("mapPathToBank ~ expansion", () => {
  it("expands a leading ~ to the home directory (prefix match included)", () => {
    const cfg = { mapPathToBank: { "~/scratch-zone": "scratch" } } as never;
    expect(deriveBankId(cfg, `${process.env.HOME}/scratch-zone/some/repo`)).toBe("scratch");
    expect(deriveBankId(cfg, "/elsewhere/scratch-zone")).not.toBe("scratch");
  });
});
