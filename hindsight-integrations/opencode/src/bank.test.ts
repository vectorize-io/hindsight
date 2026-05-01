import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

vi.mock("node:child_process", () => ({
  execFileSync: vi.fn(),
}));

import { execFileSync } from "node:child_process";
import { deriveBankId, ensureBankMission } from "./bank.js";
import { makeConfig } from "./test-helpers.js";

const mockExec = vi.mocked(execFileSync);

describe("deriveBankId", () => {
  const originalEnv = { ...process.env };

  beforeEach(() => {
    // Default: simulate "not in a git repo" so the project field falls back
    // to the directory basename. Individual git-aware tests override this.
    mockExec.mockImplementation(() => {
      throw new Error("fatal: not a git repository");
    });
  });

  afterEach(() => {
    process.env = { ...originalEnv };
    mockExec.mockReset();
  });

  it("returns default bank name in static mode", () => {
    expect(deriveBankId(makeConfig(), "/home/user/project")).toBe("opencode");
  });

  it("returns configured bankId in static mode", () => {
    const config = makeConfig({ bankId: "my-bank" });
    expect(deriveBankId(config, "/home/user/project")).toBe("my-bank");
  });

  it("adds prefix in static mode", () => {
    const config = makeConfig({ bankIdPrefix: "dev", bankId: "my-bank" });
    expect(deriveBankId(config, "/home/user/project")).toBe("dev-my-bank");
  });

  it("composes from granularity fields in dynamic mode", () => {
    const config = makeConfig({
      dynamicBankId: true,
      dynamicBankGranularity: ["agent", "project"],
      agentName: "opencode",
    });
    expect(deriveBankId(config, "/home/user/my-project")).toBe("opencode::my-project");
  });

  it("uses default granularity when not specified", () => {
    const config = makeConfig({
      dynamicBankId: true,
      dynamicBankGranularity: [],
    });
    expect(deriveBankId(config, "/home/user/proj")).toBe("opencode::proj");
  });

  it("preserves raw special characters", () => {
    const config = makeConfig({
      dynamicBankId: true,
      dynamicBankGranularity: ["project"],
    });
    expect(deriveBankId(config, "/home/user/my project")).toBe("my project");
  });

  it("preserves raw UTF-8 characters", () => {
    const config = makeConfig({
      dynamicBankId: true,
      dynamicBankGranularity: ["project"],
    });
    expect(deriveBankId(config, "/home/user/мой проект")).toBe("мой проект");
  });

  it("uses channel/user from env vars", () => {
    process.env.HINDSIGHT_CHANNEL_ID = "slack-general";
    process.env.HINDSIGHT_USER_ID = "user123";
    const config = makeConfig({
      dynamicBankId: true,
      dynamicBankGranularity: ["agent", "channel", "user"],
    });
    expect(deriveBankId(config, "/home/user/proj")).toBe("opencode::slack-general::user123");
  });

  it("uses defaults for missing env vars", () => {
    delete process.env.HINDSIGHT_CHANNEL_ID;
    delete process.env.HINDSIGHT_USER_ID;
    const config = makeConfig({
      dynamicBankId: true,
      dynamicBankGranularity: ["channel", "user"],
    });
    expect(deriveBankId(config, "/home/user/proj")).toBe("default::anonymous");
  });

  it("adds prefix in dynamic mode", () => {
    const config = makeConfig({
      dynamicBankId: true,
      bankIdPrefix: "dev",
      dynamicBankGranularity: ["agent"],
    });
    expect(deriveBankId(config, "/home/user/proj")).toBe("dev-opencode");
  });

  describe("project field stays directory-only (backwards compatibility)", () => {
    it("uses raw directory basename for `project` even inside a git repo", () => {
      mockExec.mockReturnValueOnce("/home/user/myproj/.git\n" as never);
      const config = makeConfig({
        dynamicBankId: true,
        dynamicBankGranularity: ["agent", "project"],
      });
      expect(deriveBankId(config, "/tmp/worktrees/myproj-feature-x")).toBe(
        "opencode::myproj-feature-x"
      );
    });

    it("does not invoke git when only `project` is in the granularity", () => {
      const config = makeConfig({
        dynamicBankId: true,
        dynamicBankGranularity: ["agent", "project"],
      });
      deriveBankId(config, "/home/user/myproj");
      expect(mockExec).not.toHaveBeenCalled();
    });
  });

  describe("gitProject field (git-aware)", () => {
    it("uses main worktree basename when running inside a regular clone", () => {
      // `git rev-parse --git-common-dir` returns the main repo's .git path.
      mockExec.mockReturnValueOnce("/home/user/myproj/.git\n" as never);
      const config = makeConfig({
        dynamicBankId: true,
        dynamicBankGranularity: ["agent", "gitProject"],
      });
      expect(deriveBankId(config, "/home/user/myproj")).toBe("opencode::myproj");
    });

    it("returns the same bank id from a linked worktree of the same repo", () => {
      // Both invocations resolve to the SAME main .git, so worktrees share the bank.
      mockExec
        .mockReturnValueOnce("/home/user/myproj/.git\n" as never)
        .mockReturnValueOnce("/home/user/myproj/.git\n" as never);
      const config = makeConfig({
        dynamicBankId: true,
        dynamicBankGranularity: ["agent", "gitProject"],
      });
      const main = deriveBankId(config, "/home/user/myproj");
      const linked = deriveBankId(config, "/tmp/worktrees/myproj-feature-x");
      expect(main).toBe("opencode::myproj");
      expect(linked).toBe(main);
    });

    it("uses bare repo basename when common-dir is the bare repo itself", () => {
      mockExec.mockReturnValueOnce("/srv/git/myrepo.git\n" as never);
      const config = makeConfig({
        dynamicBankId: true,
        dynamicBankGranularity: ["gitProject"],
      });
      expect(deriveBankId(config, "/srv/git/myrepo.git")).toBe("myrepo.git");
    });

    it("falls back to directory basename when git is unavailable or directory is not a repo", () => {
      mockExec.mockImplementationOnce(() => {
        throw new Error("git: command not found");
      });
      const config = makeConfig({
        dynamicBankId: true,
        dynamicBankGranularity: ["gitProject"],
      });
      expect(deriveBankId(config, "/tmp/random")).toBe("random");
    });

    it("does not invoke git in static mode", () => {
      const config = makeConfig({ bankId: "fixed" });
      expect(deriveBankId(config, "/home/user/myproj")).toBe("fixed");
      expect(mockExec).not.toHaveBeenCalled();
    });

    it("does not invoke git when gitProject is not in the granularity", () => {
      const config = makeConfig({
        dynamicBankId: true,
        dynamicBankGranularity: ["agent", "channel"],
      });
      expect(deriveBankId(config, "/home/user/myproj")).toBe("opencode::default");
      expect(mockExec).not.toHaveBeenCalled();
    });

    it("can combine project and gitProject as separate segments", () => {
      mockExec.mockReturnValueOnce("/home/user/myproj/.git\n" as never);
      const config = makeConfig({
        dynamicBankId: true,
        dynamicBankGranularity: ["agent", "project", "gitProject"],
      });
      expect(deriveBankId(config, "/tmp/worktrees/myproj-feature-x")).toBe(
        "opencode::myproj-feature-x::myproj"
      );
    });
  });
});

describe("ensureBankMission", () => {
  it("calls createBank on first use", async () => {
    const client = { createBank: vi.fn().mockResolvedValue({}) } as any;
    const missionsSet = new Set<string>();
    const config = makeConfig({ bankMission: "Test mission" });

    await ensureBankMission(client, "test-bank", config, missionsSet);

    expect(client.createBank).toHaveBeenCalledWith("test-bank", {
      reflectMission: "Test mission",
      retainMission: undefined,
    });
    expect(missionsSet.has("test-bank")).toBe(true);
  });

  it("skips if already set", async () => {
    const client = { createBank: vi.fn() } as any;
    const missionsSet = new Set(["test-bank"]);
    const config = makeConfig({ bankMission: "Test mission" });

    await ensureBankMission(client, "test-bank", config, missionsSet);

    expect(client.createBank).not.toHaveBeenCalled();
  });

  it("skips if no mission configured", async () => {
    const client = { createBank: vi.fn() } as any;
    const missionsSet = new Set<string>();
    const config = makeConfig({ bankMission: "" });

    await ensureBankMission(client, "test-bank", config, missionsSet);

    expect(client.createBank).not.toHaveBeenCalled();
  });

  it("does not throw on client error", async () => {
    const client = { createBank: vi.fn().mockRejectedValue(new Error("Network error")) } as any;
    const missionsSet = new Set<string>();
    const config = makeConfig({ bankMission: "Mission" });

    await expect(
      ensureBankMission(client, "test-bank", config, missionsSet)
    ).resolves.not.toThrow();
    expect(missionsSet.has("test-bank")).toBe(false);
  });

  it("passes retainMission when configured", async () => {
    const client = { createBank: vi.fn().mockResolvedValue({}) } as any;
    const missionsSet = new Set<string>();
    const config = makeConfig({ bankMission: "Reflect", retainMission: "Extract carefully" });

    await ensureBankMission(client, "test-bank", config, missionsSet);

    expect(client.createBank).toHaveBeenCalledWith("test-bank", {
      reflectMission: "Reflect",
      retainMission: "Extract carefully",
    });
  });
});
