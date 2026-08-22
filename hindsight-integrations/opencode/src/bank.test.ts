import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

vi.mock("node:child_process", () => ({
  execFileSync: vi.fn(),
}));

import { execFileSync } from "node:child_process";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  deriveBankId,
  ensureBankMission,
  resolveBankId,
  readAgentBankId,
  toBankResolver,
  createBankResolver,
} from "./bank.js";
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

describe("resolveBankId", () => {
  beforeEach(() => {
    mockExec.mockImplementation(() => {
      throw new Error("fatal: not a git repository");
    });
  });

  it("returns the per-agent entry for the running agent", () => {
    const config = makeConfig({
      bankId: "default-bank",
      hindsightBankIds: { build: "build-bank", "review-agent": "review-bank" },
    });
    expect(resolveBankId(config, "/dir", "build")).toBe("build-bank");
    expect(resolveBankId(config, "/dir", "review-agent")).toBe("review-bank");
  });

  it("applies bankIdPrefix to per-agent entries", () => {
    const config = makeConfig({
      bankId: "default-bank",
      bankIdPrefix: "dev",
      hindsightBankIds: { build: "build-bank" },
    });
    expect(resolveBankId(config, "/dir", "build")).toBe("dev-build-bank");
  });

  it("falls back to the default agentName entry", () => {
    const config = makeConfig({
      bankId: "default-bank",
      agentName: "opencode",
      hindsightBankIds: { opencode: "default-agent-bank" },
    });
    expect(resolveBankId(config, "/dir", "security-reviewer")).toBe("default-agent-bank");
    expect(resolveBankId(config, "/dir", undefined)).toBe("default-agent-bank");
  });

  it("falls back to deriveBankId when the map has no matching entry", () => {
    const config = makeConfig({ bankId: "static-bank" });
    expect(resolveBankId(config, "/dir", "build")).toBe("static-bank");
  });

  it("falls back to dynamic-granularity derivation", () => {
    const config = makeConfig({
      dynamicBankId: true,
      dynamicBankGranularity: ["agent", "project"],
    });
    expect(resolveBankId(config, "/home/user/my-project", "build")).toBe(
      "opencode::my-project"
    );
  });
});

describe("toBankResolver", () => {
  it("wraps a bare string so forAgent() always returns it", () => {
    const resolver = toBankResolver("fixed-bank");
    expect(resolver.forAgent("build")).toBe("fixed-bank");
    expect(resolver.forAgent(undefined)).toBe("fixed-bank");
    expect(resolver.forAgent(null)).toBe("fixed-bank");
  });

  it("passes through an existing BankResolver unchanged", () => {
    const resolver = { forAgent: () => "dynamic" };
    expect(toBankResolver(resolver)).toBe(resolver);
  });
});

describe("createBankResolver", () => {
  it("resolves per-agent, falling back to derivation", () => {
    mockExec.mockImplementation(() => {
      throw new Error("fatal: not a git repository");
    });
    const config = makeConfig({
      bankId: "default-bank",
      hindsightBankIds: { build: "build-bank" },
    });
    const resolver = createBankResolver(config, "/dir");
    expect(resolver.forAgent("build")).toBe("build-bank");
    expect(resolver.forAgent("other")).toBe("default-bank");
    expect(resolver.forAgent()).toBe("default-bank");
  });
});

describe("readAgentBankId (agent .md frontmatter)", () => {
  let tmpRoot: string;

  beforeEach(() => {
    // Default: simulate "not in a git repo" so deriveBankId fallback works.
    mockExec.mockImplementation(() => {
      throw new Error("fatal: not a git repository");
    });
    tmpRoot = mkdtempSync(join(tmpdir(), "hindsight-bank-test-"));
  });

  afterEach(() => {
    rmSync(tmpRoot, { recursive: true, force: true });
  });

  function writeAgentFile(
    agentName: string,
    frontmatter: Record<string, string> | null,
    body = ""
  ): string {
    const dir = join(tmpRoot, ".opencode", "agent");
    mkdirSync(dir, { recursive: true });
    const path = join(dir, `${agentName}.md`);
    let content = body;
    if (frontmatter) {
      const fmLines = Object.entries(frontmatter)
        .map(([k, v]) => `${k}: ${v}`)
        .join("\n");
      content = `---\n${fmLines}\n---\n${body}`;
    }
    writeFileSync(path, content, "utf-8");
    return path;
  }

  it("reads the bankid field from agent frontmatter", () => {
    writeAgentFile("code-reviewer", {
      bankid: "reviewer-bank-from-md",
      description: "Reviews code",
    });
    expect(readAgentBankId("code-reviewer", tmpRoot)).toBe(
      "reviewer-bank-from-md"
    );
  });

  it("supports quoted bankid values", () => {
    writeAgentFile("build", { bankid: '"quoted-bank"' });
    expect(readAgentBankId("build", tmpRoot)).toBe("quoted-bank");
  });

  it("supports single-quoted bankid values", () => {
    writeAgentFile("build", { bankid: "'single-quoted-bank'" });
    expect(readAgentBankId("build", tmpRoot)).toBe("single-quoted-bank");
  });

  it("ignores nested keys named bankid (top-level only)", () => {
    const dir = join(tmpRoot, ".opencode", "agent");
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      join(dir, "build.md"),
      // The indented `bankid:` is nested under `permission:` and must be ignored.
      `---\ndescription: x\npermission:\n  bankid: nested-bank\nbankid: top-level-bank\n---\nbody`,
      "utf-8"
    );
    expect(readAgentBankId("build", tmpRoot)).toBe("top-level-bank");
  });

  it("returns null when the agent file has no frontmatter", () => {
    writeAgentFile("build", null, "Just body, no frontmatter.");
    expect(readAgentBankId("build", tmpRoot)).toBeNull();
  });

  it("returns null when frontmatter has no bankid field", () => {
    writeAgentFile("build", { description: "No bankid here" });
    expect(readAgentBankId("build", tmpRoot)).toBeNull();
  });

  it("returns null when the agent file does not exist", () => {
    expect(readAgentBankId("nonexistent", tmpRoot)).toBeNull();
  });

  it("returns null when agentName is null or undefined", () => {
    expect(readAgentBankId(null, tmpRoot)).toBeNull();
    expect(readAgentBankId(undefined, tmpRoot)).toBeNull();
  });

  it("caches based on file mtime", () => {
    const path = writeAgentFile("build", { bankid: "v1-bank" });
    expect(readAgentBankId("build", tmpRoot)).toBe("v1-bank");
    // Overwrite with a new bankid; mtime likely advances → cache should refresh.
    // Add a tiny delay on platforms with coarse mtime resolution.
    writeFileSync(path, `---\nbankid: v2-bank\n---\nbody`, "utf-8");
    expect(readAgentBankId("build", tmpRoot)).toBe("v2-bank");
  });

  it("checks .opencode/agents/ (plural) as a fallback location", () => {
    const dir = join(tmpRoot, ".opencode", "agents");
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      join(dir, "plural-agent.md"),
      `---\nbankid: plural-bank\n---\nbody`,
      "utf-8"
    );
    expect(readAgentBankId("plural-agent", tmpRoot)).toBe("plural-bank");
  });
});

describe("resolveBankId with agent .md frontmatter precedence", () => {
  let tmpRoot: string;

  beforeEach(() => {
    mockExec.mockImplementation(() => {
      throw new Error("fatal: not a git repository");
    });
    tmpRoot = mkdtempSync(join(tmpdir(), "hindsight-bank-test-"));
  });

  afterEach(() => {
    rmSync(tmpRoot, { recursive: true, force: true });
  });

  function writeAgentFile(agentName: string, bankId: string): void {
    const dir = join(tmpRoot, ".opencode", "agent");
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      join(dir, `${agentName}.md`),
      `---\ndescription: test\nbankid: ${bankId}\n---\nbody`,
      "utf-8"
    );
  }

  it("agent .md frontmatter takes precedence over hindsightBankIds", () => {
    writeAgentFile("code-reviewer", "from-md-bank");
    const config = makeConfig({
      bankId: "default-bank",
      hindsightBankIds: { "code-reviewer": "from-map-bank" },
    });
    expect(resolveBankId(config, tmpRoot, "code-reviewer")).toBe("from-md-bank");
  });

  it("applies bankIdPrefix to the frontmatter bankid", () => {
    writeAgentFile("build", "md-bank");
    const config = makeConfig({ bankIdPrefix: "dev" });
    expect(resolveBankId(config, tmpRoot, "build")).toBe("dev-md-bank");
  });

  it("falls back to hindsightBankIds when no frontmatter bankid", () => {
    // No .md file written for this agent.
    const config = makeConfig({
      bankId: "default-bank",
      hindsightBankIds: { build: "from-map-bank" },
    });
    expect(resolveBankId(config, tmpRoot, "build")).toBe("from-map-bank");
  });

  it("falls back to deriveBankId when neither frontmatter nor map match", () => {
    const config = makeConfig({ bankId: "static-bank" });
    expect(resolveBankId(config, tmpRoot, "unknown-agent")).toBe("static-bank");
  });
});
