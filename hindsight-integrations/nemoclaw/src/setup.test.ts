import { describe, it, expect, vi, beforeEach } from "vitest";
import type { CliArgs } from "./types.js";

// Mock all external I/O before importing setup
vi.mock("child_process", () => ({
  execFile: vi.fn(),
}));
vi.mock("./policy-reader.js", () => ({
  readSandboxPolicy: vi.fn(),
}));
vi.mock("./policy-writer.js", () => ({
  hasHindsightPolicy: vi.fn(),
  mergeHindsightPolicy: vi.fn(),
  serializePolicy: vi.fn(),
}));
vi.mock("./openclaw-config.js", () => ({
  applyPluginConfig: vi.fn(),
}));
vi.mock("fs/promises", () => ({
  writeFile: vi.fn(),
  rm: vi.fn(),
}));

const BASE_ARGS: CliArgs = {
  sandbox: "my-assistant",
  apiUrl: "https://api.hindsight.vectorize.io",
  apiToken: "hsk_test123",
  bankPrefix: "my-sandbox",
  skipPolicy: false,
  skipPluginInstall: false,
  dryRun: false,
};

describe("runSetup", () => {
  beforeEach(async () => {
    vi.clearAllMocks();

    const { execFile } = await import("child_process");
    const execFileMock = vi.mocked(execFile);

    // Default: all shell commands succeed
    execFileMock.mockImplementation((_cmd, _args, callback?: unknown) => {
      if (typeof callback === "function") {
        (callback as (err: null, result: { stdout: string; stderr: string }) => void)(null, {
          stdout: "",
          stderr: "",
        });
      }
      return {} as ReturnType<typeof execFile>;
    });

    const { readSandboxPolicy } = await import("./policy-reader.js");
    vi.mocked(readSandboxPolicy).mockResolvedValue({
      version: 1,
      network_policies: { claude_code: { name: "claude_code", endpoints: [] } },
    });

    const { hasHindsightPolicy, mergeHindsightPolicy, serializePolicy } =
      await import("./policy-writer.js");
    vi.mocked(hasHindsightPolicy).mockReturnValue(false);
    vi.mocked(mergeHindsightPolicy).mockImplementation((p) => ({
      ...p,
      network_policies: { ...p.network_policies, hindsight: { name: "hindsight", endpoints: [] } },
    }));
    vi.mocked(serializePolicy).mockReturnValue("version: 1\n");

    const { applyPluginConfig } = await import("./openclaw-config.js");
    vi.mocked(applyPluginConfig).mockResolvedValue(undefined);
  });

  it("runs all steps in order for a clean install", async () => {
    const { execFile } = await import("child_process");
    const calls: string[] = [];

    vi.mocked(execFile).mockImplementation((cmd, args, callback?: unknown) => {
      calls.push(`${cmd} ${(args as string[]).join(" ")}`);
      if (typeof callback === "function") {
        (callback as (err: null, result: { stdout: string; stderr: string }) => void)(null, {
          stdout: "",
          stderr: "",
        });
      }
      return {} as ReturnType<typeof execFile>;
    });

    const { runSetup } = await import("./setup.js");
    await runSetup(BASE_ARGS);

    expect(calls.some((c) => c.includes("which openshell"))).toBe(true);
    expect(calls.some((c) => c.includes("which openclaw"))).toBe(true);
    expect(
      calls.some((c) => c.includes("openclaw plugins install @vectorize-io/hindsight-openclaw"))
    ).toBe(true);
    expect(calls.some((c) => c.includes("openshell policy set my-assistant"))).toBe(true);
    expect(calls.some((c) => c.includes("openclaw gateway restart"))).toBe(true);
  });

  it("skips plugin install when --skip-plugin-install is set", async () => {
    const { execFile } = await import("child_process");
    const calls: string[] = [];
    vi.mocked(execFile).mockImplementation((cmd, args, callback?: unknown) => {
      calls.push(`${cmd} ${(args as string[]).join(" ")}`);
      if (typeof callback === "function") {
        (callback as (err: null, result: { stdout: string; stderr: string }) => void)(null, {
          stdout: "",
          stderr: "",
        });
      }
      return {} as ReturnType<typeof execFile>;
    });

    const { runSetup } = await import("./setup.js");
    await runSetup({ ...BASE_ARGS, skipPluginInstall: true });
    expect(calls.some((c) => c.includes("plugins install"))).toBe(false);
  });

  it("skips policy update when --skip-policy is set", async () => {
    const { runSetup } = await import("./setup.js");
    const { readSandboxPolicy } = await import("./policy-reader.js");
    await runSetup({ ...BASE_ARGS, skipPolicy: true });
    expect(vi.mocked(readSandboxPolicy)).not.toHaveBeenCalled();
  });

  it("skips policy set when Hindsight policy already exists", async () => {
    const { hasHindsightPolicy } = await import("./policy-writer.js");
    vi.mocked(hasHindsightPolicy).mockReturnValue(true);

    const { execFile } = await import("child_process");
    const calls: string[] = [];
    vi.mocked(execFile).mockImplementation((cmd, args, callback?: unknown) => {
      calls.push(`${cmd} ${(args as string[]).join(" ")}`);
      if (typeof callback === "function") {
        (callback as (err: null, result: { stdout: string; stderr: string }) => void)(null, {
          stdout: "",
          stderr: "",
        });
      }
      return {} as ReturnType<typeof execFile>;
    });

    const { runSetup } = await import("./setup.js");
    await runSetup(BASE_ARGS);
    expect(calls.some((c) => c.includes("openshell policy set"))).toBe(false);
  });

  it("does not execute any shell commands in dry-run mode", async () => {
    const { execFile } = await import("child_process");
    const { applyPluginConfig } = await import("./openclaw-config.js");
    const { writeFile } = await import("fs/promises");

    const { runSetup } = await import("./setup.js");
    await runSetup({ ...BASE_ARGS, dryRun: true });

    // which checks still run (preflight), but no actual commands
    const execCalls = vi
      .mocked(execFile)
      .mock.calls.map((c) => `${c[0]} ${(c[1] as string[]).join(" ")}`);
    expect(execCalls.some((c) => c.includes("plugins install"))).toBe(false);
    expect(execCalls.some((c) => c.includes("policy set"))).toBe(false);
    expect(execCalls.some((c) => c.includes("gateway restart"))).toBe(false);
    expect(vi.mocked(applyPluginConfig)).not.toHaveBeenCalled();
    expect(vi.mocked(writeFile)).not.toHaveBeenCalled();
  });

  it("fails early if openshell is not on PATH", async () => {
    const { execFile } = await import("child_process");
    vi.mocked(execFile).mockImplementation((cmd, args, callback?: unknown) => {
      if (cmd === "which" && (args as string[])[0] === "openshell") {
        if (typeof callback === "function") {
          (callback as (err: Error) => void)(new Error("not found"));
        }
      } else {
        if (typeof callback === "function") {
          (callback as (err: null, result: { stdout: string; stderr: string }) => void)(null, {
            stdout: "",
            stderr: "",
          });
        }
      }
      return {} as ReturnType<typeof execFile>;
    });

    const { runSetup } = await import("./setup.js");
    await expect(runSetup(BASE_ARGS)).rejects.toThrow("openshell");
  });
});
