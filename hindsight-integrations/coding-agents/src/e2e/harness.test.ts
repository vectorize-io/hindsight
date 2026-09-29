import { ChildProcess, execFileSync, spawn, spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import { expect, it, vi } from "vitest";
import { appendLogLine } from "../core/log";

vi.mock("node:child_process", async (importOriginal) => ({
  ...(await importOriginal<typeof import("node:child_process")>()),
  execFileSync: vi.fn(),
  spawn: vi.fn(),
  spawnSync: vi.fn(),
}));

it("creates host-owned, owner-only diagnostics before Docker starts and collects appended logs", async () => {
  const root = mkdtempSync(join(tmpdir(), "hs-harness-test-"));
  const configPath = join(root, "config.json");
  const diagnostics = '{"event":"retain"}\n';
  writeFileSync(configPath, JSON.stringify({ apiUrl: "http://localhost:8888" }));
  vi.stubEnv("HINDSIGHT_HARNESS_E2E", "1");
  vi.stubEnv("HINDSIGHT_E2E_CONFIG", configPath);
  vi.stubEnv("HINDSIGHT_E2E_API_URL", "http://localhost:8888");
  vi.stubEnv("HINDSIGHT_E2E_API_TOKEN", "");

  vi.mocked(spawnSync).mockImplementation((command, args) => {
    if (command === "mkdir") {
      for (const directory of args!.slice(1)) mkdirSync(directory, { recursive: true });
    }
    return { pid: 1, output: [], stdout: "", stderr: "", status: 0, signal: null };
  });
  vi.mocked(execFileSync).mockImplementation((command, args) => {
    if (command === "npm") writeFileSync(join(args![2], "plugin.tgz"), "");
    return Buffer.alloc(0);
  });
  vi.mocked(spawn).mockImplementation((command, args) => {
    expect(command).toBe("docker");
    expect(args![0]).toBe("run");
    expect(args).toContain("HINDSIGHT_DIAG_FILE=/results/diagnostics.jsonl");
    const mount = args!.find((argument) => argument.endsWith(",dst=/results"));
    expect(mount).toBeDefined();
    const resultDir = mount!.slice("type=bind,src=".length, -",dst=/results".length);
    const diagnosticsPath = join(resultDir, "diagnostics.jsonl");
    const before = statSync(diagnosticsPath);
    expect(before.uid).toBe(statSync(resultDir).uid);
    expect(before.mode & 0o777).toBe(0o600);
    expect(readFileSync(diagnosticsPath, "utf8")).toBe("");

    appendLogLine(diagnosticsPath, diagnostics);
    const after = statSync(diagnosticsPath);
    expect(after.uid).toBe(before.uid);
    expect(after.ino).toBe(before.ino);
    expect(after.mode & 0o777).toBe(0o600);

    const child = new ChildProcess();
    child.stdout = new PassThrough();
    child.stderr = new PassThrough();
    queueMicrotask(() => child.emit("close", 0));
    return child;
  });

  try {
    const { runHarnessE2e } = await import("./harness");
    const result = await runHarnessE2e({
      name: "test",
      hindsightHarness: "test",
      installCommand: "hindsight-coding-agents install test",
      command: () => ["test"],
    });
    expect(spawn).toHaveBeenCalledOnce();
    expect(result.diagnostics).toBe(diagnostics);
  } finally {
    vi.unstubAllEnvs();
    vi.resetAllMocks();
    rmSync(root, { recursive: true, force: true });
  }
});
