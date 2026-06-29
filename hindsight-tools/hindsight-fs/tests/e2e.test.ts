/**
 * End-to-end integration tests.
 *
 * These spawn the *real* compiled CLI (`dist/cli.js`) against a *real* (mock)
 * HTTP server, and exercise the mirror with *real* bash commands — ls, cat,
 * grep, find, wc, head, stat — exactly how an agent or a human would use it.
 *
 * `pretest` builds dist/ before vitest runs, so the binary exists.
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from "vitest";
import { createServer, type Server } from "node:http";
import { execFile } from "node:child_process";
import { promises as fs } from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import type { MentalModel } from "../src/client.js";

const CLI = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../dist/cli.js");
const isRoot = typeof process.getuid === "function" && process.getuid() === 0;

// ── Mock Hindsight API ─────────────────────────────────

/** Mutable bank contents; tests reassign this to simulate API changes. */
let bankModels: MentalModel[] = [];
let server: Server;
let baseUrl: string;

beforeAll(async () => {
  server = createServer((req, res) => {
    if (req.url && req.url.includes("/mental-models")) {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ items: bankModels }));
    } else {
      res.statusCode = 404;
      res.end("{}");
    }
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  if (addr && typeof addr === "object") baseUrl = `http://127.0.0.1:${addr.port}`;
});

afterAll(async () => {
  await new Promise<void>((resolve) => server.close(() => resolve()));
});

// ── Process helpers ────────────────────────────────────

interface RunResult {
  code: number;
  stdout: string;
  stderr: string;
}

/** Run a command, always resolving with the exit code (never throws on non-zero). */
function run(cmd: string, args: string[], cwd?: string): Promise<RunResult> {
  // Strip ambient HINDSIGHT_* env so tests are hermetic.
  const env = { ...process.env };
  for (const k of Object.keys(env)) if (k.startsWith("HINDSIGHT_")) delete env[k];

  return new Promise((resolve) => {
    execFile(cmd, args, { cwd, env }, (err, stdout, stderr) => {
      const code =
        err && typeof (err as { code?: unknown }).code === "number"
          ? (err as { code: number }).code
          : err
            ? 1
            : 0;
      resolve({ code, stdout: stdout.toString(), stderr: stderr.toString() });
    });
  });
}

/** Invoke the hindsight-fs CLI. */
function cli(args: string[]): Promise<RunResult> {
  return run("node", [CLI, ...args]);
}

/** Run a bash one-liner (the "common bash operations" under test). */
function sh(command: string, cwd?: string): Promise<RunResult> {
  return run("bash", ["-c", command], cwd);
}

async function waitFor(
  predicate: () => Promise<boolean>,
  timeoutMs = 8000,
  stepMs = 150
): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await predicate()) return true;
    await new Promise((r) => setTimeout(r, stepMs));
  }
  return false;
}

async function fileExists(p: string): Promise<boolean> {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

// ── Fixtures ───────────────────────────────────────────

const SAMPLE: MentalModel[] = [
  {
    id: "user-preferences",
    bank_id: "demo",
    name: "User Preferences",
    source_query: "What are the user's preferences?",
    content: "The user prefers dark mode and async, written communication.",
    tags: ["ui", "comms"],
    last_refreshed_at: "2026-06-25T10:00:00Z",
    created_at: "2026-06-01T00:00:00Z",
  },
  {
    id: "project-status",
    bank_id: "demo",
    name: "Project Status",
    source_query: "What is the project status?",
    content: "Phase 2 is in progress; the API freeze is next week.",
    tags: ["project"],
  },
];

let dir: string;
const startedDirs = new Set<string>();

beforeEach(async () => {
  dir = await fs.mkdtemp(path.join(os.tmpdir(), "hsfs-e2e-"));
  bankModels = JSON.parse(JSON.stringify(SAMPLE));
});

afterEach(async () => {
  // Stop any daemon a test left running, then clean the dir.
  for (const d of startedDirs) await cli(["stop", d]).catch(() => {});
  startedDirs.clear();
  await fs.rm(dir, { recursive: true, force: true });
});

// ── Tests ──────────────────────────────────────────────

describe("e2e: real CLI + real bash", () => {
  it("syncs the bank and the files work with ls / cat / grep / find / wc / head", async () => {
    const synced = await cli(["sync", dir, "--bank", "demo", "--api-url", baseUrl]);
    expect(synced.code).toBe(0);
    expect(synced.stdout).toContain("2 mental models");

    // ls — both files present at the mount root
    const ls = await sh(`ls -1 "${dir}"`);
    expect(ls.code).toBe(0);
    const names = ls.stdout.trim().split("\n").sort();
    expect(names).toEqual(["project-status.md", "user-preferences.md"]);

    // cat — frontmatter + body are real file content
    const cat = await sh(`cat "${dir}/user-preferences.md"`);
    expect(cat.stdout).toContain("id: user-preferences");
    expect(cat.stdout).toContain("bank: demo");
    expect(cat.stdout).toContain("dark mode and async");

    // grep -rl — content is searchable across the folder
    const grep = await sh(`grep -rl "API freeze" "${dir}"`);
    expect(grep.code).toBe(0);
    expect(grep.stdout.trim()).toBe(path.join(dir, "project-status.md"));

    // find — only markdown files at the root (control data is hidden)
    const find = await sh(`find "${dir}" -maxdepth 1 -name '*.md' | wc -l | tr -d ' '`);
    expect(find.stdout.trim()).toBe("2");

    // head + wc — ordinary text tooling
    const head = await sh(`head -1 "${dir}/project-status.md"`);
    expect(head.stdout.trim()).toBe("---");
    const wc = await sh(`wc -l < "${dir}/user-preferences.md" | tr -d ' '`);
    expect(Number(wc.stdout.trim())).toBeGreaterThan(5);
  });

  it("blocks an agent's write with read-only files", async () => {
    await cli(["sync", dir, "--bank", "demo", "--api-url", baseUrl]);
    const file = path.join(dir, "user-preferences.md");

    // Permissions are r--r--r--
    const stat = await sh(`ls -l "${file}" | cut -c1-10`);
    expect(stat.stdout.trim()).toBe("-r--r--r--");

    // A naive append fails (skipped under root, which bypasses mode bits).
    if (!isRoot) {
      const write = await sh(`echo "AGENT EDIT" >> "${file}"`);
      expect(write.code).not.toBe(0);
      expect(write.stderr.toLowerCase()).toContain("permission denied");
      const after = await sh(`grep -c "AGENT EDIT" "${file}" || true`);
      expect(after.stdout.trim()).toBe("0");
    }
  });

  it("reverts a force-edited file on the next sync", async () => {
    await cli(["sync", dir, "--bank", "demo", "--api-url", baseUrl]);
    const file = path.join(dir, "user-preferences.md");

    // Force the file writable and hijack it (as a determined agent would).
    await sh(`chmod u+w "${file}" && echo "HIJACKED" > "${file}"`);
    expect((await sh(`cat "${file}"`)).stdout.trim()).toBe("HIJACKED");

    // Re-sync with identical API content → must still be reverted.
    const resync = await cli(["sync", dir]);
    expect(resync.stdout).toContain("1 reverted");
    const restored = await sh(`cat "${file}"`);
    expect(restored.stdout).toContain("dark mode and async");
    expect((await sh(`ls -l "${file}" | cut -c1-10`)).stdout.trim()).toBe("-r--r--r--");
  });

  it("prunes a file when its model is deleted from the bank", async () => {
    await cli(["sync", dir, "--bank", "demo", "--api-url", baseUrl]);
    expect(await fileExists(path.join(dir, "project-status.md"))).toBe(true);

    bankModels = bankModels.filter((m) => m.id !== "project-status");
    const resync = await cli(["sync", dir]);
    expect(resync.stdout).toContain("1 removed");

    expect((await sh(`test -f "${dir}/project-status.md"; echo $?`)).stdout.trim()).toBe("1");
    expect((await sh(`test -f "${dir}/user-preferences.md"; echo $?`)).stdout.trim()).toBe("0");
  });

  it("runs as a background daemon and refreshes files on an interval", async () => {
    const start = await cli([
      "start",
      dir,
      "--bank",
      "demo",
      "--api-url",
      baseUrl,
      "--interval",
      "1",
    ]);
    startedDirs.add(dir);
    expect(start.code).toBe(0);
    expect(start.stdout).toContain("in background");

    // Files appear shortly after starting.
    const appeared = await waitFor(() => fileExists(path.join(dir, "user-preferences.md")));
    expect(appeared).toBe(true);

    // status --json reports healthy with exit 0.
    const healthy = await waitFor(async () => (await cli(["status", dir, "--json"])).code === 0);
    expect(healthy).toBe(true);
    const status = await cli(["status", dir, "--json"]);
    const report = JSON.parse(status.stdout);
    expect(report.status).toBe("ok");
    expect(report.daemon.running).toBe(true);

    // Change the bank server-side → the daemon picks it up within an interval.
    bankModels[0].content = "The user now prefers LIGHT mode and phone calls.";
    const picked = await waitFor(
      async () =>
        (await sh(`grep -c "LIGHT mode" "${dir}/user-preferences.md" || true`)).stdout.trim() ===
        "1"
    );
    expect(picked).toBe(true);

    // Stop → status reports dead with non-zero exit.
    const stop = await cli(["stop", dir]);
    startedDirs.delete(dir);
    expect(stop.code).toBe(0);
    const dead = await cli(["status", dir, "--json"]);
    expect(dead.code).toBe(1);
    expect(JSON.parse(dead.stdout).status).toBe("dead");
  }, 20000);

  it("status exit codes: dead / stale / ok drive the exit code", async () => {
    // Never mounted → dead, exit 1.
    const never = await cli(["status", dir, "--bank", "demo", "--api-url", baseUrl, "--json"]);
    expect(never.code).toBe(1);
    expect(JSON.parse(never.stdout).status).toBe("dead");

    // Running daemon, but force the stale threshold to 0 → stale, exit 1.
    await cli(["start", dir, "--bank", "demo", "--api-url", baseUrl, "--interval", "1"]);
    startedDirs.add(dir);
    await waitFor(() => fileExists(path.join(dir, "user-preferences.md")));

    const ok = await waitFor(async () => (await cli(["status", dir, "--json"])).code === 0);
    expect(ok).toBe(true);

    const stale = await cli(["status", dir, "--stale-after", "0", "--json"]);
    expect(stale.code).toBe(1);
    expect(JSON.parse(stale.stdout).status).toBe("stale");
  }, 20000);

  it("list prints models without writing any files", async () => {
    const list = await cli(["list", "--bank", "demo", "--api-url", baseUrl, "--dir", dir]);
    expect(list.code).toBe(0);
    expect(list.stdout).toContain("user-preferences.md");
    expect(list.stdout).toContain("project-status.md");
    // No .md files were written to the mount root.
    const ls = await sh(`ls -1 "${dir}"/*.md 2>/dev/null | wc -l | tr -d ' '`);
    expect(ls.stdout.trim()).toBe("0");
  });
});
