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
import type { KnowledgeNode } from "../src/client.js";

const CLI = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../dist/cli.js");
const isRoot = typeof process.getuid === "function" && process.getuid() === 0;

// ── Mock Hindsight API (knowledge-base tree + export) ──

/** Mutable bank contents; tests reassign these to simulate API changes. */
let bankTree: KnowledgeNode[] = [];
let bankContent: Record<string, string> = {};
let server: Server;
let baseUrl: string;

beforeAll(async () => {
  server = createServer((req, res) => {
    res.setHeader("content-type", "application/json");
    if (req.url && req.url.includes("/knowledge-base/tree")) {
      res.end(JSON.stringify({ roots: bankTree }));
    } else if (req.url && req.url.includes("/knowledge-base/export")) {
      const files = Object.entries(bankContent).map(([id, content]) => ({
        path: `${id}.md`,
        content,
      }));
      res.end(JSON.stringify({ files }));
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

function cli(args: string[]): Promise<RunResult> {
  return run("node", [CLI, ...args]);
}

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
// A small knowledge base: a "Profile" folder holding one page, plus a root page.

function sampleTree(): KnowledgeNode[] {
  return [
    {
      id: "profile",
      kind: "folder",
      name: "Profile",
      parent_id: null,
      mission: "Everything about the user",
      children: [
        {
          id: "user-preferences",
          kind: "page",
          name: "User Preferences",
          parent_id: "profile",
          children: [],
        },
      ],
    },
    { id: "project-status", kind: "page", name: "Project Status", parent_id: null, children: [] },
  ];
}

function sampleContent(): Record<string, string> {
  return {
    "user-preferences":
      "---\nid: user-preferences\ntype: knowledge-page\ntitle: User Preferences\n---\n\n" +
      "The user prefers dark mode and async, written communication.\n",
    "project-status":
      "---\nid: project-status\ntitle: Project Status\n---\n\n" +
      "Phase 2 is in progress; the API freeze is next week.\n",
  };
}

let dir: string;
const startedDirs = new Set<string>();

const PREFS = path.join("profile", "user-preferences.md");

beforeEach(async () => {
  dir = await fs.mkdtemp(path.join(os.tmpdir(), "hsfs-e2e-"));
  bankTree = sampleTree();
  bankContent = sampleContent();
});

afterEach(async () => {
  for (const d of startedDirs) await cli(["stop", d]).catch(() => {});
  startedDirs.clear();
  await fs.rm(dir, { recursive: true, force: true });
});

// ── Tests ──────────────────────────────────────────────

describe("e2e: real CLI + real bash", () => {
  it("mirrors the tree and the nested files work with ls / cat / grep / find / wc / head", async () => {
    const synced = await cli(["sync", dir, "--bank", "demo", "--api-url", baseUrl]);
    expect(synced.code).toBe(0);
    expect(synced.stdout).toContain("2 pages / 1 folders");

    // Folder became a directory; pages are at their nested paths.
    expect(await fileExists(path.join(dir, PREFS))).toBe(true);
    expect(await fileExists(path.join(dir, "project-status.md"))).toBe(true);

    // cat — frontmatter + body are real file content
    const cat = await sh(`cat "${path.join(dir, PREFS)}"`);
    expect(cat.stdout).toContain("id: user-preferences");
    expect(cat.stdout).toContain("dark mode and async");

    // grep -rl — content is searchable across the tree
    const grep = await sh(`grep -rl "API freeze" "${dir}"`);
    expect(grep.code).toBe(0);
    expect(grep.stdout.trim()).toBe(path.join(dir, "project-status.md"));

    // find — two page files (excluding the hidden control dir)
    const find = await sh(
      `find "${dir}" -name '*.md' -not -path '*/.hindsight-fs/*' | wc -l | tr -d ' '`
    );
    expect(find.stdout.trim()).toBe("2");

    // head + wc — ordinary text tooling
    const head = await sh(`head -1 "${path.join(dir, "project-status.md")}"`);
    expect(head.stdout.trim()).toBe("---");
    const wc = await sh(`wc -l < "${path.join(dir, PREFS)}" | tr -d ' '`);
    expect(Number(wc.stdout.trim())).toBeGreaterThan(4);
  });

  it("blocks an agent's write with read-only files", async () => {
    await cli(["sync", dir, "--bank", "demo", "--api-url", baseUrl]);
    const file = path.join(dir, PREFS);

    const stat = await sh(`ls -l "${file}" | cut -c1-10`);
    expect(stat.stdout.trim()).toBe("-r--r--r--");

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
    const file = path.join(dir, PREFS);

    await sh(`chmod u+w "${file}" && echo "HIJACKED" > "${file}"`);
    expect((await sh(`cat "${file}"`)).stdout.trim()).toBe("HIJACKED");

    const resync = await cli(["sync", dir]);
    expect(resync.stdout).toContain("1 reverted");
    expect((await sh(`cat "${file}"`)).stdout).toContain("dark mode and async");
    expect((await sh(`ls -l "${file}" | cut -c1-10`)).stdout.trim()).toBe("-r--r--r--");
  });

  it("prunes a file (and emptied folder) when its page is deleted from the bank", async () => {
    await cli(["sync", dir, "--bank", "demo", "--api-url", baseUrl]);
    expect(await fileExists(path.join(dir, PREFS))).toBe(true);

    // Remove the user-preferences page (and its folder) from the bank.
    bankTree = [
      { id: "project-status", kind: "page", name: "Project Status", parent_id: null, children: [] },
    ];
    delete bankContent["user-preferences"];
    const resync = await cli(["sync", dir]);
    expect(resync.stdout).toContain("1 removed");

    expect((await sh(`test -f "${path.join(dir, PREFS)}"; echo $?`)).stdout.trim()).toBe("1");
    expect((await sh(`test -d "${path.join(dir, "profile")}"; echo $?`)).stdout.trim()).toBe("1");
    expect(
      (await sh(`test -f "${path.join(dir, "project-status.md")}"; echo $?`)).stdout.trim()
    ).toBe("0");
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

    const appeared = await waitFor(() => fileExists(path.join(dir, PREFS)));
    expect(appeared).toBe(true);

    const healthy = await waitFor(async () => (await cli(["status", dir, "--json"])).code === 0);
    expect(healthy).toBe(true);
    const report = JSON.parse((await cli(["status", dir, "--json"])).stdout);
    expect(report.status).toBe("ok");
    expect(report.daemon.running).toBe(true);

    // Change the bank server-side → the daemon picks it up within an interval.
    bankContent["user-preferences"] =
      "---\nid: user-preferences\n---\n\nThe user now prefers LIGHT mode.\n";
    const picked = await waitFor(
      async () =>
        (await sh(`grep -c "LIGHT mode" "${path.join(dir, PREFS)}" || true`)).stdout.trim() === "1"
    );
    expect(picked).toBe(true);

    const stop = await cli(["stop", dir]);
    startedDirs.delete(dir);
    expect(stop.code).toBe(0);
    const dead = await cli(["status", dir, "--json"]);
    expect(dead.code).toBe(1);
    expect(JSON.parse(dead.stdout).status).toBe("dead");
  }, 20000);

  it("status exit codes: dead / stale / ok drive the exit code", async () => {
    const never = await cli(["status", dir, "--bank", "demo", "--api-url", baseUrl, "--json"]);
    expect(never.code).toBe(1);
    expect(JSON.parse(never.stdout).status).toBe("dead");

    await cli(["start", dir, "--bank", "demo", "--api-url", baseUrl, "--interval", "1"]);
    startedDirs.add(dir);
    await waitFor(() => fileExists(path.join(dir, PREFS)));

    const ok = await waitFor(async () => (await cli(["status", dir, "--json"])).code === 0);
    expect(ok).toBe(true);

    const stale = await cli(["status", dir, "--stale-after", "0", "--json"]);
    expect(stale.code).toBe(1);
    expect(JSON.parse(stale.stdout).status).toBe("stale");
  }, 20000);

  it("list prints folders + pages without writing any files", async () => {
    const list = await cli(["list", "--bank", "demo", "--api-url", baseUrl, "--dir", dir]);
    expect(list.code).toBe(0);
    expect(list.stdout).toContain("profile/");
    expect(list.stdout).toContain("profile/user-preferences.md");
    expect(list.stdout).toContain("project-status.md");
    // No files were written to the mount.
    const count = await sh(
      `find "${dir}" -name '*.md' -not -path '*/.hindsight-fs/*' 2>/dev/null | wc -l | tr -d ' '`
    );
    expect(count.stdout.trim()).toBe("0");
  });
});
