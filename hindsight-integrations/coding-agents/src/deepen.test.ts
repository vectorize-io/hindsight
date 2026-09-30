/**
 * deepen's per-bank lease (#4569): a run holds it for as long as it lives — not for a fixed
 * window — and a run whose lease was taken over stops instead of ingesting alongside the new one.
 */
import { spawn, type ChildProcess } from "node:child_process";
import { mkdtempSync, readdirSync, rmSync, unlinkSync } from "node:fs";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { buildSync } from "esbuild";
import { afterAll, afterEach, beforeAll, beforeEach, expect, it } from "vitest";

// Run the real entrypoint as its own process, bundled the way `git-stderr.test.ts` does it.
let buildDir: string;
let deepen: string;
beforeAll(() => {
  buildDir = mkdtempSync(join(tmpdir(), "deepen-build-"));
  deepen = join(buildDir, "deepen.cjs");
  buildSync({
    entryPoints: [join(__dirname, "deepen.ts")],
    bundle: true,
    platform: "node",
    format: "cjs",
    outfile: deepen,
    logLevel: "silent",
  });
});
afterAll(() => rmSync(buildDir, { recursive: true, force: true }));

let home: string;
let server: Server;
let apiUrl: string;
const children: ChildProcess[] = [];

beforeEach(async () => {
  home = mkdtempSync(join(tmpdir(), "deepen-lease-"));
  // An API that never answers keeps the first run busy — and holding its lease — indefinitely.
  server = createServer(() => {});
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  apiUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

afterEach(() => {
  for (const c of children) c.kill("SIGKILL");
  children.length = 0;
  server.closeAllConnections();
  server.close();
  rmSync(home, { recursive: true, force: true });
});

function runDeepen(): {
  child: ChildProcess;
  output: () => string;
  exited: Promise<number | null>;
} {
  let out = "";
  const child = spawn(
    process.execPath,
    [deepen, "--repo", __dirname, "--bank", "lease-test", "--api-url", apiUrl],
    // TMPDIR holds the lease root, HOME the config/logs: both isolated per test.
    { env: { ...process.env, TMPDIR: home, HOME: home }, stdio: ["ignore", "pipe", "pipe"] }
  );
  children.push(child);
  child.stdout.on("data", (d) => (out += d));
  child.stderr.on("data", (d) => (out += d));
  const exited = new Promise<number | null>((r) => child.once("exit", (code) => r(code)));
  return { child, output: () => out, exited };
}

const leaseDir = () => join(home, "hindsight-coding-agent", "deepen");

async function waitForLease(): Promise<string> {
  for (let i = 0; i < 100; i++) {
    try {
      const [lock] = readdirSync(leaseDir());
      const [owner] = lock ? readdirSync(join(leaseDir(), lock)) : [];
      if (owner) return join(leaseDir(), lock, owner);
    } catch {
      /* not taken yet */
    }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error("the first run never took the lease");
}

it("a second run is a no-op while the first is alive", async () => {
  const first = runDeepen();
  await waitForLease();
  const second = runDeepen();
  expect(await second.exited).toBe(0);
  expect(second.output()).toContain("another run holds the lease");
  expect(first.child.exitCode).toBeNull();
}, 30_000);

it("a run whose lease was taken over stops", async () => {
  const first = runDeepen();
  // Deleting the owner file is what a reclaim does to the previous holder.
  unlinkSync(await waitForLease());
  expect(await first.exited).toBe(0);
}, 30_000);
