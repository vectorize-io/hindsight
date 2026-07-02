import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { promises as fs } from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { computeHealth } from "../src/health.js";
import { saveState, emptyState } from "../src/state.js";
import type { MountConfig } from "../src/config.js";

let dir: string;

function config(): MountConfig {
  return {
    dir,
    apiUrl: "http://localhost:8000",
    bankId: "acme",
    intervalSeconds: 10,
    writable: false,
  };
}

/** Write a daemon pidfile pointing at a live (or dead) process. */
async function writePid(pid: number, intervalSeconds = 10): Promise<void> {
  await fs.mkdir(path.join(dir, ".hindsight-fs"), { recursive: true });
  await fs.writeFile(
    path.join(dir, ".hindsight-fs", "daemon.pid"),
    JSON.stringify({ pid, bankId: "acme", intervalSeconds, startedAt: "2026-06-26T00:00:00Z" }),
    "utf8"
  );
}

async function writeLastSync(
  at: string | null,
  ok: boolean,
  error: string | null = null
): Promise<void> {
  const state = emptyState("acme", "http://localhost:8000");
  state.lastSyncAt = at;
  state.lastSyncOk = ok;
  state.lastError = error;
  state.files = { a: { file: "a.md", hash: "x" } };
  await saveState(dir, state);
}

beforeEach(async () => {
  dir = await fs.mkdtemp(path.join(os.tmpdir(), "hsfs-health-"));
});
afterEach(async () => {
  await fs.rm(dir, { recursive: true, force: true });
});

describe("computeHealth", () => {
  it("reports dead + unhealthy when no daemon is running", async () => {
    await writeLastSync("2026-06-26T10:00:00Z", true);
    const report = await computeHealth(config(), { now: Date.parse("2026-06-26T10:00:05Z") });
    expect(report.status).toBe("dead");
    expect(report.healthy).toBe(false);
    expect(report.daemon.running).toBe(false);
  });

  it("reports ok when daemon alive and sync is recent", async () => {
    await writePid(process.pid);
    await writeLastSync("2026-06-26T10:00:00Z", true);
    const report = await computeHealth(config(), { now: Date.parse("2026-06-26T10:00:05Z") });
    expect(report.status).toBe("ok");
    expect(report.healthy).toBe(true);
    expect(report.lastSync.ageSeconds).toBe(5);
  });

  it("reports stale when the last sync is older than the threshold", async () => {
    await writePid(process.pid, 10); // staleAfter = max(30, 15) = 30s
    await writeLastSync("2026-06-26T10:00:00Z", true);
    const report = await computeHealth(config(), { now: Date.parse("2026-06-26T10:01:00Z") }); // 60s
    expect(report.status).toBe("stale");
    expect(report.healthy).toBe(false);
  });

  it("reports failed when the daemon is alive but the last sync errored", async () => {
    await writePid(process.pid);
    await writeLastSync("2026-06-26T10:00:00Z", false, "HTTP 500");
    const report = await computeHealth(config(), { now: Date.parse("2026-06-26T10:00:05Z") });
    expect(report.status).toBe("failed");
    expect(report.healthy).toBe(false);
    expect(report.lastSync.error).toBe("HTTP 500");
  });

  it("reports stale when the daemon is up but has never synced", async () => {
    await writePid(process.pid);
    await writeLastSync(null, false);
    const report = await computeHealth(config(), { now: Date.parse("2026-06-26T10:00:05Z") });
    expect(report.status).toBe("stale");
    expect(report.lastSync.ageSeconds).toBeNull();
  });

  it("honors an explicit stale-after override", async () => {
    await writePid(process.pid);
    await writeLastSync("2026-06-26T10:00:00Z", true);
    const report = await computeHealth(config(), {
      now: Date.parse("2026-06-26T10:00:20Z"), // 20s old
      staleAfterSeconds: 10,
    });
    expect(report.status).toBe("stale");
  });
});
