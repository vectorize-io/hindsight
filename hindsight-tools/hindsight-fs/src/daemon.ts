/**
 * Background daemon management. `start` spawns a detached copy of this CLI in
 * `__run` mode (the hidden loop entrypoint), writing a pidfile and a log under
 * the mount's control directory. `stop` signals it; `status` reports liveness.
 */

import { promises as fs } from "node:fs";
import { openSync } from "node:fs";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import * as path from "node:path";
import { CONTROL_DIR, PID_FILE, LOG_FILE } from "./paths.js";
import { saveConfig, type MountConfig } from "./config.js";

interface PidRecord {
  pid: number;
  bankId: string;
  intervalSeconds: number;
  startedAt: string;
}

function controlPath(dir: string, file: string): string {
  return path.join(dir, CONTROL_DIR, file);
}

/** True if a process with `pid` is currently alive. */
export function isAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return (err as NodeJS.ErrnoException).code === "EPERM";
  }
}

export async function readPidRecord(dir: string): Promise<PidRecord | null> {
  try {
    return JSON.parse(await fs.readFile(controlPath(dir, PID_FILE), "utf8")) as PidRecord;
  } catch {
    return null;
  }
}

async function removePidFile(dir: string): Promise<void> {
  try {
    await fs.unlink(controlPath(dir, PID_FILE));
  } catch {
    /* ignore */
  }
}

/** Resolve the path to this CLI's entry script for re-spawning. */
function cliEntry(): string {
  // dist/daemon.js → dist/cli.js
  return path.join(path.dirname(fileURLToPath(import.meta.url)), "cli.js");
}

export interface StartResult {
  pid: number;
  alreadyRunning: boolean;
}

export async function startDaemon(config: MountConfig): Promise<StartResult> {
  const existing = await readPidRecord(config.dir);
  if (existing && isAlive(existing.pid)) {
    return { pid: existing.pid, alreadyRunning: true };
  }

  await fs.mkdir(path.join(config.dir, CONTROL_DIR), { recursive: true });
  await saveConfig(config);

  const logFd = openSync(controlPath(config.dir, LOG_FILE), "a");
  const child = spawn(process.execPath, [cliEntry(), "__run", "--dir", config.dir], {
    detached: true,
    stdio: ["ignore", logFd, logFd],
    env: process.env,
  });
  child.unref();

  if (child.pid === undefined) {
    throw new Error("Failed to spawn daemon process");
  }

  const record: PidRecord = {
    pid: child.pid,
    bankId: config.bankId,
    intervalSeconds: config.intervalSeconds,
    startedAt: new Date().toISOString(),
  };
  await fs.writeFile(
    controlPath(config.dir, PID_FILE),
    JSON.stringify(record, null, 2) + "\n",
    "utf8"
  );

  return { pid: child.pid, alreadyRunning: false };
}

export interface StopResult {
  stopped: boolean;
  pid: number | null;
}

export async function stopDaemon(dir: string): Promise<StopResult> {
  const record = await readPidRecord(dir);
  if (!record) return { stopped: false, pid: null };

  if (!isAlive(record.pid)) {
    await removePidFile(dir);
    return { stopped: false, pid: record.pid };
  }

  try {
    process.kill(record.pid, "SIGTERM");
  } catch {
    /* may have just exited */
  }

  // Give it a moment to exit, then force-kill if needed.
  for (let i = 0; i < 50 && isAlive(record.pid); i++) {
    await new Promise((r) => setTimeout(r, 100));
  }
  if (isAlive(record.pid)) {
    try {
      process.kill(record.pid, "SIGKILL");
    } catch {
      /* ignore */
    }
  }

  await removePidFile(dir);
  return { stopped: true, pid: record.pid };
}

export interface DaemonStatus {
  running: boolean;
  pid: number | null;
  record: PidRecord | null;
}

export async function daemonStatus(dir: string): Promise<DaemonStatus> {
  const record = await readPidRecord(dir);
  if (!record) return { running: false, pid: null, record: null };
  const running = isAlive(record.pid);
  if (!running) await removePidFile(dir);
  return { running, pid: running ? record.pid : null, record };
}

export function logPath(dir: string): string {
  return controlPath(dir, LOG_FILE);
}
