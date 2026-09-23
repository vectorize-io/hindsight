import { execFileSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import {
  linkSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";
import {
  acquireLease,
  LEASE_HEARTBEAT_MS,
  LEASE_STALE_MS,
  releaseLease,
  startLeaseHeartbeat,
  type SurveyLease,
} from "./survey-lease";

const FENCE_MARKER = "owner-token-lease-v2";
const DRAIN_MARKER = "legacy-drain-complete";
const DRAIN_TOKEN = "fence-token";
const PROCESS_START_SKEW_MS = 5_000;
const FENCE_RETRIES = 64;
const QUARANTINE_RETRIES = 64;
const PROCESS_REGISTRATION_RETRIES = 10;
const PROCESS_REGISTRATION_RETRY_MS = 25;

interface LegacyLock {
  pid?: unknown;
  ts?: unknown;
}

export interface ProcessIdentity {
  alive: boolean;
  startedAt?: number;
}

export interface DeepenLeaseOptions {
  heartbeatMs?: number;
  staleMs?: number;
  processIdentity?: (pid: number) => ProcessIdentity;
  listDeepenProcesses?: () => number[] | undefined;
  /** Test seam for simulating independent v2 process generations in one test process. */
  processId?: number;
  /** Test seam paired with `processId`. */
  currentProcessIdentity?: ProcessIdentity;
  /** Test seam for the bounded window in which a just-started v2 process registers itself. */
  waitForProcessRegistration?: (milliseconds: number) => void;
  onLost?: () => void;
  /** Test seam: an old runtime writes after one legacy generation was quarantined. */
  afterLegacyMoved?: () => void;
  /** Test seam: a quarantine pathname is replaced after discovery but before private inspection. */
  beforeQuarantineInspect?: (path: string) => void;
  /** Narrow fault-injection seam for quarantine I/O. */
  quarantineIo?: Partial<QuarantineIo>;
}

export interface QuarantineIo {
  readdir(path: string): string[];
  rename(from: string, to: string): void;
  link(from: string, to: string): void;
  unlink(path: string): void;
}

export interface DeepenLease {
  release(): void;
}

function errno(error: unknown): string | undefined {
  return (error as NodeJS.ErrnoException).code;
}

/** Process generation behind a legacy `{pid, ts}` lock. Start time distinguishes the original
 *  holder from an unrelated process that later reused its PID. */
export function systemProcessIdentity(pid: number): ProcessIdentity {
  try {
    process.kill(pid, 0);
  } catch (error) {
    return errno(error) === "ESRCH" ? { alive: false } : { alive: true };
  }
  try {
    if (process.platform === "linux") {
      return { alive: true, startedAt: statSync(`/proc/${pid}`).ctimeMs };
    }
    if (process.platform === "win32") {
      const script =
        `(Get-Process -Id ${pid} -ErrorAction Stop).StartTime.ToUniversalTime()` +
        ".Subtract([datetime]'1970-01-01').TotalMilliseconds";
      const value = Number(
        execFileSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", script], {
          encoding: "utf8",
          windowsHide: true,
        }).trim()
      );
      return Number.isFinite(value) ? { alive: true, startedAt: value } : { alive: true };
    }
    const value = Date.parse(
      execFileSync("ps", ["-p", String(pid), "-o", "lstart="], {
        encoding: "utf8",
        windowsHide: true,
        env: { ...process.env, LC_ALL: "C" },
      }).trim()
    );
    return Number.isFinite(value) ? { alive: true, startedAt: value } : { alive: true };
  } catch {
    // Liveness is still proven. Fail closed when process start time is unavailable: a later
    // session can retry, while guessing stale could overlap a genuine legacy deepener.
    return { alive: true };
  }
}

function readLegacy(path: string): LegacyLock | undefined {
  try {
    return JSON.parse(readFileSync(path, "utf8")) as LegacyLock;
  } catch {
    return undefined;
  }
}

type LegacyGenerationState = "live" | "stale" | "unknown";

function legacyGenerationState(
  lock: LegacyLock | undefined,
  identityOf: (pid: number) => ProcessIdentity
): LegacyGenerationState {
  if (
    !lock ||
    typeof lock.pid !== "number" ||
    !Number.isInteger(lock.pid) ||
    lock.pid <= 0 ||
    typeof lock.ts !== "number" ||
    !Number.isFinite(lock.ts)
  )
    return "unknown";
  const identity = identityOf(lock.pid);
  if (!identity.alive) return "stale";
  if (identity.startedAt === undefined) return "unknown";
  return identity.startedAt <= lock.ts + PROCESS_START_SKEW_MS ? "live" : "stale";
}

/** Same-user deepen processes other than this one. `undefined` means enumeration was not reliable,
 *  so migration must fail closed. During the one-time fence handoff, false positives only defer a
 *  run; missing a hidden legacy holder would duplicate ingestion. */
export function systemDeepenProcesses(): number[] | undefined {
  try {
    if (process.platform === "win32") {
      const script =
        "$p = Get-CimInstance Win32_Process | " +
        "Where-Object { $_.CommandLine -match '[\\\\/]deepen\\.js[\"'']?(?:\\s|$)' } | " +
        "Select-Object -ExpandProperty ProcessId; $p -join ','";
      const raw = execFileSync(
        "powershell.exe",
        ["-NoProfile", "-NonInteractive", "-Command", script],
        { encoding: "utf8", windowsHide: true }
      ).trim();
      if (!raw) return [];
      return raw
        .split(",")
        .map(Number)
        .filter((pid) => Number.isInteger(pid) && pid > 0 && pid !== process.pid);
    }
    const raw = execFileSync("ps", ["-x", "-o", "pid=,command="], {
      encoding: "utf8",
      windowsHide: true,
      env: { ...process.env, LC_ALL: "C" },
    });
    const pids: number[] = [];
    for (const line of raw.split("\n")) {
      const hit = line.match(/^\s*(\d+)\s+(.+)$/);
      if (!hit || !/(^|[\\/])deepen\.js["']?(?:\s|$)/.test(hit[2])) continue;
      const pid = Number(hit[1]);
      if (pid !== process.pid) pids.push(pid);
    }
    return pids;
  } catch {
    return undefined;
  }
}

function readFenceToken(path: string): string | undefined {
  try {
    if (!lstatSync(path).isDirectory()) return undefined;
    const parsed = JSON.parse(readFileSync(join(path, FENCE_MARKER), "utf8")) as {
      version?: unknown;
      token?: unknown;
    };
    return parsed.version === 2 && typeof parsed.token === "string" && parsed.token
      ? parsed.token
      : undefined;
  } catch {
    return undefined;
  }
}

function installFence(
  root: string,
  key: string,
  afterLegacyMoved?: () => void
): { directory: string; token?: string; quarantines: string[] } {
  const legacyPath = join(root, `deepen-${key}.lock`);
  const staging = mkdtempSync(join(root, "fence-"));
  const stagedToken = randomUUID();
  writeFileSync(join(staging, FENCE_MARKER), JSON.stringify({ version: 2, token: stagedToken }), {
    flag: "wx",
    mode: 0o600,
  });
  const quarantines: string[] = [];
  try {
    for (let attempt = 0; attempt < FENCE_RETRIES; attempt++) {
      try {
        renameSync(staging, legacyPath);
        return { directory: legacyPath, token: stagedToken, quarantines };
      } catch (error) {
        if (
          !(
            ["EEXIST", "ENOTEMPTY", "EPERM", "EACCES", "ENOTDIR", "EISDIR"] as Array<
              string | undefined
            >
          ).includes(errno(error))
        )
          return { directory: legacyPath, quarantines };
      }
      const existingToken = readFenceToken(legacyPath);
      if (existingToken) return { directory: legacyPath, token: existingToken, quarantines };
      try {
        if (!lstatSync(legacyPath).isFile()) return { directory: legacyPath, quarantines };
      } catch (error) {
        if (errno(error) === "ENOENT") continue;
        return { directory: legacyPath, quarantines };
      }
      const quarantine = join(root, `deepen-${key}.legacy-${randomUUID()}.lock`);
      try {
        // Atomic quarantine: whichever generation owns the canonical pathname at this instant is
        // the one moved. No later cleanup ever unlinks the shared canonical pathname.
        renameSync(legacyPath, quarantine);
        quarantines.push(quarantine);
        afterLegacyMoved?.();
      } catch (error) {
        if (errno(error) !== "ENOENT") return { directory: legacyPath, quarantines };
      }
    }
    return { directory: legacyPath, quarantines };
  } finally {
    rmSync(staging, { recursive: true, force: true });
  }
}

const DEFAULT_QUARANTINE_IO: QuarantineIo = {
  readdir: (path) => readdirSync(path),
  rename: (from, to) => renameSync(from, to),
  link: (from, to) => linkSync(from, to),
  unlink: (path) => unlinkSync(path),
};

function allQuarantines(root: string, key: string, io: QuarantineIo): string[] | undefined {
  const prefix = `deepen-${key}.legacy-`;
  try {
    return io
      .readdir(root)
      .filter((name) => name.startsWith(prefix))
      .map((name) => join(root, name));
  } catch {
    return undefined;
  }
}

/** Atomically take one quarantine into private custody, then inspect that exact generation. */
function quarantineGenerationBlocks(
  path: string,
  root: string,
  key: string,
  identityOf: (pid: number) => ProcessIdentity,
  io: QuarantineIo
): boolean {
  const moved = join(root, `deepen-${key}.legacy-custody-${randomUUID()}.lock`);
  try {
    io.rename(path, moved);
  } catch {
    // The source normally remains in the scanned namespace. Even ENOENT is uncertain: another
    // generation may have replaced or moved it, so this acquisition must retry from scratch.
    return true;
  }
  if (legacyGenerationState(readLegacy(moved), identityOf) !== "stale") {
    try {
      // Best-effort stable-name restore. Hard links are not required for safety: on filesystems
      // that reject them, custody remains under the same scanned prefix and blocks every retry.
      io.link(moved, path);
      io.unlink(moved);
    } catch {
      /* custody remains durably discoverable */
    }
    return true;
  }
  try {
    io.unlink(moved);
    return false;
  } catch {
    return true;
  }
}

function drainQuarantines(
  root: string,
  key: string,
  initial: string[],
  identityOf: (pid: number) => ProcessIdentity,
  io: QuarantineIo,
  beforeInspect?: (path: string) => void
): boolean {
  let queued = initial;
  for (let pass = 0; pass < QUARANTINE_RETRIES; pass++) {
    const discovered = allQuarantines(root, key, io);
    if (!discovered) return false;
    const paths = [...new Set([...queued, ...discovered])];
    queued = [];
    if (paths.length === 0) return true;
    for (const path of paths) {
      beforeInspect?.(path);
      if (quarantineGenerationBlocks(path, root, key, identityOf, io)) return false;
    }
  }
  return false;
}

type DrainMarkerState = "absent" | "valid" | "invalid";

function drainMarkerState(fence: string, token: string): DrainMarkerState {
  const marker = join(fence, DRAIN_MARKER);
  try {
    if (!lstatSync(marker).isDirectory()) return "invalid";
  } catch (error) {
    return errno(error) === "ENOENT" ? "absent" : "invalid";
  }
  try {
    return readdirSync(marker).join("\0") === DRAIN_TOKEN &&
      readFileSync(join(marker, DRAIN_TOKEN), "utf8") === token
      ? "valid"
      : "invalid";
  } catch {
    // Once a marker pathname exists, partial or unreadable state is never interpreted as absence.
    return "invalid";
  }
}

function publishDrainMarker(fence: string, token: string): boolean {
  const staging = mkdtempSync(join(fence, "legacy-drain-staging-"));
  try {
    writeFileSync(join(staging, DRAIN_TOKEN), token, { flag: "wx", mode: 0o600 });
    try {
      renameSync(staging, join(fence, DRAIN_MARKER));
      return true;
    } catch {
      return drainMarkerState(fence, token) === "valid";
    }
  } catch {
    return false;
  } finally {
    rmSync(staging, { recursive: true, force: true });
  }
}

function ownsLease(lease: SurveyLease): boolean {
  try {
    return lstatSync(join(lease.directory, lease.owner)).isFile();
  } catch {
    return false;
  }
}

function processMarker(root: string, pid: number, startedAt: number): string {
  return join(root, `deepen-v2-process-${pid}-${Math.trunc(startedAt)}.json`);
}

function registerV2Process(root: string, pid: number, identity: ProcessIdentity): boolean {
  if (!identity.alive || identity.startedAt === undefined) return false;
  const startedAt = Math.trunc(identity.startedAt);
  const marker = processMarker(root, pid, startedAt);
  const content = JSON.stringify({ version: 2, pid, startedAt });
  try {
    writeFileSync(marker, content, { flag: "wx", mode: 0o600 });
    return true;
  } catch (error) {
    if (errno(error) !== "EEXIST") return false;
    try {
      return readFileSync(marker, "utf8") === content;
    } catch {
      return false;
    }
  }
}

function isRegisteredV2Process(
  root: string,
  pid: number,
  identityOf: (pid: number) => ProcessIdentity
): boolean {
  const identity = identityOf(pid);
  if (!identity.alive || identity.startedAt === undefined) return false;
  const startedAt = Math.trunc(identity.startedAt);
  try {
    const parsed = JSON.parse(readFileSync(processMarker(root, pid, startedAt), "utf8")) as {
      version?: unknown;
      pid?: unknown;
      startedAt?: unknown;
    };
    return parsed.version === 2 && parsed.pid === pid && parsed.startedAt === startedAt;
  } catch {
    return false;
  }
}

function waitForProcessRegistration(milliseconds: number): void {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, milliseconds);
}

function remainingLegacyProcesses(
  root: string,
  listProcesses: () => number[] | undefined,
  identityOf: (pid: number) => ProcessIdentity,
  wait: (milliseconds: number) => void
): number[] | undefined {
  for (let attempt = 0; attempt <= PROCESS_REGISTRATION_RETRIES; attempt++) {
    const running = listProcesses();
    if (running === undefined) return undefined;
    const legacy = running.filter((pid) => !isRegisteredV2Process(root, pid, identityOf));
    if (legacy.length === 0 || attempt === PROCESS_REGISTRATION_RETRIES) return legacy;
    // A v2 runtime can become visible to `ps` just before it reaches registerV2Process. Give that
    // bounded startup window time to close; a genuine legacy process remains and still blocks.
    wait(PROCESS_REGISTRATION_RETRY_MS);
  }
  return undefined;
}

/**
 * Mixed-version admission for deepen. Current runtimes use an owner-token heartbeat lease under
 * `leases/`; old runtimes are permanently fenced by a nonempty directory at their regular-file
 * lock pathname. Every legacy file is atomically quarantined before inspection, so cleanup never
 * targets a concurrently replaced canonical pathname.
 */
export function acquireDeepenLease(
  root: string,
  key: string,
  options: DeepenLeaseOptions = {}
): DeepenLease | undefined {
  mkdirSync(root, { recursive: true, mode: 0o700 });
  const staleMs = options.staleMs ?? LEASE_STALE_MS;
  const heartbeatMs = options.heartbeatMs ?? LEASE_HEARTBEAT_MS;
  const identityOf = options.processIdentity ?? systemProcessIdentity;
  const listDeepenProcesses = options.listDeepenProcesses ?? systemDeepenProcesses;
  const processId = options.processId ?? process.pid;
  const currentIdentity = options.currentProcessIdentity ?? systemProcessIdentity(processId);
  const waitForRegistration = options.waitForProcessRegistration ?? waitForProcessRegistration;
  const io: QuarantineIo = { ...DEFAULT_QUARANTINE_IO, ...options.quarantineIo };
  // Registration is generation-specific and monotonic. It lets the one-time legacy drain ignore
  // concurrent v2 contenders without mistaking a later PID reuse for a current runtime.
  if (!registerV2Process(root, processId, currentIdentity)) return undefined;
  const lease: SurveyLease | undefined = acquireLease(join(root, "leases"), key, staleMs, "deepen");
  if (!lease) return undefined;

  let released = false;
  let lost = false;
  const lose = () => {
    if (lost) return;
    lost = true;
    options.onLost?.();
  };
  const stopHeartbeat = startLeaseHeartbeat(lease, { heartbeatMs, onLost: lose });
  const abort = () => {
    stopHeartbeat();
    releaseLease(lease);
  };

  const migration = installFence(root, key, options.afterLegacyMoved);
  if (!migration.token) {
    abort();
    return undefined;
  }

  // A valid marker is monotonic: it is token-bound to this permanent fence and can only be
  // published after legacy processes and quarantines have drained. Quarantines are still scanned
  // every time so unexpected filesystem artifacts always fail closed.
  const markerState = drainMarkerState(migration.directory, migration.token);
  if (markerState === "invalid") {
    abort();
    return undefined;
  }

  if (
    !drainQuarantines(
      root,
      key,
      migration.quarantines,
      identityOf,
      io,
      options.beforeQuarantineInspect
    )
  ) {
    abort();
    return undefined;
  }

  if (markerState === "absent") {
    // Legacy 0.6.1 wrote its lock non-exclusively, so two old contenders could both run while only
    // the last PID remained in the file. The fence now prevents new entrants; filter registered v2
    // generations and require every remaining same-user deepen.js process to drain exactly once.
    const legacy = remainingLegacyProcesses(
      root,
      listDeepenProcesses,
      identityOf,
      waitForRegistration
    );
    if (
      legacy === undefined ||
      legacy.length > 0 ||
      !ownsLease(lease) ||
      !publishDrainMarker(migration.directory, migration.token)
    ) {
      abort();
      return undefined;
    }
  }

  if (!ownsLease(lease)) {
    abort();
    return undefined;
  }

  return {
    release() {
      if (released) return;
      released = true;
      stopHeartbeat();
      releaseLease(lease);
    },
  };
}
