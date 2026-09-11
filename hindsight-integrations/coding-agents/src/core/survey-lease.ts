/**
 * Single-flight admission for the codebase survey (#4255): at most one survey per destination
 * (API + credential + bank) runs at a time, however many SessionStart hooks race the cold check.
 *
 * The lease is a lock DIRECTORY holding exactly one owner file named by a random token, and it is
 * kept alive by a HEARTBEAT: the holder rewrites the owner file's mtime every few seconds, and a
 * lease whose mtime is older than the stale threshold belongs to a dead holder and may be taken.
 * Only filesystem primitives that behave the same on macOS, Linux and Windows are used — no
 * `flock`/`fcntl` (advisory, fd-bound, unreliable over NFS), no `O_EXCL` (broken on NFS), and no
 * PID liveness probes (PIDs are recycled, `kill(pid, 0)` answers for whatever process now has the
 * number, and it cannot be probed across users at all).
 *
 * A heartbeat needs a holder that runs OUR code, and the survey agent (`claude -p`, `codex exec`,
 * …) does not — while the hook that won the lease exits immediately. So the survey runs under a
 * tiny detached supervisor (`survey-supervisor.ts` → `superviseSurvey`) that holds the lease for
 * exactly the agent's lifetime: it heartbeats, releases when the agent exits, and kills the agent
 * if it finds the lease was taken from it (a machine that slept past the stale window), so a
 * reclaimed lease never leaves two surveys running.
 */
import { spawn as realSpawn, type ChildProcess } from "node:child_process";
import { randomUUID } from "node:crypto";
import {
  mkdirSync,
  mkdtempSync,
  readdirSync,
  renameSync,
  rmdirSync,
  rmSync,
  statSync,
  unlinkSync,
  utimesSync,
  writeFileSync,
} from "node:fs";
import { join } from "node:path";

/** How often the supervisor refreshes its lease. */
export const LEASE_HEARTBEAT_MS = 5_000;
/** A lease not refreshed for this long belongs to a dead holder. Six missed heartbeats: generous
 *  enough for a loaded machine, short enough that a crashed survey is retried on the next session. */
export const LEASE_STALE_MS = 30_000;

export interface SurveyLease {
  /** The lock directory: `<root>/survey-<key>.lock`. */
  directory: string;
  /** The owner file inside it — a random token, so a generation can only ever touch its own. */
  owner: string;
}

/** Environment variable carrying the JSON `SurveySupervisorSpec` from the hook to the supervisor.
 *
 * NOT argv: the spec holds the whole agent command line (survey prompt, inline MCP config,
 * `--disallowedTools Bash Write …`), and a node process launched by the hook with that on its own
 * command line was SIGKILLed by endpoint security within milliseconds of starting — 5/5 runs on a
 * SentinelOne-managed Mac, vs 0/5 for the identical launch with the spec in the environment (#4255
 * repro). The supervisor deletes it before starting the agent, so it goes no further. */
export const SURVEY_SPEC_ENV = "HINDSIGHT_SURVEY_SPEC";

/** What the hook hands the detached supervisor (via `SURVEY_SPEC_ENV`). */
export interface SurveySupervisorSpec {
  lease: SurveyLease;
  bin: string;
  args: string[];
  heartbeatMs?: number;
}

function tryRename(from: string, to: string): boolean {
  try {
    renameSync(from, to);
    return true;
  } catch {
    return false; // ENOTEMPTY / EEXIST / EPERM: someone else holds it
  }
}

/** Remove only this generation's owner file. A replacement lease is nonempty, so the rmdir that
 *  follows cannot take a newer owner with it even if one moved in between the two calls. */
export function releaseLease(lease: SurveyLease): void {
  try {
    unlinkSync(join(lease.directory, lease.owner));
  } catch {
    return; // already released or taken over: never touch an unknown generation
  }
  try {
    rmdirSync(lease.directory);
  } catch {
    /* a new owner is already there */
  }
}

/** Refresh the lease. `false` only when the owner file is gone — the lease was reclaimed from us.
 *  Any other failure (a transient EBUSY/EPERM on Windows) is retried on the next beat; if it
 *  persists the lease goes stale and the ENOENT arrives then. */
export function heartbeatLease(lease: SurveyLease): boolean {
  try {
    const now = new Date();
    utimesSync(join(lease.directory, lease.owner), now, now);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException).code !== "ENOENT";
  }
}

/** Clear the way for a new owner when the current lease is abandoned. Returns whether it may be
 *  retried; the retry's own rename is what decides between racing reclaimers. */
function reclaimIfStale(directory: string, staleMs: number): boolean {
  const owners = readdirSync(directory);
  if (owners.length === 0) {
    // A release interrupted between its unlink and its rmdir: unowned, and on Windows a rename
    // cannot replace even an empty directory.
    try {
      rmdirSync(directory);
    } catch {
      /* a new owner moved in; the retried rename will lose */
    }
    return true;
  }
  if (owners.length !== 1) return false; // not a layout we wrote: leave it alone
  const age = Date.now() - statSync(join(directory, owners[0])).mtimeMs;
  // A future mtime (the clock stepped back) is no proof of a live holder either: a live one
  // rewrites it with the current clock within one heartbeat.
  if (Math.abs(age) <= staleMs) return false;
  releaseLease({ directory, owner: owners[0] });
  return true;
}

/**
 * Take the lease for `key` under `root`, or `undefined` when a live holder has it. Never throws:
 * anything that prevents proving admission counts as "held" — skipping a survey is cheap,
 * launching a duplicate one is not.
 *
 * The owner file is written into a private staging directory which is then RENAMED into place: a
 * rename onto a nonempty directory fails on every platform, so exactly one contender wins, and
 * unlike `mkdir` there is no instant where the lock exists without its owner.
 */
export function acquireLease(
  root: string,
  key: string,
  staleMs: number = LEASE_STALE_MS
): SurveyLease | undefined {
  let staging: string | undefined;
  try {
    mkdirSync(root, { recursive: true, mode: 0o700 });
    staging = mkdtempSync(join(root, "claim-"));
    const owner = randomUUID();
    writeFileSync(join(staging, owner), "", { flag: "wx", mode: 0o600 });
    const directory = join(root, `survey-${key}.lock`);
    if (!tryRename(staging, directory)) {
      if (!reclaimIfStale(directory, staleMs)) return undefined;
      if (!tryRename(staging, directory)) return undefined; // another reclaimer won
    }
    return { directory, owner };
  } catch {
    return undefined;
  } finally {
    if (staging) rmSync(staging, { recursive: true, force: true });
  }
}

/**
 * Run the survey agent while holding `spec.lease` (the body of the detached supervisor). Resolves
 * once the agent is gone; the lease is released when the agent exits and left alone when it was
 * lost. `stop()` ends the survey early (the supervisor's signal handlers).
 */
export function superviseSurvey(
  spec: SurveySupervisorSpec,
  spawnFn: typeof realSpawn = realSpawn
): { done: Promise<void>; stop(): void } {
  let stop = () => {};
  const done = new Promise<void>((resolve) => {
    // The hook took the lease moments ago; if it is already gone, someone else is surveying.
    if (!heartbeatLease(spec.lease)) return resolve();
    let child: ChildProcess;
    try {
      child = spawnFn(spec.bin, spec.args, { stdio: "ignore", windowsHide: true });
    } catch {
      releaseLease(spec.lease);
      return resolve();
    }
    let finished = false;
    const finish = (release: boolean) => {
      if (finished) return;
      finished = true;
      clearInterval(timer);
      if (release) releaseLease(spec.lease);
      resolve();
    };
    const timer = setInterval(() => {
      if (heartbeatLease(spec.lease)) return;
      child.kill(); // the lease now belongs to another survey
      finish(false);
    }, spec.heartbeatMs ?? LEASE_HEARTBEAT_MS);
    child.on("error", () => finish(true));
    child.once("exit", () => finish(true));
    stop = () => {
      child.kill();
      finish(true);
    };
  });
  return { done, stop: () => stop() };
}
