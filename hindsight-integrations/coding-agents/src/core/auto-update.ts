/**
 * Keep the STAGED runtime current on its own.
 *
 * `install` copies this package into ~/.hindsight/coding-agents and points every wired agent's
 * hooks at that copy (installer.ts `stageRuntime`). Nothing ever refreshed it: the only update
 * path was the user remembering to re-run `install`, so a machine could sit several versions
 * behind indefinitely — bugs stayed fixed only for people who happened to re-install.
 *
 * Once per `CHECK_INTERVAL_MS`, at session start, this asks the npm registry for the published
 * version and — when it is newer than the staged one — spawns a DETACHED
 * `npx @vectorize-io/hindsight-coding-agents@<version> update`, which re-stages the runtime and
 * touches no host config (see the `update` branch in installer.ts). Fire-and-forget: the current
 * session keeps running the version it already loaded and the next one starts on the new code.
 *
 * Deliberately narrow:
 *   - it runs ONLY from the staged copy. A checkout or an `npx` run is somebody's development or
 *     one-off invocation, and overwriting it with a published build would destroy their work.
 *   - it replaces ONLY a runtime it can prove npx downloaded (`installOrigin` below). A copy staged
 *     from `npm i -g`, from a project dependency, or from a local checkout belongs to whoever
 *     manages that source: updating it behind their back would leave `npm ls -g` reporting a
 *     version that is no longer what runs, or would silently overwrite a developer's built dist.
 *   - it needs `npx` on PATH, since that is how the updater is fetched. Without it there is
 *     nothing to spawn, so it says so once a day rather than failing a spawn each time.
 *   - it stages only. Rewiring hosts unattended would mean choosing which agents to install for,
 *     and that is the user's call (`install` spells it out for exactly this reason).
 *   - `autoUpdate: false` in ~/.hindsight/coding-agent.json (or HINDSIGHT_AUTO_UPDATE=false) turns
 *     it off entirely, for pinned or air-gapped setups.
 *
 * Known window: `stageRuntime` replaces dist/ wholesale, so a hook that happens to spawn during
 * that copy can fail to load its entry point. Already-running processes are unaffected (node has
 * read the bundle by then), the window is milliseconds once a day, and the cost of losing it is
 * one turn without memory — the same outcome as any other hook failure. Serialising against every
 * possible concurrent hook spawn would need a lock every hook takes on every turn, which is a
 * worse trade than the window it closes.
 */
import { spawn as realSpawn } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  realpathSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { Config } from "./config";
import { log } from "./log";
import { binOnPath } from "./util";

export const PACKAGE_NAME = "@vectorize-io/hindsight-coding-agents";

/** How often the registry is asked. One session a day pays a few hundred milliseconds; the rest
 *  read a timestamp off disk and move on. */
export const CHECK_INTERVAL_MS = 24 * 60 * 60 * 1000;

/** Registry call budget. A slow or unreachable registry must not delay a session start, and this
 *  runs before the user has typed anything. */
const FETCH_TIMEOUT_MS = 5000;

/** Where the last check's timestamp lives — inside the staged runtime, so it is removed with it. */
export function stateFile(runtimeDir: string): string {
  return join(runtimeDir, ".update-check.json");
}

/** The directory this module is running out of (the package root, one level above dist/). */
function packageRoot(): string {
  return join(dirname(fileURLToPath(import.meta.url)), "..");
}

/** Where `install` stages the runtime — the one copy this may replace. Kept as a literal rather
 *  than imported from installer.ts, which would pull the whole installer into every hook bundle. */
function stagedRuntimeDir(): string {
  return join(homedir(), ".hindsight", "coding-agents");
}

/** Same directory, compared through realpath — a symlinked or differently-spelled HOME must not
 *  read as "not the staged copy" and silently disable updates. */
function sameDir(a: string, b: string): boolean {
  try {
    return realpathSync(a) === realpathSync(b);
  } catch {
    return a === b;
  }
}

/** Version of the package this code belongs to, or "" when it cannot be read. */
export function stagedVersion(pkgRoot: string): string {
  try {
    const pkg = JSON.parse(readFileSync(join(pkgRoot, "package.json"), "utf8")) as {
      version?: string;
    };
    return typeof pkg.version === "string" ? pkg.version : "";
  } catch {
    return "";
  }
}

/**
 * Is `candidate` a later release than `current`?
 *
 * A deliberately small comparison rather than a semver dependency: this package ships
 * zero-dependency (the installer must run from a bare `npx`), and the only question asked here is
 * "did the release number go up". NOT `util.ts`'s `semverGte`, which answers a different question
 * for capability probes: it is true on equality and strips pre-release suffixes, so it would both
 * re-update a machine that is already current and drag a release onto its own release candidate. A PRERELEASE suffix loses to the same numbers without one, which
 * is what keeps a machine on `1.2.0` from being pulled onto `1.2.0-rc.1`; two prereleases of the
 * same version compare as equal, so neither drags the other around.
 */
export function isNewer(candidate: string, current: string): boolean {
  const parse = (v: string): { nums: number[]; pre: boolean } => {
    const [core = "", ...rest] = v.trim().split("-");
    return {
      nums: core.split(".").map((n) => Number.parseInt(n, 10)),
      pre: rest.length > 0,
    };
  };
  const a = parse(candidate);
  const b = parse(current);
  if (a.nums.length !== 3 || b.nums.length !== 3) return false;
  if (a.nums.some(Number.isNaN) || b.nums.some(Number.isNaN)) return false;
  for (let i = 0; i < 3; i++) {
    if (a.nums[i] !== b.nums[i]) return a.nums[i] > b.nums[i];
  }
  // Same numbers: only a release can supersede a prerelease of itself.
  return b.pre && !a.pre;
}

/** Whether enough time has passed since the last check. An unreadable/absent state file reads as
 *  "never checked", so a first run always checks and a corrupted one self-heals. */
function dueForCheck(file: string, now: number): boolean {
  try {
    const state = JSON.parse(readFileSync(file, "utf8")) as { lastCheck?: number };
    return typeof state.lastCheck !== "number" || now - state.lastCheck >= CHECK_INTERVAL_MS;
  } catch {
    return true;
  }
}

/** Written by installer.ts `stageRuntime`, naming the directory this runtime was copied from. */
const ORIGIN_FILE = ".install-origin.json";

/**
 * May this runtime be replaced automatically?
 *
 * Only when it was staged from an npx download — the documented install path, where no other tool
 * is tracking the version. `npm i -g` and a project dependency are managed by npm (an unattended
 * re-stage would leave `npm ls -g` naming a version that is no longer what runs, with no way for
 * the user to reconcile the two), and a checkout is a developer's build, which a published release
 * would silently overwrite.
 *
 * Missing marker = no. It is written on every `install`/`update` from the version that introduced
 * auto-update onward, and a machine has to re-install once to get this code at all — so a runtime
 * old enough to lack the marker is also too old to be running this check. Failing closed here
 * costs one manual install; failing open costs somebody their working tree.
 */
export function selfUpdatable(runtimeDir: string): boolean {
  try {
    const origin = JSON.parse(readFileSync(join(runtimeDir, ORIGIN_FILE), "utf8")) as {
      source?: string;
    };
    if (typeof origin.source !== "string" || !origin.source) return false;
    // npx unpacks into a `_npx/<hash>/node_modules/...` cache directory; nothing else does.
    return origin.source.split(/[\\/]/).includes("_npx");
  } catch {
    return false; // absent or unreadable — see above, this fails closed on purpose
  }
}

/**
 * Serialise updaters across concurrent sessions.
 *
 * The 24h stamp is not enough on its own: several agents starting within the same second all read
 * "due" before any of them has written it, so they all spawn an updater. Two `update` runs racing
 * means two `stageRuntime` calls, and staging is `rmSync(dist)` then `cpSync` — one process can
 * delete the directory the other is half way through writing, leaving a runtime with missing entry
 * points and every hook broken until a manual re-install. That burst is not hypothetical: this is
 * a plugin for machines that routinely run five agents at once.
 *
 * Same shape as deepen.ts's per-bank lock, and for the same reason — a TTL alone would wedge the
 * updater for its whole window after a crash, so the holder's pid decides liveness. The pid stored
 * is the detached CHILD's, because the copy happens in the child and outlives this process.
 *
 * In the OS temp dir, deliberately: it is scratch, and a reboot clearing it can only cost one
 * redundant check.
 */
const LOCK_STALE_MS = 10 * 60 * 1000;

function lockFile(): string {
  return join(tmpdir(), "hindsight-coding-agent", "auto-update.lock");
}

function acquireUpdateLock(file: string, now: number): boolean {
  try {
    const held = JSON.parse(readFileSync(file, "utf8")) as { pid?: number; ts?: number };
    if (held.ts && now - held.ts < LOCK_STALE_MS) {
      let holderAlive = false;
      if (held.pid) {
        try {
          process.kill(held.pid, 0);
          holderAlive = true;
        } catch {
          /* ESRCH: the holder died — the lock is stale NOW, not in LOCK_STALE_MS */
        }
      }
      if (holderAlive) return false;
    }
  } catch {
    /* no/unreadable lock — free */
  }
  try {
    mkdirSync(dirname(file), { recursive: true });
    writeFileSync(file, JSON.stringify({ pid: process.pid, ts: now }));
    return true;
  } catch {
    // Cannot claim the lock, so we cannot prove we are the only updater. Skip rather than race.
    return false;
  }
}

/** Hand the lock to the detached child, whose staging is what actually needs guarding. */
function holdLockFor(file: string, pid: number | undefined, now: number): void {
  try {
    if (pid === undefined) return releaseUpdateLock(file);
    writeFileSync(file, JSON.stringify({ pid, ts: now }));
  } catch {
    /* best-effort */
  }
}

function releaseUpdateLock(file: string): void {
  try {
    unlinkSync(file);
  } catch {
    /* already gone */
  }
}

/**
 * Stamp the check BEFORE acting on its result.
 *
 * The spawned update can fail — offline, a registry hiccup, a read-only home — and re-checking on
 * every session start until it succeeds would turn one broken machine into a request per session.
 * Recording the attempt bounds the retry to once per interval whatever the outcome.
 */
function stampCheck(file: string, now: number, latest: string): void {
  try {
    writeFileSync(file, JSON.stringify({ lastCheck: now, latest }));
  } catch {
    /* best-effort: an unwritable state file means we re-check next session, nothing worse */
  }
}

/** Ask the registry for the published version, or "" if it cannot be determined. */
async function latestVersion(fetchImpl: typeof fetch): Promise<string> {
  try {
    const r = await fetchImpl(`https://registry.npmjs.org/${PACKAGE_NAME}/latest`, {
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
      headers: { accept: "application/vnd.npm.install-v1+json" },
    });
    if (!r.ok) return "";
    const body = (await r.json()) as { version?: string };
    return typeof body.version === "string" ? body.version : "";
  } catch {
    return "";
  }
}

export interface AutoUpdateOptions {
  /** Package root to treat as "where this code runs from" (tests). */
  pkgRoot?: string;
  /** The staged runtime directory this may update (tests); defaults to ~/.hindsight/coding-agents. */
  runtimeDir?: string;
  /** Cross-process updater lock (tests); defaults to one in the OS temp dir. */
  lockFile?: string;
  /** Seams for the two ownership guards (tests). */
  selfUpdatable?: (runtimeDir: string) => boolean;
  binOnPath?: (bin: string) => boolean;
  spawn?: typeof realSpawn;
  fetch?: typeof fetch;
  now?: number;
}

/**
 * Check for a newer release and, if there is one, spawn the detached updater. Awaitable so tests
 * (and callers that want to) can observe it; production call sites fire and forget. Never throws.
 *
 * Returns the version an update was started for, or "" when nothing was done.
 */
export async function maybeAutoUpdate(
  cfg: Pick<Config, "autoUpdate">,
  opts: AutoUpdateOptions = {}
): Promise<string> {
  try {
    if (!cfg.autoUpdate) return "";
    // The survey's own headless session must not race the runtime out from under its parent.
    if (process.env.HINDSIGHT_DISABLE_HOOKS) return "";

    const pkgRoot = opts.pkgRoot ?? packageRoot();
    const runtime = opts.runtimeDir ?? stagedRuntimeDir();
    if (!existsSync(runtime)) return "";
    // Only the staged copy updates itself — see the module doc.
    if (!sameDir(pkgRoot, runtime)) return "";

    const now = opts.now ?? Date.now();
    const file = stateFile(runtime);
    if (!dueForCheck(file, now)) return "";

    const current = stagedVersion(pkgRoot);
    if (!current) return ""; // cannot tell what is installed — never guess and overwrite it
    // Both guards sit after the interval gate and stamp like any other "checked, nothing to do"
    // outcome, so each states its reason at most once a day instead of on every session start.
    if (!(opts.selfUpdatable ?? selfUpdatable)(runtime)) {
      log.info("auto-update", "runtime is managed outside npx — leaving its version alone");
      stampCheck(file, now, "");
      return "";
    }
    // Probed BEFORE the registry call: with no npx there is nothing to spawn, so asking npm for a
    // version we could not install anyway is a request for nothing.
    if (!(opts.binOnPath ?? binOnPath)("npx")) {
      log.info("auto-update", "npx is not on PATH — skipping the update check");
      stampCheck(file, now, "");
      return "";
    }

    // Claimed before the registry call, so a burst of simultaneous session starts makes ONE
    // request and can only ever produce one updater.
    const lock = opts.lockFile ?? lockFile();
    if (!acquireUpdateLock(lock, now)) return "";
    try {
      const latest = await latestVersion(opts.fetch ?? fetch);
      stampCheck(file, now, latest);
      if (!latest || !isNewer(latest, current)) {
        releaseUpdateLock(lock);
        return "";
      }

      log.info("auto-update", `updating the Hindsight runtime ${current} -> ${latest}`);
      const child = (opts.spawn ?? realSpawn)(
        "npx",
        ["-y", `${PACKAGE_NAME}@${latest}`, "update"],
        {
          detached: true,
          stdio: "ignore",
          windowsHide: true,
        }
      );
      // A spawn failure (no npx on PATH, EACCES) arrives asynchronously as an 'error' event; an
      // unhandled one would take the session start down with it.
      child.on("error", (e) => {
        log.warn("auto-update", `update spawn failed: ${e.message}`);
        releaseUpdateLock(lock);
      });
      child.unref();
      holdLockFor(lock, child.pid, now);
      return latest;
    } catch (e) {
      releaseUpdateLock(lock);
      throw e;
    }
  } catch {
    return ""; // an update check must never break a session
  }
}
