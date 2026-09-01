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
import { existsSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { Config } from "./config";
import { log } from "./log";

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
 * "did the release number go up". A PRERELEASE suffix loses to the same numbers without one, which
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

    const latest = await latestVersion(opts.fetch ?? fetch);
    stampCheck(file, now, latest);
    if (!latest || !isNewer(latest, current)) return "";

    log.info("auto-update", `updating the Hindsight runtime ${current} -> ${latest}`);
    const child = (opts.spawn ?? realSpawn)("npx", ["-y", `${PACKAGE_NAME}@${latest}`, "update"], {
      detached: true,
      stdio: "ignore",
      windowsHide: true,
    });
    // A spawn failure (no npx on PATH, EACCES) arrives asynchronously as an 'error' event; an
    // unhandled one would take the session start down with it.
    child.on("error", (e) => log.warn("auto-update", `update spawn failed: ${e.message}`));
    child.unref();
    return latest;
  } catch {
    return ""; // an update check must never break a session
  }
}
