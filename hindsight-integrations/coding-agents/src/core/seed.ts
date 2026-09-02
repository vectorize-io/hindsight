/**
 * Per-bank seed-consent state, for the auto-seed flow (Task 10).
 *
 * (Cold-repo detection itself lives in core/session-start.ts, which needs a tri-state result —
 * cold / warm / server-unreachable — that a plain boolean can't express.)
 *
 * Fail-safe throughout: state read/write never throws — a corrupt or unwritable state file must
 * not break the hook.
 */
import { spawn as realSpawn } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export const DEFAULT_SEED_LIMIT = 300;

/** Spawn a DETACHED background run of the deepen engine for `repoDir`. Idempotent server-side
 *  (per-bank lock + dedup by document id), so firing it every session start is safe — each run does
 *  only the missing work (cold seed, new conversations, the next diff-deepening batch). Fire-and-forget:
 *  the child outlives this process; extraction is server-side/async so nothing here blocks. Never throws. */
export function startBackgroundSeed(
  repoDir: string,
  opts: { limit?: number; harness?: string; enginePath?: string; spawn?: typeof realSpawn } = {}
): void {
  try {
    const spawnFn = opts.spawn ?? realSpawn;
    const enginePath =
      opts.enginePath ?? join(dirname(fileURLToPath(import.meta.url)), "deepen.js");
    const limit = opts.limit ?? DEFAULT_SEED_LIMIT;
    // The engine writes the leveled plugin.log itself (core/log.ts), so the child's stdio can be
    // fully detached — no fd redirect to manage.
    const child = spawnFn(
      "node",
      [
        enginePath,
        "--repo",
        repoDir,
        "--gitlog-limit",
        String(limit),
        ...(opts.harness ? ["--harness", opts.harness] : []),
      ],
      {
        detached: true,
        stdio: "ignore",
        windowsHide: true,
      }
    );
    // spawn() failures (ENOENT/EACCES/fd exhaustion/sandboxed environments) often arrive
    // ASYNCHRONOUSLY as an 'error' event on the child, not as a synchronous throw — an
    // unhandled 'error' event would crash the caller. Swallow it: fire-and-forget best-effort.
    child.on("error", () => {});
    child.unref();
  } catch {
    /* best-effort: a failed spawn must not break the caller */
  }
}

export interface SeedControlResult {
  ok: boolean;
  message: string;
}

/** Execute a seed-control command. Pure-ish: bank + connection are passed in; spawn + stateDir injectable. */
export function seedControl(
  command: string,
  args: {
    repo: string;
    bankId: string;
    limit?: number;
    spawn?: typeof realSpawn;
  }
): SeedControlResult {
  if (command === "seed") {
    startBackgroundSeed(args.repo, { limit: args.limit, spawn: args.spawn });
    return {
      ok: true,
      message: `🧠 Hindsight is learning this repo → memory bank “${args.bankId}”`,
    };
  }
  return { ok: false, message: "usage: hindsight-seed seed --repo <dir>" };
}
