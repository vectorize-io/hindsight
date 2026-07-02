/**
 * Harness-agnostic incremental git sync — keeps a backfilled bank current with new commits.
 *
 * It diffs the commits reachable from a target ref (default `origin/main`, falling back to HEAD) against
 * the commit document_ids already in Hindsight (tag `source:git`, id `git:<sha>`) and async-retains ONLY
 * the missing commits, using the SAME per-commit encoding the backfill uses (see retainCommit). The check
 * is SET-BASED, so it is correct across rebases / force-push: stale commits simply remain in memory and
 * whatever the ref now points at gets ingested. Everything is best-effort and non-blocking — a sync
 * failure (offline, no remote, unconfigured bank) must never break the agent.
 */
import { execFileSync } from "node:child_process";
import type { HindsightClient } from "./hindsight";
import { retainCommit, repoNameOf } from "./git";
import { pool } from "./util";

/** Run git, returning trimmed stdout or null on ANY failure (not a repo, unknown ref, offline …). */
function gitTry(repo: string, ...args: string[]): string | null {
  try {
    return execFileSync("git", ["-C", repo, ...args], {
      encoding: "utf8",
      maxBuffer: 1 << 28,
    }).trim();
  } catch {
    return null;
  }
}

export interface SyncOpts {
  ref?: string; // preferred target ref (default origin/main; falls back to HEAD if absent)
  fetch?: boolean; // `git fetch` the ref's remote before diffing (default false: no network side effect)
  concurrency?: number; // parallel retains (default 4 — sync is background, keep it gentle)
  log?: (m: string) => void;
}

export interface SyncResult {
  ref: string | null; // the ref actually synced (null if the repo has no commits)
  total: number; // commits reachable from ref
  ingested: number; // new commits retained this run
  failures: number; // retains that failed to enqueue
  inSync: boolean; // true when nothing needed ingesting
}

/** The preferred ref if it resolves to a commit, else HEAD, else null (empty repo / not a git dir). */
function resolveRef(repo: string, preferred: string): string | null {
  if (gitTry(repo, "rev-parse", "--verify", "--quiet", `${preferred}^{commit}`) !== null)
    return preferred;
  if (gitTry(repo, "rev-parse", "--verify", "--quiet", "HEAD^{commit}") !== null) return "HEAD";
  return null;
}

export async function syncGit(
  client: HindsightClient,
  repo: string,
  opts: SyncOpts = {}
): Promise<SyncResult> {
  const log = opts.log ?? (() => {});
  const preferred = opts.ref || "origin/main";

  if (opts.fetch) {
    // `origin/main` -> fetch `main` from remote `origin`. Best-effort: ignore failures (offline / no remote).
    const m = /^([^/]+)\/(.+)$/.exec(preferred);
    if (m) gitTry(repo, "fetch", "--quiet", m[1], m[2]);
  }

  const ref = resolveRef(repo, preferred);
  if (!ref) {
    log("[sync] no git ref to sync");
    return { ref: null, total: 0, ingested: 0, failures: 0, inSync: true };
  }

  const revs = gitTry(repo, "rev-list", ref);
  const shas = (revs ? revs.split("\n") : []).filter(Boolean);
  if (!shas.length) return { ref, total: 0, ingested: 0, failures: 0, inSync: true };

  const ingestedIds = await client.listDocumentIds("source:git"); // Set of `git:<sha>` already in the bank
  const missing = shas.filter((sha) => !ingestedIds.has(`git:${sha}`));
  if (!missing.length) {
    log(`[sync] in sync — all ${shas.length} commits on ${ref} already ingested`);
    return { ref, total: shas.length, ingested: 0, failures: 0, inSync: true };
  }

  const repoName = repoNameOf(repo);
  log(
    `[sync] ${missing.length}/${shas.length} commits on ${ref} not in memory — retaining (async) …`
  );
  let failures = 0;
  await pool(
    missing,
    opts.concurrency ?? 4,
    async (sha) => {
      await retainCommit(client, repo, sha, repoName); // async retain: extraction enqueued server-side
    },
    () => {
      failures++;
    }
  );

  const ingested = missing.length - failures;
  log(
    `[sync] retained ${ingested} new commit(s) under 'git'` +
      (failures ? `, ${failures} failed to enqueue` : "")
  );
  return { ref, total: shas.length, ingested, failures, inSync: false };
}
