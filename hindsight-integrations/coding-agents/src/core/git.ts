/**
 * Harness-agnostic git ingestion. Two paths:
 *  - `ingestGit`/`retainCommit`: per-commit, full message + full diff (opt-in, `--diffs`). Expensive
 *    (one extraction op per commit).
 *  - `ingestGitLog`/`gitLogText`: the last N commit MESSAGES ONLY, aggregated into ONE document
 *    (default). Cheap (one extraction op total).
 */
import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { projectNameOf } from "./bank";
import type { HindsightClient } from "./hindsight";
import type { RetainStamp } from "./retain-stamp";
import { pool } from "./util";

const US = "\x1f";
const RS = "\x1e"; // record separator between commits in gitLogText

function git(repo: string, ...args: string[]): string {
  return execFileSync("git", ["-C", repo, ...args], { encoding: "utf8", maxBuffer: 1 << 28 });
}

/** The bank-facing name for a repo — WORKTREE-AWARE (all worktrees produce the main checkout's
 *  name, mirroring bank resolution) and path-spelling-proof ("." names the directory). Document
 *  ids derived from this must be identical from every worktree of the same repo. */
export function repoNameOf(repo: string): string {
  return projectNameOf(resolve(repo));
}

/** HEAD's sha, or null on any failure (empty repo, not a repo). */
export function gitHeadSha(dir: string): string | null {
  try {
    return git(dir, "rev-parse", "HEAD").trim() || null;
  } catch {
    return null;
  }
}

/**
 * Count commits reachable from HEAD but NOT from `sinceSha` (`git rev-list --count sinceSha..HEAD`).
 * Branch-robust "new commits since a baseline": counts what's reachable from wherever HEAD is now, so
 * switching to a behind-branch yields 0 and a feature branch counts only its own new commits. Returns
 * null when `sinceSha` is unknown to the repo (rebased/gc'd/foreign) or on any git error — the caller
 * treats null as "not a reachable baseline" rather than "0 new".
 */
export function commitsSince(dir: string, sinceSha: string): number | null {
  try {
    const n = Number.parseInt(git(dir, "rev-list", "--count", `${sinceSha}..HEAD`).trim(), 10);
    return Number.isFinite(n) ? n : null;
  } catch {
    return null;
  }
}

/** True iff `dir` is a git repo with at least one commit. Fast (--max-count=1). False on non-git/empty/error. */
export function hasGitHistory(dir: string): boolean {
  try {
    return git(dir, "rev-list", "--max-count=1", "HEAD").trim().length > 0;
  } catch {
    return false;
  }
}

/**
 * Retain ONE commit (full message + full diff) under the `git` strategy. The document_id is `git:<sha>`,
 * so a later run (or the live git-sync) can tell what's already ingested and stays idempotent. Shared by
 * both the backfill and the incremental sync so the encoding is identical across the two entry points.
 */
export async function retainCommit(
  client: HindsightClient,
  repo: string,
  sha: string,
  repoName: string,
  stamp?: RetainStamp
): Promise<void> {
  const [h, an, ae, aISO, cISO, subj, body] = git(
    repo,
    "show",
    "-s",
    `--format=%H${US}%an${US}%ae${US}%aI${US}%cI${US}%s${US}%b`,
    sha
  ).split(US);
  const msg = (subj + (body?.trim() ? "\n\n" + body.trim() : "")).trim();
  const diff = git(repo, "show", "--format=", sha); // FULL diff, uncapped
  const content =
    `REF-ID: git:${sha.slice(0, 12)}\n` +
    `Git commit ${sha.slice(0, 12)} in the ${repoName} repository (${an}, ${aISO}).\n\n` +
    `Message:\n${msg}\n\nDiff:\n${diff}`;
  await client.retain(
    content,
    `git commit in ${repoName}`,
    `git:${sha}`,
    [...new Set([...(stamp?.tags ?? []), "source:git"])],
    "git",
    {
      timestamp: aISO, // the memory's timestamp is the commit's author date
      metadata: {
        ...stamp?.metadata,
        source: "git",
        repo: repoName,
        commit: h,
        short_sha: sha.slice(0, 12),
        author: an,
        author_email: ae,
        authored_at: aISO,
        committed_at: cISO,
        subject: subj,
      },
    }
  );
}

export async function ingestGit(
  client: HindsightClient,
  repo: string,
  opts: {
    limit?: number;
    concurrency?: number;
    log?: (m: string) => void;
    stampFor?: () => RetainStamp;
  } = {}
): Promise<number> {
  const log = opts.log ?? (() => {});
  // NEWEST-first: recent commits (the project's own decision commits) extract before the ancient
  // upstream noise, so the decisions that matter aren't starved at the tail of the extraction queue.
  let shas = git(repo, "rev-list", "HEAD").trim().split("\n").filter(Boolean);
  if (opts.limit) shas = shas.slice(0, opts.limit); // most recent N (validate the machine on a slice first)
  const repoName = repoNameOf(repo);
  log(`[git] ingesting ${shas.length} commits (full message + full diff, no filter) …`);
  let failures = 0;
  await pool(
    shas,
    opts.concurrency ?? 8,
    async (sha) => {
      await retainCommit(client, repo, sha, repoName, opts.stampFor?.());
    },
    (i, e) => {
      failures++;
      log(`  ! commit ${i} failed to enqueue: ${(e as Error).message?.slice(0, 120)}`);
    },
    (done) => {
      if (done % 25 === 0) log(`  ${done}/${shas.length}`);
    }
  );
  log(`[git] done: ${shas.length} commits ingested under strategy 'git'`);
  return failures;
}

/**
 * The last `limit` commits as an aggregated MESSAGES-ONLY block (no diffs) — one record per commit:
 * "<shortsha> <date> <author>\n<subject>\n\n<body>", newest first, separated by a divider line.
 * Thin wrapper over `git log`; empty repo (no commits) -> "".
 */
export function gitLogText(repo: string, limit: number): string {
  let raw: string;
  try {
    raw = git(
      repo,
      "log",
      `-n`,
      String(limit),
      "--no-merges",
      "--date=short",
      `--format=%h${US}%ad${US}%an${US}%s${US}%b${RS}`
    );
  } catch {
    return ""; // no commits yet (or not a git repo) — caller handles empty
  }
  return raw
    .split(RS)
    .map((rec) => rec.trim())
    .filter(Boolean)
    .map((rec) => {
      const [sha, date, author, subj, body] = rec.split(US);
      const header = `${sha} ${date} ${author}`;
      const msg = subj + (body?.trim() ? "\n\n" + body.trim() : "");
      return `${header}\n${msg}`;
    })
    .join("\n\n---\n\n");
}

/**
 * Ingest the aggregated commit-message history (last `opts.limit` commits, no diffs) as ONE document —
 * a single retain/extraction op, orders of magnitude cheaper than per-commit full-diff ingestion. The
 * document_id is `gitlog:<repoName>` so a re-seed replaces it (idempotent) rather than duplicating.
 * Tags keep `source:git` (alongside `source:git-log`) so the cold-repo check (`listDocumentIds("source:git")`)
 * still sees the bank as warm after a default (message-only) seed.
 */
export async function ingestGitLog(
  client: HindsightClient,
  repo: string,
  opts: { limit: number; log?: (m: string) => void; stampFor?: () => RetainStamp } = { limit: 300 }
): Promise<number> {
  const log = opts.log ?? (() => {});
  const text = gitLogText(repo, opts.limit);
  if (!text) {
    log("[gitlog] no commits found — skipping");
    return 0;
  }
  const repoName = repoNameOf(repo);
  const n = text.split("\n\n---\n\n").length;
  log(`[gitlog] ingesting last ${n} commit messages for ${repoName} as ONE document …`);
  try {
    // The gitlog-head:<sha> tag makes freshness a single tag query: the deepen engine re-upserts
    // this document (same id — replaces, never duplicates) only when HEAD has moved past it.
    const head = gitHeadSha(repo);
    const stamp = opts.stampFor?.();
    const tags = [
      ...new Set([
        ...(stamp?.tags ?? []),
        "source:git",
        "source:git-log",
        ...(head ? [`gitlog-head:${head}`] : []),
      ]),
    ];
    if (Object.keys(stamp?.metadata ?? {}).length) {
      await client.retain(
        text,
        `git commit-message history (last ${n}) for ${repoName}`,
        `gitlog:${repoName}`,
        tags,
        "gitlog",
        { metadata: stamp?.metadata }
      );
    } else {
      await client.retain(
        text,
        `git commit-message history (last ${n}) for ${repoName}`,
        `gitlog:${repoName}`,
        tags,
        "gitlog"
      );
    }
    log(`[gitlog] done: ${n} commit messages ingested as 1 document under strategy 'gitlog'`);
    return 0;
  } catch (e) {
    log(`  ! gitlog ingest failed: ${(e as Error).message?.slice(0, 120)}`);
    return 1;
  }
}
