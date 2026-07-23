/** Harness-agnostic git ingestion: every commit (full message + full diff) under the `git` strategy. */
import { execFileSync } from "node:child_process";
import type { HindsightClient } from "./hindsight";
import { pool } from "./util";

const US = "\x1f";

function git(repo: string, ...args: string[]): string {
  return execFileSync("git", ["-C", repo, ...args], { encoding: "utf8", maxBuffer: 1 << 28 });
}

/** The bank-facing name for a repo (its directory basename). */
export function repoNameOf(repo: string): string {
  return repo.replace(/\/+$/, "").split("/").pop() || "repo";
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
  repoName: string
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
  await client.retain(content, `git commit in ${repoName}`, `git:${sha}`, ["source:git"], "git", {
    timestamp: aISO, // the memory's timestamp is the commit's author date
    metadata: {
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
  });
}

export async function ingestGit(
  client: HindsightClient,
  repo: string,
  opts: { limit?: number; concurrency?: number; log?: (m: string) => void } = {}
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
      await retainCommit(client, repo, sha, repoName);
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
