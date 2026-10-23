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
import { uuidV5 } from "./uuid";

const US = "\x1f";
const RS = "\x1e"; // record separator between commits in gitLogText
const GITLOG_HEAD = "gitlog-head:"; // tag naming the commit the git-log document was written at
const GITLOG_FORMAT = `--format=%h${US}%ad${US}%an${US}%s${US}%b${RS}`;
const GITLOG_DIVIDER = "\n\n---\n\n"; // between two commit records in the git-log document

function runGit(repo: string, args: string[], env?: NodeJS.ProcessEnv): string {
  // `log.showSignature=true` in the user's config makes `git log`/`git show` print signature
  // checks to stdout ahead of the formatted output, which we parse as data (#5269).
  return execFileSync("git", ["-C", repo, "-c", "log.showSignature=false", ...args], {
    encoding: "utf8",
    maxBuffer: 1 << 28,
    windowsHide: true,
    env,
    // execFileSync otherwise forwards failed git stderr into the host's TUI,
    // even when callers catch the error. Keep diagnostics on the error instead.
    stdio: ["ignore", "pipe", "pipe"],
  });
}

function git(repo: string, ...args: string[]): string {
  return runGit(repo, args);
}

/** For a sha read back from the bank, which may name a commit this clone never had: a partial
 *  clone would otherwise ask its remote for it, taking seconds on every run to fail (#5606). */
function gitLocal(repo: string, ...args: string[]): string {
  return runGit(repo, args, { ...process.env, GIT_NO_LAZY_FETCH: "1" });
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
 *
 * `sinceSha` is read back from the bank (survey-baseline document ids, the git-log document's
 * gitlog-head tag), so only a full object name reaches git: `git rev-list` parses an argument that
 * starts with "-" as an option, and `--output=<path>..HEAD` creates that file before failing.
 */
export function commitsSince(dir: string, sinceSha: string): number | null {
  if (!/^[0-9a-f]{40}(?:[0-9a-f]{24})?$/.test(sinceSha)) return null;
  try {
    const n = Number.parseInt(gitLocal(dir, "rev-list", "--count", `${sinceSha}..HEAD`).trim(), 10);
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
 * Up to the last `limit` commits as an aggregated MESSAGES-ONLY block (no diffs) — one record per
 * commit: "<shortsha> <date> <author>\n<subject>\n\n<body>", OLDEST first, separated by a divider
 * line. Thin wrapper over `git log`; empty repo (no commits) -> "".
 *
 * Built so a new commit only APPENDS to the previous text (#5003). The server's delta retain keeps
 * every chunk whose text is unchanged at the same index, so an append re-extracts only the tail.
 * This used to be the newest `limit` commits, newest first: every commit shifted the whole text,
 * no chunk survived, and the entire history was re-extracted and re-consolidated each time.
 * Flipping the order alone is not enough once the window is full — dropping the oldest commit
 * shifts the text again — so the window's START moves in fixed steps of half the limit (counted
 * from the root commit), and the document holds between limit/2 and `limit` commits. The history
 * is rewritten once per step instead of once per commit. Commits that land in the middle of the
 * date order (a merged branch, a rebase) still shift the text, which costs the full re-extraction
 * every commit used to.
 *
 * This text is now what a FULL write sends — the first seed, or a history the stored document no
 * longer describes. Keeping an existing document current appends only the new commits instead
 * (see appendGitLog), so neither the window's steps nor a merged branch rewrites what is stored.
 */
export function gitLogText(repo: string, limit: number): string {
  let raw: string;
  try {
    const total = Number(git(repo, "rev-list", "--count", "--no-merges", "HEAD").trim());
    const step = Math.max(1, Math.ceil(limit / 2));
    const start = total > limit ? Math.ceil((total - limit) / step) * step : 0;
    raw = git(
      repo,
      "log",
      `-n`,
      String(total - start),
      "--no-merges",
      "--reverse", // applied after -n: the newest N, printed oldest first
      "--date=short",
      GITLOG_FORMAT
    );
  } catch {
    return ""; // no commits yet (or not a git repo) — caller handles empty
  }
  return gitLogRecords(raw).join(GITLOG_DIVIDER);
}

/** `git log --format=GITLOG_FORMAT` output as one "<shortsha> <date> <author>\n<message>" record
 *  per commit, in the order git printed them. */
function gitLogRecords(raw: string): string[] {
  return raw
    .split(RS)
    .map((rec) => rec.trim())
    .filter(Boolean)
    .map((rec) => {
      const [sha, date, author, subj, body] = rec.split(US);
      const header = `${sha} ${date} ${author}`;
      const msg = subj + (body?.trim() ? "\n\n" + body.trim() : "");
      return `${header}\n${msg}`;
    });
}

/**
 * The records of the non-merge commits reachable from `head` but not from `baseSha`, OLDEST first —
 * exactly what the git-log document written at `baseSha` lacks, in the order they belong at its
 * end. Null when `baseSha` is not a full object name or not an ancestor of `head` (a rebase, a
 * force-push, a branch that diverged from the last writer): there the document holds commits
 * `head` does not, and only a full re-send describes its history.
 */
function gitLogRecordsSince(repo: string, baseSha: string, head: string): string[] | null {
  if (!/^[0-9a-f]{40}(?:[0-9a-f]{24})?$/.test(baseSha)) return null;
  try {
    gitLocal(repo, "merge-base", "--is-ancestor", baseSha, head);
  } catch {
    return null; // exit 1 = not an ancestor; anything else (unknown object) is no better
  }
  try {
    return gitLogRecords(
      git(
        repo,
        "log",
        "--no-merges",
        "--reverse",
        "--date=short",
        GITLOG_FORMAT,
        `${baseSha}..${head}`
      )
    );
  } catch {
    return null;
  }
}

/**
 * The author date (strict ISO 8601) of the newest commit `gitLogText` would include — i.e. the newest
 * non-merge commit reachable from HEAD, matching that function's traversal. This is the aggregated
 * document's temporal anchor: without it the retain API stamps "now" and the extraction prompt gets
 * `Event Date: Unknown`, so every extracted fact lands with a null occurred_start/occurred_end.
 * Null on an empty repo, a non-repo, or any git error — the caller then omits the timestamp.
 */
export function gitLogNewestAuthorDate(repo: string): string | null {
  try {
    return git(repo, "log", "-n", "1", "--no-merges", "--format=%aI").trim() || null;
  } catch {
    return null;
  }
}

type GitLogOpts = { limit: number; log?: (m: string) => void; stampFor?: () => RetainStamp };

/**
 * Write `content` to the repository's canonical git-log document — a replace, or with `append` an
 * append onto the stored body. Both paths go through here so the document's tags, context and
 * timestamp never depend on which one wrote last.
 */
async function retainGitLog(
  client: HindsightClient,
  repo: string,
  content: string,
  opts: GitLogOpts,
  head: string | null,
  append?: { operationId: string }
): Promise<void> {
  const repoName = repoNameOf(repo);
  // The gitlog-head:<sha> tag records the commit this document is written at: the deepen engine
  // writes it again only when HEAD has moved past that commit (see gitLogIsCurrent), and an
  // append starts from it (see appendGitLog). An append REPLACES the document's tags with these,
  // so the tag always names the newest commit the stored body holds — which is why the caller
  // passes the HEAD it built `content` from rather than this reading HEAD again.
  const stamp = opts.stampFor?.();
  const tags = [
    ...new Set([
      ...(stamp?.tags ?? []),
      "source:git",
      "source:git-log",
      ...(head ? [`${GITLOG_HEAD}${head}`] : []),
    ]),
  ];
  await client.retain(
    content,
    // No commit count: an append makes the document longer than any count a replace wrote.
    `git commit-message history for ${repoName}`,
    `gitlog:${repoName}`,
    tags,
    "gitlog",
    {
      // Anchor the document in time on the newest commit it contains, so extraction can date the
      // facts it pulls out of these messages. Previously no timestamp was passed at all: retain
      // then stamped "now", the extraction prompt got `Event Date: Unknown`, and every fact landed
      // with a null occurred_start/occurred_end — invisible to temporal search (#3602).
      timestamp: gitLogNewestAuthorDate(repo) ?? undefined,
      // `retain` only sets metadata when it is truthy, so an empty stamp sends none.
      metadata: Object.keys(stamp?.metadata ?? {}).length ? stamp?.metadata : undefined,
      ...(append ? { updateMode: "append" as const, operationId: append.operationId } : {}),
    }
  );
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
  opts: GitLogOpts = { limit: 300 }
): Promise<number> {
  const log = opts.log ?? (() => {});
  const head = gitHeadSha(repo);
  const text = gitLogText(repo, opts.limit);
  if (!text) {
    log("[gitlog] no commits found — skipping");
    return 0;
  }
  const repoName = repoNameOf(repo);
  const n = text.split(GITLOG_DIVIDER).length;
  log(`[gitlog] ingesting last ${n} commit messages for ${repoName} as ONE document …`);
  // A full write replaces the stored document, so commits that fell out of the window lose their
  // facts. Say so instead of leaving it to be noticed in the bank (#5708).
  const total = Number(git(repo, "rev-list", "--count", "--no-merges", "HEAD").trim());
  if (total > n) {
    log(
      `[gitlog] dropping the ${total - n} oldest commit(s): the document holds at most seedLimit (${opts.limit})`
    );
  }
  try {
    await retainGitLog(client, repo, text, opts, head);
    log(`[gitlog] done: ${n} commit messages ingested as 1 document under strategy 'gitlog'`);
    return 0;
  } catch (e) {
    log(`  ! gitlog ingest failed: ${(e as Error).message?.slice(0, 120)}`);
    return 1;
  }
}

/**
 * Bring the git-log document up to HEAD by APPENDING only the commits it lacks, oldest first.
 *
 * A replace re-sends the whole log for every new commit. The server's delta retain keeps a stored
 * chunk only when the text before it is unchanged, so whatever moves the text — a commit merged
 * into the middle of the date order, the window's start advancing — makes it re-extract the whole
 * log, every time. An append cannot move what is stored: the new records land after it, and only
 * they are extracted.
 *
 * Returns null when an append cannot describe HEAD, and the caller replaces instead: no document
 * or no gitlog-head tag (first seed), a recorded commit that is not an ancestor of HEAD (a rebase,
 * a force-push, a diverged worktree), more new commits than the seed `limit` (a replace bounded by
 * the limit is smaller), or a server that cannot append safely (no `operation_id` dedupe, or a bank
 * that does not store document text — see HindsightClient.supportsAppendRetain).
 */
async function appendGitLog(
  client: HindsightClient,
  repo: string,
  opts: GitLogOpts,
  head: string
): Promise<number | null> {
  const log = opts.log ?? (() => {});
  const canonical = `gitlog:${repoNameOf(repo)}`;
  const tags = await client.documentTags?.(canonical).catch(() => undefined);
  const base = tags?.find((t) => t.startsWith(GITLOG_HEAD))?.slice(GITLOG_HEAD.length);
  if (!base) return null;
  const records = gitLogRecordsSince(repo, base, head);
  if (records === null || records.length > opts.limit) return null;
  if (!records.length) {
    // Only merge commits since the recorded one: the log has nothing to add.
    log("[gitlog] no new non-merge commits — skipping");
    return 0;
  }
  const appendable = await client.supportsAppendRetain?.().catch(() => false);
  if (!appendable) return null;
  // The server joins an append onto the stored body with one "\n"; the leading divider makes the
  // join the same divider that separates every other pair of records.
  const content = GITLOG_DIVIDER.slice(1) + records.join(GITLOG_DIVIDER);
  // Same base + same new commits = same operation, so a re-submitted append is collapsed into the
  // first one server-side rather than applied twice.
  const operationId = uuidV5(`${client.bank}\n${canonical}\nappend\n${base}\n${content}`);
  log(`[gitlog] appending ${records.length} new commit message(s) to ${canonical} …`);
  try {
    await retainGitLog(client, repo, content, opts, head, { operationId });
    return 0;
  } catch (e) {
    log(`  ! gitlog append failed: ${(e as Error).message?.slice(0, 120)}`);
    return 1;
  }
}

/**
 * Whether this repository's canonical git-log document already reflects HEAD, so re-ingesting it
 * would add no commit it lacks. One tag query answers the usual case, a document written at HEAD.
 * Otherwise the document's gitlog-head tag names the commit it WAS written at, and it is still
 * current when HEAD is an ancestor of that commit (`commitsSince` counts 0): re-ingesting there
 * would only replace the log with an older one.
 *
 * The exact-HEAD query used to be the whole check. Every worktree of a repository writes the same
 * `gitlog:<repoName>` document, so returning to a worktree behind the last writer (A -> B -> A)
 * re-sent and re-extracted the whole log, although B's copy held every commit of A's (#4661). A HEAD
 * with a commit the document lacks (new work, a worktree that diverged) still reads as stale, and so
 * does a recorded commit this clone does not know (a same-named foreign repository, a rebase).
 * Server errors reject: the callers decide what an unreachable server means. `documentTags` is
 * optional so a client without it (SessionStart's minimal test clients) keeps the exact-HEAD check.
 */
export async function gitLogIsCurrent(
  client: Pick<HindsightClient, "listDocumentIds"> & Partial<Pick<HindsightClient, "documentTags">>,
  repo: string,
  head: string
): Promise<boolean> {
  const canonical = `gitlog:${repoNameOf(repo)}`;
  if ((await client.listDocumentIds(`${GITLOG_HEAD}${head}`, "all_strict")).has(canonical))
    return true;
  const tags = (await client.documentTags?.(canonical)) ?? [];
  return tags.some(
    (tag) => tag.startsWith(GITLOG_HEAD) && commitsSince(repo, tag.slice(GITLOG_HEAD.length)) === 0
  );
}

/**
 * Ensure this repository's canonical aggregated git-log document is current: append the commits
 * it lacks when it was written at an ancestor of HEAD (appendGitLog), replace it otherwise.
 *
 * Older deepen versions enumerated every `source:git-log` document in the bank and deleted every
 * non-canonical id. A bank can be shared by unrelated repositories, so that name comparison was
 * not an ownership check and could cascade-delete foreign memories (#3877). The canonical retain is
 * already an idempotent upsert; ambiguous legacy documents are deliberately left untouched.
 */
export async function syncGitLog(
  client: HindsightClient,
  repo: string,
  opts: GitLogOpts
): Promise<number> {
  const log = opts.log ?? (() => {});
  const head = gitHeadSha(repo);
  const current = head !== null && (await gitLogIsCurrent(client, repo, head).catch(() => false));
  if (current) {
    log("[gitlog] current with HEAD — skipping");
    return 0;
  }
  return (
    (head ? await appendGitLog(client, repo, opts, head) : null) ?? ingestGitLog(client, repo, opts)
  );
}
