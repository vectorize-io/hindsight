#!/usr/bin/env node
/**
 * deepen — the background ingestion engine. NOT a user-facing CLI (no bin entry): the runtime
 * spawns it detached at every session start (core/seed.ts), and harnesses that need deterministic
 * ingestion (the benchmark, the live e2e suite) run it directly and poll `dist/status.js`.
 *
 * IDEMPOTENT and RESUMABLE — safe to fire every session; each run does only the missing work:
 *   1. configure the bank (missions + retain strategies; PUT/PATCH, no reset — a fresh bank IS the
 *      reset path)
 *   2. ingest conversations not yet in the bank (`--conversations` file via the harness's
 *      chatReader; dedup by `chat:<id>` document id — live sessions arrive via write-back, so this
 *      is history import, not sync)
 *   3. seed the aggregated commit-message history (ONE cheap document) if absent
 *   4. progressively DEEPEN: ingest the next batch of not-yet-ingested commits individually with
 *      their full diffs, NEWEST first (recent decisions matter most), up to DIFF_BATCH per run and
 *      DEEPEN_DIFF_TARGET total — full precision arrives across sessions without a big-bang ingest
 *   5. drain this run's extractions, then create the knowledge pages if the bank has none —
 *      pages-last makes `syncStatus().synced` a real completion marker
 *
 * A per-bank lock file makes concurrent session starts a no-op (stale locks expire).
 */
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { deriveBankId } from "./core/bank";
import { ingestChats } from "./core/chat";
import { applyBankConfig, loadConfig } from "./core/config";
import { commitsSince, gitHeadSha, ingestGitLog, repoNameOf, retainCommit } from "./core/git";
import { SURVEY_DOC_IDS } from "./core/survey";
import { buildPageTrigger } from "./core/missions";
import { HindsightClient } from "./core/hindsight";
import { DEEPEN_DIFF_TARGET } from "./core/status";
import { pool } from "./core/util";
import { getHarness, HARNESS_NAMES } from "./harness/registry";
import { diag } from "./core/diag";
import { buildRetainStamp } from "./core/retain-stamp";
import { describeError, log as plog, setLogLevel } from "./core/log";

const DIFF_BATCH = 50; // per-run cap on per-commit diff ingestion (bounded session cost)
const LOCK_STALE_MS = 30 * 60 * 1000;

function arg(name: string, def?: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  if (i >= 0 && i + 1 < process.argv.length) return process.argv[i + 1];
  return process.argv.includes(`--${name}`) ? "true" : def;
}

const REPO = arg("repo");
const cfg0 = loadConfig({ harness: arg("harness") ?? undefined, path: arg("config") });
const BANK =
  arg("bank") ?? (REPO ? deriveBankId(cfg0, REPO, arg("harness") ?? cfg0.harness) : cfg0.bankId);
const resolved0 = BANK
  ? applyBankConfig(cfg0, BANK, REPO ?? undefined)
  : { cfg: cfg0, bankId: BANK };
const cfg = resolved0.cfg;
const FINAL_BANK = resolved0.bankId ?? BANK;
if (cfg.disabled) {
  console.log(`deepen: bank ${FINAL_BANK} is disabled (banks override) — nothing to do`);
  process.exit(0);
}
const HARNESS = arg("harness") ?? cfg0.harness;
const API_URL = arg("api-url") ?? cfg.apiUrl;
const API_TOKEN = arg("api-token") ?? cfg.apiToken;
const CONV = arg("conversations");
const GITLOG_LIMIT = arg("gitlog-limit") ? Number(arg("gitlog-limit")) : (cfg.seedLimit ?? 300);
// harness override (benchmark/e2e want deterministic depth regardless of user config)
const GIT_INGEST =
  (["message", "full", "none"] as const).find((m) => m === arg("git-ingest")) ?? cfg.gitIngest;

if (!REPO || !BANK) {
  console.error(
    "usage: node deepen.js --repo <path> [--bank <id>] [--harness <name>] " +
      "[--conversations f.json] [--api-url U] [--api-token X] [--config path] [--gitlog-limit N] [--git-ingest message|full|none]\n" +
      `harnesses: ${HARNESS_NAMES.join(", ")}`
  );
  process.exit(1);
}

setLogLevel(cfg.logLevel);
// Foreground runs (benchmark/e2e) read stdout; background runs are followed via the leveled
// plugin.log — every engine line goes to both.
const log = (m: string) => {
  console.log(`${new Date().toISOString()} ${m}`);
  plog.info("deepen", m);
};

// ── per-bank lock: concurrent session starts must not double-ingest ─────────────
// Scratch, not state: the lock only guards against concurrent double-ingest cost. In the OS
// temp dir so ~/.hindsight holds ONLY the config file (a reboot clearing it is harmless).
const LOCK_DIR = join(tmpdir(), "hindsight-coding-agent");
const LOCK = join(LOCK_DIR, `deepen-${encodeURIComponent(FINAL_BANK ?? "")}.lock`);

function acquireLock(): boolean {
  try {
    const held = JSON.parse(readFileSync(LOCK, "utf8")) as { pid?: number; ts?: number };
    if (held.ts && Date.now() - held.ts < LOCK_STALE_MS) {
      // TTL alone is not enough: a killed run would block its bank for LOCK_STALE_MS. The lock
      // already records the holder's pid — if that process is gone, the lock is stale NOW.
      let holderAlive = false;
      if (held.pid) {
        try {
          process.kill(held.pid, 0);
          holderAlive = true;
        } catch {
          /* ESRCH: holder is dead — treat as stale */
        }
      }
      if (holderAlive) return false; // live run in progress
    }
  } catch {
    /* no/invalid lock — free */
  }
  try {
    mkdirSync(LOCK_DIR, { recursive: true });
    writeFileSync(LOCK, JSON.stringify({ pid: process.pid, ts: Date.now() }));
    return true;
  } catch {
    return false;
  }
}

async function main() {
  if (!acquireLock()) {
    log(`deepen: another run holds the lock for ${FINAL_BANK} — nothing to do`);
    return;
  }
  const t0 = Date.now();
  diag("deepen", "deepen_started", { bank: FINAL_BANK });
  try {
    const harness = await getHarness(HARNESS);
    const client = new HindsightClient({
      apiUrl: API_URL,
      apiToken: API_TOKEN,
      bank: FINAL_BANK!,
      // Names the repository in every seeded page's query, so page synthesis can tell this
      // project's decisions from those of a dependency it merely discusses (#3476). Same
      // worktree-aware name the gitlog document id uses, so all worktrees agree on it.
      project: repoNameOf(REPO!),
      maxParallelRetains: cfg.maxParallelRetains,
      observationScopes: cfg.observationScopes,
      log,
    });
    log(`deepen -> ${client.apiUrl} bank=${FINAL_BANK} harness=${harness.name}`);
    const stampFor = (sessionId?: string) =>
      buildRetainStamp(cfg, {
        directory: REPO!,
        harness: HARNESS,
        bankId: FINAL_BANK!,
        sessionId,
      });

    await client.configureBank({ pageTrigger: buildPageTrigger(cfg) });
    if (client.knowledgePagesSupported === false) {
      diag(harness.name, "knowledge_pages_unavailable", {
        bank: FINAL_BANK,
        apiUrl: client.apiUrl,
      });
    }

    const gitIds = await client.listDocumentIds("source:git");

    // chats FIRST: few, and they carry the decisions that make memory necessary — never starved
    // behind the git flood. Dedup against what's already in the bank (chat:<id>).
    const chatIds = await client.listDocumentIds("source:chat").catch(() => new Set<string>());
    const all = await harness.chatReader.read({ conversations: CONV, repo: REPO });
    const sessions = all.filter((s, i) => !chatIds.has(`chat:${s.id || `s${i}`}`));
    if (all.length !== sessions.length)
      log(`[chat] ${all.length - sessions.length} conversations already ingested — skipping those`);
    const chatFails = await ingestChats(client, sessions, {
      concurrency: cfg.maxParallelRetains,
      log,
      stampFor,
    });

    // ── git: seeding and syncing are the SAME code — this idempotent pass runs every session,
    // so "keep the bank current" is just "run it again". cfg.gitIngest picks the depth:
    //   none    → git contributes nothing
    //   message → ONE aggregated commit-message doc, re-upserted when HEAD moves (same doc id, so
    //             it replaces — the gitlog-head:<sha> tag makes freshness a single tag query)
    //   full    → message doc + progressive per-commit full diffs, newest first (new commits land
    //             at the top of rev-list, so the next run ingests them: that IS the sync)
    let gitFails = 0;
    if (GIT_INGEST === "none") {
      log("[git] gitIngest=none — git ingestion disabled");
    } else {
      const head = gitHeadSha(REPO!);
      const gitlogCurrent =
        head !== null &&
        (await client.listDocumentIds(`gitlog-head:${head}`).catch(() => new Set())).size > 0;
      if (gitlogCurrent) {
        log("[gitlog] current with HEAD — skipping");
      } else {
        gitFails += await ingestGitLog(client, REPO!, { limit: GITLOG_LIMIT, log, stampFor });
      }
      // Self-cleanup: earlier versions named the gitlog doc per WORKTREE (gitlog:my-repo-wt2 …),
      // duplicating the history in the shared bank. Delete any gitlog doc that isn't the
      // canonical (worktree-aware) id.
      try {
        const canonical = `gitlog:${repoNameOf(REPO!)}`;
        const logDocs = await client.listDocumentIds("source:git-log");
        for (const id of logDocs) {
          if (id !== canonical) {
            await client.deleteDocument(id);
            log(`[gitlog] removed stale duplicate ${id} (canonical: ${canonical})`);
          }
        }
      } catch {
        /* cleanup is best-effort */
      }

      if (GIT_INGEST === "full") {
        // progressive depth: next batch of un-ingested commits, newest first, full message + diff.
        try {
          const shas = execFileSync(
            "git",
            ["-C", REPO!, "rev-list", `-n`, String(DEEPEN_DIFF_TARGET), "HEAD"],
            { encoding: "utf8" }
          )
            .trim()
            .split("\n")
            .filter(Boolean)
            .filter((sha) => !gitIds.has(`git:${sha}`))
            .slice(0, DIFF_BATCH);
          if (shas.length) {
            const repoName = repoNameOf(REPO!);
            log(`[deepen] ingesting ${shas.length} commits with full diffs (newest first) …`);
            await pool(
              shas,
              cfg.maxParallelRetains,
              (sha) => retainCommit(client, REPO!, sha, repoName, stampFor()),
              () => {
                gitFails++;
              }
            );
          } else {
            log(`[deepen] recent history fully deepened (target ${DEEPEN_DIFF_TARGET})`);
          }
        } catch {
          log("[deepen] no git history — skipping diff deepening");
        }
      }
    }

    // Survey-marker cosmetics: once the survey's findings exist, flip the newest reachable
    // baseline marker from "researching…" to "completed" (lazy — the detached survey agent can't
    // reliably do it itself). The `survey-state:done` tag makes this a one-time upsert.
    try {
      const uploads = await client.listDocumentIds("source:upload").catch(() => new Set<string>());
      if (SURVEY_DOC_IDS.some((id) => uploads.has(id))) {
        const markers = await client.listDocumentIds("source:survey-baseline");
        const done = await client
          .listDocumentIds("survey-state:done")
          .catch(() => new Set<string>());
        let best: { id: string; sha: string; behind: number } | undefined;
        for (const id of markers) {
          if (done.has(id)) continue;
          const sha = id.replace(/^survey-baseline:/, "");
          const behind = commitsSince(REPO!, sha);
          if (behind !== null && (!best || behind < best.behind)) best = { id, sha, behind };
        }
        if (best) {
          const stamp = stampFor();
          const content =
            `✅ Codebase survey completed — baseline commit ${best.sha.slice(0, 12)}. ` +
            `(Internal marker: no memories are extracted from this document.)`;
          const tags = [...new Set([...stamp.tags, "source:survey-baseline", "survey-state:done"])];
          await client.retain(
            content,
            "hindsight codebase-survey baseline",
            best.id,
            tags,
            "survey",
            {
              // `retain` only sets metadata when it is truthy, so an empty stamp sends none.
              metadata: Object.keys(stamp.metadata).length ? stamp.metadata : undefined,
            }
          );
          log(`[survey] marker ${best.id} flipped to completed`);
        }
      }
    } catch {
      /* cosmetics — best-effort */
    }

    await client.drain(client.opIds, "extraction");

    // The drain above only covers operations THIS run enqueued — consolidation and the template's
    // page refreshes run server-side on their own schedule. `synced` requires ZERO active ops, so
    // wait (bounded) for the bank to fully settle before declaring the run complete.
    const settleDeadline = Date.now() + 15 * 60 * 1000;
    for (;;) {
      const active = await client.activeOperations().catch(() => 0);
      if (active === 0) break;
      if (Date.now() > settleDeadline) {
        log(`[deepen] ${active} server-side op(s) still active at settle timeout — proceeding`);
        break;
      }
      log(`[deepen] waiting for ${active} server-side op(s) to settle …`);
      await new Promise((r) => setTimeout(r, 5000));
    }
    // (knowledge pages need no separate pass: configureBank seeds them through the knowledge-base
    // API every run, matched by name — syncStatus's `synced` stays sound because it also requires
    // the gitlog seed present AND zero active extraction operations.)

    const failures = chatFails + gitFails;
    diag("deepen", "deepen_done", {
      bank: FINAL_BANK,
      ms: Date.now() - t0,
      newChats: sessions.length,
      failures,
    });
    log(
      `\n✅ deepen complete in ${((Date.now() - t0) / 1000).toFixed(1)}s${failures ? ` (${failures} items failed to enqueue)` : ""}.`
    );
  } finally {
    try {
      unlinkSync(LOCK);
    } catch {
      /* best-effort */
    }
  }
}

main().catch((e) => {
  diag("deepen", "deepen_failed", {
    bank: FINAL_BANK,
    error: describeError(e),
  });
  console.error("deepen failed:", (e as Error).message || e);
  try {
    unlinkSync(LOCK);
  } catch {
    /* best-effort */
  }
  process.exit(1);
});
