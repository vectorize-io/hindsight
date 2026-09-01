/**
 * Shared `SessionStart` lifecycle: deterministically auto-seeds a cold repo's bank from its git
 * history (in the background, non-blocking), keeps warm banks deepening on every start, and
 * injects a short visible note plus the live
 * knowledge-page roster + guidance preamble that tells the agent to consult the repo's pages.
 *
 * Earlier design injected an instruction asking the AGENT to pose a y/n question to the user and
 * then run a seed command itself. Live testing showed that doesn't work: the model surfaces the
 * question, then plows ahead with the user's actual task and never runs the command — so nothing
 * ever seeds. Fix: the hook does the seeding itself. There is nothing left for the agent to decide
 * or execute for seeding, so no shell command (and no shell-escaping) is needed anymore.
 *
 * Tri-state cold check: a boolean "is it cold?" would collapse "cold" and "server unreachable"
 * into the same outcome, which is wrong here — a transient outage must not get treated the same
 * as "already seeded" and silently suppress seeding forever. So `buildSessionStartContext` calls
 * `client.listDocumentIds` directly so it can tell the three cases apart:
 *   - throws (server unreachable)      -> no seed, no state written  (try again next session)
 *   - non-empty set (warm/pre-seeded)  -> no seed, seededAt written  (remember, skip enumerating)
 *   - empty set (cold)                 -> start the background seed, seededAt written, note added
 */
import { readFileSync } from "node:fs";
import { gitHeadSha, hasGitHistory, commitsSince, repoNameOf } from "./git";
import { DEEPEN_DIFF_TARGET } from "./status";
import { startBackgroundSeed } from "./seed";
import { maybeAutoUpdate } from "./auto-update";
import { syncCompanionSkill } from "./skill-sync";
import { SURVEY_DOC_IDS, startCodebaseSurvey, type SurveyHarness } from "./survey";
import { applyBankConfig, loadConfig } from "./config";
import { DAEMON_WAIT_SESSION_START_MS, ensureDaemon } from "./daemon";
import type { Config } from "./config";
import { deriveBankIdOrSkip } from "./bank";
import { brandWord } from "./brand";
import { diag } from "./diag";
import { setLogLevel } from "./log";
import { parsePageList, buildKnowledgePreamble, type PageRef } from "./knowledge-injection";
import type { ClientOpts, RetainOpts } from "./hindsight";
import { buildRetainStamp } from "./retain-stamp";
import { HindsightClient } from "./hindsight";
import { sessionCacheFile, sessionRootDir, writeSessionCache } from "./session-cache";

/** Minimal client shape `buildSessionStartContext` needs. */
interface SeedContextClient {
  listDocumentIds(tag: string, tagsMatch?: "all" | "all_strict"): Promise<Set<string>>;
  listPages(): Promise<unknown>;
  knowledgePagesSupported?: boolean;
  // Optional: used to write the survey-baseline marker (Option A). HindsightClient has it; the
  // minimal test clients omit it, and the baseline write guards on its presence.
  retain?(
    content: string,
    context: string,
    documentId: string,
    tags: string[],
    strategy: string,
    opts?: RetainOpts
  ): Promise<void>;
}

/**
 * The session banner: one line — the gradient "Hindsight" wordmark (same colors as the API
 * server's banner) plus the repo's bank. Shown on EVERY session start; "learning" on a cold
 * repo (first ingest running), "remembering" once the bank is warm.
 */
export function buildSeedBanner(bankId: string, cold = true, gitNote?: string): string {
  // Two lines: what Hindsight DOES for this repo (value, not mechanism), then where the memory
  // lives + sync state. The brand word leads line 1 so the host TUI's message prefix lands there.
  const headline = cold
    ? `${brandWord()} is learning this repo — ingesting its decisions, conventions and history`
    : `${brandWord()} is tracking the decisions, conventions and history of this repo`;
  const details = `  ↳ memory bank “${bankId}”` + (gitNote ? ` · ${gitNote}` : "");
  return `${headline}\n${details}`;
}

/**
 * One-phrase git-sync state for the banner (the syncStatus contract, condensed): whether the bank
 * is current with the repo's commits. Cheap — reuses the cold-check's doc-id set plus ONE tag query
 * (gitlog-head:<sha>, the freshness marker the deepen engine maintains). Returns undefined when
 * there's nothing meaningful to say (gitIngest off, no git, cold bank — "learning" already covers it).
 */
async function gitSyncNote(args: {
  client: SeedContextClient;
  cwd: string;
  gitIds: Set<string>;
  mode: "message" | "full" | "none";
  cold: boolean;
}): Promise<string | undefined> {
  const { client, cwd, gitIds, mode, cold } = args;
  if (mode === "none" || cold) return undefined;
  const head = gitHeadSha(cwd);
  if (!head) return undefined;
  const gitlogCurrent = await client
    .listDocumentIds(`gitlog-head:${head}`, "all_strict")
    .then((s) => s.has(`gitlog:${repoNameOf(cwd)}`))
    .catch(() => undefined);
  if (gitlogCurrent === undefined) return undefined; // server hiccup: say nothing rather than guess
  if (mode === "message") return gitlogCurrent ? "git in sync" : "catching up on new commits";
  // full: deepening progress = per-commit docs vs the recent-history target
  const deepened = [...gitIds].filter((id) => id.startsWith("git:")).length;
  let target = DEEPEN_DIFF_TARGET;
  try {
    const { execFileSync } = await import("node:child_process");
    const n = Number(
      execFileSync("git", ["-C", cwd, "rev-list", "--count", "HEAD"], {
        encoding: "utf8",
        windowsHide: true,
      }).trim()
    );
    if (n > 0) target = Math.min(DEEPEN_DIFF_TARGET, n);
  } catch {
    /* keep the default target */
  }
  return gitlogCurrent && deepened >= target
    ? "git in sync"
    : `syncing git history (${Math.min(deepened, target)}/${target})`;
}

/** Split SessionStart output: `systemMessage` renders in the terminal (user-visible);
 *  `additionalContext` is injected into the model's context only. */
export interface SessionStartOutput {
  systemMessage?: string;
  additionalContext?: string;
  /** Internal lifecycle signal: defer one auto-reflect while a new bank gains useful content. */
  deferInitialReflect?: boolean;
}

/** Host-specific adapter for the shared SessionStart core. Claude and Codex use the
 * Claude-compatible envelope; Cursor's native hooks require snake-case `additional_context`. */
export interface SessionStartHookSpec {
  harness: string;
  parse(event: Record<string, unknown>): { cwd?: string; sessionId?: string };
  emit(output: SessionStartOutput): unknown;
}

/**
 * Build this session's additionalContext: (maybe) kick off a background auto-seed of a cold repo
 * and (always, unless the hook is disabled) append the knowledge-page bank mission. See module doc
 * for the tri-state cold check. Never throws.
 */
export async function buildSessionStartContext(args: {
  cwd: string;
  /** Where the session started — see `sessionRootDir`. Same as `cwd` on a fresh session; they
   *  differ on a resume, where the bank was resolved from the ORIGINAL root. */
  sessionRoot?: string;
  bankId: string;
  cfg: Config;
  client: SeedContextClient;
  harness?: string;
  stateDir?: string;
  hasGit?: (dir: string) => boolean;
  startSeed?: (repoDir: string, opts?: { limit?: number; harness?: string }) => void;
  startSurvey?: (
    repoDir: string,
    opts?: { harness?: SurveyHarness; model?: string; budgetUsd?: number }
  ) => void;
  headSha?: (dir: string) => string | null;
  commitsSince?: (dir: string, sinceSha: string) => number | null;
}): Promise<SessionStartOutput> {
  const { cwd, bankId, cfg, client, stateDir } = args;
  const t0 = Date.now();
  let cold: boolean | undefined; // undefined = not checked (autoSeed off / non-git / unreachable)
  let gitDocIds: Set<string> | undefined; // the cold-check's doc set, reused by the banner's git note
  const harness = args.harness ?? "claude-code";
  const hasGit = args.hasGit ?? hasGitHistory;
  const startSeed = args.startSeed ?? startBackgroundSeed;
  const startSurvey = args.startSurvey ?? startCodebaseSurvey;
  const resolveHeadSha = args.headSha ?? gitHeadSha;
  const countCommitsSince = args.commitsSince ?? commitsSince;
  // sessionRoot as well as cwd, so a stamped {gitProject} names the project the bank id does.
  const retainStamp = () =>
    buildRetainStamp(cfg, { directory: cwd, sessionRoot: args.sessionRoot, harness, bankId });

  // Codebase-survey baseline (Option A): the bank is the only state, so the HEAD at the last survey
  // is recorded IN the bank as a tiny `survey-baseline:<sha>` marker doc (tag source:survey-baseline;
  // content = the bare sha so extraction yields no facts), mirroring the deepen engine's gitlog-head
  // marker. The commit-count re-survey reads these back by tag.
  const SURVEY_BASELINE_TAG = "source:survey-baseline";
  const SURVEY_BASELINE_PREFIX = "survey-baseline:";
  const recordSurveyBaseline = (sha: string): void => {
    if (!client.retain) return; // minimal client (some tests) — nothing to record
    try {
      const stamp = retainStamp();
      const content =
        `🛰️ Hindsight is researching this codebase — survey started at commit ${sha.slice(0, 12)}. ` +
        `(Internal marker: no memories are extracted from this document.)`;
      const tags = [...new Set([...stamp.tags, SURVEY_BASELINE_TAG])];
      // Fire-and-forget; Promise.resolve tolerates a non-Promise return (e.g. a test spy).
      const retained = client.retain(
        content,
        "hindsight codebase-survey baseline",
        `${SURVEY_BASELINE_PREFIX}${sha}`,
        tags,
        "survey",
        // `retain` only sets metadata when it is truthy, so an empty stamp sends none.
        { metadata: Object.keys(stamp.metadata).length ? stamp.metadata : undefined }
      );
      void Promise.resolve(retained).catch(() => {});
    } catch {
      /* best-effort — a failed baseline write never breaks SessionStart */
    }
  };

  // The banner is USER-FACING and must ride `systemMessage` (the only hook field Claude Code
  // renders in the terminal); `additionalContext` is model-only and would show the human nothing.
  // The knowledge preamble is model context and stays in `additionalContext`.
  let systemMessage: string | undefined;

  if (cfg.autoSeed !== false) {
    if (hasGit(cwd)) {
      // The LIVE bank is the ONLY state (cold-check-wins): delete the bank and the next session is
      // a true first-open — no client-side flags can contradict it. Opting out of memory is the
      // `disabled` config field (global or per-harness), not a hidden state file.
      {
        let docIds: Set<string> | undefined;
        try {
          docIds = await client.listDocumentIds("source:git", "all_strict");
        } catch {
          docIds = undefined; // server unreachable: transient — do nothing, try again next session
        }

        cold = docIds !== undefined ? docIds.size === 0 : undefined;
        gitDocIds = docIds;
        if (docIds !== undefined) {
          // ALWAYS fire the background deepen engine when the server is reachable — it is
          // idempotent (per-bank lock, dedup by document id) and each run does only the missing
          // work: cold seed, newly appeared conversations, the next per-commit diff batch. The
          // one-time extras stay cold-gated below.
          // Pass the ASKING harness through: without it deepen.js falls back to the config
          // loader's harness default and misfiles this session's survey + git history into the
          // wrong bank (e.g. `opencode::<project>` for a claude-code session). See #3247.
          startSeed(cwd, { limit: cfg.seedLimit, harness });
          // Cold iff the bank has zero source:git docs (an undefined result — server error — is
          // NOT treated as cold; we never surveyed/noted on an unconfirmed-empty bank).
          if (docIds.size === 0) {
            if (cfg.codebaseSurvey !== false) {
              // Run the survey under the current harness's own CLI (falls back to any available agent).
              startSurvey(cwd, {
                harness: harness as SurveyHarness,
                model: cfg.surveyModel,
                budgetUsd: cfg.surveyBudgetUsd,
              });
              const sha = resolveHeadSha(cwd);
              if (sha) recordSurveyBaseline(sha); // baseline for the commit-count re-survey below
            }
            diag(harness, "seed_started", { bank: bankId });
          } else if (cfg.codebaseSurvey !== false && cfg.surveyRefreshCommits > 0) {
            // WARM bank: re-run the survey once enough commits have accrued since the last one, so
            // structural pages track an evolving architecture. Branch-robust: read the
            // survey-baseline:<sha> markers and take the MIN reachable `sha..HEAD` count (= the most
            // recent survey that's an ancestor of HEAD; dead-branch/rebased markers are unreachable →
            // ignored). No reachable baseline yet (pre-feature bank, or all markers gone) => record
            // HEAD as the baseline WITHOUT surveying (no surprise spend).
            const sha = resolveHeadSha(cwd);
            if (sha) {
              const markers = await client
                .listDocumentIds(SURVEY_BASELINE_TAG, "all_strict")
                .catch(() => new Set<string>());
              const counts: number[] = [];
              for (const id of markers) {
                if (!id.startsWith(SURVEY_BASELINE_PREFIX)) continue;
                const n = countCommitsSince(cwd, id.slice(SURVEY_BASELINE_PREFIX.length));
                if (n !== null) counts.push(n);
              }
              const sinceLast = counts.length ? Math.min(...counts) : null;
              // A baseline without FINDINGS means the surveyed agent died before ingesting (no
              // CLI on PATH, budget kill) — the marker alone must not suppress retries forever.
              const uploads = await client
                .listDocumentIds("source:upload", "all_strict")
                .catch(() => new Set<string>());
              const findingsAbsent =
                counts.length > 0 && !SURVEY_DOC_IDS.some((id) => uploads.has(id));
              if ((sinceLast !== null && sinceLast >= cfg.surveyRefreshCommits) || findingsAbsent) {
                startSurvey(cwd, {
                  harness: harness as SurveyHarness,
                  model: cfg.surveyModel,
                  budgetUsd: cfg.surveyBudgetUsd,
                });
                recordSurveyBaseline(sha);
                diag(harness, "survey_refresh", {
                  bank: bankId,
                  commits: sinceLast,
                  retry: findingsAbsent,
                });
              } else if (sinceLast === null) {
                recordSurveyBaseline(sha); // first baseline, or reset after a rebase — no survey
              }
            }
          }
        }
      }
    }
  }

  // An actual empty roster, like a cold git-doc result, means this bank has no useful material to
  // synthesize yet. A failed roster request is NOT evidence of that, so keep the two cases apart.
  let pages: PageRef[] = [];
  let pageListKnown = false;
  try {
    pages = parsePageList(await client.listPages());
    pageListKnown = true;
  } catch {
    if (client.knowledgePagesSupported === false) {
      diag(harness, "knowledge_pages_unavailable", { bank: bankId });
    }
    /* fail-open preamble; preserve first-prompt reflect eligibility on a transient outage */
  }
  const additionalContext = buildKnowledgePreamble(pages, { reflectOnNewGoals: !cfg.autoReflect });
  const deferInitialReflect = cold === true || (pageListKnown && pages.length === 0);

  // The banner shows on EVERY session — Hindsight's presence is part of the product, not a
  // one-time setup note. Wording tracks the bank state: cold = "learning", else "remembering";
  // warm banners also condense the syncStatus contract into a git-sync phrase.
  let gitNote: string | undefined;
  if (gitDocIds) {
    gitNote = await gitSyncNote({
      client,
      cwd,
      gitIds: gitDocIds,
      mode: cfg.gitIngest,
      cold: cold === true,
    }).catch(() => undefined);
  }
  systemMessage = buildSeedBanner(bankId, cold === true, gitNote);

  // ALWAYS record the session start (warm sessions used to log nothing — undebuggable).
  diag(harness, "session_start", { bank: bankId, cold, pages: pages.length, ms: Date.now() - t0 });

  return { systemMessage, additionalContext, deferInitialReflect };
}

/** Run one SessionStart hook invocation: stdin event in, (maybe) an additionalContext object on stdout. */
export async function runSessionStartHook(
  spec: SessionStartHookSpec,
  makeClient: (opts: ClientOpts) => SeedContextClient = (o) => new HindsightClient(o)
): Promise<void> {
  // Anti-recursion: the codebase survey's own headless claude session (core/survey.ts) sets this
  // so its hooks are a no-op — it must not re-seed/re-survey its own survey session.
  if (process.env.HINDSIGHT_DISABLE_HOOKS) return;

  // Whole-body try/catch (unlike runHook/runRetainHook, which only guard individual steps): a
  // throw here happens during session bootstrap, before the agent has done anything — more
  // disruptive than a failure mid-prompt — so nothing in this function may ever escape it.
  try {
    let ev: Record<string, unknown> = {};
    try {
      ev = JSON.parse(readFileSync(0, "utf8")) as Record<string, unknown>;
    } catch {
      return; // no/invalid event: stay silent
    }
    const { harness } = spec;
    const { cwd: rawCwd, sessionId } = spec.parse(ev);
    const cwd = rawCwd || process.cwd();

    let cfg = loadConfig({ harness });
    setLogLevel(cfg.logLevel);
    syncCompanionSkill(harness); // keep the installed skill current with the package version
    if (cfg.disabled) return;
    // …and keep the package itself current. AFTER the disabled check, unlike the skill sync above:
    // `disabled` means an inert plugin, and a network call plus a background npm install is not
    // inert. It also keeps the two harness families symmetric — the plugin hosts never construct a
    // RuntimeCore when disabled (harness/plugin-entry.ts), so they already skip this.
    // Detached and rate-limited to once a day; the update lands for the NEXT session.
    void maybeAutoUpdate(cfg);

    // Recorded HERE, on the session's first hook, so every later hook of this session resolves the
    // same bank however far the agent navigates (#3563).
    const sessionRoot = sessionRootDir(harness, sessionId, cwd);
    const derived = deriveBankIdOrSkip(cfg, cwd, harness, sessionRoot);
    if (derived === null) return; // repository unidentifiable: no bank, no seed, no injection
    const resolved = applyBankConfig(cfg, derived, cwd);
    cfg = resolved.cfg;
    const bankId = resolved.bankId;
    if (cfg.disabled) return; // per-bank opt-out (banks.<id> override)
    // Daemon mode: warm it up now, before the user has typed anything. The start itself is
    // detached; we wait only briefly, so an already-running daemon is adopted immediately while a
    // cold one keeps coming up in the background and is picked up by a later turn.
    await ensureDaemon(cfg, harness, { waitMs: DAEMON_WAIT_SESSION_START_MS });
    const client = makeClient({
      apiUrl: cfg.apiUrl,
      apiToken: cfg.apiToken,
      bank: bankId,
      maxParallelRetains: cfg.maxParallelRetains,
      observationScopes: cfg.observationScopes,
    });

    const out = await buildSessionStartContext({ cwd, sessionRoot, bankId, cfg, client, harness });
    if (out.deferInitialReflect && sessionId) {
      writeSessionCache(sessionCacheFile(harness, sessionId), { deferInitialReflect: true });
    }
    // The lifecycle computes one host-neutral output; the registry owns each host's wire schema.
    const payload = spec.emit(out);
    if (out.systemMessage || out.additionalContext) {
      process.stdout.write(JSON.stringify(payload));
    }
  } catch {
    /* SessionStart must never throw and break the session */
  }
}
