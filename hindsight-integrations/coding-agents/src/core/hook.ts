/**
 * Shared runtime for HOOK-based harnesses (Claude Code, Codex, Cursor CLI, ...).
 *
 * The ONE runtime path (see docs/superpowers/specs/2026-07-27-reflect-pages-runtime.md):
 *   - REFLECT once per session, on the first prompt: agentic synthesis over the bank returning the
 *     root-cause decision with exact values. Cached per session, re-injected every turn.
 *   - KNOWLEDGE PAGES every turn: the page set is fetched on a cadence and matched LOCALLY against
 *     the prompt (section-level lexical scoring — no server/LLM call); the top sections are
 *     injected with provenance. Fast like recall, organized like reflect.
 *   - The tool-guide/page-roster block re-injects on the same cadence.
 *
 * Every outcome is recorded in the diagnostics file — a memory-less session can't masquerade as a
 * memory session. A failure never breaks the agent.
 *
 * A harness plugs in with a HookSpec: its name, how to read (prompt, cwd, sessionId) from its
 * stdin event, and how to wrap injected context in its native output schema. The pure logic lives
 * in `buildHookOutput` (client + cache file in, injection string out) so it's unit-testable
 * without stdin/stdout; `runHook` is thin plumbing around it, with a `makeClient` seam for tests.
 */
import { existsSync, readFileSync } from "node:fs";
import { deriveBankId } from "./bank";
import type { Config } from "./config";
import { applyBankConfig, loadConfig } from "./config";
import { diag, diagFilePath } from "./diag";
import { describeError, log, setLogLevel } from "./log";
import { startBackgroundSeed } from "./seed";
import type { ClientOpts } from "./hindsight";
import { HindsightClient } from "./hindsight";
import { brandWord } from "./brand";
import { buildReflectQuery, buildSystemInjection } from "./inject";
import type { PageRef } from "./knowledge-injection";
import { buildRosterRefresh, parsePageList } from "./knowledge-injection";
import {
  readSessionCache,
  sessionCacheFile,
  sessionRootDir,
  writeSessionCache,
  type SessionCache,
} from "./session-cache";

export interface HookEventFields {
  prompt?: string;
  cwd?: string;
  sessionId?: string;
}

export interface HookSpec {
  /** Harness name — config `harnesses.<name>` section, {harness} template field, diag records. */
  harness: string;
  /** Read the fields out of the harness's stdin event (shapes differ per harness). */
  parse(event: Record<string, unknown>): HookEventFields;
  /** Some hosts execute hook commands from their global config directory. Those hosts must provide
   * a workspace path in the event; falling back to process.cwd() would create a bank for config. */
  requireCwd?: boolean;
  /** Wrap injected context (and an optional user-facing notice) in the harness's native
   *  hook-output schema. Harnesses whose schema has no user-visible channel ignore `notice`. */
  emit(context: string, notice?: string, event?: Record<string, unknown>): unknown;
}

/** Minimal client shape `buildHookOutput` needs — `HindsightClient` satisfies it structurally. */
interface HookClient {
  reflect(query: string, opts: { budget?: string; timeoutMs?: number }): Promise<string>;
  listPages(): Promise<unknown>;
  knowledgePagesSupported?: boolean;
}

/**
 * Cap on the once-per-session reflect. INVARIANT: this MUST stay below every harness's
 * UserPromptSubmit/PreInvocation hook timeout (currently 30s in the supported hook harnesses) —
 * otherwise the host kills the hook mid-reflect before the
 * result is cached, so the injection is discarded AND the reflect re-fires (uncached) every turn.
 * Raise the harness timeout in lockstep if you raise this.
 */
const HOOK_REFLECT_CAP_MS = 25_000;

export interface HookOutput {
  /** The model-facing injection block, or undefined when there's nothing to inject. */
  context?: string;
  /** User-facing line(s) — set only on the reflect turn (its goal + result preview). */
  notice?: string;
  /** Page count from the session's roster cache — 0 signals a bank the engine never built. */
  pagesKnown?: number;
}

/**
 * Pure hook logic: reflect once per session (cached); knowledge-page sections + roster on every
 * turn. Returns the injection plus a per-turn user-facing notice.
 */
export async function buildHookOutput(args: {
  harness: string;
  prompt: string;
  cfg: Config;
  client: HookClient;
  cacheFile: string;
}): Promise<HookOutput> {
  const { harness, prompt, cfg, client, cacheFile } = args;

  const cached = readSessionCache(cacheFile);
  const turns = (cached.turns ?? 0) + 1;

  // ── reflect: once per session, on the first prompt ────────────────────────────
  // autoReflect false = tool-only mode: no injected synthesis; the roster's tool guide instead
  // instructs the agent to call hindsight_reflect itself when a new goal is set.
  let reflectAnswer = cached.reflectAnswer;
  let reflectRanThisTurn = false;
  // Set ONLY by the catch below. An empty answer is not a failure: reflect can legitimately have
  // nothing to say on a sparse bank (diag records that as reflect_empty), and reporting it as a
  // failure would tell the user the plugin broke on exactly the sessions where it did not.
  let reflectFailed = false;
  const deferInitialReflect = cached.deferInitialReflect === true;
  if (deferInitialReflect) {
    // A new bank has no useful history yet. Do not burn the once-per-session synthesis on prompt
    // one; this marker is deliberately consumed below so prompt two remains eligible to reflect.
    diag(harness, "reflect_deferred_new_bank", { query: prompt.slice(0, 80) });
  } else if (cfg.autoReflect && reflectAnswer === undefined) {
    reflectRanThisTurn = true;
    const t0 = Date.now();
    try {
      reflectAnswer = await client.reflect(buildReflectQuery(prompt), {
        // Automatic reflection runs inside a hard 25s hook window. Hindsight's low budget is the
        // supported default for bounded reflect calls; callers that explicitly invoke the MCP
        // tool still get the deeper high-budget path.
        budget: "low",
        timeoutMs: Math.min(cfg.reflectTimeoutMs, HOOK_REFLECT_CAP_MS),
      });
      diag(harness, reflectAnswer ? "reflect_ok" : "reflect_empty", {
        ms: Date.now() - t0,
        chars: reflectAnswer.length,
        query: prompt.slice(0, 80),
        // Verbatim injected synthesis — the ONLY durable record of what the agent actually saw
        // (post-mortems otherwise depend on the harness happening to persist hook context).
        answer: reflectAnswer.slice(0, 8000),
      });
    } catch (e) {
      reflectAnswer = ""; // ran and failed — don't retry every turn; the diag trail records it
      reflectFailed = true;
      log.warn(harness, "reflect failed — session runs without memory", {
        error: describeError(e),
      });
      diag(harness, "reflect_failed", {
        ms: Date.now() - t0,
        error: describeError(e),
        query: prompt.slice(0, 80),
      });
    }
  }

  // ── knowledge-page roster (ids + titles only): refreshed on the cadence ────────
  const cadence = cfg.pageRefreshEveryTurns;
  const stale = !cached.pages || (cadence > 0 && turns - cached.pages.atTurn >= cadence);
  let pages = cached.pages?.list ?? [];
  if (stale) {
    const t0 = Date.now();
    try {
      pages = parsePageList(await client.listPages());
      diag(harness, "pages_ok", { ms: Date.now() - t0, count: pages.length });
    } catch (e) {
      diag(
        harness,
        client.knowledgePagesSupported === false ? "knowledge_pages_unavailable" : "pages_failed",
        {
          ms: Date.now() - t0,
          error: describeError(e),
        }
      );
    }
  }

  writeSessionCache(cacheFile, {
    turns,
    reflectAnswer,
    pages: { atTurn: stale ? turns : (cached.pages?.atTurn ?? turns), list: pages },
  } satisfies SessionCache);

  const blocks: string[] = [];
  // Hook-injected context lands in the USER MESSAGE and persists in the transcript — unlike a
  // system prompt, it accumulates. The reflect block is injected exactly ONCE, the turn reflect
  // ran. No cadence re-injection: replaying the turn-1 synthesis at arbitrary later turns reads
  // as random noise once the session drifts, and the agent holds hindsight_reflect for a FRESH
  // pass if compaction or drift makes it need memory again.
  if (reflectAnswer && reflectRanThisTurn) blocks.push(buildSystemInjection(reflectAnswer));
  // Knowledge pages are NOT auto-injected: the agent pulls them through
  // hindsight_search_knowledge_pages when a question warrants it — an unprompted injection on
  // every turn (even a plain "yes") read as phantom research. The roster below keeps the tool
  // and the page names in front of the agent.
  if (cadence > 0 && turns % cadence === 0) {
    blocks.push(buildRosterRefresh(pages, { reflectOnNewGoals: !cfg.autoReflect }));
  }
  const kept = blocks.filter(Boolean);

  // User-facing notice ONLY on the turn reflect actually ran (showing its assigned goal and a
  // preview of what came back). Ordinary turns stay silent — page knowledge is now pulled via
  // the hindsight_search_knowledge_pages tool, which is visible as a real tool call.
  let notice: string | undefined;
  if (reflectRanThisTurn && reflectAnswer) {
    const q = prompt.replace(/\s+/g, " ").trim();
    const excerpt = q.length > 48 ? `${q.slice(0, 48)}…` : q;
    const preview = reflectAnswer.replace(/\s+/g, " ").trim();
    notice =
      `${brandWord()} · goal: recall this repo's past decisions about “${excerpt}”\n` +
      `↳ ${preview.length > 140 ? `${preview.slice(0, 140)}…` : preview}`;
  } else if (reflectFailed) {
    // The failure is already in the diag trail and plugin.log, but both are files nobody is
    // tailing mid-session, so a memory-less session looked exactly like a healthy one (#3443).
    // One terse line pointing at the trail — not an explanation, and not advice to the agent:
    // this fires at most once per session, on the turn reflect ran.
    notice = `${brandWord()} · no memory this turn — see ${diagFilePath()}`;
  }

  return { context: kept.length ? kept.join("\n\n") : undefined, notice, pagesKnown: pages.length };
}

/** Run one hook invocation: stdin event in, (maybe) an injection object on stdout. */
export async function runHook(
  spec: HookSpec,
  makeClient: (opts: ClientOpts) => HookClient = (o) => new HindsightClient(o)
): Promise<void> {
  // Anti-recursion: the codebase survey's own headless session sets this so its hooks are a no-op.
  if (process.env.HINDSIGHT_DISABLE_HOOKS) return;

  let ev: Record<string, unknown> = {};
  try {
    ev = JSON.parse(readFileSync(0, "utf8")) as Record<string, unknown>;
  } catch {
    return; // no/invalid event: stay silent
  }
  const { prompt: rawPrompt, cwd: rawCwd, sessionId } = spec.parse(ev);
  if (spec.requireCwd && !rawCwd) return;
  const cwd = rawCwd || process.cwd();
  const prompt = (rawPrompt || "").trim();
  if (!prompt) return;

  let cfg = loadConfig({ harness: spec.harness });
  setLogLevel(cfg.logLevel);
  if (cfg.disabled) {
    log.debug(spec.harness, "hook skipped: disabled");
    return;
  }

  const out = (context: string | undefined, notice?: string) =>
    process.stdout.write(JSON.stringify(spec.emit(context ?? "", notice, ev)));

  const sessionRoot = sessionRootDir(spec.harness, sessionId, cwd);
  const resolved = applyBankConfig(cfg, deriveBankId(cfg, cwd, spec.harness, sessionRoot), cwd);
  cfg = resolved.cfg;
  const bankId = resolved.bankId;
  if (cfg.disabled) {
    log.debug(spec.harness, "hook skipped: bank disabled via banks override", { bank: bankId });
    return;
  }
  const client = makeClient({
    apiUrl: cfg.apiUrl,
    apiToken: cfg.apiToken,
    bank: bankId,
    maxParallelRetains: cfg.maxParallelRetains,
    observationScopes: cfg.observationScopes,
  });
  const cacheFile = sessionCacheFile(spec.harness, sessionId || "no-session");

  // Safety net on EVERY harness: on the session's FIRST prompt (no cache file yet), fire the
  // ingestion engine. Normally the shared SessionStart lifecycle already did — the engine's
  // per-bank lock makes this a no-op — but a session that predates installation never received
  // SessionStart. This limited fallback heals that case without duplicating any survey logic.
  if (cfg.autoSeed !== false && !existsSync(cacheFile)) {
    startBackgroundSeed(cwd, { limit: cfg.seedLimit, harness: spec.harness });
  }

  const output = await buildHookOutput({ harness: spec.harness, prompt, cfg, client, cacheFile });
  // Mid-session heal: a bank with ZERO pages means the engine never built it (e.g. the session
  // predates the install, so no SessionStart and the first-prompt net already passed). Fire the
  // idempotent engine — the per-bank lock makes repeats free while it builds.
  if (
    cfg.autoSeed !== false &&
    client.knowledgePagesSupported !== false &&
    output.pagesKnown === 0
  ) {
    startBackgroundSeed(cwd, { limit: cfg.seedLimit, harness: spec.harness });
  }
  if (output.context || output.notice) out(output.context, output.notice);
}
