/**
 * Host adapter runtime for PERSISTENT-PLUGIN harnesses (opencode, Kilo, Cline). It delegates SessionStart and
 * prompt behavior to the same core lifecycle as fresh-process hook harnesses; this class keeps only
 * the host-specific injection, toast, and incremental-transcript responsibilities.
 *
 * A harness adapter feeds it three normalized events and reads two values back:
 *   - seedIfCold(repoPath)          : plugin load -> cold-check auto-seed + compute the page preamble
 *   - onPrompt(sessionId, prompt)   : each user turn -> recall + build this turn's injection
 *   - getInjection(sessionId)       : the system-prompt text to inject this turn (or undefined)
 *   - toolSpecs()                   : the hindsight_* knowledge/recall tools to register natively
 *   - onTranscript(sessionId, turns): full transcript -> write back every N turns (on by default)
 *   - onSessionIdle(sessionId)      : assistant finished -> refetch + write back the completed
 *                                     exchange (the Stop-equivalent these hosts lack)
 * No opencode/claude specifics live here — only the memory logic.
 */
import type { Config } from "./config";
import { DAEMON_WAIT_RETAIN_MS, DAEMON_WAIT_SESSION_START_MS, ensureDaemon } from "./daemon";
import { diag } from "./diag";
import { describeError, log, setLogLevel } from "./log";
import type { HindsightClient } from "./hindsight";
import { buildKnowledgeTools, type ToolSpec } from "./knowledge-tools";
import { buildPageTrigger } from "./missions";
import { retainLiveSession, type TransportTurn } from "./chat";
import { memoryCursorStore } from "./retain-cursor";
import { buildRetainStamp } from "./retain-stamp";
import { buildSessionStartContext } from "./session-start";
import { buildHookOutput } from "./hook";
import { sessionCacheFile, writeSessionCache } from "./session-cache";

const HARNESS = "opencode";

export class RuntimeCore {
  private readonly injection = new Map<string, string>(); // sessionId -> this turn's injection block
  private readonly turnCount = new Map<string, number>(); // sessionId -> user-turn counter (cadence)
  private readonly sessionState = new Map<string, { startTs: string; retainedTurns: number }>();
  /** Live write-back cursors. In memory, unlike the hook harnesses': this host outlives the session. */
  private readonly cursors = memoryCursorStore();
  /** Pulls a session's CURRENT transcript from the host (set by the adapter); see onSessionIdle. */
  private fetchTranscript?: (sessionId: string) => Promise<TransportTurn[]>;
  private lastInjection = ""; // most recent turn's injection block, keyed by nothing (see getInjection)
  private deferInitialReflect = false;
  /** Host-notice channel (opencode/Kilo: client.tui.showToast via the adapter). Optional, fail-open. */
  private notify?: (title: string, message: string) => void;
  private preamble = ""; // SessionStart-equivalent knowledge preamble, computed once at seedIfCold

  constructor(
    private readonly client: HindsightClient,
    private readonly bankId: string,
    private readonly cfg: Config,
    /**
     * Which persistent-plugin host loaded us. Scopes diagnostics, the session cache file and the
     * retained transcript's harness field, so Kilo sessions don't masquerade as opencode ones.
     * Defaults to opencode, the original (and only other) plugin harness.
     */
    readonly harness: string = HARNESS,
    /** Workspace root this host opened — the SAME directory bank resolution used, so a
     *  `{gitProject}` in retainTags names the repo the bank was derived from. */
    private readonly projectDir: string = process.cwd()
  ) {
    setLogLevel(cfg.logLevel);
  }

  setNotifier(notify: (title: string, message: string) => void): void {
    this.notify = notify;
  }

  /**
   * Supply a way to READ the session's transcript back from the host. Without it `onSessionIdle`
   * no-ops, so hosts that can't refetch keep the old (lagging) turn-driven write-back rather than
   * retaining a stale transcript.
   */
  setTranscriptSource(fetch: (sessionId: string) => Promise<TransportTurn[]>): void {
    this.fetchTranscript = fetch;
  }

  /** The hindsight_* knowledge + recall tools, bound to this bank, for the harness to register natively. */
  toolSpecs(): ToolSpec[] {
    return buildKnowledgeTools(this.client, this.bankId, {
      repoDir: this.projectDir,
      harness: this.harness,
      pageTrigger: buildPageTrigger(this.cfg),
      stampFor: () =>
        buildRetainStamp(this.cfg, {
          directory: this.projectDir,
          harness: this.harness,
          bankId: this.bankId,
        }),
    });
  }

  get writeBackEnabled(): boolean {
    return this.cfg.retainSessions;
  }

  /**
   * Plugin load (SessionStart-equivalent): on a cold repo, deterministically start the background
   * git-log seed + codebase survey, and compute the knowledge-page preamble (tool guide + roster)
   * that onPrompt injects on the session's first turn. Reuses the exact hook-harness logic
   * (`buildSessionStartContext`) so opencode seeds identically. Never throws.
   */
  async seedIfCold(repoPath: string | undefined): Promise<void> {
    // Anti-recursion: a headless survey session runs the agent (which loads this plugin) with
    // HINDSIGHT_DISABLE_HOOKS=1 — the tools stay registered (toolSpecs, so the survey can ingest),
    // but seeding/recall/write-back must no-op or the survey would re-seed itself (see core/survey.ts).
    if (process.env.HINDSIGHT_DISABLE_HOOKS) return;
    // Daemon mode: this is the SessionStart of a persistent-plugin host, so it owns the same
    // warm-up the hook harnesses do in `runSessionStartHook` — start it before the user has typed
    // anything, wait only briefly, and let a cold one keep coming up in the background. Without it
    // these hosts never started a daemon at all and every request failed with ECONNREFUSED (#3524);
    // `runSessionStartHook` is a hook-only wrapper, so calling `buildSessionStartContext` directly
    // (as every plugin harness does) skipped the ensure entirely.
    await ensureDaemon(this.cfg, this.harness, { waitMs: DAEMON_WAIT_SESSION_START_MS });
    try {
      const out = await buildSessionStartContext({
        cwd: repoPath || process.cwd(),
        bankId: this.bankId,
        cfg: this.cfg,
        client: this.client,
        harness: this.harness,
      });
      // The seed note is user-facing; opencode has no visible-system channel at load, so surface it
      // on stderr (shows in the plugin log / console) rather than dropping it.
      // Banner: via the host's toast API when available (stderr corrupts the TUI); always logged.
      if (out.systemMessage) {
        const plain = out.systemMessage.replace(/\x1b\[[0-9;]*m/g, "");
        log.info(this.harness, plain);
        this.notify?.("Hindsight", plain.replace(/^Hindsight is /, "Is "));
      }
      this.preamble = out.additionalContext ?? "";
      this.deferInitialReflect = out.deferInitialReflect === true;
    } catch {
      /* seeding + preamble are best-effort — a cold-check failure never breaks the agent */
    }
  }

  /**
   * Each user turn delegates to the same `buildHookOutput` used by hook harnesses. The cache file
   * intentionally replaces the old in-memory reflect/page state, keeping lifecycle behavior
   * identical across hosts while this adapter retains only delivery-specific state.
   */
  async onPrompt(sessionId: string | undefined, prompt: string): Promise<void> {
    if (process.env.HINDSIGHT_DISABLE_HOOKS) return; // anti-recursion (see seedIfCold)
    if (!sessionId || !prompt.trim()) return;
    const turns = (this.turnCount.get(sessionId) ?? 0) + 1;
    this.turnCount.set(sessionId, turns);

    const cacheFile = sessionCacheFile(this.harness, sessionId);
    if (this.deferInitialReflect) {
      // `seedIfCold` has no session id. Transfer its SessionStart decision to the first concrete
      // session here; `buildHookOutput` consumes it exactly once, like hook harnesses do.
      this.deferInitialReflect = false;
      writeSessionCache(cacheFile, { deferInitialReflect: true });
    }
    const output = await buildHookOutput({
      harness: this.harness,
      prompt,
      cfg: this.cfg,
      client: this.client,
      cacheFile,
    });

    const blocks: string[] = [];
    // The preamble is the SessionStart-equivalent; inject it once, on the first turn (later turns get
    // the periodic refresh below). Empty until seedIfCold resolves — if the first prompt races ahead
    // of plugin-load seeding, the roster refresh still delivers the tool guide on cadence.
    if (turns === 1 && this.preamble) blocks.push(this.preamble);
    if (output.context) blocks.push(output.context);
    // OpenCode has no user-message hook channel; use its native toast instead of stderr, which
    // renders inside the TUI input line. The shared output owns when a notice exists.
    if (output.notice) {
      const preview = output.notice.replace(/\s+/g, " ").trim().slice(0, 140);
      log.info(this.harness, "reflect goal", { preview });
      this.notify?.("Hindsight · recalled past decisions", preview);
    }
    const block = blocks.filter(Boolean).join("\n\n");
    this.injection.set(sessionId, block);
    this.lastInjection = block; // for consumers that can't supply a sessionId (see getInjection)
  }

  /**
   * The system-prompt text to inject this turn (built by the preceding onPrompt), or undefined.
   * opencode's `experimental.chat.system.transform` hook fires with NO sessionId (input is just
   * `{model}`), so a session-keyed lookup returns nothing there — fall back to the most recent
   * turn's block (`lastInjection`). The completion's system.transform fires right after that
   * session's `chat.message`/onPrompt, so `lastInjection` is this turn's block.
   */
  getInjection(sessionId: string | undefined): string | undefined {
    const keyed = sessionId ? this.injection.get(sessionId) : undefined;
    return keyed ?? this.lastInjection ?? undefined;
  }

  /**
   * Full normalized transcript (rich: user/assistant text + tool calls/outputs): upsert every turn
   * when enabled. On by default for persistent-plugin hosts (parity with hook-harness Stop
   * write-back), so opencode sessions compound into memory; opt out with `retainSessions: false`.
   *
   * There is deliberately no client-side cadence. Batching write-backs meant holding turns in a
   * process the host can close at any moment, with no reliable signal on the way out — the flush a
   * cadence needs would have to ride a session-end hook, and those are cancelled at shutdown. The
   * server coalesces instead: queued retains for one document fold into a single execution
   * (`engine.retain.fold`), so submitting every turn costs one extraction, not one per turn, and
   * nothing is ever held somewhere it can be lost.
   */
  async onTranscript(sessionId: string, turns: TransportTurn[]): Promise<void> {
    if (process.env.HINDSIGHT_DISABLE_HOOKS) return; // anti-recursion (see seedIfCold)
    if (!this.writeBackEnabled || !sessionId || !turns.length) return;
    const st = this.stateFor(sessionId);
    this.retain(sessionId, turns, st.startTs);
  }

  /**
   * Session went idle — the assistant has finished replying.
   *
   * This is the Stop-equivalent these hosts lack. `onTranscript` alone is driven by
   * `messages.transform`, which the host runs while BUILDING a request, so the transcript it hands
   * us never contains the reply that request produced: write-back always lagged one user turn, and
   * a session's final exchange was never retained at all. Refetching here (rather than replaying a
   * cached list) is the whole point — the cached one predates the reply.
   *
   * Cadence is deliberately bypassed: idle means "there may be nothing after this", so a pending
   * turn must not wait for a threshold it will never reach.
   */
  async onSessionIdle(sessionId: string): Promise<void> {
    if (process.env.HINDSIGHT_DISABLE_HOOKS) return; // anti-recursion (see seedIfCold)
    if (!this.writeBackEnabled || !sessionId || !this.fetchTranscript) return;
    let turns: TransportTurn[];
    try {
      turns = await this.fetchTranscript(sessionId);
    } catch (e) {
      diag(this.harness, "idle_fetch_failed", {
        error: describeError(e),
        session: sessionId,
      });
      return;
    }
    if (!turns.length) return;
    const st = this.stateFor(sessionId);
    // idle can fire more than once for one exchange (and again on a session with no new activity);
    // only retain when this transcript actually grew past what we last wrote.
    if (turns.length <= st.retainedTurns) return;
    st.retainedTurns = turns.length;
    this.retain(sessionId, turns, st.startTs, "idle");
  }

  private stateFor(sessionId: string): {
    startTs: string;
    retainedTurns: number;
  } {
    let st = this.sessionState.get(sessionId);
    if (!st) {
      st = { startTs: new Date().toISOString(), retainedTurns: 0 };
      this.sessionState.set(sessionId, st);
    }
    return st;
  }

  /** Fire-and-forget upsert of the session transcript; never throws into the host. */
  private retain(
    sessionId: string,
    turns: TransportTurn[],
    startTs: string,
    trigger = "turn"
  ): void {
    const t0 = Date.now();
    // Daemon mode: the write path is the last chance to get a daemon up — the Stop-hook role, for
    // hosts that have no Stop hook. A daemon that idled out mid-session (daemonIdleTimeout) would
    // otherwise take the whole exchange with it. The wait sits INSIDE the fire-and-forget chain, so
    // unlike the Stop hook it costs the host nothing: this call already returns before the retain
    // does. Deliberately not gated on the result — retain proceeds either way, so an unreachable
    // daemon produces the same `retain_failed` diagnostic as an unreachable Cloud server.
    void ensureDaemon(this.cfg, this.harness, { waitMs: DAEMON_WAIT_RETAIN_MS })
      .then(() =>
        retainLiveSession(this.client, sessionId, turns, startTs, this.harness, {
          cursors: this.cursors,
          stamp: buildRetainStamp(this.cfg, {
            directory: this.projectDir,
            harness: this.harness,
            bankId: this.bankId,
            sessionId,
          }),
        })
      )
      .then(() =>
        diag(this.harness, "retain_ok", {
          ms: Date.now() - t0,
          turns: turns.length,
          session: sessionId,
          trigger,
        })
      )
      .catch((e) =>
        diag(this.harness, "retain_failed", {
          ms: Date.now() - t0,
          error: describeError(e),
          session: sessionId,
          trigger,
        })
      );
  }
}
