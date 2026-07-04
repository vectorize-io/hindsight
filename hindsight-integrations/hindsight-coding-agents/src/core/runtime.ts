/**
 * Harness-agnostic runtime: the reflect-once → inject → (opt-in) write-back state machine.
 *
 * A harness adapter feeds it three normalized events and reads one value back:
 *   - onTask(sessionId, query)      : first task message -> reflect once, cache the answer
 *   - getInjection(sessionId)       : the system-prompt text to inject this turn (or undefined)
 *   - onTranscript(sessionId, turns): full user/assistant transcript -> maybe write back
 * No opencode/claude-code specifics live here — only the memory logic.
 */
import type { HindsightClient } from "./hindsight";
import { appendFileSync } from "node:fs";
import { buildSystemInjection } from "./inject";
import { retainLiveSession, type TransportTurn } from "./chat";
import { syncGit } from "./sync";

export interface RuntimeOpts {
  retainSessions?: boolean; // enable live write-back (off by default: don't pollute a backfilled bank)
  retainEveryTurns?: number; // upsert every N user turns (default 5)
  reflectTimeoutMs?: number;
  gitSync?: boolean; // keep the bank current with new commits on load (default off: opt-in)
  gitSyncRef?: string; // target ref for sync (default origin/main; falls back to HEAD)
  gitSyncFetch?: boolean; // git fetch the ref before diffing (default off: no network side effect)
}

export class RuntimeCore {
  private readonly memory = new Map<string, string>(); // sessionId -> surfaced decision
  private readonly reflected = new Set<string>(); // sessions we've already reflected for
  private readonly sessionState = new Map<string, { startTs: string; retainedUsers: number }>();
  private readonly retainSessions: boolean;
  private readonly retainEvery: number;
  private readonly reflectTimeoutMs: number;
  private readonly gitSync: boolean;
  private readonly gitSyncRef: string;
  private readonly gitSyncFetch: boolean;
  private gitSyncStarted = false; // once-per-process guard for syncGitOnce

  constructor(
    private readonly client: HindsightClient,
    opts: RuntimeOpts = {}
  ) {
    this.retainSessions = !!opts.retainSessions;
    this.retainEvery = opts.retainEveryTurns || 5;
    this.reflectTimeoutMs = opts.reflectTimeoutMs || 120000;
    this.gitSync = !!opts.gitSync; // default off: opt-in
    this.gitSyncRef = opts.gitSyncRef || "origin/main";
    this.gitSyncFetch = !!opts.gitSyncFetch;
  }

  get writeBackEnabled(): boolean {
    return this.retainSessions;
  }

  /**
   * On-demand reflect: the SAME synthesized, root-cause answer the plugin injects automatically, but for
   * an explicit query the agent asks at any point. Best-effort: returns "" on error/timeout so a caller
   * (e.g. a harness tool) can degrade gracefully rather than surface a failure to the model.
   */
  async reflectNow(query: string): Promise<string> {
    if (!query.trim()) return "";
    try {
      return await this.client.reflect(query, { budget: "high", timeoutMs: this.reflectTimeoutMs });
    } catch {
      return "";
    }
  }

  /**
   * Once per process: async-retain any commits on the target ref not yet in the bank, keeping memory
   * current with the repo since the backfill (or the last run). Fire-and-forget and best-effort — a sync
   * failure never blocks or breaks the agent. `repoPath` comes from the harness's plugin context.
   */
  async syncGitOnce(repoPath: string | undefined): Promise<void> {
    if (!this.gitSync || !repoPath || this.gitSyncStarted) return;
    this.gitSyncStarted = true;
    try {
      const r = await syncGit(this.client, repoPath, {
        ref: this.gitSyncRef,
        fetch: this.gitSyncFetch,
      });
      if (r.ingested)
        console.error(`hindsight: git-sync retained ${r.ingested} new commit(s) from ${r.ref}`);
    } catch {
      /* best-effort — memory sync never breaks the agent */
    }
  }

  /** First task message of a session: reflect on the symptom once, cache the root-cause answer. */
  async onTask(sessionId: string, query: string): Promise<void> {
    if (!sessionId || this.reflected.has(sessionId) || !query.trim()) return;
    this.reflected.add(sessionId);
    const t0 = Date.now();
    try {
      const ans = await this.client.reflect(query, {
        budget: "high",
        timeoutMs: this.reflectTimeoutMs,
      });
      if (ans) this.memory.set(sessionId, ans);
      this.diag(ans ? "reflect_ok" : "reflect_empty", { ms: Date.now() - t0, chars: ans.length, query: query.slice(0, 80) });
    } catch (e) {
      /* memory is best-effort — never break the agent; but never fail silently either: a memory
         run whose reflect quietly failed is indistinguishable from a no-memory run without this. */
      this.diag("reflect_failed", { ms: Date.now() - t0, error: String((e as Error)?.message || e).slice(0, 200), query: query.slice(0, 80) });
    }
  }

  /** Append a reflect-outcome record to the diagnostics file (default /tmp/hindsight-plugin.log). */
  private diag(event: string, extra: Record<string, unknown> = {}): void {
    try {
      appendFileSync(process.env.HINDSIGHT_DIAG_FILE || "/tmp/hindsight-plugin.log",
        JSON.stringify({ ts: new Date().toISOString(), event, ...extra }) + "\n");
    } catch {
      /* diagnostics must not break the agent either */
    }
  }

  /** The system-prompt text to inject this turn (persisted across interventions), or undefined. */
  getInjection(sessionId: string | undefined): string | undefined {
    const mem = sessionId ? this.memory.get(sessionId) : undefined;
    return mem ? buildSystemInjection(mem) : undefined;
  }

  /** Full normalized transcript (user/assistant turns): upsert every N user turns when enabled. */
  async onTranscript(sessionId: string, turns: TransportTurn[]): Promise<void> {
    if (!this.retainSessions || !sessionId || !turns.length) return;
    const users = turns.filter((t) => t.role === "user").length;
    let st = this.sessionState.get(sessionId);
    if (!st) {
      st = { startTs: new Date().toISOString(), retainedUsers: 0 };
      this.sessionState.set(sessionId, st);
    }
    if (users - st.retainedUsers >= this.retainEvery) {
      st.retainedUsers = users;
      void retainLiveSession(this.client, sessionId, turns, st.startTs).catch(() => {});
    }
  }
}
