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
import { buildSystemInjection } from "./inject";
import { retainLiveSession, type TransportTurn } from "./chat";

export interface RuntimeOpts {
  retainSessions?: boolean;   // enable live write-back (off by default: don't pollute a backfilled bank)
  retainEveryTurns?: number;  // upsert every N user turns (default 5)
  reflectTimeoutMs?: number;
}

export class RuntimeCore {
  private readonly memory = new Map<string, string>();   // sessionId -> surfaced decision
  private readonly reflected = new Set<string>();        // sessions we've already reflected for
  private readonly sessionState = new Map<string, { startTs: string; retainedUsers: number }>();
  private readonly retainSessions: boolean;
  private readonly retainEvery: number;
  private readonly reflectTimeoutMs: number;

  constructor(private readonly client: HindsightClient, opts: RuntimeOpts = {}) {
    this.retainSessions = !!opts.retainSessions;
    this.retainEvery = opts.retainEveryTurns || 5;
    this.reflectTimeoutMs = opts.reflectTimeoutMs || 120000;
  }

  get writeBackEnabled(): boolean { return this.retainSessions; }

  /** First task message of a session: reflect on the symptom once, cache the root-cause answer. */
  async onTask(sessionId: string, query: string): Promise<void> {
    if (!sessionId || this.reflected.has(sessionId) || !query.trim()) return;
    this.reflected.add(sessionId);
    try {
      const ans = await this.client.reflect(query, { budget: "high", timeoutMs: this.reflectTimeoutMs });
      if (ans) this.memory.set(sessionId, ans);
    } catch { /* memory is best-effort — never break the agent */ }
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
    if (!st) { st = { startTs: new Date().toISOString(), retainedUsers: 0 }; this.sessionState.set(sessionId, st); }
    if (users - st.retainedUsers >= this.retainEvery) {
      st.retainedUsers = users;
      void retainLiveSession(this.client, sessionId, turns, st.startTs).catch(() => {});
    }
  }
}
