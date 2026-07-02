/**
 * hindsight-coding-opencode — long-term memory for coding agents, reflect + INJECT.
 *
 * READ: when a task arrives, the plugin asks the project's memory (Hindsight `reflect`) about the
 * task's symptom and PUSHES the synthesized root-cause answer into the system prompt.
 * WRITE (opt-in): with HINDSIGHT_RETAIN_SESSIONS on, it also binds the live session INTO memory —
 * every few turns it upserts the full user/assistant transcript (tool calls/comments dropped) under a
 * stable per-session document_id, so future sessions can recall it. Off by default (benchmark: a
 * pre-backfilled bank must not be polluted by the agent writing its own solves back).
 *
 * Env: HINDSIGHT_API_URL (default http://localhost:8888), HINDSIGHT_BANK_ID, HINDSIGHT_API_TOKEN,
 *      HINDSIGHT_DISABLED (hard off-switch), HINDSIGHT_RETAIN_SESSIONS (enable live write),
 *      HINDSIGHT_RETAIN_EVERY_TURNS (default 5).
 */
import type { Plugin } from "@opencode-ai/plugin";

const env = (k: string, d = "") => process.env[k] ?? d;

const HindsightCodingPlugin: Plugin = async () => {
  if (env("HINDSIGHT_DISABLED")) return {}; // inert: same agent, no memory (baseline parity)

  const apiUrl = env("HINDSIGHT_API_URL", "http://localhost:8888").replace(/\/$/, "");
  const apiToken = env("HINDSIGHT_API_TOKEN") || undefined;
  const bankId = env("HINDSIGHT_BANK_ID", "coding");

  // Reflect ONCE per session (on the task message); the surfaced memory is injected every turn.
  const memory = new Map<string, string>();
  const reflected = new Set<string>();

  const TIMEOUT_MS = Number(env("HINDSIGHT_REFLECT_TIMEOUT_MS")) || 120000;

  async function reflect(query: string): Promise<string> {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (apiToken) headers["Authorization"] = `Bearer ${apiToken}`;
    // Bounded: never hang the agent if the memory server is slow/loaded — skip injection instead.
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
    try {
      const resp = await fetch(`${apiUrl}/v1/default/banks/${encodeURIComponent(bankId)}/reflect`, {
        method: "POST",
        headers,
        body: JSON.stringify({ query, budget: "high" }),
        signal: ctrl.signal,
      });
      if (!resp.ok) throw new Error(`reflect ${resp.status}`);
      const data = (await resp.json()) as { text?: string };
      return (data.text || "").trim();
    } finally {
      clearTimeout(timer);
    }
  }

  const textOf = (parts: { type?: string; text?: string }[]) =>
    (parts || []).filter((p) => p?.type === "text" && p.text).map((p) => p!.text).join("\n").trim();

  // ── live session write-back (opt-in) ────────────────────────────────────────
  const RETAIN_SESSIONS = ["1", "true"].includes(env("HINDSIGHT_RETAIN_SESSIONS").toLowerCase());
  const RETAIN_EVERY = Number(env("HINDSIGHT_RETAIN_EVERY_TURNS")) || 5;
  const sessionState = new Map<string, { startTs: string; retainedUsers: number }>();

  async function retainSession(sid: string, turns: { role: string; content: string; timestamp?: string }[], startTs: string) {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (apiToken) headers["Authorization"] = `Bearer ${apiToken}`;
    // stable document_id per session => same-id UPSERT: Hindsight reprocesses the FULL conversation,
    // so the distributed decision is extracted from the whole thing (not scattered per-turn).
    await fetch(`${apiUrl}/v1/default/banks/${encodeURIComponent(bankId)}/memories`, {
      method: "POST", headers,
      body: JSON.stringify({ items: [{
        content: JSON.stringify(turns),                 // JSON user/assistant transcript (roles preserved)
        context: "opencode agent session",
        document_id: `conversation:${sid}`,             // stable per session (upsert), timestamp carried below
        tags: ["source:chat"],
        strategy: "chat",
        timestamp: startTs,
        metadata: { source: "chat", session_id: sid, ref_id: `conversation:${sid}` },
      }], async: true }),
    }).catch(() => {}); // best-effort; never break the agent
  }

  return {
    // On the task message, reflect on its symptom and cache the surfaced decision for this session.
    "chat.message": async (input: { sessionID?: string }, output: { parts: { type?: string; text?: string }[] }) => {
      const sid = input.sessionID;
      if (!sid || reflected.has(sid)) return; // once per session
      const q = textOf(output.parts);
      if (!q) return;
      reflected.add(sid);
      try {
        const ans = await reflect(q);
        if (ans) memory.set(sid, ans);
      } catch {
        /* memory is best-effort — never break the agent */
      }
    },
    // Push the surfaced decision into the system prompt (every turn, so it persists across interventions).
    "experimental.chat.system.transform": async (input: { sessionID?: string }, output: { system: string[] }) => {
      const mem = input.sessionID ? memory.get(input.sessionID) : undefined;
      if (!mem) return;
      output.system.push(
        "Relevant project memory, surfaced from THIS repository's git history and past developer " +
        "conversations — a past decision that likely explains this issue. If it states an EXACT rule " +
        "or literal values (specific strings, numbers, set members, mappings), apply them PRECISELY as " +
        "given — the hidden tests depend on those exact choices; do not substitute your own guess. " +
        "Verify against the current code before editing:\n\n" + mem,
      );
    },
    // WRITE-BACK (opt-in): bind the live session into memory. Every RETAIN_EVERY user turns, upsert the
    // FULL filtered transcript (user/assistant TEXT only — tool calls/outputs & reasoning dropped) under
    // a stable per-session document_id. Note: our injected memory lives in the SYSTEM prompt, not in
    // messages, so it is never re-ingested (no feedback loop).
    "experimental.chat.messages.transform": async (
      _input: unknown,
      output: { messages: { info?: { role?: string; sessionID?: string; time?: { created?: number } }; parts: { type?: string; text?: string }[] }[] },
    ) => {
      if (!RETAIN_SESSIONS) return;
      const msgs = output.messages || [];
      const sid = msgs.find((m) => m.info?.sessionID)?.info?.sessionID;
      if (!sid) return;
      const turns: { role: string; content: string; timestamp?: string }[] = [];
      for (const m of msgs) {
        const role = m.info?.role;
        if (role !== "user" && role !== "assistant") continue; // drop non-conversational message roles
        const text = textOf(m.parts);                          // text parts only => drops tool calls/comments
        if (!text) continue;
        const created = m.info?.time?.created;                 // per-turn timestamp (Unix ms) -> ISO
        turns.push({ role, content: text, ...(created ? { timestamp: new Date(created).toISOString() } : {}) });
      }
      const users = turns.filter((t) => t.role === "user").length;
      let st = sessionState.get(sid);
      if (!st) { st = { startTs: new Date().toISOString(), retainedUsers: 0 }; sessionState.set(sid, st); }
      if (turns.length && users - st.retainedUsers >= RETAIN_EVERY) {
        st.retainedUsers = users;
        void retainSession(sid, turns, st.startTs);
      }
    },
  };
};

export default HindsightCodingPlugin;
export { HindsightCodingPlugin };
