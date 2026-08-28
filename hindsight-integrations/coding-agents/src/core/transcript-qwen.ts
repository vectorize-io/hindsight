/**
 * Qwen Code session transcript (JSONL) reader — a HYBRID of the two schemas this package already
 * parses, which is why it needs its own reader instead of reusing either.
 *
 * The ENVELOPE is Claude Code's, field for field: one record per line carrying
 * `uuid`/`parentUuid`/`sessionId`/`timestamp`/`type`/`cwd`/`version`/`gitBranch`, stored at
 * `~/.qwen/projects/<slugified-cwd>/chats/<sessionId>.jsonl` and handed to the Stop hook as
 * `transcript_path`. The BODY is Gemini's: `message.parts[]` rather than `message.content[]`, and
 * an assistant record's `message.role` reads **"model"** — so, exactly as in transcript.ts, role is
 * driven from the record's own `type` and never from the redundant `message.role`.
 *
 * What we keep, normalized to the SAME `TransportTurn[]` shape as readClaudeTranscript so the live
 * write-back (retainLiveSession) renders every harness identically:
 *   - real user prompts
 *   - assistant prose
 *   - each `functionCall` part → a compact `role:"action"` turn (name + primary target, no args)
 *
 * What we drop:
 *   - `type:"system"` records — UI telemetry, attribution/file-history snapshots, slash-command
 *     invocations. Not conversation, and the BULK of the file: 5,385 of 9,962 records (54%) across
 *     the live transcripts this reader was measured against.
 *   - `type:"tool_result"` records (`functionResponse` parts) — tool OUTPUT, the mechanical noise
 *     the shared `actionLine` convention exists to keep out of the bank. The matching CALL is a
 *     `functionCall` part on the assistant record, so nothing is lost by dropping these whole.
 *   - parts flagged `thought: true` — the model's reasoning, like Claude `thinking` and Codex
 *     `reasoning`.
 *   - SYNTHETIC user records. `type:"user"` is not a user turn in Qwen: background-task
 *     notifications, cron re-entries and goal-runtime traffic are all written as `type:"user"`, and
 *     they outnumbered genuine prompts 307 to 69 in the measured corpus. `provenance` is the
 *     discriminator (see `isRealUser`) — Qwen's analogue of Claude's `isMeta` and dsh's
 *     `source.kind === "user"`. Retaining them files machine scaffolding as the user's own words.
 *
 * Fail-open: never throws on a missing file, a malformed line, or a line that parses to a
 * non-object JSON value (`null`, a number, an array, …).
 */
import type { TransportTurn } from "./chat";
import { readJsonlTail } from "./jsonl";
import { actionLine, stripInjectedMemory } from "./transcript-util";
import { log } from "./log";

/**
 * Qwen ECHOES OUR OWN INJECTION BACK INTO ITS TRANSCRIPT, so removing it is load-bearing rather
 * than defensive: left in, every Stop would re-ingest the memories that turn's recall had just
 * injected — the retain→recall feedback loop `stripInjectedMemory` exists to prevent.
 *
 * That shared stripper cannot do it here. Qwen HTML-escapes `<`/`>` in `additionalContext` before
 * the model sees it, so what lands on disk is `&lt;hindsight_memories&gt;` and `MEMORY_TAG_RE` —
 * which matches raw tags — sees nothing. Measured: all 55 injections in the live corpus are stored
 * escaped, 0 raw, and all 55 sit inside the wrapper.
 *
 * But the wrapper must NOT be matched as a substring anywhere. Qwen's hook contract is explicit
 * that the tag is "a provenance marker, not ... a general trust boundary", that consumers "must
 * not infer that arbitrary tag-like user text is hook provenance", and that hook output is escaped
 * so it "cannot close or forge the tag". A regex replace over every text part therefore deletes
 * genuine user-authored prose that merely quotes the tag — which, in a repo whose subject IS this
 * integration, is a prompt someone will really write.
 *
 * Two pieces of evidence identify a real injection, per that contract:
 *   1. `systemPayload.hookContext` is a string. Then `systemPayload.displayText` is the pre-hook
 *      projection and "never includes the hook context", so it IS the user turn — use it directly
 *      and ignore the parts.
 *   2. Compatibility fallback for released `displayText`-only records: a COMPLETE tagged context
 *      occupying the FINAL part, after at least one other part. Then drop that whole part.
 * Anything else is left alone, including legacy bare-context records, which the contract says
 * "keep their model-bound display behavior because the context cannot be separated reliably".
 */
const HOOK_CONTEXT_PART_RE =
  /^\s*<qwen:user-prompt-submit-context>[\s\S]*<\/qwen:user-prompt-submit-context>\s*$/;

/** Whether this part is ENTIRELY a hook-context wrapper (not merely one that mentions the tag). */
function isWholeHookContextPart(part: Part): boolean {
  return typeof part.text === "string" && HOOK_CONTEXT_PART_RE.test(part.text);
}

/** One entry of a Gemini `Content.parts[]` — the four shapes Qwen actually writes. */
interface Part {
  text?: string;
  /** Model reasoning rides on a normal text part, flagged only by this. */
  thought?: boolean;
  functionCall?: { name?: string; args?: unknown };
  functionResponse?: unknown;
}

interface TranscriptLine {
  type?: string;
  subtype?: string;
  provenance?: string;
  /** Written only for user-prompt records that carried UserPromptSubmit hook context. */
  systemPayload?: { displayText?: unknown; hookContext?: unknown };
  /** Subagent envelope markers. Present ONLY in subagent transcripts, which carry no provenance. */
  agentId?: unknown;
  isSidechain?: unknown;
  timestamp?: string;
  message?: {
    role?: string;
    parts?: Part[];
  };
}

interface RenderedLine {
  role: string;
  content: string;
}

/**
 * Whether a `type:"user"` record is something the HUMAN sent.
 *
 * When `provenance` is present it is authoritative — `real_user` (and its one interjection subtype,
 * `mid_turn_user_message`) versus the `system`/`goal_runtime` scaffolding. When it is absent we
 * fall back exactly the way Qwen's own `legacySafeProvenance` does, because SUBAGENT transcripts
 * use a THIRD envelope — `{agentId, agentName, isSidechain, …}` with no `provenance` and no
 * `subtype` — and a strict `provenance === "real_user"` test would read them as 100% synthetic and
 * retain nothing at all.
 */
function isRealUser(line: TranscriptLine): boolean {
  // Present and well-formed: authoritative, and the ONLY accepting case on a main transcript.
  if (typeof line.provenance === "string") return line.provenance === "real_user";
  // Present but malformed (null, a number, an object). Do NOT fall through to the subtype
  // heuristic: an unrecognized provenance is unknown provenance, and unknown provenance on a
  // record that HAS the field is not evidence of a human. Measured on the live corpus, every
  // record in a main transcript carries a string provenance, so this can only be corruption.
  if (line.provenance !== undefined) return false;
  // Absent entirely: the subagent envelope, which is a different shape carrying neither
  // `provenance` nor `subtype`. Require a positive marker of that shape rather than inferring it
  // from the absence of a subtype — otherwise any field-stripped record reads as human.
  const isSubagentEnvelope = line.agentId !== undefined || line.isSidechain !== undefined;
  if (!isSubagentEnvelope) return false;
  return line.subtype === undefined || line.subtype === "mid_turn_user_message";
}

/**
 * Render one record's `parts` into turns: text parts join into one prose turn (Qwen's injected-
 * context echo and any injected memory stripped); each `functionCall` becomes its own compact
 * `role:"action"` turn (name + primary target, via `actionLine`); `thought` and `functionResponse`
 * parts are dropped.
 */
function renderLine(parts: Part[] | undefined, type: string): RenderedLine[] {
  if (!Array.isArray(parts)) return [];

  // Compatibility pairing evidence (see HOOK_CONTEXT_PART_RE): a complete tagged context in the
  // FINAL part, after at least one other part. Never a substring, never a lone part.
  const last = parts.length > 1 ? parts[parts.length - 1] : undefined;
  const dropLast =
    !!last && typeof last === "object" && isWholeHookContextPart(last) ? parts.length - 1 : -1;

  const texts: string[] = [];
  const actions: RenderedLine[] = [];
  for (const [i, part] of parts.entries()) {
    if (!part || typeof part !== "object") continue;
    if (i === dropLast) continue; // the injected-context part, positively identified
    if (part.thought === true) continue; // model reasoning, like Claude `thinking`
    if (typeof part.text === "string") {
      const text = stripInjectedMemory(part.text).trim();
      if (text) texts.push(text);
    } else if (part.functionCall && typeof part.functionCall.name === "string") {
      // `args` is a real object here (unlike Codex's raw JSON string), so no parse step.
      actions.push({
        role: "action",
        content: actionLine(part.functionCall.name, part.functionCall.args),
      });
    }
    // functionResponse: dropped — outputs are mechanical noise for extraction
  }

  const out: RenderedLine[] = [];
  const joined = texts.join("\n").trim();
  if (joined) out.push({ role: type, content: joined });
  out.push(...actions);
  return out;
}

/** Parse a Qwen Code chat JSONL into normalized turns (text + tool calls).
 *  Drops system records, tool results, thought parts, synthetic user records, Qwen's echo of our
 *  own injection, and empty turns. Never throws on bad lines. */
export function readQwenTranscript(path: string): TransportTurn[] {
  const turns: TransportTurn[] = [];
  try {
    collectQwenTurns(path, turns);
  } catch (err) {
    // The fail-open contract is only honoured if it covers LAZY I/O. readJsonlTail guards
    // statSync/openSync, but the generator it returns reads on each iteration, and those reads
    // throw here rather than at the call. A DIRECTORY passed as transcript_path is the real case:
    // on Linux it passes both statSync and openSync, then throws EISDIR on the first readSync —
    // after this function has already returned its generator. runRetainHook calls the reader
    // OUTSIDE buildRetain's catch, so an uncaught throw rejects the whole Stop hook and the turn
    // is never retained. Yield what we parsed before the fault instead.
    log.warn("qwen-code", "transcript read failed — retaining what was parsed", {
      path,
      turns: turns.length,
      error: err instanceof Error ? err.message : String(err),
    });
  }
  return turns;
}

/** The parse loop. Separated so a lazy read fault leaves the caller holding the partial result. */
function collectQwenTurns(path: string, turns: TransportTurn[]): void {
  for (const rawLine of readJsonlTail(path, { scope: "qwen-code" }).lines) {
    const trimmed = rawLine.trim();
    if (!trimmed) continue;

    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      continue;
    }
    // JSON.parse accepts non-object top-level values (`null`, numbers, booleans, arrays);
    // guard here so a corrupt/truncated line can't reach a property access below and throw.
    if (typeof parsed !== "object" || parsed === null) continue;
    const line = parsed as TranscriptLine;

    if (line.type !== "user" && line.type !== "assistant") continue;
    if (line.type === "user" && !isRealUser(line)) continue;
    if (typeof line.message !== "object" || line.message === null) continue;

    // Primary pairing evidence: when `systemPayload.hookContext` is a string, `displayText` is
    // the pre-hook projection and "never includes the hook context" (Qwen's hook contract), so it
    // IS the user turn. Prefer it over reconstructing from parts — no tag matching involved.
    // Tool calls cannot occur on such a record (it is a user prompt), so parts are not consulted.
    const sp = line.systemPayload;
    if (
      line.type === "user" &&
      sp &&
      typeof sp === "object" &&
      typeof sp.hookContext === "string" &&
      typeof sp.displayText === "string"
    ) {
      const content = stripInjectedMemory(sp.displayText).trim();
      if (content) {
        const turn: TransportTurn = { role: "user", content };
        if (typeof line.timestamp === "string") turn.timestamp = line.timestamp;
        turns.push(turn);
      }
      continue;
    }

    // `type` is validated as "user" | "assistant" above; drive role from it (Qwen's assistant
    // records carry the Gemini `message.role: "model"`, so the nested role is worse than redundant
    // here). One record can yield a prose turn plus one action turn per tool call.
    for (const rendered of renderLine(line.message.parts, line.type)) {
      const turn: TransportTurn = { role: rendered.role, content: rendered.content };
      if (typeof line.timestamp === "string") turn.timestamp = line.timestamp;
      turns.push(turn);
    }
  }
}
