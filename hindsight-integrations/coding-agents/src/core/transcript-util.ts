/** Shared transcript-rendering helpers used by both the Claude (transcript.ts) and Codex
 *  (transcript-codex.ts) session readers. */

/** Cap on any single rendered tool input or tool result (mirrors v1's 2000-char cap): small
 *  edits/commands are captured verbatim while a giant Write/output is bounded. */
export const TOOL_TEXT_CAP = 2000;

/** Injected context — stripped from retained text so a write-back never re-ingests its own
 *  injected memory (a retain→reflect feedback loop). Covers every block the hooks inject, PLUS
 *  the host's own hook-transport wrappers, which arrive as ordinary USER messages and would
 *  otherwise be extracted as things the user said (#3023):
 *    <hook_prompt>        codex surfaces hook stdout/errors this way
 *    <task-notification>  Claude Code reports a background task's outcome — task id, tool-use id,
 *                         status; measured at 39 of these across 400 local transcripts, each one
 *                         the entire message
 *    <system-reminder>    same class; today it only rides inside `tool_result` blocks, which the
 *                         Claude reader already drops, so this is insurance against it moving
 *    <hindsight_knowledge>, <hindsight_knowledge_refresh>  the pre-rename names of the page
 *                         preamble tags, still in transcripts recorded before it. Remove in 2.x.
 *  Tag-structural, not content-guessing. The block is removed and any surrounding text KEPT — a
 *  message that is nothing but a wrapper then renders empty and is dropped as a no-content turn. */
const MEMORY_TAG_RE =
  /<(hook_prompt|task-notification|system-reminder|hindsight_memory|hindsight_memories|hindsight_bank|relevant_memories|user_feedback|hindsight_pages|hindsight_page_refresh|hindsight_knowledge|hindsight_knowledge_refresh)\b[\s\S]*?<\/\1>/g;

export function stripInjectedMemory(s: string): string {
  return s.replace(MEMORY_TAG_RE, "");
}

export function truncate(s: string, max = TOOL_TEXT_CAP): string {
  return s.length > max ? `${s.slice(0, max)}… (truncated)` : s;
}

/** Keys tried, in order, to find a tool call's primary target for the compact action line. */
const TARGET_KEYS = [
  "file_path",
  "path",
  "notebook_path",
  "command",
  "pattern",
  "query",
  "url",
  "name",
  "id",
] as const;

const ACTION_TARGET_CAP = 100;

/**
 * Compact one tool call into an action line: the tool name plus its primary target — a file path,
 * command, pattern, … — with NO full arguments and NO output (e.g. `Edit boltons/strutils.py`).
 * Retaining raw args/outputs buries the session's decisions in mechanical noise; the extractor only
 * needs WHAT was touched. The one exception is a memory note, whose text is kept (memoryNoteText).
 */
export function actionLine(tool: string, input: unknown): string {
  let target = "";
  if (input && typeof input === "object") {
    const rec = input as Record<string, unknown>;
    for (const k of TARGET_KEYS) {
      const v = rec[k];
      if (typeof v === "string" && v.trim()) {
        target = v.trim().split("\n")[0];
        break;
      }
    }
  } else if (typeof input === "string") {
    target = input.trim().split("\n")[0];
  }
  if (target.length > ACTION_TARGET_CAP) target = `${target.slice(0, ACTION_TARGET_CAP)}…`;
  const line = target ? `${tool} ${target}` : tool;
  const note = memoryNoteText(input);
  return note ? `${line}\n${truncate(note, MEMORY_NOTE_CAP)}` : line;
}

/** Larger than TOOL_TEXT_CAP: an agent's progress note grows to ~5KB over a run (CL-Bench, 15
 *  questions), and cutting it at 2KB drops the newest lessons, which it appends at the end. */
const MEMORY_NOTE_CAP = 8000;

/** A file under a `memory/` or `memories/` folder, or a `MEMORY.md`: where an agent keeps the
 *  notes it writes for its future self (Claude Code's auto memory, Anthropic's memory protocol). */
const MEMORY_NOTE_PATH_RE = /(^|[\\/])(memory|memories)[\\/]|(^|[\\/])MEMORY\.md$/i;

/**
 * The text an agent wrote into one of its memory notes, so the write-back keeps the lesson in the
 * agent's own words. Every other file write stays a bare action line, but a note IS the decision:
 * reduced to `Write memories/prices.md`, extraction had to rebuild "prc is cents, prc_usd is NULL
 * for most rows" from the session's queries, and on CL-Bench it kept the unit and dropped the
 * caveat that the agent itself had written down. `content` is a Write, `new_string` an Edit.
 */
function memoryNoteText(input: unknown): string {
  if (!input || typeof input !== "object") return "";
  const rec = input as Record<string, unknown>;
  const path = rec.file_path ?? rec.path;
  if (typeof path !== "string" || !MEMORY_NOTE_PATH_RE.test(path)) return "";
  const text = rec.content ?? rec.new_string;
  return typeof text === "string" ? text.trim() : "";
}
