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
 *  Tag-structural, not content-guessing. The block is removed and any surrounding text KEPT — a
 *  message that is nothing but a wrapper then renders empty and is dropped as a no-content turn. */
const MEMORY_TAG_RE =
  /<(hook_prompt|task-notification|system-reminder|hindsight_memory|hindsight_memories|hindsight_bank|relevant_memories|user_feedback|hindsight_knowledge|hindsight_knowledge_refresh)\b[\s\S]*?<\/\1>/g;

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
 * needs WHAT was touched.
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
  return target ? `${tool} ${target}` : tool;
}
