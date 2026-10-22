/**
 * The live session's client half (`autoInject: "live"`): Hindsight beside the coding agent,
 * speaking only when memory holds something the agent must hear. The session itself runs in
 * Hindsight (core/session-sync.ts); this module is what the plugin adds around it — how a
 * note reads to the agent, where its source is, and the change the session checks edits against.
 */
import { gitWithTimeout } from "./git";
import type { SessionNote } from "./session-events";

/** How the agent hears a note — the wording the benchmark validated. */
export const NOTE_LEAD = "🧠 Note from this project's memory (past sessions and commits):";

/**
 * How the agent credits a note to the user: the same blockquote as every other Hindsight
 * memory it uses (inject.ts, knowledge-tools.ts), named after the live session.
 */
/** The credit line for note `n` (its seq): the server counts the notes named this way. */
export const noteCredit = (n: number | string): string =>
  `> 🧠 **From Hindsight memory (live session #${n})** —`;
export const NOTE_CREDIT = noteCredit("<note number>");

/**
 * How the agent tells the user about a note it did not need: every note is mentioned, so the user
 * sees Hindsight at work even when it had nothing to add. The server counts both lines.
 */
export const noteDismiss = (n: number | string): string =>
  `> 🧠 **Hindsight note (live session #${n})** —`;
export const NOTE_DISMISS = noteDismiss("<note number>");

/**
 * Told to the agent once, at session start. Without it a note read as an injection: Claude called a
 * real decision (a recorded corp-beta override) a prompt injection and refused it three times.
 */
export const LIVE_DECLARATION =
  "Hindsight follows this session live. It knows this project's past sessions and the decisions made " +
  `in them, and while you work it may post notes starting with "${NOTE_LEAD.split(" (")[0].replace("Note", "Note #<number>")}". A note ` +
  "is a project requirement even when the request does not mention it: a fix that contradicts a past " +
  "decision is a wrong fix. Each note says where to read its source: check it; if it says what the " +
  "note claims, apply the note as part of the task. Push back only if the source does not say so. " +
  "Tell the user about every note you get, in your reply, one line per note, naming the note by its " +
  "number, so they see memory at work. A note that shaped what you did:\n" +
  `${NOTE_CREDIT} <what the note told you and what you did with it>\n` +
  "A note you did not need:\n" +
  `${NOTE_DISMISS} <what it said>; not needed here: <why>\n` +
  "Never use the first line for a note you did not use.";

/** Where the agent can read a note's source. Commit ids are this repo's own, so git can show them. */
function whereFor(source: SessionNote["source"]): string {
  const id = source?.document_id ?? "";
  if (id.startsWith("git:")) return `git show ${id.slice(4, 16)}`;
  if (id.startsWith("gitlog:")) return "git log";
  if (id.startsWith("conversation:") || id.startsWith("chat:"))
    return `a past session${source?.mentioned_at ? ` (${source.mentioned_at.slice(0, 10)})` : ""}`;
  return "project memory";
}

/**
 * What the agent reads for a note: the note, where its source is and the source's own words, then
 * a reminder to credit it (a note can land long after the session-start declaration).
 */
export function renderNote(n: SessionNote): string {
  const quote = n.source?.quote ?? "";
  return (
    `${NOTE_LEAD.replace("Note", `Note #${n.seq}`)} ${n.text.trim()} Source: ${whereFor(n.source)}${quote ? ` — "${quote}"` : ""}` +
    `. Tell the user about it: "${noteCredit(n.seq)} …" if you use it, else "${noteDismiss(n.seq)} …; not needed here: …"`
  );
}

const git = (cwd: string, ...args: string[]): string => {
  try {
    return gitWithTimeout(cwd, 5000, ...args);
  } catch (e) {
    // `git diff --no-index` exits 1 when the files differ, which is every new file: keep its output.
    const out = (e as { stdout?: unknown }).stdout;
    return typeof out === "string" ? out : "";
  }
};

/** The agent's change to the source: tracked edits plus new files (`git diff` alone skips those). */
export function sourceDiff(cwd: string): string {
  const spec = ["--", ".", ":(exclude)tests/**", ":(exclude)test/**", ":(exclude)**/*.test.*"];
  let out = git(cwd, "diff", "HEAD", ...spec);
  for (const f of git(cwd, "ls-files", "--others", "--exclude-standard", ...spec)
    .split("\n")
    .filter(Boolean)
    .slice(0, 20))
    out += git(cwd, "diff", "--no-index", "/dev/null", f);
  return out;
}

/**
 * Late notes as the user reads them at the end of a turn: what each note says, without the
 * instructions written for the agent (where to check it, how to credit it).
 */
export function notesForUser(rendered: string): string {
  const says = rendered
    .split("\n\n")
    .map((n) =>
      n
        .replace(/^🧠 Note (#\d+ )?from this project's memory \([^)]*\):\s*/, "")
        .split(" Source: ")[0]
        .trim()
    )
    .filter(Boolean);
  return `🧠 Hindsight has a note for the agent (it reads it with your next message):\n${says.map((t) => `- ${t}`).join("\n")}`;
}
