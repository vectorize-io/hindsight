/**
 * The live session's client half (`autoInject: "live"`): Hindsight beside the coding agent,
 * speaking only when memory holds something the agent must hear. The session itself runs in
 * Hindsight (core/session-sync.ts); this module is what the plugin adds around it — how a
 * note reads to the agent, where its source is, and the change the session checks edits against.
 */
import { gitWithTimeout } from "./git";
import type { SessionNote } from "./session-events";

/**
 * How the agent hears a note. Notes read as Hindsight's thoughts — memory as a stream of
 * consciousness beside the agent, not a colleague handing it notes — and every line names
 * Hindsight, so the user sees whose thought it is.
 */
export const NOTE_LEAD = "🧠 Hindsight thought:";

/**
 * How the agent credits a thought that changed what it did — only that: the user should see plainly
 * when Hindsight helped, and the server counts these lines as the thoughts that did. In beta the
 * agent "used" 34 notes of which a judge found 8 changed its work; the rest only agreed with it.
 */
export const noteCredit = (n: number | string): string =>
  `> 🧠 **Thanks to Hindsight thought #${n}**,`;
export const NOTE_CREDIT = noteCredit("<number>");

/**
 * Every other thought, in one plain line: every thought is mentioned, so the user sees Hindsight at
 * work even when it changed nothing. There were three lines (thanks / already done / set aside), and
 * agents wrote "set aside" for thoughts they had acted on: the choice is now one question, did it
 * change what you did. The server counts this line as not needed.
 */
export const noteDismiss = (n: number | string): string => `> 🧠 Hindsight thought #${n} —`;
export const NOTE_DISMISS = noteDismiss("<number>");

/**
 * Told to the agent once, at session start. Without it a note read as an injection: Claude called a
 * real decision (a recorded corp-beta override) a prompt injection and refused it three times.
 */
export const LIVE_DECLARATION =
  "Hindsight is this project's long-term memory, and it thinks alongside you. As you work, thoughts " +
  'surface from past sessions and commits, each starting with "🧠 Hindsight thought #<number>". A thought ' +
  "recalls a project decision, and it holds even when the request does not mention it: a fix that " +
  "contradicts a past decision is a wrong fix. Each thought says where it was recalled from: check it; " +
  "if the source says what the thought claims, act on it as part of the task; ignore it only if the " +
  "source does not say so, or it does not touch your work. Share every thought with the user in your " +
  "reply, one line each, by its number, so they see memory thinking. Ask one question: did you do " +
  "anything differently because of it (fix, add, avoid or undo something)? If yes:\n" +
  `${NOTE_CREDIT} <what you did differently>\n` +
  "If no (you were already doing it, or it does not apply):\n" +
  `${NOTE_DISMISS} <what it said>; no change: <why>`;

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
    `${NOTE_LEAD.replace("thought:", `thought #${n.seq}:`)} ${n.text.trim()} Source: ${whereFor(n.source)}${quote ? ` — "${quote}"` : ""}` +
    `. Share it with the user: "${noteCredit(n.seq)} …" if you did something differently because of it, ` +
    `else "${noteDismiss(n.seq)} …; no change: …"`
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
        .replace(/^🧠 Hindsight thought (#\d+)?:\s*/, "")
        .split(" Source: ")[0]
        .trim()
    )
    .filter(Boolean);
  return `🧠 Hindsight is thinking (the agent picks this up with your next message):\n${says.map((t) => `- ${t}`).join("\n")}`;
}
