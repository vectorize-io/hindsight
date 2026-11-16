/**
 * `autoInject: "tree"`: the bank's observations as a folder of markdown files the agent reads like
 * any other files (`GET /observations/tree`, the server's observation tree). At session start the
 * folder is rebuilt from the bank and the agent is told where it is, with the root index inline.
 *
 * Why a folder and not an injection: on CL-Bench database_exploration (Sonnet 5.5) an agent that
 * read its memory as files, written this way before each session, scored 0.787 and 0.742 against
 * 0.776 for Anthropic's own memory-folder prompt, with no reflect call on the hot path.
 */
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, isAbsolute, join, normalize, sep } from "node:path";

export interface ObservationTreeFile {
  path: string;
  kind: "index" | "topic";
  content: string;
}

export interface ObservationTreeFolder {
  path: string;
  folders: ObservationTreeFolder[];
  files: ObservationTreeFile[];
}

/** Root index text inlined in the session-start context; the rest stays in the files. Claude Code
 *  replaces hook context over 10,000 characters with a file path the model rarely opens. */
const INLINE_INDEX_CAP = 6000;

/** `~/.hindsight/memory-tree/<bank>/memory`. Outside the repo, so it never lands in a commit. The
 *  last segment is `memory` on purpose: a note the agent writes there matches the write-back's
 *  memory-note rule (transcript-util memoryNoteText), so its text is retained, not just its path. */
export function memoryTreeDir(bankId: string, home = homedir()): string {
  return join(home, ".hindsight", "memory-tree", bankId.replace(/[^A-Za-z0-9._-]/g, "_"), "memory");
}

/** Replace `dir` with the tree's files. Returns the root's index text ("" when the tree has none). */
export function writeMemoryTree(dir: string, tree: ObservationTreeFolder): string {
  // ponytail: wipes the folder each session start; a second concurrent session of the same bank
  // loses notes it has not written back yet. Per-session folders if that ever matters.
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  const stack = [tree];
  while (stack.length) {
    const folder = stack.pop()!;
    for (const f of folder.files) {
      // Paths come from the server: never let one escape the folder.
      const rel = normalize(f.path);
      if (!rel || isAbsolute(rel) || rel.split(sep).includes("..")) continue;
      const out = join(dir, rel);
      mkdirSync(dirname(out), { recursive: true });
      writeFileSync(out, f.content);
    }
    stack.push(...folder.folders);
  }
  const root = tree.files.find((f) => f.kind === "index") ?? tree.files[0];
  return root?.content ?? "";
}

/** The session-start context pointing the agent at the folder. The `<hindsight_memory>` tag keeps it
 *  out of the session write-back (transcript-util MEMORY_TAG_RE). */
export function buildMemoryTreeContext(dir: string, rootIndex: string): string {
  const cut = rootIndex.lastIndexOf("\n", INLINE_INDEX_CAP);
  const index =
    rootIndex.length > INLINE_INDEX_CAP
      ? `${rootIndex.slice(0, cut > 0 ? cut : INLINE_INDEX_CAP)}\n(… rest in index.md)`
      : rootIndex;
  return (
    "<hindsight_memory>\n" +
    `What this project's memory has learned from past sessions is in ${dir}, as markdown files: ` +
    "one file per topic, and an index.md in each folder. Before you start, read the files that bear " +
    "on your task, and grep the folder for the names you are working with: past mistakes and their " +
    "fixes are recorded there. To keep a lesson for future sessions, write it as a short note in " +
    "that folder; notes are saved to memory when the session ends, and the folder is rebuilt from " +
    "memory at the next session start.\n\n" +
    (index.trim() ? `${index.trim()}\n` : "The folder is empty: nothing has been learned yet.\n") +
    "</hindsight_memory>"
  );
}
