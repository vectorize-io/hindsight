/**
 * Import a harness's PAST sessions from local disk — the migration path off the older per-agent
 * plugins.
 *
 * Day to day, conversations reach a bank through live write-back, so a fresh install starts from
 * git history alone and knows nothing of what you discussed last month. The old per-agent plugins
 * stored their memory in differently-scoped banks (`claude-code::<project>` vs this package's
 * per-repo `coding-agent::{gitProject}`), and the server's bank import restores a whole bank rather
 * than merging — so those banks cannot be folded together. Re-reading the transcripts the agent
 * already wrote to disk sidesteps that entirely: the same conversations are re-extracted into
 * whichever bank is current.
 *
 * Scoped to ONE repo on purpose. This machine has ~14k Claude sessions; importing all of them would
 * cost extraction on every unrelated project. Each harness below can answer "which sessions belong
 * to this directory" cheaply.
 *
 * Only file-based harnesses are supported. opencode, Kilo, Cursor, Cline, Copilot and Devin keep
 * history in SQLite (`opencode.db`, `store.db`, …) whose schemas are internal and unversioned;
 * reading them would break on any upstream change, so they report as unsupported rather than
 * silently importing nothing.
 */
import { execFileSync } from "node:child_process";
import {
  closeSync,
  existsSync,
  openSync,
  readdirSync,
  readFileSync,
  readSync,
  statSync,
} from "node:fs";
import { createHash } from "node:crypto";
import { homedir } from "node:os";
import { join } from "node:path";
// Namespace import on purpose: `zstdDecompressSync` only exists on Node 22.15+, and a NAMED import
// of a missing builtin export is a load-time SyntaxError — which would break every entry point that
// pulls this module in, not just the dsh backfill (guarded below).
import * as zlib from "node:zlib";
import type { ChatSession } from "./types";
import { readClaudeTranscript } from "./transcript";
import { readCodexTranscript } from "./transcript-codex";
import { readDcodeTranscript } from "./transcript-dcode";
import { readDshEvents, type DshSessionEvent } from "./transcript-dsh";
import { zstdDecompressFrames } from "./zstd-frames";
import type { TransportTurn } from "./chat";

export interface HistoryImport {
  supported: boolean;
  /** Why, when unsupported — surfaced to the user rather than failing silently. */
  reason?: string;
  sessions: ChatSession[];
  /** Sessions skipped because nothing in them proves which repo they belong to. */
  unattributed?: number;
}

/** Claude encodes a project directory as its absolute path with EVERY non-alphanumeric character
 *  replaced by `-` — separators, dots, underscores, spaces, `+`, `@`, all of them. Case is kept,
 *  and runs are not collapsed (`/a-1/-b` -> `-a-1--b`), so this is a 1:1 character substitution.
 *  Verified against Claude Code 2.1.241: `hs_under_test` -> `hs-under-test`, and
 *  `hs+odd@repo v2` -> `hs-odd-repo-v2`. */
export function claudeProjectDir(repoDir: string, home = homedir()): string {
  return join(home, ".claude", "projects", repoDir.replace(/[^a-zA-Z0-9]/g, "-"));
}

/** Is `dir` the repo itself or somewhere inside it? */
export function withinRepo(dir: string | undefined, repoDir: string): boolean {
  return (
    !!dir && (dir === repoDir || dir.startsWith(repoDir.endsWith("/") ? repoDir : repoDir + "/"))
  );
}

/** First `cwd` recorded in a Claude transcript, i.e. where that session was working. */
function claudeSessionCwd(file: string): string | undefined {
  try {
    for (const line of readFileSync(file, "utf8").split("\n", 400)) {
      if (!line.includes('"cwd"')) continue;
      const cwd = (JSON.parse(line) as { cwd?: string }).cwd;
      if (cwd) return cwd;
    }
  } catch {
    /* unreadable or truncated transcript */
  }
  return undefined;
}

function toSession(id: string, turns: TransportTurn[]): ChatSession | undefined {
  // `action` turns are tool-call breadcrumbs; the interchange format carries prose only.
  const prose = turns
    .filter((t) => t.role === "user" || t.role === "assistant")
    .map((t) => ({
      role: t.role,
      text: t.content,
      ...(t.timestamp ? { timestamp: t.timestamp } : {}),
    }));
  return prose.length ? { id, turns: prose } : undefined;
}

/**
 * First line of a file, read in chunks.
 *
 * Codex's `session_meta` header is a single line that carries the agent's full base instructions —
 * tens of KB. Reading a fixed prefix and splitting on newline truncated it mid-string, so every
 * rollout failed to parse and the import silently found nothing. Capped so a file with no newline
 * can't pull an unbounded amount into memory.
 */
function firstLine(path: string, cap = 1_000_000): string | undefined {
  const fd = openSync(path, "r");
  try {
    const chunk = Buffer.alloc(64 * 1024);
    let acc = "";
    while (acc.length < cap) {
      const n = readSync(fd, chunk, 0, chunk.length, null);
      if (n <= 0) break;
      acc += chunk.subarray(0, n).toString("utf8");
      const nl = acc.indexOf("\n");
      if (nl !== -1) return acc.slice(0, nl);
    }
    return acc.length && acc.length < cap ? acc : undefined;
  } finally {
    closeSync(fd);
  }
}

/**
 * The entries of a directory, or none when there is nothing to list.
 *
 * `existsSync` answered only half of that: it is TRUE for a regular file sitting where a directory
 * was expected — a stray `~/.claude/projects/-Users-x-dev-repo-sub` — and the `readdirSync` behind
 * it then threw ENOTDIR out of `importLocalHistory`, which promises never to throw. One junk entry
 * killed the whole `--import-conversations` run instead of costing that entry (#3771).
 */
function listDir(dir: string): string[] {
  try {
    return readdirSync(dir);
  } catch {
    return []; // missing, unreadable, or a stray file where a directory was expected
  }
}

function jsonlFiles(dir: string): string[] {
  return listDir(dir)
    .filter((f) => f.endsWith(".jsonl"))
    .map((f) => join(dir, f));
}

/**
 * Claude Code: one directory per LAUNCH directory, one .jsonl per session.
 *
 * Running Claude from a subdirectory creates its own project dir, so an exact match on the repo
 * root silently misses that history (on a real machine, 64 of 107 project dirs were nested under
 * another). Candidate dirs are prefiltered by encoded-name prefix — cheap, since the encoding is
 * order-preserving — and then confirmed against the `cwd` recorded INSIDE each session, because the
 * name alone is ambiguous: `/` and `.` both encode to `-`, so `repo-sub` may be the subdirectory
 * `repo/sub` or an unrelated sibling repo called `repo-sub`.
 */
function claudeHistory(repoDir: string, home: string): HistoryImport {
  const root = join(home, ".claude", "projects");
  const exact = claudeProjectDir(repoDir, home);
  const prefix = exact + "-";
  const dirs = listDir(root)
    .map((d) => join(root, d))
    .filter((d) => d === exact || d.startsWith(prefix));
  const sessions: ChatSession[] = [];
  let unattributed = 0;
  for (const file of dirs.flatMap(jsonlFiles)) {
    // ONLY the cwd recorded inside the session may attribute it to a repo. Falling back to the
    // directory name would be a guess: every non-alphanumeric encodes to `-`, so `repo-sub` is either the
    // subdirectory `repo/sub` or an unrelated sibling repo — and a wrong guess files someone
    // else's conversation into this repo's memory, which is worse than importing nothing.
    // (Measured: 400/400 sampled sessions record a cwd, so this skips ~nothing in practice.)
    const cwd = claudeSessionCwd(file);
    if (!cwd) {
      unattributed++;
      continue;
    }
    if (!withinRepo(cwd, repoDir)) continue;
    try {
      const id = file
        .split("/")
        .pop()!
        .replace(/\.jsonl$/, "");
      const s = toSession(id, readClaudeTranscript(file));
      if (s) sessions.push(s);
    } catch {
      /* a single unreadable transcript must not abort the import */
    }
  }
  return { supported: true, sessions, unattributed };
}

/**
 * Codex: rollouts are partitioned by DATE, not project, so the repo is read from the `session_meta`
 * header each file opens with — cheap enough to check without parsing the whole transcript.
 */
function codexHistory(repoDir: string, home: string): HistoryImport {
  const root = join(home, ".codex", "sessions");
  if (!existsSync(root)) return { supported: true, sessions: [] };
  const files: string[] = [];
  const walk = (dir: string): void => {
    for (const entry of readdirSync(dir)) {
      const p = join(dir, entry);
      if (statSync(p).isDirectory()) walk(p);
      else if (entry.endsWith(".jsonl")) files.push(p);
    }
  };
  try {
    walk(root);
  } catch {
    return { supported: true, sessions: [] };
  }

  const sessions: ChatSession[] = [];
  for (const file of files) {
    try {
      const head = firstLine(file);
      if (!head) continue;
      const meta = JSON.parse(head) as { payload?: { cwd?: string; id?: string } };
      if (!withinRepo(meta?.payload?.cwd, repoDir)) continue;
      const s = toSession(meta.payload?.id ?? file, readCodexTranscript(file));
      if (s) sessions.push(s);
    } catch {
      /* skip unreadable/short files */
    }
  }
  return { supported: true, sessions };
}

/**
 * Dcode: transcripts are named for their THREAD, and carry no cwd of their own — the working
 * directory lives only in the LangGraph checkpoint store (`.state/sessions.db`), whose schema is
 * exactly the internal, unversioned kind this module refuses to read.
 *
 * `dcode threads list --json` is the supported way to ask the same question. It is a declared
 * contract, not a schema leak: the payload is `{schema_version, command, data:[…]}` and each row
 * carries `thread_id` and `cwd`. Verified against deepagents-code 0.1.65.
 */
export const DCODE_THREADS_SCHEMA_VERSION = 1;

interface DcodeThread {
  thread_id?: unknown;
  cwd?: unknown;
}

/** Path Dcode materializes a thread's transcript at: a truncated readable prefix, `--`, and the
 *  sha256 of the FULL thread id (`hooks/transcript.py:_safe_component`). Only the digest is
 *  load-bearing, so match on it and let the prefix vary. */
function dcodeTranscriptPath(root: string, threadId: string): string | undefined {
  const digest = createHash("sha256").update(threadId, "utf8").digest("hex");
  const suffix = `--${digest}.jsonl`;
  try {
    const match = readdirSync(root).find((entry) => entry.endsWith(suffix));
    return match ? join(root, match) : undefined;
  } catch {
    return undefined;
  }
}

function dcodeHistory(
  repoDir: string,
  home: string,
  runCli: (args: string[]) => string
): HistoryImport {
  const root = join(process.env.DEEPAGENTS_HOME || join(home, ".deepagents"), "transcripts");
  if (!existsSync(root)) return { supported: true, sessions: [] };

  let listed: unknown;
  try {
    listed = JSON.parse(runCli(["threads", "list", "--json"]));
  } catch {
    // No dcode on PATH (or it failed): the transcripts are unattributable without it.
    return {
      supported: false,
      reason:
        "dcode transcripts record no working directory, so `dcode threads list --json` is the " +
        "only way to tell which sessions belong to this repo — and the dcode CLI is not runnable " +
        "here",
      sessions: [],
    };
  }
  const payload = listed as { schema_version?: unknown; data?: unknown };
  if (payload?.schema_version !== DCODE_THREADS_SCHEMA_VERSION || !Array.isArray(payload.data)) {
    return {
      supported: false,
      reason: `unrecognized \`dcode threads list --json\` schema (expected schema_version ${DCODE_THREADS_SCHEMA_VERSION})`,
      sessions: [],
    };
  }

  const sessions: ChatSession[] = [];
  let unattributed = 0;
  for (const row of payload.data as DcodeThread[]) {
    const threadId = typeof row?.thread_id === "string" ? row.thread_id : undefined;
    if (!threadId) continue;
    if (typeof row.cwd !== "string") {
      unattributed++;
      continue;
    }
    if (!withinRepo(row.cwd, repoDir)) continue;
    const file = dcodeTranscriptPath(root, threadId);
    if (!file) continue; // listed thread whose projection was never materialized
    try {
      const session = toSession(threadId, readDcodeTranscript(file));
      if (session) sessions.push(session);
    } catch {
      /* a single unreadable transcript must not abort the import */
    }
  }
  return { supported: true, sessions, ...(unattributed ? { unattributed } : {}) };
}

/**
 * DeepSeek Harness: `$DSH_HOME/sessions/<project>/<encoded-id>/session.jsonl(.zstd)`.
 *
 * The project directory is a lossy, truncated rendering of the session's cwd, so it is used only to
 * narrow the walk — attribution still comes from the `cwd` in each log's header line, exactly like
 * the Claude reader. Logs are zstd by default (see core/zstd-frames.ts for why a plain decompress
 * of the whole file reads back only the header line).
 */
function dshHistory(repoDir: string, home: string): HistoryImport {
  const root = process.env.DSH_HOME
    ? join(process.env.DSH_HOME, "sessions")
    : join(home, ".dsh", "sessions");
  if (!existsSync(root)) return { supported: true, sessions: [] };
  if (typeof zlib.zstdDecompressSync !== "function") {
    return {
      supported: false,
      reason:
        "reading dsh session logs needs Node's built-in Zstandard support (Node 22.15+); " +
        `this import is running on ${process.version}`,
      sessions: [],
    };
  }
  const sessions: ChatSession[] = [];
  let unattributed = 0;
  for (const dir of readdirSync(root).map((project) => join(root, project))) {
    let sessionDirs: string[];
    try {
      sessionDirs = readdirSync(dir).map((id) => join(dir, id));
    } catch {
      continue; // a stray file where a project directory was expected
    }
    for (const sessionDir of sessionDirs) {
      const file = ["session.jsonl.zstd", "session.jsonl"]
        .map((name) => join(sessionDir, name))
        .find((candidate) => existsSync(candidate));
      if (!file) continue;
      try {
        const lines = readDshLog(file);
        const header = JSON.parse(lines[0] ?? "{}") as { cwd?: string; id?: string };
        if (!header.cwd) {
          unattributed++;
          continue;
        }
        if (!withinRepo(header.cwd, repoDir)) continue;
        const events = lines.slice(1).flatMap((line) => {
          try {
            return [JSON.parse(line) as DshSessionEvent];
          } catch {
            return []; // a packed chunk row or a torn tail line: not conversation either way
          }
        });
        const s = toSession(header.id ?? sessionDir, readDshEvents(events));
        if (s) sessions.push(s);
      } catch {
        /* a single unreadable log must not abort the import */
      }
    }
  }
  return { supported: true, sessions, unattributed };
}

/** The logical JSONL lines of one dsh session log, decompressing when the artifact is zstd. */
function readDshLog(file: string): string[] {
  const bytes = readFileSync(file);
  const text = file.endsWith(".zstd") ? zstdDecompressFrames(bytes) : bytes.toString("utf8");
  return text.split("\n").filter((line) => line.trim());
}

const SQLITE_HISTORY =
  "keeps session history in an internal SQLite database, whose schema is unversioned and would " +
  "break on any upstream change";

/** Read a harness's past sessions for one repo. Never throws. */
export function importLocalHistory(
  harness: string,
  repoDir: string,
  home = homedir(),
  /** Seam for tests: runs `dcode <args>` and returns stdout. */
  runDcodeCli?: (args: string[]) => string
): HistoryImport {
  switch (harness) {
    case "claude-code":
      return claudeHistory(repoDir, home);
    case "codex":
      return codexHistory(repoDir, home);
    case "dcode":
      return dcodeHistory(
        repoDir,
        home,
        runDcodeCli ??
          ((args) =>
            execFileSync("dcode", args, { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }))
      );
    case "dsh":
      return dshHistory(repoDir, home);
    case "opencode":
    // opencode v2 keeps sessions in the SAME `opencode.db` v1 does.
    case "opencode2":
    case "kilo":
    case "cursor-cli":
    case "cline-cli":
    case "copilot-cli":
    case "devin-cli":
      return { supported: false, reason: `${harness} ${SQLITE_HISTORY}`, sessions: [] };
    default:
      return { supported: false, reason: `no local history reader for ${harness}`, sessions: [] };
  }
}
