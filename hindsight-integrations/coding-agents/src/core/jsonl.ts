/**
 * Streaming JSONL reading for the harness transcript readers.
 *
 * Every live-transcript reader used to do `readFileSync(path, "utf8").split("\n")`. Past V8's
 * maximum string length (~537M chars) that throws ERR_STRING_TOO_LONG before a single record is
 * parsed; the readers caught it as "unreadable file" and returned no turns, so the Stop hook exited
 * successfully having retained nothing (#3292).
 *
 * So: stream the file a chunk at a time. Memory is bounded by the largest record plus whatever the
 * reader keeps — and the readers keep only rendered prose and one-line tool calls, dropping tool
 * results and thinking, which are the bulk of a transcript's bytes.
 *
 * The file is always read WHOLE. An earlier version read only the last 32MB, but a sliding tail
 * window shifts every turn's index on each read, which breaks the absolute turn count and prefix
 * fingerprint the append cursor (core/retain-cursor.ts) relies on — any session over the cap fell
 * back to re-sending its entire transcript every turn, forever (#4380).
 */
import { closeSync, openSync, readSync } from "node:fs";
import { StringDecoder } from "node:string_decoder";

/** Read granularity. Large enough that a multi-MB transcript is a few hundred syscalls. */
const CHUNK_BYTES = 64 * 1024;

/**
 * Every record of a JSONL file, oldest first, one complete line at a time.
 *
 * Fail-open like the readers it serves: a missing or unopenable file yields no records rather than
 * throwing. The fd is opened lazily (on first iteration) and closed even if the consumer abandons
 * the generator early — for..of calls return() for us.
 */
export function* readJsonl(path: string): Generator<string> {
  let fd: number;
  try {
    fd = openSync(path, "r");
  } catch {
    return;
  }
  try {
    const buffer = Buffer.allocUnsafe(CHUNK_BYTES);
    // StringDecoder holds back a trailing partial UTF-8 sequence, so a multi-byte character split
    // across two chunks is reassembled instead of becoming two replacement chars.
    const decoder = new StringDecoder("utf8");
    let pending = "";
    let position = 0;

    for (;;) {
      const bytesRead = readSync(fd, buffer, 0, buffer.length, position);
      if (bytesRead <= 0) break;
      position += bytesRead;
      pending += decoder.write(buffer.subarray(0, bytesRead));

      let newline: number;
      while ((newline = pending.indexOf("\n")) !== -1) {
        yield pending.slice(0, newline);
        pending = pending.slice(newline + 1);
      }
    }

    pending += decoder.end();
    // A final record with no trailing newline still counts.
    if (pending) yield pending;
  } finally {
    closeSync(fd);
  }
}
