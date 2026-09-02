/**
 * Bounded JSONL reading for the harness transcript readers.
 *
 * Every live-transcript reader used to do `readFileSync(path, "utf8").split("\n")`. Two things go
 * wrong with that as a session grows, and the second is silent:
 *
 *   - past V8's maximum string length (~537M chars) `readFileSync` throws ERR_STRING_TOO_LONG
 *     before a single record is parsed. The readers catch that as "unreadable file" and return no
 *     turns, so the Stop hook exits successfully having retained nothing — an agent that had been
 *     running for weeks just stopped updating memory, with no error anywhere (#3292).
 *   - well before that limit, decoding hundreds of MB into one string (and then a turn per line)
 *     costs several times the file size in heap, in a hook process that must not fall over.
 *
 * So: stream the file a chunk at a time, and read only the LAST `maxBytes` of it. Truncating the
 * head rather than the tail is what makes the cap safe to apply everywhere — the recent exchange is
 * the part worth retaining, and the incremental write-back (core/retain-cursor.ts) only sends turns
 * added since its cursor anyway. A transcript over the cap is reported, never dropped in silence.
 */
import { closeSync, openSync, readSync, statSync } from "node:fs";
import { StringDecoder } from "node:string_decoder";
import { log } from "./log";

/** Read granularity. Large enough that a multi-MB transcript is a few hundred syscalls. */
const CHUNK_BYTES = 64 * 1024;

/**
 * Most transcript we will decode, per read.
 *
 * A parsed transcript costs roughly 3x its bytes in heap (one string per line plus the turn
 * objects), so this keeps a hook process near ~100MB even in the worst case. It is far above any
 * real coding session: the largest transcripts seen in the wild are single-digit MB, and a file
 * over this has already stopped being a conversation anyone can extract meaning from.
 */
export const MAX_TRANSCRIPT_BYTES = 32 * 1024 * 1024;

export interface JsonlTail {
  /** Complete records, oldest first. Never holds more than one line plus a chunk in memory. */
  lines: Generator<string>;
  /** Bytes skipped from the head because the file exceeded the cap; 0 when it was read whole. */
  skippedBytes: number;
}

/**
 * The last `maxBytes` of a JSONL file, one complete record at a time.
 *
 * Fail-open like the readers it serves: a missing or unreadable file yields no records rather than
 * throwing. `scope` only names the harness in the truncation warning.
 */
export function readJsonlTail(path: string, opts: { scope: string; maxBytes?: number }): JsonlTail {
  const maxBytes = opts.maxBytes ?? MAX_TRANSCRIPT_BYTES;
  let size: number;
  try {
    size = statSync(path).size;
  } catch {
    return { lines: emptyLines(), skippedBytes: 0 };
  }
  const skippedBytes = size > maxBytes ? size - maxBytes : 0;
  if (skippedBytes > 0) {
    // The whole point of #3292: an oversized transcript must not fail quietly.
    log.warn(opts.scope, "transcript too large — retaining the most recent portion only", {
      path,
      sizeBytes: size,
      skippedBytes,
    });
  }
  return { lines: streamLines(path, skippedBytes), skippedBytes };
}

function* emptyLines(): Generator<string> {
  /* nothing to yield — see the fail-open contract above */
}

/** Yield complete lines from `start` to EOF. The fd is opened lazily (on first iteration) and
 *  closed even if the consumer abandons the generator early — for..of calls return() for us. */
function* streamLines(path: string, start: number): Generator<string> {
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
    // Begin one byte EARLY so the cut can be classified: if that byte is the newline ending the
    // previous record, the cut was clean and the "partial" we drop is an empty string, costing
    // nothing. Otherwise we really did land inside a record and drop that fragment.
    let position = start > 0 ? start - 1 : 0;
    let dropPartial = start > 0;

    for (;;) {
      const bytesRead = readSync(fd, buffer, 0, buffer.length, position);
      if (bytesRead <= 0) break;
      position += bytesRead;
      pending += decoder.write(buffer.subarray(0, bytesRead));

      let newline: number;
      while ((newline = pending.indexOf("\n")) !== -1) {
        const line = pending.slice(0, newline);
        pending = pending.slice(newline + 1);
        if (dropPartial) {
          dropPartial = false;
          continue;
        }
        yield line;
      }
    }

    pending += decoder.end();
    // A final record with no trailing newline still counts (and is not the dropped partial).
    if (pending && !dropPartial) yield pending;
  } finally {
    closeSync(fd);
  }
}
