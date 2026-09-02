import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MAX_TRANSCRIPT_BYTES, readJsonlTail } from "./jsonl";

// Spy on the REAL readFileSync (not a stub that throws): the point is to prove nothing in this path
// calls it, which is what #3292 came down to — past V8's maximum string length it throws
// ERR_STRING_TOO_LONG, and the readers turned that into "no turns" with no error anywhere.
vi.mock("node:fs", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:fs")>();
  return { ...actual, readFileSync: vi.fn(actual.readFileSync) };
});

let root: string;
let file: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "hs-jsonl-"));
  file = join(root, "transcript.jsonl");
  vi.mocked(readFileSync).mockClear();
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

const read = (maxBytes?: number) => {
  const tail = readJsonlTail(file, { scope: "test", maxBytes });
  return { lines: [...tail.lines], skippedBytes: tail.skippedBytes };
};

describe("readJsonlTail", () => {
  it("yields every record of a file under the cap", () => {
    writeFileSync(file, ["a", "b", "c"].join("\n") + "\n");
    expect(read()).toEqual({ lines: ["a", "b", "c"], skippedBytes: 0 });
  });

  it("yields a final record that has no trailing newline", () => {
    writeFileSync(file, "a\nb");
    expect(read().lines).toEqual(["a", "b"]);
  });

  it("preserves blank and whitespace-only records for the caller to skip", () => {
    // The readers decide what to drop (they trim and skip empties); this must not silently change
    // the record stream they used to get from split("\n").
    writeFileSync(file, "a\n\n b \nc\n");
    expect(read().lines).toEqual(["a", "", " b ", "c"]);
  });

  it("keeps \\r on CRLF records, exactly as split('\\n') did", () => {
    writeFileSync(file, "a\r\nb\r\n");
    expect(read().lines).toEqual(["a\r", "b\r"]);
  });

  it("reads the TAIL when the file exceeds the cap, dropping the record it lands inside", () => {
    // The cap lands mid-record; that fragment must be discarded whole rather than emitted as a
    // half line that fails to parse.
    const records = ["1111111111", "2222222222", "3333333333", "4444444444"];
    writeFileSync(file, records.join("\n") + "\n"); // 44 bytes
    const { lines, skippedBytes } = read(25);
    expect(skippedBytes).toBe(19);
    expect(lines).toEqual(["3333333333", "4444444444"]);
  });

  it("keeps whole records when the cap lands exactly on a boundary", () => {
    writeFileSync(file, ["aaaa", "bbbb", "cccc"].join("\n") + "\n"); // 15 bytes
    // Last 10 bytes start exactly at "bbbb"; the leading newline is the dropped 'partial'.
    expect(read(10).lines).toEqual(["bbbb", "cccc"]);
  });

  it("reassembles a multi-byte character split across the read boundary", () => {
    // 'é' is 2 bytes; place it so its first byte ends one 64KB chunk and its second starts the next.
    const pad = "a".repeat(65535);
    writeFileSync(file, `${pad}é-tail\nsecond\n`);
    const lines = read().lines;
    expect(lines[0].endsWith("é-tail")).toBe(true);
    expect(lines[0]).not.toContain("�"); // no replacement char: the sequence survived
    expect(lines[1]).toBe("second");
  });

  it("yields nothing for a missing file instead of throwing", () => {
    expect(read()).toEqual({ lines: [], skippedBytes: 0 });
  });

  it("yields nothing for an empty file", () => {
    writeFileSync(file, "");
    expect(read().lines).toEqual([]);
  });

  it("closes the file even when the consumer stops early", () => {
    writeFileSync(file, ["a", "b", "c"].join("\n") + "\n");
    const { lines } = readJsonlTail(file, { scope: "test" });
    for (const line of lines) {
      expect(line).toBe("a");
      break; // for..of calls the generator's return(), which must run the finally that closes the fd
    }
    expect(lines.next().done).toBe(true);
  });

  it("never reads the whole file into one string", () => {
    writeFileSync(file, "a\nb\n");
    expect(read().lines).toEqual(["a", "b"]);
    expect(vi.mocked(readFileSync)).not.toHaveBeenCalled();
  });

  it("caps at 32MB by default", () => {
    expect(MAX_TRANSCRIPT_BYTES).toBe(32 * 1024 * 1024);
  });
});
