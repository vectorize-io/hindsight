import { rmSync } from "node:fs";
import { afterEach, describe, expect, it } from "vitest";
import {
  fileCursorStore,
  readSessionCache,
  sessionCacheFile,
  writeSessionCache,
} from "./session-cache";

const HARNESS = "codex-cursor-test";
const sessions = ["s1", "s2"];

afterEach(() => {
  for (const s of sessions) {
    rmSync(sessionCacheFile(HARNESS, s), { force: true });
    rmSync(sessionCacheFile(HARNESS, s).replace(/\.json$/, ".retain.json"), { force: true });
  }
});

describe("fileCursorStore", () => {
  it("round-trips a cursor across processes (a Stop hook has no memory)", () => {
    // Two stores, as two separate hook invocations would build them.
    fileCursorStore(HARNESS).write("s1", { turns: 4, fingerprint: "abc", bank: "b1", dirty: true });
    expect(fileCursorStore(HARNESS).read("s1")).toEqual({
      turns: 4,
      fingerprint: "abc",
      bank: "b1",
      dirty: true,
    });
  });

  it("keeps cursors separate per session", () => {
    const store = fileCursorStore(HARNESS);
    store.write("s1", { turns: 1, fingerprint: "a", bank: "b1" });
    store.write("s2", { turns: 9, fingerprint: "b", bank: "b1" });
    expect(store.read("s1")).toEqual({ turns: 1, fingerprint: "a", bank: "b1" });
    expect(store.read("s2")).toEqual({ turns: 9, fingerprint: "b", bank: "b1" });
  });

  it("reads as absent when the session was never written — the caller then replaces", () => {
    expect(fileCursorStore(HARNESS).read("s1")).toBeUndefined();
  });

  it("survives the prompt hook rewriting the session cache", () => {
    // The regression this file exists for. The cursor used to be a FIELD of the session cache, and
    // the prompt hook writes a fresh {turns, reflectAnswer, pages} object rather than merging — so
    // every user prompt dropped it, the next Stop found no cursor, and the incremental write-back
    // silently degraded to a full replace on every hook harness after a session's first turn.
    const store = fileCursorStore(HARNESS);
    store.write("s1", { turns: 5, fingerprint: "f", bank: "b1" });

    writeSessionCache(sessionCacheFile(HARNESS, "s1"), {
      turns: 2,
      pages: { atTurn: 2, list: [] },
    });

    expect(store.read("s1")).toEqual({ turns: 5, fingerprint: "f", bank: "b1" });
  });

  it("leaves the session cache alone", () => {
    // The inverse direction: writing a cursor must not disturb the recall state either.
    const file = sessionCacheFile(HARNESS, "s1");
    writeSessionCache(file, { turns: 3, reflectAnswer: "already ran" });
    fileCursorStore(HARNESS).write("s1", { turns: 2, fingerprint: "f", bank: "b1" });
    expect(readSessionCache(file)).toEqual({ turns: 3, reflectAnswer: "already ran" });
  });

  it("never exposes a half-written cursor", () => {
    // Writes go through a temp file and a rename, so a reader sees the old value or the new one.
    const store = fileCursorStore(HARNESS);
    store.write("s1", { turns: 1, fingerprint: "a", bank: "b1" });
    for (let i = 2; i <= 30; i++) {
      store.write("s1", { turns: i, fingerprint: "a".repeat(i * 200), bank: "b1" });
      const seen = store.read("s1");
      expect(seen?.turns).toBe(i);
      expect(seen?.fingerprint).toHaveLength(i * 200);
    }
  });
});
