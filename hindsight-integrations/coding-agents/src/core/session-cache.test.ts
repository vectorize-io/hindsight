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
  for (const s of sessions) rmSync(sessionCacheFile(HARNESS, s), { force: true });
});

describe("fileCursorStore", () => {
  it("round-trips a cursor across processes (a Stop hook has no memory)", () => {
    // Two stores, as two separate hook invocations would build them.
    fileCursorStore(HARNESS).write("s1", { turns: 4, fingerprint: "abc", dirty: true });
    expect(fileCursorStore(HARNESS).read("s1")).toEqual({
      turns: 4,
      fingerprint: "abc",
      dirty: true,
    });
  });

  it("keeps cursors separate per session", () => {
    const store = fileCursorStore(HARNESS);
    store.write("s1", { turns: 1, fingerprint: "a" });
    store.write("s2", { turns: 9, fingerprint: "b" });
    expect(store.read("s1")).toEqual({ turns: 1, fingerprint: "a" });
    expect(store.read("s2")).toEqual({ turns: 9, fingerprint: "b" });
  });

  it("reads as absent when the session was never written — the caller then replaces", () => {
    expect(fileCursorStore(HARNESS).read("s1")).toBeUndefined();
  });

  it("preserves the rest of the session cache it shares a file with", () => {
    const file = sessionCacheFile(HARNESS, "s1");
    writeSessionCache(file, { turns: 3, reflectAnswer: "already ran" });
    fileCursorStore(HARNESS).write("s1", { turns: 2, fingerprint: "f" });
    expect(readSessionCache(file)).toEqual({
      turns: 3,
      reflectAnswer: "already ran",
      retain: { turns: 2, fingerprint: "f" },
    });
  });
});
