import { describe, expect, it } from "vitest";
import type { TransportTurn } from "./chat";
import { fingerprintTurns, memoryCursorStore, planRetain } from "./retain-cursor";

const turn = (i: number): TransportTurn => ({ role: "user", content: `turn ${i}` });
const turns = (n: number, from = 0): TransportTurn[] =>
  Array.from({ length: n }, (_, i) => turn(from + i));

const BANK = "coding-agent::repo";

const cursorFor = (all: TransportTurn[], count: number) => ({
  turns: count,
  fingerprint: fingerprintTurns(all, count),
  bank: BANK,
});

const SUPPORTED = { appendSupported: true, bank: BANK };

describe("planRetain", () => {
  it("appends only the turns added since the last write", () => {
    const all = turns(5);
    expect(planRetain(all, cursorFor(all, 3), SUPPORTED)).toEqual({ mode: "append", fromTurn: 3 });
  });

  it("replaces the whole document on the first write of a session", () => {
    expect(planRetain(turns(3), undefined, SUPPORTED)).toEqual({ mode: "replace" });
  });

  it("replaces when the previous write was never confirmed", () => {
    // The one case that would DUPLICATE turns inside the document if we appended anyway: the
    // server may or may not hold the last slice, and only a replace settles it.
    const all = turns(5);
    expect(planRetain(all, { ...cursorFor(all, 3), dirty: true }, SUPPORTED)).toEqual({
      mode: "replace",
    });
  });

  it("replaces when the transcript was rewritten rather than extended", () => {
    // Compaction: same turn count, different content. Appending would splice two conversations.
    const original = turns(5);
    const stale = cursorFor(original, 3);
    const compacted = [{ role: "user", content: "summary of earlier work" }, ...turns(4, 10)];
    expect(planRetain(compacted, stale, SUPPORTED)).toEqual({ mode: "replace" });
  });

  it("replaces when the transcript shrank below the cursor", () => {
    const all = turns(5);
    expect(planRetain(turns(2), cursorFor(all, 4), SUPPORTED)).toEqual({ mode: "replace" });
  });

  it("replaces when the server cannot deduplicate a resubmitted write", () => {
    const all = turns(5);
    expect(planRetain(all, cursorFor(all, 3), { appendSupported: false, bank: BANK })).toEqual({
      mode: "replace",
    });
  });

  it("replaces when the session moved to another bank", () => {
    // Same session id, different bank: the hook derives the bank from each event's cwd, so a
    // session that moves between repos (#3133) would otherwise append its tail into a bank that
    // holds no document for it, silently losing everything before the move.
    const all = turns(5);
    expect(planRetain(all, cursorFor(all, 3), { ...SUPPORTED, bank: "other-repo" })).toEqual({
      mode: "replace",
    });
  });

  it("skips when nothing was added since the last write", () => {
    const all = turns(4);
    expect(planRetain(all, cursorFor(all, 4), SUPPORTED)).toEqual({ mode: "skip" });
  });

  it("skips an empty transcript", () => {
    expect(planRetain([], undefined, SUPPORTED)).toEqual({ mode: "skip" });
  });
});

describe("fingerprintTurns", () => {
  it("changes when the retained prefix changes, not when later turns are appended", () => {
    const all = turns(5);
    expect(fingerprintTurns([...all, turn(99)], 3)).toBe(fingerprintTurns(all, 3));
    const edited = [turn(0), { role: "user", content: "edited" }, ...turns(3, 2)];
    expect(fingerprintTurns(edited, 3)).not.toBe(fingerprintTurns(all, 3));
  });

  it("distinguishes prefixes of different length", () => {
    const all = turns(5);
    expect(fingerprintTurns(all, 2)).not.toBe(fingerprintTurns(all, 3));
  });
});

describe("memoryCursorStore", () => {
  it("keeps cursors per session", () => {
    const store = memoryCursorStore();
    expect(store.read("a")).toBeUndefined();
    store.write("a", { turns: 2, fingerprint: "f", bank: "b1" });
    store.write("b", { turns: 7, fingerprint: "g", bank: "b1" });
    expect(store.read("a")).toEqual({ turns: 2, fingerprint: "f", bank: "b1" });
    expect(store.read("b")).toEqual({ turns: 7, fingerprint: "g", bank: "b1" });
  });
});
