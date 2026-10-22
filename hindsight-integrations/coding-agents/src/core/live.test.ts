import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  NOTE_CREDIT,
  NOTE_DISMISS,
  NOTE_LEAD,
  LIVE_DECLARATION,
  renderNote,
  sourceDiff,
  noteCredit,
  noteDismiss,
} from "./live";

const where = (document_id: string, mentioned_at?: string) =>
  renderNote({ seq: 1, text: "K.", source: { document_id, mentioned_at } }).split("Source: ")[1];

describe("the note's source", () => {
  it("renders the note, where to read it, and the quote", () => {
    expect(
      renderNote({
        seq: 1,
        text: "K is 1024.",
        source: { document_id: "git:abc", quote: "memory as 1024" },
      })
    ).toBe(
      `${NOTE_LEAD.replace("Note", "Note #1")} K is 1024. Source: git show abc — "memory as 1024". Tell the user about it: "${noteCredit(1)} …" if you use it, else "${noteDismiss(1)} …; not needed here: …"`
    );
    expect(renderNote({ seq: 1, text: "K is 1024." })).toBe(
      `${NOTE_LEAD.replace("Note", "Note #1")} K is 1024. Source: project memory. Tell the user about it: "${noteCredit(1)} …" if you use it, else "${noteDismiss(1)} …; not needed here: …"`
    );
  });

  it("asks the agent to mention every note: credit one it used, name one it did not need", () => {
    // The server counts these exact lines (engine/sessions/service.py CREDIT_MARK / DISMISS_MARK).
    expect(noteCredit(8)).toContain("From Hindsight memory (live session #8)");
    expect(noteDismiss(8)).toContain("Hindsight note (live session #8)");
    expect(LIVE_DECLARATION).toContain(NOTE_CREDIT);
    expect(LIVE_DECLARATION).toContain(NOTE_DISMISS);
  });

  it("names where the agent can read each kind of document", () => {
    expect(where("git:0123456789abcdef")).toMatch(/^git show 0123456789ab\./);
    expect(where("gitlog:main")).toMatch(/^git log\./);
    expect(where("conversation:s1", "2026-10-01T10:00:00Z")).toMatch(
      /^a past session \(2026-10-01\)\./
    );
    expect(where("survey:x")).toMatch(/^project memory\./);
  });
});

describe("sourceDiff", () => {
  it("includes new files, not only edits to tracked ones", () => {
    const repo = mkdtempSync(join(tmpdir(), "live-diff-"));
    execFileSync("git", ["init", "-q", repo]);
    writeFileSync(join(repo, "units.py"), "K = 1024\n");
    expect(sourceDiff(repo)).toContain("+K = 1024");
  });
});
