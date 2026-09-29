import { describe, expect, it } from "vitest";
import { normalizeNote } from "../src/frontmatter";

describe("normalizeNote", () => {
  it("returns the raw body (trimmed) with the folder when there is no frontmatter", () => {
    const note = normalizeNote("  # Title\n\njust a body  ", "Work/Clients");
    expect(note.body).toBe("# Title\n\njust a body");
    expect(note.tags).toEqual([]);
    expect(note.timestamp).toBeUndefined();
    expect(note.metadata).toEqual({ folder: "Work/Clients" });
  });

  it("omits the folder key when the folder is empty (vault root)", () => {
    const note = normalizeNote("body", "");
    expect(note.metadata).toEqual({});
  });

  it("lifts block-list tags and aliases, dedupes, and strips the frontmatter", () => {
    const raw = [
      "---",
      "tags:",
      "  - alpha",
      "  - beta",
      "  - alpha",
      "aliases:",
      "  - Alpha",
      "---",
      "body text",
    ].join("\n");
    const note = normalizeNote(raw, "");
    expect(note.tags).toEqual(["alpha", "beta", "Alpha"]);
    expect(note.body).toBe("body text");
  });

  it("parses an inline flow list of tags", () => {
    const note = normalizeNote('---\ntags: [a, "b", c]\n---\nbody', "");
    expect(note.tags).toEqual(["a", "b", "c"]);
  });

  it("lifts created into the timestamp", () => {
    const note = normalizeNote("---\ncreated: 2026-03-01T08:00:00Z\n---\nbody", "");
    expect(note.timestamp).toBe("2026-03-01T08:00:00Z");
  });

  it("uses the first of created/date and ignores the later one", () => {
    const note = normalizeNote("---\ncreated: 2026-01-01\ndate: 2026-12-31\n---\nbody", "");
    expect(note.timestamp).toBe("2026-01-01");
  });

  it("lifts other scalars into metadata with quotes stripped", () => {
    const note = normalizeNote('---\nstatus: "active"\nauthor: ben\n---\nbody', "Notes");
    expect(note.metadata).toEqual({ folder: "Notes", status: "active", author: "ben" });
  });

  it("treats an unterminated frontmatter block as plain body", () => {
    const raw = "---\ntags:\n  - alpha\nstill going";
    const note = normalizeNote(raw, "");
    expect(note.tags).toEqual([]);
    expect(note.body).toBe(raw);
  });
});
