import { describe, it, expect } from "vitest";
import { stringifyFrontmatter, parseDocument, type Frontmatter } from "../src/frontmatter.js";
import { renderMentalModel, fileNameFor } from "../src/format.js";

describe("frontmatter", () => {
  it("round-trips scalars and arrays", () => {
    const data: Frontmatter = {
      id: "user-preferences",
      name: "User Preferences",
      tags: ["team", "ui"],
      empty: [],
      flag: true,
      n: 7,
      missing: null,
    };
    const text = `${stringifyFrontmatter(data)}\n\nbody here\n`;
    const parsed = parseDocument(text);
    expect(parsed.body.trim()).toBe("body here");
    expect(parsed.frontmatter.id).toBe("user-preferences");
    expect(parsed.frontmatter.name).toBe("User Preferences");
    expect(parsed.frontmatter.tags).toEqual(["team", "ui"]);
    expect(parsed.frontmatter.empty).toEqual([]);
    expect(parsed.frontmatter.flag).toBe(true);
    expect(parsed.frontmatter.n).toBe(7);
    expect(parsed.frontmatter.missing).toBeNull();
  });

  it("quotes strings containing YAML-significant characters", () => {
    const text = stringifyFrontmatter({ q: "What are the user's needs: now?" });
    expect(text).toContain('q: "What are the user');
    const parsed = parseDocument(`${text}\n\nx\n`);
    expect(parsed.frontmatter.q).toBe("What are the user's needs: now?");
  });

  it("treats a document with no frontmatter as pure body", () => {
    const parsed = parseDocument("# just markdown\n");
    expect(parsed.frontmatter).toEqual({});
    expect(parsed.body).toBe("# just markdown\n");
  });
});

describe("renderMentalModel", () => {
  it("emits frontmatter + content body", () => {
    const md = renderMentalModel({
      id: "team-comms",
      bank_id: "acme",
      name: "Team Comms",
      tags: ["team"],
      source_query: "How does the team communicate?",
      content: "The team prefers async updates.",
      last_refreshed_at: "2026-01-01T00:00:00Z",
      created_at: "2025-12-01T00:00:00Z",
      is_stale: false,
    });
    const parsed = parseDocument(md);
    expect(parsed.frontmatter.id).toBe("team-comms");
    expect(parsed.frontmatter.bank).toBe("acme");
    expect(parsed.frontmatter.is_stale).toBe(false);
    expect(parsed.body.trim()).toBe("The team prefers async updates.");
  });

  it("uses a placeholder when content is empty", () => {
    const md = renderMentalModel({ id: "x", bank_id: "b", name: "X", content: "" });
    expect(md).toContain("has not been generated yet");
  });
});

describe("fileNameFor", () => {
  it("produces safe markdown filenames", () => {
    expect(fileNameFor("user-preferences")).toBe("user-preferences.md");
    expect(fileNameFor("Weird Name!!")).toBe("weird-name.md");
  });
});
