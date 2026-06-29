import { describe, it, expect } from "vitest";
import { stringifyFrontmatter, parseDocument, type Frontmatter } from "../src/frontmatter.js";
import { slug, planMirror } from "../src/format.js";
import type { KnowledgeSnapshot } from "../src/client.js";

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

describe("slug", () => {
  it("produces safe path segments", () => {
    expect(slug("user-preferences")).toBe("user-preferences");
    expect(slug("Weird Name!!")).toBe("weird-name");
    expect(slug("Billing Policy")).toBe("billing-policy");
    expect(slug("")).toBe("untitled");
  });
});

describe("planMirror", () => {
  it("nests pages under folder dirs and uses the page's OKF content", () => {
    const snapshot: KnowledgeSnapshot = {
      roots: [
        {
          id: "f1",
          kind: "folder",
          name: "Policies",
          parent_id: null,
          children: [{ id: "p1", kind: "page", name: "Billing", parent_id: "f1", children: [] }],
        },
        { id: "p2", kind: "page", name: "Glossary", parent_id: null, children: [] },
      ],
      content: new Map([["p1", "# Billing\n\nNet-30.\n"]]), // p2 intentionally has no content
    };

    const plan = planMirror(snapshot);
    expect(plan.folderCount).toBe(1);
    expect(plan.pageCount).toBe(2);
    expect(plan.dirs).toEqual(["policies"]);

    const billing = plan.files.find((f) => f.relPath === "policies/billing.md");
    expect(billing?.content).toContain("Net-30.");

    const glossary = plan.files.find((f) => f.relPath === "glossary.md");
    expect(glossary?.content).toContain("has not been generated yet"); // placeholder
  });
});
