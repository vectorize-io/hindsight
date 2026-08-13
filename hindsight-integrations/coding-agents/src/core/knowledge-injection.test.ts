import { describe, expect, it } from "vitest";
import { parsePageList, buildKnowledgePreamble, buildRosterRefresh } from "./knowledge-injection";

describe("parsePageList", () => {
  it("extracts {id,title} from the page list shape, tolerating junk", () => {
    const raw = {
      items: [
        { id: "p1", name: "Component map" },
        { id: "p2", name: "Core concepts" },
        { nope: 1 },
      ],
    };
    expect(parsePageList(raw)).toEqual([
      { id: "p1", title: "Component map" },
      { id: "p2", title: "Core concepts" },
    ]);
  });
  it("returns [] for null/garbage", () => {
    expect(parsePageList(null)).toEqual([]);
    expect(parsePageList(42 as unknown)).toEqual([]);
  });
});

describe("buildKnowledgePreamble", () => {
  it("lists the pages and gives a when-to-call guide for the FULL tool suite", () => {
    const out = buildKnowledgePreamble([{ id: "p1", title: "Component map" }]);
    expect(out).toContain("<hindsight_knowledge>");
    expect(out).toContain("Component map");
    expect(out).toContain("p1");
    // Every meaningful tool must be named with a when-to-call, not just pages.
    expect(out).toContain("hindsight_list_knowledge_pages");
    expect(out).toContain("hindsight_read_knowledge_page");
    expect(out).toContain("hindsight_reflect");
    expect(out).toContain("hindsight_capture_initiative");
    expect(out).toContain("hindsight_ingest_document");
    expect(out).toContain("hindsight_recall");
  });
  it("has an empty-state line when there are no pages", () => {
    const out = buildKnowledgePreamble([]);
    expect(out).toMatch(/no knowledge pages yet|still learning/i);
  });

  it("recall-only mode names only the exposed workflow tool", () => {
    const out = buildKnowledgePreamble([], { toolAllowlist: ["recall"] });
    expect(out).toContain("hindsight_recall");
    expect(out).not.toContain("hindsight_reflect");
    expect(out).not.toContain("hindsight_ingest_document");
    expect(out).not.toContain("hindsight_list_knowledge_pages");
    expect(out).not.toContain("No knowledge pages yet");
  });

  it("empty mode injects no tool guidance", () => {
    expect(buildKnowledgePreamble([], { toolAllowlist: [] })).toBe("");
    expect(buildRosterRefresh([], { toolAllowlist: [] })).toBe("");
  });

  it("does not name page tools excluded from a partial allowlist", () => {
    const out = buildKnowledgePreamble([], { toolAllowlist: ["search_knowledge_pages"] });
    expect(out).toContain("hindsight_search_knowledge_pages");
    expect(out).not.toContain("hindsight_list_knowledge_pages");
    expect(out).not.toContain("hindsight_read_knowledge_page");
  });
});

describe("buildRosterRefresh", () => {
  it("lists current pages and re-states the full tool guide", () => {
    const out = buildRosterRefresh([{ id: "p1", title: "Component map" }]);
    expect(out).toContain("Component map");
    expect(out).toContain("p1");
    for (const tool of [
      "hindsight_list_knowledge_pages",
      "hindsight_read_knowledge_page",
      "hindsight_capture_initiative",
      "hindsight_ingest_document",
    ]) {
      expect(out).toContain(tool);
    }
  });
  it("still emits the full tool guide when there are no pages yet (no roster, but the guide persists)", () => {
    const out = buildRosterRefresh([]);
    expect(out).toContain("<hindsight_knowledge_refresh>");
    expect(out).toContain("hindsight_capture_initiative");
    expect(out).toContain("hindsight_ingest_document");
    // No roster block when there are no pages.
    expect(out).not.toContain("Current Hindsight knowledge pages");
  });
});
