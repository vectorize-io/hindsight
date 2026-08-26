import { describe, expect, it } from "vitest";
import type { KnowledgeNode } from "@/lib/api";
import { findRequestedKnowledgePageId } from "@/lib/knowledge-base-navigation";

function node(
  id: string,
  kind: KnowledgeNode["kind"],
  children: KnowledgeNode[] = []
): KnowledgeNode {
  return {
    id,
    kind,
    name: id,
    parent_id: null,
    mental_model_id: kind === "page" ? `mm-${id}` : null,
    managed: false,
    description: null,
    tags: [],
    timestamp: null,
    is_stale: null,
    trigger: null,
    children,
  };
}

const tree = [node("folder-1", "folder", [node("page-nested", "page")]), node("page-root", "page")];

describe("findRequestedKnowledgePageId", () => {
  it("returns a nested page when it belongs to the loaded bank", () => {
    expect(findRequestedKnowledgePageId(tree, "page-nested")).toBe("page-nested");
  });

  it("rejects a page id that is absent from the loaded bank", () => {
    expect(findRequestedKnowledgePageId(tree, "page-from-another-bank")).toBeNull();
  });

  it("rejects folder ids", () => {
    expect(findRequestedKnowledgePageId(tree, "folder-1")).toBeNull();
  });

  it("returns null when no page was requested", () => {
    expect(findRequestedKnowledgePageId(tree, null)).toBeNull();
  });
});
