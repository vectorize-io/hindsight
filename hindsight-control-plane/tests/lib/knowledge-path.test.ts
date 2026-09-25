import { describe, expect, it } from "vitest";
import type { KnowledgeNode } from "@/lib/api";
import { buildPathIndex } from "@/lib/knowledge-path";

function node(id: string, name: string, parent_id: string | null): KnowledgeNode {
  return {
    id,
    kind: parent_id === null || id.startsWith("kf") ? "folder" : "page",
    name,
    parent_id,
    mental_model_id: null,
    managed: false,
    description: null,
    tags: [],
    timestamp: null,
    is_stale: null,
    trigger: null,
    children: [],
  };
}

describe("buildPathIndex", () => {
  const nodes = [
    node("kf1", "Engineering", null),
    node("kf2", "Retrieval", "kf1"),
    node("kp1", "Recall pipeline", "kf2"),
    node("kp2", "Overview", null),
  ];

  it("walks the parent chain, bank root first", () => {
    const paths = buildPathIndex(nodes, "demo");
    expect(paths.get("kp1")).toBe("demo / Engineering / Retrieval");
  });

  it("gives a root-level page just the bank", () => {
    expect(buildPathIndex(nodes, "demo").get("kp2")).toBe("demo");
  });

  it("names folders by their own parents, not themselves", () => {
    const paths = buildPathIndex(nodes, "demo");
    expect(paths.get("kf2")).toBe("demo / Engineering");
    expect(paths.get("kf1")).toBe("demo");
  });

  it("stops at a parent the tree doesn't contain", () => {
    // A page whose folder was filtered out of the tree must still get a path
    // rather than looping or throwing.
    const orphan = [node("kp3", "Stray", "missing-folder")];
    expect(buildPathIndex(orphan, "demo").get("kp3")).toBe("demo");
  });
});
