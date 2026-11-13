import { describe, expect, it } from "vitest";
import {
  findMatches,
  findPattern,
  treeConstellation,
  clusterLabels,
} from "@/components/observation-tree-view";
import type { ObservationTreeFile } from "@/lib/api";

const topic = (path: string, texts: string[]): ObservationTreeFile => ({
  path,
  name: path.split("/").pop()!,
  kind: "topic",
  title: path,
  observation_count: texts.length,
  about: [],
  related: [],
  observations: texts.map((text, i) => ({ id: `${path}-${i}`, text, proof_count: 1, tags: [] })),
  content: "",
});

describe("observation tree find", () => {
  const files = [
    topic("gemma/vllm.md", ["Gemma runs on vLLM", "NVFP4 is faster"]),
    topic("postgres.md", ["Postgres locks rows"]),
  ];

  it("counts matching lines per file, case-insensitive", () => {
    expect(findMatches(files, findPattern("gemma")!)).toEqual(new Map([["gemma/vllm.md", 2]]));
  });

  it("treats an invalid regex as literal text", () => {
    expect(findMatches(files, findPattern("locks (")!).size).toBe(0);
    expect(findMatches(files, findPattern("nvfp4|postgres")!)).toEqual(
      new Map([
        ["gemma/vllm.md", 1],
        ["postgres.md", 2],
      ])
    );
  });

  it("an empty query finds nothing", () => {
    expect(findPattern("  ")).toBeNull();
  });
});

describe("observation tree constellation", () => {
  it("makes one star per observation and bridges related topics, even one-sided", () => {
    // Only one side lists the relation: the bridge must still be drawn.
    const a = topic("gemma/vllm.md", ["a0", "a1"]);
    const c = { ...topic("postgres.md", ["c0"]), related: ["gemma/vllm.md"] };
    const g = treeConstellation([a, c, { ...topic("index.md", ["r"]), kind: "index" }]);
    expect(g.nodes.map((n) => [n.id, n.group])).toEqual([
      ["gemma/vllm.md-0", "gemma/vllm.md"],
      ["gemma/vllm.md-1", "gemma/vllm.md"],
      ["postgres.md-0", "postgres.md"],
    ]);
    expect(g.links.map((l) => [l.source, l.target, l.type])).toEqual([
      ["postgres.md-0", "gemma/vllm.md-0", "related"],
    ]);
  });

  it("lays a topic's related topics out right after it", () => {
    const a = { ...topic("a.md", ["a0"]), related: ["c.md"] };
    const b = topic("b.md", ["b0"]);
    const c = topic("c.md", ["c0"]);
    const g = treeConstellation([a, b, c]);
    expect([...new Set(g.nodes.map((n) => n.group))]).toEqual(["a.md", "c.md", "b.md"]);
  });

  it("tells apart topics that share a title", () => {
    const x = { ...topic("memlake/vchord.md", ["x"]), title: "vchord · worker" };
    const y = { ...topic("analytics/vchord.md", ["y"]), title: "vchord · worker" };
    const z = { ...topic("gemma.md", ["z"]), title: "Gemma" };
    expect(Object.fromEntries(clusterLabels([x, y, z]))).toEqual({
      "memlake/vchord.md": "vchord · worker · memlake",
      "analytics/vchord.md": "vchord · worker · analytics",
      "gemma.md": "Gemma",
    });
  });
});
