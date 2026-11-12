import { describe, expect, it } from "vitest";
import { findMatches, findPattern, treeConstellation } from "@/components/memory-tree-view";
import type { MemoryTreeFile } from "@/lib/api";

const topic = (path: string, texts: string[]): MemoryTreeFile => ({
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

describe("memory tree find", () => {
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

describe("memory tree constellation", () => {
  it("makes one star per observation, tied to its topic's hub and bridged to related topics", () => {
    const a = { ...topic("gemma/vllm.md", ["a0", "a1"]), related: ["postgres.md"] };
    const c = { ...topic("postgres.md", ["c0"]), related: ["gemma/vllm.md"] };
    const g = treeConstellation([a, c, { ...topic("README.md", ["r"]), kind: "readme" }]);
    expect(g.nodes.map((n) => [n.id, n.group])).toEqual([
      ["gemma/vllm.md-0", "gemma/vllm.md"],
      ["gemma/vllm.md-1", "gemma/vllm.md"],
      ["postgres.md-0", "postgres.md"],
    ]);
    expect(g.links.map((l) => [l.source, l.target, l.type])).toEqual([
      ["gemma/vllm.md-0", "gemma/vllm.md-1", "topic"],
      ["gemma/vllm.md-0", "postgres.md-0", "related"],
    ]);
  });
});
