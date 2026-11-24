import { describe, it, expect } from "vitest";
import { buildMemoryMap } from "@/lib/memory-map";

const start = new Date(2026, 0, 1);
const end = new Date(2027, 0, 1);
const row = (id: string, fact_type: string, mentioned_at: string) => ({
  id,
  fact_type,
  mentioned_at,
});
const edge = (source: string, target: string, weight: number) => ({
  data: { source, target, weight },
});

describe("buildMemoryMap", () => {
  it("keeps the year's memories as dots, with their strongest links and no text", () => {
    const map = buildMemoryMap(
      [
        {
          table_rows: [
            row("a", "world", "2026-01-01T12:00:00Z"),
            row("b", "observation", "2026-07-01T12:00:00Z"),
            row("c", "experience", "2026-12-01T12:00:00Z"),
            row("old", "world", "2025-06-01T12:00:00Z"),
          ],
          edges: [
            edge("a", "a", 1), // self link
            edge("a", "b", 0.9),
            edge("b", "a", 0.9), // the same link, the other way
            edge("a", "c", 0.5),
            edge("a", "old", 1), // to a memory outside the year
          ],
        },
      ],
      start,
      end
    );
    expect(map.nodes.map((n) => n.k)).toEqual([0, 2, 1]);
    expect(map.nodes[0].t).toBeLessThan(map.nodes[1].t);
    expect(map.nodes[1].t).toBeLessThan(map.nodes[2].t);
    expect(map.links).toEqual([
      [0, 1],
      [0, 2],
    ]);
    for (const n of map.nodes) {
      expect(Object.keys(n).sort()).toEqual(["k", "t", "x", "y"]);
      expect(n.x).toBeGreaterThanOrEqual(0);
      expect(n.x).toBeLessThanOrEqual(1);
    }
  });

  it("is empty when no memory falls in the year", () => {
    expect(
      buildMemoryMap([{ table_rows: [row("x", "world", "2025-01-01T00:00:00Z")] }], start, end)
    ).toEqual({
      nodes: [],
      links: [],
    });
  });
});
