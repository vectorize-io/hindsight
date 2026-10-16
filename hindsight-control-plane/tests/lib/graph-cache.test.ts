import { beforeEach, describe, expect, it, vi } from "vitest";

const getGraph = vi.fn();
vi.mock("@/lib/api", () => ({ client: { getGraph: (...a: unknown[]) => getGraph(...a) } }));

import {
  GRAPH_FRESH_MS,
  cachedGraph,
  fetchGraph,
  isFresh,
  prefetchGraph,
  sameGraph,
  storeGraph,
} from "@/lib/graph-cache";

let bank = 0;
const params = () => ({ bank_id: `b${++bank}`, type: "world", limit: 1000 });

describe("graph cache", () => {
  beforeEach(() => getGraph.mockReset());

  it("keys by every filter, treating absent and empty alike", () => {
    const p = params();
    storeGraph(p, { n: 1 });
    expect(cachedGraph({ ...p, q: undefined, tags: undefined })?.data).toEqual({ n: 1 });
    expect(cachedGraph({ ...p, type: "experience" })).toBeUndefined();
    expect(cachedGraph({ ...p, tags: ["a"] })).toBeUndefined();
  });

  it("is fresh only for a while", () => {
    const p = params();
    storeGraph(p, {}, 1000);
    const entry = cachedGraph(p)!;
    expect(isFresh(entry, 1000 + GRAPH_FRESH_MS - 1)).toBe(true);
    expect(isFresh(entry, 1000 + GRAPH_FRESH_MS)).toBe(false);
  });

  it("joins a running prefetch instead of reading again", async () => {
    let resolve!: (v: unknown) => void;
    getGraph.mockReturnValueOnce(new Promise((r) => (resolve = r)));
    const p = params();
    const warming = prefetchGraph(p);
    const joined = fetchGraph(p, new AbortController().signal);
    resolve({ n: 2 });
    expect(await joined).toEqual({ n: 2 });
    await warming;
    expect(getGraph).toHaveBeenCalledTimes(1);
    expect(cachedGraph(p)?.data).toEqual({ n: 2 });
  });

  it("does not prefetch what is already cached", async () => {
    const p = params();
    storeGraph(p, {});
    await prefetchGraph(p);
    expect(getGraph).not.toHaveBeenCalled();
  });

  it("swallows a failed prefetch and lets the next read retry", async () => {
    getGraph.mockRejectedValueOnce(new Error("boom")).mockResolvedValueOnce({ n: 3 });
    const p = params();
    await prefetchGraph(p);
    expect(cachedGraph(p)).toBeUndefined();
    expect(await fetchGraph(p)).toEqual({ n: 3 });
  });

  it("evicts the oldest entries past its size", () => {
    const first = params();
    storeGraph(first, {});
    for (let i = 0; i < 9; i++) storeGraph(params(), {});
    expect(cachedGraph(first)).toBeUndefined();
  });

  it("calls two reads the same only when nothing visible differs", () => {
    const g = (text: string, edges = 2) => ({
      total_units: 5,
      edges: Array(edges).fill({}),
      table_rows: [{ id: "a", text }],
    });
    expect(sameGraph(g("x"), g("x"))).toBe(true);
    expect(sameGraph(g("x"), g("y"))).toBe(false);
    expect(sameGraph(g("x"), g("x", 3))).toBe(false);
    expect(sameGraph(null, g("x"))).toBe(false);
  });
});
