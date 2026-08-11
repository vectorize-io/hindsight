import { afterEach, describe, expect, it, vi } from "vitest";
import { HindsightClient } from "./hindsight";

afterEach(() => vi.restoreAllMocks());

describe("HindsightClient project-scoped reflect", () => {
  it("scopes ordinary reflect and supports explicit bank-wide reflect", async () => {
    const bodies: unknown[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init: RequestInit) => {
        bodies.push(JSON.parse(String(init.body)));
        return { ok: true, status: 200, json: async () => ({ text: "answer" }) } as Response;
      })
    );
    const client = new HindsightClient({
      apiUrl: "http://x",
      bank: "default",
      projectScope: { projectTag: "project:potcodev", globalTags: ["scope:global"] },
    });
    await client.reflect("why?", { budget: "high" });
    await client.reflect("compare projects", { budget: "high", unscoped: true });
    expect(bodies[0]).toEqual({
      query: "why?",
      budget: "high",
      tag_groups: [
        {
          or: [
            { tags: ["project:potcodev"], match: "any_strict" },
            { tags: ["scope:global"], match: "any_strict" },
          ],
        },
      ],
    });
    expect(bodies[1]).toEqual({ query: "compare projects", budget: "high" });
  });
});
