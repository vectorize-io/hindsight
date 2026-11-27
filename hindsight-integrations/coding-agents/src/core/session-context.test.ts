import { describe, expect, it, vi } from "vitest";
import { loadSessionContext, parseSessionContext } from "./session-context";
import { resolveConfig } from "./config";
import { buildSessionStartContext } from "./session-start";

describe("parseSessionContext", () => {
  it("reads mental_model:/page: entries in order; a bare id is a mental model", () => {
    expect(parseSessionContext(["page:kp-1", "mm-rules", " mental_model: mm-2 "])).toEqual([
      { kind: "page", ref: "kp-1" },
      { kind: "mental_model", ref: "mm-rules" },
      { kind: "mental_model", ref: "mm-2" },
    ]);
  });

  it("drops unknown kinds, empty ids and non-strings", () => {
    expect(parseSessionContext(["doc:x", "page:", "", 3, "page:ok"])).toEqual([
      { kind: "page", ref: "ok" },
    ]);
    expect(parseSessionContext(undefined)).toEqual([]);
    expect(parseSessionContext("page:x")).toEqual([]);
  });

  it("comes through resolveConfig, empty by default", () => {
    expect(resolveConfig({}).sessionContext).toEqual([]);
    expect(resolveConfig({ sessionContext: ["page:kp-1"] }).sessionContext).toEqual([
      { kind: "page", ref: "kp-1" },
    ]);
  });
});

describe("loadSessionContext", () => {
  const client = {
    getPage: vi.fn(async (id: string) => ({ name: `Page ${id}`, body: `body of ${id}` })),
    getMentalModel: vi.fn(async (id: string) => {
      if (id === "gone") throw new Error("mental model not found: gone");
      if (id === "empty") return { name: "Empty", content: "  " };
      return { name: "Standing rules", content: "Never touch prod config." };
    }),
  };

  it("renders every readable entry in the configured order, skipping failures and empty ones", async () => {
    const out = await loadSessionContext(
      client,
      parseSessionContext(["mm-rules", "gone", "page:kp-1", "empty"])
    );
    expect(out).toContain("<hindsight_session_context>");
    expect(out.indexOf("## Standing rules\nNever touch prod config.")).toBeLessThan(
      out.indexOf("## Page kp-1\nbody of kp-1")
    );
    expect(out).not.toContain("Empty");
    expect(client.getMentalModel).toHaveBeenCalledWith("gone", { timeoutMs: 5000 });
  });

  it("returns nothing when no entry is configured or none could be read", async () => {
    expect(await loadSessionContext(client, [])).toBe("");
    expect(await loadSessionContext(client, parseSessionContext(["gone"]))).toBe("");
  });
});

describe("sessionContext at session start", () => {
  it("lands after the pages preamble in the session's context", async () => {
    const out = await buildSessionStartContext({
      cwd: "/repo/dir",
      bankId: "bank-1",
      cfg: resolveConfig({ autoSeed: false, sessionContext: ["mm-rules"] }),
      client: {
        listDocumentIds: async () => new Set(["git:a"]),
        listPages: async () => ({ items: [] }),
        getPage: async () => ({}),
        getMentalModel: async () => ({ name: "Standing rules", content: "Tests first." }),
      },
      detectLegacyPlugin: () => undefined,
    });
    const ctx = out.additionalContext ?? "";
    expect(ctx.indexOf("</hindsight_pages>")).toBeLessThan(ctx.indexOf("## Standing rules"));
    expect(ctx).toContain("Tests first.");
  });
});
