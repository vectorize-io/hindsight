import { describe, expect, it } from "vitest";
import { editedRetainItem } from "@/lib/edited-retain-item";

describe("editedRetainItem", () => {
  it("keeps an undated document undated", () => {
    const item = editedRetainItem("new", { id: "d1", retain_params: { event_date: null } });
    expect(item.timestamp).toBe("unset");
  });

  it("leaves the timestamp out when none was stored", () => {
    expect(editedRetainItem("new", { id: "d1", retain_params: {} })).toEqual({
      content: "new",
      document_id: "d1",
    });
  });

  it("replays everything the original retain supplied", () => {
    const item = editedRetainItem("new", {
      id: "d1",
      tags: ["t"],
      retain_params: {
        context: "ctx",
        event_date: "2026-01-02T03:04:05",
        metadata: { k: "v" },
        strategy: "survey",
        entities: [{ text: "Widget", type: "CONCEPT" }],
        resolve_entities: false,
        observation_scopes: "per_tag",
      },
    });
    expect(item).toEqual({
      content: "new",
      document_id: "d1",
      context: "ctx",
      timestamp: "2026-01-02T03:04:05",
      metadata: { k: "v" },
      tags: ["t"],
      strategy: "survey",
      entities: [{ text: "Widget", type: "CONCEPT" }],
      resolve_entities: false,
      observation_scopes: "per_tag",
    });
  });
});
