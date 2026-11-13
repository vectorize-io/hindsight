/**
 * Reflect's Based On lists the knowledge documents the agent read: they arrive inside the recall
 * tool's output in the trace, not in based_on.
 */
import { describe, expect, it, vi } from "vitest";

vi.mock("next-intl", () => ({ useTranslations: () => (key: string) => key }));

const { knowledgeDocuments } = await import("@/components/think-view");

describe("knowledgeDocuments", () => {
  it("collects the recall calls' documents once each, and nothing else", () => {
    const doc = (id: string, text: string) => ({ document_id: id, source: `${id}.md`, text });
    const trace = {
      tool_calls: [
        { tool: "search_observations", output: { documents: [doc("x", "no")] } },
        { tool: "recall", output: { documents: [doc("a", "one"), doc("b", "two")] } },
        { tool: "recall", output: { documents: [doc("a", "one"), doc("a", "three")] } },
        { tool: "recall", output: { memories: [] } },
      ],
    };
    expect(knowledgeDocuments(trace).map((d) => `${d.document_id}:${d.text}`)).toEqual([
      "a:one",
      "b:two",
      "a:three",
    ]);
    expect(knowledgeDocuments(undefined)).toEqual([]);
  });
});
