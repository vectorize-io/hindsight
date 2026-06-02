import { describe, expect, it } from "vitest";
import { retrievedNotes } from "../src/reflect-util";
import type { ReflectResponse } from "../src/types";

describe("retrievedNotes", () => {
  it("extracts document_ids from recall results and nested observation source_facts", () => {
    const response: ReflectResponse = {
      text: "answer",
      // based_on facts carry no document_id (server omits it) — must come from the trace.
      based_on: { memories: [{ id: "1", text: "fact" }] },
      trace: {
        tool_calls: [
          {
            tool: "recall",
            input: { query: "acme" },
            output: { results: [{ id: "a", text: "x", document_id: "Work/Clients/acme.md" }] },
          },
          {
            tool: "search_observations",
            input: { query: "worried" },
            output: {
              observations: [{ id: "o1", text: "consolidated" }], // no doc id
              source_facts: {
                f1: { id: "f1", text: "src", document_id: "Personal/morning-pages.md" },
              },
            },
          },
        ],
      },
    };

    expect(retrievedNotes(response)).toEqual(["Personal/morning-pages.md", "Work/Clients/acme.md"]);
  });

  it("returns empty when nothing was retrieved", () => {
    expect(retrievedNotes({ text: "hi" })).toEqual([]);
  });
});
