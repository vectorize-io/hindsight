import { describe, expect, it } from "vitest";
import { retrievedNotes, retrievedNotesDetailed } from "../src/reflect-util";
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

describe("retrievedNotesDetailed", () => {
  it("pairs each note with its retrieved text snippets", () => {
    const response: ReflectResponse = {
      text: "answer",
      trace: {
        tool_calls: [
          {
            tool: "recall",
            input: { query: "acme" },
            output: {
              results: [{ id: "a", text: "Acme cares about SOC2", document_id: "Work/acme.md" }],
            },
          },
          {
            tool: "search_observations",
            input: { query: "worried" },
            output: {
              source_facts: {
                f1: { id: "f1", text: "felt scattered", document_id: "Personal/journal.md" },
              },
            },
          },
        ],
      },
    };

    expect(retrievedNotesDetailed(response)).toEqual([
      { docId: "Personal/journal.md", snippets: ["felt scattered"] },
      { docId: "Work/acme.md", snippets: ["Acme cares about SOC2"] },
    ]);
  });
});
