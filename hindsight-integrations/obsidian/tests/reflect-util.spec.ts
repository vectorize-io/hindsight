import { describe, expect, it } from "vitest";
import { groundedNotes, retrievedNotes, retrievedNotesDetailed } from "../src/reflect-util";
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

describe("groundedNotes", () => {
  it("returns only the notes the answer cited, joined to the trace by fact id", () => {
    const response: ReflectResponse = {
      text: "Jon's favorite band is Tool.",
      // The answer is grounded on just one fact...
      based_on: { memories: [{ id: "f-tool", text: "Jon's favorite band is Tool" }] },
      trace: {
        tool_calls: [
          {
            tool: "recall",
            input: { query: "Jon band" },
            // ...even though the agent's scratchpad retrieved several notes.
            output: {
              results: [
                { id: "f-tool", text: "Jon's favorite band is Tool", document_id: "Music/Jon.md" },
                { id: "f-x", text: "Jon likes hiking", document_id: "People/Jon-misc.md" },
                { id: "f-y", text: "unrelated", document_id: "Random/notes.md" },
              ],
            },
          },
        ],
      },
    };

    // Only the cited note appears — not the full scratchpad.
    expect(groundedNotes(response)).toEqual([
      { docId: "Music/Jon.md", snippets: ["Jon's favorite band is Tool"] },
    ]);
  });

  it("dedupes by note and preserves citation order", () => {
    const response: ReflectResponse = {
      text: "answer",
      based_on: {
        memories: [
          { id: "b1", text: "from acme" },
          { id: "a1", text: "from journal" },
          { id: "b2", text: "also acme" },
        ],
      },
      trace: {
        tool_calls: [
          {
            tool: "recall",
            input: { query: "q" },
            output: {
              results: [
                { id: "b1", text: "from acme", document_id: "Work/acme.md" },
                { id: "a1", text: "from journal", document_id: "Personal/journal.md" },
                { id: "b2", text: "also acme", document_id: "Work/acme.md" },
              ],
            },
          },
        ],
      },
    };

    expect(groundedNotes(response)).toEqual([
      { docId: "Work/acme.md", snippets: ["from acme", "also acme"] },
      { docId: "Personal/journal.md", snippets: ["from journal"] },
    ]);
  });

  it("falls back to the full retrieved list when nothing was cited", () => {
    const response: ReflectResponse = {
      text: "answer",
      // No based_on (some responses omit it) — don't show an empty panel.
      trace: {
        tool_calls: [
          {
            tool: "recall",
            input: { query: "q" },
            output: { results: [{ id: "a", text: "x", document_id: "Work/acme.md" }] },
          },
        ],
      },
    };

    expect(groundedNotes(response)).toEqual([{ docId: "Work/acme.md", snippets: ["x"] }]);
  });
});
