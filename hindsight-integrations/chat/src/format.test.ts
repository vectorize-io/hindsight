import { describe, it, expect } from "vitest";
import { formatMemoriesAsSystemPrompt } from "./format.js";
import type { RecallResult, EntityState } from "./types.js";

function makeMemory(overrides: Partial<RecallResult> = {}): RecallResult {
  return {
    id: "mem-1",
    text: "User prefers dark mode",
    type: "experience",
    ...overrides,
  };
}

function makeEntities(): Record<string, EntityState> {
  return {
    "ent-1": {
      entity_id: "ent-1",
      canonical_name: "Alice",
      observations: [
        { text: "Works at Acme Corp" },
        { text: "Prefers TypeScript", mentioned_at: "2025-01-01T00:00:00Z" },
      ],
    },
  };
}

describe("formatMemoriesAsSystemPrompt", () => {
  it("returns empty string when no memories and no entities", () => {
    expect(formatMemoriesAsSystemPrompt([], null)).toBe("");
    expect(formatMemoriesAsSystemPrompt([], {})).toBe("");
    expect(formatMemoriesAsSystemPrompt([], undefined)).toBe("");
  });

  it("formats memories with default preamble", () => {
    const result = formatMemoriesAsSystemPrompt(
      [makeMemory(), makeMemory({ id: "mem-2", text: "Likes coffee", type: "world" })],
      null
    );

    expect(result).toContain("You have access to the following memories about this user");
    expect(result).toContain("<memories>");
    expect(result).toContain("- User prefers dark mode [experience]");
    expect(result).toContain("- Likes coffee [world]");
    expect(result).toContain("</memories>");
    expect(result).not.toContain("<entity_observations>");
  });

  it("formats memories without type suffix when type is null", () => {
    const result = formatMemoriesAsSystemPrompt([makeMemory({ type: null })], null);
    expect(result).toContain("- User prefers dark mode\n");
    expect(result).not.toContain("[");
  });

  it("includes entity observations", () => {
    const result = formatMemoriesAsSystemPrompt([makeMemory()], makeEntities());

    expect(result).toContain("<memories>");
    expect(result).toContain("<entity_observations>");
    expect(result).toContain("## Alice");
    expect(result).toContain("- Works at Acme Corp");
    expect(result).toContain("- Prefers TypeScript");
    expect(result).toContain("</entity_observations>");
  });

  it("shows only entities when no memories", () => {
    const result = formatMemoriesAsSystemPrompt([], makeEntities());

    expect(result).not.toContain("<memories>");
    expect(result).toContain("<entity_observations>");
    expect(result).toContain("## Alice");
  });

  it("uses custom preamble", () => {
    const result = formatMemoriesAsSystemPrompt([makeMemory()], null, {
      preamble: "Here is what I know:",
    });
    expect(result.startsWith("Here is what I know:")).toBe(true);
  });

  it("limits memories with maxMemories", () => {
    const memories = [
      makeMemory({ id: "1", text: "First" }),
      makeMemory({ id: "2", text: "Second" }),
      makeMemory({ id: "3", text: "Third" }),
    ];
    const result = formatMemoriesAsSystemPrompt(memories, null, {
      maxMemories: 2,
    });
    expect(result).toContain("First");
    expect(result).toContain("Second");
    expect(result).not.toContain("Third");
  });

  it("filters by includeTypes", () => {
    const memories = [
      makeMemory({ id: "1", text: "World fact", type: "world" }),
      makeMemory({ id: "2", text: "Experience", type: "experience" }),
      makeMemory({ id: "3", text: "Observation", type: "observation" }),
    ];
    const result = formatMemoriesAsSystemPrompt(memories, null, {
      includeTypes: ["world", "observation"],
    });
    expect(result).toContain("World fact");
    expect(result).not.toContain("Experience");
    expect(result).toContain("Observation");
  });

  it("excludes entities when includeEntities is false", () => {
    const result = formatMemoriesAsSystemPrompt([makeMemory()], makeEntities(), {
      includeEntities: false,
    });
    expect(result).toContain("<memories>");
    expect(result).not.toContain("<entity_observations>");
  });
});
