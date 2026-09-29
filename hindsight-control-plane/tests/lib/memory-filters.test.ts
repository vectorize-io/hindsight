import { describe, it, expect } from "vitest";
import {
  AppliedMemoryFilters,
  hasActiveMemoryFilters,
  shouldShowEmptyBankState,
} from "@/lib/memory-filters";

const NONE: AppliedMemoryFilters = { q: "", tags: [] };

describe("hasActiveMemoryFilters", () => {
  it("is false with no filters applied", () => {
    expect(hasActiveMemoryFilters(NONE)).toBe(false);
  });

  it("is false for a whitespace-only query", () => {
    expect(hasActiveMemoryFilters({ q: "   ", tags: [] })).toBe(false);
  });

  it("is true for a text query", () => {
    expect(hasActiveMemoryFilters({ q: "paris", tags: [] })).toBe(true);
  });

  it("is true for a tag filter", () => {
    expect(hasActiveMemoryFilters({ q: "", tags: ["trip"] })).toBe(true);
  });

  it("is true for the global observation scope, whose tag set is empty", () => {
    expect(hasActiveMemoryFilters({ q: "", tags: [], tagsMatch: "exact" })).toBe(true);
  });
});

describe("shouldShowEmptyBankState", () => {
  it("shows for an empty bank with no filters", () => {
    expect(shouldShowEmptyBankState(0, NONE)).toBe(true);
  });

  it("does not show when the bank has memories", () => {
    expect(shouldShowEmptyBankState(42, NONE)).toBe(false);
  });

  // Issue #3670: a search that matched nothing must keep the filter bar on
  // screen, otherwise the query can't be cleared without leaving the page.
  it("does not show for a search that matched nothing", () => {
    expect(shouldShowEmptyBankState(0, { q: "zzzz", tags: [] })).toBe(false);
  });

  it("does not show for a tag filter that matched nothing", () => {
    expect(shouldShowEmptyBankState(0, { q: "", tags: ["nope"] })).toBe(false);
  });

  it("does not show while the total is still unknown", () => {
    expect(shouldShowEmptyBankState(undefined, NONE)).toBe(false);
  });
});
