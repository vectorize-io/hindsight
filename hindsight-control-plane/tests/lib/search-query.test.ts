import { describe, expect, it } from "vitest";

import { isSearchQueryCleared } from "../../src/lib/search-query";

describe("isSearchQueryCleared", () => {
  it.each([
    ["query", ""],
    [" query ", "   "],
  ])("recognizes an active query being cleared", (previousQuery, nextQuery) => {
    expect(isSearchQueryCleared(previousQuery, nextQuery)).toBe(true);
  });

  it.each([
    ["", ""],
    ["query", "que"],
    ["   ", ""],
  ])("does not request a reload for %j -> %j", (previousQuery, nextQuery) => {
    expect(isSearchQueryCleared(previousQuery, nextQuery)).toBe(false);
  });
});
