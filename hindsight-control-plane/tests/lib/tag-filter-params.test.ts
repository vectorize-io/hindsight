import { describe, expect, it } from "vitest";
import { appendTagFilter, isTagFilterActive, readTagFilter } from "@/lib/tag-filter-params";

const read = (qs: string) => readTagFilter(new URLSearchParams(qs));

describe("tag filter params", () => {
  it("forwards repeated tags with a known mode", () => {
    expect(read("tags=user:dan&tags=team&tags_match=any_strict")).toEqual({
      tags: ["user:dan", "team"],
      tags_match: "any_strict",
    });
  });

  it("drops empty tags instead of filtering on the empty tag", () => {
    expect(read("tags=")).toEqual({});
    expect(read("tags=&tags=a")).toEqual({ tags: ["a"] });
  });

  it("drops an unknown mode instead of forwarding a 422", () => {
    expect(read("tags=a&tags_match=bogus")).toEqual({ tags: ["a"] });
  });

  it("keeps tags_match only while the filter is active", () => {
    expect(read("tags_match=any_strict")).toEqual({});
    expect(read("tags=a&tags_match=all_strict")).toEqual({ tags: ["a"], tags_match: "all_strict" });
    // exact with no tags is the untagged scope — still a filter.
    expect(read("tags_match=exact")).toEqual({ tags_match: "exact" });
  });

  it("is active for tags, or exact alone", () => {
    expect(isTagFilterActive({ tags_match: "any_strict" })).toBe(false);
    expect(isTagFilterActive({ tags: [], tags_match: "exact" })).toBe(true);
    expect(isTagFilterActive({ tags: ["a"] })).toBe(true);
  });

  it("appends the same normalized form it reads", () => {
    const params = new URLSearchParams();
    appendTagFilter(params, { tags: ["a", ""], tags_match: "any" });
    expect(params.toString()).toBe("tags=a&tags_match=any");
    const none = new URLSearchParams();
    appendTagFilter(none, { tags: [], tags_match: "any_strict" });
    expect(none.toString()).toBe("");
  });
});
