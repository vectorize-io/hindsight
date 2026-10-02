import { describe, expect, it } from "vitest";
import { readTagFilter } from "@/lib/tag-filter-params";

const read = (query: string) => readTagFilter(new URLSearchParams(query));

describe("readTagFilter", () => {
  it("forwards repeated tags with a known mode", () => {
    expect(read("tags=user:dan&tags=team&tags_match=any_strict")).toEqual({
      tags: ["user:dan", "team"],
      tags_match: "any_strict",
    });
  });

  it("sends nothing without tags, even when a mode is given", () => {
    expect(read("tags_match=any_strict")).toEqual({});
    expect(read("tags=")).toEqual({});
  });

  it("keeps exact with no tags — it is itself a filter for untagged entities", () => {
    expect(read("tags_match=exact")).toEqual({ tags: [], tags_match: "exact" });
  });

  it("drops an unknown mode instead of forwarding it", () => {
    expect(read("tags=user:dan&tags_match=bogus")).toEqual({ tags: ["user:dan"] });
  });
});
