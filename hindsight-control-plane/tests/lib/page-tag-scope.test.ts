import { describe, it, expect } from "vitest";
import { createTagsMatch, includesUntagged, updateTagsMatch } from "@/lib/page-tag-scope";

describe("includesUntagged", () => {
  it("reads an unset tags_match as excluding untagged memories", () => {
    // Null/undefined is what the API returns for a page that never stored one;
    // the server resolves that to all_strict as soon as the page has tags.
    expect(includesUntagged(undefined)).toBe(false);
    expect(includesUntagged(null)).toBe(false);
    expect(includesUntagged("all_strict")).toBe(false);
    expect(includesUntagged("any_strict")).toBe(false);
    expect(includesUntagged("exact")).toBe(false);
  });

  it("reads the inclusive modes as including untagged memories", () => {
    expect(includesUntagged("all")).toBe(true);
    expect(includesUntagged("any")).toBe(true);
  });
});

describe("createTagsMatch", () => {
  it("sends nothing when the page has no tags", () => {
    // Nothing to widen: an untagged page already matches the whole bank.
    expect(createTagsMatch([], true)).toBeUndefined();
    expect(createTagsMatch([], false)).toBeUndefined();
  });

  it("sends nothing when the page keeps the strict default", () => {
    expect(createTagsMatch(["homelab"], false)).toBeUndefined();
  });

  it("widens to 'all' when the page opts into untagged memories", () => {
    expect(createTagsMatch(["type:runbook", "homelab"], true)).toBe("all");
  });
});

describe("updateTagsMatch", () => {
  it("sends nothing when the checkbox was not moved", () => {
    // The guard that keeps a page on "any"/"exact" from being flattened to
    // all_strict by an unrelated edit such as a rename.
    expect(updateTagsMatch(["homelab"], false, false)).toBeUndefined();
    expect(updateTagsMatch(["homelab"], true, true)).toBeUndefined();
  });

  it("sends nothing when the same edit cleared the tags", () => {
    expect(updateTagsMatch([], true, false)).toBeUndefined();
    expect(updateTagsMatch([], false, true)).toBeUndefined();
  });

  it("widens to 'all' when untagged memories are switched on", () => {
    expect(updateTagsMatch(["homelab"], true, false)).toBe("all");
  });

  it("narrows back to 'all_strict' when they are switched off", () => {
    expect(updateTagsMatch(["homelab"], false, true)).toBe("all_strict");
  });
});
