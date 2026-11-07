import { describe, expect, it } from "vitest";
import { tagChange } from "@/lib/tag-change";

describe("tagChange", () => {
  it("sends only the tags that changed", () => {
    expect(tagChange(["a", "b"], ["b", "c"])).toEqual({ add_tags: ["c"], remove_tags: ["a"] });
  });

  it("is empty when nothing changed, whatever the order", () => {
    expect(tagChange(["a", "b"], ["b", "a"])).toEqual({ add_tags: [], remove_tags: [] });
  });

  it("removes everything when the list is cleared", () => {
    expect(tagChange(["a", "b"], [])).toEqual({ add_tags: [], remove_tags: ["a", "b"] });
  });
});
