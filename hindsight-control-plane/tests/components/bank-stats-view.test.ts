import { describe, expect, it } from "vitest";

import { getCausalLinkCount } from "@/components/bank-stats-view";

describe("getCausalLinkCount", () => {
  it("combines canonical and legacy causal link types", () => {
    expect(
      getCausalLinkCount({
        caused_by: 7,
        causes: 3,
        enables: 2,
        prevents: 1,
        temporal: 20,
        semantic: 10,
        entity: 5,
      })
    ).toBe(13);
  });

  it("treats absent causal link types as zero", () => {
    expect(getCausalLinkCount({ temporal: 2 })).toBe(0);
  });
});
