import { describe, expect, it } from "vitest";
import { MIN_IDEMPOTENT_RETAIN_VERSION } from "./hindsight";
import { semverGte } from "./util";

describe("semverGte", () => {
  it("compares numerically, not lexically", () => {
    expect(semverGte("0.10.0", "0.9.0")).toBe(true);
    expect(semverGte("0.9.0", "0.10.0")).toBe(false);
  });

  it("decides the retain-idempotency floor correctly", () => {
    // Pinned to the real constant: operation_id first shipped in v0.8.6 (#2937/#2947).
    const floor = MIN_IDEMPOTENT_RETAIN_VERSION;
    expect(floor).toBe("0.8.6");
    expect(semverGte("0.8.6", floor)).toBe(true);
    expect(semverGte("0.8.5", floor)).toBe(false); // one patch short
    expect(semverGte("0.9.0", floor)).toBe(true);
    expect(semverGte("1.0.0", floor)).toBe(true);
  });

  it("treats an unknown version as too old — a capability probe must fail closed", () => {
    expect(semverGte(undefined, "0.8.6")).toBe(false);
    expect(semverGte("", "0.8.6")).toBe(false);
    expect(semverGte("dev", "0.8.6")).toBe(false);
  });

  it("compares by the numeric part of a pre-release or build-tagged version", () => {
    expect(semverGte("0.9.0rc1", "0.8.6")).toBe(true);
    expect(semverGte("0.8.5+dev", "0.8.6")).toBe(false);
  });

  it("treats a missing patch component as zero", () => {
    expect(semverGte("0.9", "0.8.6")).toBe(true);
    expect(semverGte("0.8", "0.8.6")).toBe(false);
  });
});
