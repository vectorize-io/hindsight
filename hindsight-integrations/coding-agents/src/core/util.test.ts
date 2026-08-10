import { describe, expect, it } from "vitest";
import { semverGte } from "./util";

describe("semverGte", () => {
  it("compares numerically, not lexically", () => {
    expect(semverGte("0.10.0", "0.9.0")).toBe(true);
    expect(semverGte("0.9.0", "0.10.0")).toBe(false);
  });

  it("decides the retain-idempotency floor correctly", () => {
    expect(semverGte("0.8.6", "0.8.6")).toBe(true);
    expect(semverGte("0.8.5", "0.8.6")).toBe(false); // one patch short
    expect(semverGte("0.9.0", "0.8.6")).toBe(true);
    expect(semverGte("1.0.0", "0.8.6")).toBe(true);
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
