import { describe, expect, it } from "vitest";

import { bankDisplayLabel } from "@/lib/bank-label";

describe("bankDisplayLabel", () => {
  it("uses the bank name when present", () => {
    expect(bankDisplayLabel({ bank_id: "project-9d6f1542fbee", name: "Tarn (project)" })).toBe(
      "Tarn (project)"
    );
  });

  it("falls back to bank_id when the bank has no display name", () => {
    expect(bankDisplayLabel({ bank_id: "default", name: null })).toBe("default");
  });

  it("falls back to bank_id when the display name is blank", () => {
    expect(bankDisplayLabel({ bank_id: "project-9d6f1542fbee", name: "  " })).toBe(
      "project-9d6f1542fbee"
    );
  });
});
