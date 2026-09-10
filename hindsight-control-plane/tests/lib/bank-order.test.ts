import { describe, expect, it } from "vitest";
import { hoistCurrentBank } from "@/lib/bank-order";
import type { BankInfo } from "@/lib/bank-context";

function bank(bank_id: string): BankInfo {
  return {
    bank_id,
    name: null,
    mission: null,
    created_at: null,
    updated_at: null,
    fact_count: 0,
    last_document_at: null,
    last_write_at: null,
  };
}

const ids = (banks: BankInfo[]) => banks.map((b) => b.bank_id);

describe("hoistCurrentBank", () => {
  it("pins the current bank first and keeps the rest in server order", () => {
    const banks = [bank("a"), bank("b"), bank("c")];
    expect(ids(hoistCurrentBank(banks, "c"))).toEqual(["c", "a", "b"]);
  });

  it("leaves the list untouched when the current bank is already first", () => {
    const banks = [bank("a"), bank("b")];
    expect(hoistCurrentBank(banks, "a")).toBe(banks);
  });

  it("leaves the list untouched when no bank is selected", () => {
    const banks = [bank("a"), bank("b")];
    expect(hoistCurrentBank(banks, null)).toBe(banks);
  });

  it("leaves the list untouched when the current bank is not on the loaded page", () => {
    const banks = [bank("a"), bank("b")];
    expect(hoistCurrentBank(banks, "zz")).toBe(banks);
  });

  it("does not drop or duplicate banks", () => {
    const banks = [bank("a"), bank("b"), bank("c"), bank("d")];
    const out = hoistCurrentBank(banks, "b");
    expect(ids(out)).toEqual(["b", "a", "c", "d"]);
    expect(out).toHaveLength(banks.length);
  });
});
