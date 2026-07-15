import type { BankInfo } from "@/lib/bank-context";

export function bankDisplayLabel(bank: Pick<BankInfo, "bank_id" | "name">): string {
  const name = bank.name?.trim();
  return name || bank.bank_id;
}
