import type { BankInfo } from "@/lib/bank-context";

/**
 * Banks arrive already ordered by last write descending, one page at a time, so the
 * selector renders them in server order — with one exception: the bank you are already
 * in is pinned to the top, so the list opens on where you are instead of making you
 * hunt for it in a long page. The rest keep their relative order.
 *
 * A bank that has not been paged in yet cannot be hoisted; the list is returned
 * unchanged in that case.
 */
export function hoistCurrentBank(banks: BankInfo[], currentBank: string | null): BankInfo[] {
  if (!currentBank) return banks;
  const idx = banks.findIndex((b) => b.bank_id === currentBank);
  if (idx <= 0) return banks;
  return [banks[idx], ...banks.slice(0, idx), ...banks.slice(idx + 1)];
}
