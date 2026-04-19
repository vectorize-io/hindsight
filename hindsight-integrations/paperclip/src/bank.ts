/**
 * Bank ID derivation — maps Paperclip company/agent identity onto Hindsight bank IDs.
 *
 * Default format: "paperclip::{companyId}::{agentId}"
 *
 * bankGranularity: ['company']        → "paperclip::{companyId}"
 * bankGranularity: ['agent']          → "paperclip::{agentId}"
 * bankGranularity: ['company','agent'] → "paperclip::{companyId}::{agentId}"
 */

export interface BankContext {
  companyId: string;
  agentId: string;
}

export interface BankConfig {
  bankGranularity?: Array<"company" | "agent">;
}

export function deriveBankId(context: BankContext, config: BankConfig): string {
  const granularity = config.bankGranularity ?? ["company", "agent"];
  const parts: string[] = ["paperclip"];

  for (const field of granularity) {
    if (field === "company") parts.push(context.companyId);
    if (field === "agent") parts.push(context.agentId);
  }

  return parts.join("::");
}
